"""Lightning callback that drives TensorBoard visualisation.

:class:`ImageLogger` caches the first batch of every training /
validation epoch (on rank 0), runs an automatic-mode forward pass at
epoch end, and hands the predictions off to
:func:`_log_predictions` which emits the panels defined in
:mod:`brainbow.callbacks.tensorboard.heads`.
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from einops import rearrange

from brainbow.callbacks.tensorboard.heads import _log_predictions
from brainbow.callbacks.tensorboard.tags import TagContext
from brainbow.callbacks.tensorboard.viz import _to_2d
from brainbow.losses import AFF_SLICE, AFFINITY_OFFSETS, N_ATTRACTIVE, SEM_SLICE


class ImageLogger(pl.Callback):
    """Log sample images to TensorBoard at the end of every *n*-th epoch.

    Logs visualisations for both **training** and **validation** batches
    using **automatic** mode (image-only forward).

    Lifecycle (per epoch, rank-0 only)::

        on_{train,validation}_batch_end(batch_idx == 0)
            -> cache first batch on CPU (self._{train,val}_batch)

        on_{train,validation}_epoch_end
            -> eval mode + autocast + no_grad
            -> forward(images) -> [B, HEAD_CHANNELS, ...] unified head
            -> _log_predictions(tb, ctx, ...)   # heads.py orchestrator

    All tags live under ``{stage}/{mode}/...`` where
    ``stage`` ∈ {``train``, ``val``} and ``mode`` = ``"automatic"``.
    The affinity panels show a curated subset of offsets (all attractive
    nearest-neighbours plus a few long-range repulsive ones)::

        {stage}/automatic/true/image
        {stage}/automatic/true/label
        {stage}/automatic/true/aff/{offset}                     (3-D only)
        {stage}/automatic/true/wan_decoder                      (Cosmos + VAE only)
        {stage}/automatic/pred/sem
        {stage}/automatic/pred/raw
        {stage}/automatic/pred/aff/{offset}
        {stage}/automatic/pred/label/pre                        (Mutex Watershed, 3-D)
        {stage}/automatic/pred/label/mul                        (× predicted sem)

    This matches the scalar hierarchy emitted by
    :class:`brainbow.modules.base.BaseCircuitModule`
    (``{stage}/automatic/{head}/loss[/<component>]`` and
    ``{stage}/automatic/{head}/metric/<name>``) so each head's images
    and scalars collapse into the same TensorBoard group.

    Args:
        every_n_epochs: log every *n* epochs (default 1).
        max_images: maximum batch elements to log (default 4).
        spatial_dims: 2 or 3 — controls central-slice extraction for 3-D.
        mode: mode name to place after the stage (default ``"automatic"``).
        projection_algorithm: Manifold reducer for instance embeddings.
            One of ``"pca"`` (default, linear), ``"svd"`` (linear, no
            centering), or ``"umap"`` (non-linear, highlights local
            cluster structure but ~10-100× slower).
        projection_backend: Backend for the projection.  ``"auto"`` picks
            cuML on CUDA inputs when RAPIDS is available, else torch /
            umap-learn.  Explicit choices: ``"cuml"`` (forces GPU),
            ``"torch"`` (pca/svd CPU or CUDA SVD), ``"umap-learn"``
            (forces CPU UMAP).
        geometry_style: Renderer family for the ``pred/dir`` and
            ``pred/cov`` panels.  ``"glyph"`` (default) draws
            matplotlib quiver arrows for ``dir`` and ellipse glyphs for
            ``cov`` -- the most literal reading.  ``"flow"`` uses a
            vectorised optical-flow-style HSV colour map (no
            matplotlib, ~10× faster).  Both styles composite onto the
            raw EM with the soft predicted sem as the per-pixel blend
            weight.  Validated against
            :data:`brainbow.callbacks.tensorboard.heads.GEOMETRY_STYLES`
            at construction time so a typo in the YAML fails fast.
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        max_images: int = 4,
        spatial_dims: int = 2,
        mode: str = "automatic",
        projection_algorithm: str = "pca",
        projection_backend: str = "auto",
        geometry_style: str = "glyph",
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(every_n_epochs, 1)
        self.max_images = max_images
        self.spatial_dims = spatial_dims
        self.mode = mode
        self.projection_algorithm = projection_algorithm
        self.projection_backend = projection_backend
        self.geometry_style = geometry_style
        self._train_batch: Optional[Dict[str, torch.Tensor]] = None
        self._val_batch: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Batch capture (ALL ranks, first batch of each epoch)
    # ------------------------------------------------------------------
    # Captured on every rank -- under FSDP the epoch-end forward must run
    # on all ranks (the model's per-layer all-gathers are collectives), so
    # every rank needs its own batch to feed.  Only rank 0 logs the result.

    @staticmethod
    def _detach_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx == 0:
            self._train_batch = self._detach_batch(batch)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            self._val_batch = self._detach_batch(batch)

    # ------------------------------------------------------------------
    # Epoch-end dispatch
    # ------------------------------------------------------------------

    def _get_tb(self, trainer: pl.Trainer):
        """Return TensorBoard SummaryWriter or None."""
        logger = trainer.logger
        if logger is None:
            return None
        tb = getattr(logger, "experiment", None)
        if tb is None or not hasattr(tb, "add_images"):
            return None
        return tb


    @torch.no_grad()
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # ALL ranks must run ``_run_visualization`` (its model forward does
        # FSDP per-layer all-gathers -- collectives that need every rank);
        # only rank 0 actually logs (``tb`` is None elsewhere, and the
        # forward bails before the logging stage on non-rank-0).  Do NOT
        # early-return on non-rank-0, or the collectives desync and hang.
        # The epoch gate is rank-consistent (same ``current_epoch``).
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            self._train_batch = None
            return
        if self._train_batch is None:
            return

        tb = self._get_tb(trainer)  # real on rank 0, None elsewhere
        was_training = pl_module.training
        pl_module.eval()
        try:
            self._run_visualization(tb, trainer, pl_module, self._train_batch, stage="train")
        finally:
            self._train_batch = None
            if was_training:
                pl_module.train()

    @torch.no_grad()
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # See on_train_epoch_end: ALL ranks run the forward (collective);
        # only rank 0 logs.  No non-rank-0 early-return.
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            self._val_batch = None
            return
        if self._val_batch is None:
            return

        tb = self._get_tb(trainer)  # real on rank 0, None elsewhere
        was_training = pl_module.training
        pl_module.eval()
        try:
            self._run_visualization(tb, trainer, pl_module, self._val_batch, stage="val")
        finally:
            self._val_batch = None
            if was_training:
                pl_module.train()

    # ------------------------------------------------------------------
    # Internal: forward + dispatch
    # ------------------------------------------------------------------

    def _run_visualization(
        self, tb, trainer, pl_module, batch, *, stage: str,
    ):
        epoch = pl_module.current_epoch
        # Gate autocast on the actual device the module lives on, **not**
        # on ``torch.cuda.is_available()``: the latter is True whenever
        # any GPU is visible to the process, which would spuriously flip
        # CUDA autocast on when the user has chosen a CPU trainer.
        device_type = str(pl_module.device).split(":")[0]
        autocast_enabled = device_type == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            device_type=device_type, enabled=autocast_enabled,
        ):
            images = batch["image"].to(pl_module.device)
            if images.dim() == self.spatial_dims + 1:
                images = rearrange(images, "b ... -> b 1 ...")

            labels = batch["label"].to(pl_module.device)
            if labels.dim() == self.spatial_dims + 2:
                labels = rearrange(labels, "b 1 ... -> b ...")

            n = min(images.shape[0], self.max_images)
            # Route the head forward through the (FSDP/DDP-)WRAPPED root
            # module ``trainer.model`` -> ``LightningModule.forward`` ->
            # ``self.model``.  Under FSDP this triggers the root unit's
            # native param unshard (the frozen VAE + heads live there); a
            # direct ``pl_module.model(...)`` call bypasses it and sees
            # sharded 1-D weights ("weight should have at least three
            # dimensions").  Identical behaviour under DDP / single.
            fwd_module = getattr(trainer, "model", None) or pl_module
            head_pred = fwd_module(images[:n])

            # Wan-VAE reconstruction panel (Cosmos diagnostic) is a wrapper
            # METHOD, not the forward, so it can't ride the root unshard --
            # under FSDP its params would be sharded.  Skip it under FSDP
            # (panel suppressed); keep it on DDP / single where params are
            # full-shape.
            try:
                from pytorch_lightning.strategies import FSDPStrategy
                is_fsdp = isinstance(getattr(trainer, "strategy", None), FSDPStrategy)
            except Exception:
                is_fsdp = False
            wan_decoder = (
                None if is_fsdp
                else getattr(pl_module.model, "wan_decoder_output", None)
            )
            wan_decoder_pred = (
                wan_decoder(images[:n]) if callable(wan_decoder) else None
            )

        # Autocast-returned tensors may be bf16/fp16.  Cast back to fp32
        # so every downstream op in this callback (colour LUTs, eigh
        # decomposition for the cov overlay, TB image encoders) operates
        # in a single, display-friendly dtype.
        head_pred = head_pred.float()
        if wan_decoder_pred is not None:
            wan_decoder_pred = wan_decoder_pred.float()

        # The forward above is the only collective part (FSDP per-layer
        # all-gathers), and every rank has now run it.  Non-rank-0 ranks
        # have no TensorBoard writer, so they bail here -- the remaining
        # work (clustering, manifold projection, rendering, ``tb.add_*``)
        # is purely local and only the master needs to do it.
        if tb is None:
            return

        # Mutex Watershed instance segmentation of the predicted affinities
        # (3-D only).  Run on the full head BEFORE central-slice extraction
        # so the agglomeration sees the whole volume, then slice the label
        # map for display.  Restricted to the predicted foreground (sem).
        agglomerator = getattr(pl_module, "agglomerator", None)
        seg_pred_2d = None
        offsets = AFFINITY_OFFSETS
        n_attractive = N_ATTRACTIVE
        if agglomerator is not None:
            offsets = getattr(agglomerator, "offsets", offsets)
            n_attractive = getattr(agglomerator, "n_attractive", n_attractive)
            if self.spatial_dims == 3:
                aff = head_pred[:, AFF_SLICE].float()
                sem_fg = head_pred[:, SEM_SLICE][:, 0] > 0.5
                seg_3d = agglomerator(aff, sem_fg)            # [n, D, H, W] long
                seg_pred_2d = rearrange(
                    _to_2d(rearrange(seg_3d, "b ... -> b 1 ...")),
                    "b 1 ... -> b ...",
                )

        images_2d = _to_2d(images[:n])
        labels_2d = rearrange(
            _to_2d(rearrange(labels[:n], "b ... -> b 1 ...")),
            "b 1 ... -> b ...",
        )
        wan_decoder_2d = (
            _to_2d(wan_decoder_pred) if wan_decoder_pred is not None else None
        )

        ctx = TagContext(stage=stage, mode=self.mode)
        _log_predictions(
            tb, ctx, images_2d, labels_2d,
            head_pred, self.spatial_dims, n, epoch,
            offsets=offsets,
            n_attractive=n_attractive,
            labels_3d=labels[:n] if self.spatial_dims == 3 else None,
            seg_pred_2d=seg_pred_2d,
            wan_decoder_2d=wan_decoder_2d,
        )
        del head_pred, wan_decoder_pred


__all__ = ["ImageLogger"]
