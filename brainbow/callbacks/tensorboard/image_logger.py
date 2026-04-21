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


class ImageLogger(pl.Callback):
    """Log sample images to TensorBoard at the end of every *n*-th epoch.

    Logs visualisations for both **training** and **validation** batches
    using **automatic** mode (image-only forward).

    Lifecycle (per epoch, rank-0 only)::

        on_{train,validation}_batch_end(batch_idx == 0)
            -> cache first batch on CPU (self._{train,val}_batch)

        on_{train,validation}_epoch_end
            -> eval mode + autocast + no_grad
            -> forward(images)
            -> optional build_boundary_target(labels, images)
            -> _log_predictions(tb, ctx, ...)   # heads.py orchestrator
                   -> _log_semantic / _log_instance
                      _log_geometry / _log_boundary

    All tags live under ``{stage}/{mode}/...`` where
    ``stage`` ∈ {``train``, ``val``} and ``mode`` = ``"automatic"``::

        {stage}/automatic/true/image
        {stage}/automatic/true/label
        {stage}/automatic/semantic/pred
        {stage}/automatic/instance/pred/{pca|svd|umap}
        {stage}/automatic/instance/pred/label
        {stage}/automatic/geometry/pred/{dir_centroid|cov|raw}
        {stage}/automatic/boundary/pred/{raw,min,avg,max}
        {stage}/automatic/boundary/true/{min,avg,max}
        {stage}/automatic/boundary/{pred,true}/{t,b,u,d,l,r}

    ``boundary/true/raw`` is intentionally **not** emitted: it would
    duplicate ``{stage}/automatic/true/image`` pixel-for-pixel (the
    boundary ``raw`` channel is literally the input image).

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
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        max_images: int = 4,
        spatial_dims: int = 2,
        mode: str = "automatic",
        projection_algorithm: str = "pca",
        projection_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(every_n_epochs, 1)
        self.max_images = max_images
        self.spatial_dims = spatial_dims
        self.mode = mode
        self.projection_algorithm = projection_algorithm
        self.projection_backend = projection_backend
        self._train_batch: Optional[Dict[str, torch.Tensor]] = None
        self._val_batch: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Batch capture (rank-0 only, first batch of each epoch)
    # ------------------------------------------------------------------

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
        if batch_idx == 0 and trainer.global_rank == 0:
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
        if batch_idx == 0 and trainer.global_rank == 0:
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
        if trainer.global_rank != 0:
            self._train_batch = None
            return
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        if self._train_batch is None:
            return
        tb = self._get_tb(trainer)
        if tb is None:
            return

        batch = self._train_batch
        was_training = pl_module.training
        pl_module.eval()
        try:
            self._run_visualization(tb, pl_module, batch, stage="train")
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
        if trainer.global_rank != 0:
            self._val_batch = None
            return
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        if self._val_batch is None:
            return
        tb = self._get_tb(trainer)
        if tb is None:
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            self._run_visualization(
                tb, pl_module, self._val_batch, stage="val",
            )
        finally:
            self._val_batch = None
            if was_training:
                pl_module.train()

    # ------------------------------------------------------------------
    # Internal: forward + dispatch
    # ------------------------------------------------------------------

    def _maybe_build_boundary_target(
        self,
        preds: Dict[str, torch.Tensor],
        criterion: Any,
        images: torch.Tensor,
        labels: torch.Tensor,
        n: int,
    ) -> Optional[torch.Tensor]:
        """Return the 16-channel boundary GT target if it should be logged.

        Gated by:
          * ``weight_boundary > 0`` on the criterion, AND
          * ``spatial_dims == 3`` (target construction uses z/y/x coords), AND
          * a ``"boundary"`` key present in the model predictions.

        Returns ``None`` otherwise (the GT panel is then skipped while
        the prediction panel is still emitted by :func:`_log_boundary`).
        """
        weight_boundary = float(getattr(criterion, "weight_boundary", 0.0) or 0.0)
        if weight_boundary <= 0.0 or self.spatial_dims != 3:
            return None
        if "boundary" not in preds:
            return None

        from brainbow.losses.boundary import build_boundary_target

        img_bcdhw = images[:n]
        img_for_target = (
            rearrange(img_bcdhw, "b 1 ... -> b ...")
            if img_bcdhw.dim() == self.spatial_dims + 2
            else img_bcdhw
        )
        lbl_for_target = (
            rearrange(labels[:n], "b 1 ... -> b ...")
            if labels[:n].dim() == self.spatial_dims + 2
            else labels[:n]
        )
        return build_boundary_target(lbl_for_target, img_for_target).float()

    def _run_visualization(
        self, tb, pl_module, batch, *, stage: str,
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
            preds_auto = pl_module.model(images[:n])

        # Autocast-returned tensors may be bf16/fp16.  Cast back to fp32
        # so every downstream op in this callback (colour LUTs,
        # matplotlib renderers, TB image encoders) operates in a single,
        # display-friendly dtype.
        preds_auto = {
            k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in preds_auto.items()
        }
        clusterer = (
            getattr(pl_module, "clusterer", None)
            or getattr(pl_module, "_clusterer", None)
        )

        criterion = getattr(pl_module, "criterion", None)
        geom_loss = getattr(criterion, "geometry_loss", None) if criterion else None
        dir_target = getattr(geom_loss, "dir_target", "centroid") if geom_loss else "centroid"
        sem_loss = getattr(criterion, "semantic_loss", None) if criterion else None
        active_classes = getattr(sem_loss, "active_classes", None) if sem_loss else None

        boundary_target = self._maybe_build_boundary_target(
            preds_auto, criterion, images, labels, n,
        )

        images_2d = _to_2d(images[:n])
        labels_2d = rearrange(
            _to_2d(rearrange(labels[:n], "b ... -> b 1 ...")),
            "b 1 ... -> b ...",
        )

        ctx = TagContext(stage=stage, mode=self.mode)
        _log_predictions(
            tb, ctx, images_2d, labels_2d,
            preds_auto, self.spatial_dims, n, epoch,
            clusterer=clusterer, dir_target=dir_target,
            active_classes=active_classes,
            projection_algorithm=self.projection_algorithm,
            projection_backend=self.projection_backend,
            boundary_target=boundary_target,
        )
        del preds_auto, boundary_target


__all__ = ["ImageLogger"]
