"""
Base Lightning module shared by every Brainbow training recipe.

All modules in :mod:`brainbow.modules` (``Vista3DModule``,
``CosmosTransfer3DModule``) run the same training / evaluation loop:

* forward the volume through the wrapper (``self.model``)
* apply :class:`brainbow.losses.CombinedLoss`
* accumulate per-head metrics during validation / test
* all-reduce once per epoch and log under a single scalar hierarchy

This module captures that loop so the subclasses only have to declare
:attr:`_model_cls`, :attr:`_loss_cls` and (optionally) override
``configure_optimizers`` / freeze-scheduling hooks.

Scalar tag hierarchy
--------------------
All scalars live under ``{stage}/{mode}/...`` where ``stage`` is
``train`` | ``val`` | ``test`` and ``mode`` is ``"automatic"`` (the only
supported mode today — structured so ``"prompted"`` can slot in later)::

    {stage}/{mode}/loss                         # global total
    {stage}/{mode}/{head}/loss                  # per-head total
    {stage}/{mode}/{head}/loss/{component}      # per-head loss breakdown
    {stage}/{mode}/{head}/metric/{name}         # per-head eval metric
    {stage}/{mode}/eff_w/{head}                 # learned task weight

This matches the image tags emitted by
:class:`brainbow.callbacks.tensorboard.ImageLogger` so images and
scalars for a given head collapse into the same TensorBoard group.
"""

import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange, reduce

from brainbow.inference.clusterer import build_clusterer

logger = logging.getLogger(__name__)
from brainbow.metrics import (
    compute_per_batch_ari,
    compute_per_batch_ami,
    compute_per_batch_dice,
    compute_per_batch_iou,
    compute_per_batch_voi,
    compute_per_batch_ted,
)
from brainbow.losses import EMB_SLICE, SEM_SLICE

_SPATIAL_AXES = {2: "h w", 3: "d h w"}


def _head_field(
    loss_config: Dict[str, Any], head: str, field: str, default: Any = None,
) -> Any:
    """Read a sub-loss field from a nested ``weight_<field>`` mapping."""
    v = loss_config.get(f"weight_{head}")
    if isinstance(v, Mapping) and field in v:
        return v[field]
    return default


class BaseCircuitModule(pl.LightningModule):
    """Shared Lightning loop for Brainbow's segmentation modules.

    Subclasses **must** define:

    * :attr:`_model_cls`  -- model wrapper class (called via
      :meth:`_build_model` with the ``model_config`` dict)
    * :attr:`_loss_cls`   -- loss class (typically
      :class:`brainbow.losses.CombinedLoss`)

    Subclasses **may** override:

    * :attr:`_SPATIAL_DIMS` -- 2 or 3 (default 3)
    * :attr:`_MODE`         -- scalar-tag segment after ``stage`` (default
      ``"automatic"``)
    * :meth:`_build_model`        -- construct the wrapper (default
      forwards every ``model_config`` entry as kwargs)
    * :meth:`configure_optimizers` -- keeps the default AdamW + optional
      cosine schedule if not overridden
    """

    _SPATIAL_DIMS: int = 3
    _MODE: str = "automatic"
    _model_cls: type
    _loss_cls: type

    # Populated by ``__init_subclass__`` based on ``_SPATIAL_DIMS``.
    _EXPAND_PATTERN: str
    _SQUEEZE_PATTERN: str

    # ------------------------------------------------------------------
    # Class bookkeeping
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dims = getattr(cls, "_SPATIAL_DIMS", None)
        if dims is None:
            return
        if dims not in _SPATIAL_AXES:
            raise ValueError(
                f"{cls.__name__}._SPATIAL_DIMS={dims} is invalid. "
                f"Must be one of {sorted(_SPATIAL_AXES)}."
            )
        axes = _SPATIAL_AXES[dims]
        cls._EXPAND_PATTERN = f"b {axes} -> b 1 {axes}"
        cls._SQUEEZE_PATTERN = f"b 1 {axes} -> b {axes}"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs:
            warnings.warn(
                f"{type(self).__name__} ignoring unknown kwargs: {sorted(kwargs)}",
                stacklevel=2,
            )

        self.optimizer_config = dict(optimizer_config or {})
        self.training_config = dict(training_config or {})
        loss_config = dict(loss_config or {})

        self.model = self._build_model(dict(model_config or {}))
        self.criterion = self._loss_cls(**loss_config)

        clusterer_config = dict(self.training_config.get("clusterer", {}) or {})
        clusterer_name = clusterer_config.pop("name", "soft_meanshift")
        clusterer_config.setdefault(
            "bandwidth",
            _head_field(loss_config, "emb", "delta_v", default=0.5),
        )
        clusterer_config.setdefault(
            "normalize_embeddings",
            _head_field(loss_config, "emb", "normalize_embeddings", default=False),
        )
        self.clusterer = build_clusterer(clusterer_name, **clusterer_config)

        self._eval_accum: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

    def _build_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        """Instantiate the wrapper.

        Default: forward every ``model_config`` entry as kwargs to
        :attr:`_model_cls`.  Cosmos overrides this to add freeze-schedule
        + backbone bookkeeping; Vista keeps the default.
        """
        return self._model_cls(**model_config)

    # ------------------------------------------------------------------
    # Tag helpers
    # ------------------------------------------------------------------

    def _scalar_prefix(self, stage: str) -> str:
        """Return ``"{stage}/{mode}"``, e.g. ``"train/automatic"``."""
        return f"{stage}/{self._MODE}"

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kw: Any) -> Dict[str, torch.Tensor]:
        return self.model(x, **kw)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_meta_tensor(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Strip MONAI MetaTensor subclasses at the batch boundary.

        MetaTensor's ``__torch_function__`` override can interfere with
        mixed-dtype backward passes; plain ``torch.Tensor`` is safer.
        """
        return {
            k: v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def _prepare_targets(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Build the targets dict consumed by ``self.criterion``.

        Also pre-builds ``targets["_cached_targets"]`` inside this
        no-grad scope so avg / affinity / geometry target ops don't pay
        autograd-tape overhead on every step.
        """
        ndim_with_channel = self._SPATIAL_DIMS + 2
        squeeze = self._SQUEEZE_PATTERN

        labels = batch["label"]
        if labels.dim() == ndim_with_channel:
            labels = rearrange(labels, squeeze)

        # Binary FG/BG semantic target: foreground is anything with a
        # positive instance id.  No multi-class semantic supervision is
        # populated by any current datamodule.
        targets: Dict[str, Any] = {"labels": labels}
        needs_raw = (
            getattr(self.criterion, "weight_raw", 0.0) > 0
        )
        if "image" in batch and needs_raw:
            targets["raw_image"] = batch["image"]
        for key in ("label_direction", "label_covariance"):
            if key in batch:
                targets[key] = batch[key]
        targets["_cached_targets"] = self.criterion.build_targets(
            targets["labels"], targets,
        )
        return targets

    def _expand_image_channel(self, images: torch.Tensor) -> torch.Tensor:
        """Ensure a singleton channel axis (``[B, D, H, W] → [B, 1, D, H, W]``)."""
        if images.dim() == self._SPATIAL_DIMS + 1:
            return rearrange(images, self._EXPAND_PATTERN)
        return images

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int,
    ) -> Optional[torch.Tensor]:
        batch = self._strip_meta_tensor(batch)
        images = self._expand_image_channel(batch["image"])

        # ``_prepare_targets`` is @no_grad and builds ``_cached_targets``
        # so avg / aff / geometry target precompute ops don't pay
        # autograd-tape overhead on every step.
        targets = self._prepare_targets(batch)

        head = self.model(images)
        losses = self.criterion(head, targets)
        total_loss = losses["loss"]

        # Finite-loss guard.  ``total_loss.isnan().any() or .isinf().any()``
        # would force a device→host sync **every step** (each ``.any()``
        # materialises a Python bool).  With ``gradient_clip_val=1.0`` and
        # ``bf16-mixed`` already in place, the guard is belt-and-suspenders;
        # we run it on a configurable cadence instead.  Default cadence is
        # ``training.log_every_n_steps`` so it lines up with TB logging
        # already paying for a sync.
        check_every = int(self.training_config.get(
            "check_loss_finite_every_n_steps",
            self.training_config.get("log_every_n_steps", 100),
        ))
        if check_every > 0 and self.global_step % check_every == 0:
            if not torch.isfinite(total_loss).all():
                nan_keys = [
                    k for k, v in losses.items()
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                ]
                warnings.warn(
                    f"NaN/Inf total loss at step {self.global_step} — "
                    f"skipping backward (keys={nan_keys}).",
                    stacklevel=2,
                )
                return None

        prefix = self._scalar_prefix("train")
        bs = images.shape[0]
        for name, value in losses.items():
            # ``loss`` (the global total) is the only scalar we surface on
            # the progress bar / per-step; the rest are epoch-averaged.
            # Field entries whose weight is zero are absent from
            # ``losses``; no extra filter is needed here.
            is_total = name == "loss"
            self.log(
                f"{prefix}/{name}", value,
                on_step=is_total,
                on_epoch=True,
                prog_bar=is_total,
                batch_size=bs,
            )

        return total_loss

    # ------------------------------------------------------------------
    # Evaluation — accumulate per-batch, all-reduce once per epoch
    # ------------------------------------------------------------------

    def _accum(self, name: str, value: Any, weight: float) -> None:
        v = value.item() if isinstance(value, torch.Tensor) else float(value)
        acc = self._eval_accum[name]
        acc[0] += v * weight
        acc[1] += weight

    @torch.no_grad()
    def _eval_step_and_accumulate(
        self, batch: Dict[str, torch.Tensor], stage: str,
    ) -> None:
        batch = self._strip_meta_tensor(batch)
        images = self._expand_image_channel(batch["image"])

        targets = self._prepare_targets(batch)
        head = self.model(images)
        losses = self.criterion(head, targets)

        prefix = self._scalar_prefix(stage)
        bs = float(images.shape[0])
        for name, val in losses.items():
            self._accum(f"{prefix}/{name}", val, bs)

        self._accumulate_metrics(head, targets, prefix, bs)

        del head, losses

    def _accumulate_metrics(
        self,
        head: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        """Compute per-head classification / segmentation metrics.

        Metrics are computed from fixed slices of the unified head:
        semantic from ``SEM_SLICE`` and instance IDs from clustering the
        embedding slice ``EMB_SLICE``.
        """
        self._accumulate_semantic_metrics(head, targets, prefix, bs)
        self._accumulate_instance_metrics(head, targets, prefix, bs)

    def _accumulate_semantic_metrics(
        self,
        head_pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        sem_probs = head_pred[:, SEM_SLICE]
        sem_pred = (sem_probs[:, 0] > 0.5).long()
        sem_gt = (targets["labels"] > 0).long()
        metric = f"{prefix}/sem/metric"
        self._accum(
            f"{metric}/acc",
            reduce((sem_pred == sem_gt).float(), "b ... -> ", "mean"),
            bs,
        )
        self._accum(
            f"{metric}/iou",
            compute_per_batch_iou(sem_pred, sem_gt, num_classes=2), bs,
        )
        self._accum(
            f"{metric}/dice",
            compute_per_batch_dice(sem_pred, sem_gt, num_classes=2), bs,
        )

    def _accumulate_instance_metrics(
        self,
        head_pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        fg_mask = targets["labels"] > 0
        if not fg_mask.any():
            return
        ins_pred, _, _ = self.clusterer(head_pred[:, EMB_SLICE].float(), fg_mask)
        ins_gt = targets["labels"]
        metric = f"{prefix}/emb/metric"
        self._accum(f"{metric}/ari", compute_per_batch_ari(ins_pred, ins_gt), bs)
        self._accum(f"{metric}/ami", compute_per_batch_ami(ins_pred, ins_gt), bs)
        voi = compute_per_batch_voi(ins_pred, ins_gt)
        self._accum(f"{metric}/voi", voi.total, bs)
        self._accum(f"{metric}/voi_split", voi.split, bs)
        self._accum(f"{metric}/voi_merge", voi.merge, bs)
        self._accum(f"{metric}/ted", compute_per_batch_ted(ins_pred, ins_gt), bs)
        del ins_pred

    def _reduce_and_log_accum(self, stage: str) -> None:
        if not self._eval_accum:
            return

        names = sorted(self._eval_accum)
        sums = torch.tensor(
            [self._eval_accum[n][0] for n in names], device=self.device,
        )
        counts = torch.tensor(
            [self._eval_accum[n][1] for n in names], device=self.device,
        )

        if self.trainer.world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        prefix = self._scalar_prefix(stage)
        prog_bar_names = {
            f"{prefix}/loss",
            f"{prefix}/semantic/metric/acc",
            f"{prefix}/semantic/metric/iou",
            f"{prefix}/semantic/metric/dice",
            f"{prefix}/instance/metric/ari",
        }
        for i, name in enumerate(names):
            if counts[i] > 0:
                avg = (sums[i] / counts[i]).item()
                self.log(
                    name, avg,
                    prog_bar=(name in prog_bar_names),
                    sync_dist=False,
                    rank_zero_only=True,
                )

        self._eval_accum.clear()

    # ------------------------------------------------------------------
    # Validation / Test hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._eval_accum = defaultdict(lambda: [0.0, 0.0])

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int,
    ) -> None:
        self._eval_step_and_accumulate(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._reduce_and_log_accum("val")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_start(self) -> None:
        self._eval_accum = defaultdict(lambda: [0.0, 0.0])

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int,
    ) -> None:
        self._eval_step_and_accumulate(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._reduce_and_log_accum("test")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Optimizer (default: plain AdamW + optional cosine schedule)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        lr = self.optimizer_config.get("lr", 1e-4)
        wd = self.optimizer_config.get("weight_decay", 1e-5)

        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)

        return self._maybe_wrap_scheduler(optimizer)

    def _maybe_wrap_scheduler(self, optimizer: Any) -> Any:
        """Wrap the optimizer with a cosine-warmup schedule if configured."""
        sched_cfg = self.optimizer_config.get("scheduler", {})
        stype = str(sched_cfg.get("type", "cosine") or "").lower()

        if stype in ("cosine", "cosine_warmup"):
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR, LinearLR, SequentialLR,
            )

            warmup_epochs = sched_cfg.get("warmup_epochs", 5)
            t_max = sched_cfg.get("T_max", 100)
            eta_min = sched_cfg.get("eta_min", 1e-7)

            warmup = LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(t_max - warmup_epochs, 1), eta_min=eta_min,
            )
            scheduler = SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if stype:
            warnings.warn(
                f"Unknown scheduler type '{stype}', using no scheduler. "
                "Supported: 'cosine', 'cosine_warmup'.",
                stacklevel=2,
            )
        return optimizer
