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

_SPATIAL_AXES = {2: "h w", 3: "d h w"}


def _head_weight(
    loss_config: Dict[str, Any], head: str, default: float = 1.0,
) -> float:
    """Extract the scalar head weight from a (nested or flat) loss config."""
    v = loss_config.get(f"weight_{head}", default)
    if isinstance(v, Mapping):
        return float(v.get("weight", default))
    return float(v)


def _head_field(
    loss_config: Dict[str, Any], head: str, field: str, default: Any = None,
) -> Any:
    """Read a sub-loss field from either the nested head dict or the flat form.

    Checks ``loss_config["weight_<head>"][field]`` first (new nested
    form), then falls back to ``loss_config[field]`` (legacy flat form).
    """
    v = loss_config.get(f"weight_{head}")
    if isinstance(v, Mapping) and field in v:
        return v[field]
    return loss_config.get(field, default)


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

        disabled_heads = frozenset(
            name
            for name in ("semantic", "instance", "geometry", "boundary")
            if _head_weight(loss_config, name, default=1.0) == 0
        )
        self._disabled_heads = disabled_heads

        self.model = self._build_model(dict(model_config or {}))
        self.criterion = self._loss_cls(
            spatial_dims=self._SPATIAL_DIMS, **loss_config,
        )

        clusterer_config = dict(self.training_config.get("clusterer", {}) or {})
        clusterer_name = clusterer_config.pop("name", "soft_meanshift")
        clusterer_config.setdefault(
            "bandwidth",
            _head_field(loss_config, "instance", "delta_v", default=0.5),
        )
        clusterer_config.setdefault(
            "normalize_embeddings",
            _head_field(loss_config, "instance", "normalize_embeddings", default=False),
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

        Also pre-builds ``targets["_cached_weights"]`` (instance pixel
        weights + geometry targets) **inside this no-grad scope** so the
        autograd tape doesn't track those ops on the way to the model
        forward.  Boundary/membrane voxels are applied upstream by the
        data pipeline (``data.find_boundaries`` → ``FindBoundariesd``),
        not here.
        """
        ndim_with_channel = self._SPATIAL_DIMS + 2
        squeeze = self._SQUEEZE_PATTERN

        labels = batch["label"]
        if labels.dim() == ndim_with_channel:
            labels = rearrange(labels, squeeze)

        # Binary FG/BG semantic target: foreground is anything with a
        # positive instance id.  No multi-class semantic supervision is
        # populated by any current datamodule.
        targets: Dict[str, Any] = {
            "semantic_labels": (labels > 0).long(),
            "labels": labels,
        }
        needs_raw = (
            self.criterion.weight_geometry > 0
            or self.criterion.weight_boundary > 0
        )
        if "image" in batch and needs_raw:
            targets["raw_image"] = batch["image"]
        for key in ("label_direction", "label_covariance"):
            if key in batch:
                targets[key] = batch[key]
        # Build the per-step shared targets (geometry per-batch dirs/covs,
        # full 10-channel boundary target, 6-aff target shared with
        # InstanceLoss) once here, inside the @no_grad context, so the
        # ops don't pay autograd-tape overhead on every step.
        targets["_cached_targets"] = self.criterion._build_targets(
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

        # ``_prepare_targets`` is @no_grad and now also builds
        # ``_cached_weights`` so the geometry / instance precompute ops
        # don't pay autograd-tape overhead on every step.
        targets = self._prepare_targets(batch)

        predictions = self.model(images)
        losses = self.criterion(predictions, targets)
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
            # Disabled-head entries are already absent from ``losses`` —
            # ``CombinedLoss.forward`` only adds e.g. ``semantic/...``
            # keys when ``self.semantic_loss is not None`` — so we don't
            # need an extra disabled-head filter here.
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
        predictions = self.model(images)
        losses = self.criterion(predictions, targets)

        prefix = self._scalar_prefix(stage)
        bs = float(images.shape[0])
        for name, val in losses.items():
            self._accum(f"{prefix}/{name}", val, bs)

        self._accumulate_metrics(predictions, targets, prefix, bs)

        del predictions, losses

    def _accumulate_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        """Compute per-head classification / segmentation metrics.

        Heads that are absent from ``predictions`` (because
        ``weight_<head>=0``) are silently skipped — the numbers that show
        up in TensorBoard always reflect heads the deployed model will
        actually produce.
        """
        if "semantic" in predictions:
            self._accumulate_semantic_metrics(predictions, targets, prefix, bs)
        if "instance" in predictions:
            self._accumulate_instance_metrics(predictions, targets, prefix, bs)

    def _accumulate_semantic_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        # ``predictions["semantic"]`` is *already* a tensor of per-channel
        # sigmoid probabilities -- the model wrapper applies sigmoid to
        # the semantic head before anything downstream (loss, metrics,
        # tensorboard) sees it.
        sem_probs = predictions["semantic"]
        sem_loss = getattr(self.criterion, "semantic_loss", None)
        active = getattr(sem_loss, "active_classes", None) if sem_loss else None
        if active is not None and active < sem_probs.shape[1]:
            sem_probs = sem_probs[:, :active]

        if sem_probs.shape[1] == 1:
            sem_pred = (sem_probs[:, 0] > 0.5).long()
            n_cls = 2
        else:
            # Multi-channel sigmoid (multi-label).  ``argmax`` is monotone
            # under sigmoid so the class ranking matches what the old
            # logits path produced.
            sem_pred = sem_probs.argmax(dim=1)
            n_cls = sem_probs.shape[1]

        sem_gt = targets["semantic_labels"]
        head = f"{prefix}/semantic/metric"
        self._accum(
            f"{head}/acc",
            reduce((sem_pred == sem_gt).float(), "b ... -> ", "mean"),
            bs,
        )
        self._accum(
            f"{head}/iou",
            compute_per_batch_iou(sem_pred, sem_gt, num_classes=n_cls), bs,
        )
        self._accum(
            f"{head}/dice",
            compute_per_batch_dice(sem_pred, sem_gt, num_classes=n_cls), bs,
        )

    def _accumulate_instance_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        fg_mask = targets["labels"] > 0
        if not fg_mask.any():
            return
        ins_pred, _, _ = self.clusterer(
            predictions["instance"].float(), fg_mask,
        )
        ins_gt = targets["labels"]
        head = f"{prefix}/instance/metric"
        self._accum(f"{head}/ari", compute_per_batch_ari(ins_pred, ins_gt), bs)
        self._accum(f"{head}/ami", compute_per_batch_ami(ins_pred, ins_gt), bs)
        voi = compute_per_batch_voi(ins_pred, ins_gt)
        self._accum(f"{head}/voi", voi.total, bs)
        self._accum(f"{head}/voi_split", voi.split, bs)
        self._accum(f"{head}/voi_merge", voi.merge, bs)
        self._accum(f"{head}/ted", compute_per_batch_ted(ins_pred, ins_gt), bs)
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
