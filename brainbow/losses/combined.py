"""
Unified head loss for the 30-channel Vista-style task head.

Composes six per-field sub-losses plus two derived face-affinity terms
into a single dict-returning module.  See :mod:`brainbow.losses._common`
for the canonical channel layout and direction tables.

Field summary
-------------

::

    field   ch slice    activation   sub-loss             extras
    -----   ---------   ----------   ------------------   ----------------
    raw     [ 0,  1)    linear       L1 / MSE / Smooth-L1 vs the (clipped) input image
    sem     [ 1,  2)    sigmoid      BCE + Dice           vs (label > 0)
    dir     [ 2,  5)    linear       L1 / MSE / Smooth-L1 vs centroid-direction field, fg-only
    cov     [ 5, 11)    linear       L1 / MSE / Smooth-L1 vs upper-triangle covariance, fg-only
    avg     [11, 14)    linear       L1 / MSE / Smooth-L1 vs normalised centroid (z, y, x), fg-only
                                     + 12-ch ``aff_avg`` (BCE + Dice on derived face-affinity)
    emb     [14, 30)    linear       discriminative pull / push / norm (centroid-based)
                                     + 12-ch ``aff_emb`` (BCE + Dice on derived face-affinity)

The two ``aff_*`` paths share the same 12-channel binary aff target
``aff_target = label_aff(label, background=self.background)``.

Configuration schema
--------------------

Each ``weight_<field>`` argument is either a scalar (only the field
weight) or a mapping ``{weight: ..., **sub_kwargs}`` whose entries are
forwarded into the field's sub-loss configuration::

    weight_sem:
      weight: 1.0          # 0 disables the field entirely
      weight_ce: 1.0
      weight_dice: 1.0
      class_weights: [0.1]

A nested mapping that omits ``weight`` defaults to ``weight: 1.0`` --
a user who wrote a nested block clearly intended to enable the field;
silent disablement on a missing key would be a footgun.

Output dict
-----------

::

    loss                                   # global total
    loss/{field}                           # per-field total (weighted)
    loss/{field}/{sub}                     # per-field breakdown,
                                           # un-weighted (so each sub
                                           # value is comparable across
                                           # runs with different weights)

with ``{field}`` ∈ ``{raw, sem, dir, cov, avg, emb, aff_avg, aff_emb}``
and ``{sub}`` ∈ ``{ce, dice, pull, push, norm}`` where applicable.

The Lightning module prefixes everything with ``{stage}/{mode}/`` so a
TB tag like ``train/automatic/loss/aff_emb/dice`` sits next to its
image counterpart ``train/automatic/pred/emb/aff/01_t1`` etc.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from monai.losses import DiceLoss

from brainbow.losses._common import (
    AFF_CHANNELS,
    AVG_SLICE,
    COV_SLICE,
    DIR_SLICE,
    DIRECTIONS,
    EMB_SLICE,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    RAW_SLICE,
    SEM_SLICE,
    affinity_target,
    canonical_regression_name,
    regression_loss_fn,
    shift_replicate,
    soft_aff_from_field,
    stable_bce_on_probs,
)


HeadConfig = Union[float, int, Mapping[str, Any]]


def _split_field(cfg: HeadConfig) -> Tuple[float, Dict[str, Any]]:
    """Split ``weight_<field>`` into ``(weight, sub_kwargs)``.

    Scalar shorthand is supported: ``weight_sem: 1.0`` is the same as
    ``weight_sem: { weight: 1.0 }`` (no sub-knobs).  A nested mapping
    without ``weight:`` defaults to ``weight: 1.0`` -- a nested block
    almost certainly means "I want this field on".
    """
    if isinstance(cfg, Mapping):
        d = dict(cfg)
        return float(d.pop("weight", 1.0)), d
    return float(cfg), {}


# ---------------------------------------------------------------------------
# Avg target builder (per-instance normalised centroid, broadcast to fg)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_avg_target(labels: torch.Tensor) -> torch.Tensor:
    """Build the ``avg`` target ``[B, 3, D, H, W]`` from instance labels.

    For every foreground voxel the target is the instance's centroid
    ``(z, y, x)`` divided by ``(D, H, W)`` so the value lives in
    ``[0, 1]``.  Background voxels are zero.

    Centroid is computed in a single fully-vectorised pass per batch
    element (``scatter_add_`` + ``bincount``); no Python loop over
    voxels.
    """
    B, D, H, W = labels.shape
    norm = labels.new_tensor([D, H, W], dtype=torch.float32).clamp(min=1.0)
    target = labels.new_zeros(B, 3, D, H, W, dtype=torch.float32)

    for b in range(B):
        lbl_b = labels[b]
        fg = lbl_b > 0
        if not fg.any():
            continue
        _, inverse = torch.unique(lbl_b[fg], return_inverse=True)
        K = int(inverse.max().item()) + 1
        fg_idx = torch.nonzero(fg, as_tuple=False).to(torch.float32)  # [M, 3]

        sums = fg_idx.new_zeros(K, 3)
        sums.scatter_add_(0, repeat(inverse, "m -> m c", c=3), fg_idx)
        counts = (
            torch.bincount(inverse, minlength=K).float().clamp(min=1.0)
        )
        centroids = sums / rearrange(counts, "k -> k 1")

        voxel_avg = (centroids / norm)[inverse]                       # [M, 3]
        target[b][:, fg] = rearrange(voxel_avg, "m c -> c m")
    return target


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """Unified loss for the 30-channel head.

    Args:
        weight_raw / _sem / _dir / _cov / _avg / _emb /
            _aff_emb / _aff_avg:
            Field-level config -- scalar or
            ``{weight: ..., **sub_kwargs}``.  ``weight: 0`` disables
            the field entirely (its sub-loss is short-circuited and
            its scalar output is omitted from the return dict).
        background: Label value treated as background when building the
            12-channel aff target.  Default ``-1`` keeps every voxel
            (including ``FindBoundariesd``-zeroed boundary voxels) in
            the supervision footprint, which removes the checkerboard
            artifact along instance edges that ``background=0``
            otherwise produces.  Pass ``None`` to disable masking
            entirely.
        ignore_index: label value excluded from the semantic BCE term.
    """

    num_channels: int = HEAD_CHANNELS

    def __init__(
        self,
        weight_raw: HeadConfig = 1.0,
        weight_sem: HeadConfig = 1.0,
        weight_dir: HeadConfig = 1.0,
        weight_cov: HeadConfig = 1.0,
        weight_avg: HeadConfig = 1.0,
        weight_emb: HeadConfig = 1.0,
        weight_aff_emb: HeadConfig = 1.0,
        weight_aff_avg: HeadConfig = 1.0,
        *,
        background: Optional[int] = -1,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.background = int(background) if background is not None else None
        self.ignore_index = int(ignore_index)

        # ----- raw -----
        self.weight_raw, raw_kw = _split_field(weight_raw)
        self.loss_raw = canonical_regression_name(raw_kw.pop("loss", "l1"))
        self._raw_fn = regression_loss_fn(self.loss_raw)

        # ----- sem (sigmoid BCE + Dice) -----
        self.weight_sem, sem_kw = _split_field(weight_sem)
        self.sem_weight_ce = float(sem_kw.pop("weight_ce", 1.0))
        self.sem_weight_dice = float(sem_kw.pop("weight_dice", 1.0))
        self.sem_class_weights = sem_kw.pop("class_weights", None)
        if self.sem_class_weights is not None:
            pw = torch.tensor(
                list(map(float, self.sem_class_weights)),
                dtype=torch.float32,
            ).view(1, -1, 1, 1, 1)
            self.register_buffer("_sem_pos_weight", pw, persistent=False)
        else:
            self._sem_pos_weight = None
        self._sem_dice = DiceLoss(
            sigmoid=False, softmax=False,
            include_background=True, reduction="mean",
            batch=True,
        )

        # ----- dir / cov / avg (foreground-only L1 regression) -----
        self.weight_dir, dir_kw = _split_field(weight_dir)
        self.loss_dir = canonical_regression_name(dir_kw.pop("loss", "l1"))
        self._dir_fn = regression_loss_fn(self.loss_dir)

        self.weight_cov, cov_kw = _split_field(weight_cov)
        self.loss_cov = canonical_regression_name(cov_kw.pop("loss", "l1"))
        self._cov_fn = regression_loss_fn(self.loss_cov)

        self.weight_avg, avg_kw = _split_field(weight_avg)
        self.loss_avg = canonical_regression_name(avg_kw.pop("loss", "l1"))
        self._avg_fn = regression_loss_fn(self.loss_avg)

        # ----- emb (discriminative pull/push/norm) -----
        self.weight_emb, emb_kw = _split_field(weight_emb)
        self.emb_weight_pull = float(emb_kw.pop("weight_pull", 1.0))
        self.emb_weight_push = float(emb_kw.pop("weight_push", 1.0))
        self.emb_weight_norm = float(emb_kw.pop("weight_norm", 0.001))
        self.delta_v = float(emb_kw.pop("delta_v", 0.5))
        self.delta_d = float(emb_kw.pop("delta_d", 1.5))
        self.normalize_embeddings = bool(emb_kw.pop("normalize_embeddings", False))
        self.max_hard_pairs = int(emb_kw.pop("max_hard_pairs", 0))

        # ----- aff_emb / aff_avg (BCE + Dice on derived face-aff) -----
        self.weight_aff_emb, aff_emb_kw = _split_field(weight_aff_emb)
        self.weight_aff_avg, aff_avg_kw = _split_field(weight_aff_avg)
        # Defaults: BCE off, Dice on -- matches the tested configuration in
        # the previous boundary-head loss.  Each path can override.
        self.aff_emb_weight_ce = float(aff_emb_kw.pop("weight_ce", 0.0))
        self.aff_emb_weight_dice = float(aff_emb_kw.pop("weight_dice", 1.0))
        self.aff_emb_tau = float(aff_emb_kw.pop("tau", 1.0))
        self.aff_avg_weight_ce = float(aff_avg_kw.pop("weight_ce", 0.0))
        self.aff_avg_weight_dice = float(aff_avg_kw.pop("weight_dice", 1.0))
        self.aff_avg_tau = float(aff_avg_kw.pop("tau", 1.0))
        # Optional per-direction class_weights (12 channels) for the BCE
        # path; broadcastable over [B, 12, D, H, W].
        for path, kw, attr in (
            ("aff_emb", aff_emb_kw, "_aff_emb_pos_weight"),
            ("aff_avg", aff_avg_kw, "_aff_avg_pos_weight"),
        ):
            cw = kw.pop("class_weights", None)
            if cw is not None:
                if len(cw) != AFF_CHANNELS:
                    raise ValueError(
                        f"{path}.class_weights must have length "
                        f"{AFF_CHANNELS}; got {len(cw)}."
                    )
                pw = torch.tensor(
                    list(map(float, cw)), dtype=torch.float32,
                ).view(1, AFF_CHANNELS, 1, 1, 1)
                self.register_buffer(attr, pw, persistent=False)
            else:
                setattr(self, attr, None)
        self.aff_eps = float(aff_emb_kw.pop("aff_eps", 1e-5))
        self._aff_dice = DiceLoss(
            sigmoid=False, softmax=False,
            include_background=True, reduction="mean",
            batch=True,
            smooth_nr=self.aff_eps, smooth_dr=self.aff_eps,
        )

        # Warn about unused kwargs (typo-defence).
        for path, kw in (
            ("weight_raw", raw_kw),
            ("weight_sem", sem_kw),
            ("weight_dir", dir_kw),
            ("weight_cov", cov_kw),
            ("weight_avg", avg_kw),
            ("weight_emb", emb_kw),
            ("weight_aff_emb", aff_emb_kw),
            ("weight_aff_avg", aff_avg_kw),
        ):
            if kw:
                import warnings
                warnings.warn(
                    f"CombinedLoss: ignoring unknown {path} keys: "
                    f"{sorted(kw)}",
                    stacklevel=2,
                )

        # Cached zero scalar (lazy device migration; non-persistent).
        self.register_buffer(
            "_zero_scalar", torch.tensor(0.0), persistent=False,
        )

    def _zero(self, device: torch.device) -> torch.Tensor:
        z = self._zero_scalar
        if z.device != device:
            self._zero_scalar = z = torch.zeros((), device=device)
        return z

    # ------------------------------------------------------------------
    # Target precomputation (once per step, called by the Lightning module)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_targets(
        self,
        labels: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the per-step shared targets used by :meth:`forward`.

        Builds (only what's needed for the active fields):

        * ``avg``  -- ``[B, 3, D, H, W]`` per-instance centroid /
          ``(D, H, W)``.
        * ``aff``  -- ``[B, 12, D, H, W]`` binary 2nd-order face-affinity
          target derived from labels (shared by ``aff_emb`` and
          ``aff_avg``).
        * ``dir``  -- pulled from ``batch["label_direction"]`` if
          ``compute_geometry`` was set in the datamodule.  When the
          field is absent, a zero target is built lazily inside
          :meth:`_loss_dir` (the loss short-circuits if no fg voxels).
        * ``cov``  -- ditto for ``batch["label_covariance"]``.

        Returns a dict; keys absent here are interpreted by
        :meth:`forward` as "no cached target, build lazily".
        """
        out: Dict[str, torch.Tensor] = {}

        if self.weight_avg > 0:
            out["avg"] = build_avg_target(labels.long())

        if self.weight_aff_emb > 0 or self.weight_aff_avg > 0:
            out["aff"] = affinity_target(labels.long(), background=self.background)

        if batch is not None:
            if self.weight_dir > 0 and "label_direction" in batch:
                out["dir"] = batch["label_direction"]
            if self.weight_cov > 0 and "label_covariance" in batch:
                out["cov"] = batch["label_covariance"]

        return out

    # ------------------------------------------------------------------
    # Sub-losses
    # ------------------------------------------------------------------

    def _loss_raw(
        self, pred: torch.Tensor, raw_image: torch.Tensor,
    ) -> torch.Tensor:
        """Dense L1 / MSE / Smooth-L1 reconstruction over every voxel.

        ``pred`` is linear (no sigmoid).  The target is the (normalised)
        input image, taken as-is -- no clamp -- so a faithfully
        normalised input drives a faithful regression.
        """
        if raw_image.dim() == 4:
            raw_image = rearrange(raw_image, "b ... -> b 1 ...")
        return self._raw_fn(pred.float(), raw_image.detach().to(torch.float32))

    def _loss_sem(
        self, probs: torch.Tensor, labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """BCE + Dice on the binary semantic head (post-sigmoid).

        Returns a dict ``{loss, ce, dice}`` with un-weighted sub-values.
        """
        target = rearrange(
            (labels > 0).float(), "b ... -> b 1 ...",
        )
        valid = (labels != self.ignore_index).float()
        valid_mask = rearrange(valid, "b ... -> b 1 ...")

        ce = self._zero(probs.device)
        if self.sem_weight_ce > 0:
            per_voxel = stable_bce_on_probs(
                probs, target, pos_weight=self._sem_pos_weight,
            )
            per_voxel = per_voxel * valid_mask.to(per_voxel.dtype)
            denom = valid_mask.sum().clamp(min=1.0) * per_voxel.shape[1]
            ce = per_voxel.sum() / denom

        dice = self._zero(probs.device)
        if self.sem_weight_dice > 0:
            dice = self._sem_dice(probs, target * valid_mask)

        total = self.sem_weight_ce * ce + self.sem_weight_dice * dice
        return {"loss": total, "ce": ce, "dice": dice}

    def _loss_fg_regression(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fg: torch.Tensor,
        fn,
    ) -> torch.Tensor:
        """Foreground-only regression mean.

        ``pred`` and ``target`` are ``[B, C, D, H, W]``; ``fg`` is
        ``[B, D, H, W]`` bool.  The mean is taken over all foreground
        voxels and all channels.  Returns a 0-d zero scalar when no
        voxel is foreground (avoids a NaN division).
        """
        n_fg = fg.sum()
        if n_fg.item() == 0:
            return pred.new_zeros(())
        per_voxel = fn(pred.float(), target.to(pred.dtype), reduction="none")
        fg_mask = rearrange(fg.float(), "b ... -> b 1 ...").expand_as(per_voxel)
        return (per_voxel * fg_mask).sum() / (n_fg.float() * pred.shape[1])

    def _loss_emb(
        self, embed: torch.Tensor, labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Discriminative pull / push / norm on the embedding field.

        Pull: per-instance hinge² toward the mean centroid (margin
        ``delta_v``).  Push: pairwise hinge² between mean centroids
        (margin ``2 * delta_d``).  Norm: L2 regulariser on the
        centroids.  All averaged across batch elements that contain at
        least one foreground voxel.
        """
        device = embed.device
        emb_flat = rearrange(embed, "b e ... -> b e (...)")
        if self.normalize_embeddings:
            emb_flat = torch.nn.functional.normalize(emb_flat, dim=1, eps=1e-6)
        lbl_flat = rearrange(labels, "b ... -> b (...)")

        L_pull = torch.zeros((), device=device)
        L_push = torch.zeros((), device=device)
        L_norm = torch.zeros((), device=device)
        n_valid = 0

        for b in range(embed.shape[0]):
            fg = lbl_flat[b] > 0
            if not fg.any():
                continue
            n_valid += 1
            _, inverse = torch.unique(lbl_flat[b][fg], return_inverse=True)
            K = int(inverse.max().item()) + 1
            E = emb_flat.shape[1]
            emb_fg = rearrange(emb_flat[b, :, fg], "e m -> m e").float()

            # Mean centroid per instance.  Intentionally NOT @no_grad:
            # the centroid is part of the discriminative loss's autograd
            # graph (push / norm rely on gradients flowing through it).
            sums = torch.zeros(K, E, device=device, dtype=torch.float32)
            sums.scatter_add_(0, repeat(inverse, "m -> m e", e=E), emb_fg)
            counts = (
                torch.bincount(inverse, minlength=K).float().clamp(min=1.0)
            )
            centers = sums / rearrange(counts, "k -> k 1")

            # Pull: voxel -> centroid hinge².
            ctr_per_voxel = centers[inverse]                          # [M, E]
            diff = emb_fg - ctr_per_voxel
            dist = reduce(diff ** 2, "m e -> m", "sum").clamp(min=1e-12).sqrt()
            per_voxel = (dist - self.delta_v).clamp(min=0).pow(2)
            pull_sums = torch.zeros(K, device=device, dtype=torch.float32)
            pull_sums.scatter_add_(0, inverse, per_voxel)
            L_pull = L_pull + (pull_sums / counts).mean()

            # Push: pairwise centroid hinge².  ``max_hard_pairs`` keeps
            # only the top-k hardest pairs (smallest distance) so the
            # backward gradient is bounded; the forward still
            # materialises the K x K x E difference tensor.
            if K > 1:
                pw = (
                    rearrange(centers, "i e -> i 1 e")
                    - rearrange(centers, "j e -> 1 j e")
                )
                pw_dist = (
                    reduce(pw ** 2, "i j e -> i j", "sum")
                    .clamp(min=1e-12).sqrt()
                )
                triu = torch.triu_indices(K, K, offset=1, device=device)
                hinge = (
                    2 * self.delta_d - pw_dist[triu[0], triu[1]]
                ).clamp(min=0).pow(2)
                if self.max_hard_pairs > 0 and hinge.numel() > self.max_hard_pairs:
                    hinge, _ = hinge.topk(self.max_hard_pairs)
                L_push = L_push + hinge.mean()

            # Norm: L2 on centroids (skip when normalize_embeddings=True
            # since centroids are already on the unit sphere).
            if not self.normalize_embeddings:
                L_norm = L_norm + (
                    reduce(centers ** 2, "k e -> k", "sum")
                    .clamp(min=1e-12).sqrt().mean()
                )

        n = max(n_valid, 1)
        pull = L_pull / n
        push = L_push / n
        norm = L_norm / n
        total = (
            self.emb_weight_pull * pull
            + self.emb_weight_push * push
            + self.emb_weight_norm * norm
        )
        return {"loss": total, "pull": pull, "push": push, "norm": norm}

    def _loss_aff_path(
        self,
        field: torch.Tensor,
        labels: torch.Tensor,
        aff_target: torch.Tensor,
        tau: float,
        weight_ce: float,
        weight_dice: float,
        pos_weight: Optional[torch.Tensor],
        *,
        normalize_field: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generic 12-channel derived-affinity loss for a continuous field.

        Used by both ``aff_emb`` (field = embedding, 16-D) and
        ``aff_avg`` (field = avg, 3-D).  The kernel
        ``exp(-tau * sum_c |field[v] - shift_replicate(field[v], dir)|)``
        produces a soft 12-channel face-affinity score that is
        supervised against the binary ``aff_target``, masked to
        foreground-foreground face pairs (so background-background
        pairs don't contaminate the dice numerator / denominator).
        """
        if field.dim() != 5:
            raise ValueError(
                f"_loss_aff_path: field must be [B, C, D, H, W]; got "
                f"{tuple(field.shape)}."
            )
        if normalize_field:
            field = torch.nn.functional.normalize(field, p=2, dim=1, eps=1e-6)

        aff_pred = soft_aff_from_field(field, tau=tau)        # [B, 12, D, H, W]
        aff_t = aff_target.to(dtype=aff_pred.dtype, device=aff_pred.device)

        with torch.no_grad():
            if self.background is None:
                fg = torch.ones_like(labels, dtype=aff_pred.dtype)
            else:
                fg = (labels != self.background).to(dtype=aff_pred.dtype)
            pair_mask = torch.stack([
                fg * shift_replicate(fg, axis, shift)
                for _, axis, shift in DIRECTIONS
            ], dim=1)                                         # [B, 12, D, H, W]

        ce = self._zero(field.device)
        if weight_ce > 0:
            per_voxel = stable_bce_on_probs(
                aff_pred, aff_t, pos_weight=pos_weight,
            )
            per_voxel = per_voxel * pair_mask.to(per_voxel.dtype)
            denom = pair_mask.sum().clamp(min=1.0)
            ce = per_voxel.sum() / denom

        dice = self._zero(field.device)
        if weight_dice > 0:
            dice = self._aff_dice(aff_pred * pair_mask, aff_t * pair_mask)

        total = weight_ce * ce + weight_dice * dice
        return {"loss": total, "ce": ce, "dice": dice}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        head: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run all enabled sub-losses and aggregate.

        Args:
            head: ``[B, 30, D, H, W]`` model output.  Activation policy
                applied by the wrapper: sigmoid on ch 1 (sem); linear
                everywhere else.
            targets: Dict.  Required keys::

                ``labels``      [B, D, H, W] integer instance ids.

                Optional / per-field::

                ``raw_image``    [B, D, H, W] or [B, 1, D, H, W];
                                 needed iff ``weight_raw > 0``.
                ``label_direction`` [B, 3, D, H, W];  ``weight_dir > 0``.
                ``label_covariance``[B, 6, D, H, W]; ``weight_cov > 0``.
                ``_cached_targets`` dict from :meth:`build_targets`;
                                 used as a per-step cache so the
                                 12-aff target / avg target / dir-cov
                                 lookups don't run twice when both the
                                 loss and image-logger consume them.

        Returns:
            Dict whose keys are listed in the module docstring.
            Per-field totals are emitted only for active (``weight > 0``)
            fields.  ``loss`` is the global weighted sum.
        """
        if head.shape[1] != HEAD_CHANNELS:
            raise ValueError(
                f"CombinedLoss expects head with {HEAD_CHANNELS} channels; "
                f"got {head.shape[1]}."
            )
        labels = targets["labels"]
        device = head.device
        zero = self._zero(device)
        out: Dict[str, torch.Tensor] = {}

        # ----- assemble cached / on-the-fly targets -----
        cached: Dict[str, torch.Tensor] = targets.get("_cached_targets") or {}
        if not cached and (
            self.weight_avg > 0 or self.weight_aff_emb > 0 or self.weight_aff_avg > 0
        ):
            cached = self.build_targets(labels, batch=targets)

        # Promote `targets` keys for direction / covariance into the
        # cache; a fresh `cached` dict from `build_targets` already has
        # them but a user-supplied one might not.
        for k in ("dir", "cov"):
            if k not in cached and self._field_active(k):
                src = {"dir": "label_direction", "cov": "label_covariance"}[k]
                if src in targets:
                    cached[k] = targets[src]

        # ----- per-field sub-losses -----
        total = zero.clone()
        fg = labels > 0

        # raw
        if self.weight_raw > 0:
            raw_pred = head[:, RAW_SLICE]
            raw_image = targets.get("raw_image")
            if raw_image is None:
                raise KeyError(
                    "CombinedLoss requires `targets['raw_image']` when "
                    "weight_raw > 0; pass the normalised image."
                )
            l_raw = self._loss_raw(raw_pred, raw_image)
            out["loss/raw"] = l_raw
            total = total + self.weight_raw * l_raw

        # sem
        if self.weight_sem > 0:
            sem_pred = head[:, SEM_SLICE]
            sem = self._loss_sem(sem_pred, labels)
            out["loss/sem"] = sem["loss"]
            if self.sem_weight_ce > 0:
                out["loss/sem/ce"] = sem["ce"]
            if self.sem_weight_dice > 0:
                out["loss/sem/dice"] = sem["dice"]
            total = total + self.weight_sem * sem["loss"]

        # dir
        if self.weight_dir > 0:
            dir_pred = head[:, DIR_SLICE]
            dir_target = cached.get("dir")
            if dir_target is None:
                # No precomputed direction field; treat as zero target
                # so the sub-loss is well-defined but does nothing
                # useful.  Better than crashing: dropping `compute_
                # geometry` should disable the field via its weight.
                l_dir = zero
            else:
                l_dir = self._loss_fg_regression(
                    dir_pred, dir_target, fg, self._dir_fn,
                )
            out["loss/dir"] = l_dir
            total = total + self.weight_dir * l_dir

        # cov
        if self.weight_cov > 0:
            cov_pred = head[:, COV_SLICE]
            cov_target = cached.get("cov")
            if cov_target is None:
                l_cov = zero
            else:
                l_cov = self._loss_fg_regression(
                    cov_pred, cov_target, fg, self._cov_fn,
                )
            out["loss/cov"] = l_cov
            total = total + self.weight_cov * l_cov

        # avg
        if self.weight_avg > 0:
            avg_pred = head[:, AVG_SLICE]
            avg_target = cached.get("avg")
            if avg_target is None:
                avg_target = build_avg_target(labels.long())
            l_avg = self._loss_fg_regression(
                avg_pred, avg_target, fg, self._avg_fn,
            )
            out["loss/avg"] = l_avg
            total = total + self.weight_avg * l_avg

        # emb (discriminative)
        if self.weight_emb > 0:
            emb_pred = head[:, EMB_SLICE]
            emb = self._loss_emb(emb_pred, labels)
            out["loss/emb"] = emb["loss"]
            out["loss/emb/pull"] = emb["pull"]
            out["loss/emb/push"] = emb["push"]
            out["loss/emb/norm"] = emb["norm"]
            total = total + self.weight_emb * emb["loss"]

        # aff_emb / aff_avg -- both consume the same 12-channel
        # ``aff`` target.
        aff_target = cached.get("aff")
        if self.weight_aff_emb > 0:
            if aff_target is None:
                aff_target = affinity_target(
                    labels.long(), background=self.background,
                )
                cached["aff"] = aff_target
            emb_pred = head[:, EMB_SLICE]
            ae = self._loss_aff_path(
                emb_pred, labels, aff_target,
                tau=self.aff_emb_tau,
                weight_ce=self.aff_emb_weight_ce,
                weight_dice=self.aff_emb_weight_dice,
                pos_weight=self._aff_emb_pos_weight,
                normalize_field=self.normalize_embeddings,
            )
            out["loss/aff_emb"] = ae["loss"]
            if self.aff_emb_weight_ce > 0:
                out["loss/aff_emb/ce"] = ae["ce"]
            if self.aff_emb_weight_dice > 0:
                out["loss/aff_emb/dice"] = ae["dice"]
            total = total + self.weight_aff_emb * ae["loss"]

        if self.weight_aff_avg > 0:
            if aff_target is None:
                aff_target = affinity_target(
                    labels.long(), background=self.background,
                )
                cached["aff"] = aff_target
            avg_pred = head[:, AVG_SLICE]
            aa = self._loss_aff_path(
                avg_pred, labels, aff_target,
                tau=self.aff_avg_tau,
                weight_ce=self.aff_avg_weight_ce,
                weight_dice=self.aff_avg_weight_dice,
                pos_weight=self._aff_avg_pos_weight,
                normalize_field=False,
            )
            out["loss/aff_avg"] = aa["loss"]
            if self.aff_avg_weight_ce > 0:
                out["loss/aff_avg/ce"] = aa["ce"]
            if self.aff_avg_weight_dice > 0:
                out["loss/aff_avg/dice"] = aa["dice"]
            total = total + self.weight_aff_avg * aa["loss"]

        out["loss"] = total
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _field_active(self, name: str) -> bool:
        """Whether the named field will run in :meth:`forward`."""
        return getattr(self, f"weight_{name}") > 0

    def __repr__(self) -> str:
        active = [
            n for n in HEAD_LAYOUT
            if self._field_active(n)
        ] + [
            n for n in ("aff_emb", "aff_avg") if self._field_active(n)
        ]
        return (
            f"{self.__class__.__name__}("
            f"channels={self.num_channels}, "
            f"background={self.background}, "
            f"active={active})"
        )
