"""
Boundary loss: per-instance centroid colour + raw intensity + face-affinity,
with two parallel affinity supervision paths.

The boundary head emits **10** channels per voxel:

  =====  ========================================================
  ch     meaning                                                supervision
  =====  ========================================================
  0      *raw*    (raw, normalised image intensity)             regression vs raw
  1-3    *avg*    (normalised z, y, x of instance centroid)     regression vs avg
  4-9    *aff*    (direct face-affinity prediction;             BCE + Dice + IoU
                   T, B, U, D, L, R; Z-Y-X order)                 vs binary aff target
  =====  ========================================================

In addition to the model's *direct* aff prediction (ch 4-9), the loss
also derives a **soft 6-face affinity from the predicted avgloc** (ch 1-3)
using::

    aff_avg[c] = exp(-tau * sum_i |avg[i] - shift_replicate(avg[i], dir_c)|)

and supervises that derived signal against the same binary aff target.
This ties the predicted avg field explicitly to the boundary structure:
two voxels in the same instance must share their predicted centroid, and
two voxels across an instance boundary must disagree on it — and the
derived-aff loss makes that pressure explicit on top of the regression
loss on ch 1-3.

Affinity targets use **SAME / replicate** padding (the boundary voxel is
compared to itself, ``aff = 1`` for foreground, masked to ``0`` for
background) -- no masking needed at the volume edge.

Channel summary::

    raw (1) + avg (3) + aff_pred (6) = 10  (model output)
    aff_avg (6, derived from avg)         = 6  conceptual, computed in loss
    -------------------------------------------------------------
    raw + avg + aff_pred + aff_avg        = 16 conceptual supervision slots

Normalisation of the avg channels is relative to the patch size, so they
live in ``[0, 1]`` regardless of anisotropy:: ``R = z/D, G = y/H, B = x/W``.
Background voxels (``label == 0``) have avg == 0 and are excluded from
the avg regression by ``foreground_only_loc=True``.

Centroids are computed in a single fully-vectorised pass (no Python loops
over voxels) via :func:`torch.scatter_add_` (CUDA path) or
:func:`numpy.bincount` (CPU path).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from monai.losses import DiceLoss

from brainbow.losses._common import (
    canonical_regression_name,
    regression_loss_fn,
    stable_bce_on_probs,
)


# Channel layout owned by this module.  All callers (TensorBoard
# heads/image-logger, the CombinedLoss key hierarchy, etc.) re-import
# these constants rather than re-declaring magic numbers, so any future
# layout change propagates automatically.
_N_RAW: int = 1            # ch 0
_N_AVG: int = 3            # ch 1..3
_N_AFF: int = 6            # ch 4..9 (T, B, U, D, L, R; Z-Y-X order)
_BOUNDARY_CHANNELS: int = _N_RAW + _N_AVG + _N_AFF      # 10
_AVG_START: int = _N_RAW                                 # 1
_AVG_END: int = _N_RAW + _N_AVG                          # 4
_AFF_START: int = _AVG_END                               # 4
_AFF_END: int = _BOUNDARY_CHANNELS                       # 10

# (name, axis_in_[B,D,H,W], shift).  ``shift == +1`` means: the shifted
# tensor at position ``i`` equals the input at position ``i-1`` (i.e. the
# neighbour one voxel *earlier* along ``axis``); shift -1 is the mirror.
# Axis indexing is relative to the 4-D ``[B, D, H, W]`` layout -- axis 1
# is Z, axis 2 is Y, axis 3 is X.  Channel order is **Z, Y, X** (slowest
# to fastest axis); each axis contributes (-1, +1) in that order.
_DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("T", 1, +1),   # top    : z - 1
    ("B", 1, -1),   # bottom : z + 1
    ("U", 2, +1),   # up     : y - 1
    ("D", 2, -1),   # down   : y + 1
    ("L", 3, +1),   # left   : x - 1
    ("R", 3, -1),   # right  : x + 1
)
_AFF_NAMES: tuple[str, ...] = tuple(name for name, _, _ in _DIRECTIONS)


# ---------------------------------------------------------------------------
# Affinity target helpers (SAME / replicate padding)
# ---------------------------------------------------------------------------


def _shift_replicate_torch(
    x: torch.Tensor, axis: int, shift: int,
) -> torch.Tensor:
    """Shift a torch tensor along ``axis`` by ``shift`` with replicate pad.

    ``shift == +1`` pads the front by replicating the first slab (so
    ``out[0] == x[0]`` and ``out[i] == x[i-1]`` for ``i > 0``); ``-1``
    pads the back.
    """
    N = x.size(axis)
    if shift == 1:
        head = x.narrow(axis, 0, 1)
        body = x.narrow(axis, 0, N - 1)
        return torch.cat([head, body], dim=axis)
    body = x.narrow(axis, 1, N - 1)
    tail = x.narrow(axis, -1, 1)
    return torch.cat([body, tail], dim=axis)


def _shift_replicate_np(x: np.ndarray, axis: int, shift: int) -> np.ndarray:
    """NumPy counterpart of :func:`_shift_replicate_torch`."""
    N = x.shape[axis]
    if shift == 1:
        head = np.take(x, indices=[0], axis=axis)
        body = np.take(x, indices=np.arange(0, N - 1), axis=axis)
        return np.concatenate([head, body], axis=axis)
    body = np.take(x, indices=np.arange(1, N), axis=axis)
    tail = np.take(x, indices=[N - 1], axis=axis)
    return np.concatenate([body, tail], axis=axis)


@torch.no_grad()
def _affinity_target_torch(
    labels: torch.Tensor,
    background: Optional[int] = None,
) -> torch.Tensor:
    """Build the 6-channel affinity target from ``[B, D, H, W]`` labels.

    Args:
        labels: Integer instance ids of shape ``[B, D, H, W]``.
        background: When set, voxels whose label equals this value are
            masked to ``0`` across all 6 face-affinity channels. This
            suppresses the spurious ``0 == 0 -> 1`` signal at background
            voxels (and at the boundary voxels zeroed by
            :class:`FindBoundariesd`). ``None`` -> legacy behavior
            (``aff = labels == shifted_labels`` everywhere).
    """
    per_dir = [
        (labels == _shift_replicate_torch(labels, axis, shift)).to(torch.float32)
        for _, axis, shift in _DIRECTIONS
    ]
    out = rearrange(torch.stack(per_dir, dim=0), "c b d h w -> b c d h w")
    if background is not None:
        mask = rearrange(
            (labels != background).to(torch.float32), "b ... -> b 1 ...",
        )
        out = out * mask
    return out


def _affinity_target_np(
    labels: np.ndarray,
    background: Optional[int] = None,
) -> np.ndarray:
    """NumPy counterpart of :func:`_affinity_target_torch`."""
    per_dir = [
        (labels == _shift_replicate_np(labels, axis, shift)).astype(np.float32)
        for _, axis, shift in _DIRECTIONS
    ]
    out = np.stack(per_dir, axis=1)          # [B, 6, D, H, W]
    if background is not None:
        mask = (labels != background).astype(np.float32)[:, None]
        out = out * mask
    return out


# ---------------------------------------------------------------------------
# Soft 6-face affinity derived from a 3-channel avgloc field
# ---------------------------------------------------------------------------


def soft_aff_from_avg(
    avg: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Derive a soft 6-face affinity from a continuous ``[B, 3, D, H, W]`` field.

    Mirrors :func:`_affinity_target_torch` but operates on a continuous
    avg field instead of integer labels.  Similarity uses an
    L1-distance kernel with bandwidth ``tau``::

        aff[c] = exp(-tau * sum_i |avg[i] - shift_replicate(avg[i], dir_c)|)

    Voxels that share an instance share their predicted centroid, so
    the L1 distance vanishes and the kernel evaluates to ``1``; voxels
    across an instance boundary disagree on the centroid and the
    similarity decays smoothly.  At the volume edge SAME / replicate
    padding compares the voxel to itself, giving ``aff = 1`` (consistent
    with the binary target on foreground voxels).

    Args:
        avg: Predicted avgloc of shape ``[B, 3, D, H, W]``.  Channels
            are interpreted as the 3 anisotropic-normalised centroid
            components (Z, Y, X order) but the function only relies on
            the L1 distance, so any ordering works.
        tau: Bandwidth of the soft kernel (positive).  Larger ``tau``
            sharpens the decay (more like a hard binary signal); smaller
            ``tau`` softens it.

    Returns:
        ``[B, 6, D, H, W]`` similarity tensor in ``(0, 1]``.
    """
    if avg.dim() != 5 or avg.shape[1] != _N_AVG:
        raise ValueError(
            f"soft_aff_from_avg expects [B, {_N_AVG}, D, H, W]; "
            f"got shape {tuple(avg.shape)}"
        )
    # ``_DIRECTIONS`` uses the BDHW (4-D) convention where Z/Y/X live at
    # tensor axes 1/2/3.  Here ``avg`` is BCDHW (5-D) so the matching
    # spatial axes are 2/3/4 -- shift on ``axis + 1``.
    per_dir = []
    for _, axis, shift in _DIRECTIONS:
        diff = (avg - _shift_replicate_torch(avg, axis + 1, shift)).abs()
        # L1 distance summed over the 3 avg channels -> [B, D, H, W].
        l1 = reduce(diff, "b c d h w -> b d h w", "sum")
        per_dir.append(torch.exp(-tau * l1))
    return rearrange(torch.stack(per_dir, dim=0), "c b d h w -> b c d h w")


# ---------------------------------------------------------------------------
# 10-channel target builders (CUDA / CPU paths)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _boundary_target_torch(
    labels: torch.Tensor,
    image: torch.Tensor,
    background: Optional[int] = None,
) -> torch.Tensor:
    """Vectorised 10-channel boundary target on the same device as ``labels``.

    Args:
        labels: ``[B, D, H, W]`` integer instance ids (``0`` = background).
        image:  ``[B, D, H, W]`` raw (normalised) image intensities.
        background: Forwarded to :func:`_affinity_target_torch` to mask
            spurious bg-bg ``1`` values on the 6 affinity channels.
            ``None`` preserves the legacy unmasked behavior.

    Returns:
        ``[B, 10, D, H, W]`` float target tensor with channels
        ``[raw(1) | avg(3) | aff(6)]``.
    """
    B, D, H, W = labels.shape
    dims_t = labels.new_tensor([D, H, W], dtype=torch.float32)
    norm = rearrange(dims_t, "c -> 1 c").clamp(min=1.0)

    target = labels.new_zeros((B, _BOUNDARY_CHANNELS, D, H, W), dtype=torch.float32)
    target[:, 0] = image.to(torch.float32)

    for b in range(B):
        lbl_b = labels[b]
        fg = lbl_b > 0
        if not fg.any():
            continue

        unique_ids, inverse = torch.unique(lbl_b[fg], return_inverse=True)
        K = unique_ids.shape[0]

        fg_idx = torch.nonzero(fg, as_tuple=False).to(torch.float32)  # [M, 3] (z, y, x)

        # Centroid via scatter_add_ + bincount -- single on-device pass.
        sum_coords = fg_idx.new_zeros((K, 3))
        inv3 = repeat(inverse, "m -> m c", c=3)
        sum_coords.scatter_add_(0, inv3, fg_idx)
        counts = torch.bincount(inverse, minlength=K).to(torch.float32).clamp_(min=1.0)
        cen_coords = sum_coords / rearrange(counts, "k -> k 1")

        # Broadcast normalised per-instance centroid to every fg voxel.
        voxel_avg = (cen_coords / norm)[inverse]                          # [M, 3]
        target[b, _AVG_START:_AVG_END][:, fg] = rearrange(
            voxel_avg, "m c -> c m",
        )

    target[:, _AFF_START:_AFF_END] = _affinity_target_torch(
        labels, background=background,
    )
    return target


@torch.no_grad()
def _boundary_target_scipy(
    labels_np: np.ndarray,
    image_np: np.ndarray,
    background: Optional[int] = None,
) -> np.ndarray:
    """Reference 10-channel boundary-target builder via ``numpy.bincount``.

    CPU fallback used when the input tensors live on CPU.  Centroid
    computation is fully vectorised; no Python loop over voxels.
    """
    B, D, H, W = labels_np.shape
    target = np.zeros((B, _BOUNDARY_CHANNELS, D, H, W), dtype=np.float32)
    target[:, 0] = image_np.astype(np.float32, copy=False)
    dims = np.array([D, H, W], dtype=np.float32).clip(min=1.0)

    for b in range(B):
        lbl = labels_np[b]
        fg = lbl > 0
        if not fg.any():
            continue

        unique_ids, inverse = np.unique(lbl[fg], return_inverse=True)
        K = unique_ids.shape[0]

        coords = np.stack(np.nonzero(fg), axis=1).astype(np.float32)      # [M, 3]
        counts = np.bincount(inverse, minlength=K).astype(np.float32).clip(min=1.0)
        cen = np.stack([
            np.bincount(inverse, weights=coords[:, a], minlength=K) / counts
            for a in range(3)
        ], axis=1).astype(np.float32)                                     # [K, 3]

        voxel_avg = (cen / dims)[inverse]                                 # [M, 3]
        target[b, _AVG_START:_AVG_END][:, fg] = voxel_avg.T

    target[:, _AFF_START:_AFF_END] = _affinity_target_np(
        labels_np, background=background,
    )
    return target


@torch.no_grad()
def build_boundary_target(
    labels: torch.Tensor,
    image: torch.Tensor,
    background: Optional[int] = 0,
) -> torch.Tensor:
    """Build a ``[B, 10, D, H, W]`` boundary target from labels + image.

    Picks the on-device torch path for CUDA tensors and the NumPy path
    for CPU tensors.  Exposed as a module-level function so callers
    outside :class:`BoundaryLoss` can pre-build the target (e.g. the
    image-logger callback).

    Args:
        labels: ``[B, D, H, W]`` integer instance ids.
        image:  ``[B, D, H, W]`` raw (normalised) image intensities.
        background: Label value treated as background when building the
            6-channel affinity target (ch 4-9).  When set, voxels with
            ``labels == background`` get affinity ``0`` (suppresses the
            spurious ``0 == 0 -> 1`` signal at background voxels and at
            the boundary voxels zeroed by :class:`FindBoundariesd`).
            Default ``0``.  Pass ``None`` to opt out and reproduce the
            pre-fix unmasked targets.

    Returns:
        Float tensor ``[B, 10, D, H, W]`` with layout
        ``[raw(1) | avg(3) | aff(6)]``.
    """
    if labels.dim() != 4:
        raise ValueError(
            f"labels must be [B, D, H, W]; got shape {tuple(labels.shape)}"
        )
    if image.shape != labels.shape:
        raise ValueError(
            f"image shape {tuple(image.shape)} != labels shape "
            f"{tuple(labels.shape)}"
        )

    if labels.is_cuda:
        return _boundary_target_torch(
            labels.long(), image.float(), background=background,
        )

    target_np = _boundary_target_scipy(
        labels.detach().cpu().long().numpy(),
        image.detach().cpu().float().numpy(),
        background=background,
    )
    return torch.from_numpy(target_np).to(labels.device)


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


class BoundaryLoss(nn.Module):
    """Regression + binary loss on a 10-channel boundary target map.

    The target map is built on-the-fly from ``labels`` (+ ``image``):

    - Channel 0:    ``raw``  (per-voxel raw image intensity)
    - Channels 1-3: ``avg``  (per-instance centroid xyz / (D, H, W))
    - Channels 4-9: ``aff``  (6 face-neighbour affinities in Z-Y-X
                              order: T, B, U, D, L, R)

    On top of the **direct** affinity prediction (model ch 4-9), the
    loss also derives a **soft affinity from the predicted avgloc**
    (model ch 1-3) via :func:`soft_aff_from_avg` and supervises that
    derived signal against the same binary aff target.  The two
    aff sub-loss bundles share the same CE / Dice / IoU sub-loss
    weights but each carries its own path weight (``weight_aff_pred``
    for the direct path, ``weight_aff_avg`` for the derived path).

    Channel 0 is supervised everywhere.  Channels 1-3 are foreground-only
    when ``foreground_only_loc=True``.  Channels 4-9 (both direct and
    derived) are supervised everywhere -- boundary voxels use SAME /
    replicate padding so they compare to themselves (aff = 1 on FG, 0
    on BG).

    Args:
        loss_avg:  Regression loss name for the 3 avgloc channels
            (``"l1"``, ``"mse"``, ``"smooth_l1"`` + aliases).
        loss_raw:  Regression loss name for the raw-intensity channel.
        weight_avg:   Weight of the 3 avg-location channels (ch 1-3).
        weight_raw:   Weight of the raw-intensity channel (ch 0).
        weight_aff_pred: Path weight on the **direct** aff sub-losses
            (operating on model ch 4-9).
        weight_aff_avg:  Path weight on the **derived** aff sub-losses
            (operating on :func:`soft_aff_from_avg(model_avg, tau)`).
            ``0`` disables the derived path entirely.
        weight_ce:    Weight of the BCE sub-loss on the 6 affinity
            channels.  Applied to **both** the direct and derived
            paths; the path weights above scale the per-path total.
            The model wrapper applies ``sigmoid`` to all 10 boundary
            channels before this loss sees them, so the direct aff is
            already in ``[0, 1]``; the derived aff is in ``(0, 1]`` by
            construction.  The BCE is computed on probabilities (not
            logits) via :func:`F.binary_cross_entropy`, with
            ``pos_weight = class_weights`` when provided.
        weight_dice:  Weight of the soft-Dice sub-loss on the 6 affinity
            channels (MONAI :class:`DiceLoss` with ``sigmoid=False`` --
            the wrapper has already applied sigmoid).
        weight_iou:   Weight of the soft-Jaccard sub-loss on the 6
            affinity channels (MONAI :class:`DiceLoss` with
            ``sigmoid=False, jaccard=True``).
        tau:    Bandwidth of the soft similarity kernel used by
            :func:`soft_aff_from_avg`.  Default ``1.0``.
        class_weights: Optional per-affinity-channel weight list (length
            6, one entry per direction T/B/U/D/L/R).  Plumbed into the
            BCE sub-loss as ``pos_weight``.  No effect on the Dice / IoU
            sub-losses.
        foreground_only_loc:  If True, the avgloc loss is averaged
            over foreground voxels only (strongly recommended -- the
            target is zero on background, so averaging everywhere would
            down-weight instance gradients proportionally to foreground
            fraction).  Default ``True``.
        aff_eps: Numerical stabiliser passed as both ``smooth_nr`` and
            ``smooth_dr`` to :class:`monai.losses.DiceLoss`.
        background: Label value treated as background when building the
            6 affinity targets.  When set (default ``0``), voxels whose
            label equals this value contribute ``0`` to all 6
            face-affinity channels.  Pass ``None`` to opt out.
    """

    num_channels: int = _BOUNDARY_CHANNELS

    def __init__(
        self,
        loss_avg: str = "smooth_l1",
        loss_raw: str = "l1",
        weight_avg: float = 1.0,
        weight_raw: float = 1.0,
        weight_aff_pred: float = 1.0,
        weight_aff_avg: float = 1.0,
        weight_ce: float = 0.0,
        weight_dice: float = 1.0,
        weight_iou: float = 0.0,
        tau: float = 1.0,
        class_weights: Optional[List[float]] = None,
        foreground_only_loc: bool = True,
        aff_eps: float = 1e-5,
        background: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.loss_avg = canonical_regression_name(loss_avg)
        self.loss_raw = canonical_regression_name(loss_raw)
        self._loss_avg_fn = regression_loss_fn(loss_avg)
        self._loss_raw_fn = regression_loss_fn(loss_raw)
        self.weight_avg = float(weight_avg)
        self.weight_raw = float(weight_raw)
        self.weight_aff_pred = float(weight_aff_pred)
        self.weight_aff_avg = float(weight_aff_avg)
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.weight_iou = float(weight_iou)
        self.tau = float(tau)
        self.class_weights = (
            list(map(float, class_weights)) if class_weights is not None else None
        )
        self.foreground_only_loc = bool(foreground_only_loc)
        self.aff_eps = float(aff_eps)
        self.background = (
            int(background) if background is not None else None
        )

        # BCE pos_weight reshaped to [1, 6, 1, 1, 1] so it broadcasts
        # over [B, 6, D, H, W]; None means uniform weighting.
        if self.class_weights is not None:
            if len(self.class_weights) != _N_AFF:
                raise ValueError(
                    f"class_weights must have length {_N_AFF} (one per affinity "
                    f"channel T/B/U/D/L/R); got {len(self.class_weights)}."
                )
            pw = torch.tensor(self.class_weights, dtype=torch.float32).view(
                1, _N_AFF, 1, 1, 1,
            )
            self.register_buffer("_aff_pos_weight", pw, persistent=False)
        else:
            self._aff_pos_weight = None

        # Sigmoid is applied externally (in the model wrapper) to every
        # boundary channel: BCE consumes probabilities directly via
        # F.binary_cross_entropy, and the Dice / IoU sub-losses are MONAI
        # DiceLoss with sigmoid=False so they do not re-apply any
        # activation.
        _dice_kwargs = dict(
            sigmoid=False,
            include_background=True,
            reduction="mean",
            batch=True,                       # aggregate over B + spatial per channel
            smooth_nr=self.aff_eps,
            smooth_dr=self.aff_eps,
        )
        self._aff_dice = DiceLoss(**_dice_kwargs)
        self._aff_iou = DiceLoss(**_dice_kwargs, jaccard=True)

    @property
    def task_channels(self) -> int:
        """Expected width of the boundary head prediction tensor (10)."""
        return _BOUNDARY_CHANNELS

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_target(
        self,
        labels: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Full 10-channel boundary target for ``(labels, image)``.

        Thin wrapper around :func:`build_boundary_target` that passes
        through this loss's :attr:`background` setting.  Exposed so
        callers (image-logger, custom precompute paths) can hand the
        target back via ``cached_target=`` to skip the second build in
        :meth:`forward`.
        """
        return build_boundary_target(labels, image, background=self.background)

    # ------------------------------------------------------------------
    # Per-voxel weights (not used by this head)
    # ------------------------------------------------------------------

    def compute_weights(self, labels: torch.Tensor) -> None:
        return None

    # ------------------------------------------------------------------
    # Sub-losses
    # ------------------------------------------------------------------

    def _compute_loss_raw(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Dense raw-intensity loss on ch 0."""
        return self._loss_raw_fn(pred, target)

    def _compute_loss_avg(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """3-RGB centroid regression loss on ch 1-3.

        Args:
            pred:   ``[B, 3, D, H, W]`` avgloc prediction.
            target: ``[B, 3, D, H, W]`` avgloc target.
            labels: ``[B, D, H, W]``   instance ids.
        """
        per_voxel = self._loss_avg_fn(pred, target, reduction="none")
        if self.foreground_only_loc:
            fg = rearrange(labels > 0, "b ... -> b 1 ...").expand_as(per_voxel)
            n_fg = fg.sum().clamp(min=1)
            return (per_voxel * fg).sum() / n_fg
        return per_voxel.mean()

    def _compute_loss_aff(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """CE + Dice + IoU sub-losses on a 6-channel sigmoid affinity tensor.

        Used for **both** the direct prediction (ch 4-9) and the
        derived-from-avgloc soft affinity.  ``pred`` is **already** in
        ``[0, 1]``: the model wrapper applies sigmoid to every boundary
        channel and :func:`soft_aff_from_avg` returns probabilities by
        construction (``exp(-tau * L1)``).  Returns a dict with keys
        ``ce``, ``dice``, ``iou``; each sub-term is only computed when
        its weight is non-zero, the rest are filled with a zero tensor
        matching the prediction dtype/device so downstream logging can
        treat them uniformly.

        - ``ce``   : binary cross-entropy on probabilities via
          :func:`torch.nn.functional.binary_cross_entropy`, with optional
          ``pos_weight = class_weights`` (broadcast over the 6 affinity
          channels).  Inputs are clamped to ``[eps, 1 - eps]`` to keep
          ``log(p)`` finite at the boundaries.
        - ``dice`` : :class:`monai.losses.DiceLoss` with ``sigmoid=False``
          and ``batch=True`` so intersections / unions are summed over
          batch + spatial per channel and the final ``1 - dice`` is
          averaged across the 6 direction channels.
        - ``iou``  : same MONAI loss with ``jaccard=True``.
        """
        zero = pred.new_zeros(())
        out: Dict[str, torch.Tensor] = {"ce": zero, "dice": zero, "iou": zero}
        if self.weight_ce > 0:
            per_voxel = stable_bce_on_probs(
                pred, target, pos_weight=self._aff_pos_weight,
            )
            out["ce"] = per_voxel.mean()
        if self.weight_dice > 0:
            out["dice"] = self._aff_dice(pred, target)
        if self.weight_iou > 0:
            out["iou"] = self._aff_iou(pred, target)
        return out

    @staticmethod
    def _aff_total(
        terms: Dict[str, torch.Tensor],
        weight_ce: float,
        weight_dice: float,
        weight_iou: float,
    ) -> torch.Tensor:
        return (
            weight_ce * terms["ce"]
            + weight_dice * terms["dice"]
            + weight_iou * terms["iou"]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        image: torch.Tensor,
        cached_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the boundary regression + dual-aff loss.

        Args:
            prediction:     ``[B, 10, D, H, W]`` model output (post-sigmoid;
                see :class:`brainbow.models.cosmos_transfer_2_5.decoder._DecoderAdapter3D`).
                Channel layout: ``[raw(1) | avg(3) | aff_pred(6)]``.
            labels:         ``[B, D, H, W]``    instance ids.
            image:          ``[B, D, H, W]``    normalised image.
            cached_target:  Optional precomputed ``[B, 10, D, H, W]`` target
                (see :meth:`build_target`).  Useful under DDP where each
                step otherwise rebuilds it twice (loss + image logger).

        Returns:
            Dict with keys ``loss``, ``raw``, ``avg``, ``aff`` (sum of
            both aff paths weighted by ``weight_aff_pred`` /
            ``weight_aff_avg``), ``aff_pred`` (direct path), ``aff_avg``
            (derived path), and the per-sub-term breakdown
            ``aff_pred_ce / aff_pred_dice / aff_pred_iou`` and
            ``aff_avg_ce / aff_avg_dice / aff_avg_iou``.
        """
        if prediction.shape[1] != _BOUNDARY_CHANNELS:
            raise ValueError(
                f"BoundaryLoss expects {_BOUNDARY_CHANNELS}-channel prediction; "
                f"got {prediction.shape[1]} channels."
            )
        if labels.dim() == 5 and labels.shape[1] == 1:
            labels = rearrange(labels, "b 1 d h w -> b d h w")
        if image.dim() == 5 and image.shape[1] == 1:
            image = rearrange(image, "b 1 d h w -> b d h w")

        target = (
            cached_target if cached_target is not None
            else self.build_target(labels, image)
        )
        target = target.to(dtype=prediction.dtype, device=prediction.device)

        loss_raw = self._compute_loss_raw(prediction[:, 0], target[:, 0])
        loss_avg = self._compute_loss_avg(
            prediction[:, _AVG_START:_AVG_END],
            target[:, _AVG_START:_AVG_END],
            labels,
        )

        aff_target = target[:, _AFF_START:_AFF_END]

        # Direct path: the model's predicted ch 4-9 (already sigmoid).
        pred_aff = prediction[:, _AFF_START:_AFF_END]
        aff_pred_terms = self._compute_loss_aff(pred_aff, aff_target)
        loss_aff_pred = self._aff_total(
            aff_pred_terms, self.weight_ce, self.weight_dice, self.weight_iou,
        )

        # Derived path: soft 6-aff from the predicted avgloc (ch 1-3).
        # Skip the kernel + sub-losses entirely when the path weight is
        # zero so we don't pay for unused gradient propagation.
        if self.weight_aff_avg > 0:
            avg_aff = soft_aff_from_avg(
                prediction[:, _AVG_START:_AVG_END], tau=self.tau,
            )
            aff_avg_terms = self._compute_loss_aff(avg_aff, aff_target)
            loss_aff_avg = self._aff_total(
                aff_avg_terms, self.weight_ce, self.weight_dice, self.weight_iou,
            )
        else:
            zero = prediction.new_zeros(())
            aff_avg_terms = {"ce": zero, "dice": zero, "iou": zero}
            loss_aff_avg = zero

        loss_aff = (
            self.weight_aff_pred * loss_aff_pred
            + self.weight_aff_avg * loss_aff_avg
        )

        total = (
            self.weight_raw * loss_raw
            + self.weight_avg * loss_avg
            + loss_aff
        )

        return {
            "loss": total,
            "raw": loss_raw,
            "avg": loss_avg,
            "aff": loss_aff,
            "aff_pred": loss_aff_pred,
            "aff_avg": loss_aff_avg,
            "aff_pred_ce": aff_pred_terms["ce"],
            "aff_pred_dice": aff_pred_terms["dice"],
            "aff_pred_iou": aff_pred_terms["iou"],
            "aff_avg_ce": aff_avg_terms["ce"],
            "aff_avg_dice": aff_avg_terms["dice"],
            "aff_avg_iou": aff_avg_terms["iou"],
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"channels={self.task_channels}, "
            f"loss_avg='{self.loss_avg}', loss_raw='{self.loss_raw}', "
            f"weight_raw={self.weight_raw}, "
            f"weight_avg={self.weight_avg}, "
            f"weight_aff_pred={self.weight_aff_pred}, "
            f"weight_aff_avg={self.weight_aff_avg}, "
            f"weight_ce={self.weight_ce}, "
            f"weight_dice={self.weight_dice}, "
            f"weight_iou={self.weight_iou}, "
            f"tau={self.tau}, "
            f"class_weights={self.class_weights}, "
            f"foreground_only_loc={self.foreground_only_loc}, "
            f"background={self.background})"
        )
