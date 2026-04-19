"""
Brainbow loss: per-instance spatial colouring + raw intensity + face-affinity.

For every connected component ``label > 0`` in the batch this loss builds
a dense **16-channel** per-voxel target out of purely geometric and
image statistics -- no learnable parameters:

  =====  ========================================================
  ch     meaning
  =====  ========================================================
  0      *raw*  := raw (normalised) image intensity at the voxel
  1-3    R,G,B  := normalised  z,y,x  of the instance's **min**  (bbox min)
  4-6    R,G,B  := normalised  z,y,x  of the instance's **avg**  (centroid)
  7-9    R,G,B  := normalised  z,y,x  of the instance's **max**  (bbox max)
  10-15  *aff*  := binary face-affinity to 6 neighbours (U,D,L,R,T,B)
  =====  ========================================================

Normalisation is relative to the **patch size**, so the 9 localisation
channels live in ``[0, 1]`` regardless of anisotropy or patch resolution::

    R = z / D, G = y / H, B = x / W

Background voxels (``label == 0``) are zeroed on the 9 localisation
channels; channel 0 carries the raw image value for every voxel.

The 6 affinity channels encode, for every voxel ``v``, whether it shares
its instance label with each of its 6 face neighbours (same-label = 1,
different-label = 0).  Direction order::

    ch 10 : U  (up    = y - 1)
    ch 11 : D  (down  = y + 1)
    ch 12 : L  (left  = x - 1)
    ch 13 : R  (right = x + 1)
    ch 14 : T  (top   = z - 1)
    ch 15 : B  (bottom= z + 1)

Boundary voxels (where the neighbour would fall outside the crop) use
**SAME padding** (replicate) so the voxel is compared to itself -- the
boundary affinity is therefore always ``1`` (self-connected) and every
voxel contributes a well-defined target, no masking required.

The per-instance (min, centroid, max) statistics are computed with a
single pass of :func:`scipy.ndimage.find_objects` + centroid-via-
``np.bincount`` (CPU path) or on-device :func:`torch.scatter_reduce_`
(GPU path) -- fully vectorised, no Python loops over voxels.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from brainbow.losses._common import canonical_regression_name, regression_loss_fn


_BRAINBOW_CHANNELS: int = 16
_N_LOC: int = 10          # ch 0..9 : raw + min/avg/max (10 channels)
_N_AFF: int = 6           # ch 10..15 : U, D, L, R, T, B

# (name, axis_in_[B,D,H,W], shift).  ``shift == +1`` means: the shifted
# tensor at position ``i`` equals the input at position ``i-1`` (i.e. the
# neighbour one voxel *earlier* along ``axis``); shift -1 is the mirror.
# Axis indexing is relative to the 4-D ``[B, D, H, W]`` layout -- axis 1
# is Z, axis 2 is Y, axis 3 is X.
_DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("U", 2, +1),   # up     : y - 1
    ("D", 2, -1),   # down   : y + 1
    ("L", 3, +1),   # left   : x - 1
    ("R", 3, -1),   # right  : x + 1
    ("T", 1, +1),   # top    : z - 1
    ("B", 1, -1),   # bottom : z + 1
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
def _affinity_target_torch(labels: torch.Tensor) -> torch.Tensor:
    """Build the 6-channel affinity target from ``[B, D, H, W]`` labels."""
    per_dir = [
        (labels == _shift_replicate_torch(labels, axis, shift)).to(torch.float32)
        for _, axis, shift in _DIRECTIONS
    ]
    # stack along channel axis -> [B, 6, D, H, W]
    return rearrange(torch.stack(per_dir, dim=0), "c b d h w -> b c d h w")


def _affinity_target_np(labels: np.ndarray) -> np.ndarray:
    """NumPy counterpart of :func:`_affinity_target_torch`."""
    per_dir = [
        (labels == _shift_replicate_np(labels, axis, shift)).astype(np.float32)
        for _, axis, shift in _DIRECTIONS
    ]
    return np.stack(per_dir, axis=1)          # [B, 6, D, H, W]


# ---------------------------------------------------------------------------
# Low-level builders (used by BrainbowLoss._build_target_*)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _brainbow_target_torch(
    labels: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Vectorised 16-channel brainbow target on the same device as ``labels``.

    Args:
        labels: ``[B, D, H, W]`` integer instance ids (``0`` = background).
        image:  ``[B, D, H, W]`` raw (normalised) image intensities.

    Returns:
        ``[B, 16, D, H, W]`` float target tensor.
    """
    B, D, H, W = labels.shape
    dims_t = labels.new_tensor([D, H, W], dtype=torch.float32)
    norm = rearrange(dims_t, "c -> 1 c").clamp(min=1.0)

    target = labels.new_zeros((B, _BRAINBOW_CHANNELS, D, H, W), dtype=torch.float32)
    target[:, 0] = image.to(torch.float32)

    for b in range(B):
        lbl_b = labels[b]
        fg = lbl_b > 0
        if not fg.any():
            continue

        unique_ids, inverse = torch.unique(lbl_b[fg], return_inverse=True)
        K = unique_ids.shape[0]

        fg_idx = torch.nonzero(fg, as_tuple=False).to(torch.float32)  # [M, 3] (z, y, x)

        # Min / max / sum per-instance via scatter_reduce_ (fully on-device).
        INF = torch.finfo(torch.float32).max
        min_coords = fg_idx.new_full((K, 3), INF)
        max_coords = fg_idx.new_full((K, 3), -INF)
        sum_coords = fg_idx.new_zeros((K, 3))

        inv3 = repeat(inverse, "m -> m c", c=3)
        min_coords.scatter_reduce_(0, inv3, fg_idx, reduce="amin", include_self=True)
        max_coords.scatter_reduce_(0, inv3, fg_idx, reduce="amax", include_self=True)
        sum_coords.scatter_add_(0, inv3, fg_idx)
        counts = torch.bincount(inverse, minlength=K).to(torch.float32).clamp_(min=1.0)
        cen_coords = sum_coords / rearrange(counts, "k -> k 1")

        # Concat min|cen|max -> [K, 9] normalised RGB colours.
        rgb9 = torch.cat(
            [min_coords / norm, cen_coords / norm, max_coords / norm], dim=1,
        )

        # Broadcast per-instance colour to every foreground voxel.
        voxel_rgb = rgb9[inverse]                                         # [M, 9]
        target[b, 1:_N_LOC][:, fg] = rearrange(voxel_rgb, "m c -> c m")

    # Affinity ch 10-15 -- SAME-pad face equality over the whole batch.
    target[:, _N_LOC:_BRAINBOW_CHANNELS] = _affinity_target_torch(labels)
    return target


@torch.no_grad()
def _brainbow_target_scipy(
    labels_np: np.ndarray,
    image_np: np.ndarray,
) -> np.ndarray:
    """Reference 16-channel brainbow-target builder via ``scipy.ndimage``.

    CPU fallback used when the input tensors live on CPU.  ``find_objects``
    is a single O(N) pass over the label volume, which is faster than
    iterating :func:`torch.unique` for very fragmented labels.
    """
    from scipy.ndimage import find_objects

    B, D, H, W = labels_np.shape
    target = np.zeros((B, _BRAINBOW_CHANNELS, D, H, W), dtype=np.float32)
    target[:, 0] = image_np.astype(np.float32, copy=False)
    dims = np.array([D, H, W], dtype=np.float32).clip(min=1.0)

    for b in range(B):
        lbl = labels_np[b]
        fg = lbl > 0
        if not fg.any():
            continue

        unique_ids, inverse = np.unique(lbl[fg], return_inverse=True)
        K = unique_ids.shape[0]

        # Bounding boxes give min/max in a single pass.
        slices = find_objects(lbl.astype(np.int64))
        min_rgb = np.zeros((K, 3), dtype=np.float32)
        max_rgb = np.zeros((K, 3), dtype=np.float32)
        for k, uid in enumerate(unique_ids):
            sl = slices[int(uid) - 1]
            if sl is None:
                continue
            for a in range(3):
                min_rgb[k, a] = sl[a].start
                max_rgb[k, a] = sl[a].stop - 1

        # Vectorised centroids via np.bincount.
        coords = np.stack(np.nonzero(fg), axis=1).astype(np.float32)      # [M, 3]
        counts = np.bincount(inverse, minlength=K).astype(np.float32).clip(min=1.0)
        cen_rgb = np.stack([
            np.bincount(inverse, weights=coords[:, a], minlength=K) / counts
            for a in range(3)
        ], axis=1).astype(np.float32)

        voxel_rgb = np.concatenate(
            [min_rgb / dims, cen_rgb / dims, max_rgb / dims], axis=1,
        )[inverse]                                                        # [M, 9]
        target[b, 1:_N_LOC][:, fg] = voxel_rgb.T

    # Affinity ch 10-15 -- SAME-pad face equality over the whole batch.
    target[:, _N_LOC:_BRAINBOW_CHANNELS] = _affinity_target_np(labels_np)
    return target


@torch.no_grad()
def build_brainbow_target(
    labels: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Build a ``[B, 16, D, H, W]`` brainbow target from labels + image.

    Picks the on-device torch path for CUDA tensors and the NumPy /
    scipy path for CPU tensors.  Exposed as a module-level function so
    callers outside :class:`BrainbowLoss` can pre-build the target (e.g.
    the image-logger callback).
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
        return _brainbow_target_torch(labels.long(), image.float())

    target_np = _brainbow_target_scipy(
        labels.detach().cpu().long().numpy(),
        image.detach().cpu().float().numpy(),
    )
    return torch.from_numpy(target_np).to(labels.device)


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


class BrainbowLoss(nn.Module):
    """Regression + binary loss on a 16-channel brainbow target map.

    The target map is built on-the-fly from ``labels`` (+ ``image``):

    - Channel  0:    ``raw``  (per-voxel raw image intensity)
    - Channels 1-3:  ``min``  (per-instance bbox-min xyz / (D,H,W))
    - Channels 4-6:  ``avg``  (per-instance centroid     / (D,H,W))
    - Channels 7-9:  ``max``  (per-instance bbox-max xyz / (D,H,W))
    - Channels 10-15: ``aff`` (6 face-neighbour affinities U,D,L,R,T,B)

    Channel 0 is supervised everywhere.  Channels 1-9 are foreground-only.
    Channels 10-15 are supervised everywhere -- boundary voxels use
    SAME / replicate padding so they compare to themselves (aff = 1).

    Args:
        loss_loc:  Regression loss name for the 9 localisation channels
            (``"l1"``, ``"mse"``, ``"smooth_l1"`` + aliases).
        loss_raw:  Regression loss name for the raw-intensity channel.
        weight_min:   Weight of the 3 min-location channels (ch 1-3).
        weight_avg:   Weight of the 3 avg-location channels (ch 4-6).
        weight_max:   Weight of the 3 max-location channels (ch 7-9).
        weight_raw:   Weight of the raw-intensity channel (ch 0).
        weight_aff:   Weight of the 6 affinity channels (ch 10-15).
        foreground_only_loc:  If True, the localisation loss is averaged
            over foreground voxels only (strongly recommended -- the
            target is zero on background, so averaging everywhere would
            down-weight instance gradients proportionally to foreground
            fraction).  Default ``True``.
        aff_eps: Numerical stabiliser in the soft-Dice denominator.  The
            affinity sub-loss is ``1 - (2 Σ p t + ε) / (Σ p + Σ t + ε)``
            where ``p = sigmoid(logits)``, averaged across the 6
            direction channels (and batch).
    """

    num_channels: int = _BRAINBOW_CHANNELS

    def __init__(
        self,
        loss_loc: str = "smooth_l1",
        loss_raw: str = "l1",
        weight_min: float = 1.0,
        weight_avg: float = 1.0,
        weight_max: float = 1.0,
        weight_raw: float = 1.0,
        weight_aff: float = 1.0,
        foreground_only_loc: bool = True,
        aff_eps: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_loc = canonical_regression_name(loss_loc)
        self.loss_raw = canonical_regression_name(loss_raw)
        self._loss_loc_fn = regression_loss_fn(loss_loc)
        self._loss_raw_fn = regression_loss_fn(loss_raw)
        self.weight_min = float(weight_min)
        self.weight_avg = float(weight_avg)
        self.weight_max = float(weight_max)
        self.weight_raw = float(weight_raw)
        self.weight_aff = float(weight_aff)
        self.foreground_only_loc = bool(foreground_only_loc)
        self.aff_eps = float(aff_eps)

    @property
    def task_channels(self) -> int:
        """Expected width of the brainbow head prediction tensor (16)."""
        return _BRAINBOW_CHANNELS

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_target_raw(self, image: torch.Tensor) -> torch.Tensor:
        """Raw-intensity target (ch 0) -- identity on the input image."""
        return rearrange(image.to(torch.float32), "b ... -> b 1 ...")

    @torch.no_grad()
    def _build_target_loc(
        self,
        labels: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Per-instance localisation target (ch 1-9, min / avg / max)."""
        full = build_brainbow_target(labels, image)
        return full[:, 1:_N_LOC]

    @torch.no_grad()
    def _build_target_aff(self, labels: torch.Tensor) -> torch.Tensor:
        """6-channel SAME-pad affinity target (ch 10-15)."""
        if labels.is_cuda:
            return _affinity_target_torch(labels.long())
        lbl_np = labels.detach().cpu().long().numpy()
        return torch.from_numpy(_affinity_target_np(lbl_np)).to(labels.device)

    @torch.no_grad()
    def build_target(
        self,
        labels: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Full 16-channel brainbow target for ``(labels, image)``."""
        return build_brainbow_target(labels, image)

    # Backwards-compat alias.
    def compute_target(
        self, labels: torch.Tensor, image: torch.Tensor,
    ) -> torch.Tensor:
        return self.build_target(labels, image)

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

    def _compute_loss_loc(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Grouped (min / avg / max) 3-RGB localisation loss on ch 1-9.

        Args:
            pred:   ``[B, 9, D, H, W]`` localisation prediction.
            target: ``[B, 9, D, H, W]`` localisation target.
            labels: ``[B, D, H, W]``   instance ids.
        """
        per_voxel = self._loss_loc_fn(pred, target, reduction="none")
        if self.foreground_only_loc:
            fg = rearrange(labels > 0, "b ... -> b 1 ...").expand_as(per_voxel)
            n_fg = fg.sum().clamp(min=1)
            per_group = reduce(per_voxel * fg, "b (g c) ... -> g", "sum", g=3) / n_fg
        else:
            per_group = reduce(per_voxel, "b (g c) ... -> g", "mean", g=3)

        loss_min, loss_avg, loss_max = per_group.unbind(0)
        return {"min": loss_min, "avg": loss_avg, "max": loss_max}

    def _compute_loss_aff(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Soft-Dice loss on sigmoid affinities (ch 10-15).

        ``1 - mean_c [ (2 Σ p t + ε) / (Σ p + Σ t + ε) ]`` where
        ``p = sigmoid(logits)``, summed over every voxel of every batch
        element, and the mean is taken across the 6 direction channels.

        No mask: SAME-pad guarantees every voxel has a valid target
        (boundary voxels supervise ``aff == 1`` -- self-connected), so
        the Dice score is well-defined on the full volume.
        """
        p = pred.sigmoid()
        intersection = reduce(p * target, "b c ... -> c", "sum")
        p_sum = reduce(p, "b c ... -> c", "sum")
        t_sum = reduce(target, "b c ... -> c", "sum")
        dice = (2.0 * intersection + self.aff_eps) / (p_sum + t_sum + self.aff_eps)
        return 1.0 - dice.mean()

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
        """Compute the brainbow regression + affinity loss.

        Args:
            prediction:     ``[B, 16, D, H, W]`` model output.  Channels
                0-9 are regression, 10-15 are affinity logits.
            labels:         ``[B, D, H, W]``    instance ids.
            image:          ``[B, D, H, W]``    normalised image.
            cached_target:  Optional precomputed ``[B, 16, D, H, W]`` target
                (see :meth:`build_target`).  Useful under DDP where each
                step otherwise rebuilds it twice (loss + image logger).

        Returns:
            Dict with keys ``loss``, ``raw``, ``min``, ``avg``, ``max``,
            ``aff``.
        """
        if prediction.shape[1] != _BRAINBOW_CHANNELS:
            raise ValueError(
                f"BrainbowLoss expects {_BRAINBOW_CHANNELS}-channel prediction; "
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
        loc_losses = self._compute_loss_loc(
            prediction[:, 1:_N_LOC],
            target[:, 1:_N_LOC],
            labels,
        )
        loss_aff = self._compute_loss_aff(
            prediction[:, _N_LOC:_BRAINBOW_CHANNELS],
            target[:, _N_LOC:_BRAINBOW_CHANNELS],
        )

        total = (
            self.weight_raw * loss_raw
            + self.weight_min * loc_losses["min"]
            + self.weight_avg * loc_losses["avg"]
            + self.weight_max * loc_losses["max"]
            + self.weight_aff * loss_aff
        )

        return {
            "loss": total,
            "raw": loss_raw,
            **loc_losses,
            "aff": loss_aff,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"channels={self.task_channels}, "
            f"loss_loc='{self.loss_loc}', loss_raw='{self.loss_raw}', "
            f"weight_raw={self.weight_raw}, "
            f"weight_min={self.weight_min}, "
            f"weight_avg={self.weight_avg}, "
            f"weight_max={self.weight_max}, "
            f"weight_aff={self.weight_aff}, "
            f"foreground_only_loc={self.foreground_only_loc})"
        )
