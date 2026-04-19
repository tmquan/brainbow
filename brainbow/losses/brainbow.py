"""
Brainbow loss: per-instance spatial colouring + raw-intensity reconstruction.

For every connected component ``label > 0`` in the batch this loss builds
a dense **10-channel** per-voxel target out of purely geometric and
image statistics -- no learnable parameters:

  ====  =====================================================
  ch    meaning
  ====  =====================================================
  0     *raw*  := raw (normalised) image intensity at that voxel
  1-3   R,G,B  := normalised  z,y,x  of the instance's **min**  (bbox min)
  4-6   R,G,B  := normalised  z,y,x  of the instance's **avg**  (centroid)
  7-9   R,G,B  := normalised  z,y,x  of the instance's **max**  (bbox max)
  ====  =====================================================

Normalisation is relative to the **patch size**, so every channel lives
in ``[0, 1]`` regardless of anisotropy or patch resolution:

    R = z / D, G = y / H, B = x / W

Background voxels (``label == 0``) are zeroed on the 9 localisation
channels; channel 0 carries the raw image value for every voxel so the
model simultaneously learns to *reconstruct* the input -- a cheap but
useful autoencoder-style auxiliary signal.

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


_BRAINBOW_CHANNELS: int = 10


# ---------------------------------------------------------------------------
# Low-level builders (used by BrainbowLoss._build_target_*)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _brainbow_target_torch(
    labels: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Vectorised 10-channel brainbow target on the same device as ``labels``.

    Args:
        labels: ``[B, D, H, W]`` integer instance ids (``0`` = background).
        image:  ``[B, D, H, W]`` raw (normalised) image intensities.

    Returns:
        ``[B, 10, D, H, W]`` float target tensor.
    """
    B, D, H, W = labels.shape
    device = labels.device
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
        target[b, 1:_BRAINBOW_CHANNELS][:, fg] = rearrange(voxel_rgb, "m c -> c m")

    return target


@torch.no_grad()
def _brainbow_target_scipy(
    labels_np: np.ndarray,
    image_np: np.ndarray,
) -> np.ndarray:
    """Reference 10-channel brainbow-target builder via ``scipy.ndimage``.

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
        target[b, 1:_BRAINBOW_CHANNELS][:, fg] = voxel_rgb.T

    return target


@torch.no_grad()
def build_brainbow_target(
    labels: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Build a ``[B, 10, D, H, W]`` brainbow target from labels + image.

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
    """Regression loss on a 10-channel brainbow target map.

    The target map is built on-the-fly from ``labels`` + ``image``:

    - Channel  0:   ``raw``  (per-voxel raw image intensity)
    - Channels 1-3: ``min``  (per-instance bbox-min xyz, normalised by D,H,W)
    - Channels 4-6: ``avg``  (per-instance centroid, normalised by D,H,W)
    - Channels 7-9: ``max``  (per-instance bbox-max xyz, normalised by D,H,W)

    The 9 localisation channels are foreground-only (background voxels
    are zero) while channel 0 is supervised everywhere.

    Args:
        loss_loc:  Regression loss name for the 9 localisation channels.
        loss_raw:  Regression loss name for the raw-intensity channel.
        weight_min:   Weight of the 3 min-location channels (ch 1-3).
        weight_avg:   Weight of the 3 avg-location channels (ch 4-6).
        weight_max:   Weight of the 3 max-location channels (ch 7-9).
        weight_raw:   Weight of the raw-intensity channel (ch 0).
        foreground_only_loc:  If True, the localisation loss is averaged
            over foreground voxels only (strongly recommended -- the
            target is zero on background, so averaging everywhere would
            down-weight instance gradients proportionally to foreground
            fraction).  Default ``True``.
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
        foreground_only_loc: bool = True,
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
        self.foreground_only_loc = bool(foreground_only_loc)

    @property
    def task_channels(self) -> int:
        """Expected width of the brainbow head prediction tensor (10)."""
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
        """Per-instance localisation target (ch 1-9, min / avg / max).

        Returned as a ``[B, 9, D, H, W]`` tensor -- the raw channel is
        *not* concatenated here; :meth:`build_target` does that for us.
        """
        # The on-device kernels produce the full 10-channel stack (raw
        # at ch 0 + localisation at ch 1-9) in one pass since that is
        # the path they are most efficient on.  Slice off the raw
        # channel when only the localisation part is asked for.
        full = build_brainbow_target(labels, image)
        return full[:, 1:_BRAINBOW_CHANNELS]

    @torch.no_grad()
    def build_target(
        self,
        labels: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Full 10-channel brainbow target for ``(labels, image)``."""
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
        """Compute the brainbow regression loss.

        Args:
            prediction:     ``[B, 10, D, H, W]`` model output.
            labels:         ``[B, D, H, W]``    instance ids.
            image:          ``[B, D, H, W]``    normalised image.
            cached_target:  Optional precomputed ``[B, 10, D, H, W]`` target
                (see :meth:`build_target`).  Useful under DDP where each
                step otherwise rebuilds it twice (loss + image logger).

        Returns:
            Dict with keys ``loss``, ``raw``, ``min``, ``avg``, ``max``.
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
            prediction[:, 1:_BRAINBOW_CHANNELS],
            target[:, 1:_BRAINBOW_CHANNELS],
            labels,
        )

        total = (
            self.weight_raw * loss_raw
            + self.weight_min * loc_losses["min"]
            + self.weight_avg * loc_losses["avg"]
            + self.weight_max * loc_losses["max"]
        )

        return {
            "loss": total,
            "raw": loss_raw,
            **loc_losses,
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
            f"foreground_only_loc={self.foreground_only_loc})"
        )
