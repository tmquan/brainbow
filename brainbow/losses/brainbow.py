"""
Brainbow loss: per-instance spatial colouring + raw-intensity reconstruction.

For every connected component ``label > 0`` in the batch this loss builds
a dense **10-channel** per-voxel target out of purely geometric and
image statistics -- no learnable parameters:

  ====  =====================================================
  ch    meaning
  ====  =====================================================
  0     *rawval*  := the raw (normalised) image intensity at that voxel
  1-3   R,G,B  := normalised  z,y,x  of the instance's **minloc**
  4-6   R,G,B  := normalised  z,y,x  of the instance's **avgloc** (centroid)
  7-9   R,G,B  := normalised  z,y,x  of the instance's **maxloc**
  ====  =====================================================

Normalisation is done relative to the **patch size**, so every channel
lives in ``[0, 1]`` regardless of anisotropy or patch resolution:

  .. math::  R = z / D, \\quad  G = y / H, \\quad  B = x / W

Background voxels (``label == 0``) are zeroed on the 9 localisation
channels; channel 0 carries the raw image value for every voxel so the
model simultaneously learns to *reconstruct* the input -- a cheap but
useful autoencoder-style auxiliary signal.

The per-instance (min, centroid, max) statistics are computed with a
single pass of :func:`scipy.ndimage.find_objects` + centroid-via-
``np.bincount`` -- fully vectorised, no Python loops over voxels.  On
GPU tensors the computation runs on-device via :func:`torch.bincount`
and a single :func:`torch.scatter_reduce_` call.

A regression loss (``l1``, ``mse``, or ``smooth_l1``) is then applied
between the model's 10-channel "brainbow" head output and this target.
Optional per-channel weights let you trade off localisation vs.
reconstruction.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


@torch.no_grad()
def _brainbow_target_torch(
    labels: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Vectorised brainbow target on the same device as ``labels``.

    Args:
        labels: ``[B, D, H, W]`` integer instance ids (``0`` = background).
        image: ``[B, D, H, W]`` raw (normalised) image intensities.

    Returns:
        ``[B, 10, D, H, W]`` float target tensor.
    """
    B, D, H, W = labels.shape
    device = labels.device
    dims_t = labels.new_tensor([D, H, W], dtype=torch.float32)

    target = labels.new_zeros((B, 10, D, H, W), dtype=torch.float32)
    target[:, 0] = image.to(torch.float32)

    # Precompute flat (z, y, x) coordinate grids once per call; both
    # arrays are tiny compared to the label volume, and re-building
    # them per-batch keeps the CUDA memory peak flat.
    zz = torch.arange(D, device=device, dtype=torch.float32)
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)

    for b in range(B):
        lbl_b = labels[b]
        fg = lbl_b > 0
        if not fg.any():
            continue

        fg_labels = lbl_b[fg]
        unique_ids, inverse = torch.unique(fg_labels, return_inverse=True)
        K = unique_ids.shape[0]

        fg_idx = torch.nonzero(fg, as_tuple=False)  # [M, 3] in (z, y, x)
        z = fg_idx[:, 0].to(torch.float32)
        y = fg_idx[:, 1].to(torch.float32)
        x = fg_idx[:, 2].to(torch.float32)

        # --- min / max / sum per-instance via scatter_reduce_ (fully on-device) ---
        INF = torch.finfo(torch.float32).max
        min_coords = z.new_full((K, 3), INF)
        max_coords = z.new_full((K, 3), -INF)
        sum_coords = z.new_zeros((K, 3))

        coords = torch.stack([z, y, x], dim=1)  # [M, 3]
        inv3 = repeat(inverse, "m -> m c", c=3)
        min_coords.scatter_reduce_(0, inv3, coords, reduce="amin", include_self=True)
        max_coords.scatter_reduce_(0, inv3, coords, reduce="amax", include_self=True)
        sum_coords.scatter_add_(0, inv3, coords)
        counts = torch.bincount(inverse, minlength=K).to(torch.float32).clamp_(min=1.0)
        centroid_coords = sum_coords / rearrange(counts, "k -> k 1")

        # --- normalise by patch dims → [K, 3] each, all in [0, 1] ---
        norm = rearrange(dims_t, "c -> 1 c").clamp(min=1.0)
        min_rgb = min_coords / norm
        cen_rgb = centroid_coords / norm
        max_rgb = max_coords / norm

        # --- broadcast per-instance colour to every foreground voxel ---
        # voxel-level RGBs: gather by `inverse` into [M, 9]
        voxel_rgb = torch.cat([min_rgb, cen_rgb, max_rgb], dim=1)[inverse]  # [M, 9]

        # scatter back into the dense target tensor; fancy-indexing on a
        # 3-D boolean mask is the fastest way on both CPU and CUDA.
        target[b, 1:10][:, fg] = rearrange(voxel_rgb, "m c -> c m")

    return target


@torch.no_grad()
def _brainbow_target_scipy(
    labels_np: np.ndarray,
    image_np: np.ndarray,
) -> np.ndarray:
    """Reference brainbow-target builder via ``scipy.ndimage``.

    Pure-NumPy path used when the input tensors live on CPU and SciPy
    is available; avoids importing scipy when the GPU path is taken.
    """
    from scipy.ndimage import find_objects  # local: only hit on CPU

    B, D, H, W = labels_np.shape
    target = np.zeros((B, 10, D, H, W), dtype=np.float32)
    target[:, 0] = image_np.astype(np.float32, copy=False)
    dims = np.array([D, H, W], dtype=np.float32).clip(min=1.0)

    for b in range(B):
        lbl = labels_np[b]
        if not lbl.any():
            continue

        fg = lbl > 0
        fg_flat = lbl[fg]
        unique_ids, inverse = np.unique(fg_flat, return_inverse=True)
        K = unique_ids.shape[0]

        # Bounding boxes via single O(N) pass
        slices = find_objects(lbl.astype(np.int64))
        min_rgb = np.zeros((K, 3), dtype=np.float32)
        max_rgb = np.zeros((K, 3), dtype=np.float32)
        for k, uid in enumerate(unique_ids):
            sl = slices[int(uid) - 1]
            if sl is None:       # id present but hole: treat as single voxel
                continue
            for a in range(3):
                min_rgb[k, a] = sl[a].start
                max_rgb[k, a] = sl[a].stop - 1

        # Centroid via bincount (vectorised)
        idx = np.nonzero(fg)                     # 3 × M  (z, y, x)
        coords = np.stack(idx, axis=1).astype(np.float32)
        counts = np.bincount(inverse, minlength=K).astype(np.float32).clip(min=1.0)
        cen_rgb = np.stack([
            np.bincount(inverse, weights=coords[:, a], minlength=K) / counts
            for a in range(3)
        ], axis=1).astype(np.float32)

        min_rgb /= dims
        cen_rgb /= dims
        max_rgb /= dims

        voxel_rgb = np.concatenate([min_rgb, cen_rgb, max_rgb], axis=1)[inverse]  # [M, 9]
        target[b, 1:10][:, fg] = voxel_rgb.T                                       # [9, M]

    return target


@torch.no_grad()
def build_brainbow_target(
    labels: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Build a ``[B, 10, D, H, W]`` brainbow target from labels + image.

    Picks the on-device torch path for CUDA tensors and the NumPy /
    scipy path for CPU tensors, since ``find_objects`` is faster than
    iterating ``torch.unique`` for very fragmented label volumes.
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


_LOSS_FN = {
    "l1": F.l1_loss,
    "mae": F.l1_loss,
    "l2": F.mse_loss,
    "mse": F.mse_loss,
    "smooth_l1": F.smooth_l1_loss,
    "huber": F.smooth_l1_loss,
}


def _resolve_loss(name: str):
    key = name.lower().replace("-", "_")
    if key not in _LOSS_FN:
        raise ValueError(
            f"Unknown loss type '{name}'. Choose from: {sorted(set(_LOSS_FN))}"
        )
    return _LOSS_FN[key]


_BRAINBOW_CHANNELS: int = 10


class BrainbowLoss(nn.Module):
    """Regression loss on a 10-channel brainbow target map.

    The target map is built on-the-fly from ``labels`` + ``image``:

    - Channel  0:    ``rawval``  (per-voxel raw image intensity)
    - Channels 1-3:  ``minloc``  normalised by (D, H, W)
    - Channels 4-6:  ``avgloc``  (centroid) normalised by (D, H, W)
    - Channels 7-9:  ``maxloc``  normalised by (D, H, W)

    The 9 localisation channels are foreground-only (background voxels
    are zero) while channel 0 is supervised everywhere.

    Args:
        loss_loc:  Regression loss for the 9 localisation channels.
        loss_raw:  Regression loss for the raw-intensity channel.
        weight_minloc:    Weight of the 3 minloc channels.
        weight_avgloc:    Weight of the 3 avgloc channels.
        weight_maxloc:    Weight of the 3 maxloc channels.
        weight_rawval:    Weight of the raw-intensity channel.
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
        weight_minloc: float = 1.0,
        weight_avgloc: float = 1.0,
        weight_maxloc: float = 1.0,
        weight_rawval: float = 1.0,
        foreground_only_loc: bool = True,
    ) -> None:
        super().__init__()
        self._loss_loc = _resolve_loss(loss_loc)
        self._loss_raw = _resolve_loss(loss_raw)
        self.weight_minloc = float(weight_minloc)
        self.weight_avgloc = float(weight_avgloc)
        self.weight_maxloc = float(weight_maxloc)
        self.weight_rawval = float(weight_rawval)
        self.foreground_only_loc = bool(foreground_only_loc)

        # Pre-register per-channel group weights as a buffer so ``.to()``
        # moves them along with the module.
        group_w = torch.tensor(
            [weight_minloc] * 3 + [weight_avgloc] * 3 + [weight_maxloc] * 3,
            dtype=torch.float32,
        )
        self.register_buffer("_loc_channel_weights", group_w, persistent=False)

    # ------------------------------------------------------------------
    # Target precomputation (for CombinedLoss._compute_targets hook)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_target(
        self,
        labels: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic, gradient-free target builder; see module docstring."""
        return build_brainbow_target(labels, image)

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
            labels:         ``[B, D, H, W]`` instance ids.
            image:          ``[B, D, H, W]`` normalised image.
            cached_target:  Optional precomputed ``[B, 10, D, H, W]`` target
                (see :meth:`compute_target`).  When supplied, the same
                tensor is reused -- useful under DDP where each step
                otherwise rebuilds it twice (loss + image logger).

        Returns:
            Dict with keys ``loss``, ``minloc``, ``avgloc``, ``maxloc``,
            and ``rawval``.
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
            cached_target
            if cached_target is not None
            else self.compute_target(labels, image)
        )
        target = target.to(dtype=prediction.dtype, device=prediction.device)

        # --- localisation sub-losses (per channel group, channels 1-9) ---
        pred_loc = prediction[:, 1:10]
        true_loc = target[:, 1:10]

        if self.foreground_only_loc:
            fg = rearrange(labels > 0, "b d h w -> b 1 d h w").expand_as(pred_loc)
            n_fg = fg.sum().clamp(min=1)
            per_voxel = self._loss_loc(pred_loc, true_loc, reduction="none") * fg
            per_group = reduce(per_voxel, "b (g c) d h w -> g", "sum", g=3) / n_fg
        else:
            per_voxel = self._loss_loc(pred_loc, true_loc, reduction="none")
            per_group = reduce(per_voxel, "b (g c) d h w -> g", "mean", g=3)

        loss_min, loss_avg, loss_max = per_group.unbind(0)

        # --- raw-intensity sub-loss (channel 0; dense, fg + bg) ---
        loss_raw = self._loss_raw(prediction[:, 0], target[:, 0])

        total = (
            self.weight_rawval * loss_raw
            + self.weight_minloc * loss_min
            + self.weight_avgloc * loss_avg
            + self.weight_maxloc * loss_max
        )

        return {
            "loss": total,
            "minloc": loss_min,
            "avgloc": loss_avg,
            "maxloc": loss_max,
            "rawval": loss_raw,
        }
