"""Boundary detection with cucim GPU acceleration and skimage fallback."""

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from brainbow.transforms.edt import _use_gpu


def find_boundaries(
    label,
    mode: str = "inner",
    connectivity: int = 1,
    **kwargs,
):
    """Find boundaries between labeled regions.

    Accepts ``numpy.ndarray`` or CUDA ``torch.Tensor``.  When a CUDA
    tensor is passed and cucim is available, the entire operation stays
    on the GPU via DLPack zero-copy — no CPU roundtrip.

    Args:
        label: Integer label array ``[*spatial]`` (numpy or torch).
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).
        connectivity: Neighbourhood connectivity.  ``1`` = face-adjacent
            only (6-connected in 3D, thinnest boundaries).  Higher values
            include edge/corner neighbours (up to 26-connected in 3D).

    Returns:
        Boolean boundary mask, same type and device as *label*.
    """
    is_tensor = isinstance(label, torch.Tensor)

    if is_tensor and label.is_cuda and _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.segmentation import (
                find_boundaries as _cucim_fb,
            )
            cp_label = cp.from_dlpack(label)
            cp_bnd = _cucim_fb(cp_label, mode=mode,
                               connectivity=connectivity, **kwargs)
            return torch.from_dlpack(cp_bnd)
        except Exception:
            pass

    if is_tensor:
        label_np = label.detach().cpu().numpy()
    else:
        label_np = label

    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.segmentation import (
                find_boundaries as _cucim_fb,
            )
            return cp.asnumpy(_cucim_fb(cp.asarray(label_np), mode=mode,
                                        connectivity=connectivity, **kwargs))
        except Exception:
            pass

    from skimage.segmentation import find_boundaries as _skimage_fb
    return _skimage_fb(label_np, mode=mode, connectivity=connectivity, **kwargs)


# ------------------------------------------------------------------
# Anisotropy-aware boundary detection
# ------------------------------------------------------------------

def _find_boundaries_xy(
    label_3d: np.ndarray,
    mode: str = "inner",
    connectivity: int = 1,
) -> np.ndarray:
    """Detect boundaries only in the xy plane (z-neighbours ignored).

    For anisotropic volumes where z-resolution is much coarser than xy
    (e.g. 6 × 6 × 30 nm), 3-D boundary detection creates boundaries
    that are physically 5× thicker along z.  This function checks only
    xy-neighbours via vectorised numpy shifts over the full volume —
    no Python loop over depth slices.

    Args:
        label_3d: ``[D, H, W]`` integer label array.
        mode: ``'inner'`` (foreground boundary), ``'outer'`` (background
            boundary), or ``'thick'`` (both sides).
        connectivity: 1 = 4-connected in xy (face-adjacent, thinnest).
            2 = 8-connected (adds diagonal xy neighbours).

    Returns:
        Boolean boundary mask, same shape as *label_3d*.
    """
    padded = np.pad(label_3d, ((0, 0), (1, 1), (1, 1)), mode="edge")
    center = padded[:, 1:-1, 1:-1]

    boundary = (
        (center != padded[:, 1:-1, 2:])
        | (center != padded[:, 1:-1, :-2])
        | (center != padded[:, 2:, 1:-1])
        | (center != padded[:, :-2, 1:-1])
    )

    if connectivity >= 2:
        boundary |= (
            (center != padded[:, 2:, 2:])
            | (center != padded[:, 2:, :-2])
            | (center != padded[:, :-2, 2:])
            | (center != padded[:, :-2, :-2])
        )

    if mode == "inner":
        boundary &= label_3d > 0
    elif mode == "outer":
        boundary &= label_3d == 0

    return boundary


def boundary_mask_batch(
    labels: torch.Tensor,
    mode: str = "inner",
    connectivity: int = 1,
) -> torch.Tensor:
    """Batch boundary mask using thinnest connectivity (6-connected in 3D).

    Args:
        labels: Instance labels [B, *spatial].
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).
        connectivity: 1 = face-adjacent only (thinnest).

    Returns:
        Boolean mask [B, *spatial], True at boundary voxels.
    """
    parts = []
    for b in range(labels.shape[0]):
        bnd = find_boundaries(labels[b], mode=mode, connectivity=connectivity)
        if isinstance(bnd, np.ndarray):
            parts.append(torch.from_numpy(bnd).to(labels.device))
        else:
            parts.append(bnd)
    return torch.stack(parts)


class FindBoundariesd(MapTransform, Randomizable):
    """Zero out boundary voxels in instance labels (label × (1 − boundary)).

    Uses thinnest boundary (connectivity=1) so boundary voxels are
    zeroed: ``label[boundary] = 0``.

    **Anisotropy handling** — When ``pixel_size`` is provided and the
    z-resolution is more than 2× coarser than xy, boundary detection is
    restricted to the xy plane (no z-neighbours).  This prevents
    physically thick boundaries along the low-resolution axis.

    For single-channel (binary) semantic segmentation, boundary voxels
    become 0 (background) in the keyed label and the derived semantic
    target ``(label > 0)``.  This teaches the semantic head to predict
    thin gaps between touching instances, aiding instance separation.

    The keyed array is the only thing modified, so the caller controls
    scope: applying this to ``label`` erodes everything derived from it
    (sem + affinity targets), while applying it to a dedicated
    ``sem_label`` copy (the datamodule's ``boundary_target: semantic``
    mode) erodes the foreground target only and leaves the instance
    ``label`` -- hence the affinity targets -- untouched.

    Expects input labels in ``[C, *spatial]`` format (post
    ``EnsureChannelFirstd``).  Each channel is processed independently.

    Args:
        keys: Keys of instance label maps to process.
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).
        connectivity: 1 = face-adjacent only (thinnest boundaries).
        prob: Probability of applying the transform per sample.
        pixel_size: Voxel dimensions ``(z, y, x)`` in physical units,
            matching the array dimension order.  When z is >2× coarser
            than xy, xy-only boundary detection is used automatically.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "inner",
        connectivity: int = 1,
        prob: float = 1.0,
        pixel_size: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(keys)
        self.mode = mode
        self.connectivity = connectivity
        self.prob = prob
        self._do_transform = True

        self._xy_only = False
        if pixel_size is not None and len(pixel_size) == 3:
            z_res = float(pixel_size[0])
            xy_min = float(min(pixel_size[1], pixel_size[2]))
            if xy_min > 0 and z_res / xy_min > 2.0:
                self._xy_only = True

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)

        d = dict(data)

        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)

            if is_tensor:
                device = arr.device
                label_np = arr.cpu().numpy().copy()
            else:
                label_np = np.array(arr, copy=True)

            if label_np.ndim > 1:
                for c in range(label_np.shape[0]):
                    label_np[c] = self._process_volume(label_np[c])
            else:
                label_np = self._process_volume(label_np)

            if is_tensor:
                label_np = torch.from_numpy(label_np).to(device)

            d[key] = label_np

        return d

    def _process_volume(self, vol: np.ndarray) -> np.ndarray:
        """Detect boundaries and zero them for a single spatial volume."""
        if self._xy_only and vol.ndim == 3:
            boundaries = _find_boundaries_xy(
                vol, mode=self.mode, connectivity=self.connectivity,
            )
        else:
            bnd = find_boundaries(
                vol, mode=self.mode, connectivity=self.connectivity,
            )
            boundaries = bnd if isinstance(bnd, np.ndarray) else np.asarray(bnd)

        vol[boundaries] = 0
        return vol
