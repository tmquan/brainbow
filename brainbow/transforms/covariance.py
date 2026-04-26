"""Spatial covariance field for instance segmentation labels.

For each foreground instance, computes the 2nd-order spatial statistics
(covariance matrix) from ``moments_central`` provided by
``cucim.skimage.measure.regionprops`` (GPU) or
``skimage.measure.regionprops`` (CPU), then stores the upper-triangle
entries at every pixel of the instance.
"""

import numpy as np
from monai.config import KeysCollection

from brainbow.transforms._region_field import RegionFieldTransformd
from brainbow.transforms.edt import _use_gpu


def _regionprops(label_np: np.ndarray):
    """Run regionprops — cucim GPU when available, skimage CPU otherwise.

    The cucim path can fail at *any* of: cupy/cucim import (missing
    package on a CPU-only build), CUDA-side allocation (OOM on a busy
    device), or kernel launch (driver mismatch).  Each of these
    surfaces as a different exception class, so we catch the union of
    them rather than bare ``Exception`` and fall through to skimage.
    """
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.measure import regionprops
            return regionprops(cp.asarray(label_np))
        except (ImportError, ModuleNotFoundError, MemoryError, RuntimeError):
            pass
    from skimage.measure import regionprops
    return regionprops(label_np)


def _to_numpy(arr) -> np.ndarray:
    """cupy or numpy array -> numpy."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def compute_covariance_field(
    label: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """Compute per-pixel spatial covariance for each instance.

    Extracts covariance entries from ``regionprops.moments_central``
    rather than computing deltas from scratch.

    Args:
        label: Instance label array ``[*spatial]``.  Background is 0.
        normalized: Divide each covariance matrix by its trace so
            that values are scale-invariant.

    Returns:
        Upper-triangle covariance ``[S*(S+1)//2, *spatial]`` where
        ``S = label.ndim``.  Channel order for 2-D is
        ``(Cyy, Cyx, Cxx)``; for 3-D ``(Czz, Czy, Czx, Cyy, Cyx, Cxx)``.
    """
    label_np = np.asarray(label, dtype=np.int64)
    S = label_np.ndim
    n_tri = S * (S + 1) // 2
    shape = label_np.shape
    cov_field = np.zeros((n_tri,) + shape, dtype=np.float32)

    if not np.any(label_np > 0):
        return cov_field

    tri_pairs = [(i, j) for i in range(S) for j in range(i, S)]

    for p in _regionprops(label_np):
        mu = _to_numpy(p.moments_central)
        area = float(mu[(0,) * S])
        if area < 2:
            continue

        mask = label_np == int(p.label)

        for ch, (i, j) in enumerate(tri_pairs):
            idx = [0] * S
            idx[i] += 1
            idx[j] += 1
            cov_field[ch][mask] = float(mu[tuple(idx)]) / area

    if normalized:
        trace = np.zeros(shape, dtype=np.float32)
        for ch, (i, j) in enumerate(tri_pairs):
            if i == j:
                trace += cov_field[ch]
        fg = label_np > 0
        trace_safe = np.where(fg, np.clip(trace, 1e-8, None), 1.0)
        cov_field[:, fg] /= trace_safe[fg]

    return cov_field


class Covarianced(RegionFieldTransformd):
    """Compute per-pixel spatial covariance field for instance labels.

    Reads instance labels from each key and stores the covariance field
    under ``{key}_covariance``.  Input labels are expected in
    ``[C, *spatial]`` format (post ``EnsureChannelFirstd``); the first
    channel is used.

    Args:
        keys: Keys of instance label maps.
        spatial_dims: Number of spatial dimensions (2 or 3).
        normalized: Normalize covariance by its trace.
    """

    output_suffix = "_covariance"

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
        normalized: bool = True,
    ) -> None:
        super().__init__(keys, spatial_dims=spatial_dims)
        self.normalized = normalized

    def _compute(self, label_np: np.ndarray) -> np.ndarray:
        return compute_covariance_field(label_np, normalized=self.normalized)
