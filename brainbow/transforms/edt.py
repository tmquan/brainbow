"""GPU-accelerated image processing utilities with cucim, scipy fallback.

Provides numpy-in / numpy-out wrappers for:

- Euclidean distance transform (EDT)  — ``cucim.core.operations.morphology``
- Gaussian filter                     — ``cucim.skimage.filters``
- Per-label centroid                  — ``cucim.skimage.measure.regionprops``
- Connected-component labeling        — ``cucim.skimage.measure``
- Binary structure generation         — ``scipy.ndimage`` (no cucim equivalent)

When cucim is installed and a CUDA device is available, operations run
on the GPU.  In forked DataLoader workers the CPU path is used
automatically since CUDA contexts do not survive ``fork()``.
"""

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np

_pid_gpu_cache: dict = {}

_CUCIM_AVAILABLE: Optional[bool] = None


def _cucim_available() -> bool:
    global _CUCIM_AVAILABLE
    if _CUCIM_AVAILABLE is None:
        try:
            import cucim  # noqa: F401
            _CUCIM_AVAILABLE = True
        except ImportError:
            _CUCIM_AVAILABLE = False
    return _CUCIM_AVAILABLE


def _use_gpu() -> bool:
    """Return True when the GPU (cucim/cupy) code path should be used.

    Checks cucim availability, the ``NEURONS_FORCE_CPU`` env var, and
    whether CUDA is functional in the current process (handles fork).
    Result is cached per-PID so forked workers re-probe once.
    """
    if not _cucim_available():
        return False
    if os.environ.get("NEURONS_FORCE_CPU", ""):
        return False
    pid = os.getpid()
    if pid in _pid_gpu_cache:
        return _pid_gpu_cache[pid]
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        _pid_gpu_cache[pid] = True
        return True
    except Exception:
        _pid_gpu_cache[pid] = False
        return False


# ------------------------------------------------------------------
# Euclidean distance transform
# ------------------------------------------------------------------

def distance_transform_edt(
    mask: np.ndarray,
    sampling: Optional[Union[float, Sequence[float]]] = None,
) -> np.ndarray:
    """Euclidean distance transform with cucim GPU acceleration."""
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.core.operations.morphology import (
                distance_transform_edt as _cucim_edt,
            )
            result = _cucim_edt(cp.asarray(mask))
            return cp.asnumpy(result)
        except Exception:
            pass
    from scipy.ndimage import distance_transform_edt as _scipy_edt
    return _scipy_edt(mask, sampling=sampling)


def edt_with_indices(
    mask: np.ndarray,
    sampling: Optional[Union[float, Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Euclidean distance + indices of the nearest zero (seed) voxel.

    For each True voxel in ``mask`` returns:
    * ``distance[*spatial]`` -- Euclidean distance to the nearest False
      voxel.
    * ``indices[ndim, *spatial]`` -- coordinates of the nearest False
      voxel (one component per spatial axis, in axis order).

    The False voxels themselves get ``distance == 0`` and
    ``indices[..., v] == v``.

    Used by the skeleton-relative geometry pipeline to compute, in a
    single pass, both the per-voxel distance to the nearest skeleton
    voxel (``rad``) and the index of that voxel (used by ``dir`` /
    ``cov`` to derive the centripetal direction and the Voronoi-cell
    assignment in :class:`brainbow.transforms.SkeletonGeometryd`).

    GPU path: cucim's ``distance_transform_edt(..., return_indices=True)``;
    CPU path: scipy's equivalent.  Same API on both, so callers don't
    need to branch.

    Args:
        mask: Bool / 0-1 array ``[*spatial]``.  True = query voxel,
            False = seed voxel.  Conventionally we pass ``~skel`` so the
            seeds are skeleton voxels.
        sampling: Voxel anisotropy (one value per axis) -- forwarded to
            the underlying EDT call.  ``None`` means isotropic
            voxel-unit distances.

    Returns:
        ``(distance, indices)`` numpy arrays.  ``distance`` has dtype
        ``float32``; ``indices`` has dtype ``int32`` (cucim) or ``int64``
        (scipy).
    """
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.core.operations.morphology import (
                distance_transform_edt as _cucim_edt,
            )
            dist, idx = _cucim_edt(
                cp.asarray(mask), return_indices=True, sampling=sampling,
            )
            return cp.asnumpy(dist).astype(np.float32, copy=False), cp.asnumpy(idx)
        except Exception:
            pass
    from scipy.ndimage import distance_transform_edt as _scipy_edt
    dist, idx = _scipy_edt(
        mask, sampling=sampling, return_indices=True,
    )
    return dist.astype(np.float32, copy=False), idx


# ------------------------------------------------------------------
# Gaussian filter
# ------------------------------------------------------------------

def gaussian_filter(
    input: np.ndarray,
    sigma: Union[float, Sequence[float]],
    **kwargs,
) -> np.ndarray:
    """Gaussian filter with cucim GPU acceleration.

    GPU path uses ``cucim.skimage.filters.gaussian``; CPU fallback
    uses ``scipy.ndimage.gaussian_filter``.
    """
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.filters import gaussian as _cucim_gaussian
            mode = kwargs.pop("mode", "reflect")
            cval = kwargs.pop("cval", 0.0)
            result = _cucim_gaussian(
                cp.asarray(input),
                sigma=sigma,
                mode=mode,
                cval=cval,
                preserve_range=True,
            )
            return cp.asnumpy(result)
        except Exception:
            pass
    from scipy.ndimage import gaussian_filter as _scipy_gf
    return _scipy_gf(input, sigma, **kwargs)


# ------------------------------------------------------------------
# Per-label centroid
# ------------------------------------------------------------------

def centroid(
    labels: np.ndarray,
    index: Optional[Union[int, Sequence[int], np.ndarray]] = None,
) -> Union[Tuple[float, ...], list]:
    """Compute unweighted centroids for labeled regions.

    GPU path uses ``cucim.skimage.measure.regionprops``; CPU fallback
    uses ``scipy.ndimage.center_of_mass``.

    Returns the same format as ``scipy.ndimage.center_of_mass``:
    a single tuple when *index* is scalar, or a list of tuples.
    """
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.measure import regionprops
            props = regionprops(cp.asarray(labels))
            prop_map = {
                int(p.label): tuple(float(c) for c in p.centroid)
                for p in props
            }
            if index is None:
                return list(prop_map.values())
            idx = np.atleast_1d(np.asarray(index))
            zero = (0.0,) * labels.ndim
            result = [prop_map.get(int(uid), zero) for uid in idx]
            return result[0] if np.ndim(index) == 0 else result
        except Exception:
            pass
    from scipy.ndimage import center_of_mass as _scipy_com
    ones = np.ones_like(labels, dtype=np.float32)
    return _scipy_com(ones, labels, index)


# ------------------------------------------------------------------
# Connected-component labeling
# ------------------------------------------------------------------

def label(
    input: np.ndarray,
    structure: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Connected-component labeling with cucim GPU acceleration.

    Returns ``(labeled_array, num_features)`` matching scipy's API.
    """
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.measure import label as _cucim_label
            labeled = _cucim_label(cp.asarray(input))
            return cp.asnumpy(labeled), int(labeled.max())
        except Exception:
            pass
    from scipy.ndimage import label as _scipy_label
    return _scipy_label(input, structure=structure)


# ------------------------------------------------------------------
# Binary structure generation (no cucim equivalent — scipy only)
# ------------------------------------------------------------------

def generate_binary_structure(rank: int, connectivity: int) -> np.ndarray:
    """Generate binary structure element (always CPU, result is tiny)."""
    from scipy.ndimage import generate_binary_structure as _scipy_gbs
    return _scipy_gbs(rank, connectivity)
