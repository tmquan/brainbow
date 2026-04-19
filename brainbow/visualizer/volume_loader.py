"""
Unified volume loader that delegates to the existing preprocessors.

Wraps :mod:`brainbow.preprocessors` to load any supported format into a
:class:`VolumeData` with shape, dtype, and anisotropic voxel spacing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from brainbow.preprocessors import (
    HDF5Preprocessor,
    TIFFPreprocessor,
    NRRDPreprocessor,
    NFTYPreprocessor,
)

_hdf5 = HDF5Preprocessor()
_tiff = TIFFPreprocessor()
_nrrd = NRRDPreprocessor()
_nfty = NFTYPreprocessor()

_EXT_MAP = {
    ".h5": _hdf5, ".hdf5": _hdf5, ".hdf": _hdf5, ".he5": _hdf5,
    ".tif": _tiff, ".tiff": _tiff,
    ".nrrd": _nrrd, ".nhdr": _nrrd,
    ".nii": _nfty, ".nii.gz": _nfty, ".gz": _nfty,
    ".npy": None, ".npz": None,
}


def _resolve_ext(path: str) -> str:
    """Return the canonical extension, handling ``.nii.gz``."""
    p = Path(path)
    if len(p.suffixes) >= 2 and p.suffixes[-2:] == [".nii", ".gz"]:
        return ".nii.gz"
    return p.suffix.lower()


def _load_numpy(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        return np.asarray(data[list(data.files)[0]])
    return np.asarray(np.load(path))


def _extract_spacing_hdf5(path: str) -> Optional[Tuple[float, ...]]:
    """Try to read spacing / resolution from HDF5 root attrs."""
    import h5py
    try:
        with h5py.File(path, "r", locking=False) as f:
            for attr_name in ("spacing", "resolution", "voxel_size"):
                if attr_name in f.attrs:
                    return tuple(float(v) for v in f.attrs[attr_name])
    except Exception:
        pass
    return None


class VolumeData:
    """In-memory 3-D volume with anisotropic spacing metadata.

    Attributes:
        data: ``np.ndarray`` of shape ``(Z, Y, X)``.
        spacing: voxel size as ``(sz, sy, sx)``.  Defaults to isotropic
            ``(1.0, 1.0, 1.0)`` when the file has no spacing info.
    """

    def __init__(
        self,
        data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        if data.ndim == 4 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Expected 3-D volume, got shape {data.shape}")
        self.data = data
        self.spacing = spacing

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def to_uint8(self) -> np.ndarray:
        """Return data normalised to ``[0, 255]`` uint8."""
        d = self.data.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-8:
            return np.zeros_like(d, dtype=np.uint8)
        return ((d - lo) / (hi - lo) * 255).astype(np.uint8)

    def slice_axis(self, axis: int, idx: int) -> np.ndarray:
        """Extract a 2-D slice along *axis* (0=Z/axial, 1=Y/coronal, 2=X/sagittal)."""
        idx = max(0, min(idx, self.data.shape[axis] - 1))
        slices = [slice(None)] * 3
        slices[axis] = idx
        return self.data[tuple(slices)]

    def aspect_ratio(self, axis: int) -> float:
        """Pixel aspect ratio (height / width) for a slice orthogonal to *axis*.

        Accounts for anisotropic voxel spacing.  The returned value should
        be applied as ``canvas_height / canvas_width`` or used to scale
        the vertical axis in a renderer.

        axis 0 (axial  / XY plane): rows=Y, cols=X  -> sy / sx
        axis 1 (coronal/ XZ plane): rows=Z, cols=X  -> sz / sx
        axis 2 (sagittal/YZ plane): rows=Z, cols=Y  -> sz / sy
        """
        sz, sy, sx = self.spacing
        if axis == 0:
            return sy / sx
        elif axis == 1:
            return sz / sx
        else:
            return sz / sy

    def slice_physical_size(self, axis: int) -> Tuple[float, float]:
        """Physical (height, width) in spacing units for a slice along *axis*."""
        sz, sy, sx = self.spacing
        Z, Y, X = self.shape
        if axis == 0:
            return (Y * sy, X * sx)
        elif axis == 1:
            return (Z * sz, X * sx)
        else:
            return (Z * sz, Y * sy)


def load_volume(
    path: str,
    key: Optional[str] = None,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> VolumeData:
    """Load a volume file, auto-detecting the format via preprocessors.

    Args:
        path: path to HDF5, TIFF, NRRD, NIfTI, or NumPy file.
        key: HDF5 dataset key (auto-detected if ``None``).
        spacing: explicit ``(sz, sy, sx)`` voxel spacing.  When ``None``
            the loader tries to read it from file metadata, falling
            back to ``(1.0, 1.0, 1.0)``.

    Returns:
        :class:`VolumeData` instance.
    """
    ext = _resolve_ext(path)
    preprocessor = _EXT_MAP.get(ext)

    if ext in (".npy", ".npz"):
        arr = _load_numpy(path)
        sp = spacing or (1.0, 1.0, 1.0)
        return VolumeData(data=arr, spacing=sp)

    if preprocessor is None:
        supported = sorted(set(_EXT_MAP.keys()) - {None})
        raise ValueError(f"Unsupported format '{ext}'. Supported: {supported}")

    # Load via the preprocessor
    if isinstance(preprocessor, HDF5Preprocessor):
        if key:
            preprocessor = HDF5Preprocessor(default_key=key)
        arr = preprocessor.load(path, key=key)
        file_spacing = spacing or _extract_spacing_hdf5(path)
    elif isinstance(preprocessor, NRRDPreprocessor):
        arr = preprocessor.load(path)
        file_spacing = spacing or preprocessor.get_spacing(path)
    elif isinstance(preprocessor, NFTYPreprocessor):
        arr = preprocessor.load(path)
        file_spacing = spacing
    else:
        arr = preprocessor.load(path)
        file_spacing = spacing

    sp = file_spacing or (1.0, 1.0, 1.0)
    if len(sp) < 3:
        sp = (1.0,) * (3 - len(sp)) + tuple(sp)

    return VolumeData(data=np.asarray(arr), spacing=sp[:3])
