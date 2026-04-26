"""
Volume I/O utilities.

Why this file exists
--------------------
A thin façade over :mod:`brainbow.preprocessors` that picks the right
preprocessor by file suffix so callers don't have to.

Public surface
--------------
* :func:`find_folder` -- recursively locate a file by basename or
  extension under a search root.  Used by every dataset to resolve
  ``vol`` / ``seg`` keys against ``data_root``.
* :func:`load_volume` / :func:`save_volume` -- format-agnostic reader
  / writer; suffix dispatches to a :class:`BasePreprocessor` subclass.
* :func:`ensure_data` -- "make sure this is a numpy array" coerce with
  a common error path.

Extending this module
---------------------
A new format only needs a new :class:`BasePreprocessor` subclass; the
suffix dispatch here will pick it up automatically once the class is
registered in ``preprocessors/__init__.py``.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

# Supported extensions grouped by format family
SUPPORTED_EXTENSIONS: List[str] = [
    ".h5", ".hdf5", ".hdf",
    ".tiff", ".tif",
    ".nrrd", ".nhdr",
    ".npy", ".npz",
]


def ensure_data(path: Union[str, Path]) -> Path:
    """
    Ensure path exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_folder(
    root: Union[str, Path],
    base_name: str,
    extensions: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Search for a file by base name across multiple supported extensions.

    Scans the given root directory for ``base_name`` combined with each
    extension in priority order and returns the first match.

    Args:
        root: Directory to search in.
        base_name: Base filename without extension (e.g., 'AC4_inputs').
        extensions: List of extensions to try, in priority order.
            Defaults to all supported connectomics extensions.

    Returns:
        Path to the first matching file, or None if not found.

    Example:
        >>> path = find_folder("/data/snemi3d", "AC4_inputs")
        >>> # Returns e.g. Path("/data/snemi3d/AC4_inputs.h5")
    """
    root = Path(root)
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    for ext in extensions:
        candidate = root / f"{base_name}{ext}"
        if candidate.exists():
            return candidate

    return None


def load_volume(
    path: Union[str, Path],
    format: Optional[str] = None,
    key: str = "main",
) -> np.ndarray:
    """
    Load volume data from file.

    Automatically detects format based on extension if not specified.

    Args:
        path: Path to volume file.
        format: File format ('h5', 'tiff', 'nrrd'). Auto-detected if None.
        key: Dataset key for HDF5 files.

    Returns:
        Numpy array containing volume data.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If format is unsupported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Volume file not found: {path}")

    if format is None:
        format = path.suffix.lower().lstrip(".")

    if format in ("h5", "hdf5", "hdf"):
        import h5py

        with h5py.File(path, "r", locking=False) as f:
            if key in f:
                return f[key][:]
            else:
                keys = list(f.keys())
                if keys:
                    return f[keys[0]][:]
                raise KeyError(f"No datasets found in {path}")

    elif format in ("tiff", "tif"):
        import tifffile

        return tifffile.imread(str(path))

    elif format in ("nrrd", "nhdr"):
        import nrrd

        data, _ = nrrd.read(str(path))
        return data

    elif format == "npy":
        return np.load(path)

    elif format == "npz":
        data = np.load(path)
        keys = list(data.keys())
        if keys:
            return data[keys[0]]
        raise KeyError(f"No arrays found in {path}")

    else:
        raise ValueError(f"Unsupported format: {format}")


def save_volume(
    data: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    format: Optional[str] = None,
    key: str = "main",
    compression: Optional[str] = "gzip",
) -> None:
    """
    Save volume data to file.

    Automatically detects format based on extension if not specified.

    Args:
        data: Volume data as numpy array or torch tensor.
        path: Output file path.
        format: File format ('h5', 'tiff', 'nrrd'). Auto-detected if None.
        key: Dataset key for HDF5 files.
        compression: Compression for HDF5 files.
    """
    path = Path(path)
    ensure_data(path.parent)

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if format is None:
        format = path.suffix.lower().lstrip(".")

    if format in ("h5", "hdf5", "hdf"):
        import h5py

        with h5py.File(path, "w", locking=False) as f:
            f.create_dataset(key, data=data, compression=compression)

    elif format in ("tiff", "tif"):
        import tifffile

        tifffile.imwrite(str(path), data)

    elif format in ("nrrd", "nhdr"):
        import nrrd

        nrrd.write(str(path), data)

    elif format == "npy":
        np.save(path, data)

    elif format == "npz":
        np.savez_compressed(path, **{key: data})

    else:
        raise ValueError(f"Unsupported format: {format}")
