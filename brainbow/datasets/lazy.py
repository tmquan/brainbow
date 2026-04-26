"""
Lazy-loading dataset for 3D volumetric connectomics data.

Reads only the requested patch from disk on each __getitem__ call,
keeping system memory usage constant regardless of volume count/size.
Designed for DDP training where each rank would otherwise load full
volumes, exhausting system RAM.

Supports HDF5 (chunked reads) and TIFF (memory-mapped reads).
File handles are cached per worker to avoid the overhead of opening
and closing HDF5 files on every sample.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_thread_local = threading.local()


class _VolumeHandle:
    """Lightweight metadata handle for a single volume — no data in RAM."""

    __slots__ = ("image_path", "label_path", "image_key", "label_key",
                 "shape", "name")

    def __init__(
        self,
        image_path: Path,
        label_path: Optional[Path],
        shape: Tuple[int, ...],
        name: str,
        image_key: Optional[str] = None,
        label_key: Optional[str] = None,
    ) -> None:
        self.image_path = image_path
        self.label_path = label_path
        self.image_key = image_key
        self.label_key = label_key
        self.shape = shape
        self.name = name


def _resolve_hdf5_key(path: Path) -> Optional[str]:
    """Find the first dataset key in an HDF5 file without loading data."""
    import h5py
    with h5py.File(str(path), "r", locking=False) as f:
        for k in ("main", "data", "raw", "volume", "image", "label"):
            if k in f:
                return k
        for k in f.keys():
            if isinstance(f[k], h5py.Dataset):
                return k
    return None


def _get_shape(path: Path, key: Optional[str] = None) -> Tuple[int, ...]:
    """Read volume shape from file metadata without loading data."""
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        import h5py
        if key is None:
            key = _resolve_hdf5_key(path)
        with h5py.File(str(path), "r", locking=False) as f:
            return tuple(f[key].shape)
    elif suffix in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            return tuple(series.shape)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _get_h5_handle(path: str):
    """Return a cached HDF5 file handle for the current worker thread.

    Handles are stored in thread-local storage so each DataLoader worker
    keeps its own set of open files.  This avoids the overhead of
    opening and parsing the HDF5 superblock on every sample.
    """
    import h5py
    cache = getattr(_thread_local, "h5_cache", None)
    if cache is None:
        cache = {}
        _thread_local.h5_cache = cache
    handle = cache.get(path)
    if handle is None:
        handle = h5py.File(path, "r", swmr=True, locking=False)
        cache[path] = handle
    return handle


def _read_patch(
    path: Path,
    slices: Tuple[slice, ...],
    key: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Read a spatial patch from disk without loading the full volume."""
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        if key is None:
            key = _resolve_hdf5_key(path)
        f = _get_h5_handle(str(path))
        data = f[key][slices]
        if dtype is not None:
            data = data.astype(dtype)
        return data
    elif suffix in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.memmap(str(path))
        data = np.array(arr[slices])
        if dtype is not None:
            data = data.astype(dtype)
        return data
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _find_file(search_dir: Path, base_name: str) -> Optional[Path]:
    """Find a volume file by base name with common extensions."""
    for ext in (".h5", ".hdf5", ".tif", ".tiff"):
        candidate = search_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    for d in search_dir.iterdir():
        if d.is_dir() and d.name == base_name:
            for ext in (".h5", ".hdf5", ".tif", ".tiff"):
                for f in d.rglob(f"*{ext}"):
                    return f
    return None


class LazyVolDataset(Dataset):
    """Memory-efficient dataset that reads 3D patches on-demand from disk.

    Instead of loading entire volumes into RAM, this dataset stores only
    file paths and volume shapes (~bytes per volume). Each ``__getitem__``
    reads a random patch directly from the HDF5/TIFF file using sliced
    I/O, materializing only the patch_size crop in memory.

    Memory usage: O(num_volumes × metadata) ≈ negligible
    vs. CacheDataset: O(num_volumes × volume_size) ≈ GBs

    Args:
        root_dir: Root directory containing data files.
        volumes: List of ``{vol, seg}`` dicts specifying volume names.
        patch_size: Spatial crop size ``(D, H, W)`` for random patches.
        transform: MONAI transform pipeline to apply to each patch.
        num_samples: Virtual epoch length (random patches per epoch).
        normalize: Whether to normalize images to [0, 1] using per-volume
            min/max (pre-computed from metadata, not full load).
        min_foreground: Minimum fraction of non-zero voxels in the label
            patch.  When > 0, patches below this threshold are rejected
            and re-sampled (up to ``max_foreground_retries`` attempts).
        max_foreground_retries: Maximum re-sampling attempts before
            accepting whatever patch was drawn.
    """

    def __init__(
        self,
        root_dir: str,
        volumes: List[Dict[str, str]],
        patch_size: Tuple[int, int, int],
        transform: Optional[Callable] = None,
        num_samples: int = 16000,
        normalize: bool = True,
        deterministic: bool = False,
        min_foreground: float = 0.0,
        max_foreground_retries: int = 50,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.deterministic = deterministic
        self.num_samples = num_samples
        self.normalize = normalize
        self.min_foreground = float(min_foreground)
        self.max_foreground_retries = int(max_foreground_retries)

        self._handles: List[_VolumeHandle] = []
        self._cum_voxels: List[int] = []
        self._norm_params: Dict[str, Tuple[float, float]] = {}

        self._discover_volumes(volumes)

        if not self._handles:
            raise ValueError(f"No volumes found in {root_dir}")

        total = sum(np.prod(h.shape) for h in self._handles)
        logger.info(
            "LazyVolDataset: %d volumes, %s total voxels, "
            "patch_size=%s, num_samples=%d, min_fg=%.0f%%, ~%.1f MB metadata",
            len(self._handles), f"{total:,}", patch_size, num_samples,
            self.min_foreground * 100,
            len(self._handles) * 0.001,
        )

    def _discover_volumes(self, volumes: List[Dict[str, str]]) -> None:
        """Scan volume files and store metadata only — no data loaded."""
        cumulative = 0
        for vol_spec in volumes:
            vol_root = Path(vol_spec.get("root", str(self.root_dir)))
            vol_name = vol_spec["vol"]

            img_path = _find_file(vol_root, vol_name)
            if img_path is None:
                logger.warning("Volume image not found: %s in %s", vol_name, vol_root)
                continue

            img_key = _resolve_hdf5_key(img_path) if img_path.suffix.lower() in (".h5", ".hdf5") else None
            shape = _get_shape(img_path, img_key)

            seg_name = vol_spec.get("seg")
            label_path = None
            label_key = None
            if seg_name:
                label_path = _find_file(vol_root, seg_name)
                if label_path is not None and label_path.suffix.lower() in (".h5", ".hdf5"):
                    label_key = _resolve_hdf5_key(label_path)

            handle = _VolumeHandle(
                image_path=img_path,
                label_path=label_path,
                shape=shape,
                name=vol_name,
                image_key=img_key,
                label_key=label_key,
            )
            self._handles.append(handle)
            cumulative += int(np.prod(shape))
            self._cum_voxels.append(cumulative)

        if self.normalize and self._handles:
            self._compute_norm_params()

    # ------------------------------------------------------------------
    # Per-volume normalisation parameters
    # ------------------------------------------------------------------
    # Reads a handful of *small* centred patches (not full slices) to
    # estimate min/max, then caches the result to a ``<file>.norm.json``
    # sidecar so subsequent ranks / runs load instantly.  Full-slice
    # sampling on chunked gzip HDF5 can take 10+ s per volume, which
    # multiplied across ranks (setup runs per-rank) easily turns into a
    # multi-minute hang before Lightning's sanity-check banner prints.
    _NORM_PROBE_SIZE = 512      # ≤ a few MB per probe; trivial to decode
    _NORM_PROBE_COUNT = 5       # how many probes to average min/max over

    def _norm_cache_path(self, handle: _VolumeHandle) -> Path:
        return handle.image_path.with_suffix(handle.image_path.suffix + ".norm.json")

    def _read_norm_cache(self, handle: _VolumeHandle) -> Optional[Tuple[float, float]]:
        import json
        p = self._norm_cache_path(handle)
        if not p.exists():
            return None
        try:
            with open(p, "r") as f:
                data = json.load(f)
            return float(data["min"]), float(data["max"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # OSError: race with concurrent writer / unreadable file.
            # ValueError / json.JSONDecodeError: corrupted sidecar.
            # KeyError: schema mismatch (older format).  In all cases
            # we treat the cache as cold and force a recompute.
            return None

    def _write_norm_cache(self, handle: _VolumeHandle, vmin: float, vmax: float) -> None:
        import json, os
        p = self._norm_cache_path(handle)
        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump({"min": vmin, "max": vmax}, f)
            os.replace(tmp, p)
        except OSError:
            pass   # non-fatal: read-only filesystem, concurrent rank, etc.

    def _compute_norm_params(self) -> None:
        """Per-volume min/max from small centred probes with sidecar caching."""
        probe = self._NORM_PROBE_SIZE
        n_probes = self._NORM_PROBE_COUNT

        for h in self._handles:
            cached = self._read_norm_cache(h)
            if cached is not None:
                self._norm_params[h.name] = cached
                logger.debug("Norm params cache hit for %s: %s", h.name, cached)
                continue

            spatial = self._spatial_shape(h)
            has_channel = len(h.shape) > len(spatial)
            z_dim = spatial[0]

            z_indices = sorted(
                set(
                    max(0, min(i, z_dim - 1))
                    for i in (0, z_dim // 4, z_dim // 2, 3 * z_dim // 4, z_dim - 1)
                )
            )[:n_probes]

            vmin, vmax = float("inf"), float("-inf")
            for zi in z_indices:
                # Centred (probe × probe) crop at z=zi; orders of
                # magnitude fewer chunks than a full-slice read.
                sl = [slice(zi, zi + 1)]
                for axis_len in spatial[1:]:
                    half = min(probe, axis_len) // 2
                    centre = axis_len // 2
                    sl.append(slice(max(0, centre - half), min(axis_len, centre + half)))
                sl_tuple = tuple(sl)
                if has_channel:
                    sl_tuple = (slice(None),) + sl_tuple
                patch = _read_patch(h.image_path, sl_tuple, h.image_key, dtype=np.float32)
                vmin = min(vmin, float(patch.min()))
                vmax = max(vmax, float(patch.max()))

            self._norm_params[h.name] = (vmin, vmax)
            self._write_norm_cache(h, vmin, vmax)
            logger.debug("Norm params for %s: min=%.4f, max=%.4f", h.name, vmin, vmax)

    def __len__(self) -> int:
        return self.num_samples

    def _pick_volume(self, index: int) -> _VolumeHandle:
        """Select a volume, weighted by total voxel count."""
        rng = np.random.RandomState(index)
        total = self._cum_voxels[-1]
        r = rng.randint(0, total)
        for i, cum in enumerate(self._cum_voxels):
            if r < cum:
                return self._handles[i]
        return self._handles[-1]

    def _random_patch_slices(
        self, shape: Tuple[int, ...], rng: np.random.RandomState,
    ) -> Tuple[slice, ...]:
        """Generate random crop slices that fit within the volume."""
        slices = []
        for dim_size, patch_dim in zip(shape, self.patch_size):
            max_start = max(0, dim_size - patch_dim)
            start = rng.randint(0, max_start + 1)
            slices.append(slice(start, start + patch_dim))
        return tuple(slices)

    def _spatial_shape(self, handle: _VolumeHandle) -> Tuple[int, ...]:
        """Strip leading channel dim if present, returning only spatial dims."""
        shape = handle.shape
        if len(shape) == len(self.patch_size) + 1:
            return shape[1:]
        return shape

    def _check_foreground(
        self, handle: _VolumeHandle, crop_slices: Tuple[slice, ...],
    ) -> bool:
        """Return True if the label patch has enough foreground."""
        if self.min_foreground <= 0 or handle.label_path is None:
            return True
        label = _read_patch(
            handle.label_path, crop_slices, handle.label_key, dtype=np.int64,
        )
        fg_frac = float(np.count_nonzero(label)) / label.size
        return fg_frac >= self.min_foreground

    def __getitem__(self, index: int) -> Dict[str, Any]:
        seed = index if self.deterministic else index + int(torch.randint(0, 2**31, (1,)).item())
        rng = np.random.RandomState(seed)
        handle = self._pick_volume(index)
        spatial = self._spatial_shape(handle)

        crop_slices = self._random_patch_slices(spatial, rng)
        if len(handle.shape) > len(spatial):
            full_slices = (slice(None),) + crop_slices
        else:
            full_slices = crop_slices

        for _ in range(self.max_foreground_retries):
            if self._check_foreground(handle, full_slices):
                break
            crop_slices = self._random_patch_slices(spatial, rng)
            full_slices = ((slice(None),) + crop_slices
                           if len(handle.shape) > len(spatial)
                           else crop_slices)

        image = _read_patch(
            handle.image_path, full_slices, handle.image_key, dtype=np.float32,
        )

        if self.normalize and handle.name in self._norm_params:
            vmin, vmax = self._norm_params[handle.name]
            if vmax > vmin:
                image = (image - vmin) / (vmax - vmin)

        sample: Dict[str, Any] = {
            "image": image,
            "volume": handle.name,
        }

        if handle.label_path is not None:
            label = _read_patch(
                handle.label_path, full_slices, handle.label_key, dtype=np.int64,
            )
            sample["label"] = label

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
