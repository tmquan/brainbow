"""Random resolution zoom for multi-dataset resolution harmonisation.

Different EM datasets have different native resolutions:

    SNEMI3D / Neurons     6 × 6 × 30 nm
    MICrONS               8 × 8 × 40 nm

``RandResolutionZoomd`` randomly zooms each training patch so that,
regardless of native resolution, the model sees the target resolution
range (default 6–8 nm XY, 30–40 nm Z).  The 5:1 Z:XY ratio is
maintained.

Zoom factors are computed as ``native / target`` per axis:

* zoom > 1 → upsample (simulate higher resolution than native)
* zoom < 1 → downsample (simulate lower resolution than native)
* zoom = 1 → no change

The target resolution is sampled once (Y axis) and the other axes are
derived from the native anisotropy ratio, so ``pixel_size`` proportions
are preserved (e.g. all datasets keep their 5:1 Z:XY ratio).

When training on mixed datasets with different native resolutions,
use ``resolution_map`` to specify per-volume resolutions.  The
transform reads the ``"volume"`` key from the sample dict and looks
up the matching entry (longest prefix match).

Output spatial size is preserved via center-crop / zero-pad.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


# Default target range: (min_nm, max_nm) per axis in (Z, Y, X) order.
# Spans SNEMI3D (6×6×30) to MICrONS (8×8×40), keeping 5:1 Z:XY ratio.
DEFAULT_TARGET_RANGE: Tuple[Tuple[float, float], ...] = (
    (30.0, 40.0),  # Z
    (6.0, 8.0),    # Y
    (6.0, 8.0),    # X
)


def _zoom_volume(
    vol: torch.Tensor,
    zoom: Sequence[float],
    mode: str = "trilinear",
) -> torch.Tensor:
    """Zoom a (C, D, H, W) tensor and crop/pad back to the original size.

    For nearest-neighbour mode (labels), large integer IDs are remapped
    to compact sequential values before the float32 interpolation and
    restored afterwards.  This avoids precision loss that would merge
    distinct segment IDs (e.g. MICrONS uint64 IDs ~8.6e17).
    """
    orig_shape = vol.shape[1:]  # (D, H, W)
    new_shape = [max(1, round(s * z)) for s, z in zip(orig_shape, zoom)]

    # For nearest-neighbour, remap to small IDs to avoid float32 precision loss
    remap_lut = None
    work = vol
    if mode == "nearest":
        uniq = torch.unique(vol)
        if uniq.numel() > 0 and (uniq.max().item() > 2**23 or uniq.min().item() < -(2**23)):
            remap_lut = uniq  # original IDs, sorted
            fwd = torch.zeros(int(uniq.max().item()) + 1, dtype=vol.dtype,
                              device=vol.device) if uniq.max().item() < 1_000_000 else None
            if fwd is not None:
                for i, uid in enumerate(uniq):
                    fwd[uid.item()] = i
                work = fwd[vol]
            else:
                work = torch.zeros_like(vol)
                for i, uid in enumerate(uniq):
                    work[vol == uid] = i

    # Interpolate
    vol_5d = rearrange(work.float(), "c d h w -> 1 c d h w")
    zoomed = F.interpolate(vol_5d, size=new_shape, mode=mode,
                           align_corners=False if mode != "nearest" else None)
    zoomed = rearrange(zoomed, "1 c d h w -> c d h w")

    # Center-crop / zero-pad back to orig_shape
    out = torch.zeros_like(vol)
    for d in range(3):
        zs = zoomed.shape[d + 1]
        os = orig_shape[d]
        if zs >= os:
            start = (zs - os) // 2
            zoomed = _slice_dim(zoomed, d + 1, start, start + os)
        else:
            pad_before = (os - zs) // 2
            zoomed = _pad_dim(zoomed, d + 1, pad_before, os - zs - pad_before)

    if remap_lut is not None:
        idx = zoomed.long().clamp(0, remap_lut.numel() - 1)
        out[:] = remap_lut[idx]
    else:
        out[:] = zoomed
    return out


def _slice_dim(t: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    """Slice tensor along a given dimension."""
    idx = [slice(None)] * t.ndim
    idx[dim] = slice(start, end)
    return t[tuple(idx)]


def _pad_dim(t: torch.Tensor, dim: int, before: int, after: int) -> torch.Tensor:
    """Pad tensor with zeros along a given dimension."""
    # F.pad uses reversed dim order and (left, right) per dim from last
    ndim = t.ndim
    pad = [0] * (2 * ndim)
    rev_dim = ndim - 1 - dim
    pad[2 * rev_dim] = before
    pad[2 * rev_dim + 1] = after
    return F.pad(t, pad, mode="constant", value=0)


class RandResolutionZoomd(MapTransform, Randomizable):
    """Randomly zoom image and label to simulate a different physical resolution.

    Args:
        keys: Keys to apply the zoom to (e.g. ``["image", "label"]``).
        native_resolution: Default native voxel size in nm, ``(Z, Y, X)``.
        target_range: Per-axis ``(min_nm, max_nm)`` in ``(Z, Y, X)`` order.
            Default covers the SNEMI3D–MICrONS range (30–40 Z, 6–8 XY).
        prob: Probability of applying the zoom (0–1).
        label_keys: Subset of *keys* that should use nearest-neighbour
            interpolation (default: ``{"label"}``).
        resolution_map: Optional mapping from volume-name prefixes to
            native resolutions ``(Z, Y, X)``.  When the sample contains
            a ``"volume"`` key, the longest matching prefix is used
            instead of *native_resolution*.  Example::

                {"minnie65": (40, 8, 8), "AC4": (30, 6, 6)}

        volume_key: Sample dict key that holds the volume name
            (default ``"volume"``).
    """

    def __init__(
        self,
        keys: KeysCollection,
        native_resolution: Tuple[float, float, float],
        target_range: Tuple[Tuple[float, float], ...] = DEFAULT_TARGET_RANGE,
        prob: float = 1.0,
        label_keys: Optional[set] = None,
        resolution_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
        volume_key: str = "volume",
    ) -> None:
        super().__init__(keys)
        self.native_resolution = np.asarray(native_resolution, dtype=np.float64)
        self.target_range = np.asarray(target_range, dtype=np.float64)  # (3, 2)
        self.prob = float(prob)
        self.label_keys = label_keys or {"label"}
        self.resolution_map: Optional[Dict[str, np.ndarray]] = None
        if resolution_map:
            self.resolution_map = {
                k: np.asarray(v, dtype=np.float64) for k, v in resolution_map.items()
            }
        self.volume_key = volume_key

        self._do_zoom = False
        self._zoom: Optional[np.ndarray] = None

    def _resolve_native(self, data: Optional[Dict]) -> np.ndarray:
        """Look up per-volume native resolution, falling back to default."""
        if data is None or self.resolution_map is None:
            return self.native_resolution
        vol_name = data.get(self.volume_key, "")
        if not vol_name:
            return self.native_resolution
        best_prefix = ""
        for prefix in self.resolution_map:
            if vol_name.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
        if best_prefix:
            return self.resolution_map[best_prefix]
        return self.native_resolution

    def randomize(self, data=None) -> None:  # noqa: D102
        self._do_zoom = self.R.random() < self.prob
        if self._do_zoom:
            native = self._resolve_native(data)
            ref_lo, ref_hi = self.target_range[1]  # Y range
            ref_target = self.R.uniform(ref_lo, ref_hi)
            ratios = native / native[1]
            target = np.clip(
                ref_target * ratios,
                self.target_range[:, 0],
                self.target_range[:, 1],
            )
            self._zoom = native / target

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)
        if not self._do_zoom or self._zoom is None:
            return data

        d = dict(data)
        zoom = self._zoom.tolist()

        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)
            if not is_tensor:
                arr = torch.as_tensor(np.asarray(arr))

            mode = "nearest" if key in self.label_keys else "trilinear"
            result = _zoom_volume(arr, zoom, mode=mode)

            if is_tensor and hasattr(d[key], "meta"):
                from monai.data import MetaTensor
                result = MetaTensor(result, meta=d[key].meta, applied_operations=d[key].applied_operations)

            d[key] = result

        return d
