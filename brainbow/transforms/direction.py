"""Direction field transform: per-pixel vectors toward instance centroids.

For each foreground instance in a label map, computes the vector from
every pixel to the instance's centroid and (optionally) normalises to
unit length.  Background pixels receive zero vectors.

The centroid computation is GPU-accelerated via cucim/cupy when
available (see :mod:`brainbow.transforms.edt`).
"""

from typing import Dict

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform

from brainbow.transforms.edt import centroid as _centroid


def compute_direction_field(
    label: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute per-pixel direction vectors pointing toward instance centroids.

    Args:
        label: Instance label array ``[*spatial]``.  Background is 0.
        normalize: Normalise vectors to unit length.

    Returns:
        Direction field ``[S, *spatial]`` where ``S = label.ndim``.
        Channels correspond to spatial axes in the same order as the
        input dimensions (e.g. ``(z, y, x)`` for 3-D).
    """
    label_np = np.asarray(label, dtype=np.int64)
    S = label_np.ndim
    shape = label_np.shape
    direction = np.zeros((S,) + shape, dtype=np.float32)

    uids = np.unique(label_np)
    uids = uids[uids > 0]

    if len(uids) == 0:
        return direction

    centroids = _centroid(label_np, uids)
    if not isinstance(centroids, list):
        centroids = [centroids]

    coords = np.meshgrid(
        *[np.arange(s, dtype=np.float32) for s in shape],
        indexing="ij",
    )

    for uid, centroid in zip(uids, centroids):
        mask = label_np == uid
        for d in range(S):
            direction[d][mask] = centroid[d] - coords[d][mask]

    if normalize:
        mag = np.sqrt(np.sum(direction ** 2, axis=0, keepdims=True))
        np.clip(mag, 1e-8, None, out=mag)
        direction /= mag

    return direction


class Directiond(MapTransform):
    """Compute per-pixel direction field toward instance centroids.

    Reads instance labels from each key and stores the direction field
    under ``{key}_direction``.  Input labels are expected in
    ``[C, *spatial]`` format (post ``EnsureChannelFirstd``); the first
    channel is used.

    Args:
        keys: Keys of instance label maps.
        spatial_dims: Number of spatial dimensions (2 or 3).
        normalize: Normalise direction vectors to unit length.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
        normalize: bool = True,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims
        self.normalize = normalize

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)

            if is_tensor:
                device = arr.device
                label_np = arr.cpu().numpy()
            else:
                label_np = np.asarray(arr)

            # Strip leading non-spatial dims (channel) to get [*spatial]
            while label_np.ndim > self.spatial_dims:
                label_np = label_np[0]

            direction = compute_direction_field(label_np, normalize=self.normalize)

            if is_tensor:
                direction = torch.from_numpy(direction).to(device)

            d[f"{key}_direction"] = direction

        return d
