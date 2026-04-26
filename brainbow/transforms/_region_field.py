"""
Shared base for per-pixel region-field transforms.

Why this file exists
--------------------
:class:`brainbow.transforms.Directiond` and
:class:`brainbow.transforms.Covarianced` both compute a *per-pixel
field derived from the instance label map* (centroid-pointing vectors
and 2nd-order moments, respectively).  Both used to carry an identical
``__call__`` body whose only job was MetaTensor housekeeping (extract
device, strip leading non-spatial dims, run the numpy compute, write
the result back as the same tensor / array type that came in).  This
module is the single shared implementation; concrete subclasses only
declare the output suffix and the numpy-domain compute function.

Public surface
--------------
* :class:`RegionFieldTransformd`

Extending this module
---------------------
Subclass :class:`RegionFieldTransformd` and override

* ``output_suffix`` — appended to each key (e.g. ``"_direction"``);
* ``_compute(label_np: np.ndarray) -> np.ndarray`` — the field built
  from a single ``[*spatial]`` instance-label array.

The base class handles everything else: tensor↔numpy coercion, device
restoration, leading-channel stripping, and writing the result under
``{key}{output_suffix}``.
"""

from typing import Dict

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform

__all__ = ["RegionFieldTransformd"]


class RegionFieldTransformd(MapTransform):
    """Base class for per-pixel region-derived field transforms.

    Subclasses implement :meth:`_compute` (numpy in, numpy out) and set
    :attr:`output_suffix`.  Tensor / device / leading-channel handling
    is shared.

    Args:
        keys: Keys of instance label maps.
        spatial_dims: Number of spatial dimensions (2 or 3).  Leading
            dims past this count (e.g. a leading channel from
            ``EnsureChannelFirstd``) are stripped before
            :meth:`_compute` runs.
    """

    output_suffix: str = ""

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims

    def _compute(self, label_np: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, data: Dict) -> Dict:
        if not self.output_suffix:
            raise RuntimeError(
                f"{type(self).__name__} must set a non-empty `output_suffix`."
            )
        d = dict(data)

        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)

            if is_tensor:
                device = arr.device
                label_np = arr.cpu().numpy()
            else:
                device = None
                label_np = np.asarray(arr)

            # Strip leading non-spatial dims (channel) to get [*spatial].
            while label_np.ndim > self.spatial_dims:
                label_np = label_np[0]

            field = self._compute(label_np)

            if is_tensor:
                field = torch.from_numpy(field).to(device)

            d[f"{key}{self.output_suffix}"] = field

        return d
