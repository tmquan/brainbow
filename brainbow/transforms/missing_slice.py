"""Missing-section (missing z-slice) augmentation for ssTEM volumes.

Serial-section EM (e.g. CREMI's ssTEM *Drosophila* volumes) frequently
contains **missing / damaged sections**: whole z-slices that are blank,
corrupted, or dropped during acquisition.  :class:`RandMissingSliced`
simulates this defect so the model learns to bridge affinities across a
ruined section instead of treating it as a hard boundary.

The transform operates on the **image only** -- the instance ``label``
(hence the affinity / sem targets) is left intact, so the network is
supervised to predict the *true* connectivity through the corrupted
section (the standard connectomics convention, cf. PyTorch-Connectomics
``MissingSection``).

Only meaningful for 3-D volumes (``[C, D, H, W]``); a no-op on 2-D
slice-mode tensors.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

_FILL_MODES = ("zero", "mean", "replicate")


class RandMissingSliced(MapTransform, Randomizable):
    """Randomly blank out whole z-sections of the image (CREMI-style defect).

    Args:
        keys: Keys to corrupt.  Defaults to ``["image"]`` -- the label is
            deliberately left untouched so the affinity / sem targets keep
            supervising the true connectivity through the missing section.
        prob: Probability of applying the augmentation to a sample (0-1).
        max_slices: When applied, drop ``k`` sections with
            ``k ~ Uniform{1, .., max_slices}``.
        fill: How to fill a dropped section:
            ``"zero"`` -- set to 0 (a black / blank section; matches the
            already-``[0,1]``-normalised image).  ``"mean"`` -- set to the
            patch mean (a flat grey section).  ``"replicate"`` -- copy the
            nearest surviving neighbour section (a duplicated section, the
            other common ssTEM defect).
        consecutive: If ``True`` the dropped sections form one contiguous
            run; if ``False`` (default) they are scattered independently.
        z_axis: Spatial axis index of the section (depth) axis, **excluding**
            the leading channel dim.  Default ``0`` (the ``D`` in
            ``[C, D, H, W]``).
    """

    def __init__(
        self,
        keys: KeysCollection = ("image",),
        prob: float = 0.0,
        max_slices: int = 2,
        fill: str = "zero",
        consecutive: bool = False,
        z_axis: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.prob = float(prob)
        self.max_slices = max(1, int(max_slices))
        if fill not in _FILL_MODES:
            raise ValueError(f"fill must be one of {_FILL_MODES}; got {fill!r}.")
        self.fill = fill
        self.consecutive = bool(consecutive)
        self.z_axis = int(z_axis)

        self._do_transform = False
        self._drop: List[int] = []

    def randomize(self, n_sections: int) -> None:
        self._do_transform = self.R.random() < self.prob
        if not self._do_transform or n_sections <= 0:
            self._do_transform = False
            return
        k = int(self.R.randint(1, self.max_slices + 1))
        k = min(k, n_sections)
        if self.consecutive:
            start = int(self.R.randint(0, n_sections - k + 1))
            self._drop = list(range(start, start + k))
        else:
            self._drop = sorted(
                int(i) for i in self.R.choice(n_sections, size=k, replace=False)
            )

    def _section_count(self, arr) -> int:
        # arr is channel-first; the section (depth) axis is z_axis + 1.
        # Only 3-D volumes [C, D, H, W] have a depth axis.
        if arr.ndim < 4:
            return 0
        return int(arr.shape[self.z_axis + 1])

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        # Drive randomness off the first present key so all keys share the
        # same dropped sections within a sample.
        first = next(iter(self.key_iterator(d)), None)
        if first is None:
            return d
        self.randomize(self._section_count(d[first]))
        if not self._do_transform or not self._drop:
            return d

        zdim = self.z_axis + 1  # account for the channel axis
        for key in self.key_iterator(d):
            arr = d[key]
            if (isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)) and arr.ndim < 4:
                continue  # 2-D slice mode: nothing to drop

            # Copy before mutating so we never corrupt a shared / cached source
            # array (CacheDataset reuses the same object across epochs).
            if isinstance(arr, torch.Tensor):
                arr = arr.clone()
            elif isinstance(arr, np.ndarray):
                arr = arr.copy()

            if self.fill == "mean":
                fill_val = float(arr.mean())
            else:
                fill_val = 0.0

            surviving = [i for i in range(arr.shape[zdim]) if i not in self._drop]
            for z in self._drop:
                idx = [slice(None)] * arr.ndim
                if self.fill == "replicate" and surviving:
                    src = min(surviving, key=lambda s: abs(s - z))
                    src_idx = [slice(None)] * arr.ndim
                    src_idx[zdim] = slice(src, src + 1)
                    idx[zdim] = slice(z, z + 1)
                    arr[tuple(idx)] = arr[tuple(src_idx)]
                else:
                    idx[zdim] = slice(z, z + 1)
                    arr[tuple(idx)] = fill_val
            d[key] = arr
        return d


__all__ = ["RandMissingSliced"]
