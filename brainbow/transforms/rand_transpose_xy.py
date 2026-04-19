"""Random Y ↔ X transpose augmentation for volumetric data.

Combined with axis flips and 90° rotations this completes the full
dihedral-8 symmetry group of the XY plane, ensuring the model sees
every possible in-plane orientation of the tissue.
"""

from typing import Dict

import numpy as np
import torch
from einops import rearrange
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


class RandTransposeXYd(MapTransform, Randomizable):
    """Randomly swap the last two spatial axes (Y ↔ X).

    Works for both 2-D ``(C, H, W)`` and 3-D ``(C, D, H, W)`` tensors.

    Args:
        keys: Keys to apply the transpose to (e.g. ``["image", "label"]``).
        prob: Probability of applying the transpose (0–1).
    """

    def __init__(self, keys: KeysCollection, prob: float = 0.5) -> None:
        super().__init__(keys)
        self.prob = prob
        self._do_transform = False

    def randomize(self, data=None) -> None:
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Dict) -> Dict:
        self.randomize()
        if not self._do_transform:
            return data
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            if isinstance(val, torch.Tensor):
                d[key] = rearrange(val, "... h w -> ... w h").contiguous()
            elif isinstance(val, np.ndarray):
                d[key] = np.ascontiguousarray(np.swapaxes(val, -2, -1))
        return d
