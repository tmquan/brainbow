"""Foreground-ratio-aware random spatial cropping.

Wraps ``RandSpatialCropd`` with rejection sampling so that every returned
patch meets a minimum foreground fraction.  Repeatedly draws random crops
and keeps the first one where the fraction of foreground voxels (label > 0)
meets the threshold.  Falls back to the best crop seen after *max_attempts*.
"""

from typing import Dict, Sequence, Union

from monai.config import KeysCollection
from monai.transforms import MapTransform, RandSpatialCropd


class RandSpatialCropForegroundd(MapTransform):
    """``RandSpatialCropd`` with rejection sampling on foreground ratio.

    Args:
        keys: Keys of the corresponding items to crop.
        spatial_size: Output patch size ``(D, H, W)`` or ``(H, W)``.
        label_key: Key whose values are used for foreground detection.
        min_foreground: Minimum fraction of foreground (label > 0) voxels
            required in the crop.  Patches below this are re-drawn.
        max_attempts: Maximum rejection-sampling iterations before
            falling back to the best crop seen.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        label_key: str = "label",
        min_foreground: float = 0.5,
        max_attempts: int = 50,
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.min_foreground = min_foreground
        self.max_attempts = max_attempts
        self._cropper = RandSpatialCropd(
            keys=keys, roi_size=spatial_size, random_size=False,
        )

    def __call__(self, data: Dict) -> Dict:
        best, best_ratio = None, -1.0
        for _ in range(self.max_attempts):
            cropped = self._cropper(dict(data))
            label = cropped[self.label_key]
            fg_ratio = float((label > 0).float().mean())
            if fg_ratio >= self.min_foreground:
                return cropped
            if fg_ratio > best_ratio:
                best, best_ratio = cropped, fg_ratio
        return best  # type: ignore[return-value]
