"""
Domain-specific MONAI dictionary transforms for connectomics / EM data.

Why this package exists
-----------------------
The MONAI augmentation pipeline can be assembled from MONAI's
built-ins (RandFlip, Rand3DElastic, ...) for ~90% of connectomics
needs, but a few connectomics-specific operations are not in MONAI:

* Crop hygiene (re-label connected components after a random crop).
* Boundary-map construction at load time.
* Foreground-biased crop sampling for sparse instance volumes.
* Resolution-zoom augmentation that harmonises pixel sizes across
  datasets in a multi-dataset run.
* Random Y<->X transpose, completing the dihedral-8 symmetry group of
  the XY plane (combined with flips + 90 deg rotations).

All transforms here are MONAI ``MapTransform`` (dict-in / dict-out)
wrappers so they slot into the same :class:`monai.transforms.Compose`
pipelines as the standard MONAI ones.

Public surface
--------------
* :class:`Labeld`                       -- CC-relabel after random crop.
* :class:`FindBoundariesd`              -- zero out instance boundaries.
* :class:`RandSpatialCropForegroundd`   -- foreground-biased crop.
* :class:`RandTransposeXYd`             -- random Y<->X transpose.
* :class:`RandResolutionZoomd`          -- random resolution zoom.

Elastic deformation uses MONAI's :class:`Rand3DElasticd` directly --
it's configured by the datamodule and not re-exported here.

Extending this module
---------------------
A new transform should be a :class:`MapTransform` subclass placed in
its own ``transforms/<name>.py`` file and re-exported here.  The
datamodule's pipeline assembly lives in
:meth:`brainbow.datamodules.base.CircuitDataModule.get_train_transforms`.
"""

from brainbow.transforms.label import Labeld
from brainbow.transforms.find_boundaries import FindBoundariesd
from brainbow.transforms.rand_crop_foreground import RandSpatialCropForegroundd
from brainbow.transforms.rand_transpose_xy import RandTransposeXYd
from brainbow.transforms.resolution_zoom import RandResolutionZoomd

__all__ = [
    "Labeld",
    "FindBoundariesd",
    "RandSpatialCropForegroundd",
    "RandTransposeXYd",
    "RandResolutionZoomd",
]
