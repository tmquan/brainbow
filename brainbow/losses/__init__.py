"""
Loss functions for connectomics segmentation.

Why this package exists
-----------------------
A "head" in brainbow is a per-voxel prediction tensor; a "loss" is the
function that scores one head against a deterministic supervision
target built directly from the labels.  Every loss here follows the
same skeleton (see :mod:`brainbow.losses._common`) so they are
pluggable into :class:`CombinedLoss` without bespoke glue.

Public surface
--------------
Standalone task losses
    * :class:`SemanticLoss`  -- CE + IoU + Dice on a per-voxel class map.
    * :class:`InstanceLoss`  -- pull / push / norm discriminative loss
      on a per-voxel embedding.
    * :class:`GeometryLoss`  -- raw / cov-upper-tri / dir regression
      (10 channels in 3-D, 5 in 2-D).
    * :class:`BoundaryLoss`  -- 10-channel head: raw(1) + avg RGB(3) +
      6-channel face-affinity, plus a derived soft 6-aff computed from
      the predicted avgloc for dual supervision.  (Formerly
      ``BrainbowLoss``.)
    * :func:`build_boundary_target` -- standalone target builder for
      :class:`BoundaryLoss`, used by :class:`ImageLogger` for previews.

Combined loss
    * :class:`CombinedLoss` -- weighted sum used by every Lightning
      module.  Heads with weight ``0.0`` are not instantiated.

Extending this package
----------------------
See ``doc/CONTRIBUTING.md`` "How to add a new task loss" for the recipe.
The loss skeleton + naming convention is documented in
:mod:`brainbow.losses._common`.
"""

from brainbow.losses.semantic import SemanticLoss
from brainbow.losses.instance import InstanceLoss
from brainbow.losses.geometry import GeometryLoss
from brainbow.losses.boundary import BoundaryLoss, build_boundary_target
from brainbow.losses.combined import CombinedLoss

__all__ = [
    "SemanticLoss",
    "InstanceLoss",
    "GeometryLoss",
    "BoundaryLoss",
    "build_boundary_target",
    "CombinedLoss",
]
