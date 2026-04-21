"""
Loss functions for connectomics segmentation.

Standalone task losses:
- :class:`SemanticLoss`  -- CE + IoU + Dice
- :class:`InstanceLoss`  -- pull / push / norm
- :class:`GeometryLoss`  -- raw / cov / dir (10 channels in 3-D)
- :class:`BoundaryLoss`  -- 16-channel head: raw(1) + min/avg/max RGB(9)
                           + 6-channel face-affinity (aff, SAME-pad).
                           (Formerly ``BrainbowLoss``.)

Combined multi-head loss used by every Lightning module:
- :class:`CombinedLoss`  -- weighted sum of the task losses
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
