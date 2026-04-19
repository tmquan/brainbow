"""
Loss functions for connectomics segmentation.

Standalone task losses:
- :class:`SemanticLoss`  -- CE + IoU + Dice
- :class:`InstanceLoss`  -- pull / push / norm
- :class:`GeometryLoss`  -- dir / cov / raw
- :class:`BrainbowLoss`  -- 9-channel per-instance (min, avg, max)loc colouring
                           + 1-channel raw image reconstruction

Combined multi-head loss used by every Lightning module:
- :class:`CombinedLoss`  -- weighted sum of the task losses
"""

from brainbow.losses.semantic import SemanticLoss
from brainbow.losses.instance import InstanceLoss
from brainbow.losses.geometry import GeometryLoss
from brainbow.losses.brainbow import BrainbowLoss, build_brainbow_target
from brainbow.losses.combined import CombinedLoss

__all__ = [
    "SemanticLoss",
    "InstanceLoss",
    "GeometryLoss",
    "BrainbowLoss",
    "build_brainbow_target",
    "CombinedLoss",
]
