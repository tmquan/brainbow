"""Loss package for the unified 30-channel Brainbow head.

The public loss surface is intentionally small:

* :class:`CombinedLoss` consumes the model's single ``[B, 30, *spatial]``
  head tensor and supervises raw, semantic foreground, direction,
  covariance, average-centroid, embedding, and the two derived
  12-channel affinity paths.
* :mod:`brainbow.losses._common` owns the canonical channel layout,
  12-direction affinity convention, field-slicing helpers, and shared
  numerical utilities.
"""

from brainbow.losses.combined import CombinedLoss
from brainbow.losses._common import (
    AFF_CHANNELS,
    AFF_NAMES,
    AVG_SLICE,
    COV_SLICE,
    DIR_SLICE,
    EMB_SLICE,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    RAW_SLICE,
    SEM_SLICE,
    affinity_target,
    slice_head,
    soft_aff_from_field,
    upper_tri_to_matrix,
)

__all__ = [
    "CombinedLoss",
    "HEAD_CHANNELS",
    "HEAD_LAYOUT",
    "RAW_SLICE",
    "SEM_SLICE",
    "DIR_SLICE",
    "COV_SLICE",
    "AVG_SLICE",
    "EMB_SLICE",
    "AFF_CHANNELS",
    "AFF_NAMES",
    "slice_head",
    "affinity_target",
    "soft_aff_from_field",
    "upper_tri_to_matrix",
]
