"""Loss package for the unified 32-channel Brainbow head.

The public loss surface is intentionally small:

* :class:`CombinedLoss` consumes the model's single ``[B, 32, *spatial]``
  head tensor and supervises raw, semantic foreground, binary skeleton
  mask, skeleton-relative direction, Voronoi-cell covariance,
  distance-to-skeleton scalar, average-centroid, embedding, and the
  two derived 12-channel affinity paths.
* :mod:`brainbow.losses._common` owns the canonical channel layout,
  the contiguous ``SIGMOID_SLICE`` over the (sem, skl) classification
  block, :func:`apply_head_activations`, the 12-direction affinity
  convention, field-slicing helpers, and shared numerical utilities.
"""

from brainbow.losses.combined import CombinedLoss, build_avg_target
from brainbow.losses._common import (
    AFF_CHANNELS,
    AFF_NAMES,
    AVG_SLICE,
    COV_SLICE,
    DIR_SLICE,
    EMB_SLICE,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    RAD_SLICE,
    RAW_SLICE,
    SEM_SLICE,
    SIGMOID_SLICE,
    SKL_SLICE,
    affinity_target,
    apply_head_activations,
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
    "SKL_SLICE",
    "DIR_SLICE",
    "COV_SLICE",
    "RAD_SLICE",
    "AVG_SLICE",
    "EMB_SLICE",
    "SIGMOID_SLICE",
    "AFF_CHANNELS",
    "AFF_NAMES",
    "slice_head",
    "apply_head_activations",
    "affinity_target",
    "build_avg_target",
    "soft_aff_from_field",
    "upper_tri_to_matrix",
]
