"""Loss package for the affinity + foreground Mutex Watershed head.

The public loss surface is intentionally small:

* :class:`AffinityFGLoss` consumes the model's single
  ``[B, HEAD_CHANNELS, *spatial]`` head tensor and supervises the
  per-offset affinities and the scalar foreground probability.  At
  evaluation the affinities are agglomerated into instances by the
  Mutex Watershed (:mod:`brainbow.inference.mutex_watershed`).
* :class:`DiceBCEFocalLoss` is the composite Dice + BCE + Focal
  supervisor used by :class:`AffinityFGLoss` for the foreground head --
  exposed here so external consumers can instantiate it directly with
  the same numerics.
* :mod:`brainbow.losses._common` owns the canonical channel layout
  (:data:`AFFINITY_OFFSETS`, :data:`AFF_SLICE`, :data:`FG_SLICE`),
  :func:`apply_head_activations`, the affinity-target / validity-mask
  builders, field-slicing helpers, and shared numerical utilities
  (including :func:`stable_bce_on_probs`).
"""

from brainbow.losses.affinity import AffinityFGLoss
from brainbow.losses.dice_bce_focal import DiceBCEFocalLoss
from brainbow.losses._common import (
    AFF_CHANNELS,
    AFF_NAMES,
    AFF_SLICE,
    AFFINITY_OFFSETS,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    N_AFF,
    N_PULL,
    RAW_SLICE,
    SEM_SLICE,
    SIGMOID_SLICE,
    affinity_target_from_offsets,
    affinity_validity_mask,
    apply_head_activations,
    shift_nd,
    shift_replicate,
    slice_head,
    stable_bce_on_probs,
)

__all__ = [
    "AffinityFGLoss",
    "DiceBCEFocalLoss",
    "HEAD_CHANNELS",
    "HEAD_LAYOUT",
    "AFFINITY_OFFSETS",
    "AFF_SLICE",
    "SEM_SLICE",
    "RAW_SLICE",
    "SIGMOID_SLICE",
    "N_AFF",
    "N_PULL",
    "AFF_CHANNELS",
    "AFF_NAMES",
    "slice_head",
    "apply_head_activations",
    "affinity_target_from_offsets",
    "affinity_validity_mask",
    "shift_nd",
    "shift_replicate",
    "stable_bce_on_probs",
]
