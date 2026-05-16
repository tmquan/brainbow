"""
Inference utilities for connectomics segmentation.

This subpackage holds the **post-training** path: turn a trained model
plus a (possibly very large) volume into a discrete instance-id map.

Public surface
--------------
- :func:`brainbow.inference.sliding_window.sliding_window_inference`
  -- patch-wise blended inference over volumes that don't fit on the GPU.
- :func:`brainbow.inference.clusterer.build_clusterer`
  -- factory for the per-instance clustering step (soft mean-shift,
  HDBSCAN, spatial connected-components).
- :class:`brainbow.inference.clusterer.SoftMeanShift` and the other
  ``*Clusterer`` classes that ``build_clusterer`` returns.

Extending this module: a new clusterer should subclass
:class:`brainbow.inference.clusterer._BaseUnsupervisedClusterer` and
register itself in :func:`build_clusterer`'s ``_CLUSTERER_REGISTRY``.
Sliding-window aggregation operates on the unified 32-channel head
tensor; see :mod:`sliding_window` for the gaussian-blended patch
fusion logic.
"""

from brainbow.inference.clusterer import (
    HDBSCANClusterer,
    SoftMeanShift,
    SpatialCCClusterer,
    build_clusterer,
)
from brainbow.inference.sliding_window import sliding_window_inference

__all__ = [
    "HDBSCANClusterer",
    "SoftMeanShift",
    "SpatialCCClusterer",
    "build_clusterer",
    "sliding_window_inference",
]
