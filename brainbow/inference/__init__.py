"""
Inference utilities for connectomics segmentation.

This subpackage holds the **post-training** path: turn a trained model
plus a (possibly very large) volume into a discrete instance-id map.

Public surface
--------------
- :func:`brainbow.inference.sliding_window.sliding_window_inference`
  -- patch-wise blended inference over volumes that don't fit on the GPU.
- :class:`brainbow.inference.mutex_watershed.MutexWatershed` -- the
  parameter-free agglomeration that turns predicted affinities into
  instance ids (the production eval / inference path; see
  :doc:`MUTEXWATERSHED`).  It dispatches per-input between
  :func:`~brainbow.inference.mutex_watershed.mws_cp` (GPU, cupy Boruvka,
  default for CUDA inputs) and
  :func:`~brainbow.inference.mutex_watershed.mws_np` (CPU, numpy/numba,
  exact reference + fallback).

Sliding-window aggregation operates on the affinity + sem + raw head
tensor (``C = HEAD_CHANNELS``); see :mod:`sliding_window` for the
gaussian-blended patch fusion logic.
"""

from brainbow.inference.mutex_watershed import (
    MutexWatershed,
    mutex_watershed,
    mws_cp,
    mws_np,
)
from brainbow.inference.sliding_window import sliding_window_inference

__all__ = [
    "MutexWatershed",
    "mutex_watershed",
    "mws_np",
    "mws_cp",
    "sliding_window_inference",
]
