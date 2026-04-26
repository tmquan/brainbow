"""
PyTorch Lightning callbacks bundled with brainbow.

Why this package exists
-----------------------
The training loop should stay free of visualisation, memory tracking
and other side-effects; we collect them here as Lightning callbacks
that the training script wires in declaratively from the Hydra config.

Public surface
--------------
* :class:`CudaEmptyCacheCallback` -- empties the PyTorch CUDA caching
  allocator around validation epochs, which is the easiest knob against
  slow-growing reserved memory on multi-day runs.
* :class:`CudaMemoryLoggerCallback` -- logs allocated / reserved /
  fragmentation gauges to TensorBoard under ``cuda_memory/*`` so you
  can tell a leak from a fragmentation drift at a glance.
* :class:`ImageLogger` -- the heavy-lifter.  Renders per-head panels at
  every epoch end into the canonical ``{stage}/{mode}/[{head}/]{panel}``
  TB tag hierarchy (see :mod:`brainbow.callbacks.tensorboard.tags`).

Extending this module
---------------------
Add a new ``brainbow/callbacks/<name>.py`` defining a
:class:`pytorch_lightning.callbacks.Callback` subclass and re-export
it here.  Wire it into the training script's ``setup_callbacks``.
"""

from brainbow.callbacks.memory import (
    CudaEmptyCacheCallback,
    CudaMemoryLoggerCallback,
)
from brainbow.callbacks.tensorboard import ImageLogger

__all__ = [
    "CudaEmptyCacheCallback",
    "CudaMemoryLoggerCallback",
    "ImageLogger",
]
