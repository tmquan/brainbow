"""
Training callbacks for connectomics segmentation.

Includes:
- ImageLogger: TensorBoard image/slice logger for epoch-end visualisation
"""

from brainbow.callbacks.memory import CudaEmptyCacheCallback
from brainbow.callbacks.tensorboard import ImageLogger

__all__ = [
    "CudaEmptyCacheCallback",
    "ImageLogger",
]
