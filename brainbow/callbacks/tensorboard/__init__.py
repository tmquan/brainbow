"""
TensorBoard image logger package.

All tags emitted by this subpackage follow a single, head-oriented
hierarchy so that images and scalars cluster together in the
TensorBoard UI::

    {stage}/{mode}/[{head}/]{panel}

where

* ``stage``  -- ``"train"`` | ``"val"``
* ``mode``   -- ``"automatic"`` (single mode today; structured so
  additional modes such as ``"prompted"`` can slot in later)
* ``head``   -- ``"semantic"`` | ``"instance"`` | ``"geometry"`` |
  ``"brainbow"``, omitted for mode-level panels (``image``, ``label``)
* ``panel``  -- head-specific sub-tag (e.g. ``prediction``, ``pca``,
  ``dir_centroid``, ``pred/raw``)

The scalar logs emitted by
:class:`brainbow.modules.base.BaseCircuitModule` use the same
``{stage}/{mode}/{head}/...`` root for both training and validation,
so the per-head groups collapse identically in the Images and
Scalars tabs.

Module layout::

    tags.py           -- TagContext, HEADS
    viz.py            -- low-level image utilities (slicing, colour, projection)
    geometry.py       -- matplotlib glyph + quiver renderers
    heads.py          -- per-head panel loggers + _log_predictions orchestrator
    image_logger.py   -- Lightning callback (ImageLogger)

Works for both 2-D slices and 3-D volumes (takes a central slice).
"""

from brainbow.callbacks.tensorboard.heads import _log_predictions
from brainbow.callbacks.tensorboard.image_logger import ImageLogger
from brainbow.callbacks.tensorboard.tags import HEADS, TagContext

__all__ = [
    "HEADS",
    "ImageLogger",
    "TagContext",
    "_log_predictions",
]
