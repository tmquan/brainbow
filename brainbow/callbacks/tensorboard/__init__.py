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
  ``"boundary"``, omitted for mode-level panels (``true/image``,
  ``true/label``)
* ``panel``  -- head-specific sub-tag (e.g. ``pred``, ``pred/pca``,
  ``pred/dir_centroid``, ``pred/raw``, ``true/t``)

The scalar logs emitted by
:class:`brainbow.modules.base.BaseCircuitModule` use the same
``{stage}/{mode}/{head}/...`` root for both training and validation,
so the per-head groups collapse identically in the Images and
Scalars tabs.

Module layout::

    tags.py           -- TagContext, HEADS (tag assembly; single source
                         of truth for the ``{stage}/{mode}/[{head}/]``
                         hierarchy)
    viz.py            -- low-level image utilities (central-slice,
                         per-image min-max normalise, HSV palette,
                         manifold projection of embeddings)
    geometry.py       -- matplotlib glyph (covariance) + quiver (dir)
                         renderers; isolated so the rest of the package
                         stays NumPy / PyTorch-only
    heads.py          -- per-head panel loggers and the
                         ``_log_predictions`` orchestrator
    image_logger.py   -- Lightning callback ``ImageLogger`` (cache first
                         batch of each epoch, forward under eval +
                         autocast, dispatch)

End-to-end flow (rank-0 only, once per ``every_n_epochs``)::

        ┌────────────────────────┐
        │ DataLoader batch       │  images, labels, optional
        │ (PL Trainer)           │  direction / covariance fields
        └──────────┬─────────────┘
                   ▼
        ┌────────────────────────┐   cache first batch of the epoch on
        │ ImageLogger            │   CPU and release when the epoch
        │ .on_*_batch_end        │   ends (see _detach_batch)
        └──────────┬─────────────┘
                   ▼
        ┌────────────────────────┐   move batch back to device, run a
        │ ImageLogger            │   single eval-mode forward under
        │ ._run_visualization    │   autocast, cast preds back to fp32
        └──────────┬─────────────┘
                   ▼
        ┌────────────────────────┐   (optional) rebuild the 16-channel
        │ build_boundary_target  │   boundary target for the ``true/*``
        │ (losses.boundary)      │   panels
        └──────────┬─────────────┘
                   ▼
        ┌────────────────────────┐   mode-level panels + four per-head
        │ heads._log_predictions │   loggers.  Tags are produced via
        │                        │   a single TagContext so the hier-
        │                        │   archy is enforced in one place
        └──────────┬─────────────┘
                   ▼
        ┌────────────────────────┐
        │ SummaryWriter          │
        │ .add_images(...)       │
        └────────────────────────┘

Works for both 2-D slices and 3-D volumes (takes a central slice
through ``brainbow.callbacks.tensorboard.viz._to_2d``).
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
