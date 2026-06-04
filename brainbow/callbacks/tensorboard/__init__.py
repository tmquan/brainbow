"""
TensorBoard image logger package.

All tags emitted by this subpackage follow a single hierarchy so that
images and scalars cluster together in the TensorBoard UI::

    {stage}/{mode}/{panel}

where

* ``stage``  -- ``"train"`` | ``"val"``
* ``mode``   -- ``"automatic"`` (single mode today; structured so
  additional modes such as ``"prompted"`` can slot in later)
* ``panel``  -- e.g. ``true/image``, ``true/label``, ``true/aff/*``,
  ``true/wan_decoder``, ``pred/sem``, ``pred/raw``, ``pred/aff/*``,
  ``pred/label/{pre,mul}`` (the Mutex Watershed instances).  Affinity
  panels are named by their offset (``brainbow.losses.AFF_NAMES``, e.g.
  ``01_pull_z1`` / ``04_push_y3``) with a 1-based numeric prefix so the
  alphabetical TB sort keeps them in offset order.

The scalar logs emitted by
:class:`brainbow.modules.base.BaseCircuitModule` use the same
``{stage}/{mode}/loss/...`` root for both training and validation, so
each predicted field's images sit near its loss scalars.

Module layout::

    tags.py           -- TagContext (tag assembly; single source of
                         truth for the ``{stage}/{mode}/`` hierarchy)
    viz.py            -- low-level image utilities (central-slice,
                         per-image min-max normalise, label HSV palette)
    heads.py          -- affinity + sem + raw panel logger (incl. the
                         Mutex Watershed ``pred/label`` instances)
    image_logger.py   -- Lightning callback ``ImageLogger`` (cache first
                         batch of each epoch, forward under eval +
                         autocast, dispatch)

    (``geometry.py`` -- legacy ``pred/dir`` / ``pred/cov`` overlays for
    the retired skeleton-geometry head; no longer wired in.)

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
        │ ImageLogger            │   per-image eval-mode forward under
        │ ._run_visualization    │   autocast, cast preds back to fp32
        └──────────┬─────────────┘
                   ▼
        ┌────────────────────────┐   affinity + sem + raw head panels
        │ heads._log_predictions │   (pred/sem, pred/raw, pred/aff/*,
        │                        │   Mutex Watershed pred/label/*)
        └──────────┬─────────────┘
                   ▼
        ┌────────────────────────┐
        │ SummaryWriter          │
        │ .add_images(...)       │
        └────────────────────────┘

Works for both 2-D slices and 3-D volumes (takes a central slice
through ``brainbow.callbacks.tensorboard.viz._to_2d``).
"""

from brainbow.callbacks.tensorboard.image_logger import ImageLogger

__all__ = ["ImageLogger"]
