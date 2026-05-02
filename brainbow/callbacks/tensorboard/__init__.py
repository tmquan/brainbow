"""
TensorBoard image logger package.

All tags emitted by this subpackage follow a single hierarchy so that
images and scalars cluster together in the TensorBoard UI::

    {stage}/{mode}/{panel}

where

* ``stage``  -- ``"train"`` | ``"val"``
* ``mode``   -- ``"automatic"`` (single mode today; structured so
  additional modes such as ``"prompted"`` can slot in later)
* ``panel``  -- e.g. ``true/image``, ``true/avg/val``,
  ``true/wan_decoder``, ``pred/raw``, ``pred/sem``, ``pred/avg/val``,
  ``pred/avg/aff/01_t1``, ``pred/emb/_pca``, ``pred/label/{pre,mul}``.  Affinity panels are prefixed with their 1-based
  position in :data:`brainbow.losses.DIRECTIONS` so the alphabetical
  TB sort pairs each axis (T/B, U/D, L/R) on consecutive panels.

The scalar logs emitted by
:class:`brainbow.modules.base.BaseCircuitModule` use the same
``{stage}/{mode}/loss/...`` root for both training and validation, so
each predicted field's images sit near its loss scalars.

Module layout::

    tags.py           -- TagContext (tag assembly; single source of
                         truth for the ``{stage}/{mode}/`` hierarchy)
    viz.py            -- low-level image utilities (central-slice,
                         per-image min-max normalise, HSV palette,
                         manifold projection of embeddings)
    geometry.py       -- vectorised optical-flow-style HSV overlays for
                         the predicted ``dir`` (Middlebury hue/value of
                         the centroid-direction vector) and ``cov``
                         (principal-eigenvector orientation with
                         anisotropy → saturation) panels, mask-composited
                         on top of the raw EM image (foreground = pure
                         flow colour, background = raw EM)
    heads.py          -- unified-head panel logger
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
        ┌────────────────────────┐   unified 30-channel head panels
        │ heads._log_predictions │   (raw, sem, dir, cov, avg, emb,
        │                        │   derived 12-aff panels)
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
