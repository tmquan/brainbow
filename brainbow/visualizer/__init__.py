"""
Web-based volume visualizer with a neuroglancer-style 4-panel layout.

Why this package exists
-----------------------
TensorBoard images are great for **per-step debugging** but a full
volume in 3-D is much easier to skim in an interactive viewer.  The
visualizer is a small FastAPI server that streams chunks of an HDF5 /
NRRD / TIFF / NIfTI volume into a WebGL raymarcher in the browser.  Use
it to QA datasets and predictions; do not depend on it from training.

Public surface
--------------
* :class:`VolumeData` -- tiny container for a loaded volume + spacing.
* :func:`load_volume` -- format-agnostic loader (HDF5 / NRRD / TIFF /
  NIfTI) backed by :mod:`brainbow.preprocessors`.

The FastAPI app and CLI live in :mod:`brainbow.visualizer.app` and
:mod:`brainbow.visualizer.__main__` respectively; launch with::

    python -m brainbow.visualizer --raw vol.h5 --seg seg.h5

Extending this module
---------------------
To support a new format, add the corresponding
:class:`brainbow.preprocessors.BasePreprocessor` subclass; the visualizer
will pick it up via :func:`load_volume` automatically.
"""

from brainbow.visualizer.volume_loader import VolumeData, load_volume

__all__ = ["VolumeData", "load_volume"]
