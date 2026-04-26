"""
Format-agnostic preprocessor classes for connectomics volumes.

Why this package exists
-----------------------
Connectomics raw and segmentation volumes ship in several incompatible
formats (HDF5, NRRD, multi-page TIFF, NIfTI).  Each preprocessor here
isolates the format-specific I/O so the rest of brainbow can speak in
plain :class:`numpy.ndarray` and :class:`torch.Tensor`.

Required overrides for new preprocessors
----------------------------------------
* :attr:`BasePreprocessor.supported_extensions` -- iterable of suffixes
  this class can read (e.g. ``(".h5", ".hdf5")``).
* :meth:`BasePreprocessor.load` -- read a file into a numpy array.
* :meth:`BasePreprocessor.validate` -- sniff a file for compatibility.

Optional overrides
------------------
* :meth:`save` -- writing back is optional; the base class raises
  :class:`NotImplementedError` if a leaf doesn't override it (e.g. read-
  only formats are fine).
* :meth:`get_metadata`, :meth:`get_shape` -- the base implementations
  read the full file; override for a cheap header-only path on large
  volumes.

Extending this module
---------------------
Add a new ``<format>.py`` module that subclasses :class:`BasePreprocessor`,
register it in ``__init__.py``, and update ``utils/io.py`` if you want
:func:`brainbow.utils.load_volume` to dispatch to it by suffix.
"""

from brainbow.preprocessors.base import BasePreprocessor
from brainbow.preprocessors.hdf5 import HDF5Preprocessor
from brainbow.preprocessors.tiff import TIFFPreprocessor
from brainbow.preprocessors.nrrd import NRRDPreprocessor
from brainbow.preprocessors.nfty import NFTYPreprocessor

__all__ = [
    "BasePreprocessor",
    "HDF5Preprocessor",
    "TIFFPreprocessor",
    "NRRDPreprocessor",
    "NFTYPreprocessor",
]
