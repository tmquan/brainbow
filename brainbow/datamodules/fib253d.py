"""
FIB-25 (3D) DataModule for PyTorch Lightning.

Thin subclass of :class:`MICRONSDataModule`: FIB-25 crops share the
MICRONS HDF5 layout (dataset key ``main``, ``[Z, Y, X]``), so only the
leaf :attr:`dataset_class` differs.  Lazy 3-D patch mode
(``slice_mode: false`` with ``patch_size``) reads patches on demand via
:class:`brainbow.datasets.LazyVolDataset`, exactly as for MICrONS.
"""

import logging

from brainbow.datamodules.microns import MICRONSDataModule
from brainbow.datasets import FIB253DDataset

logger = logging.getLogger(__name__)


class FIB253DDataModule(MICRONSDataModule):
    """PyTorch Lightning DataModule for the FIB-25 (3D) dataset.

    Inherits the full MICrONS datamodule behaviour (eager slice mode /
    lazy 3-D patch mode, the shared MONAI augmentation pipeline, and the
    dataloader hooks); only the underlying :attr:`dataset_class` is
    swapped to :class:`FIB253DDataset`.
    """

    dataset_class = FIB253DDataset


__all__ = ["FIB253DDataModule"]
