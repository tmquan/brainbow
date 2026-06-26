"""
CREMI (3D) DataModule for PyTorch Lightning.

Thin subclass of :class:`MICRONSDataModule`: the converted CREMI crops
share the MICRONS HDF5 layout (dataset key ``main``, ``[Z, Y, X]``), so
only the leaf :attr:`dataset_class` differs.  Lazy 3-D patch mode
(``slice_mode: false`` with ``patch_size``) reads patches on demand via
:class:`brainbow.datasets.LazyVolDataset`.
"""

import logging

from brainbow.datamodules.microns import MICRONSDataModule
from brainbow.datasets import CREMI3DDataset

logger = logging.getLogger(__name__)


class CREMI3DDataModule(MICRONSDataModule):
    """PyTorch Lightning DataModule for the CREMI (3D) dataset.

    Inherits the full MICrONS datamodule behaviour (eager slice mode /
    lazy 3-D patch mode, shared MONAI augmentation pipeline, dataloader
    hooks); only the underlying :attr:`dataset_class` is swapped to
    :class:`CREMI3DDataset`.
    """

    dataset_class = CREMI3DDataset


__all__ = ["CREMI3DDataModule"]
