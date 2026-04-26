"""
PyTorch Lightning DataModules for connectomics datasets.

Why this package exists
-----------------------
A datamodule sits between the raw dataset (one of
:mod:`brainbow.datasets`) and the Lightning ``Trainer``: it owns the
**MONAI augmentation pipeline**, the **patch / slice sampling
strategy**, and the **DataLoader** configuration (batch size, workers,
``forkserver`` multiprocessing context).  All architectures share the
same datamodule contract -- swapping the backbone does not require a
new datamodule.

Public surface
--------------
* :class:`CircuitDataModule` -- shared base.  Implements
  ``get_train_transforms`` / ``get_val_transforms`` / ``setup`` and the
  three dataloader hooks; subclasses only declare ``dataset_class`` and
  any per-leaf overrides.
* :class:`SNEMI3DDataModule`, :class:`MICRONSDataModule`,
  :class:`NeuronsDataModule` -- one leaf per dataset.

Extending this module
---------------------
A new datamodule is typically a 50-line subclass declaring its
``dataset_class``.  See ``doc/CONTRIBUTING.md`` "How to add a new
dataset" for the full recipe.
"""

from brainbow.datamodules.base import CircuitDataModule
from brainbow.datamodules.snemi3d import SNEMI3DDataModule
from brainbow.datamodules.microns import MICRONSDataModule
from brainbow.datamodules.neurons import NeuronsDataModule

__all__ = [
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "MICRONSDataModule",
    "NeuronsDataModule",
]
