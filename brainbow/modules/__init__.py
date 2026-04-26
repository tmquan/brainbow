"""
PyTorch Lightning modules for connectomics segmentation training.

Why this package exists
-----------------------
A Lightning module is the **glue** between a backbone wrapper
(:mod:`brainbow.models`), the loss (:class:`brainbow.losses.CombinedLoss`),
and the Trainer.  Brainbow keeps the loop in one shared base
(:class:`BaseCircuitModule`) so that every architecture gets the same
training step, evaluation accumulation, scalar tag hierarchy and
clusterer wiring for free.

Architecture-specific concerns -- freeze schedules, optimiser param-
group splits, gradient sanitisation -- live in a per-backbone
``base.py`` (:class:`BaseCosmosModule`, :class:`BaseVistaModule`); the
concrete ``module.py`` is then a ~20-line declaration of which
``model_cls`` and ``loss_cls`` to use.

Public surface
--------------
* :class:`BaseCircuitModule` -- shared training / eval loop.
* :class:`BaseCosmosModule` -- adds the freeze schedule and
  ``dit_backbone_lr`` parameter-group split.
* :class:`BaseVistaModule` -- adds Vista-specific wiring.
* :class:`CosmosTransfer3DModule` -- concrete Lightning class for the
  Cosmos-Transfer 2.5 backbone (4 heads).
* :class:`Vista3DModule` -- concrete Lightning class for the Vista
  backbone (3 heads, no boundary head).

Extending this module
---------------------
See ``doc/CONTRIBUTING.md`` "How to add a new model architecture".
"""

from brainbow.modules.base import BaseCircuitModule
from brainbow.modules.cosmos_transfer_2_5 import (
    BaseCosmosModule,
    CosmosTransfer3DModule,
)
from brainbow.modules.vista import BaseVistaModule, Vista3DModule

__all__ = [
    "BaseCircuitModule",
    "BaseVistaModule",
    "BaseCosmosModule",
    "Vista3DModule",
    "CosmosTransfer3DModule",
]
