"""
PyTorch Lightning modules for connectomics segmentation training.

Why this package exists
-----------------------
A Lightning module is the **glue** between a backbone wrapper
(:mod:`brainbow.models`), the loss (:class:`brainbow.losses.AffinityFGLoss`),
and the Trainer.  Brainbow keeps the loop in one shared base
(:class:`BaseCircuitModule`) so that every architecture gets the same
training step, evaluation accumulation, scalar tag hierarchy and
clusterer wiring for free.

Architecture-specific concerns -- freeze schedules, optimiser param-
group splits, gradient sanitisation -- live in a per-family base class.
The Cosmos 2.5 family (Predict + Transfer) shares one
:class:`BaseCosmosModule` from
:mod:`brainbow.modules.cosmos_2_5_common`; concrete subclasses are
~20-line declarations of which ``model_cls`` and ``loss_cls`` to use.

Public surface
--------------
* :class:`BaseCircuitModule` -- shared training / eval loop.
* :class:`BaseCosmosModule` -- adds the freeze schedule and
  ``dit_backbone_lr`` / ``controlnet_lr`` parameter-group split.
* :class:`BaseVistaModule` -- adds Vista-specific wiring.
* :class:`Cosmos3Nano3DModule` -- concrete Lightning class for the
  Cosmos 3 (Nano) omni backbone (16B MoT, no ControlNet).  The default.
* :class:`CosmosPredict3DModule` -- concrete Lightning class for the
  Cosmos-Predict 2.5 backbone (base DiT, no ControlNet).
* :class:`CosmosTransfer3DModule` -- concrete Lightning class for the
  Cosmos-Transfer 2.5 backbone (base DiT + ControlNet).
* :class:`Vista3DModule` -- concrete Lightning class for the Vista
  backbone (3 heads, no boundary head).

Extending this module
---------------------
See ``doc/CONTRIBUTING.md`` "How to add a new model architecture".
"""

from brainbow.modules.base import BaseCircuitModule
from brainbow.modules.cosmos_2_5_common import BaseCosmosModule
from brainbow.modules.cosmos_3_nano import Cosmos3Nano3DModule
from brainbow.modules.cosmos_predict_2_5 import CosmosPredict3DModule
from brainbow.modules.cosmos_transfer_2_5 import CosmosTransfer3DModule
from brainbow.modules.vista import BaseVistaModule, Vista3DModule

__all__ = [
    "BaseCircuitModule",
    "BaseCosmosModule",
    "BaseVistaModule",
    "Cosmos3Nano3DModule",
    "CosmosPredict3DModule",
    "CosmosTransfer3DModule",
    "Vista3DModule",
]
