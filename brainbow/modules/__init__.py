"""
PyTorch Lightning modules for connectomics segmentation training.

All modules inherit from :class:`BaseCircuitModule`, which owns the
shared training / evaluation loop and scalar logging hierarchy.
Per-backbone specialisations (freeze schedules, optimiser param groups)
live in :class:`BaseVistaModule` and :class:`BaseCosmosModule`.
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
