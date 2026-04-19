"""
Model architectures for connectomics segmentation.

All model wrappers share a three-head structure (semantic, instance,
geometry) attached via :class:`VistaTaskHead3D`.
"""

from brainbow.models.base import BaseModel
from brainbow.models.cosmos_transfer_2_5 import CosmosTransfer3DWrapper
from brainbow.models.vista import Vista3DWrapper

__all__ = [
    "BaseModel",
    "Vista3DWrapper",
    "CosmosTransfer3DWrapper",
]
