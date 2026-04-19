"""
Cosmos-Transfer 2.5 Lightning-module package.

Splits the generic Cosmos training scaffolding (freeze schedule,
NaN/Inf-gradient handling, backbone-vs-heads optimiser groups) from
the concrete 3-D Lightning module that wires the
:class:`CosmosTransfer3DWrapper` model into the base class.

Module layout::

    base.py    -- BaseCosmosModule         (freeze schedule + optimiser split)
    module.py  -- CosmosTransfer3DModule   (concrete 3-D Lightning module)

Both symbols are re-exported at package level for backward
compatibility with the previous ``brainbow.modules.cosmos`` /
``brainbow.modules.cosmostransfer3d_module`` imports::

    from brainbow.modules.cosmos_transfer_2_5 import (
        BaseCosmosModule, CosmosTransfer3DModule,
    )
"""

from brainbow.modules.cosmos_transfer_2_5.base import BaseCosmosModule
from brainbow.modules.cosmos_transfer_2_5.module import CosmosTransfer3DModule

__all__ = ["BaseCosmosModule", "CosmosTransfer3DModule"]
