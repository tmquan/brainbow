"""
Cosmos-Predict 2.5 Lightning-module package.

Re-exports :class:`BaseCosmosModule` (shared base, lives in
:mod:`brainbow.modules.cosmos_2_5_common`) and the concrete 3-D
Lightning module that wires the
:class:`CosmosPredict3DWrapper` model into the base class.

Module layout::

    module.py  -- CosmosPredict3DModule (concrete 3-D Lightning module)

The base class lives in :mod:`brainbow.modules.cosmos_2_5_common.base`
and is shared with :mod:`brainbow.modules.cosmos_transfer_2_5`.
"""

from brainbow.modules.cosmos_2_5_common.base import BaseCosmosModule
from brainbow.modules.cosmos_predict_2_5.module import CosmosPredict3DModule

__all__ = ["BaseCosmosModule", "CosmosPredict3DModule"]
