"""
Cosmos 3 (Nano) Lightning-module package.

Re-exports :class:`BaseCosmosModule` (shared base, lives in
:mod:`brainbow.modules.cosmos_2_5_common`) and the concrete 3-D
Lightning module that wires the :class:`Cosmos3Nano3DWrapper` model
into the base class.

Module layout::

    module.py  -- Cosmos3Nano3DModule (concrete 3-D Lightning module)

The base class lives in :mod:`brainbow.modules.cosmos_2_5_common.base`
and is shared with :mod:`brainbow.modules.cosmos_predict_2_5` and
:mod:`brainbow.modules.cosmos_transfer_2_5`.
"""

from brainbow.modules.cosmos_3_nano.base import BaseCosmosModule
from brainbow.modules.cosmos_3_nano.module import Cosmos3Nano3DModule

__all__ = ["BaseCosmosModule", "Cosmos3Nano3DModule"]
