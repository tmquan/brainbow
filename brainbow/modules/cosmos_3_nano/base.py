"""Cosmos 3 (Nano) Lightning-module base (soft re-export).

The training / evaluation / logging loop for every Cosmos backbone is
implemented once in :class:`brainbow.modules.cosmos_2_5_common.base.BaseCosmosModule`.
Cosmos 3 reuses it verbatim, but imports it through this module so the
``cosmos_3_nano`` package never references the ``cosmos_2_5_common`` /
``cosmos_predict_2_5`` namespaces directly -- a thin "soft link" that
keeps the Cosmos 3 backbone self-contained at the import level while the
heavy shared logic stays in one place.
"""

from brainbow.modules.cosmos_2_5_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
