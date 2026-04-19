"""
Vista Lightning-module package.

Splits the backbone-agnostic base class from the concrete 3-D module
so downstream code can depend on the shared training / freeze-schedule
logic without pulling in the :class:`Vista3DWrapper` model.

Module layout::

    base.py    -- BaseVistaModule   (shared construction + param-group logic)
    module.py  -- Vista3DModule     (concrete 3-D Lightning module)

Both symbols are re-exported at package level for backward compatibility
with the previous ``brainbow.modules.vista`` / ``brainbow.modules.vista3d_module``
imports::

    from brainbow.modules.vista import BaseVistaModule, Vista3DModule
"""

from brainbow.modules.vista.base import BaseVistaModule
from brainbow.modules.vista.module import Vista3DModule

__all__ = ["BaseVistaModule", "Vista3DModule"]
