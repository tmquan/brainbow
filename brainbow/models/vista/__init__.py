"""
Vista3D model package.

Splits the wrapper from the shared VISTA3D-style task head so the head
can be reused by the Cosmos-Transfer 3D wrapper (and any future 3-D
dense-prediction wrapper) without importing the full Vista3D backbone.

Module layout::

    heads.py     -- VistaTaskHead3D  (MONAI UnetrBasicBlock head)
    wrapper.py   -- Vista3DWrapper   (SegResNetDS2 + 3 heads)
    hf_loader.py -- MONAI/VISTA3D-HF pretrained-encoder loader
"""

from brainbow.models.vista.heads import VistaTaskHead3D
from brainbow.models.vista.wrapper import Vista3DWrapper

__all__ = [
    "Vista3DWrapper",
    "VistaTaskHead3D",
]
