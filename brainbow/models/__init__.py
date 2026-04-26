"""
Model architectures for connectomics segmentation.

Two end-to-end backbone wrappers live here:

* :class:`CosmosTransfer3DWrapper` -- Cosmos-Transfer 2.5 (DiT + VAE)
  with **four** parallel task heads (semantic, instance, geometry,
  boundary).
* :class:`Vista3DWrapper` -- SegResNetDS2 backbone with **three**
  parallel task heads (semantic, instance, geometry; no boundary head).

Each wrapper attaches its task heads via
:class:`brainbow.models.vista.VistaTaskHead3D` so the post-backbone
refinement stack is shared.

The abstract :class:`BaseModel` lays out the common contract every
wrapper honours -- ``forward`` returns a dict keyed by head name, plus a
``get_output_channels()`` helper for downstream code that needs to know
the channel widths without inspecting outputs.
"""

from brainbow.models.base import BaseModel
from brainbow.models.cosmos_transfer_2_5 import CosmosTransfer3DWrapper
from brainbow.models.vista import Vista3DWrapper

__all__ = [
    "BaseModel",
    "Vista3DWrapper",
    "CosmosTransfer3DWrapper",
]
