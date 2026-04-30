"""
Model architectures for connectomics segmentation.

Two end-to-end backbone wrappers live here:

* :class:`CosmosTransfer3DWrapper` -- Cosmos-Transfer 2.5 (DiT + VAE)
  with a single unified 30-channel head
  (``raw|sem|dir|cov|avg|emb``).
* :class:`Vista3DWrapper` -- SegResNetDS2 backbone with the same
  unified 30-channel head, for fast local iteration.

Both wrappers project their backbone features through
:class:`brainbow.models.vista.VistaTaskHead3D` so the post-backbone
refinement stack is shared.

The abstract :class:`BaseModel` lays out the common contract every
wrapper honours: ``forward`` returns a single tensor of shape
``[B, HEAD_CHANNELS, ...]`` (default ``HEAD_CHANNELS = 30``) and
``get_output_channels()`` returns that integer width for downstream
code that needs to know the head shape without running a forward
pass.  The fixed-slice layout is owned by
:data:`brainbow.losses.HEAD_LAYOUT`.
"""

from brainbow.models.base import BaseModel
from brainbow.models.cosmos_transfer_2_5 import CosmosTransfer3DWrapper
from brainbow.models.vista import Vista3DWrapper

__all__ = [
    "BaseModel",
    "Vista3DWrapper",
    "CosmosTransfer3DWrapper",
]
