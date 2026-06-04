"""
Model architectures for connectomics segmentation.

Four end-to-end backbone wrappers live here:

* :class:`Cosmos3Nano3DWrapper` -- Cosmos 3 (Nano) 16B omni
  Mixture-of-Transformers (``Cosmos3OmniTransformer`` + Wan VAE) with
  the unified 32-channel head.  The default backbone.  Shares all
  scaffolding with the Cosmos 2.5 family via
  :mod:`brainbow.models.cosmos_2_5_common`.
* :class:`CosmosTransfer3DWrapper` -- Cosmos-Transfer 2.5 (base DiT +
  ControlNet residual branch + VAE) with a single unified 32-channel
  head (``raw|sem|dir|cov|avg|emb``).
* :class:`CosmosPredict3DWrapper` -- Cosmos-Predict 2.5 (base DiT +
  VAE, no ControlNet) with the same unified 32-channel head.  Shares
  all scaffolding with Transfer via
  :mod:`brainbow.models.cosmos_2_5_common`.
* :class:`Vista3DWrapper` -- SegResNetDS2 backbone with the same
  unified 32-channel head, for fast local iteration.

All wrappers project their backbone features through
:class:`brainbow.models.vista.VistaTaskHead3D` so the post-backbone
refinement stack is shared.

The abstract :class:`BaseModel` lays out the common contract every
wrapper honours: ``forward`` returns a single tensor of shape
``[B, HEAD_CHANNELS, ...]`` (the affinity + sem + raw head,
``HEAD_CHANNELS = N_AFF + 2``) and ``get_output_channels()`` returns
that integer width for downstream code that needs to know the head
shape without running a forward pass.  The fixed-slice layout is owned
by :data:`brainbow.losses.HEAD_LAYOUT`.
"""

from brainbow.models.base import BaseModel
from brainbow.models.cosmos_3_nano import Cosmos3Nano3DWrapper
from brainbow.models.cosmos_predict_2_5 import CosmosPredict3DWrapper
from brainbow.models.cosmos_transfer_2_5 import CosmosTransfer3DWrapper
from brainbow.models.vista import Vista3DWrapper

__all__ = [
    "BaseModel",
    "Cosmos3Nano3DWrapper",
    "CosmosPredict3DWrapper",
    "CosmosTransfer3DWrapper",
    "Vista3DWrapper",
]
