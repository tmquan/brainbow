"""
Cosmos-Transfer2.5 **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.
"""

from brainbow.losses import CombinedLoss
from brainbow.models.cosmos_transfer_2_5 import CosmosTransfer3DWrapper
from brainbow.modules.cosmos_transfer_2_5.base import BaseCosmosModule


class CosmosTransfer3DModule(BaseCosmosModule):
    """Cosmos-Transfer2.5 3-D volumetric segmentation module.

    Four output heads:

    * ``semantic`` ``[B, L, D, H, W]``      -- per-voxel class logits.
    * ``instance`` ``[B, E, D, H, W]``      -- discriminative embedding.
    * ``geometry`` ``[B, G, D, H, W]``      -- raw + dir + cov-upper-tri.
    * ``boundary`` ``[B, C, D, H, W]``      -- raw + avg RGB + 6 direct face-affinity (10 channels).

    Heads with weight ``0`` in the loss config are not constructed; see
    :class:`brainbow.losses.CombinedLoss`.
    """

    _model_cls = CosmosTransfer3DWrapper
    _loss_cls = CombinedLoss
