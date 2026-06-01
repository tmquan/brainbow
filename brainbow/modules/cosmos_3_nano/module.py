"""
Cosmos 3 (Nano) **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.
"""

from brainbow.losses import CombinedLoss
from brainbow.models.cosmos_3_nano import Cosmos3Nano3DWrapper
from brainbow.modules.cosmos_2_5_common.base import BaseCosmosModule


class Cosmos3Nano3DModule(BaseCosmosModule):
    """Cosmos 3 (Nano) 3-D volumetric segmentation module.

    Like Cosmos-Predict (and unlike Cosmos-Transfer), Cosmos 3 has no
    ControlNet branch, so :meth:`_extra_model_kwargs` returns the empty
    default; everything else (in/head channels, ``freeze_dit_backbone``
    schedule, optimiser param-group split on ``model.dit.*``) is
    inherited from :class:`BaseCosmosModule`.
    """

    _model_cls = Cosmos3Nano3DWrapper
    _loss_cls = CombinedLoss
