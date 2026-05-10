"""
Cosmos-Predict 2.5 **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.
"""

from brainbow.losses import CombinedLoss
from brainbow.models.cosmos_predict_2_5 import CosmosPredict3DWrapper
from brainbow.modules.cosmos_2_5_common.base import BaseCosmosModule


class CosmosPredict3DModule(BaseCosmosModule):
    """Cosmos-Predict 2.5 3-D volumetric segmentation module.

    No ControlNet branch (Predict is the upstream base DiT), so
    :meth:`_extra_model_kwargs` returns the empty default; the rest
    (in/head channels, freeze schedule, optimiser groups) is inherited
    from :class:`BaseCosmosModule`.
    """

    _model_cls = CosmosPredict3DWrapper
    _loss_cls = CombinedLoss
