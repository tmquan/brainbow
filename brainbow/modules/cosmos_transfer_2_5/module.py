"""
Cosmos-Transfer 2.5 **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.
"""

from typing import Any, Dict

from brainbow.losses import CombinedLoss
from brainbow.models.cosmos_transfer_2_5 import CosmosTransfer3DWrapper
from brainbow.modules.cosmos_2_5_common.base import BaseCosmosModule


class CosmosTransfer3DModule(BaseCosmosModule):
    """Cosmos-Transfer 2.5 3-D volumetric segmentation module.

    Forwards the Transfer-specific ``controlnet_revision`` and
    ``freeze_controlnet`` knobs through to
    :class:`CosmosTransfer3DWrapper` via :meth:`_extra_model_kwargs`;
    the rest (in/head channels, freeze schedule, optimiser groups) is
    inherited from :class:`BaseCosmosModule`.
    """

    _model_cls = CosmosTransfer3DWrapper
    _loss_cls = CombinedLoss

    def _extra_model_kwargs(
        self, model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "controlnet_revision": model_config.get("controlnet_revision"),
            "freeze_controlnet": model_config.get("freeze_controlnet", False),
        }
