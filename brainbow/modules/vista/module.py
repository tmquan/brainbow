"""Vista3D Lightning module for volumetric connectomics segmentation."""

from brainbow.losses import CombinedLoss
from brainbow.models.vista import Vista3DWrapper
from brainbow.modules.vista.base import BaseVistaModule


class Vista3DModule(BaseVistaModule):
    """Vista3D volumetric segmentation module.

    Three output heads: ``semantic`` ``[B, C, D, H, W]``,
    ``instance`` ``[B, E, D, H, W]``, ``geometry`` ``[B, G, D, H, W]``.
    """

    _SPATIAL_DIMS = 3
    _model_cls = Vista3DWrapper
    _loss_cls = CombinedLoss
