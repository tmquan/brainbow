"""Vista3D Lightning module for volumetric connectomics segmentation."""

from brainbow.losses import AffinityFGLoss
from brainbow.models.vista import Vista3DWrapper
from brainbow.modules.vista.base import BaseVistaModule


class Vista3DModule(BaseVistaModule):
    """Vista3D volumetric segmentation module.

    Emits the single ``[B, HEAD_CHANNELS, D, H, W]`` affinity + sem + raw
    head (raw logits / linear values) supervised by ``AffinityFGLoss``.
    """

    _SPATIAL_DIMS = 3
    _model_cls = Vista3DWrapper
    _loss_cls = AffinityFGLoss
