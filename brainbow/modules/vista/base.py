"""
Vista Lightning module for connectomics segmentation.

Thin specialisation of
:class:`~brainbow.modules.base.BaseCircuitModule` for the Vista 2-D /
3-D U-Net-style wrappers.  All training / evaluation / logging lives
in the base class; this module just constructs the wrapper.
"""

from typing import Any, Dict

import torch

from brainbow.modules.base import BaseCircuitModule


class BaseVistaModule(BaseCircuitModule):
    """Abstract base for Vista 2-D / 3-D modules.

    Subclasses **must** define:

    * :attr:`_SPATIAL_DIMS` -- 2 or 3
    * :attr:`_model_cls`    -- model wrapper class
    * :attr:`_loss_cls`     -- loss class (typically
      :class:`brainbow.losses.CombinedLoss`)
    """

    def _build_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        return self._model_cls(
            in_channels=model_config.get("in_channels", 1),
            num_classes=model_config.get(
                "semantic_channels",
                model_config.get("num_classes", 16),
            ),
            instance_channels=model_config.get("instance_channels", 10),
            feature_size=model_config.get("feature_size", 64),
            encoder_name=model_config.get("encoder_name", "vista3d"),
            dropout=model_config.get("dropout", 0.0),
            pretrained=model_config.get("pretrained", False),
            cache_dir=model_config.get("cache_dir"),
            hf_token=model_config.get("hf_token"),
        )
