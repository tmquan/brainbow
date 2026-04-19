"""
Base model class defining the interface for all segmentation models.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for segmentation models.

    All models must implement:
    - forward(): Main forward pass returning a dict with at least 'logits'
    - get_output_channels(): Return number of output channels

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels/classes.
        spatial_dims: Spatial dimensions (2 or 3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape [B, C, ...spatial_dims...].

        Returns:
            Dictionary containing model outputs. Must include 'logits' key
            with the main output tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_channels(self) -> int:
        """
        Get the number of output channels.

        Returns:
            Number of output channels/classes.
        """
        raise NotImplementedError

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_dit_backbone(self) -> None:
        """Freeze backbone parameters. Override in subclasses."""
        pass

    def unfreeze_dit_backbone(self) -> None:
        """Unfreeze backbone parameters. Override in subclasses."""
        pass

    def freeze_vae_encoder(self) -> None:
        """Freeze VAE encoder parameters. Override in subclasses."""
        pass

    def unfreeze_vae_encoder(self) -> None:
        """Unfreeze VAE encoder parameters. Override in subclasses."""
        pass

    def freeze_vae_decoder(self) -> None:
        """Freeze VAE decoder parameters. Override in subclasses."""
        pass

    def unfreeze_vae_decoder(self) -> None:
        """Unfreeze VAE decoder parameters. Override in subclasses."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_channels={self.in_channels},\n"
            f"  out_channels={self.out_channels},\n"
            f"  spatial_dims={self.spatial_dims},\n"
            f"  num_parameters={self.get_num_parameters():,}\n"
            f")"
        )
