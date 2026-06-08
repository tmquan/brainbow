"""
Abstract base for backbone wrappers in :mod:`brainbow.models`.

Why this file exists
--------------------
A "model wrapper" in brainbow is the *whole* network: encoder, decoder
and the single dense head that produces the ``[B, HEAD_CHANNELS, *spatial]``
affinity + sem + raw tensor.  This file declares the minimum contract
every wrapper honours so that downstream code
(:class:`brainbow.modules.base.BaseCircuitModule`,
:class:`brainbow.callbacks.tensorboard.ImageLogger`,
:func:`brainbow.inference.sliding_window_inference`) can stay agnostic
of the specific backbone.

Public surface
--------------
* :class:`BaseModel` -- abstract :class:`torch.nn.Module` whose
  :meth:`forward` returns the ``[B, HEAD_CHANNELS, *spatial]`` head tensor
  and whose :meth:`get_output_channels` returns ``HEAD_CHANNELS``.

Note on subclassing
-------------------
The concrete wrappers (Cosmos-Predict / Transfer / Cosmos3-Nano / Vista)
inherit directly from :class:`torch.nn.Module` rather than
:class:`BaseModel`; the single-tensor ``forward`` contract is still
respected.  New backbone wrappers are encouraged to inherit
:class:`BaseModel` for consistency.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base for backbone wrappers.

    Required overrides
    ------------------
    * :meth:`forward(x)` -- return the ``[B, HEAD_CHANNELS, *spatial]``
      head tensor of raw logits / linear values.  No activation is applied
      in ``forward``: the ``aff + sem`` channels are logits and ``raw`` is
      linear; each consumer applies its own activation.
    * :meth:`get_output_channels()` -- the integer ``HEAD_CHANNELS``.
      Used by sliding-window inference and the image logger to allocate
      output buffers without a real forward pass.

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[B, C, *spatial]``.

        Returns:
            The ``[B, HEAD_CHANNELS, *spatial]`` head tensor of raw
            logits / linear values (no activation applied: the
            ``aff + sem`` channels are logits, ``raw`` is linear).
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
