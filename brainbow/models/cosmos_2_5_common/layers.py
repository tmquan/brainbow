"""Shared primitive layers for the Cosmos 2.5 wrappers.

Used by both :mod:`brainbow.models.cosmos_transfer_2_5` (the base DiT
plus ControlNet) and :mod:`brainbow.models.cosmos_predict_2_5` (base
DiT only).  Kept dependency-light so every other submodule in the
package can import from here without pulling in the heavyweight
diffusers / VAE classes required by the wrappers.
"""

import torch
import torch.nn as nn
from einops import repeat

_SPATIAL_DIMS = 3
_CONV = nn.Conv3d


def _NORM(ch: int) -> nn.GroupNorm:
    """GroupNorm with the largest power-of-two number of groups that divides ``ch``."""
    num_groups = max(g for g in (1, 2, 4, 8, 16, 32) if ch % g == 0)
    return nn.GroupNorm(num_groups, ch)


class _PointwiseLinear(nn.Module):
    """Drop-in replacement for Conv{2,3}d(k=1) using nn.Linear.

    Avoids non-contiguous gradient strides that cause DDP warnings.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from einops import rearrange

        x_in = rearrange(x, "b c ... -> b ... c").to(self.linear.weight.dtype)
        return rearrange(self.linear(x_in), "b ... c -> b c ...")


def _adapt_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Adapt input channels to 3-ch RGB expected by Cosmos.

    For single-channel EM volumes, repeats grayscale to 3 channels.
    This preserves the VAE encoder's pretrained input distribution
    without introducing learnable parameters.
    """
    if x.shape[1] == 3:
        return x
    return repeat(x, "b 1 ... -> b 3 ...")


__all__ = [
    "_CONV",
    "_NORM",
    "_PointwiseLinear",
    "_SPATIAL_DIMS",
    "_adapt_to_rgb",
]
