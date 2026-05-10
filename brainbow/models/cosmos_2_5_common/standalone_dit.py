"""Random-init 3-D DiT fallback for the Cosmos 2.5 family.

Used by both Cosmos-Transfer 2.5 and Cosmos-Predict 2.5 wrappers when
neither ``diffusers`` nor the upstream ``cosmos_*`` package can provide
pretrained weights.  Shape-compatible with the official variants
(hidden dim / layer count / head count follow
:class:`_VariantConfigBase`) but trained from scratch.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from brainbow.models.cosmos_2_5_common.variants import _VariantConfigBase


class _DiTBlock(nn.Module):
    """Single DiT block with adaptive layer norm (standalone fallback)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if timestep_emb is not None:
            mod = self.adaLN_modulation(timestep_emb)
            shift1, scale1, shift2, scale2 = rearrange(
                mod, "b (n d) -> n b d", n=4,
            ).unbind(0)
            h = (
                self.norm1(x)
                * (1 + rearrange(scale1, "b d -> b 1 d"))
                + rearrange(shift1, "b d -> b 1 d")
            )
        else:
            h = self.norm1(x)

        h, _ = self.attn(h, h, h)
        x = x + h

        if timestep_emb is not None:
            h = (
                self.norm2(x)
                * (1 + rearrange(scale2, "b d -> b 1 d"))
                + rearrange(shift2, "b d -> b 1 d")
            )
        else:
            h = self.norm2(x)

        return x + self.mlp(h)


class _StandaloneDiT3D(nn.Module):
    """Minimal 3-D DiT matching the Cosmos 2.5 family's shape.

    Patch embedding operates on volumetric patches
    ``(P_d, P_h, P_w) = (patch_size,) * 3`` producing a 1-D sequence of
    tokens processed by self-attention blocks.
    """

    def __init__(self, cfg: _VariantConfigBase) -> None:
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = cfg.patch_size
        self.latent_channels = cfg.latent_channels

        patch_input_dim = cfg.latent_channels * cfg.patch_size ** 3
        self.patch_embed = nn.Linear(patch_input_dim, cfg.hidden_dim)

        self.timestep_embed = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.blocks = nn.ModuleList([
            _DiTBlock(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio)
            for _ in range(cfg.num_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        feature_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run 3-D DiT and optionally return intermediate features.

        Args:
            latent: ``[B, C_lat, D_lat, H_lat, W_lat]``.
            timestep: Scalar or ``[B]`` for adaptive norm.
            feature_layers: Block indices whose outputs to collect.

        Returns:
            ``(final_hidden [B, N, D], intermediates {idx: [B, N, D]})``.
        """
        _param_dtype = self.patch_embed.weight.dtype
        latent = latent.to(dtype=_param_dtype)

        B, _C, _D, _H, _W = latent.shape
        P = self.patch_size

        patches = rearrange(
            latent,
            "b c (d p1) (h p2) (w p3) -> b (d h w) (c p1 p2 p3)",
            p1=P, p2=P, p3=P,
        )
        x = self.patch_embed(patches)

        if timestep is not None:
            if timestep.dim() == 0:
                timestep = repeat(timestep, "-> b 1", b=B)
            elif timestep.dim() == 1:
                timestep = rearrange(timestep, "b -> b 1")
            t_emb = self.timestep_embed(timestep.to(dtype=_param_dtype))
        else:
            t_emb = None

        feature_layers = feature_layers or []
        intermediates: Dict[int, torch.Tensor] = {}

        for idx, block in enumerate(self.blocks):
            x = block(x, t_emb)
            if idx in feature_layers:
                intermediates[idx] = x

        x = self.norm_out(x)
        return x, intermediates


__all__ = ["_DiTBlock", "_StandaloneDiT3D"]
