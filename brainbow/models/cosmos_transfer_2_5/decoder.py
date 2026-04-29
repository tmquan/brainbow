"""Decoder-side modules for the Cosmos-Transfer 3D wrapper.

The decoder hosts one unified Vista-style task head that emits the
canonical ``[B, 30, D, H, W]`` tensor consumed by
``brainbow.losses.CombinedLoss``.  Activation policy is applied exactly
once here: sigmoid on the semantic channel only, linear everywhere else.
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from brainbow.models.cosmos_transfer_2_5.layers import (
    _NORM,
    _PointwiseLinear,
)
from brainbow.models.vista import VistaTaskHead3D
from brainbow.losses import HEAD_CHANNELS, SEM_SLICE

logger = logging.getLogger(__name__)


class _FeatureProjector3D(nn.Module):
    """Fuse multi-layer DiT features into a 3-D spatial feature map."""

    def __init__(
        self,
        hidden_dim: int,
        num_feature_layers: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        total_in = hidden_dim * num_feature_layers
        self.proj = nn.Sequential(
            _PointwiseLinear(total_in, out_dim * 2),
            _NORM(out_dim * 2),
            nn.GELU(),
            _PointwiseLinear(out_dim * 2, out_dim),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        d: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        spatial = [
            rearrange(f, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            for f in features
        ]
        fused = torch.cat(spatial, dim=1)
        return self.proj(fused)


class _ProgressiveUpsampler3D(nn.Module):
    """Progressive 3-D upsampling (each stage doubles spatial dims)."""

    def __init__(self, in_dim: int, out_dim: int, num_stages: int) -> None:
        super().__init__()
        dims = self._interpolate_dims(in_dim, out_dim, num_stages + 1)
        layers: List[nn.Module] = []
        for i in range(num_stages):
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(
                    dims[i], dims[i + 1],
                    kernel_size=4, stride=2, padding=1,
                ),
                _NORM(dims[i + 1]),
                nn.GELU(),
            ))
        self.stages = nn.ModuleList(layers)

    @staticmethod
    def _interpolate_dims(start: int, end: int, n: int) -> List[int]:
        if n <= 1:
            return [start]
        step = (end - start) / (n - 1)
        dims = [
            max(8, int(round((start + i * step) / 8)) * 8)
            for i in range(n)
        ]
        dims[0], dims[-1] = start, end
        return dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class _DecoderAdapter3D(nn.Module):
    """Reuses pretrained VAE decoder for multi-head volumetric segmentation.

    Replaces the decoder's final output convolution with four parallel
    task heads while preserving all pretrained upsampling weights.

    Freeze policy:
      - Decoder body (early / mid blocks): frozen
      - Last up-block + output norm: trainable
      - Task heads: trainable (randomly initialised)
    """

    def __init__(
        self,
        vae_decoder: Optional[nn.Module],
        latent_channels: int,
        feature_size: int,
        spatial_compression: int,
        temporal_compression: int,
        dropout: float = 0.0,
        freeze_vae_decoder: bool = False,
        head_channels: int = HEAD_CHANNELS,
    ) -> None:
        super().__init__()
        self._has_pretrained = vae_decoder is not None
        self.head_channels = int(head_channels)

        if vae_decoder is not None:
            self.to_latent = _PointwiseLinear(feature_size, latent_channels)
            self.decoder_body = vae_decoder
            self._hidden_ch = self._replace_conv_out()
            if freeze_vae_decoder:
                self._freeze_body()
        else:
            self.to_latent = None
            num_up_spatial = int(math.log2(spatial_compression))
            num_up_temporal = int(math.log2(temporal_compression))
            num_stages = max(num_up_spatial, num_up_temporal)
            self.decoder_body = _ProgressiveUpsampler3D(
                in_dim=feature_size, out_dim=feature_size,
                num_stages=num_stages,
            )
            self._hidden_ch = feature_size

        # VISTA3D-style unified task head.  It mirrors MONAI's
        # ``ClassMappingClassify.image_post_mapping`` (2× residual
        # UnetrBasicBlock at a shared refinement width with instance
        # norm) and replaces the class-embedding mask-attention with a
        # 1×1 conv so we can emit the 30-channel dense field.  Refinement
        # runs at ``feature_size`` so parameter cost stays
        # independent of the VAE decoder's output width (``_hidden_ch``
        # can be much larger on the 14B variant).
        self.head = VistaTaskHead3D(
            in_channels=self._hidden_ch,
            out_channels=self.head_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )

    def _replace_conv_out(self) -> int:
        for attr in ("conv_out", "output_conv", "proj_out", "final_conv"):
            if hasattr(self.decoder_body, attr):
                final = getattr(self.decoder_body, attr)
                if hasattr(final, "in_channels"):
                    ch = final.in_channels
                elif hasattr(final, "weight") and final.weight.dim() >= 2:
                    ch = final.weight.shape[1]
                else:
                    continue
                setattr(self.decoder_body, attr, nn.Identity())
                logger.info(
                    "Replaced decoder.%s (hidden_ch=%d) with Identity.", attr, ch,
                )
                return ch
        logger.warning(
            "Could not find decoder final conv; using latent_channels as hidden_ch."
        )
        return self.to_latent.linear.out_features

    def _freeze_body(self) -> None:
        for p in self.decoder_body.parameters():
            p.requires_grad = False
        for attr in ("up_blocks", "up"):
            if hasattr(self.decoder_body, attr):
                blocks = getattr(self.decoder_body, attr)
                if hasattr(blocks, "__len__") and len(blocks) > 0:
                    for p in blocks[-1].parameters():
                        p.requires_grad = True
                break
        for attr in ("conv_norm_out", "norm_out"):
            if hasattr(self.decoder_body, attr):
                for p in getattr(self.decoder_body, attr).parameters():
                    p.requires_grad = True
                break

    def _unfreeze_body(self) -> None:
        for p in self.decoder_body.parameters():
            p.requires_grad = True

    def forward(
        self, features: torch.Tensor, target_size: tuple,
    ) -> torch.Tensor:
        if self._has_pretrained:
            latent = self.to_latent(features)
            body_dtype = next(self.decoder_body.parameters()).dtype
            decoded = self.decoder_body(latent.to(body_dtype))
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
            decoded = decoded.to(features.dtype)
        else:
            decoded = self.decoder_body(features)
        if decoded.shape[-3:] != target_size:
            decoded = F.interpolate(
                decoded, size=target_size, mode="trilinear", align_corners=False,
            )
        out = self.head(decoded)
        return torch.cat(
            [
                out[:, :SEM_SLICE.start],
                out[:, SEM_SLICE].sigmoid(),
                out[:, SEM_SLICE.stop:],
            ],
            dim=1,
        )


__all__ = [
    "_DecoderAdapter3D",
    "_FeatureProjector3D",
    "_ProgressiveUpsampler3D",
]
