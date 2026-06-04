"""Decoder-side modules shared by the Cosmos 2.5 wrappers.

The decoder hosts one unified task head that emits the canonical
``[B, HEAD_CHANNELS, D, H, W]`` affinity + sem + raw tensor consumed by
``brainbow.losses.AffinityFGLoss``.  Activation policy is applied exactly
once here via :func:`brainbow.losses.apply_head_activations`: sigmoid on
the contiguous affinity + sem run (:data:`brainbow.losses.SIGMOID_SLICE`),
linear on the trailing raw-reconstruction channel.
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from brainbow.models.cosmos_2_5_common.layers import (
    _NORM,
    _PointwiseLinear,
)
from brainbow.models.vista import VistaTaskHead3D
from brainbow.losses import HEAD_CHANNELS, apply_head_activations

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

    Replaces the decoder's final output convolution with the unified
    affinity + sem + raw task head while preserving all pretrained upsampling
    weights.  The pretrained ``conv_out`` itself is kept on the side as
    :attr:`original_conv_out` (frozen) so the original Wan pixel
    reconstruction can still be emitted for diagnostic visualisation
    (see :meth:`wan_reconstruct` and the ``true/wan_decoder``
    TensorBoard panel) -- otherwise the ``setattr(..., Identity())``
    swap would garbage-collect the pretrained weights.

    Freeze policy:
      - Decoder body (early / mid blocks): frozen
      - Last up-block + output norm: trainable
      - Task head: trainable (randomly initialised)
      - Preserved ``original_conv_out``: frozen (pretrained weights only).
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

        # Will be populated by ``_replace_conv_out`` when a pretrained
        # decoder is provided; stays ``None`` for the random-init
        # standalone path (where there is no pretrained reconstruction
        # to emit).
        self.original_conv_out: Optional[nn.Module] = None

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
        # 1×1 conv so we can emit the HEAD_CHANNELS dense field.  Refinement
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
                # Preserve the pretrained final conv on the side BEFORE
                # overwriting it with Identity inside the body.  Without
                # this snapshot the weights become unreachable as soon
                # as ``setattr`` swaps in ``nn.Identity()``, and the
                # only way to recover them is a fresh HF download.
                # Keeping them frozen mirrors their pretrained-only
                # contract -- they are diagnostic, never optimised.
                self.original_conv_out = final
                for p in self.original_conv_out.parameters():
                    p.requires_grad = False
                setattr(self.decoder_body, attr, nn.Identity())
                logger.info(
                    "Replaced decoder.%s (hidden_ch=%d) with Identity; "
                    "preserved original conv_out as adapter.original_conv_out "
                    "(frozen) for `true/wan_decoder` reconstruction.",
                    attr, ch,
                )
                return ch
        logger.warning(
            "Could not find decoder final conv; using latent_channels as "
            "hidden_ch.  `true/wan_decoder` panel will be unavailable."
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
        decoded = self._decode_body(features, target_size)
        out = self.head(decoded)
        return apply_head_activations(out)

    def _decode_body(
        self, features: torch.Tensor, target_size: tuple,
    ) -> torch.Tensor:
        """Run features through ``decoder_body`` (with the swapped-out
        ``Identity`` final conv) and resize to ``target_size``.

        Returns the post-norm-out, pre-conv-out feature volume that the
        unified head consumes.  Shared between :meth:`forward` and
        :meth:`wan_reconstruct` so both paths see identical body
        activations on the same call.
        """
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
        return decoded

    @torch.no_grad()
    def wan_reconstruct(
        self, features: torch.Tensor, target_size: tuple,
    ) -> Optional[torch.Tensor]:
        """Pretrained Wan-VAE pixel reconstruction from DiT features.

        Mirrors :meth:`forward`'s body pass and then applies the
        original Wan ``conv_out`` (preserved at construction in
        :attr:`original_conv_out` before the adapter swapped it for
        ``Identity``) instead of the unified task head.

        Diagnostic only -- shows what the pretrained Wan decoder
        believes the model's learned latent should decode to in pixel
        space.  At epoch 0 this matches the input closely; as
        ``decoder_body`` drifts under training the reconstruction
        degrades, which is itself a useful "is the latent staying in
        the VAE's prior?" signal.

        Returns:
            ``[B, 3, D, H, W]`` RGB reconstruction in roughly
            ``[-1, 1]`` (Wan's tanh-equivalent output range), or
            ``None`` if the wrapper was built without a pretrained VAE.
        """
        if not self._has_pretrained or self.original_conv_out is None:
            return None
        decoded = self._decode_body(features, target_size)
        body_dtype = next(self.original_conv_out.parameters()).dtype
        rgb = self.original_conv_out(decoded.to(body_dtype))
        return rgb.to(features.dtype)


__all__ = [
    "_DecoderAdapter3D",
    "_FeatureProjector3D",
    "_ProgressiveUpsampler3D",
]
