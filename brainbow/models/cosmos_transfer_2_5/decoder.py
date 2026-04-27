"""Decoder-side modules for the Cosmos-Transfer 3D wrapper.

Contains the multi-layer feature projector, a random-init progressive
upsampler, and the pretrained-VAE-aware :class:`_DecoderAdapter3D`
that hosts the four task heads (semantic, instance, geometry,
boundary).

Head activations (applied exactly once, here, before loss / metric /
visualisation consume the outputs)::

    semantic :  sigmoid on every channel (per-channel multi-label
                binary probabilities)
    instance :  linear (discriminative embedding; unbounded Euclidean
                space -- sigmoid would collapse it into the unit
                hypercube)
    geometry :  sigmoid on ch 0 (``raw``, [0, 1] target); linear on
                ``cov`` and ``dir`` (signed values -- sigmoid would
                kill the sign).  Channel layout is owned by
                :class:`brainbow.losses.geometry.GeometryLoss`::
                    [raw(1) | cov(S*(S+1)/2) | dir(S)]
    boundary :  sigmoid on every channel (every target lives in
                [0, 1]: raw intensity, normalised centroid xyz colour,
                binary face affinities).  Channel layout is owned by
                :mod:`brainbow.losses.boundary`::
                    [raw(1) | avg(3) | aff_pred(6)]  -> 10 channels

Keeping the activation policy in a single place (this file) means the
loss modules consume probabilities directly and the TensorBoard
image logger never has to re-apply any activation.
"""

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from brainbow.models.cosmos_transfer_2_5.layers import (
    _NORM,
    _PointwiseLinear,
)
from brainbow.models.vista import VistaTaskHead3D

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
        num_classes: int,
        instance_channels: int,
        geometry_channels: int,
        boundary_channels: int,
        spatial_compression: int,
        temporal_compression: int,
        dropout: float = 0.0,
        freeze_vae_decoder: bool = False,
        disabled_heads: Optional[frozenset] = None,
    ) -> None:
        super().__init__()
        self._has_pretrained = vae_decoder is not None
        self._disabled_heads: frozenset = disabled_heads or frozenset()

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

        # VISTA3D-style task heads.  Each one mirrors MONAI's
        # ``ClassMappingClassify.image_post_mapping`` (2× residual
        # UnetrBasicBlock at a shared refinement width with instance
        # norm) and replaces the class-embedding mask-attention with a
        # 1×1 conv so we can emit continuous targets (instance
        # embeddings, geometry regressions) as well as class logits.
        # Refinement runs at ``feature_size`` so parameter cost stays
        # independent of the VAE decoder's output width (``_hidden_ch``
        # can be much larger on the 14B variant).
        self.head_semantic = VistaTaskHead3D(
            in_channels=self._hidden_ch,
            out_channels=num_classes,
            refine_channels=feature_size,
            dropout=dropout,
        )
        self.head_instance = VistaTaskHead3D(
            in_channels=self._hidden_ch,
            out_channels=instance_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )
        self.head_geometry = VistaTaskHead3D(
            in_channels=self._hidden_ch,
            out_channels=geometry_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )
        self.head_boundary = VistaTaskHead3D(
            in_channels=self._hidden_ch,
            out_channels=boundary_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )

        for name in self._disabled_heads:
            head = getattr(self, f"head_{name}", None)
            if head is not None:
                head.requires_grad_(False)

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
    ) -> Dict[str, torch.Tensor]:
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
        out: Dict[str, torch.Tensor] = {}
        if "semantic" not in self._disabled_heads:
            # Semantic head is sigmoid-only (multi-label per-channel
            # binary).  Apply the activation here so loss / metrics /
            # tensorboard all consume probabilities directly -- there is
            # exactly one sigmoid in the pipeline, and it lives here.
            out["semantic"] = self.head_semantic(decoded).sigmoid()
        if "instance" not in self._disabled_heads:
            # Instance head stays linear: the discriminative embedding
            # loss uses unbounded Euclidean space (delta_v / delta_d
            # margins + centroid-norm regulariser) and a sigmoid would
            # trap every voxel in the unit hypercube.
            out["instance"] = self.head_instance(decoded)
        if "geometry" not in self._disabled_heads:
            # Geometry head layout: [raw(1) | cov(S*(S+1)/2) | dir(S)].
            # - raw (ch 0)    : [0, 1] target -> sigmoid.
            # - cov (ch 1..)  : trace-normalised covariance with signed
            #   off-diagonals; leave linear (sigmoid would kill the sign).
            # - dir (last S)  : signed unit-vector components in [-1, 1];
            #   leave linear for the same reason.
            # Concatenation (not in-place) keeps autograd happy and avoids
            # a view that would break torch.compile.
            geom = self.head_geometry(decoded)
            out["geometry"] = torch.cat(
                [geom[:, :1].sigmoid(), geom[:, 1:]], dim=1,
            )
        if "boundary" not in self._disabled_heads:
            # Boundary head: every target lives in [0, 1] --
            # ch 0 is the normalised raw image, ch 1-3 is the
            # per-instance centroid xyz / (D, H, W), and ch 4-9 are
            # binary face affinities.  A single sigmoid on all 10
            # channels is therefore the correct activation; the BCE /
            # Dice / IoU sub-losses on the affinity block expect
            # pre-sigmoided probabilities.  The loss also derives a
            # second 6-face affinity from the predicted ch 1-3 (see
            # :func:`brainbow.losses.boundary.soft_aff_from_avg`); that
            # derived signal lives entirely on the loss side -- the
            # decoder still emits exactly 10 channels.
            out["boundary"] = self.head_boundary(decoded).sigmoid()
        return out


__all__ = [
    "_DecoderAdapter3D",
    "_FeatureProjector3D",
    "_ProgressiveUpsampler3D",
]
