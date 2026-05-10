"""The public :class:`CosmosPredict3DWrapper` model.

Cosmos-Predict 2.5 is the **base DiT** of the Cosmos 2.5 family, with
no ControlNet residual branch.  All scaffolding (HuggingFace download,
diffusers DiT + Wan-VAE instantiation, multi-layer feature extraction,
unified-head decoder adapter, freeze / gradient-checkpointing
plumbing) is inherited verbatim from
:class:`brainbow.models.cosmos_2_5_common.wrapper_base._BaseCosmos25Wrapper`;
the only things this file owns are the Predict variant registry and
the (currently empty) extension-hook overrides.
"""

import logging

from brainbow.models.cosmos_2_5_common.wrapper_base import _BaseCosmos25Wrapper
from brainbow.models.cosmos_predict_2_5.variants import _VARIANT_CONFIGS

logger = logging.getLogger(__name__)


class CosmosPredict3DWrapper(_BaseCosmos25Wrapper):
    """Cosmos-Predict 2.5 base DiT for volumetric connectomics segmentation.

    A single unified task head produces ``[B, 30, D, H, W]``.  Channel
    layout is owned by :mod:`brainbow.losses._common`: raw(1), sem(1),
    dir(3), cov(6), avg(3), emb(16).

    Cosmos-Predict 2.5 is natively a video / world-model DiT;
    text/image/video conditioning is fed null embeddings inside the
    wrapper so the backbone behaves as a pure feature extractor.

    Args:
        in_channels: Number of input channels (1 for EM volumes).
        head_channels: Unified head width (default 30).
        feature_size: Internal feature map channel count after projection.
        variant: ``"2B"`` or ``"14B"`` model variant.
        checkpoint_variant: HuggingFace revision string.
        dtype: Weight dtype (``"bf16"``, ``"fp16"``, ``"fp32"``).
        pretrained: Auto-pull the variant's HuggingFace checkpoint
            (``nvidia/Cosmos-Predict2.5-2B``) on first instantiation.
        freeze_dit_backbone: Whether to freeze the pretrained DiT.
            Unlike Cosmos-Transfer (which leans on a trainable
            ControlNet branch and freezes the base), Predict has no
            side branch -- if you want any backbone adaptation at all,
            set this to ``False`` (or to a positive int N for an
            N-epoch frozen warm-up before unfreezing).
        freeze_vae_decoder: Whether to freeze the Wan-VAE decoder body
            (last up-block + output norm stay trainable regardless;
            see :class:`._DecoderAdapter3D`).
        freeze_vae_encoder: Whether to freeze the Wan-VAE encoder.
        gradient_checkpointing: Activation checkpointing on DiT blocks.
        feature_layers: DiT block indices to extract features from.
        cache_dir: HuggingFace download cache directory.
        hf_token: HuggingFace authentication token.
        dropout: Dropout probability for heads.

    Example::

        >>> model = CosmosPredict3DWrapper(in_channels=1, variant="2B")
        >>> x = torch.randn(1, 1, 32, 64, 64)
        >>> out = model(x)
        >>> out.shape   # [1, 30, 32, 64, 64]
    """

    _variant_configs = _VARIANT_CONFIGS


__all__ = ["CosmosPredict3DWrapper"]
