"""
Cosmos 3 (Nano) **3D** model wrapper for volumetric connectomics segmentation.

Adapts the Cosmos 3 omni transformer (``Cosmos3OmniTransformer``, 16B
Mixture-of-Transformers) as a feature extractor for the unified
32-channel volumetric segmentation head:

``raw(1) | sem(1) | skl(1) | dir(3) | cov(6) | rad(1) | avg(3) | emb(16)``

See :mod:`brainbow.losses._common` for the canonical slice constants.

Cosmos 3 is NVIDIA's omnimodal world-model family (text / image / video
/ audio / action).  We drive only its diffusion (video) tower and feed
null conditioning for every other modality, so the depth axis of the EM
volume maps directly to the model's temporal axis::

    EM volume  [B, C, D, H, W]  <->  video  [B, C, T, H, W]

The Wan2.2-TI2V VAE compresses 16x spatially / 4x temporally into a
48-channel latent grid; the omni transformer then processes the full 3D
latent.

Module layout::

    variants.py       -- Cosmos 3 variant registry (real HF config numbers)
    wrapper.py        -- Cosmos3Nano3DWrapper (public API, thin subclass
                         of _BaseCosmos25Wrapper)

All shared scaffolding (layers, hf_loader, standalone_dit, decoder,
wrapper base class) lives in :mod:`brainbow.models.cosmos_2_5_common`
and is reused unchanged.

References:
    - HuggingFace: nvidia/Cosmos3-Nano
    - https://github.com/nvidia/cosmos
"""

from brainbow.models.cosmos_3_nano.wrapper import Cosmos3Nano3DWrapper

__all__ = ["Cosmos3Nano3DWrapper"]
