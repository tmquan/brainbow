"""
Cosmos-Transfer 2.5 **3D** model wrapper for volumetric connectomics segmentation.

Adapts the Cosmos-Transfer 2.5 DiT backbone (2B or 14B) as a feature
extractor for the volumetric segmentation head:

``aff(N_AFF) | sem(1) | raw(1)``  (``HEAD_CHANNELS`` total)

See :mod:`brainbow.losses._common` for the canonical slice constants and
the affinity offset set.

Cosmos-Transfer 2.5 is natively a video model with temporal + spatial
dimensions.  For volumetric EM data the depth axis maps directly to the
temporal axis, making the 3D adaptation architecturally natural::

    EM volume  [B, C, D, H, W]  <->  video  [B, C, T, H, W]

The VAE encoder compresses along all three axes (temporal_compression x
for depth, spatial_compression x for height/width).  The DiT backbone
then processes the full 3D latent grid.

Module layout::

    variants.py       -- _VariantConfig (extends _VariantConfigBase with
                         hf_revision_controlnet) + _VARIANT_CONFIGS registry
    wrapper.py        -- CosmosTransfer3DWrapper (public API, ControlNet add-on)

All shared scaffolding (layers, hf_loader, standalone_dit, decoder,
wrapper base class) lives in :mod:`brainbow.models.cosmos_2_5_common`
and is reused by :mod:`brainbow.models.cosmos_predict_2_5`.

References:
    - https://github.com/nvidia-cosmos/cosmos-transfer2.5
    - HuggingFace: nvidia/Cosmos-Transfer2.5-2B, nvidia/Cosmos-Transfer2.5-14B
"""

from brainbow.models.cosmos_transfer_2_5.wrapper import CosmosTransfer3DWrapper

__all__ = ["CosmosTransfer3DWrapper"]
