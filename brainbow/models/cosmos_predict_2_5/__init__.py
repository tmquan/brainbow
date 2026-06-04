"""
Cosmos-Predict 2.5 **3D** model wrapper for volumetric connectomics segmentation.

Adapts the Cosmos-Predict 2.5 DiT backbone (2B or 14B) as a feature
extractor for the volumetric segmentation head:

``aff(N_AFF) | sem(1) | raw(1)``  (``HEAD_CHANNELS`` total)

See :mod:`brainbow.losses._common` for the canonical slice constants and
the affinity offset set.

Cosmos-Predict 2.5 is the **base DiT** of the Cosmos 2.5 family,
upstream of Cosmos-Transfer 2.5 (which adds a ControlNet residual
branch on top).  As with Transfer, the depth axis of the EM volume
maps directly to the DiT's temporal axis, so the 3-D adaptation is
architecturally natural::

    EM volume  [B, C, D, H, W]  <->  video  [B, C, T, H, W]

The VAE encoder compresses along all three axes (temporal_compression x
for depth, spatial_compression x for height/width).  The DiT backbone
then processes the full 3D latent grid.

Module layout::

    variants.py       -- Predict-specific variant registry
    wrapper.py        -- CosmosPredict3DWrapper (public API, thin
                         subclass of _BaseCosmos25Wrapper)

All shared scaffolding (layers, hf_loader, standalone_dit, decoder,
wrapper base class) lives in :mod:`brainbow.models.cosmos_2_5_common`
and is reused by :mod:`brainbow.models.cosmos_transfer_2_5`.

References:
    - https://github.com/nvidia-cosmos/cosmos-predict2
    - HuggingFace: nvidia/Cosmos-Predict2.5-2B
"""

from brainbow.models.cosmos_predict_2_5.wrapper import CosmosPredict3DWrapper

__all__ = ["CosmosPredict3DWrapper"]
