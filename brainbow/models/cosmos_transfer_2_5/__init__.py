"""
Cosmos-Transfer2.5 **3D** model wrapper for volumetric connectomics segmentation.

Adapts the Cosmos-Transfer2.5 DiT backbone (2B or 14B) as a feature
extractor for the four-head volumetric segmentation task:

- **Semantic**: per-voxel class logits  (``semantic_channels`` channels)
- **Instance**: per-voxel embedding vectors  (``instance_channels`` channels)
- **Geometry**: per-voxel raw intensity, covariance, and direction.
  Channel layout mirrors BoundaryLoss: ``ch 0`` = raw, then cov
  upper-triangle (S*(S+1)/2 channels), then dir (S channels).
- **Boundary**: per-voxel raw-intensity + per-instance centroid colour +
  direct face-affinity (``boundary_channels`` = 10: 1 raw, 3 avg RGB,
  6 face-affinity neighbours in Z-Y-X order T/B/U/D/L/R).  The loss
  also derives a soft 6-aff from the predicted avgloc for dual
  supervision; see :class:`brainbow.losses.BoundaryLoss`.

Cosmos-Transfer2.5 is natively a video model with temporal + spatial
dimensions.  For volumetric EM data the depth axis maps directly to the
temporal axis, making the 3D adaptation architecturally natural::

    EM volume  [B, C, D, H, W]  <->  video  [B, C, T, H, W]

The VAE encoder compresses along all three axes (temporal_compression x
for depth, spatial_compression x for height/width).  The DiT backbone
then processes the full 3D latent grid.

Module layout::

    layers.py         -- shared primitives (_NORM, _PointwiseLinear, _adapt_to_rgb)
    variants.py       -- _VariantConfig dataclass + _VARIANT_CONFIGS registry
    hf_loader.py      -- rank-aware HuggingFace snapshot download
    standalone_dit.py -- _DiTBlock / _StandaloneDiT3D (random-init fallback)
    decoder.py        -- _FeatureProjector3D, _ProgressiveUpsampler3D,
                         _DecoderAdapter3D (VAE decoder + task heads)
    wrapper.py        -- CosmosTransfer3DWrapper (public API)

References:
    - https://github.com/nvidia-cosmos/cosmos-transfer2.5
    - HuggingFace: nvidia/Cosmos-Transfer2.5-2B, nvidia/Cosmos-Transfer2.5-14B
"""

from brainbow.models.cosmos_transfer_2_5.wrapper import CosmosTransfer3DWrapper

__all__ = ["CosmosTransfer3DWrapper"]
