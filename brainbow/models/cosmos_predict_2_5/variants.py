"""Variant registry for Cosmos-Predict 2.5 checkpoints.

Cosmos-Predict 2.5 is the **base DiT** in the Cosmos 2.5 family --
upstream of Cosmos-Transfer 2.5 (which is "Predict + ControlNet
residual branch").  For volumetric EM segmentation we use it purely as
a feature extractor: the upstream text/image/video conditioning is
fed null embeddings inside the wrapper.

Architectural fields are inherited verbatim from
:class:`brainbow.models.cosmos_2_5_common.variants._VariantConfigBase`;
no Predict-specific extension fields are needed (no ControlNet).

Release notes
-------------
The 2B variant lives at ``nvidia/Cosmos-Predict2.5-2B``.  The
``hf_revision`` default below matches Cosmos-Transfer 2.5's base DiT
revision (``diffusers/general``) so the two backbones share a base
when used side-by-side.  Override via ``model.hf_revision`` in the
recipe config if a different branch (e.g. ``diffusers/text2world``,
``diffusers/video2world``, ``diffusers/auto-multi-view``) is desired.
"""

from typing import Dict

from brainbow.models.cosmos_2_5_common.variants import _VariantConfigBase


_VARIANT_CONFIGS: Dict[str, _VariantConfigBase] = {
    "2B": _VariantConfigBase(
        hf_repo_id="nvidia/Cosmos-Predict2.5-2B",
        # Match Cosmos-Transfer 2.5's base revision so Predict and
        # Transfer share the same upstream base DiT weights.  Override
        # via ``model.hf_revision`` in the recipe config to pin a
        # different branch (text2world / video2world / etc.).
        hf_revision="diffusers/general",
        hidden_dim=2048,
        num_layers=28,
        num_heads=16,
        latent_channels=16,
        spatial_compression=8,
        temporal_compression=4,
        estimated_vram_gb=10.0,
        max_sequence_length=32768,
    ),
    # NOTE: Cosmos-Predict 2.5-14B has not been publicly released on
    # HuggingFace.  Architecture is kept for training from scratch
    # (`pretrained=False`); `hf_repo_id=None` prevents silent failure
    # when `pretrained=True`.
    "14B": _VariantConfigBase(
        hf_repo_id=None,
        hf_revision=None,
        hidden_dim=5120,
        num_layers=40,
        num_heads=40,
        latent_channels=16,
        spatial_compression=8,
        temporal_compression=4,
        estimated_vram_gb=46.0,
        max_sequence_length=32768,
    ),
}


__all__ = ["_VARIANT_CONFIGS"]
