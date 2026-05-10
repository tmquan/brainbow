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
The 2B variant lives at ``nvidia/Cosmos-Predict2.5-2B``.  As of
2026-05 the diffusers-format branches published on that repo are::

    diffusers/base/post-trained   # default below
    diffusers/base/pre-trained    # earlier checkpoint, less polished

We pin ``diffusers/base/post-trained`` -- the closest analog to
Cosmos-Transfer 2.5's ``diffusers/general`` base DiT -- so Predict
and Transfer warm-start from comparable weights.
"""

from typing import Dict

from brainbow.models.cosmos_2_5_common.variants import _VariantConfigBase


_VARIANT_CONFIGS: Dict[str, _VariantConfigBase] = {
    "2B": _VariantConfigBase(
        hf_repo_id="nvidia/Cosmos-Predict2.5-2B",
        # Predict 2.5 ships the diffusers-format base DiT under
        # ``diffusers/base/{post,pre}-trained`` rather than the
        # ``diffusers/general`` branch Transfer uses.  Pin the
        # post-trained branch (the closer analog to Transfer's
        # ``general``); pass ``diffusers/base/pre-trained`` (or a
        # custom branch / commit) via the variant override if needed.
        hf_revision="diffusers/base/post-trained",
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
