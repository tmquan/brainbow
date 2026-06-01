"""Variant registry for Cosmos 3 (Nano) checkpoints.

Cosmos 3 is NVIDIA's omnimodal world-model family (released 2026-05-31).
Unlike the Cosmos 2.5 stack -- a plain ``CosmosTransformer3DModel`` DiT
-- the Cosmos 3 generator is a 16B **Mixture-of-Transformers (MoT)**
omni model (``Cosmos3OmniTransformer``) that jointly handles text /
image / video / audio / action.  For volumetric EM segmentation we use
only its diffusion (video) tower as a feature extractor and feed null
conditioning for every other modality (see
:class:`brainbow.models.cosmos_3_nano.wrapper.Cosmos3Nano3DWrapper`).

The shared architectural fields live on
:class:`brainbow.models.cosmos_2_5_common.variants._VariantConfigBase`;
this module adds the Cosmos 3-specific MoT fields
(``head_dim`` / ``num_key_value_heads`` / ``intermediate_size`` /
``use_moe``) so the standalone fallback can reconstruct the shape.

All numbers below are mirrored verbatim from the published HF configs:

* ``transformer/config.json`` (``Cosmos3OmniTransformer``):
  ``hidden_size=4096``, ``num_hidden_layers=36``,
  ``num_attention_heads=32``, ``num_key_value_heads=8`` (GQA),
  ``head_dim=128``, ``intermediate_size=12288`` (MLP ratio 3.0),
  ``latent_channel=48``, ``latent_patch_size=2``,
  ``max_position_embeddings=262144``, ``use_moe=true``.
* ``vae/config.json`` (``AutoencoderKLWan``, Wan2.2-TI2V VAE):
  ``z_dim=48``, ``scale_factor_spatial=16``, ``scale_factor_temporal=4``.

Release notes
-------------
Only the **Nano** (16B) generator is wired here, pinned at ``main`` on
``nvidia/Cosmos3-Nano``.  ``Cosmos3-Super`` (64B) ships in a separate
repo; add a ``"SUPER"`` entry once its ``transformer/config.json`` is
mirrored (do **not** guess its shape -- copy the published numbers).

Only BF16 is officially supported by Cosmos 3 (per the model card);
``model.dtype`` should stay ``bf16``.
"""

from dataclasses import dataclass
from typing import Dict

from brainbow.models.cosmos_2_5_common.variants import _VariantConfigBase


@dataclass
class _VariantConfig(_VariantConfigBase):
    """Cosmos 3 variant config (adds MoT / GQA shape fields).

    The extra fields are not consumed by the shared diffusers load path
    (which reads the shape straight off the pretrained checkpoint) but
    are required by the random-init standalone fallback and document the
    omni-transformer geometry at a glance.
    """

    # Per-head attention dim (``head_dim`` in the HF config).  Note
    # ``num_heads * head_dim = 32 * 128 = 4096 = hidden_dim`` here.
    head_dim: int = 128
    # Grouped-query attention: KV heads < query heads.
    num_key_value_heads: int = 8
    # Feed-forward inner dim (``intermediate_size``); MLP ratio is
    # ``intermediate_size / hidden_dim`` and is also surfaced via the
    # base ``mlp_ratio`` field for the standalone DiT.
    intermediate_size: int = 12288
    # Mixture-of-Transformers / MoE routing (``use_moe`` in the config).
    use_moe: bool = True


_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "NANO": _VariantConfig(
        hf_repo_id="nvidia/Cosmos3-Nano",
        hf_revision="main",
        hidden_dim=4096,
        num_layers=36,
        num_heads=32,
        latent_channels=48,
        spatial_compression=16,
        temporal_compression=4,
        # 16B params @ bf16 ~= 32 GB of weights before activations /
        # optimiser state; a frozen feature-extractor forward fits in
        # ~40 GB, an end-to-end fine-tune needs materially more.
        estimated_vram_gb=40.0,
        max_sequence_length=262144,
        patch_size=2,
        mlp_ratio=12288 / 4096,  # = 3.0
        head_dim=128,
        num_key_value_heads=8,
        intermediate_size=12288,
        use_moe=True,
    ),
}


__all__ = ["_VARIANT_CONFIGS", "_VariantConfig"]
