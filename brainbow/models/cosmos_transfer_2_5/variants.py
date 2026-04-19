"""Variant registry for Cosmos-Transfer2.5 checkpoints.

One :class:`_VariantConfig` entry captures both architectural shape
(hidden dim, layer count, compression ratios) and download metadata
(HF repo id + revision) so the wrapper can spin up a 2B or 14B
variant without branching on variant-specific constants elsewhere.

Release notes
-------------
As of 2026-04, only the **2B** Cosmos-Transfer2.5 variant is published
to HuggingFace (``nvidia/Cosmos-Transfer2.5-2B``).  The ``14B`` entry
below keeps the architectural spec so training from scratch
(``pretrained=False``) is possible, but its ``hf_repo_id`` is ``None``
-- HF auto-pull will refuse to proceed and the wrapper will raise a
clear error rather than silently falling back to random weights when
``pretrained=True``.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class _VariantConfig:
    """Architecture and download metadata for a Cosmos-Transfer2.5 variant."""

    hf_repo_id: Optional[str]
    hf_revision: Optional[str]
    hidden_dim: int
    num_layers: int
    num_heads: int
    latent_channels: int
    spatial_compression: int
    temporal_compression: int
    estimated_vram_gb: float
    max_sequence_length: int
    patch_size: int = 2
    mlp_ratio: float = 4.0


_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "2B": _VariantConfig(
        hf_repo_id="nvidia/Cosmos-Transfer2.5-2B",
        hf_revision="diffusers/general",
        hidden_dim=2048,
        num_layers=28,
        num_heads=16,
        latent_channels=16,
        spatial_compression=8,
        temporal_compression=4,
        estimated_vram_gb=12.0,
        max_sequence_length=32768,
    ),
    # NOTE: Cosmos-Transfer2.5-14B has not been publicly released on
    # HuggingFace.  Architecture is kept for training from scratch
    # (`pretrained=False`); `hf_repo_id=None` prevents silent failure
    # when `pretrained=True`.
    "14B": _VariantConfig(
        hf_repo_id=None,
        hf_revision=None,
        hidden_dim=5120,
        num_layers=40,
        num_heads=40,
        latent_channels=16,
        spatial_compression=8,
        temporal_compression=4,
        estimated_vram_gb=48.0,
        max_sequence_length=32768,
    ),
}


__all__ = ["_VARIANT_CONFIGS", "_VariantConfig"]
