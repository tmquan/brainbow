"""Shared variant-config dataclass for the Cosmos 2.5 family.

Each concrete backbone (Transfer / Predict) keeps its own
``_VARIANT_CONFIGS`` registry under its package, but the row dataclass
is defined here so the shared :class:`_BaseCosmos25Wrapper` can stay
agnostic of which family it's serving.  Backbone-specific extensions
(e.g. Transfer's ``hf_revision_controlnet``) extend this dataclass
in their own ``variants.py``.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class _VariantConfigBase:
    """Architecture and download metadata common to every Cosmos 2.5 variant."""

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


__all__ = ["_VariantConfigBase"]
