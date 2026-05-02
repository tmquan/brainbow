"""Variant registry for Cosmos-Transfer2.5 checkpoints.

One :class:`_VariantConfig` entry captures both architectural shape
(hidden dim, layer count, compression ratios) and download metadata
(HF repo id + revisions) so the wrapper can spin up a 2B or 14B
variant without branching on variant-specific constants elsewhere.

ControlNet split
----------------
Cosmos-Transfer2.5 is structured as **base DiT + ControlNet**:

* ``hf_revision`` (e.g. ``diffusers/general``) holds the full base
  ``CosmosTransformer3DModel`` -- the "upper" path that does the bulk
  of the work and we keep frozen by default.
* ``hf_revision_controlnet`` (e.g. ``diffusers/controlnet/general/edge``)
  holds a small ``CosmosControlNetModel`` (a few replicated transformer
  blocks) whose ``control_block_samples`` are injected into the base
  every ``controlnet_block_every_n`` blocks.  This is the residual
  branch we keep trainable.

The two revisions live in the **same** repo (``nvidia/Cosmos-Transfer2.5-2B``);
the loader in :mod:`.wrapper` downloads both and instantiates the
matching diffusers classes.

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
    # Branch holding the ControlNet residual weights.  ``None`` disables
    # the ControlNet load path (variant trains on base DiT only).
    hf_revision_controlnet: Optional[str] = None


_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "2B": _VariantConfig(
        hf_repo_id="nvidia/Cosmos-Transfer2.5-2B",
        hf_revision="diffusers/general",
        # ``edge`` is the closest analog to EM contrast (Canny-style
        # high-frequency structure); other shipped modalities are
        # ``depth`` / ``seg`` / ``blur``.  Override via
        # ``model.controlnet_revision`` in the recipe config.
        hf_revision_controlnet="diffusers/controlnet/general/edge",
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
        hf_revision_controlnet=None,
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
