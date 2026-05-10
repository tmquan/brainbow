"""Shared scaffolding for the Cosmos 2.5 backbone wrappers.

This package factors out everything Cosmos-Transfer 2.5 and
Cosmos-Predict 2.5 have in common -- shared primitive layers, the
random-init standalone DiT fallback, the rank-aware HuggingFace
snapshot downloader, the unified-head decoder adapter, and the
abstract :class:`_BaseCosmos25Wrapper` that ties them together.

The two backbone-specific packages
(:mod:`brainbow.models.cosmos_transfer_2_5`,
:mod:`brainbow.models.cosmos_predict_2_5`) only own:

* their variant registry (different ``hf_repo_id`` / ``hf_revision``)
* the (optional) extension hooks of :class:`_BaseCosmos25Wrapper`
  (Cosmos-Transfer adds a ControlNet branch; Cosmos-Predict has none)

Module layout::

    layers.py         -- shared primitives (_NORM, _PointwiseLinear, _adapt_to_rgb)
    variants.py       -- _VariantConfigBase dataclass
    hf_loader.py      -- rank-aware HuggingFace snapshot download
    standalone_dit.py -- _DiTBlock / _StandaloneDiT3D (random-init fallback)
    decoder.py        -- _FeatureProjector3D, _ProgressiveUpsampler3D,
                         _DecoderAdapter3D (VAE decoder + unified head)
    wrapper_base.py   -- _BaseCosmos25Wrapper (abstract public API)
"""

from brainbow.models.cosmos_2_5_common.decoder import (
    _DecoderAdapter3D,
    _FeatureProjector3D,
    _ProgressiveUpsampler3D,
)
from brainbow.models.cosmos_2_5_common.hf_loader import _download_from_hf
from brainbow.models.cosmos_2_5_common.layers import (
    _CONV,
    _NORM,
    _PointwiseLinear,
    _SPATIAL_DIMS,
    _adapt_to_rgb,
)
from brainbow.models.cosmos_2_5_common.standalone_dit import (
    _DiTBlock,
    _StandaloneDiT3D,
)
from brainbow.models.cosmos_2_5_common.variants import _VariantConfigBase
from brainbow.models.cosmos_2_5_common.wrapper_base import _BaseCosmos25Wrapper

__all__ = [
    "_BaseCosmos25Wrapper",
    "_CONV",
    "_DecoderAdapter3D",
    "_DiTBlock",
    "_FeatureProjector3D",
    "_NORM",
    "_PointwiseLinear",
    "_ProgressiveUpsampler3D",
    "_SPATIAL_DIMS",
    "_StandaloneDiT3D",
    "_VariantConfigBase",
    "_adapt_to_rgb",
    "_download_from_hf",
]
