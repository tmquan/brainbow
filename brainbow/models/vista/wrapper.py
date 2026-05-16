"""Vista3D wrapper with one unified 32-channel dense-prediction head."""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from brainbow.models.vista.heads import VistaTaskHead3D
from brainbow.models.vista.hf_loader import (
    DEFAULT_VISTA3D_REPO,
    DEFAULT_VISTA3D_REVISION,
    load_pretrained_vista3d_encoder,
)
from brainbow.losses import HEAD_CHANNELS, apply_head_activations

logger = logging.getLogger(__name__)

_SPATIAL_DIMS = 3
# Upstream MONAI VISTA3D trains SegResNetDS2 with this width; matching
# it is what lets the pretrained encoder load cleanly.
_VISTA3D_PRETRAINED_FEATURE_SIZE = 48


class Vista3DWrapper(nn.Module):
    """
    3D version of the Vista architecture for volumetric segmentation.

    Args:
        in_channels: Number of input channels (default: 1 for EM).
        head_channels: Unified dense head width (default 32).
        feature_size: Base feature dimension from backbone (default: 64).
            Set to 48 to load the pretrained MONAI VISTA3D encoder
            cleanly (upstream uses ``init_filters=48``).
        encoder_name: Vista3D internal encoder ('segresnet' or 'swin').
        pretrained: If true, download and load the MONAI VISTA3D encoder
            weights from HuggingFace (``MONAI/VISTA3D-HF``).  Only the
            SegResNetDS2 encoder is loaded; task heads remain randomly
            initialised.  Silently falls back to random init on network
            or shape-mismatch errors (with a warning).
        hf_repo_id / hf_revision / cache_dir / hf_token: Optional
            overrides for the HuggingFace download.

    Example:
        >>> model = Vista3DWrapper(in_channels=1)
        >>> x = torch.randn(1, 1, 64, 64, 64)
        >>> out = model(x)
        >>> out.shape   # [1, 32, 64, 64, 64]
    """

    def __init__(
        self,
        in_channels: int = 1,
        head_channels: int = HEAD_CHANNELS,
        feature_size: int = 64,
        encoder_name: str = "vista3d",
        dropout: float = 0.0,
        pretrained: bool = False,
        hf_repo_id: str = DEFAULT_VISTA3D_REPO,
        hf_revision: str = DEFAULT_VISTA3D_REVISION,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.head_channels = int(head_channels)
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS
        self.dropout = dropout
        self._pretrained = pretrained

        self._build_backbone(encoder_name, **kwargs)

        if pretrained:
            self._maybe_load_pretrained_encoder(
                hf_repo_id=hf_repo_id,
                hf_revision=hf_revision,
                cache_dir=cache_dir,
                hf_token=hf_token,
            )

        # VISTA3D-style unified task head.  It mirrors MONAI's real
        # ``ClassMappingClassify.image_post_mapping`` (2× residual
        # UnetrBasicBlock with instance norm) and replaces the class
        # embedding mask-attention with a per-voxel 1×1 projection so
        # we can emit the 32-channel dense field consumed by
        # ``CombinedLoss``.  Refinement runs at ``feature_size`` — the
        # same width the SegResNetDS2 encoder emits — matching the
        # reference VISTA3D network exactly.
        self.head = VistaTaskHead3D(
            in_channels=feature_size,
            out_channels=self.head_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )

    def _build_backbone(self, encoder_name: str, **kwargs: Any) -> None:
        """Build backbone encoder: SegResNetDS2 (VISTA3D encoder) or SegResNet fallback."""
        if encoder_name in ("vista3d", "segresnet_ds2"):
            try:
                from monai.networks.nets.segresnet_ds import SegResNetDS2
                self.backbone = SegResNetDS2(
                    spatial_dims=_SPATIAL_DIMS,
                    in_channels=self.in_channels,
                    out_channels=self.feature_size,
                    init_filters=self.feature_size,
                    blocks_down=(1, 2, 2, 4, 4),
                    norm="instance",
                    dsdepth=1,
                )
                return
            except ImportError:
                import warnings
                warnings.warn(
                    "SegResNetDS2 not available, falling back to SegResNet. "
                    "Install monai>=1.3 for Vista3D encoder support.",
                    stacklevel=2,
                )

        from monai.networks.nets import SegResNet
        self.backbone = SegResNet(
            spatial_dims=_SPATIAL_DIMS,
            in_channels=self.in_channels,
            out_channels=self.feature_size,
            init_filters=self.feature_size,
            dropout_prob=self.dropout,
        )
        self._use_vista3d = False

    def _maybe_load_pretrained_encoder(
        self,
        hf_repo_id: str,
        hf_revision: str,
        cache_dir: Optional[str],
        hf_token: Optional[str],
    ) -> None:
        """Attempt to load the MONAI VISTA3D encoder; fall back to random init.

        Failure modes:
        * Network / auth error during download  -> warn, keep random init.
        * Shape mismatch (``feature_size != 48``) -> warn, partial load.
        * Backbone is not a SegResNetDS2 (i.e. encoder_name != 'vista3d')
          -> warn, skip; the pretrained keys would not match.
        """
        if not getattr(self, "_use_vista3d", True):
            logger.warning(
                "pretrained=True but encoder_name != 'vista3d'; cannot "
                "load MONAI VISTA3D encoder weights into a plain "
                "SegResNet.  Falling back to random initialisation.",
            )
            return

        if self.feature_size != _VISTA3D_PRETRAINED_FEATURE_SIZE:
            logger.warning(
                "pretrained=True with feature_size=%d -- upstream VISTA3D "
                "uses init_filters=%d, so all encoder tensors will fail "
                "shape-matching.  Set `feature_size=%d` (and matching "
                "head `in_channels`) for a full pretrained load.",
                self.feature_size,
                _VISTA3D_PRETRAINED_FEATURE_SIZE,
                _VISTA3D_PRETRAINED_FEATURE_SIZE,
            )

        try:
            load_pretrained_vista3d_encoder(
                self.backbone,
                repo_id=hf_repo_id,
                revision=hf_revision,
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception as exc:
            logger.warning(
                "Vista3D pretrained-encoder load failed (%s).  Falling "
                "back to random initialisation.", exc,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unified ``[B, 32, D, H, W]`` head tensor.

        Activation policy: sigmoid on the semantic and skeleton channels
        (the contiguous run :data:`brainbow.losses.SIGMOID_SLICE`);
        raw / dir / cov / rad / avg / emb stay linear.
        """
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        out = self.head(feat)
        return apply_head_activations(out)
