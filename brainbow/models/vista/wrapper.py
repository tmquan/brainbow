"""
Vista3D model wrapper for volumetric connectomics segmentation.

3D version of the Vista architecture with three parallel task heads:
- Semantic: per-voxel class logits (num_classes channels)
- Instance: per-voxel embedding vectors for discriminative clustering
  (instance_channels channels)
- Geometry: per-voxel raw-intensity reconstruction, covariance
  upper-triangle, and direction.  Channel layout (must match
  :class:`brainbow.losses.GeometryLoss`)::

      ch 0                  : raw intensity
      ch 1 .. S*(S+1)/2     : covariance upper-triangle
      ch -S ..              : per-voxel direction toward instance centroid

  ``geometry_channels = 1 + S * (S + 1) // 2 + S`` (= ``10`` in 3-D).
"""

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
from brainbow.models.vista.point_prompt_encoder import PointPromptEncoder

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
        num_classes: Number of semantic classes (default: 16).
            Set higher than currently needed to leave headroom for
            future class additions without retraining the backbone.
        instance_channels: Per-voxel instance embedding dimensionality
            (default: 10).
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
        >>> model = Vista3DWrapper(
        ...     in_channels=1, num_classes=16, instance_channels=10,
        ... )
        >>> x = torch.randn(1, 1, 64, 64, 64)
        >>> out = model(x)
        >>> out['semantic'].shape   # [1, 16, 64, 64, 64]
        >>> out['instance'].shape   # [1, 10, 64, 64, 64]
        >>> out['geometry'].shape   # [1, 10, 64, 64, 64]  (raw=1 + cov_tri=6 + dir=3)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 16,
        instance_channels: int = 10,
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
        self.num_classes = num_classes
        self.instance_channels = instance_channels
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS
        self.dropout = dropout
        self._pretrained = pretrained

        S = _SPATIAL_DIMS
        # Geometry head layout: raw (1) + cov upper-tri (S*(S+1)/2) + dir (S).
        # Matches BoundaryLoss channel convention with raw at ch 0.
        self.geometry_channels = 1 + S * (S + 1) // 2 + S

        self._build_backbone(encoder_name, **kwargs)

        if pretrained:
            self._maybe_load_pretrained_encoder(
                hf_repo_id=hf_repo_id,
                hf_revision=hf_revision,
                cache_dir=cache_dir,
                hf_token=hf_token,
            )

        # VISTA3D-style task heads.  Each head mirrors MONAI's real
        # ``ClassMappingClassify.image_post_mapping`` (2× residual
        # UnetrBasicBlock with instance norm) and replaces the class
        # embedding mask-attention with a per-voxel 1×1 projection so
        # we can emit continuous targets (instance embeddings, geometry
        # regressions) as well as class logits.  Refinement runs at
        # ``feature_size`` — the same width the SegResNetDS2 encoder
        # emits — matching the reference VISTA3D network exactly.
        self.head_semantic = VistaTaskHead3D(
            in_channels=feature_size,
            out_channels=num_classes,
            refine_channels=feature_size,
            dropout=dropout,
        )
        self.head_instance = VistaTaskHead3D(
            in_channels=feature_size,
            out_channels=instance_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )
        self.head_geometry = VistaTaskHead3D(
            in_channels=feature_size,
            out_channels=self.geometry_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )

        self.point_encoder = PointPromptEncoder(
            num_classes=num_classes,
            feature_size=feature_size,
            spatial_dims=_SPATIAL_DIMS,
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

    def forward(
        self,
        x: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        point_prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone + three parallel heads.

        Args:
            x: Input tensor [B, C, D, H, W].
            semantic_ids: Optional per-voxel semantic class labels [B, D, H, W].
                Passed through so the loss can compute per-class instance losses.
            point_prompts: Optional dict with keys ``pos_points``,
                ``neg_points``, ``target_semantic_ids``, ``target_instance_ids``
                as produced by :func:`sample_point_prompts`.
        """
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if point_prompts is not None:
            feat = feat + self.point_encoder(
                pos_points=point_prompts["pos_points"],
                neg_points=point_prompts["neg_points"],
                target_semantic_ids=point_prompts["target_semantic_ids"],
                target_instance_ids=point_prompts["target_instance_ids"],
                spatial_shape=feat.shape[2:],
            )

        # Semantic head is sigmoid-only (multi-label per-channel binary);
        # apply the activation here so loss / metrics / tensorboard all
        # consume probabilities directly -- there is exactly one sigmoid
        # in the pipeline, and it lives here.
        out: Dict[str, torch.Tensor] = {
            "semantic": self.head_semantic(feat).sigmoid(),
            "instance": self.head_instance(feat),
            "geometry": self.head_geometry(feat),
        }
        if semantic_ids is not None:
            out["semantic_ids"] = semantic_ids
        return out
