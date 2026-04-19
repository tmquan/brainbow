"""
Point prompt encoder for interactive/proofread segmentation.

Converts sparse point coordinates with semantic class and instance identity
into a dense spatial feature map that is added (residual) to backbone
features before the task heads.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class PointPromptEncoder(nn.Module):
    """Encode sparse point prompts into a dense feature volume.

    Builds a sparse indicator volume with ``num_classes + 3`` channels:

    - ``pos_map``      [B, 1, ...]            +1 at positive point locations
    - ``neg_map``      [B, 1, ...]            +1 at negative point locations
    - ``semantic_map`` [B, num_classes, ...]   one-hot of target class at all
      point locations
    - ``instance_map`` [B, 1, ...]            normalised instance id at
      positive point locations

    A small conv block projects this to ``[B, feature_size, ...]`` so it can
    be added as a residual to backbone features.  When no points are provided
    the sparse volume is all zeros, producing near-zero output after BN,
    so the model degrades to automatic mode.

    Args:
        num_classes:  Number of semantic classes.
        feature_size: Output channel dimension (must match backbone output).
        spatial_dims: 2 or 3.
    """

    def __init__(
        self,
        num_classes: int,
        feature_size: int,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.spatial_dims = spatial_dims
        in_ch = num_classes + 3  # pos + neg + one-hot class + instance
        num_groups = max(g for g in (1, 2, 4, 8, 16, 32) if feature_size % g == 0)

        if spatial_dims == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, feature_size, 3, padding=1),
                nn.GroupNorm(num_groups, feature_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, feature_size, 3, padding=1),
                nn.GroupNorm(num_groups, feature_size),
                nn.ReLU(inplace=True),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize conv weights near zero so the encoder starts as a no-op."""
        for m in self.conv.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.normal_(m.weight, std=1e-4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_sparse_volume(
        self,
        pos_points: List[torch.Tensor],
        neg_points: List[torch.Tensor],
        target_semantic_ids: torch.Tensor,
        target_instance_ids: torch.Tensor,
        spatial_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Scatter points into a dense indicator volume.

        Returns:
            Tensor of shape ``[B, num_classes + 3, *spatial_shape]``.
        """
        B = len(pos_points)
        C = self.num_classes + 3
        device = target_semantic_ids.device
        vol = torch.zeros(B, C, *spatial_shape, device=device)

        for b in range(B):
            pts_pos = pos_points[b]  # [N_pos, spatial_dims]
            pts_neg = neg_points[b]  # [N_neg, spatial_dims]
            sem_id = target_semantic_ids[b].long().clamp(0, self.num_classes - 1)
            ins_id = target_instance_ids[b].float()
            ins_max = ins_id.abs().clamp(min=1.0)
            norm_iid = ins_id / ins_max

            # Unified 2D / 3D indexing: convert [N, S] coords to a tuple
            # of 1-D index tensors for advanced indexing.
            def _scatter(channel: int, idx: torch.Tensor, value: float) -> None:
                """Write *value* at spatial positions *idx* into vol[b, channel]."""
                coords = tuple(idx[:, d] for d in range(self.spatial_dims))
                vol[(b, channel) + coords] = value

            if pts_pos.numel() > 0:
                idx = pts_pos.long()
                _scatter(0, idx, 1.0)                                  # pos_map
                _scatter(2 + sem_id, idx, 1.0)                         # semantic_map
                _scatter(-1, idx, norm_iid)                            # instance_map

            if pts_neg.numel() > 0:
                idx = pts_neg.long()
                _scatter(1, idx, 1.0)                                  # neg_map
                _scatter(2 + sem_id, idx, 1.0)                         # semantic_map

        return vol

    def forward(
        self,
        pos_points: List[torch.Tensor],
        neg_points: List[torch.Tensor],
        target_semantic_ids: torch.Tensor,
        target_instance_ids: torch.Tensor,
        spatial_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Encode point prompts into a dense feature volume.

        Args:
            pos_points: List (length B) of ``[N_pos, spatial_dims]`` tensors.
            neg_points: List (length B) of ``[N_neg, spatial_dims]`` tensors.
            target_semantic_ids: ``[B]`` int tensor — semantic class per sample.
            target_instance_ids: ``[B]`` int tensor — instance label per sample.
            spatial_shape: Spatial dimensions of the backbone feature map.

        Returns:
            Dense feature tensor ``[B, feature_size, *spatial_shape]``.
        """
        vol = self._build_sparse_volume(
            pos_points, neg_points,
            target_semantic_ids, target_instance_ids,
            spatial_shape,
        )
        return self.conv(vol)
