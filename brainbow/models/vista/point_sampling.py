"""
Point prompt sampling utilities for interactive/proofread training.

Samples positive and negative points from ground-truth semantic and
instance masks, returning the structured dict expected by
:class:`brainbow.models.vista.PointPromptEncoder`.

Lives inside :mod:`brainbow.models.vista` because Vista is the only
model family that supports proofread-mode training (Cosmos runs
automatic-only with null prompt embeddings).
"""

from typing import Any, Dict, List

import torch


def _sample_coords(mask: torch.Tensor, n: int) -> torch.Tensor:
    """Sample *n* voxel coordinates from a boolean mask (with replacement)."""
    indices = mask.nonzero(as_tuple=False)  # [K, spatial_dims]
    if indices.shape[0] == 0:
        return indices[:0]  # empty, preserving spatial_dims columns
    sel = torch.randint(indices.shape[0], (n,), device=indices.device)
    return indices[sel]


@torch.no_grad()
def sample_point_prompts(
    semantic_labels: torch.Tensor,
    instance_labels: torch.Tensor,
    num_pos: int = 5,
    num_neg: int = 5,
    sample_mode: str = "class",
) -> Dict[str, Any]:
    """Sample point prompts from GT masks for one batch.

    Args:
        semantic_labels: ``[B, *spatial]`` integer semantic class labels.
        instance_labels: ``[B, *spatial]`` integer instance labels (0 = bg).
        num_pos: Number of positive points per sample.
        num_neg: Number of negative points per sample.
        sample_mode: ``"class"`` picks a random foreground *class*;
            ``"instance"`` picks a random foreground *instance*.

    Returns:
        Dict with keys consumed by :class:`PointPromptEncoder`:

        - ``"pos_points"``: list of ``[N_pos, spatial_dims]`` tensors
        - ``"neg_points"``: list of ``[N_neg, spatial_dims]`` tensors
        - ``"target_semantic_ids"``: ``[B]`` int tensor
        - ``"target_instance_ids"``: ``[B]`` int tensor
    """
    B = semantic_labels.shape[0]
    device = semantic_labels.device
    spatial_dims = semantic_labels.dim() - 1

    pos_points: List[torch.Tensor] = []
    neg_points: List[torch.Tensor] = []
    sem_ids: List[int] = []
    inst_ids: List[int] = []

    for b in range(B):
        sem = semantic_labels[b]     # [*spatial]
        inst = instance_labels[b]    # [*spatial]

        if sample_mode == "instance":
            fg_ids = inst.unique()
            fg_ids = fg_ids[fg_ids > 0]
            if fg_ids.numel() == 0:
                pos_points.append(torch.zeros(0, spatial_dims, device=device))
                neg_points.append(torch.zeros(0, spatial_dims, device=device))
                sem_ids.append(0)
                inst_ids.append(0)
                continue

            target_inst = fg_ids[torch.randint(fg_ids.numel(), (1,), device=device)].item()
            pos_mask = inst == target_inst
            neg_mask = ~pos_mask

            target_sem = sem[pos_mask].mode().values.item() if pos_mask.any() else 0
            sem_ids.append(int(target_sem))
            inst_ids.append(int(target_inst))
        else:  # "class"
            fg_classes = sem.unique()
            fg_classes = fg_classes[fg_classes > 0]
            if fg_classes.numel() == 0:
                pos_points.append(torch.zeros(0, spatial_dims, device=device))
                neg_points.append(torch.zeros(0, spatial_dims, device=device))
                sem_ids.append(0)
                inst_ids.append(0)
                continue

            target_sem = fg_classes[torch.randint(fg_classes.numel(), (1,), device=device)].item()
            pos_mask = sem == target_sem
            neg_mask = ~pos_mask

            sem_ids.append(int(target_sem))
            inst_ids.append(0)

        pos_points.append(_sample_coords(pos_mask, num_pos))
        neg_points.append(_sample_coords(neg_mask, num_neg))

    return {
        "pos_points": pos_points,
        "neg_points": neg_points,
        "target_semantic_ids": torch.tensor(sem_ids, dtype=torch.long, device=device),
        "target_instance_ids": torch.tensor(inst_ids, dtype=torch.long, device=device),
    }
