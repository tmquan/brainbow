"""
Semantic segmentation metrics using MONAI.

Provides per-point and per-batch variants of:
- Dice coefficient via ``monai.metrics.DiceMetric``
- IoU / Jaccard  via ``monai.metrics.MeanIoU``

Both support multi-class evaluation with an optional ``ignore_index``.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from monai.metrics import DiceMetric, MeanIoU


def _to_onehot_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> tuple:
    """Convert integer label maps to one-hot tensors for MONAI metrics.

    Args:
        pred:   [*spatial] int/long predicted class labels.
        target: [*spatial] int/long ground truth class labels.
        num_classes: Total number of classes.
        ignore_index: Label to mask out.

    Returns:
        (pred_oh, target_oh) each shaped [1, num_classes, *spatial] float.
    """
    valid = target != ignore_index
    p = pred.clone().long()
    t = target.clone().long()
    p[~valid] = 0
    t[~valid] = 0

    p_oh = rearrange(
        F.one_hot(p, num_classes).float(), "... c -> 1 c ...",
    )
    t_oh = rearrange(
        F.one_hot(t, num_classes).float(), "... c -> 1 c ...",
    )

    if not valid.all():
        mask = rearrange(valid.float(), "... -> 1 1 ...")
        p_oh = p_oh * mask
        t_oh = t_oh * mask

    return p_oh, t_oh


# ======================================================================
# Dice
# ======================================================================

def compute_per_point_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean Dice coefficient across foreground classes for a single sample."""
    p_oh, t_oh = _to_onehot_pair(pred.cpu(), target.cpu(), num_classes, ignore_index)
    metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    result = metric(p_oh, t_oh)
    val = result.nanmean().item()
    return val if val == val else 0.0  # handle NaN


def compute_per_batch_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean Dice averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred.shape[0]):
        total += compute_per_point_dice(pred[b], target[b], num_classes, ignore_index, eps)
        count += 1
    return total / count if count > 0 else 0.0


# ======================================================================
# IoU (Jaccard)
# ======================================================================

def compute_per_point_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean IoU (Jaccard) across foreground classes for a single sample."""
    p_oh, t_oh = _to_onehot_pair(pred.cpu(), target.cpu(), num_classes, ignore_index)
    metric = MeanIoU(include_background=False, reduction="mean", ignore_empty=True)
    result = metric(p_oh, t_oh)
    val = result.nanmean().item()
    return val if val == val else 0.0  # handle NaN


def compute_per_batch_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean IoU averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred.shape[0]):
        total += compute_per_point_iou(pred[b], target[b], num_classes, ignore_index, eps)
        count += 1
    return total / count if count > 0 else 0.0
