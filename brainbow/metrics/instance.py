"""
Instance segmentation metrics for connectomics evaluation.

Provides per-point and per-batch variants of:
- ARI  (Adjusted Rand Index)
- AMI  (Adjusted Mutual Information)
- AXI  (geometric mean of ARI and AMI)
- VOI  (Variation of Information, with split/merge decomposition)
- TED  (Tolerant Edit Distance -- min split+merge corrections)
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch


# ======================================================================
# Helpers
# ======================================================================

def _prepare_flat_labels(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten and filter label tensors for metric computation."""
    pred_flat = pred_labels.cpu().numpy().ravel()
    true_flat = true_labels.cpu().numpy().ravel()

    if ignore_background:
        fg_mask = (pred_flat > 0) | (true_flat > 0)
        if not fg_mask.any():
            return np.array([]), np.array([])
        pred_flat = pred_flat[fg_mask]
        true_flat = true_flat[fg_mask]

    return pred_flat, true_flat


def _contingency_table(
    pred: np.ndarray,
    true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build contingency matrix and marginals.

    Returns (contingency, pred_counts, true_counts).
    """
    from scipy.sparse import coo_matrix

    n = len(pred)
    cont = coo_matrix(
        (np.ones(n, dtype=np.float64), (true.astype(np.int64), pred.astype(np.int64))),
        shape=(true.max() + 1, pred.max() + 1),
    ).toarray()
    true_counts = cont.sum(axis=1)
    pred_counts = cont.sum(axis=0)
    return cont, pred_counts, true_counts


# ======================================================================
# ARI
# ======================================================================

def compute_per_point_ari(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Adjusted Rand Index for a single sample.

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        ARI in [0, 1] (clamped).
    """
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return 0.0

    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return 0.0
    return max(0.0, adjusted_rand_score(true_flat, pred_flat))


def compute_per_batch_ari(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """ARI averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred_labels.shape[0]):
        total += compute_per_point_ari(pred_labels[b], true_labels[b], ignore_background)
        count += 1
    return total / count if count > 0 else 0.0


# ======================================================================
# AMI
# ======================================================================

def compute_per_point_ami(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Adjusted Mutual Information for a single sample.

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        AMI in [0, 1] (clamped).
    """
    try:
        from sklearn.metrics import adjusted_mutual_info_score
    except ImportError:
        return 0.0

    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return 0.0
    return max(0.0, adjusted_mutual_info_score(true_flat, pred_flat))


def compute_per_batch_ami(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """AMI averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred_labels.shape[0]):
        total += compute_per_point_ami(pred_labels[b], true_labels[b], ignore_background)
        count += 1
    return total / count if count > 0 else 0.0


# ======================================================================
# AXI  (geometric mean of ARI and AMI)
# ======================================================================

def compute_per_point_axi(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Geometric mean of ARI and AMI for a single sample."""
    ari = compute_per_point_ari(pred_labels, true_labels, ignore_background)
    ami = compute_per_point_ami(pred_labels, true_labels, ignore_background)
    return float(np.sqrt(max(ari, 0.0) * max(ami, 0.0)))


def compute_per_batch_axi(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """AXI averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred_labels.shape[0]):
        total += compute_per_point_axi(pred_labels[b], true_labels[b], ignore_background)
        count += 1
    return total / count if count > 0 else 0.0


# ======================================================================
# VOI  (Variation of Information)
# ======================================================================

class VOIResult(NamedTuple):
    """Variation of Information decomposed into split and merge errors."""
    split: float
    merge: float
    total: float


def compute_per_point_voi(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> VOIResult:
    """Variation of Information for a single sample.

    VOI = H(true|pred) + H(pred|true)
      - H(true|pred) = *split* error  (over-segmentation)
      - H(pred|true) = *merge* error  (under-segmentation)

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        VOIResult(split, merge, total).  Lower is better; 0 = perfect.
    """
    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return VOIResult(0.0, 0.0, 0.0)

    cont, pred_counts, true_counts = _contingency_table(pred_flat, true_flat)
    n = float(len(pred_flat))

    nz = cont > 0

    # H(true|pred) -- split error
    pred_counts_broadcast = np.broadcast_to(pred_counts[np.newaxis, :], cont.shape)
    h_true_given_pred = -np.sum((cont[nz] / n) * np.log2(cont[nz] / pred_counts_broadcast[nz]))

    # H(pred|true) -- merge error
    true_counts_broadcast = np.broadcast_to(true_counts[:, np.newaxis], cont.shape)
    h_pred_given_true = -np.sum((cont[nz] / n) * np.log2(cont[nz] / true_counts_broadcast[nz]))

    return VOIResult(
        split=float(h_true_given_pred),
        merge=float(h_pred_given_true),
        total=float(h_true_given_pred + h_pred_given_true),
    )


def compute_per_batch_voi(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> VOIResult:
    """VOI averaged over a batch [B, ...]."""
    splits, merges = [], []
    for b in range(pred_labels.shape[0]):
        r = compute_per_point_voi(pred_labels[b], true_labels[b], ignore_background)
        splits.append(r.split)
        merges.append(r.merge)
    s = float(np.mean(splits)) if splits else 0.0
    m = float(np.mean(merges)) if merges else 0.0
    return VOIResult(split=s, merge=m, total=s + m)


# ======================================================================
# TED  (Tolerant Edit Distance)
# ======================================================================

def compute_per_point_ted(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Tolerant Edit Distance for a single sample.

    Counts the minimum number of split and merge operations needed to
    transform *pred* into *true*.  A GT segment mapped to K > 1 predicted
    segments costs (K - 1) splits; a predicted segment covering M > 1 GT
    segments costs (M - 1) merges.

    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape.
        ignore_background: Exclude background (label 0).

    Returns:
        TED score (non-negative integer).  0 = perfect match.
    """
    pred_flat, true_flat = _prepare_flat_labels(pred_labels, true_labels, ignore_background)
    if len(pred_flat) == 0:
        return 0.0

    cont, _, _ = _contingency_table(pred_flat, true_flat)

    n_splits = 0
    for i in range(cont.shape[0]):
        row = cont[i]
        overlapping = (row > 0).sum()
        if overlapping > 1:
            n_splits += int(overlapping - 1)

    n_merges = 0
    for j in range(cont.shape[1]):
        col = cont[:, j]
        overlapping = (col > 0).sum()
        if overlapping > 1:
            n_merges += int(overlapping - 1)

    return float(n_splits + n_merges)


def compute_per_batch_ted(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """TED averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred_labels.shape[0]):
        total += compute_per_point_ted(pred_labels[b], true_labels[b], ignore_background)
        count += 1
    return total / count if count > 0 else 0.0
