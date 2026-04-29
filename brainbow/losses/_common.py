"""
Shared helpers + canonical constants for the unified head loss.

The Vista-style task head emits a single ``[B, 30, *spatial]`` tensor
(in 3-D) whose 30 channels carry six fields::

    ch  0       : raw      (1)   linear,  L1 vs raw image intensity
    ch  1       : sem      (1)   sigmoid, BCE + Dice vs (label > 0)
    ch  2 -  4  : dir      (3)   linear,  L1 vs unit centroid-direction field
    ch  5 - 10  : cov      (6)   linear,  L1 vs upper-triangle covariance
    ch 11 - 13  : avg      (3)   linear,  L1 vs normalised centroid (z, y, x)
                                   + derived 12-channel face-affinity loss
    ch 14 - 29  : emb      (16)  linear,  discriminative pull / push / norm
                                   + derived 12-channel face-affinity loss

The order puts the **fixed-width** fields first (raw, sem, dir, cov,
avg = 14 channels) and the **wide embedding** last so anyone slicing
``head[:, :14]`` gets the deterministic-target channels and
``head[:, 14:]`` gets the embedding without having to thread the
embedding width through every downstream consumer.

Activation policy: **sigmoid only on ch 1** (sem; classification-supervised);
linear on every other channel (regression / embedding).

The face-affinity kernel takes the per-voxel field (16-D for ``emb``,
3-D for ``avg``) and shifts it along each of 12 directions to produce
a similarity score per face pair::

    aff[c, v] = exp(-tau * sum_i |field[i, v] - shift_replicate(field[i, v], dir_c)|)

The 12 directions are the **2nd-order face neighbours** along each
spatial axis -- shift = ``+/-1`` and ``+/-2`` along each of Z, Y, X.
First-order neighbours (stride 1) capture immediate cell-cell
boundaries; second-order (stride 2) reaches one voxel further and
gives the loss extra signal at thin processes / faint membranes
where stride-1 affinity alone is noisy.

.. code-block:: text

    direction      axis  shift   neighbour offset
    -----------    ----  -----   ----------------
    T1 / B1         Z     +1/-1   z - 1     / z + 1
    U1 / D1         Y     +1/-1   y - 1     / y + 1
    L1 / R1         X     +1/-1   x - 1     / x + 1
    T2 / B2         Z     +2/-2   z - 2     / z + 2
    U2 / D2         Y     +2/-2   y - 2     / y + 2
    L2 / R2         X     +2/-2   x - 2     / x + 2

(``shift = +k`` means: the shifted value at position ``i`` equals the
input at position ``max(0, i - k)`` -- replicate-pad on the front.)

The aff is supervised on the **emb** field (16-D, at ch 14-29) and the
**avg** field (3-D, at ch 11-13).  Both use the same 12-channel binary
target derived from instance labels.

Public helpers
--------------
* :data:`HEAD_CHANNELS`, :data:`AFF_CHANNELS`, :data:`AFF_NAMES`,
  :data:`DIRECTIONS` -- channel-layout constants.
* :func:`slice_head` -- split ``[B, 30, *spatial]`` into a dict of the
  six fields.
* :func:`shift_replicate` -- shift a tensor along an axis by ``+/- k``
  voxels with replicate (edge-pad) semantics; supports any positive
  ``k``.  Pure torch op, gradient-friendly.
* :func:`affinity_target` -- build the 12-channel binary aff target
  from instance labels.
* :func:`soft_aff_from_field` -- soft 12-channel affinity from an
  arbitrary continuous field via ``exp(-tau * L1)``.
* :func:`upper_tri_to_matrix` -- expand an upper-triangle covariance
  channel stack to symmetric matrices (used by the TB visualiser).
* :func:`canonical_regression_name` / :func:`regression_loss_fn` --
  resolve user-facing names (``mse`` / ``l1`` / ``smooth_l1`` plus
  aliases) to a canonical string or ``F.*`` callable.
* :func:`stable_bce_on_probs` -- per-voxel BCE on already-sigmoided
  probabilities, with fp32 log math safe under bf16-mixed autocast.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Channel layout (single source of truth -- 30 channels in 3-D)
# ---------------------------------------------------------------------------

# Per-field channel widths.
CH_RAW: int = 1
CH_SEM: int = 1
CH_DIR: int = 3        # 3-D (Z, Y, X)
CH_COV: int = 6        # upper-triangle of a 3x3 symmetric matrix
CH_AVG: int = 3        # normalised centroid (z, y, x)
CH_EMB: int = 16       # discriminative embedding -- last so head[:, 14:] is emb

# Slice indices: ``[start, end)`` per field.  The embedding lives at
# the **end** of the channel dim so consumers that slice for the
# clusterer / TB embedding-projection panel can read ``head[:, 14:]``
# without threading the embedding width through every call.
RAW_SLICE: slice = slice(0, CH_RAW)                                   # [0, 1)
SEM_SLICE: slice = slice(RAW_SLICE.stop, RAW_SLICE.stop + CH_SEM)     # [1, 2)
DIR_SLICE: slice = slice(SEM_SLICE.stop, SEM_SLICE.stop + CH_DIR)     # [2, 5)
COV_SLICE: slice = slice(DIR_SLICE.stop, DIR_SLICE.stop + CH_COV)     # [5, 11)
AVG_SLICE: slice = slice(COV_SLICE.stop, COV_SLICE.stop + CH_AVG)     # [11, 14)
EMB_SLICE: slice = slice(AVG_SLICE.stop, AVG_SLICE.stop + CH_EMB)     # [14, 30)

HEAD_CHANNELS: int = EMB_SLICE.stop                                    # 30

# Map from field name to slice.  Every consumer (loss, wrapper, TB
# callback, sliding-window inference) reaches into this dict instead of
# hard-coding indices.  Iteration order matches the channel order.
HEAD_LAYOUT: Dict[str, slice] = {
    "raw": RAW_SLICE,
    "sem": SEM_SLICE,
    "dir": DIR_SLICE,
    "cov": COV_SLICE,
    "avg": AVG_SLICE,
    "emb": EMB_SLICE,
}


# ---------------------------------------------------------------------------
# 2nd-order face-affinity directions (12 channels in 3-D)
# ---------------------------------------------------------------------------

# (name, axis, shift) where ``axis`` is the index in a [B, D, H, W]
# 4-D tensor (1=Z, 2=Y, 3=X) and ``shift`` is the offset in voxels.
# ``shift = +k`` means: the shifted tensor at position ``i`` equals the
# input at position ``i - k`` (replicate-pad the front).  Channel order
# is **stride 1 first, then stride 2**, with Z-Y-X major sort within
# each stride -- so the first 6 entries match the legacy 6-direction
# face-affinity convention exactly.
DIRECTIONS: Tuple[Tuple[str, int, int], ...] = (
    ("T1", 1, +1), ("B1", 1, -1),
    ("U1", 2, +1), ("D1", 2, -1),
    ("L1", 3, +1), ("R1", 3, -1),
    ("T2", 1, +2), ("B2", 1, -2),
    ("U2", 2, +2), ("D2", 2, -2),
    ("L2", 3, +2), ("R2", 3, -2),
)
AFF_NAMES: Tuple[str, ...] = tuple(name for name, _, _ in DIRECTIONS)
AFF_CHANNELS: int = len(DIRECTIONS)                                    # 12


# ---------------------------------------------------------------------------
# Field slicing helpers
# ---------------------------------------------------------------------------

def slice_head(
    head: torch.Tensor,
    *,
    channel_dim: int = 1,
) -> Dict[str, torch.Tensor]:
    """Split a unified head tensor into the six named fields.

    Args:
        head: ``[B, 30, *spatial]`` (or ``[B, *spatial, 30]`` if you set
            ``channel_dim``).
        channel_dim: Axis carrying the 30 channels.  Defaults to ``1``.

    Returns:
        Dict mapping ``"raw" / "sem" / "emb" / "dir" / "cov" / "avg"``
        to the corresponding channel slice (a view of ``head``).
    """
    if head.shape[channel_dim] != HEAD_CHANNELS:
        raise ValueError(
            f"slice_head: expected {HEAD_CHANNELS} channels along axis "
            f"{channel_dim}; got {head.shape[channel_dim]}."
        )
    return {
        name: head.narrow(channel_dim, sl.start, sl.stop - sl.start)
        for name, sl in HEAD_LAYOUT.items()
    }


# ---------------------------------------------------------------------------
# Replicate-shift along an axis (supports arbitrary integer shift)
# ---------------------------------------------------------------------------

def shift_replicate(
    x: torch.Tensor, axis: int, shift: int,
) -> torch.Tensor:
    """Shift ``x`` along ``axis`` by ``shift`` voxels, replicating the edge.

    ``shift > 0`` pads the front of ``axis`` by replicating slab 0 ``shift``
    times and trims the back; ``shift < 0`` mirrors that on the back.
    ``shift == 0`` is a no-op.  The output has the same shape as ``x``.

    Used by :func:`affinity_target` and :func:`soft_aff_from_field` to
    construct the per-voxel "neighbour at offset ``shift`` along ``axis``"
    tensor that drives the face-affinity loss.

    Args:
        x:     Any tensor.
        axis:  Axis to shift along.
        shift: Voxel offset; positive or negative integer.
    """
    N = x.size(axis)
    k = abs(shift)
    if k == 0:
        return x

    # Defensive guard: the affinity directions hard-code shift in
    # {1, 2}, but a future call with ``k >= N`` would silently produce
    # an all-edge tensor.  Replicate-shift only makes sense when the
    # axis is at least as long as the offset.
    if k >= N:
        raise ValueError(
            f"shift_replicate: |shift|={k} >= axis-{axis} size {N}; "
            f"cannot replicate-shift further than the tensor extent."
        )

    if shift > 0:
        # Front-pad with k slabs of x[..., 0, ...]; trim k from back.
        head = x.narrow(axis, 0, 1)
        head_pad = head.expand(*[
            d if a != axis else k for a, d in enumerate(x.shape)
        ])
        body = x.narrow(axis, 0, N - k)
        return torch.cat([head_pad, body], dim=axis)

    # shift < 0
    body = x.narrow(axis, k, N - k)
    tail = x.narrow(axis, -1, 1)
    tail_pad = tail.expand(*[
        d if a != axis else k for a, d in enumerate(x.shape)
    ])
    return torch.cat([body, tail_pad], dim=axis)


def shift_replicate_np(x: np.ndarray, axis: int, shift: int) -> np.ndarray:
    """NumPy counterpart of :func:`shift_replicate` (for CPU target builds)."""
    N = x.shape[axis]
    k = abs(shift)
    if k == 0:
        return x
    if k >= N:
        raise ValueError(
            f"shift_replicate_np: |shift|={k} >= axis-{axis} size {N}."
        )
    if shift > 0:
        head = np.take(x, [0], axis=axis)
        head_pad = np.repeat(head, k, axis=axis)
        body = np.take(x, np.arange(0, N - k), axis=axis)
        return np.concatenate([head_pad, body], axis=axis)
    body = np.take(x, np.arange(k, N), axis=axis)
    tail = np.take(x, [N - 1], axis=axis)
    tail_pad = np.repeat(tail, k, axis=axis)
    return np.concatenate([body, tail_pad], axis=axis)


# ---------------------------------------------------------------------------
# Affinity targets and kernels (12-channel, 2nd-order)
# ---------------------------------------------------------------------------

@torch.no_grad()
def affinity_target(
    labels: torch.Tensor,
    background: Optional[int] = None,
) -> torch.Tensor:
    """Build the 12-channel binary affinity target from ``[B, D, H, W]`` labels.

    For every voxel ``v`` and every direction ``c``::

        aff[c, v] = 1  if labels[v] == labels[v + dir_c]
                  = 0  otherwise

    With replicate padding at the edge a voxel compares against itself,
    so foreground voxels at the volume boundary correctly contribute
    ``aff = 1`` in the "missing" direction.

    Args:
        labels: Integer instance ids of shape ``[B, D, H, W]``.
        background: When set, voxels whose label equals this value are
            masked to ``0`` across all 12 channels -- suppresses the
            spurious ``0 == 0 -> 1`` signal at background voxels (and
            at the boundary voxels zeroed by :class:`FindBoundariesd`).
            ``None`` -> no masking.
    """
    per_dir = [
        (labels == shift_replicate(labels, axis, shift)).to(torch.float32)
        for _, axis, shift in DIRECTIONS
    ]
    out = rearrange(torch.stack(per_dir, dim=0), "c b d h w -> b c d h w")
    if background is not None:
        mask = rearrange(
            (labels != background).to(torch.float32), "b ... -> b 1 ...",
        )
        out = out * mask
    return out


def affinity_target_np(
    labels: np.ndarray,
    background: Optional[int] = None,
) -> np.ndarray:
    """NumPy counterpart of :func:`affinity_target`."""
    per_dir = [
        (labels == shift_replicate_np(labels, axis, shift)).astype(np.float32)
        for _, axis, shift in DIRECTIONS
    ]
    out = np.stack(per_dir, axis=1)
    if background is not None:
        mask = (labels != background).astype(np.float32)[:, None]
        out = out * mask
    return out


def soft_aff_from_field(
    field: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Soft 12-channel affinity derived from a continuous field.

    Mirrors :func:`affinity_target` but on a continuous
    ``[B, C, D, H, W]`` field.  Similarity uses an L1-distance kernel
    with bandwidth ``tau``::

        aff[c, v] = exp(-tau * sum_i |field[i, v] - shift_replicate(field[i, v], dir_c)|)

    Voxels the model has placed close together in field-space (small
    L1) get aff ≈ 1; voxels placed far apart decay smoothly toward 0.
    At the volume edge replicate padding compares the voxel to itself
    so the kernel evaluates to ``1`` -- matches the binary aff target's
    edge convention.

    Used twice in :class:`brainbow.losses.combined.CombinedLoss`:

    1. on the **embedding** field (``ch 2-17``, ``C = 16``) -- a dense
       per-voxel-face complement to the centroid-based pull / push
       terms.
    2. on the **avg** field (``ch 27-29``, ``C = 3``) -- pinning the
       predicted centroid xyz to the boundary structure (two voxels in
       the same instance must share their predicted centroid; two
       across a boundary must disagree).

    Args:
        field: ``[B, C, D, H, W]`` continuous field; ``C >= 1`` arbitrary.
        tau:   Bandwidth of the soft kernel (positive).  Larger ``tau``
               sharpens the decay (more like a hard binary signal),
               smaller ``tau`` softens it.

    Returns:
        ``[B, 12, D, H, W]`` similarity tensor in ``(0, 1]``.
    """
    if field.dim() != 5:
        raise ValueError(
            f"soft_aff_from_field expects a 5-D [B, C, D, H, W] tensor; "
            f"got shape {tuple(field.shape)}."
        )
    per_dir = []
    # Field axes 2/3/4 correspond to label axes 1/2/3, so we shift on
    # ``axis + 1`` to compensate for the channel dim.
    for _, axis, shift in DIRECTIONS:
        diff = (field - shift_replicate(field, axis + 1, shift)).abs()
        l1 = diff.sum(dim=1)                                       # [B, D, H, W]
        per_dir.append(torch.exp(-tau * l1))
    return rearrange(torch.stack(per_dir, dim=0), "c b d h w -> b c d h w")


# ---------------------------------------------------------------------------
# Upper-triangle covariance helper (used by the TB visualiser)
# ---------------------------------------------------------------------------

def upper_tri_to_matrix(
    cov_tri: torch.Tensor,
    spatial_dims: int = 3,
) -> torch.Tensor:
    """Expand an upper-triangle covariance channel stack to full matrices.

    The channel order matches
    :func:`brainbow.transforms.covariance.compute_covariance_field`'s
    row-major upper-triangle::

        (i, j) for i in range(S) for j in range(i, S)

    i.e. 3-D: ``(Czz, Czy, Czx, Cyy, Cyx, Cxx)``.

    Args:
        cov_tri: ``[B, T, *spatial]`` where ``T = S*(S+1)//2``.
        spatial_dims: ``S`` (defaults to 3).

    Returns:
        ``[B, *spatial, S, S]`` symmetric matrices.
    """
    S = spatial_dims
    B = cov_tri.shape[0]
    spatial_shape = cov_tri.shape[2:]
    full = cov_tri.new_zeros(B, *spatial_shape, S, S)
    ch = 0
    for i in range(S):
        for j in range(i, S):
            full[..., i, j] = cov_tri[:, ch]
            if i != j:
                full[..., j, i] = cov_tri[:, ch]
            ch += 1
    return full


# ---------------------------------------------------------------------------
# Regression-loss name resolver
# ---------------------------------------------------------------------------

_REGRESSION_ALIASES: Dict[str, str] = {
    "mse": "mse", "l2": "mse",
    "l1": "l1", "mae": "l1",
    "smooth_l1": "smooth_l1", "huber": "smooth_l1",
}

_REGRESSION_FNS: Dict[str, Callable] = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
}


def canonical_regression_name(name: str) -> str:
    """Resolve a user-facing regression-loss name to its canonical form."""
    key = name.lower().replace("-", "_")
    if key not in _REGRESSION_ALIASES:
        raise ValueError(
            f"Unknown regression loss '{name}'. "
            f"Choose from: {sorted(set(_REGRESSION_ALIASES))}"
        )
    return _REGRESSION_ALIASES[key]


def regression_loss_fn(name: str) -> Callable:
    """Return the ``torch.nn.functional`` callable for a regression loss."""
    return _REGRESSION_FNS[canonical_regression_name(name)]


# ---------------------------------------------------------------------------
# Numerically-stable BCE on probabilities (semantic + any other [0, 1] head)
# ---------------------------------------------------------------------------

def stable_bce_on_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Per-voxel binary cross-entropy on **probabilities** (not logits).

    The model wrapper applies a single sigmoid to the semantic channel
    before the loss sees it, so BCE here consumes ``[0, 1]`` probabilities.
    The log math runs in fp32 with explicit clamping so ``log(p)`` and
    ``log(1 - p)`` stay finite under ``bf16-mixed`` autocast (bf16 has
    only ~3 decimal digits of mantissa near 1, so any ``p > ~0.992``
    rounds to exactly ``1`` and would otherwise yield ``-inf``).

    Args:
        probs:      ``[B, C, *spatial]`` already-sigmoided predictions.
        target:     ``[B, C, *spatial]`` binary target (0 / 1 floats).
        pos_weight: Optional ``[1, C, 1, ..., 1]`` per-channel positive
            weight, semantically equivalent to
            ``BCEWithLogitsLoss(pos_weight=...)`` -- it re-weights only
            the ``t == 1`` term.
        eps:        Clamp bound for numerical stability.

    Returns:
        ``[B, C, *spatial]`` per-voxel BCE.  The caller reduces (mean,
        masked sum / valid_mask, ...) as it sees fit.
    """
    p = probs.float().clamp(eps, 1.0 - eps)
    t = target.float()
    if pos_weight is not None:
        pw = pos_weight.to(p.dtype)
        return -(pw * t * p.log() + (1.0 - t) * (1.0 - p).log())
    return -(t * p.log() + (1.0 - t) * (1.0 - p).log())


__all__ = [
    # Channel layout
    "CH_RAW", "CH_SEM", "CH_DIR", "CH_COV", "CH_AVG", "CH_EMB",
    "HEAD_CHANNELS", "HEAD_LAYOUT",
    "RAW_SLICE", "SEM_SLICE", "DIR_SLICE", "COV_SLICE", "AVG_SLICE", "EMB_SLICE",
    "DIRECTIONS", "AFF_NAMES", "AFF_CHANNELS",
    # Helpers
    "slice_head",
    "shift_replicate", "shift_replicate_np",
    "affinity_target", "affinity_target_np",
    "soft_aff_from_field",
    "upper_tri_to_matrix",
    "canonical_regression_name", "regression_loss_fn",
    "stable_bce_on_probs",
]
