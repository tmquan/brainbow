"""
Shared helpers + canonical constants for the unified head loss.

The Vista-style task head emits a single ``[B, 32, *spatial]`` tensor
(in 3-D) whose 32 channels carry eight fields::

    ch  0       : raw      (1)   linear,  L1 vs raw image intensity
    ch  1       : sem      (1)   sigmoid, Dice vs (label > 0)
    ch  2       : skl      (1)   sigmoid, Dice vs binary skeleton mask
    ch  3 -  5  : dir      (3)   linear,  L1 vs unit skeleton-direction field
    ch  6 - 11  : cov      (6)   linear,  L1 vs upper-triangle Voronoi-cell
                                          local-segment covariance
    ch 12       : rad      (1)   linear,  L1 vs distance to nearest skeleton
                                          voxel (per-instance normalised by
                                          default)
    ch 13 - 15  : avg      (3)   linear,  L1 vs normalised centroid (z, y, x)
                                   + derived 12-channel face-affinity loss
    ch 16 - 31  : emb      (16)  linear,  discriminative pull / push / norm
                                   + derived 12-channel face-affinity loss

The order puts the **fixed-width** fields first (raw, sem, skl, dir,
cov, rad, avg = 16 channels) and the **wide embedding** last so anyone
slicing ``head[:, :16]`` gets the deterministic-target channels and
``head[:, 16:]`` gets the embedding without having to thread the
embedding width through every downstream consumer.

Activation policy: **sigmoid on ch 1 and ch 2** (sem + skl;
classification-supervised); linear on every other channel
(regression / embedding).  The two sigmoid channels are contiguous at
``[1, 3)`` so :data:`SIGMOID_SLICE` is a single slice that wrappers
hand to :func:`apply_head_activations`.

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

The (sem, skl) sigmoid block plus the two derived ``aff_*`` paths are
all Dice-only since May 2026; the previous per-voxel BCE helper
(``stable_bce_on_probs``) was removed along with the
``weight_ce`` / ``class_weights`` sub-knobs.  Dice is naturally
imbalance-robust on a binary positive target, so the BCE term was
just adding a tuning knob without changing the converged solution.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Channel layout (single source of truth -- 32 channels in 3-D)
# ---------------------------------------------------------------------------

# Per-field channel widths.
CH_RAW: int = 1
CH_SEM: int = 1
CH_SKL: int = 1        # binary skeleton mask
CH_DIR: int = 3        # 3-D (Z, Y, X)
CH_COV: int = 6        # upper-triangle of a 3x3 symmetric matrix
CH_RAD: int = 1        # scalar distance to nearest skeleton voxel
CH_AVG: int = 3        # normalised centroid (z, y, x)
CH_EMB: int = 16       # discriminative embedding -- last so head[:, 16:] is emb

# Slice indices: ``[start, end)`` per field.  The embedding lives at
# the **end** of the channel dim so consumers that slice for the
# clusterer / TB embedding-projection panel can read ``head[:, 16:]``
# without threading the embedding width through every call.
RAW_SLICE: slice = slice(0, CH_RAW)                                    # [0, 1)
SEM_SLICE: slice = slice(RAW_SLICE.stop, RAW_SLICE.stop + CH_SEM)      # [1, 2)
SKL_SLICE: slice = slice(SEM_SLICE.stop, SEM_SLICE.stop + CH_SKL)      # [2, 3)
DIR_SLICE: slice = slice(SKL_SLICE.stop, SKL_SLICE.stop + CH_DIR)      # [3, 6)
COV_SLICE: slice = slice(DIR_SLICE.stop, DIR_SLICE.stop + CH_COV)      # [6, 12)
RAD_SLICE: slice = slice(COV_SLICE.stop, COV_SLICE.stop + CH_RAD)      # [12, 13)
AVG_SLICE: slice = slice(RAD_SLICE.stop, RAD_SLICE.stop + CH_AVG)      # [13, 16)
EMB_SLICE: slice = slice(AVG_SLICE.stop, AVG_SLICE.stop + CH_EMB)      # [16, 32)

HEAD_CHANNELS: int = EMB_SLICE.stop                                     # 32

# Activation policy: sigmoid is applied to the union of (sem, skl).  The
# two slots are contiguous by construction so the wrapper-side concat
# stays a single in/out split (see :func:`apply_head_activations`).
SIGMOID_SLICE: slice = slice(SEM_SLICE.start, SKL_SLICE.stop)           # [1, 3)

# Map from field name to slice.  Every consumer (loss, wrapper, TB
# callback, sliding-window inference) reaches into this dict instead of
# hard-coding indices.  Iteration order matches the channel order.
HEAD_LAYOUT: Dict[str, slice] = {
    "raw": RAW_SLICE,
    "sem": SEM_SLICE,
    "skl": SKL_SLICE,
    "dir": DIR_SLICE,
    "cov": COV_SLICE,
    "rad": RAD_SLICE,
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
    """Split a unified head tensor into the named fields.

    Args:
        head: ``[B, 32, *spatial]`` (or ``[B, *spatial, 32]`` if you set
            ``channel_dim``).
        channel_dim: Axis carrying the 32 channels.  Defaults to ``1``.

    Returns:
        Dict mapping ``"raw" / "sem" / "skl" / "dir" / "cov" / "rad" /
        "avg" / "emb"`` to the corresponding channel slice (a view of
        ``head``).
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
# Activation policy: sigmoid on sem + skl, linear elsewhere
# ---------------------------------------------------------------------------

def apply_head_activations(out: torch.Tensor) -> torch.Tensor:
    """Apply the head's activation policy to a raw head tensor.

    Splits ``out`` along the channel axis at :data:`SIGMOID_SLICE`,
    applies ``sigmoid`` to that contiguous run (``sem`` and ``skl``),
    and concatenates everything back together.  All other channels are
    returned linear.

    Wrappers call this exactly once at the end of their forward pass so
    the loss receives ``[0, 1]`` probabilities on the classification
    heads and raw logits everywhere else.

    Args:
        out: ``[B, HEAD_CHANNELS, *spatial]`` head tensor with no
            activations applied yet.

    Returns:
        Tensor of the same shape, with the slot at :data:`SIGMOID_SLICE`
        passed through ``sigmoid``.
    """
    return torch.cat(
        [
            out[:, :SIGMOID_SLICE.start],
            out[:, SIGMOID_SLICE].sigmoid(),
            out[:, SIGMOID_SLICE.stop:],
        ],
        dim=1,
    )


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


__all__ = [
    # Channel layout
    "CH_RAW", "CH_SEM", "CH_SKL", "CH_DIR", "CH_COV", "CH_RAD",
    "CH_AVG", "CH_EMB",
    "HEAD_CHANNELS", "HEAD_LAYOUT",
    "RAW_SLICE", "SEM_SLICE", "SKL_SLICE", "DIR_SLICE", "COV_SLICE",
    "RAD_SLICE", "AVG_SLICE", "EMB_SLICE",
    "SIGMOID_SLICE",
    "DIRECTIONS", "AFF_NAMES", "AFF_CHANNELS",
    # Helpers
    "slice_head",
    "apply_head_activations",
    "shift_replicate", "shift_replicate_np",
    "affinity_target", "affinity_target_np",
    "soft_aff_from_field",
    "upper_tri_to_matrix",
    "canonical_regression_name", "regression_loss_fn",
]
