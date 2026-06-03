"""
Shared helpers + canonical constants for the affinity + foreground head.

The model emits a single ``[B, HEAD_CHANNELS, *spatial]`` tensor::

    ch  0 .. N_AFF-1 : aff   (N_AFF)  sigmoid, per-offset affinity in (0, 1)
    ch  N_AFF        : sem   (1)       sigmoid, foreground / boundary prob
    ch  N_AFF + 1    : raw   (1)       linear,  L1 reconstruction of the
                                        (normalised) input EM intensity

The sigmoid is applied once by the wrapper via
:func:`apply_head_activations` to the contiguous ``aff + sem`` block
(:data:`SIGMOID_SLICE`); the trailing ``raw`` channel is left linear.

The affinities are predicted for a fixed list of 3-D voxel offsets
:data:`AFFINITY_OFFSETS` ``(dz, dy, dx)``.  The first :data:`N_ATTRACTIVE`
offsets are the nearest-neighbour **attractive** edges (z, y, x); the
remainder are long-range **repulsive** edges.  This is the edge set the
Mutex Watershed (Wolf et al. 2018, *The Mutex Watershed*, CVPR) consumes
at evaluation / inference time to agglomerate voxels into instances
(see :mod:`brainbow.inference.mutex_watershed`).

The default offset set is anisotropy-aware for EM: the long-range
repulsive edges reach much further in-plane (Y, X) than across sections
(Z), matching the typical 1:5 axial:lateral resolution of connectomics
volumes.

Affinity convention
-------------------
For a voxel ``v`` and offset ``o``, the affinity target is::

    aff[o, v] = 1  iff  label[v] == label[v + o]  (both foreground)
              = 0  otherwise

i.e. a **high** affinity means "merge" (same object).  This is the
``+`` (attractive) convention; the Mutex Watershed treats short-range
offsets as attractive and the long-range offsets as repulsive (a high
long-range affinity still means "same object", a low one is evidence of
a mutual-exclusion / boundary).

Public helpers
--------------
* :data:`AFFINITY_OFFSETS`, :data:`N_ATTRACTIVE`, :data:`N_AFF`,
  :data:`AFF_SLICE`, :data:`FG_SLICE`, :data:`HEAD_CHANNELS`,
  :data:`HEAD_LAYOUT` -- channel-layout constants.
* :func:`slice_head` -- split a head tensor into ``{"aff", "fg"}``.
* :func:`apply_head_activations` -- sigmoid the whole head once.
* :func:`shift_replicate` -- shift a tensor along an axis by ``+/- k``
  voxels with replicate (edge-pad) semantics.
* :func:`shift_nd` -- replicate-shift so that ``out[v] == x[v + offset]``
  for a 3-D ``(dz, dy, dx)`` offset (used by the affinity-target /
  validity-mask builders).
* :func:`affinity_target_from_offsets` -- build the ``[B, N_AFF, ...]``
  binary affinity target from instance labels.
* :func:`canonical_regression_name` / :func:`regression_loss_fn` --
  resolve user-facing names (``mse`` / ``l1`` / ``smooth_l1`` plus
  aliases) to a canonical string or ``F.*`` callable.
* :func:`stable_bce_on_probs` -- per-voxel BCE on already-sigmoided
  probabilities, with fp32-clamped log math safe under bf16-mixed
  autocast.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Channel layout (single source of truth)
# ---------------------------------------------------------------------------

# 3-D voxel offsets ``(dz, dy, dx)``.  The first ``N_ATTRACTIVE`` are the
# nearest-neighbour attractive edges; the rest are long-range repulsive
# edges.  Anisotropy-aware: the in-plane (Y, X) reach is much longer than
# the across-section (Z) reach, matching typical EM resolution (~1:5).
AFFINITY_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    # --- attractive nearest neighbours (z, y, x) ---
    (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    # --- repulsive long-range, in-plane ---
    (0, -3, 0), (0, 0, -3),
    (0, -9, 0), (0, 0, -9),
    (0, -27, 0), (0, 0, -27),
    (0, -9, -9), (0, 9, -9),
    # --- repulsive long-range, across sections (short, anisotropic) ---
    (-2, 0, 0), (-3, 0, 0), (-4, 0, 0),
)
N_ATTRACTIVE: int = 3
N_AFF: int = len(AFFINITY_OFFSETS)                                     # 14

# Slice indices: ``[start, end)`` per field.  Affinities first, then the
# scalar foreground (semantic) probability, then the linear
# raw-reconstruction channel last.  The two sigmoid fields (aff, sem) are
# contiguous at the front so :data:`SIGMOID_SLICE` is a single slice.
AFF_SLICE: slice = slice(0, N_AFF)                                     # [0, N_AFF)
SEM_SLICE: slice = slice(N_AFF, N_AFF + 1)                            # [N_AFF, N_AFF+1)
RAW_SLICE: slice = slice(SEM_SLICE.stop, SEM_SLICE.stop + 1)          # [N_AFF+1, N_AFF+2)

HEAD_CHANNELS: int = RAW_SLICE.stop                                    # N_AFF + 2

# Sigmoid is applied to the contiguous (aff, sem) block; the trailing
# ``raw`` regression channel is left linear.
SIGMOID_SLICE: slice = slice(0, SEM_SLICE.stop)                       # [0, N_AFF+1)

# Map from field name to slice.  Every consumer (loss, wrapper, TB
# callback, sliding-window inference) reaches into this dict instead of
# hard-coding indices.  Iteration order matches the channel order.
HEAD_LAYOUT: Dict[str, slice] = {
    "aff": AFF_SLICE,
    "sem": SEM_SLICE,
    "raw": RAW_SLICE,
}

# Offset names for TensorBoard panels / logging (``a`` = attractive nn,
# ``r`` = repulsive long-range), e.g. ``01_a_z``, ``04_r_y3``.
def _offset_name(idx: int, offset: Tuple[int, int, int]) -> str:
    kind = "a" if idx < N_ATTRACTIVE else "r"
    dz, dy, dx = offset
    axis = "z" if dz else ("y" if dy else "x")
    mag = abs(dz or dy or dx)
    return f"{idx + 1:02d}_{kind}_{axis}{mag}"


AFF_NAMES: Tuple[str, ...] = tuple(
    _offset_name(i, o) for i, o in enumerate(AFFINITY_OFFSETS)
)
AFF_CHANNELS: int = N_AFF


# ---------------------------------------------------------------------------
# Field slicing helpers
# ---------------------------------------------------------------------------

def slice_head(
    head: torch.Tensor,
    *,
    channel_dim: int = 1,
) -> Dict[str, torch.Tensor]:
    """Split an affinity + sem + raw head tensor into the named fields.

    Args:
        head: ``[B, HEAD_CHANNELS, *spatial]`` (or with the channel axis
            at ``channel_dim``).
        channel_dim: Axis carrying the channels.  Defaults to ``1``.

    Returns:
        Dict mapping ``"aff"`` -> ``[B, N_AFF, *spatial]``, ``"sem"`` ->
        ``[B, 1, *spatial]``, and ``"raw"`` -> ``[B, 1, *spatial]``
        (views of ``head``).
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
# Activation policy: sigmoid on every channel (all are probabilities)
# ---------------------------------------------------------------------------

def apply_head_activations(out: torch.Tensor) -> torch.Tensor:
    """Apply the head's activation policy to a raw head tensor.

    Sigmoid the contiguous (aff, fg) block (:data:`SIGMOID_SLICE`) so the
    loss / agglomerator receive ``(0, 1)`` probabilities; leave the
    trailing ``raw`` channel linear for the L1 reconstruction.  Wrappers
    call this exactly once at the end of their forward pass.

    Args:
        out: ``[B, HEAD_CHANNELS, *spatial]`` head logits.

    Returns:
        Tensor of the same shape, with :data:`SIGMOID_SLICE` passed
        through ``sigmoid`` and the rest left linear.
    """
    return torch.cat(
        [
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

    Args:
        x:     Any tensor.
        axis:  Axis to shift along (non-negative).
        shift: Voxel offset; positive or negative integer.
    """
    N = x.size(axis)
    k = abs(shift)
    if k == 0:
        return x

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


def shift_nd(
    x: torch.Tensor,
    offset: Sequence[int],
    spatial_axes: Tuple[int, int, int] = (1, 2, 3),
) -> torch.Tensor:
    """Replicate-shift ``x`` so that ``out[v] == x[v + offset]``.

    ``offset`` is a 3-D ``(dz, dy, dx)`` voxel displacement applied along
    ``spatial_axes`` (default ``(1, 2, 3)`` for a ``[B, D, H, W]`` label
    volume).  Implemented as per-axis :func:`shift_replicate` with the
    sign flipped (``shift_replicate`` produces ``out[v] = x[v - shift]``),
    so out-of-volume positions compare the voxel against the replicated
    edge -- the same convention the affinity target uses at the boundary.

    Args:
        x:            ``[B, D, H, W]`` (or any tensor whose ``spatial_axes``
                      are the three displaced axes).
        offset:       ``(dz, dy, dx)``.
        spatial_axes: The three axes ``offset`` displaces.

    Returns:
        Tensor of the same shape as ``x``.
    """
    out = x
    for axis, comp in zip(spatial_axes, offset):
        if comp != 0:
            out = shift_replicate(out, axis, -int(comp))
    return out


# ---------------------------------------------------------------------------
# Affinity targets
# ---------------------------------------------------------------------------

@torch.no_grad()
def affinity_target_from_offsets(
    labels: torch.Tensor,
    offsets: Sequence[Sequence[int]] = AFFINITY_OFFSETS,
    background: Optional[int] = None,
) -> torch.Tensor:
    """Build the binary affinity target from ``[B, D, H, W]`` labels.

    For every voxel ``v`` and offset ``o``::

        aff[o, v] = 1  if labels[v] == labels[v + o]
                  = 0  otherwise

    With replicate padding at the edge a voxel compares against itself,
    so foreground voxels at the volume boundary contribute ``aff = 1`` in
    the "missing" direction.

    Args:
        labels:  Integer instance ids of shape ``[B, D, H, W]``.
        offsets: Iterable of ``(dz, dy, dx)`` offsets (default
            :data:`AFFINITY_OFFSETS`).
        background: When set, voxels whose label equals this value are
            masked to ``0`` across all channels -- suppresses the spurious
            ``0 == 0 -> 1`` signal at background voxels.  ``None`` -> no
            masking.

    Returns:
        ``[B, len(offsets), D, H, W]`` float (0/1) tensor.
    """
    per_offset = [
        (labels == shift_nd(labels, offset)).to(torch.float32)
        for offset in offsets
    ]
    out = rearrange(torch.stack(per_offset, dim=0), "c b d h w -> b c d h w")
    if background is not None:
        mask = rearrange(
            (labels != background).to(torch.float32), "b ... -> b 1 ...",
        )
        out = out * mask
    return out


@torch.no_grad()
def affinity_validity_mask(
    fg: torch.Tensor,
    offsets: Sequence[Sequence[int]] = AFFINITY_OFFSETS,
) -> torch.Tensor:
    """Per-offset validity mask for affinity supervision.

    An affinity edge ``(v, v + o)`` is *valid* (supervised) iff both
    endpoints are foreground.  This drops the affinities between a
    foreground voxel and the background (and between two background
    voxels), so the loss only learns within-object / across-object
    relations.

    Args:
        fg:      ``[B, D, H, W]`` boolean / float foreground mask.
        offsets: Iterable of ``(dz, dy, dx)`` offsets.

    Returns:
        ``[B, len(offsets), D, H, W]`` float (0/1) mask.
    """
    fg_f = fg.to(torch.float32)
    per_offset = [fg_f * shift_nd(fg_f, offset) for offset in offsets]
    return rearrange(torch.stack(per_offset, dim=0), "c b d h w -> b c d h w")


# ---------------------------------------------------------------------------
# Regression-loss name resolver (kept for generic regression heads)
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
# Numerically-stable BCE on probabilities
# ---------------------------------------------------------------------------

def stable_bce_on_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Per-voxel binary cross-entropy on **probabilities** (not logits).

    The model wrapper applies a single sigmoid before the loss sees the
    head, so BCE here consumes ``[0, 1]`` probabilities directly.  The
    log math runs in fp32 with explicit clamping so ``log(p)`` and
    ``log(1 - p)`` stay finite under ``bf16-mixed`` autocast.

    Args:
        probs:  ``[B, C, *spatial]`` already-activated predictions in
            ``[0, 1]``.
        target: ``[B, C, *spatial]`` binary target (0 / 1 floats).
        eps:    Clamp bound for numerical stability.

    Returns:
        ``[B, C, *spatial]`` per-voxel BCE.  The caller reduces (mean,
        masked sum / valid_mask, ...) as it sees fit.
    """
    p = probs.float().clamp(eps, 1.0 - eps)
    t = target.float()
    return -(t * p.log() + (1.0 - t) * (1.0 - p).log())


__all__ = [
    # Channel layout
    "AFFINITY_OFFSETS", "N_ATTRACTIVE", "N_AFF",
    "AFF_SLICE", "SEM_SLICE", "RAW_SLICE", "SIGMOID_SLICE",
    "HEAD_CHANNELS", "HEAD_LAYOUT",
    "AFF_NAMES", "AFF_CHANNELS",
    # Helpers
    "slice_head",
    "apply_head_activations",
    "shift_replicate", "shift_replicate_np", "shift_nd",
    "affinity_target_from_offsets", "affinity_validity_mask",
    "canonical_regression_name", "regression_loss_fn",
    "stable_bce_on_probs",
]
