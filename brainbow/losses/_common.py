"""
Shared helpers + canonical constants for the affinity + foreground head.

The model emits a single ``[B, HEAD_CHANNELS, *spatial]`` tensor of
**raw logits / linear values** (no activation in the forward pass)::

    ch  0 .. N_AFF-1 : aff   (N_AFF)  logit, per-offset affinity
    ch  N_AFF        : sem   (1)       logit, foreground / boundary
    ch  N_AFF + 1    : raw   (1)       linear, L1 reconstruction of the
                                        (normalised) input EM intensity in
                                        ``[-1, 1]``

The head applies no activation: the loss consumes the ``aff`` / ``sem``
logits directly (logit-stable BCE + sigmoid for the Dice / focal terms),
and every other consumer (metrics, Mutex Watershed, TensorBoard) applies
``sigmoid`` at its own boundary.  The trailing ``raw`` channel is linear.

The affinities are predicted for a fixed list of 3-D voxel offsets
:data:`AFFINITY_OFFSETS` ``(dz, dy, dx)``.  The first :data:`N_PULL`
offsets are the nearest-neighbour **pull** edges (z, y, x);
the remainder are long-range **push** edges.  This is the edge set the
Mutex Watershed (Wolf et al. 2018, *The Mutex Watershed*, CVPR) consumes
at evaluation / inference time to agglomerate voxels into instances
(see :mod:`brainbow.inference.mutex_watershed`).

The default offset set is anisotropy-aware for EM: the long-range
push edges reach much further in-plane (Y, X) than across sections
(Z), matching the typical 1:5 axial:lateral resolution of connectomics
volumes.

Affinity convention
-------------------
For a voxel ``v`` and offset ``o``, the affinity target is::

    aff[o, v] = 1  iff  label[v] == label[v + o]  (both foreground)
              = 0  otherwise

i.e. a **high** affinity means "merge" (same object).  This is the
``+`` (pull) convention; the Mutex Watershed treats short-range
offsets as pull and the long-range offsets as push (a high
long-range affinity still means "same object", a low one is evidence of
a mutual-exclusion / boundary).

Public helpers
--------------
* :data:`AFFINITY_OFFSETS`, :data:`N_PULL`, :data:`N_AFF`,
  :data:`AFF_SLICE`, :data:`SEM_SLICE`, :data:`RAW_SLICE`,
  :data:`HEAD_CHANNELS`, :data:`HEAD_LAYOUT` -- channel-layout constants.
* :func:`slice_head` -- split a head tensor into ``{"aff", "sem", "raw"}``.
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

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel layout (single source of truth)
# ---------------------------------------------------------------------------

# 3-D voxel offsets ``(dz, dy, dx)``.  The first ``N_PULL`` are the
# nearest-neighbour pull edges; the rest are long-range push edges.
# Anisotropy-aware: the in-plane (Y, X) reach is much longer than the
# across-section (Z) reach, matching EM resolution (~1:5).
AFFINITY_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    # --- pull nearest neighbours (z, y, x) ---
    (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    # --- push long-range, in-plane ---
    (0, -3, 0), (0, 0, -3),
    (0, -9, 0), (0, 0, -9),
    (0, -27, 0), (0, 0, -27),
    (0, -9, -9), (0, 9, -9),
    # --- push long-range, across sections (short, anisotropic) ---
    (-2, 0, 0), (-3, 0, 0), (-4, 0, 0),
)
N_PULL: int = 3
N_AFF: int = len(AFFINITY_OFFSETS)                                     # 14

# Slice indices: ``[start, end)`` per field.  Affinities first, then the
# scalar foreground (semantic) logit, then the linear raw-reconstruction
# channel last.  The head emits raw logits / linear values; each consumer
# applies its own activation (logit BCE in the loss, sigmoid for metrics /
# MWS / TensorBoard, linear for the ``raw`` channel).
AFF_SLICE: slice = slice(0, N_AFF)                                     # [0, N_AFF)
SEM_SLICE: slice = slice(N_AFF, N_AFF + 1)                            # [N_AFF, N_AFF+1)
RAW_SLICE: slice = slice(SEM_SLICE.stop, SEM_SLICE.stop + 1)          # [N_AFF+1, N_AFF+2)

HEAD_CHANNELS: int = RAW_SLICE.stop                                    # N_AFF + 2

# Map from field name to slice.  Every consumer (loss, wrapper, TB
# callback, sliding-window inference) reaches into this dict instead of
# hard-coding indices.  Iteration order matches the channel order.
HEAD_LAYOUT: Dict[str, slice] = {
    "aff": AFF_SLICE,
    "sem": SEM_SLICE,
    "raw": RAW_SLICE,
}

# Offset names for TensorBoard panels / logging (``pull`` = nearest
# neighbour, ``push`` = long-range), e.g. ``01_pull_z1``, ``04_push_y3``.
def _offset_name(idx: int, offset: Tuple[int, int, int]) -> str:
    kind = "pull" if idx < N_PULL else "push"
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
        ``[B, len(offsets), D, H, W]`` ``uint8`` (0/1) tensor.  ``uint8``
        (not ``float32``) keeps this dense ``N_AFF``-channel target cheap
        -- ~4x smaller -- since it is cached for the whole step; the loss
        casts the slices it needs to float on the fly.
    """
    n = len(offsets)
    out = labels.new_zeros((labels.shape[0], n, *labels.shape[1:]),
                           dtype=torch.uint8)
    # Write each offset in place (no 14-element ``torch.stack`` peak).
    for c, offset in enumerate(offsets):
        out[:, c] = (labels == shift_nd(labels, offset)).to(torch.uint8)
    if background is not None:
        fg = (labels != background).unsqueeze(1).to(torch.uint8)
        out *= fg
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
        ``[B, len(offsets), D, H, W]`` ``uint8`` (0/1) mask (cheap to
        cache; the loss casts to float on the fly).
    """
    fg_b = fg.to(torch.bool)
    n = len(offsets)
    out = fg_b.new_zeros((fg_b.shape[0], n, *fg_b.shape[1:]), dtype=torch.uint8)
    for c, offset in enumerate(offsets):
        out[:, c] = (fg_b & shift_nd(fg_b, offset)).to(torch.uint8)
    return out


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

    Retained as a utility for callers that already hold ``[0, 1]``
    probabilities (the head itself now emits logits -- prefer
    :func:`torch.nn.functional.binary_cross_entropy_with_logits` on the
    raw head).  The log math runs in fp32 with explicit clamping so
    ``log(p)`` and ``log(1 - p)`` stay finite under ``bf16-mixed`` autocast.

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
    "AFFINITY_OFFSETS", "N_PULL", "N_AFF",
    "AFF_SLICE", "SEM_SLICE", "RAW_SLICE",
    "HEAD_CHANNELS", "HEAD_LAYOUT",
    "AFF_NAMES", "AFF_CHANNELS",
    # Helpers
    "slice_head",
    "shift_replicate", "shift_nd",
    "affinity_target_from_offsets", "affinity_validity_mask",
    "canonical_regression_name", "regression_loss_fn",
    "stable_bce_on_probs",
]
