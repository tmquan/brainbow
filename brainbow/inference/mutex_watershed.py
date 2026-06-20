"""
Mutex Watershed agglomeration (Wolf et al. 2018).

The Mutex Watershed turns a set of per-voxel **affinities** into an
instance segmentation with *no* free threshold / seed parameters.  It is
the evaluation / inference counterpart of the affinity head supervised
by :class:`brainbow.losses.AffinityFGLoss`.

Algorithm (single pass, Kruskal-style with mutual-exclusion constraints):

* Every offset ``o`` and voxel ``v`` define an edge ``(v, v + o)``.
* Short-range (the first ``n_pull``) offsets are **pull**:
  the edge priority is the affinity ``a`` (high ``a`` -> strong "merge").
* Long-range offsets are **push**: the edge priority is ``1 - a``
  (low affinity -> strong "must separate"); these are *mutex* edges.
* Process all edges in descending priority order with a union-find:
  - an pull edge merges its two clusters **unless** they are
    already linked by an active mutex;
  - a push edge adds a mutex between its two clusters **unless**
    they are already merged.

Reference:
    S. Wolf, C. Pape, A. Bailoni, et al. "The Mutex Watershed: Efficient,
    Parameter-Free Image Partitioning." ECCV/CVPR 2018.

No external dependency (``affogato`` / ``elf``) is required -- the core
loop is JIT-compiled with numba over flat numpy arrays.  The mutex
constraints are stored as per-root singly-linked lists in flat int64
arrays (O(1) splice on union), so the whole pass stays in nopython mode.

Cost note: the edge count is ``~n_pull * |fg| + (push
edges)``.  For large EM crops keep the push ``strides`` coarse
(default ``(1, 4, 4)`` -- full in Z, every 4th in-plane) and use
``size_filter`` to drop specks; both are throughput levers, not accuracy
knobs in the usual regime.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep in this env
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return _wrap


try:
    import cupy as _cp

    _HAVE_CUPY = True
except Exception:  # pragma: no cover - cupy is optional
    _cp = None
    _HAVE_CUPY = False


# ---------------------------------------------------------------------------
# numba core
# ---------------------------------------------------------------------------

@njit(cache=True)
def _find(parent: np.ndarray, x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True)
def _mws_core(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_mutex: np.ndarray,
    order: np.ndarray,
    n_nodes: int,
    n_mutex_edges: int,
) -> np.ndarray:
    """Run the union-find + mutex pass; return the parent array.

    Mutex partners are stored as per-root singly-linked lists in the
    flat arrays ``link_next`` / ``link_to`` (with ``head`` / ``tail`` /
    ``count`` per node), spliced in O(1) on union.  A partner is stored
    as a *node* id and resolved with :func:`_find` at query time, so the
    structure never needs to migrate stale representative ids.
    """
    parent = np.arange(n_nodes)
    rank = np.zeros(n_nodes, dtype=np.int64)

    head = np.full(n_nodes, -1, dtype=np.int64)
    tail = np.full(n_nodes, -1, dtype=np.int64)
    count = np.zeros(n_nodes, dtype=np.int64)

    cap = 2 * n_mutex_edges + 1
    link_next = np.full(cap, -1, dtype=np.int64)
    link_to = np.full(cap, -1, dtype=np.int64)
    n_store = 0

    for idx in range(order.shape[0]):
        e = order[idx]
        u = edge_u[e]
        v = edge_v[e]
        ru = _find(parent, u)
        rv = _find(parent, v)
        if ru == rv:
            continue

        # Walk the smaller cluster's mutex chain to test for a constraint.
        if count[ru] <= count[rv]:
            a = ru
            b = rv
        else:
            a = rv
            b = ru
        blocked = False
        p = head[a]
        while p != -1:
            if _find(parent, link_to[p]) == b:
                blocked = True
                break
            p = link_next[p]
        if blocked:
            continue

        if edge_mutex[e]:
            # Store v on ru's chain and u on rv's chain.
            link_to[n_store] = v
            link_next[n_store] = -1
            if head[ru] == -1:
                head[ru] = n_store
            else:
                link_next[tail[ru]] = n_store
            tail[ru] = n_store
            count[ru] += 1
            n_store += 1

            link_to[n_store] = u
            link_next[n_store] = -1
            if head[rv] == -1:
                head[rv] = n_store
            else:
                link_next[tail[rv]] = n_store
            tail[rv] = n_store
            count[rv] += 1
            n_store += 1
        else:
            # Union by rank; splice rv's mutex chain into ru.
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            if head[rv] != -1:
                if head[ru] == -1:
                    head[ru] = head[rv]
                    tail[ru] = tail[rv]
                else:
                    link_next[tail[ru]] = head[rv]
                    tail[ru] = tail[rv]
                count[ru] += count[rv]
                head[rv] = -1
                tail[rv] = -1
                count[rv] = 0

    return parent


# ---------------------------------------------------------------------------
# Edge construction (numpy)
# ---------------------------------------------------------------------------

def _axis_slices(comp: int, n: int) -> Tuple[slice, slice]:
    """Source / target slices along one axis for offset component ``comp``.

    The pair is ``(v, v + comp)``: ``src`` indexes ``v``, ``tgt`` indexes
    ``v + comp``, both clipped to the valid in-bounds region.
    """
    if comp >= 0:
        return slice(0, n - comp), slice(comp, n)
    return slice(-comp, n), slice(0, n + comp)


def _build_edges(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    n_pull: int,
    strides: Sequence[int],
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build flat edge arrays ``(u, v, weight, is_mutex)`` + #mutex edges.

    ``u`` / ``v`` are flat voxel indices into the ``[D, H, W]`` grid.
    Pull edges use ``weight = affinity``; push edges use
    ``weight = 1 - affinity`` and are flagged as mutex.  Push edges
    are subsampled by ``strides`` (per-axis) to bound the edge count.
    """
    D, H, W = affinities.shape[1], affinities.shape[2], affinities.shape[3]
    grid = np.arange(D * H * W, dtype=np.int64).reshape(D, H, W)
    sz, sy, sx = (int(s) for s in strides)

    us: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    ms: List[np.ndarray] = []

    for io, offset in enumerate(offsets):
        dz, dy, dx = (int(c) for c in offset)
        zs_s, zs_t = _axis_slices(dz, D)
        ys_s, ys_t = _axis_slices(dy, H)
        xs_s, xs_t = _axis_slices(dx, W)

        aff_o = affinities[io]
        pull = io < n_pull

        u_src = grid[zs_s, ys_s, xs_s]
        v_tgt = grid[zs_t, ys_t, xs_t]
        w = aff_o[zs_s, ys_s, xs_s]

        if not pull:
            # Subsample push edges to keep the graph tractable.
            u_src = u_src[::sz, ::sy, ::sx]
            v_tgt = v_tgt[::sz, ::sy, ::sx]
            w = w[::sz, ::sy, ::sx]
            w = 1.0 - w

        u_flat = u_src.reshape(-1)
        v_flat = v_tgt.reshape(-1)
        w_flat = w.reshape(-1).astype(np.float64)

        if mask is not None:
            keep = mask.reshape(-1)[u_flat] & mask.reshape(-1)[v_flat]
            u_flat = u_flat[keep]
            v_flat = v_flat[keep]
            w_flat = w_flat[keep]

        if u_flat.size == 0:
            continue

        us.append(u_flat)
        vs.append(v_flat)
        ws.append(w_flat)
        ms.append(
            np.zeros(u_flat.size, dtype=np.bool_)
            if pull
            else np.ones(u_flat.size, dtype=np.bool_)
        )

    if not us:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0), np.empty(0, dtype=np.bool_), 0

    edge_u = np.concatenate(us)
    edge_v = np.concatenate(vs)
    edge_w = np.concatenate(ws)
    edge_m = np.concatenate(ms)
    n_mutex = int(edge_m.sum())
    return edge_u, edge_v, edge_w, edge_m, n_mutex


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------

def mutex_watershed(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    n_pull: int,
    strides: Sequence[int] = (1, 4, 4),
    mask: Optional[np.ndarray] = None,
    size_filter: int = 0,
    max_push_edges: Optional[int] = None,
) -> np.ndarray:
    """Mutex Watershed segmentation of an affinity volume (numpy / numba).

    This is the exact, sequential reference implementation (``mws_np``).

    Args:
        affinities: ``[n_offsets, D, H, W]`` float array in ``[0, 1]``;
            ``affinities[o, v]`` is ``P(label[v] == label[v + offset_o])``.
        offsets: ``(dz, dy, dx)`` per channel; ``len == n_offsets``.
        n_pull: Number of leading offsets that are pull
            (the rest are push / mutex).
        strides: Per-axis subsampling of the push edges (pull
            edges are always dense).
        mask: Optional ``[D, H, W]`` foreground mask; voxels outside it
            become background (label 0) and contribute no edges.
        size_filter: Connected components with fewer than this many
            voxels are reset to background (0).  ``0`` disables.
        max_push_edges: Hard cap on push (mutex) edges; when exceeded, only
            the top-``max_push_edges`` by separation confidence
            (``1 - affinity``) are kept (all pull edges retained).  Bounds
            the mutex bookkeeping that explodes on near-random affinities.
            ``None`` (or ``0``) means **no cap** -- run on the full edge
            set.

    Returns:
        ``[D, H, W]`` ``int64`` label volume (0 = background,
        ``1..K`` = instances, relabelled consecutively).
    """
    affinities = np.ascontiguousarray(affinities, dtype=np.float32)
    if affinities.ndim != 4:
        raise ValueError(
            f"mutex_watershed expects [n_offsets, D, H, W]; got {affinities.shape}."
        )
    if affinities.shape[0] != len(offsets):
        raise ValueError(
            f"affinities has {affinities.shape[0]} channels but {len(offsets)} "
            f"offsets were given."
        )
    _, D, H, W = affinities.shape
    n_nodes = D * H * W

    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=bool)

    edge_u, edge_v, edge_w, edge_m, n_mutex = _build_edges(
        affinities, offsets, n_pull, strides, mask,
    )

    if max_push_edges and n_mutex > int(max_push_edges):
        k = int(max_push_edges)
        push_idx = np.flatnonzero(edge_m)
        pull_idx = np.flatnonzero(~edge_m)
        top = np.argpartition(edge_w[push_idx], push_idx.size - k)[-k:]
        keep = np.concatenate([pull_idx, push_idx[top]])
        keep.sort()
        edge_u, edge_v, edge_w, edge_m = (
            edge_u[keep], edge_v[keep], edge_w[keep], edge_m[keep],
        )
        n_mutex = int(edge_m.sum())

    labels = np.zeros(n_nodes, dtype=np.int64)
    if edge_u.size > 0:
        # Descending priority (stable so equal weights keep offset order).
        order = np.argsort(-edge_w, kind="stable")
        parent = _mws_core(
            edge_u, edge_v, edge_m, order, int(n_nodes), int(n_mutex),
        )
        # Resolve every node to its root.
        roots = np.array([_find(parent, i) for i in range(n_nodes)], dtype=np.int64) \
            if not _HAVE_NUMBA else _resolve_all(parent)
    else:
        roots = np.arange(n_nodes, dtype=np.int64)

    seg = roots.reshape(D, H, W)

    if mask is not None:
        seg = np.where(mask, seg, -1)

    out = _relabel_consecutive(seg, ignore=-1)

    if size_filter > 0:
        out = _apply_size_filter(out, size_filter)
    return out


@njit(cache=True)
def _resolve_all(parent: np.ndarray) -> np.ndarray:
    n = parent.shape[0]
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = _find(parent, i)
    return roots


def _relabel_consecutive(seg: np.ndarray, ignore: int = -1) -> np.ndarray:
    """Relabel ``seg`` to ``0`` (ignored) + consecutive ``1..K``."""
    flat = seg.reshape(-1)
    fg = flat != ignore
    out = np.zeros_like(flat, dtype=np.int64)
    if fg.any():
        uniq, inv = np.unique(flat[fg], return_inverse=True)
        out[fg] = inv + 1
    return out.reshape(seg.shape)


def grow_labels_to_fill(
    seg: np.ndarray,
    max_distance: float = 0.0,
    sampling: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Grow instance labels into the unlabeled (background) band.

    The Mutex Watershed leaves voxels outside the foreground mask as
    background (``0``).  When that mask is the predicted ``sem`` head, the
    blurry / boundary-eroded membrane band is a **thick** unlabeled rind
    between segments.  This reassigns each background voxel the id of its
    nearest labeled voxel (via a Euclidean distance transform), so adjacent
    segments grow until they meet at a thin (~1-voxel) border.

    Args:
        seg: ``[D, H, W]`` (or any ND) integer label volume; ``0`` =
            background / unlabeled.
        max_distance: Only fill background voxels within this distance of a
            labeled voxel, keeping large true-background cores at ``0``.
            ``<= 0`` fills every background voxel (full watershed-style).
        sampling: Per-axis voxel spacing for an anisotropy-aware grow
            (e.g. ``pixel_size``); ``None`` = isotropic voxels.

    Returns:
        Label volume with the boundary band filled (same shape).
    """
    bg = seg == 0
    if not bg.any() or not (seg != 0).any():
        return seg
    from scipy.ndimage import distance_transform_edt

    # EDT of the background mask: distance (and nearest-labeled-voxel
    # indices) to the closest non-background voxel.  ``seg[tuple(inds)]``
    # then carries each labeled id outward into the band.
    dist, inds = distance_transform_edt(
        bg, sampling=sampling, return_indices=True,
    )
    filled = seg[tuple(inds)]
    if max_distance and max_distance > 0:
        keep_bg = bg & (dist > float(max_distance))
        filled = np.where(keep_bg, 0, filled)
    return filled


def _apply_size_filter(seg: np.ndarray, min_size: int) -> np.ndarray:
    """Reset instances smaller than ``min_size`` voxels to background."""
    flat = seg.reshape(-1)
    counts = np.bincount(flat)
    small = np.where(counts < min_size)[0]
    if small.size:
        small_set = small[small != 0]
        if small_set.size:
            kill = np.isin(flat, small_set)
            flat = flat.copy()
            flat[kill] = 0
            seg = _relabel_consecutive(
                np.where(flat == 0, -1, flat).reshape(seg.shape), ignore=-1,
            )
    return seg


# ---------------------------------------------------------------------------
# cupy GPU path (mws_cp): parallel Boruvka with priority buckets
# ---------------------------------------------------------------------------

def _cp_find_roots(parent):  # pragma: no cover - requires GPU
    """Pointer-jumping union-find: return the root of every node."""
    r = parent
    while True:
        nr = r[r]
        if bool((nr == r).all()):
            return nr
        r = nr


def mws_cp(
    affinities,
    offsets: Sequence[Sequence[int]],
    n_pull: int,
    strides: Sequence[int] = (1, 4, 4),
    mask=None,
    size_filter: int = 0,
    max_push_edges: Optional[int] = None,
    buckets: int = 16,
):  # pragma: no cover - requires GPU
    """GPU Mutex Watershed (cupy) -- parallel Boruvka with priority buckets.

    An approximation of the exact sequential :func:`mutex_watershed`
    (``mws_np``): edges are processed in ``buckets`` descending-priority
    groups, and within each bucket clusters merge in parallel Boruvka
    rounds (highest-priority pull edge per cluster, hooked higher-root ->
    lower-root so the union-find stays acyclic), with push edges installed
    as mutex constraints that block merges.  Everything runs on the GPU via
    cupy array ops -- no host transfer, no CUDA kernels, no JIT.

    Args:
        affinities: cupy ``[n_offsets, D, H, W]`` float array in ``[0, 1]``
            (typically a zero-copy view of the model's CUDA tensor).
        offsets / n_pull / strides / mask / size_filter / max_push_edges:
            as in :func:`mutex_watershed`.  ``max_push_edges=None`` -> no
            cap (full edges).
        buckets: number of descending-priority buckets (higher = closer to
            the exact sequential ordering, more rounds).

    Returns:
        cupy ``[D, H, W]`` ``int64`` label volume (0 = background).
    """
    if not _HAVE_CUPY:
        raise RuntimeError("mws_cp requires cupy, which is not available.")
    cp = _cp
    aff = cp.ascontiguousarray(affinities, dtype=cp.float32)
    n_off, D, H, W = aff.shape
    if n_off != len(offsets):
        raise ValueError(
            f"affinities has {n_off} channels but {len(offsets)} offsets."
        )
    n_vox = D * H * W
    grid = cp.arange(n_vox, dtype=cp.int64).reshape(D, H, W)

    if mask is None:
        fg_flat = cp.ones(n_vox, dtype=cp.bool_)
    else:
        fg_flat = cp.ascontiguousarray(mask, dtype=cp.bool_).reshape(-1)
    M = int(fg_flat.sum())
    if M == 0:
        return cp.zeros((D, H, W), dtype=cp.int64)
    node_of_voxel = cp.full(n_vox, -1, dtype=cp.int64)
    node_of_voxel[fg_flat] = cp.arange(M, dtype=cp.int64)

    sz, sy, sx = (int(s) for s in strides)
    us, vs, ws, ps = [], [], [], []
    for io, offset in enumerate(offsets):
        dz, dy, dx = (int(c) for c in offset)
        zs_s, zs_t = _axis_slices(dz, D)
        ys_s, ys_t = _axis_slices(dy, H)
        xs_s, xs_t = _axis_slices(dx, W)
        u_src = grid[zs_s, ys_s, xs_s]
        v_tgt = grid[zs_t, ys_t, xs_t]
        w = aff[io][zs_s, ys_s, xs_s]
        pull = io < n_pull
        if not pull:
            u_src = u_src[::sz, ::sy, ::sx]
            v_tgt = v_tgt[::sz, ::sy, ::sx]
            w = 1.0 - w[::sz, ::sy, ::sx]
        u = node_of_voxel[u_src.reshape(-1)]
        v = node_of_voxel[v_tgt.reshape(-1)]
        w = w.reshape(-1).astype(cp.float32)
        keep = (u >= 0) & (v >= 0)
        if not bool(keep.any()):
            continue
        us.append(u[keep]); vs.append(v[keep]); ws.append(w[keep])
        ps.append(cp.zeros(int(keep.sum()), dtype=cp.bool_) if pull
                  else cp.ones(int(keep.sum()), dtype=cp.bool_))

    if not us:
        return cp.zeros((D, H, W), dtype=cp.int64)
    eu = cp.concatenate(us); ev = cp.concatenate(vs)
    ew = cp.concatenate(ws); ep = cp.concatenate(ps)

    if max_push_edges:
        n_push = int(ep.sum())
        if n_push > int(max_push_edges):
            k = int(max_push_edges)
            push_idx = cp.flatnonzero(ep)
            pull_idx = cp.flatnonzero(~ep)
            top = cp.argpartition(ew[push_idx], push_idx.size - k)[-k:]
            keep = cp.concatenate([pull_idx, push_idx[top]])
            keep.sort()
            eu, ev, ew, ep = eu[keep], ev[keep], ew[keep], ep[keep]

    parent = cp.arange(M, dtype=cp.int64)
    mutex_keys = cp.empty(0, dtype=cp.int64)            # sorted canonical keys

    order = cp.argsort(-ew)                              # descending priority
    E = order.size
    bounds = cp.linspace(0, E, int(buckets) + 1).astype(cp.int64)
    bounds = [int(x) for x in cp.asnumpy(bounds)]

    for bi in range(len(bounds) - 1):
        lo, hi = bounds[bi], bounds[bi + 1]
        if hi <= lo:
            continue
        idx = order[lo:hi]
        bu, bv = eu[idx], ev[idx]
        bpush = ep[idx]
        while True:
            roots = _cp_find_roots(parent)
            # Contract: remap stored mutex keys through the *current* roots
            # (after merges a mutex (X,Z) must follow X -> find(X)).  Without
            # this the keys go stale and stop blocking -> over-merge.
            if mutex_keys.size:
                ra = roots[mutex_keys // M]; rb = roots[mutex_keys % M]
                lo = cp.minimum(ra, rb); hi = cp.maximum(ra, rb)
                keep = lo != hi
                mutex_keys = cp.unique(lo[keep] * M + hi[keep])
            ru, rv = roots[bu], roots[bv]
            alive = ru != rv
            # 1) install mutexes from alive push edges in this bucket
            pm = bpush & alive
            if bool(pm.any()):
                a = cp.minimum(ru[pm], rv[pm]); b = cp.maximum(ru[pm], rv[pm])
                keys = a * M + b
                mutex_keys = cp.union1d(mutex_keys, keys)   # sorted unique
            # 2) propose pull merges not blocked by a mutex
            pull_alive = (~bpush) & alive
            if not bool(pull_alive.any()):
                break
            ci = cp.minimum(ru, rv); cj = cp.maximum(ru, rv)
            keys = ci * M + cj
            ins = cp.searchsorted(mutex_keys, keys)
            ins = cp.clip(ins, 0, mutex_keys.size - 1) if mutex_keys.size else ins
            blocked = (mutex_keys[ins] == keys) if mutex_keys.size else cp.zeros_like(keys, dtype=cp.bool_)
            cand = pull_alive & (~blocked)
            if not bool(cand.any()):
                break
            ci_c = ci[cand]; cj_c = cj[cand]; w_c = ew[idx][cand]
            # best edge per higher-root cj: hook cj -> ci (acyclic: cj > ci)
            o2 = cp.lexsort(cp.stack([-w_c, cj_c]))      # primary cj_c, then w desc
            cj_s = cj_c[o2]
            first = cp.ones(cj_s.size, dtype=cp.bool_)
            first[1:] = cj_s[1:] != cj_s[:-1]
            win = o2[first]
            src = cj_c[win]; dst = ci_c[win]
            if not bool((src != dst).any()):
                break
            parent[src] = dst
        # next bucket

    roots = _cp_find_roots(parent)
    seg_flat = cp.full(n_vox, -1, dtype=cp.int64)
    seg_flat[fg_flat] = roots
    # relabel consecutive (0 = background / outside mask)
    out = cp.zeros(n_vox, dtype=cp.int64)
    fgm = seg_flat != -1
    if bool(fgm.any()):
        uniq, inv = cp.unique(seg_flat[fgm], return_inverse=True)
        out[fgm] = inv + 1
    seg = out.reshape(D, H, W)

    if size_filter and size_filter > 0:
        flat = seg.reshape(-1)
        counts = cp.bincount(flat)
        small = cp.flatnonzero(counts < int(size_filter))
        small = small[small != 0]
        if small.size:
            kill = cp.isin(flat, small)
            flat = flat.copy(); flat[kill] = 0
            fg2 = flat != 0
            out2 = cp.zeros_like(flat)
            if bool(fg2.any()):
                uq, iv = cp.unique(flat[fg2], return_inverse=True)
                out2[fg2] = iv + 1
            seg = out2.reshape(D, H, W)
    return seg


# ---------------------------------------------------------------------------
# torch GPU path (mws_th): parallel Boruvka, native zero-copy on the tensor
# ---------------------------------------------------------------------------

def _th_find_roots(parent: torch.Tensor) -> torch.Tensor:
    """Pointer-jumping union-find: return the root of every node."""
    r = parent
    while True:
        nr = r[r]
        if bool((nr == r).all()):
            return nr
        r = nr


@torch.no_grad()
def mws_th(
    affinities: torch.Tensor,
    offsets: Sequence[Sequence[int]],
    n_pull: int,
    strides: Sequence[int] = (1, 4, 4),
    mask: Optional[torch.Tensor] = None,
    size_filter: int = 0,
    max_push_edges: Optional[int] = None,
    buckets: int = 16,
) -> torch.Tensor:
    """GPU Mutex Watershed (torch) -- parallel Boruvka with priority buckets.

    Same algorithm as :func:`mws_cp`, but pure torch ops operating directly
    on the input CUDA tensor -- truly zero-copy (no DLPack, no host
    transfer, no cupy dependency).  An approximation of the exact
    :func:`mutex_watershed` (``mws_np``); ``max_push_edges=None`` -> no cap
    (full edges).

    Args:
        affinities: torch ``[n_offsets, D, H, W]`` float tensor in ``[0, 1]``
            on any device (the model's CUDA tensor, used in place).
        offsets / n_pull / strides / mask / size_filter / max_push_edges /
            buckets: as in :func:`mws_cp`.

    Returns:
        torch ``[D, H, W]`` ``int64`` label tensor on ``affinities.device``.
    """
    dev = affinities.device
    aff = affinities.detach().to(torch.float32)
    n_off, D, H, W = aff.shape
    if n_off != len(offsets):
        raise ValueError(
            f"affinities has {n_off} channels but {len(offsets)} offsets."
        )
    n_vox = D * H * W
    grid = torch.arange(n_vox, dtype=torch.int64, device=dev).view(D, H, W)

    if mask is None:
        fg_flat = torch.ones(n_vox, dtype=torch.bool, device=dev)
    else:
        fg_flat = mask.reshape(-1).to(torch.bool)
    M = int(fg_flat.sum())
    if M == 0:
        return torch.zeros((D, H, W), dtype=torch.int64, device=dev)
    node_of_voxel = torch.full((n_vox,), -1, dtype=torch.int64, device=dev)
    node_of_voxel[fg_flat] = torch.arange(M, dtype=torch.int64, device=dev)

    sz, sy, sx = (int(s) for s in strides)
    us, vs, ws, ps = [], [], [], []
    for io, offset in enumerate(offsets):
        dz, dy, dx = (int(c) for c in offset)
        zs_s, zs_t = _axis_slices(dz, D)
        ys_s, ys_t = _axis_slices(dy, H)
        xs_s, xs_t = _axis_slices(dx, W)
        u_src = grid[zs_s, ys_s, xs_s]
        v_tgt = grid[zs_t, ys_t, xs_t]
        w = aff[io][zs_s, ys_s, xs_s]
        pull = io < n_pull
        if not pull:
            u_src = u_src[::sz, ::sy, ::sx]
            v_tgt = v_tgt[::sz, ::sy, ::sx]
            w = 1.0 - w[::sz, ::sy, ::sx]
        u = node_of_voxel[u_src.reshape(-1)]
        v = node_of_voxel[v_tgt.reshape(-1)]
        w = w.reshape(-1).to(torch.float32)
        keep = (u >= 0) & (v >= 0)
        if not bool(keep.any()):
            continue
        us.append(u[keep]); vs.append(v[keep]); ws.append(w[keep])
        ps.append(
            torch.zeros(int(keep.sum()), dtype=torch.bool, device=dev) if pull
            else torch.ones(int(keep.sum()), dtype=torch.bool, device=dev)
        )

    if not us:
        return torch.zeros((D, H, W), dtype=torch.int64, device=dev)
    eu = torch.cat(us); ev = torch.cat(vs)
    ew = torch.cat(ws); ep = torch.cat(ps)

    if max_push_edges:
        n_push = int(ep.sum())
        if n_push > int(max_push_edges):
            k = int(max_push_edges)
            push_idx = ep.nonzero(as_tuple=True)[0]
            pull_idx = (~ep).nonzero(as_tuple=True)[0]
            top = torch.topk(ew[push_idx], k).indices
            keep = torch.cat([pull_idx, push_idx[top]])
            keep, _ = torch.sort(keep)
            eu, ev, ew, ep = eu[keep], ev[keep], ew[keep], ep[keep]

    parent = torch.arange(M, dtype=torch.int64, device=dev)
    mutex_keys = torch.empty(0, dtype=torch.int64, device=dev)  # sorted unique

    order = torch.argsort(-ew)                                   # descending
    E = order.numel()
    bounds = [int(x) for x in torch.linspace(0, E, int(buckets) + 1).tolist()]

    for bi in range(len(bounds) - 1):
        lo, hi = bounds[bi], bounds[bi + 1]
        if hi <= lo:
            continue
        idx = order[lo:hi]
        bu, bv = eu[idx], ev[idx]
        bpush = ep[idx]
        bw = ew[idx]
        while True:
            roots = _th_find_roots(parent)
            # Contract: remap stored mutex keys through current roots so a
            # merge X->Y carries mutex (X,Z) to (Y,Z); else keys go stale.
            if mutex_keys.numel():
                ra = roots[mutex_keys // M]; rb = roots[mutex_keys % M]
                klo = torch.minimum(ra, rb); khi = torch.maximum(ra, rb)
                keep = klo != khi
                mutex_keys = torch.unique(klo[keep] * M + khi[keep])
            ru, rv = roots[bu], roots[bv]
            alive = ru != rv
            # 1) install mutexes from alive push edges in this bucket
            pm = bpush & alive
            if bool(pm.any()):
                a = torch.minimum(ru[pm], rv[pm]); b = torch.maximum(ru[pm], rv[pm])
                mutex_keys = torch.unique(torch.cat([mutex_keys, a * M + b]))
            # 2) propose pull merges not blocked by a mutex
            pull_alive = (~bpush) & alive
            if not bool(pull_alive.any()):
                break
            ci = torch.minimum(ru, rv); cj = torch.maximum(ru, rv)
            keys = ci * M + cj
            if mutex_keys.numel():
                ins = torch.searchsorted(mutex_keys, keys).clamp_max(
                    mutex_keys.numel() - 1
                )
                blocked = mutex_keys[ins] == keys
            else:
                blocked = torch.zeros_like(keys, dtype=torch.bool)
            cand = pull_alive & (~blocked)
            if not bool(cand.any()):
                break
            ci_c = ci[cand]; cj_c = cj[cand]; w_c = bw[cand]
            # best pull edge per higher-root cj; hook cj -> ci (acyclic: cj>ci)
            bestw = torch.full((M,), float("-inf"), device=dev, dtype=torch.float32)
            bestw.scatter_reduce_(0, cj_c, w_c, reduce="amax", include_self=True)
            win = w_c == bestw[cj_c]
            src = cj_c[win]; dst = ci_c[win]
            if not bool((src != dst).any()):
                break
            parent[src] = dst

    roots = _th_find_roots(parent)
    seg_flat = torch.full((n_vox,), -1, dtype=torch.int64, device=dev)
    seg_flat[fg_flat] = roots
    out = torch.zeros(n_vox, dtype=torch.int64, device=dev)
    fgm = seg_flat != -1
    if bool(fgm.any()):
        _, inv = torch.unique(seg_flat[fgm], return_inverse=True)
        out[fgm] = (inv + 1).to(torch.int64)
    seg = out.view(D, H, W)

    if size_filter and size_filter > 0:
        flat = seg.reshape(-1)
        counts = torch.bincount(flat)
        small = (counts < int(size_filter)).nonzero(as_tuple=True)[0]
        small = small[small != 0]
        if small.numel():
            kill = torch.isin(flat, small)
            flat = flat.clone(); flat[kill] = 0
            fg2 = flat != 0
            out2 = torch.zeros_like(flat)
            if bool(fg2.any()):
                _, iv = torch.unique(flat[fg2], return_inverse=True)
                out2[fg2] = (iv + 1).to(torch.int64)
            seg = out2.view(D, H, W)
    return seg


# Aliases: mws_np = exact sequential (numpy/numba); mws_cp / mws_th = GPU.
mws_np = mutex_watershed


# ---------------------------------------------------------------------------
# nn.Module wrapper (drop-in for the validation agglomeration step)
# ---------------------------------------------------------------------------

class MutexWatershed(nn.Module):
    """Mutex Watershed agglomerator for batched affinity heads.

    Returns ``[B, *spatial]`` ``long`` instance ids (``0`` = background),
    the drop-in contract for the validation instance-metric path.  It is
    **non-differentiable** and used at eval / inference only.

    Dispatches per input (``backend``):

    - ``auto`` / ``torch`` / ``gpu`` -> :func:`mws_th` (torch Boruvka,
      native zero-copy directly on the CUDA tensor) for CUDA inputs.
    - ``cupy`` -> :func:`mws_cp` (cupy Boruvka, zero-copy via DLPack).
    - ``cpu`` (or any CPU-tensor input) -> the exact numpy/numba
      :func:`mws_np` reference.

    The GPU paths are approximations (ARI ~0.99 vs ``mws_np``); ``mws_np``
    is the exact reference and the automatic CPU fallback.

    Args:
        offsets: ``(dz, dy, dx)`` per affinity channel.  Defaults to
            :data:`brainbow.losses.AFFINITY_OFFSETS`.
        n_pull: Number of leading pull offsets.
        strides: Per-axis subsampling of push edges.
        size_filter: Min component size (voxels); smaller -> background.
        max_push_edges: Cap on push (mutex) edges; ``None`` = full edges.
        backend: ``auto`` / ``torch`` / ``cupy`` / ``cpu``.
        buckets: Priority buckets for the GPU Boruvka approximation.
        grow_boundaries: When ``True``, grow instance labels into the
            unlabeled (background) band after agglomeration so adjacent
            segments meet at a thin border -- removes the thick rind left
            where the foreground mask excludes the predicted membrane.
        grow_max_distance: Max voxel distance a label grows into the band
            (``<= 0`` fills all background).  Keeps large true-background
            cores at ``0``.
        gate_with_sem: Metadata for callers (e.g. the TB ImageLogger / a
            predict script): when ``True`` (default), gate the agglomeration
            with the predicted ``sem`` foreground (``sem > sem_gate_threshold``);
            when ``False``, run MWS unmasked so borders are the 1-voxel
            affinity cuts (no thick predicted-membrane rind).  The eval
            instance metric always gates with the GT foreground regardless.
        sem_gate_threshold: Probability threshold for the predicted-sem gate.
    """

    def __init__(
        self,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        n_pull: Optional[int] = None,
        strides: Sequence[int] = (1, 4, 4),
        size_filter: int = 0,
        max_push_edges: Optional[int] = None,
        backend: str = "auto",
        buckets: int = 16,
        grow_boundaries: bool = False,
        grow_max_distance: float = 0.0,
        gate_with_sem: bool = True,
        sem_gate_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        from brainbow.losses import AFFINITY_OFFSETS, N_PULL

        self.offsets = tuple(
            tuple(int(c) for c in o)
            for o in (offsets if offsets is not None else AFFINITY_OFFSETS)
        )
        self.n_pull = (
            int(n_pull) if n_pull is not None else N_PULL
        )
        self.strides = tuple(int(s) for s in strides)
        self.size_filter = int(size_filter)
        self.max_push_edges = (
            int(max_push_edges) if max_push_edges else None
        )
        self.backend = str(backend).lower()
        self.buckets = int(buckets)
        # Post-process: grow segments into the unlabeled (background) band so
        # adjacent instances meet at a thin border instead of the thick rind
        # left where the foreground mask excludes the predicted membrane.
        self.grow_boundaries = bool(grow_boundaries)
        self.grow_max_distance = float(grow_max_distance)
        # Whether callers should gate the agglomeration with the predicted
        # ``sem`` foreground (``sem > sem_gate_threshold``).  When False, MWS
        # runs unmasked -- every voxel is labeled and segment borders are the
        # 1-voxel affinity cuts (no thick predicted-membrane rind).  This is
        # metadata read by the caller (e.g. the TB ImageLogger); the eval
        # instance metric deliberately gates with the GT foreground instead.
        self.gate_with_sem = bool(gate_with_sem)
        self.sem_gate_threshold = float(sem_gate_threshold)

    def _resolve_backend(self, affinities: torch.Tensor) -> str:
        """Resolve the effective backend for this input: torch / cupy / cpu.

        ``auto`` prefers the native-zero-copy torch path (``mws_th``) on
        CUDA inputs (no DLPack, no cupy dependency), falling back to the
        exact numpy/numba ``mws_np`` on CPU.  ``cupy`` selects the DLPack
        zero-copy ``mws_cp`` path.  ``cpu`` forces ``mws_np``.
        """
        b = self.backend
        if b == "cpu" or not affinities.is_cuda:
            return "cpu"
        if b == "cupy":
            return "cupy" if _HAVE_CUPY else "torch"
        if b in ("torch", "gpu", "auto"):
            return "torch"
        return "torch"

    def _maybe_grow(self, seg: torch.Tensor) -> torch.Tensor:
        """Optionally grow labels to fill the unlabeled boundary band.

        No-op unless ``grow_boundaries`` is set.  Runs per batch element on
        the host (scipy EDT); only enabled paths pay the transfer.
        """
        if not self.grow_boundaries:
            return seg
        dev = seg.device
        seg_np = seg.detach().cpu().numpy()
        for b in range(seg_np.shape[0]):
            seg_np[b] = grow_labels_to_fill(seg_np[b], self.grow_max_distance)
        return torch.from_numpy(seg_np).to(dev)

    @torch.no_grad()
    def forward(
        self,
        affinities: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Agglomerate a batch of affinity volumes into instance ids.

        Args:
            affinities: ``[B, n_offsets, D, H, W]`` probabilities.
            foreground_mask: Optional ``[B, D, H, W]`` boolean mask.

        Returns:
            ``[B, D, H, W]`` ``long`` instance-id volume.
        """
        if affinities.dim() != 5:
            raise ValueError(
                f"MutexWatershed expects [B, n_offsets, D, H, W]; "
                f"got {tuple(affinities.shape)}."
            )
        B = affinities.shape[0]
        mode = self._resolve_backend(affinities)

        # GPU path (torch mws_th): native zero-copy -- operates directly on
        # the CUDA tensor, no DLPack / host transfer / cupy dependency.
        if mode == "torch":
            affs = affinities.detach().float()
            outs = []
            for b in range(B):
                m = (
                    foreground_mask[b].detach()
                    if foreground_mask is not None else None
                )
                outs.append(mws_th(
                    affs[b], self.offsets, self.n_pull,
                    strides=self.strides, mask=m,
                    size_filter=self.size_filter,
                    max_push_edges=self.max_push_edges,
                    buckets=self.buckets,
                ))
            return self._maybe_grow(torch.stack(outs, dim=0).to(affinities.device))

        # GPU path (cupy mws_cp): zero-copy view of the CUDA tensor via
        # DLPack -- the dense affinity volume never leaves the device.
        if mode == "cupy":
            affs = affinities.detach().float()
            outs = []
            for b in range(B):
                aff_cp = _cp.from_dlpack(affs[b].contiguous())
                mask_cp = None
                if foreground_mask is not None:
                    mask_cp = _cp.from_dlpack(
                        foreground_mask[b].detach().to(torch.uint8).contiguous()
                    ).astype(_cp.bool_)
                seg = mws_cp(
                    aff_cp, self.offsets, self.n_pull,
                    strides=self.strides, mask=mask_cp,
                    size_filter=self.size_filter,
                    max_push_edges=self.max_push_edges,
                    buckets=self.buckets,
                )
                outs.append(torch.from_dlpack(seg))
            return self._maybe_grow(torch.stack(outs, dim=0).to(affinities.device))

        # CPU fallback / reference (mws_np): single host transfer + numba.
        affs_np = affinities.detach().float().cpu().numpy()
        mask_np = (
            foreground_mask.detach().cpu().numpy().astype(bool)
            if foreground_mask is not None
            else None
        )
        out = np.empty(
            (B,) + tuple(affinities.shape[2:]), dtype=np.int64,
        )
        for b in range(B):
            out[b] = mutex_watershed(
                affs_np[b],
                self.offsets,
                self.n_pull,
                strides=self.strides,
                mask=None if mask_np is None else mask_np[b],
                size_filter=self.size_filter,
                max_push_edges=self.max_push_edges,
            )
        return self._maybe_grow(torch.from_numpy(out).to(affinities.device))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_offsets={len(self.offsets)}, "
            f"n_pull={self.n_pull}, strides={self.strides}, "
            f"size_filter={self.size_filter}, "
            f"max_push_edges={self.max_push_edges}, "
            f"backend={self.backend}, buckets={self.buckets}, "
            f"grow_boundaries={self.grow_boundaries}, "
            f"grow_max_distance={self.grow_max_distance}, "
            f"gate_with_sem={self.gate_with_sem}, "
            f"sem_gate_threshold={self.sem_gate_threshold})"
        )


__all__ = [
    "mutex_watershed",
    "mws_np",
    "mws_cp",
    "mws_th",
    "grow_labels_to_fill",
    "MutexWatershed",
]
