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
    from numba import cuda as _nbcuda

    _HAVE_CUDA = bool(_nbcuda.is_available())
except Exception:  # pragma: no cover
    _nbcuda = None
    _HAVE_CUDA = False


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
    Attractive edges use ``weight = affinity``; push edges use
    ``weight = 1 - affinity`` and are flagged as mutex.  Repulsive edges
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
) -> np.ndarray:
    """Mutex Watershed segmentation of an affinity volume.

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
# Full-GPU path (numba.cuda sequential kernel + torch edge build / relabel)
# ---------------------------------------------------------------------------

if _HAVE_CUDA:

    @_nbcuda.jit(device=True)
    def _find_cuda(parent, x):  # pragma: no cover - device code
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    @_nbcuda.jit
    def _mws_core_cuda(eu, ev, emut, order, parent, rank,
                       head, tail, count, lnext, lto):  # pragma: no cover
        """Single-thread Kruskal + mutex pass on device (mirrors ``_mws_core``).

        Inherently sequential (each edge decision depends on prior unions /
        mutexes), so this runs in one GPU thread; the speedup over the CPU
        path comes from never copying the affinity volume to the host and
        from the GPU-parallel edge build / sort / relabel around it.
        """
        if _nbcuda.grid(1) != 0:
            return
        n_store = 0
        for idx in range(order.shape[0]):
            e = order[idx]
            u = eu[e]
            v = ev[e]
            ru = _find_cuda(parent, u)
            rv = _find_cuda(parent, v)
            if ru == rv:
                continue
            if count[ru] <= count[rv]:
                a = ru
                b = rv
            else:
                a = rv
                b = ru
            blocked = False
            p = head[a]
            while p != -1:
                if _find_cuda(parent, lto[p]) == b:
                    blocked = True
                    break
                p = lnext[p]
            if blocked:
                continue
            if emut[e] != 0:
                lto[n_store] = v
                lnext[n_store] = -1
                if head[ru] == -1:
                    head[ru] = n_store
                else:
                    lnext[tail[ru]] = n_store
                tail[ru] = n_store
                count[ru] += 1
                n_store += 1
                lto[n_store] = u
                lnext[n_store] = -1
                if head[rv] == -1:
                    head[rv] = n_store
                else:
                    lnext[tail[rv]] = n_store
                tail[rv] = n_store
                count[rv] += 1
                n_store += 1
            else:
                if rank[ru] < rank[rv]:
                    tmp = ru
                    ru = rv
                    rv = tmp
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
                if head[rv] != -1:
                    if head[ru] == -1:
                        head[ru] = head[rv]
                        tail[ru] = tail[rv]
                    else:
                        lnext[tail[ru]] = head[rv]
                        tail[ru] = tail[rv]
                    count[ru] += count[rv]
                    head[rv] = -1
                    tail[rv] = -1
                    count[rv] = 0

    @_nbcuda.jit
    def _resolve_cuda(parent, out):  # pragma: no cover - device code
        i = _nbcuda.grid(1)
        if i < parent.shape[0]:
            root = i
            while parent[root] != root:
                root = parent[root]
            out[i] = root


def _relabel_consecutive_torch(
    seg_flat: torch.Tensor, ignore: int = -1,
) -> torch.Tensor:
    """Relabel a flat label tensor to ``0`` (ignored) + consecutive ``1..K``."""
    out = torch.zeros_like(seg_flat)
    fg = seg_flat != ignore
    if bool(fg.any()):
        _, inv = torch.unique(seg_flat[fg], return_inverse=True)
        out[fg] = (inv + 1).to(out.dtype)
    return out


def _apply_size_filter_torch(seg: torch.Tensor, min_size: int) -> torch.Tensor:
    """Reset instances smaller than ``min_size`` voxels to background (GPU)."""
    flat = seg.reshape(-1)
    counts = torch.bincount(flat)
    small = torch.nonzero(counts < min_size, as_tuple=False).flatten()
    small = small[small != 0]
    if small.numel():
        kill = torch.isin(flat, small)
        flat = flat.clone()
        flat[kill] = 0
        seg = _relabel_consecutive_torch(
            torch.where(flat == 0, torch.full_like(flat, -1), flat), ignore=-1,
        ).view(seg.shape)
    return seg


def mutex_watershed_gpu(
    affinities: torch.Tensor,
    offsets: Sequence[Sequence[int]],
    n_pull: int,
    strides: Sequence[int] = (1, 4, 4),
    mask: Optional[torch.Tensor] = None,
    size_filter: int = 0,
) -> torch.Tensor:
    """Full-GPU Mutex Watershed for a single ``[n_offsets, D, H, W]`` volume.

    Edge construction, foreground-compaction, priority sort, and relabel
    run on the GPU (torch); the sequential union-find + mutex pass runs as
    a single-thread :mod:`numba.cuda` kernel.  Nothing is copied to the
    host -- the result is a GPU ``LongTensor`` ``[D, H, W]``.
    """
    if not _HAVE_CUDA:
        raise RuntimeError("mutex_watershed_gpu requires numba.cuda.")
    dev = affinities.device
    n_off, D, H, W = affinities.shape
    n_vox = D * H * W
    grid = torch.arange(n_vox, device=dev, dtype=torch.int64).view(D, H, W)

    if mask is None:
        fg_flat = torch.ones(n_vox, dtype=torch.bool, device=dev)
    else:
        fg_flat = mask.reshape(-1).to(torch.bool)
    M = int(fg_flat.sum().item())
    if M == 0:
        return torch.zeros((D, H, W), dtype=torch.long, device=dev)

    node_of_voxel = torch.full((n_vox,), -1, dtype=torch.int32, device=dev)
    node_of_voxel[fg_flat] = torch.arange(M, dtype=torch.int32, device=dev)

    sz, sy, sx = (int(s) for s in strides)
    us: List[torch.Tensor] = []
    vs: List[torch.Tensor] = []
    ws: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []
    for io in range(n_off):
        dz, dy, dx = (int(c) for c in offsets[io])
        zs_s, zs_t = _axis_slices(dz, D)
        ys_s, ys_t = _axis_slices(dy, H)
        xs_s, xs_t = _axis_slices(dx, W)
        u_src = grid[zs_s, ys_s, xs_s]
        v_tgt = grid[zs_t, ys_t, xs_t]
        w = affinities[io][zs_s, ys_s, xs_s]
        pull = io < n_pull
        if not pull:
            u_src = u_src[::sz, ::sy, ::sx]
            v_tgt = v_tgt[::sz, ::sy, ::sx]
            w = (1.0 - w[::sz, ::sy, ::sx])
        u = u_src.reshape(-1)
        v = v_tgt.reshape(-1)
        w = w.reshape(-1).float()
        keep = fg_flat[u] & fg_flat[v]
        if not bool(keep.any()):
            continue
        u = node_of_voxel[u[keep]]
        v = node_of_voxel[v[keep]]
        w = w[keep]
        us.append(u)
        vs.append(v)
        ws.append(w)
        ms.append(torch.full((u.numel(),), 0 if pull else 1,
                            dtype=torch.uint8, device=dev))

    if not us:
        return torch.zeros((D, H, W), dtype=torch.long, device=dev)

    edge_u = torch.cat(us).contiguous()
    edge_v = torch.cat(vs).contiguous()
    edge_w = torch.cat(ws).contiguous()
    edge_m = torch.cat(ms).contiguous()
    n_mutex = int(edge_m.sum().item())
    try:
        order = torch.sort(edge_w, descending=True, stable=True).indices
    except TypeError:  # older torch without stable kwarg
        order = torch.argsort(edge_w, descending=True)
    order = order.to(torch.int64).contiguous()

    parent = torch.arange(M, dtype=torch.int32, device=dev).contiguous()
    rank = torch.zeros(M, dtype=torch.int32, device=dev)
    head = torch.full((M,), -1, dtype=torch.int32, device=dev)
    tail = torch.full((M,), -1, dtype=torch.int32, device=dev)
    count = torch.zeros(M, dtype=torch.int32, device=dev)
    cap = 2 * n_mutex + 1
    lnext = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    lto = torch.full((cap,), -1, dtype=torch.int32, device=dev)

    ca = _nbcuda.as_cuda_array
    _mws_core_cuda[1, 1](
        ca(edge_u), ca(edge_v), ca(edge_m), ca(order),
        ca(parent), ca(rank), ca(head), ca(tail), ca(count),
        ca(lnext), ca(lto),
    )
    roots = torch.empty(M, dtype=torch.int32, device=dev)
    threads = 256
    blocks = (M + threads - 1) // threads
    _resolve_cuda[blocks, threads](ca(parent), ca(roots))
    _nbcuda.synchronize()

    seg_flat = torch.full((n_vox,), -1, dtype=torch.int64, device=dev)
    seg_flat[fg_flat] = roots.to(torch.int64)
    out = _relabel_consecutive_torch(seg_flat, ignore=-1).view(D, H, W)
    if size_filter > 0:
        out = _apply_size_filter_torch(out, size_filter)
    return out


# ---------------------------------------------------------------------------
# nn.Module wrapper (drop-in for the validation agglomeration step)
# ---------------------------------------------------------------------------

class MutexWatershed(nn.Module):
    """Mutex Watershed agglomerator for batched affinity heads.

    Mirrors the label-output contract of the embedding clusterers
    (returns ``[B, *spatial]`` ``long`` instance ids, ``0`` = background)
    so it is a drop-in for the validation instance-metric path.  It is
    **non-differentiable** and used at eval / inference only.

    When the input affinities are on CUDA (and ``numba.cuda`` is
    available), the agglomeration runs fully on the GPU via
    :func:`mutex_watershed_gpu` -- no host transfer of the affinity
    volume.  Otherwise it falls back to the CPU numpy / numba path.

    Args:
        offsets: ``(dz, dy, dx)`` per affinity channel.  Defaults to
            :data:`brainbow.losses.AFFINITY_OFFSETS`.
        n_pull: Number of leading pull offsets.
        strides: Per-axis subsampling of push edges.
        size_filter: Min component size (voxels); smaller -> background.
    """

    def __init__(
        self,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        n_pull: Optional[int] = None,
        strides: Sequence[int] = (1, 4, 4),
        size_filter: int = 0,
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

        # GPU path: keep everything on device, process one sample at a time
        # to bound peak memory (the per-sample edge graph is the transient).
        if affinities.is_cuda and _HAVE_CUDA:
            affs = affinities.detach().float()
            outs = []
            for b in range(B):
                m = (
                    foreground_mask[b].detach()
                    if foreground_mask is not None else None
                )
                outs.append(mutex_watershed_gpu(
                    affs[b], self.offsets, self.n_pull,
                    strides=self.strides, mask=m, size_filter=self.size_filter,
                ))
            return torch.stack(outs, dim=0)

        # CPU fallback (numpy / numba).
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
            )
        return torch.from_numpy(out).to(affinities.device)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_offsets={len(self.offsets)}, "
            f"n_pull={self.n_pull}, strides={self.strides}, "
            f"size_filter={self.size_filter})"
        )


__all__ = ["mutex_watershed", "mutex_watershed_gpu", "MutexWatershed"]
