"""
Instance segmentation loss: pull / push / norm.

Dimension-agnostic -- parameterized by ``spatial_dims`` to handle both
2-D (H, W) and 3-D (D, H, W) inputs with the appropriate pool function.

Boundary and skeleton pixels receive boosted weights so the model pays
extra attention to separating touching instances and reconstructing the
medial axis.

Supports an optional **centroid-anchoring** mode where each instance's
pull target is a deterministic sinusoidal positional encoding of its
spatial center-of-mass, replacing the unstable empirical mean embedding.

Channel layout (what ``prediction`` looks like)::

    prediction: [B, E, *spatial]   instance embeddings
    labels:     [B, *spatial]      integer instance ids (0 = background)
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from brainbow.losses.boundary import (
    _DIRECTIONS,
    _affinity_target_torch,
    _shift_replicate_torch,
    soft_aff_from_field,
)
from brainbow.utils.parallel import pmap


# ---------------------------------------------------------------------------
# Helpers shared by the boundary / skeleton weight paths
# ---------------------------------------------------------------------------


def _pool_fn(spatial_dims: int):
    return F.max_pool3d if spatial_dims == 3 else F.max_pool2d


def _edt_worker(args: Tuple[np.ndarray, int]) -> Tuple[int, np.ndarray]:
    """Per-instance normalised EDT for pmap subprocesses (CPU/scipy)."""
    from brainbow.transforms.edt import distance_transform_edt
    label_np_b, uid = args
    mask = label_np_b == uid
    dt = distance_transform_edt(mask).astype(np.float32)
    max_d = dt.max()
    if max_d > 0:
        dt /= max_d
    return (uid, dt)


class _BatchElemCtx(NamedTuple):
    """Per-batch-element tensors shared across pull / push / norm."""
    b: int
    fg: torch.Tensor                # [N] bool
    inverse: torch.Tensor           # [M] zero-based instance index
    emb_fg: torch.Tensor            # [M, E]
    wgt_fg: Optional[torch.Tensor]  # [M] float or None
    spatial_shape: Tuple[int, ...]
    K: int
    E: int


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


class InstanceLoss(nn.Module):
    """Weighted discriminative pull/push/norm on instance embeddings.

    Args:
        spatial_dims: 2 for images, 3 for volumes.
        weight_pull: Weight for the pull (intra-cluster) term.
        weight_push: Weight for the push (inter-cluster) term.
        weight_norm: Weight for the centroid norm regularisation term.
        weight_edge: Boundary pixel weight multiplier (1.0 = disabled).
        weight_bone: Medial-axis pixel weight multiplier (1.0 = disabled).
        weight_aff_emb: Weight for the dense face-affinity term derived
            from the embedding via :func:`soft_aff_from_field`.  At every
            foreground-foreground face-pair, supervises the kernel
            ``exp(-tau * ||emb[v] - emb[v + dir]||_1)`` against the
            label-derived binary aff target with a (masked) soft-Dice
            loss.  Adds a per-voxel-face complement to the centroid-
            level pull / push.  ``0`` (default) disables the path so it
            costs nothing; ``> 0`` requires ``spatial_dims == 3``.
        delta_v: Pull margin (hinge threshold per embedding).
        delta_d: Push margin (half of the minimum centroid separation).
        tau: Bandwidth of the soft similarity kernel used by the
            ``aff_emb`` term (see :func:`soft_aff_from_field`).  Larger
            ``tau`` sharpens the decay, smaller softens it.  When the
            embedding is L2-normalised the L1 distance is bounded by
            ``2 * sqrt(E)`` so ``tau ≈ 1`` is a sensible default;
            otherwise pick ``tau`` so that embeddings at the typical
            push distance ``2 * delta_d`` give kernel ≈ 0.
        normalize_embeddings: L2-normalize embeddings to the unit
            hypersphere before computing pull/push (and ``aff_emb``).
            Eliminates the need for norm regularisation and bounds all
            distances to ``[0, 2]``.
        max_hard_pairs: When > 0, only the top-k hardest centroid pairs
            (smallest distance) contribute to the push loss.  Focuses
            gradient on pairs that actually need separation.
        anchor_to_centroid: Snap each instance's pull target to a
            deterministic sinusoidal encoding of its spatial
            center-of-mass instead of the empirical mean embedding.
            Push still operates on the empirical mean for gradient
            flow.  Incompatible with ``normalize_embeddings``.
        centroid_scale: Multiplier on the sinusoidal encoding so that
            the typical anchor separation matches the push margin
            ``2 * delta_d``.
        aff_eps: Numerical stabiliser in the soft-Dice numerator and
            denominator of the ``aff_emb`` sub-loss.
        background: Label value treated as background when building the
            label-derived aff target (forwarded to
            :func:`_affinity_target_torch`).  Voxels with
            ``label == background`` get aff target ``0`` across all 6
            directions and are masked out of the loss together with
            their face neighbours -- the bg embedding receives no
            ``aff_emb`` gradient.  Default ``-1`` (no masking, since
            instance ids are non-negative): boundary voxels zeroed by
            :class:`FindBoundariesd` (label ``0``) **do** contribute to
            the affinity target, which removes the checkerboard
            artifact along instance boundaries that ``background=0``
            otherwise produces.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        weight_aff_emb: float = 0.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        tau: float = 1.0,
        normalize_embeddings: bool = False,
        max_hard_pairs: int = 0,
        anchor_to_centroid: bool = False,
        centroid_scale: float = 5.0,
        aff_eps: float = 1e-5,
        background: int = -1,
    ) -> None:
        super().__init__()
        if anchor_to_centroid and normalize_embeddings:
            raise ValueError(
                "anchor_to_centroid and normalize_embeddings are mutually "
                "exclusive: anchored targets live in unbounded space."
            )
        self.spatial_dims = spatial_dims
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.weight_norm = weight_norm
        self.weight_edge = weight_edge
        self.weight_bone = weight_bone
        self.weight_aff_emb = float(weight_aff_emb)
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.tau = float(tau)
        self.normalize_embeddings = normalize_embeddings
        self.max_hard_pairs = max_hard_pairs
        self.anchor_to_centroid = anchor_to_centroid
        self.centroid_scale = centroid_scale
        self.aff_eps = float(aff_eps)
        self.background = int(background)
        self._pool = _pool_fn(spatial_dims)

        if self.weight_aff_emb > 0 and self.spatial_dims != 3:
            raise ValueError(
                "weight_aff_emb > 0 currently requires spatial_dims == 3 "
                "(the 6-face affinity primitives in brainbow.losses."
                "boundary assume a [B, D, H, W] label tensor).  Set "
                "weight_aff_emb=0 for 2-D mode."
            )

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _scatter_spatial_centroids(
        fg_indices: torch.Tensor,
        inverse: torch.Tensor,
        K: int,
        spatial_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Center-of-mass of each instance in voxel coordinates (no Python loops).

        Args:
            fg_indices: ``[M]`` flat indices of foreground pixels into
                the volume of shape ``spatial_shape``.
            inverse:    ``[M]`` zero-based instance index per fg pixel.
            K:          number of instances.
            spatial_shape: spatial dimension sizes (D, H, W) or (H, W).

        Returns:
            ``[K, S]`` centroid coordinates in voxel space.
        """
        device = fg_indices.device
        S = len(spatial_shape)
        M = fg_indices.shape[0]

        coords = torch.zeros(M, S, device=device, dtype=torch.float32)
        remainder = fg_indices.clone()
        for d in range(S - 1, -1, -1):
            coords[:, d] = (remainder % spatial_shape[d]).float()
            remainder = remainder // spatial_shape[d]

        inv_expand = repeat(inverse, "m -> m s", s=S)
        centroid_sum = torch.zeros(K, S, device=device, dtype=torch.float32)
        centroid_sum.scatter_add_(0, inv_expand, coords)
        counts = torch.bincount(inverse, minlength=K).float().clamp(min=1)
        return centroid_sum / rearrange(counts, "k -> k 1")

    @staticmethod
    @torch.no_grad()
    def _sinusoidal_centroid_encoding(
        centroids: torch.Tensor,
        spatial_shape: Tuple[int, ...],
        instance_channels: int,
        scale: float,
    ) -> torch.Tensor:
        """Multi-octave sinusoidal encoding of spatial centroids.

        Distributes frequency bands across spatial dimensions proportional
        to ``log2(resolution)``, giving higher-resolution axes more
        octaves for finer discrimination.

        Args:
            centroids:         ``[K, S]`` in voxel coordinates.
            spatial_shape:     spatial dimension sizes (D, H, W) or (H, W).
            instance_channels: target embedding dimensionality E.
            scale:             output multiplier to match pull/push margins.

        Returns:
            ``[K, E]`` deterministic target embedding vectors.
        """
        K, S = centroids.shape
        device = centroids.device

        shape_t = torch.tensor(spatial_shape, device=device, dtype=torch.float32)
        c_norm = centroids / rearrange(shape_t, "s -> 1 s").clamp(min=1)

        total_pairs = instance_channels // 2
        log_res = [math.log2(max(s, 2)) for s in spatial_shape]
        total_log = sum(log_res)

        raw_alloc = [lr / total_log * total_pairs for lr in log_res]
        pairs_per_dim = [int(a) for a in raw_alloc]
        remainder = total_pairs - sum(pairs_per_dim)
        fracs = sorted(
            ((raw_alloc[d] - pairs_per_dim[d], d) for d in range(S)),
            reverse=True,
        )
        for i in range(remainder):
            pairs_per_dim[fracs[i][1]] += 1

        features = []
        for d in range(S):
            for f in range(pairs_per_dim[d]):
                freq = (2.0 ** f) * math.pi
                features.append(torch.sin(freq * c_norm[:, d]))
                features.append(torch.cos(freq * c_norm[:, d]))

        while len(features) < instance_channels:
            features.append(torch.zeros(K, device=device))

        return scale * torch.stack(features[:instance_channels], dim=1)

    @staticmethod
    def _build_target_centers(ctx: _BatchElemCtx) -> torch.Tensor:
        """Weighted mean embedding centroid per instance.

        Returns ``[K, E]`` -- the canonical target for **pull** (when not
        anchored) and always the target for **push / norm**.

        Intentionally **not** decorated with ``@torch.no_grad`` (unlike
        the other ``_build_target_*`` helpers in this package): the
        empirical mean centroid is part of the discriminative loss's
        autograd graph, not a frozen label-derived supervision target.
        Push and norm rely on gradients flowing through these centroids
        back into the model's instance head.
        """
        E, K = ctx.E, ctx.K
        if ctx.wgt_fg is not None:
            weighted = ctx.emb_fg * rearrange(ctx.wgt_fg, "m -> m 1")
            w_sum = torch.zeros(K, device=ctx.emb_fg.device, dtype=torch.float32)
            w_sum.scatter_add_(0, ctx.inverse, ctx.wgt_fg)
        else:
            weighted = ctx.emb_fg
            w_sum = torch.bincount(ctx.inverse, minlength=K).float().clamp(min=1)

        c_sum = torch.zeros(K, E, device=ctx.emb_fg.device, dtype=torch.float32)
        c_sum.scatter_add_(0, repeat(ctx.inverse, "m -> m e", e=E), weighted)
        return c_sum / (rearrange(w_sum, "k -> k 1") + 1e-8)

    @torch.no_grad()
    def _build_target_anchors(
        self,
        ctx: _BatchElemCtx,
        fg_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Sinusoidal encoding of spatial centroids -- the anchored pull target."""
        spatial_c = self._scatter_spatial_centroids(
            fg_indices, ctx.inverse, ctx.K, ctx.spatial_shape,
        )
        return self._sinusoidal_centroid_encoding(
            spatial_c, ctx.spatial_shape, ctx.E, self.centroid_scale,
        )

    def build_target(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Instance targets are state-ful per batch element.

        Target construction happens inside :meth:`forward` because the
        ``K`` instances vary per batch element and cannot be stacked.
        This hook is kept for API symmetry with the other task losses.
        """
        return None

    # ------------------------------------------------------------------
    # Per-voxel weights (boundary + skeleton)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _boundary_weight_torch(self, label: torch.Tensor) -> torch.Tensor:
        """Inner boundary weight via find_boundaries (connectivity=1, thinnest)."""
        from brainbow.transforms.find_boundaries import boundary_mask_batch
        boundary = boundary_mask_batch(label, mode="inner", connectivity=1).float()
        return 1.0 + boundary * (self.weight_edge - 1.0)

    @torch.no_grad()
    def _boundary_weight_cpu(self, label: torch.Tensor) -> torch.Tensor:
        """Boundary weight via cucim/skimage ``find_boundaries`` (connectivity=1)."""
        from brainbow.transforms.find_boundaries import find_boundaries

        label_np = label.numpy()
        weight_np = np.ones(label_np.shape, dtype=np.float32)
        edge_scale = self.weight_edge - 1.0

        for b in range(label_np.shape[0]):
            boundary = find_boundaries(label_np[b], mode="inner", connectivity=1)
            weight_np[b][boundary] = 1.0 + edge_scale

        return torch.from_numpy(weight_np)

    @torch.no_grad()
    def _skeleton_weight_torch(self, label: torch.Tensor) -> torch.Tensor:
        """Approximate skeleton weight via iterative morphological erosion on GPU."""
        B = label.shape[0]
        weight = torch.ones_like(label, dtype=torch.float32)
        _CHUNK = 32
        spatial_tail = label.shape[-self.spatial_dims:]
        max_iter = min(spatial_tail) // 2 + 1
        ones_pattern = " ".join(["1"] * self.spatial_dims)
        broadcast_ids_pattern = f"k -> k {ones_pattern}"
        reshape_pattern = f"k c -> k c {ones_pattern}"

        for b in range(B):
            fg_ids = torch.unique(label[b])
            fg_ids = fg_ids[fg_ids > 0]
            if len(fg_ids) == 0:
                continue

            K = len(fg_ids)
            for start in range(0, K, _CHUNK):
                chunk_ids = fg_ids[start:start + _CHUNK]
                label_expanded = rearrange(label[b], "... -> 1 ...")
                chunk_ids_br = rearrange(chunk_ids, broadcast_ids_pattern)
                masks = rearrange(
                    (label_expanded == chunk_ids_br).float(),
                    "k ... -> k 1 ...",
                )

                remaining = masks.clone()
                dt = torch.zeros_like(masks)
                for layer_idx in range(1, max_iter + 1):
                    eroded = -self._pool(-remaining, 3, stride=1, padding=1)
                    removed = (remaining > 0.5) & (eroded < 0.5)
                    dt[removed] = float(layer_idx)
                    remaining = eroded * (eroded > 0.5).float()
                    if not remaining.any():
                        break

                max_d = reduce(dt, "k c ... -> k c", "max").clamp(min=1.0)
                dt = dt / rearrange(max_d, reshape_pattern)

                bone_scale = self.weight_bone - 1.0
                for i, uid in enumerate(chunk_ids):
                    m = label[b] == uid
                    weight[b][m] = 1.0 + dt[i, 0][m] * bone_scale

        return weight

    @torch.no_grad()
    def _skeleton_weight_cpu(self, label: torch.Tensor) -> torch.Tensor:
        """Skeleton weight via scipy EDT on CPU with batched pmap."""
        B = label.shape[0]
        label_np = label.cpu().numpy()
        weight_np = np.ones_like(label_np, dtype=np.float32)

        all_args = []
        all_meta = []
        for b in range(B):
            fg_ids = np.unique(label_np[b])
            fg_ids = fg_ids[fg_ids > 0]
            for uid in fg_ids:
                all_args.append((label_np[b], int(uid)))
                all_meta.append(b)

        if len(all_args) == 0:
            return torch.from_numpy(weight_np).to(label.device)

        results = pmap(_edt_worker, all_args)
        for batch_idx, (uid, dt) in zip(all_meta, results):
            m = label_np[batch_idx] == uid
            weight_np[batch_idx][m] = 1.0 + dt[m] * (self.weight_bone - 1.0)

        return torch.from_numpy(weight_np).to(label.device)

    def compute_weights(
        self,
        labels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pre-compute boundary + skeleton weight volumes.

        Always prefers the on-device torch path for GPU tensors to
        avoid CPU round-trips.  Returns ``(None, None)`` for disabled
        components so callers can skip the multiply.
        """
        weight_edge = (
            (self._boundary_weight_torch if labels.is_cuda else self._boundary_weight_cpu)(labels)
            if self.weight_edge > 1.0 else None
        )
        weight_bone = (
            (self._skeleton_weight_torch if labels.is_cuda else self._skeleton_weight_cpu)(labels)
            if self.weight_bone > 1.0 else None
        )
        return weight_edge, weight_bone

    # ------------------------------------------------------------------
    # Sub-losses (operate on a single batch element via ``ctx``)
    # ------------------------------------------------------------------

    def _compute_loss_pull(
        self,
        ctx: _BatchElemCtx,
        pull_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Per-instance weighted hinge² pull toward ``pull_centers``."""
        center_per_voxel = pull_centers[ctx.inverse]                       # [M, E]
        diff = ctx.emb_fg - center_per_voxel                               # [M, E]
        dist = reduce(diff ** 2, "m e -> m", "sum").clamp(min=1e-12).sqrt()
        per_voxel = (dist - self.delta_v).clamp(min=0).pow(2)
        if ctx.wgt_fg is not None:
            per_voxel = per_voxel * ctx.wgt_fg

        pull_sum = torch.zeros(ctx.K, device=ctx.emb_fg.device, dtype=torch.float32)
        pull_sum.scatter_add_(0, ctx.inverse, per_voxel)
        pull_count = torch.bincount(ctx.inverse, minlength=ctx.K).float().clamp(min=1)
        return (pull_sum / pull_count).mean()

    def _compute_loss_push(
        self,
        centers: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Upper-triangle pairwise hinge² push on empirical mean centers."""
        if K <= 1:
            return centers.new_zeros(())
        diff = (
            rearrange(centers, "i e -> i 1 e")
            - rearrange(centers, "j e -> 1 j e")
        )
        pw = reduce(diff ** 2, "i j e -> i j", "sum").clamp(min=1e-12).sqrt()
        triu = torch.triu_indices(K, K, offset=1, device=centers.device)
        hinge = (2 * self.delta_d - pw[triu[0], triu[1]]).clamp(min=0).pow(2)
        if self.max_hard_pairs > 0 and hinge.numel() > self.max_hard_pairs:
            hinge, _ = hinge.topk(self.max_hard_pairs)
        return hinge.mean()

    def _compute_loss_norm(self, centers: torch.Tensor) -> torch.Tensor:
        """L2 norm regulariser on empirical mean centers."""
        return reduce(centers ** 2, "k e -> k", "sum").clamp(min=1e-12).sqrt().mean()

    # ------------------------------------------------------------------
    # Dense face-affinity sub-loss on the embedding
    # ------------------------------------------------------------------

    def _compute_loss_aff_emb(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        cached_aff_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Soft-Dice on the kernel-derived face affinity of the embedding.

        Predicted aff:  ``aff_pred[c, v] = exp(-tau * ||emb[v] - shift(emb)[v, dir_c]||_1)``
        Target aff:     ``aff_target = label_aff(label, background=self.background)``
        Pair mask:      both endpoints foreground (drop bg-bg / bg-fg pairs;
                        the bg embedding is free).
        Loss:           masked soft-Dice averaged over the 6 direction
                        channels::

            dice_c = 1 - (2 * (p * t * m).sum + eps) /
                            ((p * m).sum + (t * m).sum + eps)

        Reuses :func:`brainbow.losses.boundary.soft_aff_from_field` and
        :func:`brainbow.losses.boundary._affinity_target_torch` so the
        affinity geometry is defined in exactly one place.

        Args:
            embed: ``[B, E, D, H, W]`` per-voxel instance embedding.
            label: ``[B, D, H, W]`` integer instance ids.
            cached_aff_target: Optional precomputed
                ``[B, 6, D, H, W]`` aff target.  When provided
                (typically by :meth:`CombinedLoss._build_targets` so
                BoundaryLoss and InstanceLoss share one build), the
                ``_affinity_target_torch`` call is skipped.  The cached
                tensor must have been built with the same
                ``background`` value the loss is configured with -- the
                caller is responsible for that contract.
        """
        if self.normalize_embeddings:
            embed = F.normalize(embed, p=2, dim=1, eps=1e-6)

        # Predicted soft 6-aff -- generic over the embedding's channel count.
        aff_pred = soft_aff_from_field(embed, tau=self.tau)        # [B, 6, D, H, W]

        # Label-derived 6-aff target (1 on same-instance fg-fg face-pairs,
        # 0 elsewhere; bg voxels masked to 0 by ``background`` arg).
        # Both branches converge on the same dtype + device as
        # ``aff_pred`` so downstream Dice / mask ops never trigger an
        # implicit cast (mirrors BoundaryLoss.forward).
        with torch.no_grad():
            if cached_aff_target is not None:
                aff_target = cached_aff_target
            else:
                aff_target = _affinity_target_torch(
                    label.long(), background=self.background,
                )
            aff_target = aff_target.to(
                dtype=aff_pred.dtype, device=aff_pred.device,
            )

            # Pair mask: both endpoints foreground.  Exact 0/1 floats,
            # no autograd needed -- this is the supervision footprint.
            fg = (label != self.background).to(dtype=aff_pred.dtype)
            pair_mask = torch.stack([
                fg * _shift_replicate_torch(fg, axis, shift)
                for _, axis, shift in _DIRECTIONS
            ], dim=1)                                              # [B, 6, D, H, W]

        # Soft-Dice with batch+spatial reduction per channel, masked to
        # fg-fg pairs.  Average across the 6 directions.
        p = aff_pred * pair_mask
        t = aff_target * pair_mask
        intersect = reduce(p * t, "b c d h w -> c", "sum")
        pred_sum = reduce(p, "b c d h w -> c", "sum")
        target_sum = reduce(t, "b c d h w -> c", "sum")
        dice_c = 1.0 - (2.0 * intersect + self.aff_eps) / (
            pred_sum + target_sum + self.aff_eps
        )
        return dice_c.mean()

    # ------------------------------------------------------------------
    # Per-batch-element orchestration
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        w_edge: Optional[torch.Tensor],
        w_bone: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Flatten spatial dims; combine edge + bone weights."""
        emb_flat = rearrange(embed, "b e ... -> b e (...)")
        if self.normalize_embeddings:
            emb_flat = F.normalize(emb_flat, dim=1, eps=1e-6)
        lbl_flat = rearrange(label, "b ... -> b (...)")

        if w_edge is not None and w_bone is not None:
            wgt_flat = rearrange(w_edge * w_bone, "b ... -> b (...)")
        elif w_edge is not None:
            wgt_flat = rearrange(w_edge, "b ... -> b (...)")
        elif w_bone is not None:
            wgt_flat = rearrange(w_bone, "b ... -> b (...)")
        else:
            wgt_flat = None
        return emb_flat, lbl_flat, wgt_flat

    def _make_ctx(
        self,
        b: int,
        emb_flat: torch.Tensor,
        lbl_flat: torch.Tensor,
        wgt_flat: Optional[torch.Tensor],
        spatial_shape: Tuple[int, ...],
    ) -> Optional[_BatchElemCtx]:
        """Build the per-batch-element context, or ``None`` if no foreground."""
        lbl_b = lbl_flat[b]
        fg = lbl_b > 0
        if not fg.any():
            return None

        _, inverse = torch.unique(lbl_b[fg], return_inverse=True)
        K = int(inverse.max().item()) + 1
        emb_fg = rearrange(emb_flat[b, :, fg], "e m -> m e").float()
        wgt_fg = wgt_flat[b, fg].float() if wgt_flat is not None else None

        return _BatchElemCtx(
            b=b, fg=fg, inverse=inverse, emb_fg=emb_fg, wgt_fg=wgt_fg,
            spatial_shape=spatial_shape, K=K, E=emb_fg.shape[1],
        )

    def _loss_single(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        w_edge: Optional[torch.Tensor],
        w_bone: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pull/push/norm over every batch element that has foreground.

        When ``anchor_to_centroid`` is enabled, **pull** drives each
        pixel toward a deterministic sinusoidal encoding of the
        instance's spatial center-of-mass (stable fixed target), while
        **push** and **norm** still operate on the empirical mean
        embedding (for gradient flow through the model).
        """
        spatial_shape = embed.shape[2:]
        device = embed.device
        emb_flat, lbl_flat, wgt_flat = self._prepare_batch(embed, label, w_edge, w_bone)

        L_pull = torch.zeros((), device=device)
        L_push = torch.zeros((), device=device)
        L_norm = torch.zeros((), device=device)
        n_valid = 0

        for b in range(embed.shape[0]):
            ctx = self._make_ctx(b, emb_flat, lbl_flat, wgt_flat, spatial_shape)
            if ctx is None:
                continue
            n_valid += 1

            mean_centers = self._build_target_centers(ctx)

            if self.anchor_to_centroid:
                fg_indices = torch.where(ctx.fg)[0]
                pull_centers = self._build_target_anchors(ctx, fg_indices)
            else:
                pull_centers = mean_centers

            L_pull = L_pull + self._compute_loss_pull(ctx, pull_centers)
            L_push = L_push + self._compute_loss_push(mean_centers, ctx.K)
            if not self.normalize_embeddings:
                L_norm = L_norm + self._compute_loss_norm(mean_centers)

        n = max(n_valid, 1)
        pull = L_pull / n
        push = L_push / n
        norm = L_norm / n
        total = self.weight_pull * pull + self.weight_push * push + self.weight_norm * norm
        return {"loss": total, "pull": pull, "push": push, "norm": norm}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        weight_edge: Optional[torch.Tensor] = None,
        weight_bone: Optional[torch.Tensor] = None,
        cached_aff_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute pull / push / norm + (optional) ``aff_emb`` sub-losses.

        Args:
            embed:  ``[B, E, *spatial]`` per-voxel instance embedding.
            label:  ``[B, *spatial]`` integer instance ids.
            semantic_ids: Optional per-voxel semantic class map; when
                provided, the discriminative loss is averaged across
                classes by restricting ``label`` to one class at a
                time.  In that path the cached aff target (built from
                the unmasked label) does **not** match the per-class
                ``masked_label`` and is ignored -- aff_emb rebuilds
                per class.
            weight_edge / weight_bone: Optional per-voxel weight maps
                returned by :meth:`compute_weights`; built lazily when
                both are ``None``.
            cached_aff_target: Optional ``[B, 6, D, H, W]`` aff target
                shared across heads.  When supplied (typically by
                :meth:`CombinedLoss._build_targets` reusing the
                boundary head's aff slice), the ``_affinity_target_torch``
                rebuild inside :meth:`_compute_loss_aff_emb` is
                skipped.  Ignored in the multi-class branch (see above).
        """
        if weight_edge is None and weight_bone is None:
            weight_edge, weight_bone = self.compute_weights(label)

        # Multi-class: average the discriminative loss across semantic
        # classes, restricting ``label`` to one class at a time.  The
        # aff_emb term inherits the same per-class restriction so its
        # supervision matches whatever the discriminative loss saw --
        # which means the cached aff target (built from the full
        # unmasked label) cannot be reused here.  Pass ``None`` to
        # force a per-class rebuild.
        if semantic_ids is not None:
            classes = torch.unique(semantic_ids)
            classes = classes[classes > 0]
            if len(classes) > 0:
                zero = torch.zeros((), device=embed.device)
                acc = {
                    k: zero.clone()
                    for k in ("loss", "pull", "push", "norm", "aff_emb")
                }
                for cid in classes:
                    masked_label = label * (semantic_ids == cid).long()
                    out = self._loss_single(
                        embed, masked_label, weight_edge, weight_bone,
                    )
                    out = self._maybe_add_aff_emb(
                        out, embed, masked_label, cached_aff_target=None,
                    )
                    for k in acc:
                        acc[k] = acc[k] + out.get(k, zero)
                return {k: v / len(classes) for k, v in acc.items()}

        out = self._loss_single(embed, label, weight_edge, weight_bone)
        return self._maybe_add_aff_emb(out, embed, label, cached_aff_target)

    def _maybe_add_aff_emb(
        self,
        out: Dict[str, torch.Tensor],
        embed: torch.Tensor,
        label: torch.Tensor,
        cached_aff_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Append the ``aff_emb`` sub-loss to ``out`` and roll into ``loss``.

        No-op (key filled with a zero scalar) when ``weight_aff_emb`` is
        non-positive so the kernel computation is skipped entirely on
        the dominant default-config path.
        """
        if self.weight_aff_emb > 0:
            aff_emb = self._compute_loss_aff_emb(
                embed, label, cached_aff_target=cached_aff_target,
            )
            out["aff_emb"] = aff_emb
            out["loss"] = out["loss"] + self.weight_aff_emb * aff_emb
        else:
            out["aff_emb"] = embed.new_zeros(())
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"spatial_dims={self.spatial_dims}, "
            f"weight_pull={self.weight_pull}, "
            f"weight_push={self.weight_push}, "
            f"weight_norm={self.weight_norm}, "
            f"weight_aff_emb={self.weight_aff_emb}, "
            f"weight_edge={self.weight_edge}, "
            f"weight_bone={self.weight_bone}, "
            f"delta_v={self.delta_v}, "
            f"delta_d={self.delta_d}, "
            f"tau={self.tau}, "
            f"normalize_embeddings={self.normalize_embeddings}, "
            f"max_hard_pairs={self.max_hard_pairs}, "
            f"anchor_to_centroid={self.anchor_to_centroid}, "
            f"centroid_scale={self.centroid_scale}, "
            f"aff_eps={self.aff_eps}, "
            f"background={self.background})"
        )
