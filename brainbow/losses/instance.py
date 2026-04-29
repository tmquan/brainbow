"""
Instance segmentation loss: discriminative embedding (pull / push / norm)
plus an optional dense face-affinity term derived from the embedding.

Channel layout (what ``prediction`` looks like)::

    prediction: [B, E, *spatial]   instance embeddings
    labels:     [B, *spatial]      integer instance ids (0 = background)

The loss decomposes into four scalars:

* **pull**     - per-instance hinge² distance from each foreground
                  voxel's embedding to the instance's mean embedding,
                  with margin ``delta_v``.  Drives intra-cluster
                  cohesion.
* **push**     - hinge² repulsion between every pair of instance mean
                  embeddings, with margin ``2 * delta_d`` (so any two
                  centroids end up at least ``2 * delta_d`` apart).
                  ``max_hard_pairs`` keeps only the top-k hardest pairs
                  to bound gradient memory on dense crops.
* **norm**     - L2 regulariser on the centroid norms.  Tames the
                  growth induced by the push margin.
* **aff_emb**  - optional dense face-affinity supervision.  At every
                  fg-fg face pair, ``exp(-tau * ||emb[v] - emb[v+1]||_1)``
                  is supervised against the binary aff target via
                  masked soft-Dice.  Reuses the same kernel as
                  :class:`BoundaryLoss` so the affinity geometry is
                  defined once.  Turn off with ``weight_aff_emb=0``.

Dimension-agnostic via ``spatial_dims`` (2 or 3); ``aff_emb`` is 3-D
only because the 6-face primitives in
:mod:`brainbow.losses.boundary` assume ``[B, D, H, W]``.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

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


class _BatchElemCtx(NamedTuple):
    """Per-batch-element tensors shared across pull / push / norm."""
    fg: torch.Tensor                # [N] bool
    inverse: torch.Tensor           # [M] zero-based instance index
    emb_fg: torch.Tensor            # [M, E]
    K: int
    E: int


class InstanceLoss(nn.Module):
    """Discriminative pull / push / norm on instance embeddings.

    Args:
        spatial_dims: 2 for images, 3 for volumes.
        weight_pull: Weight for the pull (intra-cluster) term.
        weight_push: Weight for the push (inter-cluster) term.
        weight_norm: Weight for the centroid norm regulariser.
        weight_aff_emb: Weight for the dense face-affinity term derived
            from the embedding via :func:`soft_aff_from_field`.  ``0``
            (default) skips the kernel entirely; ``> 0`` requires
            ``spatial_dims == 3``.
        delta_v: Pull margin (hinge threshold per embedding).
        delta_d: Push margin -- centroids end up ``>= 2 * delta_d``
            apart at convergence.
        tau: Bandwidth of the soft similarity kernel used by
            ``aff_emb`` (see :func:`soft_aff_from_field`).  When
            ``normalize_embeddings=True`` the L1 distance is bounded
            by ``2 * sqrt(E)``, so ``tau ~ 1`` is a sensible default;
            otherwise pick ``tau`` so that two embeddings at the
            typical push distance ``2 * delta_d`` give a kernel close
            to ``0``.
        normalize_embeddings: L2-normalise embeddings to the unit
            hypersphere before computing pull / push (and ``aff_emb``).
            Bounds all distances to ``[0, 2]`` and removes the need
            for a norm regulariser; pair with a smaller ``delta_d``.
        max_hard_pairs: When ``> 0``, only the top-k hardest centroid
            pairs (smallest distance) contribute to the push loss --
            concentrates gradient on pairs that actually need
            separating and bounds the push backward memory.  ``0``
            uses every upper-triangle pair.
        aff_eps: Numerical stabiliser in the soft-Dice numerator and
            denominator of the ``aff_emb`` sub-loss.
        background: Label value treated as background when building
            the ``aff_emb`` target.  Default ``-1`` keeps every voxel
            (including ``FindBoundariesd``-zeroed boundary voxels) in
            the supervision footprint, which removes the checkerboard
            artifact along instance edges that ``background=0``
            otherwise produces.  Pass ``None`` to disable masking
            entirely (semantically identical to ``-1`` since instance
            ids are non-negative; kept in the public API for symmetry
            with :class:`BoundaryLoss`).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_aff_emb: float = 0.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        tau: float = 1.0,
        normalize_embeddings: bool = False,
        max_hard_pairs: int = 0,
        aff_eps: float = 1e-5,
        background: Optional[int] = -1,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.weight_pull = float(weight_pull)
        self.weight_push = float(weight_push)
        self.weight_norm = float(weight_norm)
        self.weight_aff_emb = float(weight_aff_emb)
        self.delta_v = float(delta_v)
        self.delta_d = float(delta_d)
        self.tau = float(tau)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.max_hard_pairs = int(max_hard_pairs)
        self.aff_eps = float(aff_eps)
        self.background = int(background) if background is not None else None

        if self.weight_aff_emb > 0 and self.spatial_dims != 3:
            raise ValueError(
                "weight_aff_emb > 0 requires spatial_dims == 3 (the 6-face "
                "affinity primitives in brainbow.losses.boundary assume a "
                "[B, D, H, W] label tensor).  Set weight_aff_emb=0 for 2-D."
            )

    @property
    def task_channels(self) -> int:
        """Embedding dimensionality is set by the model wrapper, not here."""
        raise AttributeError(
            "InstanceLoss does not pin a channel count; the embedding "
            "dimensionality is set by the model wrapper "
            "(``instance_channels``)."
        )

    # ------------------------------------------------------------------
    # API symmetry stubs (other task losses build per-batch targets here;
    # the discriminative loss's targets are state-ful per batch element
    # and are computed inside ``forward``).
    # ------------------------------------------------------------------

    def build_target(self, *args, **kwargs) -> None:
        return None

    def compute_weights(self, labels: torch.Tensor) -> None:
        return None

    # ------------------------------------------------------------------
    # Sub-losses (operate on a single batch element via ``ctx``)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_target_centers(ctx: _BatchElemCtx) -> torch.Tensor:
        """Mean-embedding centroid per instance, ``[K, E]``.

        Intentionally **not** decorated with ``@torch.no_grad``: the
        empirical mean is part of the discriminative loss's autograd
        graph -- push and norm rely on gradients flowing back through
        these centroids into the model's instance head.
        """
        K, E = ctx.K, ctx.E
        sums = torch.zeros(K, E, device=ctx.emb_fg.device, dtype=torch.float32)
        sums.scatter_add_(0, repeat(ctx.inverse, "m -> m e", e=E), ctx.emb_fg)
        counts = (
            torch.bincount(ctx.inverse, minlength=K)
            .float()
            .clamp(min=1)
        )
        return sums / rearrange(counts, "k -> k 1")

    def _compute_loss_pull(
        self, ctx: _BatchElemCtx, centers: torch.Tensor,
    ) -> torch.Tensor:
        """Per-instance hinge² pull toward the centroid."""
        center_per_voxel = centers[ctx.inverse]                            # [M, E]
        diff = ctx.emb_fg - center_per_voxel                               # [M, E]
        dist = reduce(diff ** 2, "m e -> m", "sum").clamp(min=1e-12).sqrt()
        per_voxel = (dist - self.delta_v).clamp(min=0).pow(2)
        sums = torch.zeros(ctx.K, device=ctx.emb_fg.device, dtype=torch.float32)
        sums.scatter_add_(0, ctx.inverse, per_voxel)
        counts = torch.bincount(ctx.inverse, minlength=ctx.K).float().clamp(min=1)
        return (sums / counts).mean()

    def _compute_loss_push(
        self, centers: torch.Tensor, K: int,
    ) -> torch.Tensor:
        """Upper-triangle pairwise hinge² push on mean centroids."""
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

    @staticmethod
    def _compute_loss_norm(centers: torch.Tensor) -> torch.Tensor:
        """L2 norm regulariser on mean centroids."""
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
        Pair mask:      both endpoints foreground (drop bg-bg / bg-fg
                        pairs; the bg embedding is free).
        Loss:           masked soft-Dice averaged over the 6 direction
                        channels::

            dice_c = 1 - (2 * (p * t * m).sum + eps) /
                            ((p * m).sum + (t * m).sum + eps)

        Args:
            embed: ``[B, E, D, H, W]`` per-voxel instance embedding.
            label: ``[B, D, H, W]`` integer instance ids.
            cached_aff_target: Optional precomputed
                ``[B, 6, D, H, W]`` aff target.  When provided
                (typically by :meth:`CombinedLoss._build_targets` so
                BoundaryLoss and InstanceLoss share one build), the
                ``_affinity_target_torch`` call is skipped.  The
                cached tensor must be built with the same
                ``background`` value the loss is configured with --
                the caller is responsible for that contract.
        """
        if self.normalize_embeddings:
            embed = F.normalize(embed, p=2, dim=1, eps=1e-6)

        aff_pred = soft_aff_from_field(embed, tau=self.tau)        # [B, 6, D, H, W]

        with torch.no_grad():
            if cached_aff_target is None:
                aff_target = _affinity_target_torch(
                    label.long(), background=self.background,
                )
            else:
                aff_target = cached_aff_target
            aff_target = aff_target.to(
                dtype=aff_pred.dtype, device=aff_pred.device,
            )

            if self.background is None:
                fg = torch.ones_like(label, dtype=aff_pred.dtype)
            else:
                fg = (label != self.background).to(dtype=aff_pred.dtype)
            pair_mask = torch.stack([
                fg * _shift_replicate_torch(fg, axis, shift)
                for _, axis, shift in _DIRECTIONS
            ], dim=1)                                              # [B, 6, D, H, W]

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

    def _make_ctx(
        self,
        emb_b: torch.Tensor,
        lbl_b: torch.Tensor,
    ) -> Optional[_BatchElemCtx]:
        """Build the per-batch-element context, or ``None`` if no foreground."""
        fg = lbl_b > 0
        if not fg.any():
            return None
        _, inverse = torch.unique(lbl_b[fg], return_inverse=True)
        K = int(inverse.max().item()) + 1
        emb_fg = rearrange(emb_b[:, fg], "e m -> m e").float()
        return _BatchElemCtx(
            fg=fg, inverse=inverse, emb_fg=emb_fg, K=K, E=emb_fg.shape[1],
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        cached_aff_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute pull / push / norm + (optional) ``aff_emb`` sub-losses.

        Args:
            embed:  ``[B, E, *spatial]`` per-voxel instance embedding.
            label:  ``[B, *spatial]`` integer instance ids.
            cached_aff_target: Optional ``[B, 6, D, H, W]`` aff target
                shared across heads (see
                :meth:`CombinedLoss._build_targets`).  When supplied,
                the ``_affinity_target_torch`` rebuild inside
                :meth:`_compute_loss_aff_emb` is skipped.
        """
        device = embed.device

        emb_flat = rearrange(embed, "b e ... -> b e (...)")
        if self.normalize_embeddings:
            emb_flat = F.normalize(emb_flat, dim=1, eps=1e-6)
        lbl_flat = rearrange(label, "b ... -> b (...)")

        L_pull = torch.zeros((), device=device)
        L_push = torch.zeros((), device=device)
        L_norm = torch.zeros((), device=device)
        n_valid = 0

        for b in range(embed.shape[0]):
            ctx = self._make_ctx(emb_flat[b], lbl_flat[b])
            if ctx is None:
                continue
            n_valid += 1

            centers = self._build_target_centers(ctx)
            L_pull = L_pull + self._compute_loss_pull(ctx, centers)
            L_push = L_push + self._compute_loss_push(centers, ctx.K)
            if not self.normalize_embeddings:
                L_norm = L_norm + self._compute_loss_norm(centers)

        n = max(n_valid, 1)
        pull = L_pull / n
        push = L_push / n
        norm = L_norm / n
        total = (
            self.weight_pull * pull
            + self.weight_push * push
            + self.weight_norm * norm
        )

        if self.weight_aff_emb > 0:
            aff_emb = self._compute_loss_aff_emb(
                embed, label, cached_aff_target=cached_aff_target,
            )
            total = total + self.weight_aff_emb * aff_emb
        else:
            aff_emb = embed.new_zeros(())

        return {
            "loss": total,
            "pull": pull,
            "push": push,
            "norm": norm,
            "aff_emb": aff_emb,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"spatial_dims={self.spatial_dims}, "
            f"weight_pull={self.weight_pull}, "
            f"weight_push={self.weight_push}, "
            f"weight_norm={self.weight_norm}, "
            f"weight_aff_emb={self.weight_aff_emb}, "
            f"delta_v={self.delta_v}, "
            f"delta_d={self.delta_d}, "
            f"tau={self.tau}, "
            f"normalize_embeddings={self.normalize_embeddings}, "
            f"max_hard_pairs={self.max_hard_pairs}, "
            f"aff_eps={self.aff_eps}, "
            f"background={self.background})"
        )
