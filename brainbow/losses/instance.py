"""
Instance segmentation loss: pull / push / norm.

Dimension-agnostic — parameterized by ``spatial_dims`` to handle both
2-D (H, W) and 3-D (D, H, W) inputs with the appropriate pool function.

Boundary and skeleton pixels receive boosted weights so the model
pays extra attention to separating touching instances and
reconstructing the medial axis.

Supports an optional **centroid-anchoring** mode where each instance's
pull target is a deterministic sinusoidal positional encoding of its
spatial center-of-mass, replacing the unstable empirical mean embedding.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from brainbow.utils.parallel import pmap


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


def _pool_fn(spatial_dims: int):
    return F.max_pool3d if spatial_dims == 3 else F.max_pool2d


def _pad_tuple(spatial_dims: int):
    return (1, 1, 1, 1, 1, 1) if spatial_dims == 3 else (1, 1, 1, 1)


class InstanceLoss(nn.Module):
    """Weighted discriminative pull/push/norm on instance embeddings.

    Args:
        spatial_dims: 2 for images, 3 for volumes.
        weight_pull: Weight for the pull (intra-cluster) term.
        weight_push: Weight for the push (inter-cluster) term.
        weight_norm: Weight for the centroid norm regularisation term.
        weight_edge: Boundary pixel weight multiplier (1.0 = disabled).
        weight_bone: Medial-axis pixel weight multiplier (1.0 = disabled).
        delta_v: Pull margin (hinge threshold per embedding).
        delta_d: Push margin (half of the minimum centroid separation).
        normalize_embeddings: L2-normalize embeddings to the unit
            hypersphere before computing pull/push.  Eliminates the
            need for norm regularisation and bounds all distances to
            [0, 2].
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
            ``2 * delta_d``.  Higher values resolve spatially closer
            centroids but increase the target norm.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        normalize_embeddings: bool = False,
        max_hard_pairs: int = 0,
        anchor_to_centroid: bool = False,
        centroid_scale: float = 5.0,
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
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.normalize_embeddings = normalize_embeddings
        self.max_hard_pairs = max_hard_pairs
        self.anchor_to_centroid = anchor_to_centroid
        self.centroid_scale = centroid_scale

        self._pool = _pool_fn(spatial_dims)
        self._pad = _pad_tuple(spatial_dims)

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_weight_boundary(self, label: torch.Tensor) -> torch.Tensor:
        """Per-pixel boundary weight.

        GPU path (torch): morphological gradient via max_pool != min_pool.
        CPU path: cucim or skimage ``find_boundaries``.

        Always prefers the on-device torch path for GPU tensors to avoid
        costly CPU round-trips.
        """
        if label.is_cuda:
            return self._boundary_weight_torch(label)
        return self._boundary_weight_cpu(label)

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
    def _get_weight_skeleton(self, label: torch.Tensor) -> torch.Tensor:
        """Per-instance EDT skeleton weight.

        GPU path (torch): approximate L-inf EDT via morphological erosion.
        CPU path: all instances across batch via single pmap call.

        Always prefers the on-device torch path for GPU tensors to avoid
        costly CPU round-trips.
        """
        if label.is_cuda:
            return self._skeleton_weight_torch(label)

        return self._skeleton_weight_cpu(label)

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

    # ------------------------------------------------------------------
    # Centroid-anchoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _compute_spatial_centroids(
        fg_indices: torch.Tensor,
        inverse: torch.Tensor,
        K: int,
        spatial_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Center-of-mass for each instance via scatter (no Python loops).

        Args:
            fg_indices: ``[M]`` flat indices of foreground pixels into the
                volume of shape ``spatial_shape``.
            inverse: ``[M]`` zero-based instance index for each fg pixel.
            K: number of instances.
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
        to ``log2(resolution)``, giving higher-resolution axes more octaves
        for finer discrimination.

        Args:
            centroids: ``[K, S]`` centroids in voxel coordinates.
            spatial_shape: spatial dimension sizes (D, H, W) or (H, W).
            instance_channels: target embedding dimensionality E.
            scale: output multiplier to match pull/push margins.

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

    # ------------------------------------------------------------------
    # Core discriminative loss
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter_weighted_mean(emb, lbl, wgt, K):
        """Compute weighted centroid per instance using scatter.

        Args:
            emb: [E, N] embeddings for one batch element.
            lbl: [N] zero-based instance indices (0..K-1), -1 for background.
            wgt: [N] per-pixel weights.
            K: number of instances.

        Returns:
            centers: [K, E] weighted centroids.
        """
        E = emb.shape[0]
        fg = lbl >= 0
        emb_fg = emb[:, fg].float()                   # [E, M] always float32
        lbl_fg = lbl[fg]                               # [M]
        if wgt is not None:
            wgt_fg = wgt[fg].float()                   # [M] always float32
            weighted_emb = emb_fg * rearrange(wgt_fg, "m -> 1 m")
            w_sum = torch.zeros(K, device=emb.device, dtype=torch.float32)
            w_sum.scatter_add_(0, lbl_fg, wgt_fg)
        else:
            weighted_emb = emb_fg
            w_sum = torch.bincount(lbl_fg, minlength=K).float().clamp(min=1)

        c_sum = torch.zeros(E, K, device=emb.device, dtype=torch.float32)
        lbl_expand = repeat(lbl_fg, "m -> e m", e=E)
        c_sum.scatter_add_(1, lbl_expand, weighted_emb)

        centers = c_sum / (rearrange(w_sum, "k -> 1 k") + 1e-8)  # [E, K]
        return rearrange(centers, "e k -> k e")

    def _loss_single(self, embed, label, w_edge, w_bone) -> Dict[str, torch.Tensor]:
        """Pull/push/norm over all instances in the batch.

        Shapes:
            embed:  [B, E, *spatial]
            label:  [B, *spatial]
            w_edge: [B, *spatial]
            w_bone: [B, *spatial]

        When ``anchor_to_centroid`` is enabled, **pull** drives each pixel
        toward a deterministic sinusoidal encoding of the instance's
        spatial center-of-mass (stable fixed target), while **push** and
        **norm** still operate on the empirical mean embedding (for
        gradient flow through the model).
        """
        spatial_shape = embed.shape[2:]
        E = embed.shape[1]

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

        device = embed.device
        loss_pull = torch.tensor(0.0, device=device)
        loss_push = torch.tensor(0.0, device=device)
        loss_norm = torch.tensor(0.0, device=device)
        n_valid = 0

        for b in range(embed.shape[0]):
            lbl_b = lbl_flat[b]                        # [N]
            fg = lbl_b > 0

            if not fg.any():
                continue

            fg_labels = lbl_b[fg]
            unique_ids, inverse = torch.unique(fg_labels, return_inverse=True)
            K = unique_ids.shape[0]
            n_valid += 1

            remap = torch.full_like(lbl_b, -1, dtype=torch.long)
            remap[fg] = inverse

            emb_b = emb_flat[b]                        # [E, N]
            wgt_b = wgt_flat[b] if wgt_flat is not None else None

            mean_centers = self._scatter_weighted_mean(emb_b, remap, wgt_b, K)  # [K, E]

            if self.anchor_to_centroid:
                fg_indices = torch.where(fg)[0]
                spatial_c = self._compute_spatial_centroids(
                    fg_indices, inverse, K, spatial_shape,
                )
                pull_centers = self._sinusoidal_centroid_encoding(
                    spatial_c, spatial_shape, E, self.centroid_scale,
                )
            else:
                pull_centers = mean_centers

            # --- Pull: toward anchored or empirical centers ---
            center_per_pixel = pull_centers[inverse]   # [M, E]
            emb_fg = emb_b[:, fg].T                    # [M, E]

            dist = reduce(
                (emb_fg - center_per_pixel) ** 2, "m e -> m", "sum",
            ).clamp(min=1e-12).sqrt()  # [M]
            pull_per_pixel = (dist - self.delta_v).clamp(min=0).pow(2)
            if wgt_b is not None:
                pull_per_pixel = pull_per_pixel * wgt_b[fg]

            pull_sum = torch.zeros(K, device=device, dtype=torch.float32)
            pull_sum.scatter_add_(0, inverse, pull_per_pixel)
            pull_count = torch.bincount(inverse, minlength=K).float().clamp(min=1)
            b_pull = (pull_sum / pull_count).mean()
            loss_pull = loss_pull + b_pull

            # --- Push: always on empirical mean_centers for gradient ---
            if K > 1:
                pw_diff = (rearrange(mean_centers, "i e -> i 1 e") -
                           rearrange(mean_centers, "j e -> 1 j e"))
                pw = reduce(pw_diff ** 2, "i j e -> i j", "sum").clamp(min=1e-12).sqrt()
                triu = torch.triu_indices(K, K, offset=1, device=device)
                hinge = (2 * self.delta_d - pw[triu[0], triu[1]]).clamp(min=0).pow(2)
                if self.max_hard_pairs > 0 and hinge.numel() > self.max_hard_pairs:
                    hinge, _ = hinge.topk(self.max_hard_pairs)
                loss_push = loss_push + hinge.mean()

            # --- Norm: on empirical mean_centers ---
            if not self.normalize_embeddings:
                loss_norm = loss_norm + reduce(
                    mean_centers ** 2, "k e -> k", "sum",
                ).clamp(min=1e-12).sqrt().mean()

        n = max(n_valid, 1)
        pull = loss_pull / n
        push = loss_push / n
        norm = loss_norm / n
        total = self.weight_pull * pull + self.weight_push * push + self.weight_norm * norm
        return {"loss": total, "pull": pull, "push": push, "norm": norm}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_weights(self, label: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pre-compute boundary + skeleton weights (cache-friendly).

        Returns ``None`` for a weight component when the corresponding
        multiplier is <= 1.0 (disabled), avoiding a full-size ones allocation.
        """
        w_edge = self._get_weight_boundary(label) if self.weight_edge > 1.0 else None
        w_bone = self._get_weight_skeleton(label) if self.weight_bone > 1.0 else None
        return w_edge, w_bone

    def forward(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        weight_edge: Optional[torch.Tensor] = None,
        weight_bone: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if weight_edge is None and weight_bone is None:
            weight_edge, weight_bone = self.compute_weights(label)

        if semantic_ids is not None:
            classes = torch.unique(semantic_ids)
            classes = classes[classes > 0]
            if len(classes) > 0:
                zero = torch.tensor(0.0, device=embed.device)
                acc = {k: zero.clone() for k in ("loss", "pull", "push", "norm")}
                for cid in classes:
                    out = self._loss_single(
                        embed, label * (semantic_ids == cid).long(),
                        weight_edge, weight_bone,
                    )
                    for k in acc:
                        acc[k] = acc[k] + out[k]
                return {k: v / len(classes) for k, v in acc.items()}

        return self._loss_single(embed, label, weight_edge, weight_bone)
