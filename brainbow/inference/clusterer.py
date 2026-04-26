"""
Clustering modules for instance segmentation.

All embedding clusterers share a common ``nn.Module`` interface::

    clusterer(embedding, foreground_mask) ->
        labels:      [B, *spatial]        long   -- 0=background, 1..K=instances
        soft_assign: [B, K, *spatial]     float  -- soft probabilities (may be empty)
        centers:     [B, K, E]            float  -- cluster centers      (may be empty)

so they are drop-in swappable at validation / inference time.

Implementations
---------------

``SoftMeanShift``
    Differentiable temperature-scaled Gaussian mean-shift.  The only
    clusterer usable during training (preserves gradients) and the
    historical default.

``MeanShiftClusterer``
    Non-differentiable MeanShift wrapper around
    :func:`brainbow.utils.clustering.cluster_embeddings`.  Uses cuML when
    available (note: RAPIDS dropped ``MeanShift`` in cuML 23.x, so this
    currently falls back to scikit-learn on recent cuML installs).

``HDBSCANClusterer``
    Non-differentiable HDBSCAN wrapper around
    :func:`brainbow.utils.clustering.cluster_embeddings`.  Uses cuML GPU
    HDBSCAN when available (≈4× faster than SoftMeanShift on real
    validation patches), else the ``hdbscan`` package, else
    ``sklearn.cluster.HDBSCAN``.  Auto-determines ``K``.

``SpatialCCClusterer``
    Non-differentiable connected-components clusterer over the
    spatial-neighbour embedding-affinity graph (two neighbouring
    voxels are linked iff their embedding distance is below
    ``delta_v``).  Respects spatial connectivity — unrelated cells
    with similar mean embeddings cannot merge.  Uses
    ``cupyx.scipy.sparse.csgraph.connected_components`` on CUDA
    (zero-copy via DLPack), else :mod:`scipy.sparse.csgraph`.

``HoughVoting``
    Offset-based instance segmentation (not embedding based).  Kept here
    because it shares the "turn dense predictions into instance labels"
    role of the embedding clusterers, not because it shares their API.

``build_clusterer(name, **kw)``
    Factory that returns the appropriate clusterer from a config string.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat

from brainbow.utils.clustering import DeltaV, _reshape_to_spatial

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Differentiable soft mean-shift (training default)
# ---------------------------------------------------------------------------


class SoftMeanShift(nn.Module):
    """Differentiable mean-shift clustering on pixel/voxel embeddings.

    Iteratively refines K mode estimates using Gaussian-kernel weighted
    averages.  Soft assignments allow gradient flow during training;
    temperature annealing sharpens assignments for inference.

    Args:
        bandwidth: Gaussian kernel bandwidth (related to delta_var).
        num_iters: Number of mean-shift refinement iterations.
        temperature: Softmax temperature for assignment (lower = harder).
        min_cluster_size: Discard clusters smaller than this.
        normalize_embeddings: L2-normalise embeddings before clustering.
    """

    def __init__(
        self,
        bandwidth: float = 0.5,
        num_iters: int = 10,
        temperature: float = 1.0,
        min_cluster_size: int = 50,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.num_iters = num_iters
        self.temperature = temperature
        self.min_cluster_size = min_cluster_size
        self.normalize_embeddings = normalize_embeddings

    def _init_seeds(
        self,
        emb_flat: torch.Tensor,
        fg_mask: torch.Tensor,
        max_seeds: int = 256,
    ) -> torch.Tensor:
        """Subsample foreground embeddings as initial mode seeds.

        Args:
            emb_flat: [E, N] embeddings.
            fg_mask: [N] boolean foreground mask.
            max_seeds: Maximum number of seeds.

        Returns:
            [K, E] initial mode estimates.
        """
        fg_idx = torch.where(fg_mask)[0]
        if len(fg_idx) == 0:
            return rearrange(emb_flat[:, :1], "e 1 -> 1 e")
        n = min(max_seeds, len(fg_idx))
        perm = torch.randperm(len(fg_idx), device=fg_idx.device)[:n]
        return rearrange(emb_flat[:, fg_idx[perm]], "e k -> k e")

    def forward(
        self,
        embedding: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
        max_seeds: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cluster embeddings via differentiable mean-shift.

        Args:
            embedding: [B, E, *spatial] embedding tensor.
            foreground_mask: [B, *spatial] boolean mask (optional).
            max_seeds: Max initial seeds per sample.

        Returns:
            labels:       [B, *spatial] integer instance labels (0 = bg).
            soft_assign:  [B, K, *spatial] soft assignment probabilities.
            centers:      [B, K, E] final cluster centres.
        """
        B = embedding.shape[0]
        E = embedding.shape[1]
        spatial_shape = embedding.shape[2:]
        device = embedding.device

        emb_flat = rearrange(embedding, "b e ... -> b e (...)")
        if self.normalize_embeddings:
            emb_flat = F.normalize(emb_flat, dim=1, eps=1e-6)
        N = emb_flat.shape[2]

        if foreground_mask is not None:
            fg_flat = rearrange(foreground_mask, "b ... -> b (...)") > 0
        else:
            fg_flat = torch.ones(B, N, device=device, dtype=torch.bool)

        all_labels = []
        all_soft = []
        all_centers = []

        for b in range(B):
            fg_b = fg_flat[b]
            emb_b = emb_flat[b]

            if fg_b.sum() == 0:
                all_labels.append(torch.zeros(N, device=device, dtype=torch.long))
                all_soft.append(torch.zeros(1, N, device=device))
                all_centers.append(torch.zeros(1, E, device=device))
                continue

            modes = self._init_seeds(emb_b, fg_b, max_seeds)
            K = modes.shape[0]

            # --- Mean-shift iteration: refine modes via Gaussian-weighted average ---
            for _ in range(self.num_iters):
                emb_fg = emb_b[:, fg_b]                           # [E, M] foreground only
                sq_dist = self._sq_dist(modes, emb_fg)             # [K, M]

                # Gaussian kernel: weight each pixel by proximity to each mode
                weights = torch.exp(-sq_dist / (2 * self.bandwidth ** 2))
                weights_sum = reduce(weights, "k m -> k 1", "sum").clamp(min=1e-8)

                # Weighted mean update: each mode moves toward its kernel-weighted centroid
                modes = einsum(weights, emb_fg, "k m, e m -> k e") / weights_sum

                # Merge modes that have converged close together
                merged = self._merge_modes(modes)
                if merged.shape[0] < modes.shape[0]:
                    modes = merged
                    K = modes.shape[0]

            # --- Soft assignment: Gaussian distance to each final mode ---
            sq_dist_all = self._sq_dist(modes, emb_b)             # [K, N]
            logits = -sq_dist_all / (2 * self.bandwidth ** 2 * self.temperature)
            soft = F.softmax(logits, dim=0)                        # [K, N]

            # Hard assignment: argmax + 1 (0 reserved for background)
            hard = soft.argmax(dim=0) + 1
            hard[~fg_b] = 0
            hard = self._filter_small_clusters(hard, K)

            all_labels.append(hard)
            all_soft.append(soft)
            all_centers.append(modes)

        labels = _reshape_to_spatial(torch.stack(all_labels), spatial_shape)

        max_K = max(s.shape[0] for s in all_soft)
        padded_soft = []
        padded_centers = []
        for s, c in zip(all_soft, all_centers):
            k = s.shape[0]
            if k < max_K:
                s = F.pad(s, (0, 0, 0, max_K - k))
                c = F.pad(c, (0, 0, 0, max_K - k))
            padded_soft.append(s)
            padded_centers.append(c)

        soft_assign = _reshape_to_spatial(torch.stack(padded_soft), spatial_shape)
        centers = torch.stack(padded_centers)

        return labels, soft_assign, centers

    @staticmethod
    def _sq_dist(centers: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Memory-efficient squared Euclidean distance.

        Uses ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ to avoid materializing
        the [K, E, M] difference tensor.

        Args:
            centers: [K, E] cluster centres.
            points:  [E, M] data points.

        Returns:
            [K, M] squared distances.
        """
        c_sq = reduce(centers ** 2, "k e -> k 1", "sum")
        p_sq = reduce(points ** 2, "e m -> 1 m", "sum")
        cross = einsum(centers, points, "k e, e m -> k m")
        return (c_sq + p_sq - 2 * cross).clamp(min=0.0)

    def _merge_modes(
        self, modes: torch.Tensor, factor: float = 0.5,
    ) -> torch.Tensor:
        """Greedily merge modes closer than ``factor * bandwidth``.

        Keeps the first mode in each cluster of near-duplicates and
        discards the rest.  Uses vectorised mask updates.
        """
        if modes.shape[0] <= 1:
            return modes
        pw = torch.cdist(modes, modes)
        threshold = self.bandwidth * factor
        K = modes.shape[0]
        keep = torch.ones(K, device=modes.device, dtype=torch.bool)
        for i in range(K):
            if not keep[i]:
                continue
            dups = (pw[i] < threshold) & keep
            dups[i] = False
            keep[dups] = False
        return modes[keep]

    def _filter_small_clusters(
        self, labels: torch.Tensor, K: int,
    ) -> torch.Tensor:
        """Set clusters with fewer than ``min_cluster_size`` pixels to 0."""
        labels = labels.clone()
        for uid in range(1, K + 1):
            mask = labels == uid
            if mask.sum() < self.min_cluster_size:
                labels[mask] = 0
        return labels


# ---------------------------------------------------------------------------
# Non-differentiable clusterers (eval / inference)
# ---------------------------------------------------------------------------


class _BaseUnsupervisedClusterer(nn.Module):
    """Shared ``forward`` for batch-wise non-differentiable clusterers.

    Subclasses implement ``_cluster_single`` returning a ``[*spatial]``
    ``torch.long`` label tensor for one sample.  The batch loop + empty
    ``(soft_assign, centers)`` placeholders are handled here.
    """

    #: For logging / introspection.
    algorithm: str = ""

    def _cluster_single(
        self, embedding: torch.Tensor, foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cluster each sample in the batch independently.

        Args:
            embedding: ``[B, E, *spatial]`` embedding tensor.
            foreground_mask: Optional ``[B, *spatial]`` boolean mask.

        Returns:
            labels:       ``[B, *spatial]`` ``long``.
            soft_assign:  ``[B, 0, *spatial]`` empty placeholder (this
                          clusterer does not produce soft scores).
            centers:      ``[B, 0, E]`` empty placeholder.
        """
        if embedding.dim() < 3:
            raise ValueError(
                f"{type(self).__name__} expects a batched [B, E, *spatial] "
                f"tensor; got shape {tuple(embedding.shape)}."
            )

        B, E = embedding.shape[:2]
        spatial_shape = embedding.shape[2:]
        device = embedding.device

        if foreground_mask is None:
            foreground_mask = torch.ones(
                B, *spatial_shape, device=device, dtype=torch.bool,
            )

        labels_out = torch.zeros(B, *spatial_shape, device=device, dtype=torch.long)
        for b in range(B):
            fg_b = foreground_mask[b] > 0
            if not bool(fg_b.any()):
                continue
            labels_out[b] = self._cluster_single(embedding[b], fg_b)

        # Placeholders: downstream code (modules/*.py, tensorboard callback)
        # only reads the first element of the tuple.
        soft_assign = embedding.new_zeros((B, 0, *spatial_shape))
        centers = embedding.new_zeros((B, 0, E))
        return labels_out, soft_assign, centers


class HDBSCANClusterer(_BaseUnsupervisedClusterer):
    """Non-differentiable HDBSCAN clusterer (cuML GPU → CPU fallback).

    HDBSCAN auto-determines ``K`` and handles variable-density
    embeddings gracefully, making it a strong validation-time
    alternative to SoftMeanShift.

    Args:
        bandwidth: Used as ``cluster_selection_epsilon`` by default so
            the density threshold matches the discriminative-loss pull
            margin (``delta_v``).
        min_cluster_size: HDBSCAN ``min_cluster_size`` in sample space
            (i.e. relative to the subsampled point count, not the full
            volume).
        min_samples: HDBSCAN ``min_samples``; ``None`` → matches
            ``min_cluster_size``.
        cluster_selection_epsilon: Optional explicit epsilon; overrides
            ``bandwidth``.
        normalize_embeddings: L2-normalise embeddings before clustering.
            Must match the flag used at training time.
        max_points: HDBSCAN scales ~O(N²); foreground sets larger than
            this are uniformly subsampled and the remaining points are
            assigned by nearest-center propagation.  50k is a good
            default on a single GPU.
        backend: ``"auto"`` (cuML → ``hdbscan`` pkg → sklearn),
            ``"cuml"``, ``"hdbscan"``, or ``"sklearn"``.
        seed: RNG seed for subsampling reproducibility.
    """

    algorithm = "hdbscan"

    def __init__(
        self,
        bandwidth: float = 0.5,
        min_cluster_size: int = 50,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: Optional[float] = None,
        normalize_embeddings: bool = False,
        max_points: int = 50_000,
        backend: str = "auto",
        seed: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.normalize_embeddings = normalize_embeddings
        self.max_points = max_points
        self.backend = backend
        self.seed = seed

    def _cluster_single(
        self, embedding: torch.Tensor, foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Imported here to avoid a circular import at module load time
        # (utils.clustering references clusterer.SoftMeanShift).
        from brainbow.utils.clustering import cluster_embeddings

        return cluster_embeddings(
            embedding,
            foreground_mask=foreground_mask,
            algorithm="hdbscan",
            bandwidth=self.bandwidth,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            normalize_embeddings=self.normalize_embeddings,
            backend=self.backend,
            max_points=self.max_points,
            seed=self.seed,
        )


class SpatialCCClusterer(_BaseUnsupervisedClusterer):
    """Connected components on spatial-neighbour embedding affinities.

    Two foreground voxels are linked iff they are spatial neighbours
    (face-connectivity by default) and their embeddings are within
    ``delta_v`` in Euclidean distance.  Instance labels are the
    connected components of the resulting sparse graph.

    Strengths vs HDBSCAN / MeanShift:

    - **Spatial-connectivity aware.**  Two unrelated cells with
      similar mean embeddings cannot merge — there is no chain of
      spatial neighbours between them.
    - **Linear time.**  No subsampling, no ``K`` selection, one
      threshold (= training's ``delta_v``).  Runs on the native
      volume at full resolution.
    - **GPU.**  Uses ``cupyx.scipy.sparse.csgraph.connected_components``
      with zero-copy torch ↔ cupy DLPack handoff on CUDA; falls back
      to :mod:`scipy.sparse.csgraph` on CPU.

    Known failure mode: a single mis-predicted embedding voxel at a
    cell-cell boundary can leak two instances into one.  Mitigate by
    tightening ``delta_v`` below the trained margin.

    Args:
        bandwidth: Embedding-distance threshold for linking spatial
            neighbours.  Accepts either:

            - a single scalar (isotropic — identical to the original
              clusterer), or
            - a length-``n_dims`` sequence giving a separate threshold
              per spatial axis.  Use a tighter Z entry on anisotropic
              EM (e.g. ``[0.2, 0.5, 0.5]`` on 5:1 Z:XY) so that
              touching-but-distinct neurons across Z never merge.

            Semantically equal to the training pull margin
            (``delta_v``); named ``bandwidth`` to match the other
            clusterers' common kwarg so the module-layer config
            wiring (``clusterer.bandwidth``) is identical.
        min_cluster_size: Clusters smaller than this are dropped
            (mapped to background).  Applied at full resolution.
        connectivity: ``1`` for face-connectivity (6 in 3D, 4 in 2D);
            any other value enables full connectivity (26 / 8).
        normalize_embeddings: L2-normalise embeddings before computing
            distances.  Must match the flag used at training time.
        backend: Path for the connected-components pass:

            - ``"auto"``:  self-contained UF kernel on CUDA, else scipy.
            - ``"self"``:  force our self-contained CuPy union-find
                           kernel (no ``pylibcugraph`` dependency;
                           fastest).
            - ``"cupy"``:  cupyx.scipy.sparse.csgraph (needs
                           ``pylibcugraph``; kept for debugging).
            - ``"scipy"``: CPU round-trip.
    """

    algorithm = "spatial_cc"

    def __init__(
        self,
        bandwidth: DeltaV = 0.5,
        min_cluster_size: int = 50,
        connectivity: int = 1,
        normalize_embeddings: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        # Keep raw value for introspection/logging; conversion to a
        # per-axis list happens inside `cluster_spatial_cc` where the
        # spatial rank is known.  Accept both Python scalars and
        # Omegaconf ListConfig objects (Hydra-materialised YAML lists).
        if hasattr(bandwidth, "_content"):  # OmegaConf ListConfig
            bandwidth = list(bandwidth)
        if isinstance(bandwidth, (list, tuple)):
            bandwidth = [float(x) for x in bandwidth]
        self.bandwidth: DeltaV = bandwidth
        self.min_cluster_size = min_cluster_size
        self.connectivity = connectivity
        self.normalize_embeddings = normalize_embeddings
        self.backend = backend

    @property
    def delta_v(self) -> DeltaV:
        """Alias: the affinity threshold _is_ the trained pull margin."""
        return self.bandwidth

    def _cluster_single(
        self, embedding: torch.Tensor, foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        from brainbow.utils.clustering import cluster_spatial_cc

        return cluster_spatial_cc(
            embedding,
            foreground_mask=foreground_mask,
            delta_v=self.bandwidth,
            min_cluster_size=self.min_cluster_size,
            connectivity=self.connectivity,
            normalize_embeddings=self.normalize_embeddings,
            backend=self.backend,
        )


class MeanShiftClusterer(_BaseUnsupervisedClusterer):
    """Non-differentiable MeanShift clusterer (cuML → sklearn).

    Note:
        RAPIDS dropped :class:`cuml.cluster.MeanShift` in cuML 23.x.  On
        modern cuML builds this clusterer transparently falls back to
        :class:`sklearn.cluster.MeanShift` (CPU).  For a GPU-accelerated
        alternative prefer :class:`HDBSCANClusterer`.

    Args:
        bandwidth: MeanShift bandwidth (typically equal to
            discriminative-loss ``delta_v``).
        bin_seeding: Discretize initial seeds onto a grid for speed.
        min_cluster_size: Clusters smaller than this are discarded.
        normalize_embeddings: L2-normalise embeddings before clustering.
        max_points: Foreground sets larger than this are uniformly
            subsampled (MeanShift scales ~O(N²)).
        backend: ``"auto"`` | ``"cuml"`` | ``"sklearn"``.
        seed: RNG seed for subsampling reproducibility.
    """

    algorithm = "meanshift"

    def __init__(
        self,
        bandwidth: float = 0.5,
        bin_seeding: bool = True,
        min_cluster_size: int = 50,
        normalize_embeddings: bool = False,
        max_points: int = 50_000,
        backend: str = "auto",
        seed: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.bin_seeding = bin_seeding
        self.min_cluster_size = min_cluster_size
        self.normalize_embeddings = normalize_embeddings
        self.max_points = max_points
        self.backend = backend
        self.seed = seed

    def _cluster_single(
        self, embedding: torch.Tensor, foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        from brainbow.utils.clustering import cluster_embeddings

        return cluster_embeddings(
            embedding,
            foreground_mask=foreground_mask,
            algorithm="meanshift",
            bandwidth=self.bandwidth,
            min_cluster_size=self.min_cluster_size,
            normalize_embeddings=self.normalize_embeddings,
            backend=self.backend,
            max_points=self.max_points,
            bin_seeding=self.bin_seeding,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Offset-based Hough voting (unchanged, kept here for API symmetry)
# ---------------------------------------------------------------------------


class HoughVoting(nn.Module):
    """Differentiable Hough voting for offset-based instance segmentation.

    Each foreground pixel votes for a spatial location by adding its
    predicted offset to its coordinate.  Votes are accumulated into a
    smooth vote map via Gaussian splatting, then peaks are detected as
    instance centres.

    Args:
        bin_size: Spatial bin size for the vote accumulator.
        sigma: Gaussian sigma for vote splatting (in voxels).
        threshold: Relative peak threshold (fraction of max vote).
        min_votes: Minimum votes for a valid peak.
    """

    def __init__(
        self,
        bin_size: float = 2.0,
        sigma: float = 2.0,
        threshold: float = 0.3,
        min_votes: int = 50,
    ) -> None:
        super().__init__()
        self.bin_size = bin_size
        self.sigma = sigma
        self.threshold = threshold
        self.min_votes = min_votes

    @staticmethod
    def _make_coords(spatial_shape, device):
        """Build [S, *spatial] coordinate grid."""
        ranges = [torch.arange(s, device=device, dtype=torch.float32)
                  for s in spatial_shape]
        grids = torch.meshgrid(*ranges, indexing="ij")
        return torch.stack(list(reversed(grids)), dim=0)

    def forward(
        self,
        offsets: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Cluster via Hough voting on predicted offsets.

        Args:
            offsets: [B, S, *spatial] predicted spatial offsets.
            foreground_mask: [B, *spatial] boolean mask (optional).

        Returns:
            ``(labels, soft_assign, centers)`` to honour the shared
            :class:`_BaseUnsupervisedClusterer` contract.  Hough voting
            is hard (no soft assignment) and bin-coordinate based (no
            embedding-space centers), so the latter two are always
            ``None``.

            * ``labels``: ``[B, *spatial]`` long tensor (``0`` = bg).
            * ``soft_assign``: ``None``.
            * ``centers``: ``None``.
        """
        B, S = offsets.shape[:2]
        spatial_shape = offsets.shape[2:]
        device = offsets.device

        coords = self._make_coords(spatial_shape, device)
        votes = repeat(coords, "s ... -> b s ...", b=B) + offsets

        if foreground_mask is None:
            foreground_mask = torch.ones(B, *spatial_shape, device=device, dtype=torch.bool)

        all_labels = []
        for b in range(B):
            fg = foreground_mask[b]
            vote_flat = rearrange(votes[b], "s ... -> s (...)")[
                :, rearrange(fg, "... -> (...)")]

            if vote_flat.shape[1] == 0:
                all_labels.append(torch.zeros(spatial_shape, device=device, dtype=torch.long))
                continue

            bins = (vote_flat / self.bin_size).round().long()

            bin_min = reduce(bins, "s m -> s", "min")
            bins_shifted = bins - rearrange(bin_min, "s -> s 1")
            bin_max = reduce(bins_shifted, "s m -> s", "max") + 1

            acc_shape = tuple(bin_max.tolist())
            accumulator = torch.zeros(acc_shape, device=device, dtype=torch.float32)

            flat_idx = torch.zeros(bins_shifted.shape[1], device=device, dtype=torch.long)
            stride = 1
            for dim_i in range(S - 1, -1, -1):
                flat_idx = flat_idx + bins_shifted[dim_i] * stride
                stride *= acc_shape[dim_i]
            ones = torch.ones(flat_idx.shape[0], device=device, dtype=torch.float32)
            rearrange(accumulator, "... -> (...)").scatter_add_(0, flat_idx, ones)

            # Optional box-filter smoothing (unified 2D / 3D)
            if self.sigma > 0:
                k = int(3 * self.sigma) * 2 + 1
                conv_fn = F.conv3d if S == 3 else F.conv2d
                kernel = torch.ones((1, 1) + (k,) * S, device=device) / (k ** S)
                acc_nd = rearrange(accumulator, "... -> 1 1 ...")
                smoothed = conv_fn(acc_nd, kernel, padding=k // 2)
                accumulator = rearrange(smoothed, "1 1 ... -> ...")

            peak_threshold = accumulator.max() * self.threshold
            peaks_mask = accumulator >= max(peak_threshold, self.min_votes)

            if peaks_mask.sum() == 0:
                all_labels.append(torch.zeros(spatial_shape, device=device, dtype=torch.long))
                continue

            peak_coords = torch.nonzero(peaks_mask, as_tuple=False).float()

            fg_indices = torch.where(rearrange(fg, "... -> (...)"))[0]
            fg_bins = bins_shifted.float().T

            dists = torch.cdist(fg_bins, peak_coords)
            nearest = dists.argmin(dim=1) + 1

            label_flat = torch.zeros(fg.numel(), device=device, dtype=torch.long)
            label_flat[fg_indices] = nearest
            all_labels.append(_reshape_to_spatial(label_flat, spatial_shape))

        return torch.stack(all_labels), None, None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_CLUSTERER_REGISTRY: Dict[str, type] = {
    "soft_meanshift": SoftMeanShift,
    "softmeanshift": SoftMeanShift,   # convenience alias
    "meanshift": MeanShiftClusterer,
    "hdbscan": HDBSCANClusterer,
    "spatial_cc": SpatialCCClusterer,
    "spatialcc": SpatialCCClusterer,   # convenience alias
    "cc": SpatialCCClusterer,           # convenience alias
}


def build_clusterer(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a clusterer from its short name.

    Args:
        name: One of ``"soft_meanshift"`` (default, differentiable),
            ``"meanshift"``, ``"hdbscan"``, or ``"spatial_cc"``.
        **kwargs: Forwarded to the selected class's constructor.

    Returns:
        An ``nn.Module`` whose ``forward`` signature matches
        :class:`SoftMeanShift` — i.e. returns
        ``(labels, soft_assign, centers)``.

    Raises:
        ValueError: If ``name`` does not resolve to a known clusterer.
    """
    key = name.lower().strip()
    if key not in _CLUSTERER_REGISTRY:
        raise ValueError(
            f"Unknown clusterer {name!r}. Choose one of "
            f"{sorted(set(_CLUSTERER_REGISTRY))}."
        )
    cls = _CLUSTERER_REGISTRY[key]

    # Drop kwargs that the selected class does not accept so that users
    # can switch `name:` between clusterers without pruning every
    # adjacent option from their config (e.g. `backend`, `connectivity`
    # are irrelevant to `SoftMeanShift`).
    sig = inspect.signature(cls.__init__)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not accepts_var_kw:
        accepted = {
            n for n, p in sig.parameters.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ) and n != "self"
        }
        ignored = {k: v for k, v in kwargs.items() if k not in accepted}
        if ignored:
            logger.warning(
                "build_clusterer: %s does not accept %s; ignoring.",
                cls.__name__, sorted(ignored),
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**kwargs)


__all__ = [
    "SoftMeanShift",
    "MeanShiftClusterer",
    "HDBSCANClusterer",
    "SpatialCCClusterer",
    "HoughVoting",
    "build_clusterer",
]
