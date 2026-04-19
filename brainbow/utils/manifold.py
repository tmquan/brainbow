"""
Batched per-sample manifold projection for embedding visualisation.

Projects an ``[B, E, N]`` tensor to ``[B, K, N]`` by fitting an independent
dimensionality reducer per batch item.  Used by the TensorBoard callback
to visualise high-dimensional instance embeddings as RGB channels.

Algorithms
----------

* ``pca``  — Principal Component Analysis (centered SVD).  Linear,
  fast, deterministic, preserves global variance.  **Default.**
* ``svd``  — Truncated SVD without centering.  Linear, typically the
  cheapest option but mixes the data mean into the leading component.
* ``umap`` — Uniform Manifold Approximation & Projection.  Non-linear,
  emphasises local structure, can separate instance clusters more
  aggressively at the cost of ~100-1000× more compute.

Backends
--------

* ``cuml``        — RAPIDS GPU (``cuml.decomposition.PCA`` /
  ``TruncatedSVD`` / ``cuml.manifold.UMAP``).  Fastest on CUDA inputs.
* ``torch``       — Batched ``torch.linalg.svd`` (pca/svd only).
* ``umap-learn``  — CPU ``umap.UMAP`` (umap algorithm only).
* ``auto``        — cuML when the input is CUDA and RAPIDS is available,
  else the best CPU fallback for the chosen algorithm.

All backends return projections in the same device / dtype as the input.
Column signs and per-component scales are **not** guaranteed to match
across backends — consumers should apply min-max normalisation when
using the output as display channels.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from einops import rearrange, reduce


# ---------------------------------------------------------------------------
# Backend probes (memoised)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _probe_cupy() -> Optional[Any]:
    try:
        import cupy as cp
    except Exception:
        return None
    return cp


@lru_cache(maxsize=1)
def _probe_cuml_pca() -> Optional[Any]:
    if _probe_cupy() is None:
        return None
    try:
        from cuml.decomposition import PCA as _CuPCA
    except Exception:
        return None
    return _CuPCA


@lru_cache(maxsize=1)
def _probe_cuml_svd() -> Optional[Any]:
    if _probe_cupy() is None:
        return None
    try:
        from cuml.decomposition import TruncatedSVD as _CuSVD
    except Exception:
        return None
    return _CuSVD


@lru_cache(maxsize=1)
def _probe_cuml_umap() -> Optional[Any]:
    if _probe_cupy() is None:
        return None
    try:
        from cuml.manifold import UMAP as _CuUMAP
    except Exception:
        return None
    return _CuUMAP


@lru_cache(maxsize=1)
def _probe_umap_learn() -> Optional[Any]:
    try:
        import umap
    except Exception:
        return None
    return umap.UMAP


_VALID_ALGOS = ("pca", "svd", "umap")
_VALID_BACKENDS = ("auto", "cuml", "torch", "umap-learn")


# ---------------------------------------------------------------------------
# Backend dispatchers
# ---------------------------------------------------------------------------

def _pad_components(proj: torch.Tensor, n_components: int) -> torch.Tensor:
    """Zero-pad ``[B, k, N]`` to ``[B, n_components, N]`` when ``k < n_components``."""
    k = proj.shape[1]
    if k >= n_components:
        return proj[:, :n_components]
    pad = torch.zeros(
        proj.shape[0], n_components - k, proj.shape[2],
        device=proj.device, dtype=proj.dtype,
    )
    return torch.cat([proj, pad], dim=1)


def _pca_torch(flat: torch.Tensor, n_components: int, center: bool) -> torch.Tensor:
    """Batched torch-SVD PCA (``center=True``) or truncated SVD (``center=False``)."""
    x = flat
    if center:
        x = flat - reduce(flat, "b e n -> b e 1", "mean")
    try:
        _, _, Vh = torch.linalg.svd(x, full_matrices=False)    # Vh: [B, min(E,N), N]
        proj = Vh[:, :n_components]
    except (torch._C._LinAlgError, RuntimeError):
        proj = x[:, :n_components]
    return _pad_components(proj, n_components)


def _pca_cuml(flat: torch.Tensor, n_components: int, center: bool) -> torch.Tensor:
    """cuML GPU PCA / TruncatedSVD; ``flat`` must live on CUDA."""
    import cupy as cp
    estimator_cls = _probe_cuml_pca() if center else _probe_cuml_svd()
    assert estimator_cls is not None

    B, E, N = flat.shape
    k = min(n_components, E, N)
    out = torch.zeros(B, n_components, N, device=flat.device, dtype=flat.dtype)
    for b in range(B):
        samples = rearrange(flat[b], "e n -> n e").contiguous()
        x_cp = cp.from_dlpack(samples)
        try:
            est = estimator_cls(n_components=k)
            scores_cp = est.fit_transform(x_cp)               # [N, k]
            scores = torch.from_dlpack(scores_cp)
            out[b, :k] = rearrange(scores, "n k -> k n")
        except Exception:
            out[b, :k] = flat[b, :k]
    return out


def _umap_cuml(flat: torch.Tensor, n_components: int, umap_kwargs: Dict[str, Any]) -> torch.Tensor:
    """cuML GPU UMAP; ``flat`` must live on CUDA."""
    import cupy as cp
    _CuUMAP = _probe_cuml_umap()
    assert _CuUMAP is not None

    B, E, N = flat.shape
    out = torch.zeros(B, n_components, N, device=flat.device, dtype=flat.dtype)
    defaults = {
        "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean",
        "random_state": 0, "verbose": False,
    }
    defaults.update(umap_kwargs or {})
    for b in range(B):
        samples = rearrange(flat[b], "e n -> n e").contiguous()
        x_cp = cp.from_dlpack(samples)
        try:
            reducer = _CuUMAP(n_components=n_components, **defaults)
            scores_cp = reducer.fit_transform(x_cp)           # [N, n_components]
            scores = torch.from_dlpack(cp.ascontiguousarray(scores_cp))
            out[b] = rearrange(scores, "n k -> k n")
        except Exception:
            # UMAP can fail on degenerate inputs (e.g. N < n_neighbors);
            # fall back to PCA for that batch item.
            fb = _pca_cuml(flat[b:b + 1], n_components, center=True)
            out[b] = fb[0]
    return out


def _umap_cpu(flat: torch.Tensor, n_components: int, umap_kwargs: Dict[str, Any]) -> torch.Tensor:
    """CPU UMAP via umap-learn; moves data through numpy."""
    _UMAP = _probe_umap_learn()
    assert _UMAP is not None

    B, E, N = flat.shape
    out = torch.zeros(B, n_components, N, device=flat.device, dtype=flat.dtype)
    defaults = {
        "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean",
        "random_state": 0, "verbose": False,
    }
    defaults.update(umap_kwargs or {})
    flat_cpu = flat.detach().cpu().numpy()
    for b in range(B):
        samples = rearrange(flat_cpu[b], "e n -> n e")
        try:
            reducer = _UMAP(n_components=n_components, **defaults)
            scores = reducer.fit_transform(samples)           # [N, n_components]
            out[b] = rearrange(
                torch.from_numpy(scores).to(device=flat.device, dtype=flat.dtype),
                "n k -> k n",
            )
        except Exception:
            fb = _pca_torch(flat[b:b + 1], n_components, center=True)
            out[b] = fb[0]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project_batch(
    flat: torch.Tensor,
    n_components: int = 3,
    algorithm: str = "pca",
    backend: str = "auto",
    umap_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Per-sample manifold projection ``[B, E, N]`` → ``[B, n_components, N]``.

    Args:
        flat: Input tensor ``[B, E, N]`` on any device.  For image/volume
            embeddings, flatten the spatial axes before passing in.
        n_components: Number of components / output channels.  Extra
            components (beyond ``min(E, N)`` for pca/svd) are zero-filled.
        algorithm: ``"pca"`` (default), ``"svd"``, or ``"umap"``.
        backend: ``"auto"``, ``"cuml"``, ``"torch"``, or ``"umap-learn"``.
            ``"auto"`` picks cuML when available and the input lives on
            CUDA, else the best CPU fallback for the chosen algorithm.
        umap_kwargs: Extra kwargs forwarded to the UMAP estimator
            (``n_neighbors``, ``min_dist``, ``metric``, ...).  Ignored for
            non-UMAP algorithms.

    Returns:
        Projected tensor of shape ``[B, n_components, N]`` on the same
        device and dtype as ``flat``.
    """
    if flat.dim() != 3:
        raise ValueError(f"project_batch expects [B, E, N]; got {tuple(flat.shape)}")
    if algorithm not in _VALID_ALGOS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {_VALID_ALGOS}.")
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {_VALID_BACKENDS}.")

    on_cuda = flat.is_cuda
    umap_kwargs = umap_kwargs or {}

    if algorithm in ("pca", "svd"):
        center = (algorithm == "pca")
        if backend == "torch":
            return _pca_torch(flat, n_components, center=center)
        if backend == "umap-learn":
            raise ValueError("backend='umap-learn' is only valid with algorithm='umap'.")

        cuml_available = (
            _probe_cuml_pca() if center else _probe_cuml_svd()
        ) is not None

        if backend == "cuml":
            if not cuml_available:
                raise RuntimeError(f"backend='cuml' requested for {algorithm} but cuML not available.")
            if not on_cuda:
                raise RuntimeError(f"backend='cuml' requires a CUDA input tensor; got device={flat.device}.")
            return _pca_cuml(flat, n_components, center=center)

        # auto
        if on_cuda and cuml_available:
            return _pca_cuml(flat, n_components, center=center)
        return _pca_torch(flat, n_components, center=center)

    if algorithm == "umap":
        cuml_available = _probe_cuml_umap() is not None
        cpu_available = _probe_umap_learn() is not None

        if backend == "cuml":
            if not cuml_available:
                raise RuntimeError("backend='cuml' requested for UMAP but cuml.manifold.UMAP not available.")
            if not on_cuda:
                raise RuntimeError(f"backend='cuml' requires a CUDA input tensor; got device={flat.device}.")
            return _umap_cuml(flat, n_components, umap_kwargs)

        if backend in ("torch", "umap-learn"):
            if not cpu_available:
                raise RuntimeError("backend='umap-learn' requested but the `umap-learn` package is not installed.")
            return _umap_cpu(flat, n_components, umap_kwargs)

        # auto
        if on_cuda and cuml_available:
            return _umap_cuml(flat, n_components, umap_kwargs)
        if cpu_available:
            return _umap_cpu(flat, n_components, umap_kwargs)
        # No UMAP implementation installed — degrade silently to PCA so the
        # visualisation pipeline never crashes on a missing optional dep.
        return _pca_torch(flat, n_components, center=True)

    raise AssertionError(algorithm)  # pragma: no cover


def available_projectors() -> Dict[str, Dict[str, bool]]:
    """Report which projection backends are installed on this machine."""
    return {
        "pca": {
            "cuml": _probe_cuml_pca() is not None,
            "torch": True,
        },
        "svd": {
            "cuml": _probe_cuml_svd() is not None,
            "torch": True,
        },
        "umap": {
            "cuml": _probe_cuml_umap() is not None,
            "umap-learn": _probe_umap_learn() is not None,
        },
    }


__all__ = ["project_batch", "available_projectors"]
