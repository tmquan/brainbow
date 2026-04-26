"""
Cross-cutting utilities used throughout brainbow.

Why this package exists
-----------------------
A small bag of helpers that don't fit cleanly into ``datasets`` /
``transforms`` / ``inference``: volume I/O, parallel-map for CPU-bound
target construction, embedding-clustering primitives shared by
training-time losses and inference-time clusterers, and manifold
projection helpers (UMAP / PCA / t-SNE) for embedding diagnostics.

Public surface
--------------
I/O (:mod:`brainbow.utils.io`)
    * :func:`find_folder` -- recursively locate a file by name or
      extension under a search root.
    * :func:`load_volume` / :func:`save_volume` -- format-agnostic
      reader / writer dispatching to :mod:`brainbow.preprocessors`.

Clustering (:mod:`brainbow.utils.clustering`)
    * :func:`cluster_embeddings` -- generic dispatch (mean-shift /
      HDBSCAN / soft mean-shift) over a per-voxel embedding tensor.
    * :func:`cluster_embeddings_meanshift` /
      :func:`cluster_embeddings_hdbscan` /
      :func:`cluster_embeddings_soft` -- backend-specific entry points.
    * :func:`cluster_offsets_hough` -- Hough voting on instance offsets.
    * :func:`available_backends` -- which clustering backends are
      installed in the current environment.

Lower-level helpers
    * :mod:`brainbow.utils.parallel` -- ``pmap`` (forkserver-based
      parallel map for CPU work, gracefully falls back to sequential).
    * :mod:`brainbow.utils.manifold` -- ``project_batch`` for embedding
      previews in TensorBoard.

Extending this module
---------------------
Anything that is reused by **two or more** subpackages and has no
better home belongs here.  Anything used by exactly one subpackage
should live next to its consumer.
"""

from brainbow.utils.io import find_folder, load_volume, save_volume
from brainbow.utils.clustering import (
    available_backends,
    cluster_embeddings,
    cluster_embeddings_hdbscan,
    cluster_embeddings_meanshift,
    cluster_embeddings_soft,
    cluster_offsets_hough,
)

__all__ = [
    "find_folder",
    "load_volume",
    "save_volume",
    "available_backends",
    "cluster_embeddings",
    "cluster_embeddings_hdbscan",
    "cluster_embeddings_meanshift",
    "cluster_embeddings_soft",
    "cluster_offsets_hough",
]
