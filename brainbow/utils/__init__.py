"""
Cross-cutting utilities used throughout brainbow.

A small bag of helpers that don't fit cleanly into ``datasets`` /
``transforms`` / ``inference``:

I/O (:mod:`brainbow.utils.io`)
    * :func:`find_folder` -- locate a volume file in a directory by
      base name (any of the supported extensions).
    * :func:`load_volume` / :func:`save_volume` -- format-agnostic
      reader / writer dispatching to :mod:`brainbow.preprocessors`.

Clustering (:mod:`brainbow.utils.clustering`)
    * :func:`cluster_embeddings` -- generic dispatch over per-voxel
      embedding tensors (``soft_meanshift`` / ``hdbscan`` /
      ``spatial_cc``).  Used both by inference-time clusterers and by
      diagnostic notebooks.
    * :func:`cluster_spatial_cc` -- connected-components clusterer on
      the spatial-neighbour embedding-affinity graph.

Manifold projection (:mod:`brainbow.utils.manifold`)
    * :func:`project_batch` -- PCA / UMAP / t-SNE projection of
      per-voxel embeddings for TensorBoard previews.

Extending this module: anything reused by **two or more** subpackages
and with no better home belongs here.  Anything used by exactly one
subpackage should live next to its consumer.
"""

from brainbow.utils.io import find_folder, load_volume, save_volume
from brainbow.utils.clustering import (
    cluster_embeddings,
    cluster_embeddings_hdbscan,
    cluster_embeddings_soft,
    cluster_spatial_cc,
)

__all__ = [
    "find_folder",
    "load_volume",
    "save_volume",
    "cluster_embeddings",
    "cluster_embeddings_hdbscan",
    "cluster_embeddings_soft",
    "cluster_spatial_cc",
]
