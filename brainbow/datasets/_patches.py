"""
Shared 3-D patch-index generator.

Why this file exists
--------------------
``MICRONSDataset`` and ``NeuronsDataset`` both need to enumerate
overlapping 3-D patch indices over a full volume (used at construction
time when neither slice mode nor random-sampling lazy mode applies).
Both files used to carry an identical, byte-for-byte copy of the
generator; this module is the single shared implementation.

Public surface
--------------
* :func:`generate_patch_indices`

Extending this module
---------------------
The generator is dataset-agnostic and pure: it takes a volume shape, a
patch size, and an overlap fraction, and returns a list of slice
triples in row-major (z, y, x) order.  If you need a different tiling
strategy (e.g. random offsets, non-cuboidal volumes), add a new helper
alongside this one rather than overloading :func:`generate_patch_indices`.
"""

from typing import List, Sequence, Tuple

__all__ = ["generate_patch_indices"]


def generate_patch_indices(
    volume_shape: Sequence[int],
    patch_size: Sequence[int],
    overlap: float,
) -> List[Tuple[slice, slice, slice]]:
    """Tile a 3-D volume into overlapping patches.

    The grid covers the whole volume.  When the last patch would extend
    past the volume edge, it is shifted backwards so its trailing face
    lands on the volume boundary; this guarantees every voxel is
    covered without any zero-padding inside the returned slices.

    Args:
        volume_shape: ``(Z, Y, X)`` shape of the source volume.
        patch_size:   ``(pz, py, px)`` patch size (per axis).
        overlap:      Fractional overlap between adjacent patches in
            ``[0, 1)``.  ``0`` produces a non-overlapping tiling;
            ``0.5`` produces patches that share half their extent with
            their neighbours.

    Returns:
        List of ``(z_slice, y_slice, x_slice)`` triples in row-major
        order (Z varies slowest, X fastest).
    """
    all_dim_indices: List[List[Tuple[int, int]]] = []

    for dim in range(3):
        vol_size = volume_shape[dim]
        patch_dim = patch_size[dim]
        stride = max(1, int(patch_dim * (1 - overlap)))

        dim_indices: List[Tuple[int, int]] = []
        start = 0
        while start < vol_size:
            end = min(start + patch_dim, vol_size)
            if end - start < patch_dim and start > 0:
                start = max(0, end - patch_dim)
            dim_indices.append((start, end))
            if end >= vol_size:
                break
            start += stride

        all_dim_indices.append(dim_indices)

    patch_indices: List[Tuple[slice, slice, slice]] = []
    for z_start, z_end in all_dim_indices[0]:
        for y_start, y_end in all_dim_indices[1]:
            for x_start, x_end in all_dim_indices[2]:
                patch_indices.append(
                    (
                        slice(z_start, z_end),
                        slice(y_start, y_end),
                        slice(x_start, x_end),
                    )
                )

    return patch_indices
