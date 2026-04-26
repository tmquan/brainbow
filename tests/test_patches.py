"""
Tests for :func:`brainbow.datasets._patches.generate_patch_indices`.

The generator is dataset-agnostic and pure -- it just turns a volume
shape + patch size + overlap into a list of slice triples.  Tests below
verify the row-major ordering, full coverage, and the back-shift rule
that prevents zero-padded edge patches.
"""

import pytest

from brainbow.datasets._patches import generate_patch_indices


class TestGeneratePatchIndices:

    def test_exact_fit_no_overlap(self) -> None:
        slices = generate_patch_indices(
            volume_shape=(8, 8, 8),
            patch_size=(4, 4, 4),
            overlap=0.0,
        )
        # 2 patches per axis => 8 total.
        assert len(slices) == 8

    def test_overlap_increases_count(self) -> None:
        no_overlap = generate_patch_indices((16, 16, 16), (8, 8, 8), 0.0)
        half_overlap = generate_patch_indices((16, 16, 16), (8, 8, 8), 0.5)
        assert len(half_overlap) > len(no_overlap)

    def test_back_shift_keeps_patches_in_bounds(self) -> None:
        # Volume is not an exact multiple of patch size: the last patch
        # along Z must shift back so it does not run past the edge.
        slices = generate_patch_indices(
            volume_shape=(10, 8, 8),
            patch_size=(4, 4, 4),
            overlap=0.0,
        )
        for z_sl, y_sl, x_sl in slices:
            assert 0 <= z_sl.start and z_sl.stop <= 10
            assert 0 <= y_sl.start and y_sl.stop <= 8
            assert 0 <= x_sl.start and x_sl.stop <= 8
            # After the back-shift, every patch should be exactly the
            # requested size (no truncated trailing patch).
            assert (z_sl.stop - z_sl.start) == 4
            assert (y_sl.stop - y_sl.start) == 4
            assert (x_sl.stop - x_sl.start) == 4

    def test_full_voxel_coverage(self) -> None:
        # Every voxel should be inside at least one patch.
        shape = (10, 12, 14)
        slices = generate_patch_indices(shape, (4, 4, 4), overlap=0.0)
        covered = [[[False] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
        for z, y, x in slices:
            for zi in range(z.start, z.stop):
                for yi in range(y.start, y.stop):
                    for xi in range(x.start, x.stop):
                        covered[zi][yi][xi] = True
        assert all(all(all(row) for row in plane) for plane in covered)

    def test_row_major_order(self) -> None:
        # Z varies slowest, X fastest.
        slices = generate_patch_indices((8, 8, 8), (4, 4, 4), overlap=0.0)
        # First two slices share the same Z start.
        assert slices[0][0].start == slices[1][0].start
        # The first slice that changes Z should appear after exhausting Y/X.
        # With 2 patches per axis, that's index 4.
        assert slices[4][0].start == 4

    def test_volume_smaller_than_patch(self) -> None:
        # When the volume is smaller than the patch, expect a single
        # truncated slice anchored at 0.
        slices = generate_patch_indices((3, 3, 3), (4, 4, 4), overlap=0.0)
        assert len(slices) == 1
        z, y, x = slices[0]
        assert (z.start, z.stop) == (0, 3)
        assert (y.start, y.stop) == (0, 3)
        assert (x.start, x.stop) == (0, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
