"""Tests for the ``Skeletond`` transform.

The tests exercise both the kimimaro and the skimage paths.  When
kimimaro isn't installed we monkey-patch :func:`_kimimaro_available` to
keep the assertions deterministic on either install state.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from brainbow.transforms.skeleton import (
    Skeletond,
    SkeletonGeometryd,
    compute_radius_field,
    compute_skeleton_field,
    compute_skeleton_geometry,
)
from brainbow.transforms import skeleton as skeleton_module


def _cylinder_label(
    shape: tuple[int, int, int] = (16, 32, 32),
    radius: int = 3,
    instance_id: int = 1,
) -> np.ndarray:
    """A solid axis-aligned cylinder centered in the volume.

    Returns a 3-D integer label array with a single foreground
    instance.  Cylinder axis runs along Z; cross-section is a disk in
    YX of radius ``radius``.  Used to anchor the skeleton golden tests
    -- the centerline must lie on the Z column at the YX center.
    """
    D, H, W = shape
    yy, xx = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij",
    )
    cy, cx = H // 2, W // 2
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    label = np.zeros(shape, dtype=np.int64)
    label[:, disk] = instance_id
    return label


def _force_skimage_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the transform to its skimage fallback for tests that need it.

    Resets the cached availability probe so the monkey-patched value
    actually wins on the first call from this test.
    """
    monkeypatch.setattr(skeleton_module, "_KIMIMARO_AVAILABLE", False)
    monkeypatch.setattr(
        skeleton_module, "_kimimaro_available", lambda: False,
    )


class TestComputeSkeletonField:
    def test_cylinder_skimage_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A solid cylinder skeletonizes to a 1-voxel-wide Z-axis line.

        Skimage's 3-D skeletonize sometimes trims a voxel off each Z
        end of a finite cylinder (where the disk is thinner than the
        process); the test allows for that without losing the core
        invariant.
        """
        _force_skimage_backend(monkeypatch)

        D, H, W = 16, 32, 32
        label = _cylinder_label((D, H, W), radius=3)

        skl = compute_skeleton_field(label)
        assert skl.shape == (1, D, H, W)
        assert skl.dtype == np.float32

        # Skeleton voxel count is bounded by the cylinder length plus a
        # small slack for skimage edge effects.
        n_skl = int(skl.sum())
        assert 1 <= n_skl <= D

        # Every skeleton voxel sits on the disk's center column (the
        # axis of the cylinder).  Skimage may pick the literal center
        # or a near-neighbour for the unique medial axis under
        # 6-connectivity; we allow up to a 1-voxel offset.
        ones = np.argwhere(skl[0] > 0)
        cy, cx = H // 2, W // 2
        for z, y, x in ones:
            assert abs(int(y) - cy) <= 1
            assert abs(int(x) - cx) <= 1

    def test_skeleton_confined_to_foreground(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``skl == 1`` voxels must always be foreground (label > 0)."""
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((16, 32, 32), radius=3)
        skl = compute_skeleton_field(label)
        assert np.all((skl[0] == 0) | (label > 0))

    def test_empty_label_returns_zero_mask(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _force_skimage_backend(monkeypatch)
        label = np.zeros((8, 16, 16), dtype=np.int64)
        skl = compute_skeleton_field(label)
        assert skl.shape == (1, 8, 16, 16)
        assert int(skl.sum()) == 0

    def test_multi_instance_union(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two non-overlapping instances -> union of their skeletons."""
        _force_skimage_backend(monkeypatch)

        D, H, W = 12, 32, 32
        label = np.zeros((D, H, W), dtype=np.int64)
        # Instance 1: cylinder in the left half.
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        disk1 = (yy - 16) ** 2 + (xx - 8) ** 2 <= 9
        label[:, disk1] = 1
        # Instance 2: cylinder in the right half.
        disk2 = (yy - 16) ** 2 + (xx - 24) ** 2 <= 9
        label[:, disk2] = 2

        skl = compute_skeleton_field(label)
        assert skl.shape == (1, D, H, W)
        # Each instance contributes its own skeleton; the union is
        # disjoint by construction.
        n_left = int(skl[0, :, :, :W // 2].sum())
        n_right = int(skl[0, :, :, W // 2:].sum())
        assert n_left > 0 and n_right > 0

    def test_single_voxel_instance_kept(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Degenerate 1-voxel instances stay in the skeleton mask."""
        _force_skimage_backend(monkeypatch)

        label = np.zeros((4, 4, 4), dtype=np.int64)
        label[1, 1, 1] = 7
        skl = compute_skeleton_field(label)
        assert skl[0, 1, 1, 1] == 1.0
        # No other voxel should be flipped on.
        assert int(skl.sum()) == 1

    def test_falls_back_to_skimage_when_kimimaro_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If kimimaro is reported available but raises at call time,
        the transform silently falls back to skimage and still emits a
        non-empty skeleton."""
        # Pretend kimimaro is installed but make the multi-label call
        # raise -- the production code should catch and degrade.
        monkeypatch.setattr(skeleton_module, "_KIMIMARO_AVAILABLE", True)
        monkeypatch.setattr(
            skeleton_module, "_kimimaro_available", lambda: True,
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated kimimaro failure")

        monkeypatch.setattr(
            skeleton_module,
            "_skeletonize_all_kimimaro",
            _boom,
        )

        label = _cylinder_label((12, 24, 24), radius=2)
        skl = compute_skeleton_field(label)
        assert int(skl.sum()) > 0


class TestSkeletondTransform:
    def test_emits_label_skl_key_with_correct_shape(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _force_skimage_backend(monkeypatch)

        D, H, W = 8, 24, 24
        label = _cylinder_label((D, H, W), radius=2)
        # Mirror the post-EnsureChannelFirstd convention: [C, *spatial].
        data = {"label": torch.from_numpy(label).unsqueeze(0)}
        out = Skeletond(keys=["label"], spatial_dims=3)(data)
        assert "label_skl" in out
        assert out["label_skl"].shape == (1, D, H, W)
        assert out["label_skl"].dtype == torch.float32

    def test_preserves_tensor_device(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CPU-tensor input -> CPU-tensor output (numpy compute under
        the hood, device round-tripped at the boundary)."""
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((8, 16, 16), radius=2)
        tensor_label = torch.from_numpy(label).unsqueeze(0)
        out = Skeletond(keys=["label"], spatial_dims=3)({"label": tensor_label})
        assert isinstance(out["label_skl"], torch.Tensor)
        assert out["label_skl"].device == tensor_label.device

    def test_accepts_numpy_input(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((8, 16, 16), radius=2)
        out = Skeletond(keys=["label"], spatial_dims=3)(
            {"label": np.expand_dims(label, axis=0)},
        )
        assert isinstance(out["label_skl"], np.ndarray)


class TestComputeRadiusField:
    def test_cylinder_radius_voxel_units(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Un-normalised radius equals the cylinder's actual radius at
        the surface and is 0 at the centerline."""
        _force_skimage_backend(monkeypatch)

        D, H, W = 16, 32, 32
        radius = 4
        label = _cylinder_label((D, H, W), radius=radius)
        skl = compute_skeleton_field(label)
        rad = compute_radius_field(label, skl, normalize=False)

        assert rad.shape == (1, D, H, W)
        assert rad.dtype == np.float32

        # On the central column (where the skeleton lives), rad ~= 0.
        cy, cx = H // 2, W // 2
        # Take a Z-slice far from the volume edges so skimage's possible
        # end-cap trimming doesn't bite us.
        z = D // 2
        # Some skeleton voxel must sit on the (z, cy, cx) column or its
        # immediate xy-neighbour; rad there is 0 (skeleton voxel itself).
        center_block = rad[0, z, cy - 1:cy + 2, cx - 1:cx + 2]
        assert float(center_block.min()) == pytest.approx(0.0, abs=1e-5)

        # Far from the axis (at the disk edge), rad should approach
        # the cylinder radius.  Sample a surface voxel.
        max_in_slice = rad[0, z].max()
        assert max_in_slice == pytest.approx(radius, abs=1.0)

    def test_cylinder_radius_normalized(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default-normalised + inverted radius lives in [0, 1].

        Under the inverted ridge convention (the default) the
        skeleton is the peak (``rad = 1``) and the instance boundary
        decays to ``rad ≈ 0`` -- matching the background's ``0`` so
        the field is continuous across the fg/bg interface.
        """
        _force_skimage_backend(monkeypatch)

        D, H, W = 16, 32, 32
        label = _cylinder_label((D, H, W), radius=4)
        skl = compute_skeleton_field(label)
        rad = compute_radius_field(label, skl, normalize=True)

        fg = label > 0
        assert rad.shape == (1, D, H, W)
        assert float(rad.min()) >= 0.0
        # Per-instance normalisation + inversion forces the max inside
        # the instance to be exactly 1 (the skeleton itself).
        assert float(rad[0][fg].max()) == pytest.approx(1.0, abs=1e-5)
        assert float(rad.max()) <= 1.0 + 1e-5

    def test_inverted_radius_peaks_at_skeleton(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skeleton voxels carry rad == 1; boundary voxels ≈ 0."""
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((16, 32, 32), radius=4)
        skl = compute_skeleton_field(label)
        rad = compute_radius_field(label, skl, normalize=True, invert=True)

        skel_mask = skl[0] > 0
        # The skeleton ridge is at the peak of the field.
        np.testing.assert_allclose(
            rad[0][skel_mask], 1.0, atol=1e-5,
        )
        # Background carries 0, just like the un-inverted version --
        # making the fg/bg interface continuous.
        bg = label == 0
        assert float(rad[0][bg].max()) == 0.0

    def test_invert_false_returns_legacy_distance(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``invert=False`` opts back into the legacy distance field
        (skeleton == 0, boundary == 1)."""
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((16, 32, 32), radius=4)
        skl = compute_skeleton_field(label)
        rad = compute_radius_field(label, skl, normalize=True, invert=False)

        skel_mask = skl[0] > 0
        np.testing.assert_allclose(
            rad[0][skel_mask], 0.0, atol=1e-5,
        )
        fg = label > 0
        assert float(rad[0][fg].max()) == pytest.approx(1.0, abs=1e-5)

    def test_radius_zero_outside_instance(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Background voxels never receive a non-zero radius."""
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((8, 24, 24), radius=2)
        skl = compute_skeleton_field(label)
        rad = compute_radius_field(label, skl, normalize=True)

        bg = label == 0
        assert float(rad[0][bg].max()) == 0.0

    def test_two_instances_dont_leak(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A voxel in instance A measures distance to A's skeleton, not
        to a (possibly closer) skeleton voxel from instance B."""
        _force_skimage_backend(monkeypatch)

        D, H, W = 8, 32, 32
        label = np.zeros((D, H, W), dtype=np.int64)
        # Two parallel cylinders, instance 1 on the left, 2 on the right.
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        disk1 = (yy - 16) ** 2 + (xx - 8) ** 2 <= 9
        disk2 = (yy - 16) ** 2 + (xx - 24) ** 2 <= 9
        label[:, disk1] = 1
        label[:, disk2] = 2

        skl = compute_skeleton_field(label)
        # Voxel units to make the leak (if any) obvious as a smaller
        # distance.
        rad = compute_radius_field(label, skl, normalize=False)

        # Per-instance max radius matches each cylinder's own radius
        # (~3 from the disk equation).  If radii leaked across, the
        # rim voxels of instance 1 would point to instance 2's
        # skeleton instead and produce much larger distances.
        for uid in (1, 2):
            m_i = label == uid
            assert float(rad[0][m_i].max()) <= 4.0


class TestSkeletonGeometryd:
    def test_emits_skl_and_radius_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _force_skimage_backend(monkeypatch)

        D, H, W = 8, 24, 24
        label = _cylinder_label((D, H, W), radius=2)
        data = {"label": torch.from_numpy(label).unsqueeze(0)}
        out = SkeletonGeometryd(
            keys=["label"], spatial_dims=3, radius_normalize=True,
        )(data)
        assert "label_skl" in out
        assert "label_radius" in out
        assert out["label_skl"].shape == (1, D, H, W)
        assert out["label_radius"].shape == (1, D, H, W)
        # Normalised radius is bounded in [0, 1].
        assert out["label_radius"].max().item() <= 1.0 + 1e-5

    def test_radius_normalize_false_returns_voxel_units(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _force_skimage_backend(monkeypatch)

        D, H, W = 8, 24, 24
        radius = 3
        label = _cylinder_label((D, H, W), radius=radius)
        out = SkeletonGeometryd(
            keys=["label"], spatial_dims=3, radius_normalize=False,
        )({"label": torch.from_numpy(label).unsqueeze(0)})
        # In voxel units, the max should be roughly the cylinder radius.
        assert out["label_radius"].max().item() == pytest.approx(
            radius, abs=1.0,
        )

    def test_skeletond_alias_still_works(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The PR-2 ``Skeletond`` name still resolves to the same class
        so older call sites keep working."""
        _force_skimage_backend(monkeypatch)

        assert Skeletond is SkeletonGeometryd

        D, H, W = 8, 16, 16
        label = _cylinder_label((D, H, W), radius=2)
        out = Skeletond(keys=["label"], spatial_dims=3)(
            {"label": torch.from_numpy(label).unsqueeze(0)},
        )
        # PR 4 upgrades the transform to emit the full geometry quartet.
        assert "label_skl" in out
        assert "label_radius" in out
        assert "label_direction" in out
        assert "label_covariance" in out

    def test_emits_full_geometry_quartet(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The combined transform writes skl + dir + cov + rad in one
        pass with matching spatial shapes and the expected channel
        widths."""
        _force_skimage_backend(monkeypatch)

        D, H, W = 8, 24, 24
        label = _cylinder_label((D, H, W), radius=3)
        out = SkeletonGeometryd(keys=["label"], spatial_dims=3)(
            {"label": torch.from_numpy(label).unsqueeze(0)},
        )
        assert out["label_skl"].shape == (1, D, H, W)
        assert out["label_direction"].shape == (3, D, H, W)
        assert out["label_covariance"].shape == (6, D, H, W)
        assert out["label_radius"].shape == (1, D, H, W)


class TestComputeSkeletonGeometry:
    def test_dir_is_unit_norm_inside_foreground(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``dir`` is a unit vector at every fg voxel except skeleton
        voxels themselves (where it is zero by construction)."""
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((12, 24, 24), radius=3)
        fields = compute_skeleton_geometry(label)

        dir_field = fields["dir"]
        skl_field = fields["skl"]
        fg = label > 0
        skl_bool = skl_field[0] > 0

        non_skl_fg = fg & ~skl_bool
        norms = np.sqrt(
            dir_field[0] ** 2 + dir_field[1] ** 2 + dir_field[2] ** 2
        )
        # Every non-skeleton foreground voxel carries a unit vector.
        np.testing.assert_allclose(
            norms[non_skl_fg],
            np.ones(int(non_skl_fg.sum()), dtype=np.float32),
            atol=1e-5,
        )
        # And skeleton voxels carry the zero vector.
        assert float(norms[skl_bool].max()) == pytest.approx(0.0, abs=1e-6)

    def test_dir_is_centripetal_for_axial_cylinder(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For an axial cylinder, the centerline is the Z column at
        (cy, cx); ``dir`` at a rim voxel should point inward in the
        YX plane (the Z component is ~0)."""
        _force_skimage_backend(monkeypatch)

        D, H, W = 12, 32, 32
        radius = 4
        label = _cylinder_label((D, H, W), radius=radius)
        fields = compute_skeleton_geometry(label)
        dir_field = fields["dir"]

        # Rim voxel: take the right-most fg voxel on the central Z
        # slice's horizontal axis.
        z = D // 2
        cy, cx = H // 2, W // 2
        rim_x = cx + radius
        v = dir_field[:, z, cy, rim_x]
        # Should point in -X (toward the axis); Z component near 0.
        assert v[2] < -0.5, f"dir at rim should point inward, got {v}"
        assert abs(float(v[0])) < 0.2

    def test_rad_times_dir_recovers_offset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The plan's per-voxel sanity identity::

            rad(v) * R_i * dir(v) == s*(v) - v

        For the un-normalised radius (``radius_normalize=False``), the
        identity simplifies to ``rad(v) * dir(v) == s*(v) - v`` so we
        can probe it directly.
        """
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((10, 24, 24), radius=3)
        fields = compute_skeleton_geometry(label, radius_normalize=False)

        dir_field = fields["dir"]
        rad_field = fields["rad"]
        recon = dir_field * rad_field[0]                       # [3, D, H, W]
        recon_mag = np.sqrt(np.sum(recon ** 2, axis=0))

        # Magnitude of the reconstructed offset equals the radius
        # field (by construction).
        fg = label > 0
        np.testing.assert_allclose(
            recon_mag[fg], rad_field[0][fg], atol=1e-5,
        )

    def test_cov_voronoi_cell_is_trace_normalized(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With ``cov_normalized=True`` (default), the trace of the
        Voronoi-cell covariance matrix at every fg voxel is 1.0
        (modulo cells with degenerate populations whose trace is 0)."""
        _force_skimage_backend(monkeypatch)

        label = _cylinder_label((12, 24, 24), radius=3)
        fields = compute_skeleton_geometry(label, cov_normalized=True)
        cov = fields["cov"]
        # Channel order is (Czz, Czy, Czx, Cyy, Cyx, Cxx); diagonals
        # are channels 0, 3, 5.
        trace = cov[0] + cov[3] + cov[5]
        fg = label > 0
        # Either trace == 1 (non-degenerate cell) or trace == 0 (single
        # skeleton-only cell).  No other values may appear.
        nonzero = trace[fg][trace[fg] > 1e-6]
        np.testing.assert_allclose(
            nonzero, np.ones_like(nonzero), atol=1e-5,
        )

    def test_cov_smallest_eigenvector_aligns_with_tangent(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For a cylinder with Z-aligned centerline, the Voronoi-cell
        moments at every fg voxel should have their **smallest**
        eigenvalue along Z (the tangent direction) and the two larger
        eigenvalues in the YX plane (the cross-section)."""
        _force_skimage_backend(monkeypatch)

        D, H, W = 16, 24, 24
        label = _cylinder_label((D, H, W), radius=3)
        fields = compute_skeleton_geometry(label, cov_normalized=False)
        cov = fields["cov"]

        # Sample one fg voxel safely away from the volume edge so the
        # Voronoi cell is fully internal.
        z, y, x = D // 2, H // 2 + 2, W // 2
        # Reassemble the symmetric 3x3 matrix.
        m = np.array([
            [cov[0, z, y, x], cov[1, z, y, x], cov[2, z, y, x]],
            [cov[1, z, y, x], cov[3, z, y, x], cov[4, z, y, x]],
            [cov[2, z, y, x], cov[5, z, y, x], cov[5, z, y, x]],
        ], dtype=np.float64)
        # Fix the (2, 2) entry from the upper-tri stack (the line
        # above accidentally repeats Cxx); just rebuild it cleanly.
        m = np.array([
            [cov[0, z, y, x], cov[1, z, y, x], cov[2, z, y, x]],
            [cov[1, z, y, x], cov[3, z, y, x], cov[4, z, y, x]],
            [cov[2, z, y, x], cov[4, z, y, x], cov[5, z, y, x]],
        ], dtype=np.float64)

        eigvals, eigvecs = np.linalg.eigh(m)
        smallest_eigvec = eigvecs[:, 0]
        # Tangent of an axial cylinder is the Z axis; the eigenvector
        # may point in either direction (+Z or -Z), so test |z|.
        assert abs(float(smallest_eigvec[0])) > 0.9

    def test_empty_label_returns_zero_fields(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _force_skimage_backend(monkeypatch)
        label = np.zeros((6, 12, 12), dtype=np.int64)
        fields = compute_skeleton_geometry(label)
        assert fields["skl"].sum() == 0
        assert fields["dir"].sum() == 0
        assert fields["cov"].sum() == 0
        assert fields["rad"].sum() == 0
