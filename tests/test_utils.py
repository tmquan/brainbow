"""
Tests for utility functions.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from brainbow.utils.io import find_folder, load_volume, save_volume, ensure_data, SUPPORTED_EXTENSIONS
from brainbow.transforms.label import Labeld
from brainbow.metrics.instance import (
    _prepare_flat_labels,
    compute_per_point_ari as compute_ari_point,
    compute_per_point_ami as compute_ami_point,
    compute_per_batch_ari as compute_ari_batch,
    compute_per_batch_ami as compute_ami_batch,
)


class TestFindPath:
    """Tests for find_folder utility."""

    def test_finds_h5_file(self, tmp_path: Path) -> None:
        """Test finding an HDF5 file."""
        import h5py

        h5_file = tmp_path / "volume.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("main", data=np.zeros((10, 10)))

        result = find_folder(tmp_path, "volume")
        assert result is not None
        assert result.suffix == ".h5"

    def test_finds_tiff_file(self, tmp_path: Path) -> None:
        """Test finding a TIFF file."""
        import tifffile

        tiff_file = tmp_path / "image.tiff"
        tifffile.imwrite(str(tiff_file), np.zeros((10, 10), dtype=np.uint8))

        result = find_folder(tmp_path, "image")
        assert result is not None
        assert result.suffix == ".tiff"

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        """Test returning None when file not found."""
        result = find_folder(tmp_path, "nonexistent")
        assert result is None

    def test_priority_order(self, tmp_path: Path) -> None:
        """Test that h5 is preferred over tiff."""
        import h5py
        import tifffile

        h5_file = tmp_path / "data.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("main", data=np.zeros((5,)))

        tiff_file = tmp_path / "data.tiff"
        tifffile.imwrite(str(tiff_file), np.zeros((5, 5), dtype=np.uint8))

        result = find_folder(tmp_path, "data")
        assert result is not None
        assert result.suffix == ".h5"


class TestLoadSaveVolume:
    """Tests for load_volume and save_volume."""

    def test_h5_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading HDF5 volume."""
        data = np.random.rand(10, 32, 32).astype(np.float32)
        path = tmp_path / "volume.h5"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_almost_equal(data, loaded)

    def test_tiff_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading TIFF volume."""
        data = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8)
        path = tmp_path / "volume.tiff"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_equal(data, loaded)

    def test_npy_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading NPY volume."""
        data = np.random.rand(5, 16, 16).astype(np.float32)
        path = tmp_path / "volume.npy"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_almost_equal(data, loaded)

    def test_tensor_input(self, tmp_path: Path) -> None:
        """Test saving from torch tensor."""
        data = torch.randn(5, 16, 16)
        path = tmp_path / "volume.h5"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_almost_equal(data.numpy(), loaded, decimal=5)

    def test_nonexistent_raises(self) -> None:
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_volume("/nonexistent/path.h5")


class TestLabeld:
    """Tests for the :class:`Labeld` MapTransform.

    ``Labeld`` is the post-crop / post-deformation cleanup that
    splits any label whose voxels are no longer connected.  These
    tests exercise the basic 2-D and 3-D shapes that the rest of the
    pipeline relies on.
    """

    def _apply(self, labels: torch.Tensor, spatial_dims: int, **kwargs) -> torch.Tensor:
        tx = Labeld(keys="lbl", spatial_dims=spatial_dims, **kwargs)
        return tx({"lbl": labels})["lbl"]

    def test_2d_disconnected_same_label_split(self) -> None:
        labels = torch.zeros(10, 10, dtype=torch.long)
        labels[0:3, 0:3] = 5
        labels[7:10, 7:10] = 5  # same id, two disjoint regions
        result = self._apply(labels, spatial_dims=2)
        # background + 2 components.
        assert torch.unique(result).numel() == 3

    def test_3d_two_instances_preserved(self) -> None:
        labels = torch.zeros(4, 10, 10, dtype=torch.long)
        labels[:, 0:3, 0:3] = 1
        labels[:, 7:10, 7:10] = 2
        result = self._apply(labels, spatial_dims=3)
        assert torch.unique(result).numel() == 3

    def test_min_voxels_drops_slivers(self) -> None:
        labels = torch.zeros(10, 10, dtype=torch.long)
        labels[0:3, 0:3] = 1     # 9 voxels
        labels[5, 5] = 2          # 1 voxel — should be dropped
        result = self._apply(labels, spatial_dims=2, min_voxels=5)
        # Only background + the larger blob survive.
        assert torch.unique(result).numel() == 2


class TestComputeMetricsPoint:
    """Tests for per-sample ARI / AMI computation."""

    def test_perfect_ari(self) -> None:
        labels = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]])
        assert compute_ari_point(labels, labels) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_ami(self) -> None:
        labels = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]])
        assert compute_ami_point(labels, labels) == pytest.approx(1.0, abs=1e-6)

    def test_all_background(self) -> None:
        pred = torch.zeros(4, 4, dtype=torch.long)
        true = torch.zeros(4, 4, dtype=torch.long)
        assert compute_ari_point(pred, true) == 0.0
        assert compute_ami_point(pred, true) == 0.0


class TestComputeMetricsBatch:
    """Tests for batch ARI / AMI computation."""

    def test_batch_ari(self) -> None:
        pred = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        true = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        assert compute_ari_batch(pred, true) > 0.0

    def test_batch_ami(self) -> None:
        pred = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        true = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        assert compute_ami_batch(pred, true) > 0.0


class TestEnsureData:
    """Tests for ensure_data utility."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_data(new_dir)
        assert result.exists()
        assert result == new_dir

    def test_existing_directory(self, tmp_path: Path) -> None:
        result = ensure_data(tmp_path)
        assert result.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        result = ensure_data(str(tmp_path / "new"))
        assert isinstance(result, Path)


class TestSupportedExtensions:
    """Tests for module-level constants."""

    def test_contains_common_formats(self) -> None:
        assert ".h5" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS
        assert ".nrrd" in SUPPORTED_EXTENSIONS
        assert ".npy" in SUPPORTED_EXTENSIONS


class TestFindFolderCustomExtensions:
    """Additional find_folder tests."""

    def test_custom_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "data.xyz").write_text("hello")
        result = find_folder(tmp_path, "data", extensions=[".xyz"])
        assert result is not None
        assert result.suffix == ".xyz"

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        result = find_folder(tmp_path, "data", extensions=[".abc"])
        assert result is None


class TestSaveVolumeFormats:
    """Additional save/load volume format tests."""

    def test_npz_roundtrip(self, tmp_path: Path) -> None:
        data = np.random.rand(5, 8, 8).astype(np.float32)
        path = tmp_path / "volume.npz"
        save_volume(data, path)
        loaded = load_volume(path)
        np.testing.assert_array_almost_equal(data, loaded)

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "volume.xyz"
        path.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_volume(path)

    def test_save_unsupported_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            save_volume(np.zeros(5), tmp_path / "volume.xyz")

    def test_explicit_format_overrides_extension(self, tmp_path: Path) -> None:
        data = np.random.rand(4, 4).astype(np.float32)
        path = tmp_path / "volume.npy"
        save_volume(data, path, format="npy")
        loaded = load_volume(path, format="npy")
        np.testing.assert_array_almost_equal(data, loaded)


class TestPrepareFlatLabels:
    """Tests for _prepare_flat_labels helper."""

    def test_basic_flattening(self) -> None:
        pred = torch.tensor([[1, 2], [0, 1]])
        true = torch.tensor([[1, 1], [0, 2]])
        pred_flat, true_flat = _prepare_flat_labels(pred, true)
        assert len(pred_flat) == 3  # 3 foreground pixels

    def test_all_background(self) -> None:
        pred = torch.zeros(4, 4, dtype=torch.long)
        true = torch.zeros(4, 4, dtype=torch.long)
        pred_flat, true_flat = _prepare_flat_labels(pred, true)
        assert len(pred_flat) == 0

    def test_include_background(self) -> None:
        pred = torch.tensor([[0, 1], [0, 0]])
        true = torch.tensor([[0, 1], [0, 0]])
        pred_flat, true_flat = _prepare_flat_labels(pred, true, ignore_background=False)
        assert len(pred_flat) == 4


class TestComputeMetricsExtended:
    """Extended metric tests."""

    def test_random_ari_between_0_and_1(self) -> None:
        pred = torch.randint(1, 5, (10, 10))
        true = torch.randint(1, 5, (10, 10))
        ari = compute_ari_point(pred, true)
        assert 0.0 <= ari <= 1.0

    def test_random_ami_between_0_and_1(self) -> None:
        pred = torch.randint(1, 5, (10, 10))
        true = torch.randint(1, 5, (10, 10))
        ami = compute_ami_point(pred, true)
        assert 0.0 <= ami <= 1.0

    def test_batch_metrics_with_single_sample(self) -> None:
        pred = torch.tensor([[[1, 1, 2, 2]]])
        true = torch.tensor([[[1, 1, 2, 2]]])
        assert compute_ari_batch(pred, true) > 0
        assert compute_ami_batch(pred, true) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
