"""
Tests for data format preprocessors.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from brainbow.preprocessors.base import BasePreprocessor
from brainbow.preprocessors.hdf5 import HDF5Preprocessor
from brainbow.preprocessors.tiff import TIFFPreprocessor


# ---------------------------------------------------------------------------
# BasePreprocessor (abstract)
# ---------------------------------------------------------------------------

class TestBasePreprocessor:
    """Tests for the abstract BasePreprocessor."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BasePreprocessor()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self) -> List[str]:
                return [".dummy"]

            def load(self, path, **kwargs):
                return np.zeros(5)

            def validate(self, path) -> bool:
                return True

        pp = DummyPreprocessor()
        assert pp.supported_extensions == [".dummy"]
        assert pp.validate("any")
        np.testing.assert_array_equal(pp.load("any"), np.zeros(5))

    def test_to_tensor(self) -> None:
        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self) -> List[str]:
                return [".x"]

            def load(self, path, **kwargs):
                return np.zeros(3)

            def validate(self, path) -> bool:
                return True

        pp = DummyPreprocessor()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = pp.to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        np.testing.assert_array_almost_equal(t.numpy(), arr)

    def test_to_tensor_with_dtype(self) -> None:
        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self):
                return [".x"]

            def load(self, path, **kwargs):
                return np.zeros(1)

            def validate(self, path):
                return True

        pp = DummyPreprocessor()
        t = pp.to_tensor(np.array([1, 2, 3]), dtype=torch.float32)
        assert t.dtype == torch.float32

    def test_save_raises_by_default(self) -> None:
        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self):
                return [".x"]

            def load(self, path, **kwargs):
                return np.zeros(1)

            def validate(self, path):
                return True

        pp = DummyPreprocessor()
        with pytest.raises(NotImplementedError):
            pp.save(np.zeros(1), "out.x")

    def test_check_file_exists_raises(self) -> None:
        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self):
                return [".x"]

            def load(self, path, **kwargs):
                return np.zeros(1)

            def validate(self, path):
                return True

        pp = DummyPreprocessor()
        with pytest.raises(FileNotFoundError):
            pp._check_file_exists("/nonexistent/file.x")

    def test_check_extension_raises(self) -> None:
        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self):
                return [".x"]

            def load(self, path, **kwargs):
                return np.zeros(1)

            def validate(self, path):
                return True

        pp = DummyPreprocessor()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            pp._check_extension("file.wrong")

    def test_repr(self) -> None:
        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self):
                return [".x", ".y"]

            def load(self, path, **kwargs):
                return np.zeros(1)

            def validate(self, path):
                return True

        pp = DummyPreprocessor()
        r = repr(pp)
        assert "DummyPreprocessor" in r
        assert ".x" in r

    def test_get_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "dummy.x"
        f.write_bytes(b"hello")

        class DummyPreprocessor(BasePreprocessor):
            @property
            def supported_extensions(self):
                return [".x"]

            def load(self, path, **kwargs):
                return np.zeros(1)

            def validate(self, path):
                return True

        pp = DummyPreprocessor()
        meta = pp.get_metadata(str(f))
        assert meta["filename"] == "dummy.x"
        assert meta["extension"] == ".x"
        assert meta["size_bytes"] == 5


# ---------------------------------------------------------------------------
# HDF5Preprocessor
# ---------------------------------------------------------------------------

class TestHDF5Preprocessor:
    """Tests for HDF5Preprocessor."""

    def test_supported_extensions(self) -> None:
        pp = HDF5Preprocessor()
        assert ".h5" in pp.supported_extensions
        assert ".hdf5" in pp.supported_extensions

    def test_save_and_load(self, tmp_path: Path) -> None:
        pp = HDF5Preprocessor(default_key="main")
        data = np.random.rand(10, 20).astype(np.float32)
        path = str(tmp_path / "test.h5")

        pp.save(data, path)
        loaded = pp.load(path)
        np.testing.assert_array_almost_equal(data, loaded)

    def test_load_with_key(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "multi.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("alpha", data=np.ones(5))
            f.create_dataset("beta", data=np.zeros(5))

        pp = HDF5Preprocessor()
        loaded = pp.load(path, key="beta")
        np.testing.assert_array_equal(loaded, np.zeros(5))

    def test_load_auto_discovers_key(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "auto.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.arange(10))

        pp = HDF5Preprocessor(default_key="nonexistent")
        loaded = pp.load(path)
        np.testing.assert_array_equal(loaded, np.arange(10))

    def test_load_missing_key_raises(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "empty_keys.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("special", data=np.zeros(3))

        pp = HDF5Preprocessor()
        with pytest.raises(KeyError):
            pp.load(path, key="missing")

    def test_load_file_not_found(self) -> None:
        pp = HDF5Preprocessor()
        with pytest.raises(FileNotFoundError):
            pp.load("/nonexistent/file.h5")

    def test_validate_valid(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "valid.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("main", data=np.zeros(3))

        pp = HDF5Preprocessor()
        assert pp.validate(path) is True

    def test_validate_nonexistent(self) -> None:
        pp = HDF5Preprocessor()
        assert pp.validate("/nonexistent.h5") is False

    def test_validate_wrong_extension(self, tmp_path: Path) -> None:
        path = str(tmp_path / "file.txt")
        (tmp_path / "file.txt").write_text("not hdf5")
        pp = HDF5Preprocessor()
        assert pp.validate(path) is False

    def test_list_datasets(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "nested.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("a", data=np.zeros(3))
            g = f.create_group("group")
            g.create_dataset("b", data=np.ones(5))

        pp = HDF5Preprocessor()
        datasets = pp.list_datasets(path)
        assert "a" in datasets
        assert "group/b" in datasets

    def test_get_shape(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "shape.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("main", data=np.zeros((10, 20, 30)))

        pp = HDF5Preprocessor()
        shape = pp.get_shape(path)
        assert shape == (10, 20, 30)

    def test_get_metadata(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "meta.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("main", data=np.zeros((5, 10)))

        pp = HDF5Preprocessor()
        meta = pp.get_metadata(path)
        assert meta["primary_dataset"] == "main"
        assert meta["shape"] == (5, 10)

    def test_save_with_compression(self, tmp_path: Path) -> None:
        pp = HDF5Preprocessor()
        data = np.random.rand(100, 100).astype(np.float32)
        path = str(tmp_path / "compressed.h5")
        pp.save(data, path, compression="gzip", compression_opts=1)
        loaded = pp.load(path)
        np.testing.assert_array_almost_equal(data, loaded)

    def test_load_with_slices(self, tmp_path: Path) -> None:
        import h5py

        path = str(tmp_path / "sliced.h5")
        data = np.arange(100).reshape(10, 10)
        with h5py.File(path, "w") as f:
            f.create_dataset("main", data=data)

        pp = HDF5Preprocessor()
        loaded = pp.load(path, slices=(slice(0, 3), slice(0, 5)))
        assert loaded.shape == (3, 5)
        np.testing.assert_array_equal(loaded, data[:3, :5])


# ---------------------------------------------------------------------------
# TIFFPreprocessor
# ---------------------------------------------------------------------------

class TestTIFFPreprocessor:
    """Tests for TIFFPreprocessor."""

    def test_supported_extensions(self) -> None:
        pp = TIFFPreprocessor()
        assert ".tiff" in pp.supported_extensions
        assert ".tif" in pp.supported_extensions

    def test_save_and_load(self, tmp_path: Path) -> None:
        pp = TIFFPreprocessor()
        data = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
        path = str(tmp_path / "test.tiff")

        pp.save(data, path)
        loaded = pp.load(path)
        np.testing.assert_array_equal(data, loaded)

    def test_validate_valid(self, tmp_path: Path) -> None:
        import tifffile

        path = str(tmp_path / "valid.tiff")
        tifffile.imwrite(path, np.zeros((10, 10), dtype=np.uint8))

        pp = TIFFPreprocessor()
        assert pp.validate(path) is True

    def test_validate_nonexistent(self) -> None:
        pp = TIFFPreprocessor()
        assert pp.validate("/nonexistent.tiff") is False

    def test_validate_wrong_extension(self, tmp_path: Path) -> None:
        path = str(tmp_path / "file.png")
        (tmp_path / "file.png").write_bytes(b"\x89PNG")
        pp = TIFFPreprocessor()
        assert pp.validate(path) is False

    def test_get_shape(self, tmp_path: Path) -> None:
        import tifffile

        data = np.zeros((5, 32, 32), dtype=np.uint8)
        path = str(tmp_path / "shape.tiff")
        tifffile.imwrite(path, data)

        pp = TIFFPreprocessor()
        shape = pp.get_shape(path)
        assert shape == (5, 32, 32)

    def test_get_metadata(self, tmp_path: Path) -> None:
        import tifffile

        path = str(tmp_path / "meta.tiff")
        tifffile.imwrite(path, np.zeros((3, 16, 16), dtype=np.uint8))

        pp = TIFFPreprocessor()
        meta = pp.get_metadata(path)
        assert "num_pages" in meta
        assert "num_series" in meta

    def test_load_file_not_found(self) -> None:
        pp = TIFFPreprocessor()
        with pytest.raises(FileNotFoundError):
            pp.load("/nonexistent.tiff")

    def test_memmap_mode(self, tmp_path: Path) -> None:
        import tifffile

        data = np.random.randint(0, 255, (5, 16, 16), dtype=np.uint8)
        path = str(tmp_path / "memmap.tiff")
        tifffile.imwrite(path, data)

        pp = TIFFPreprocessor(memmap=True)
        loaded = pp.load(path)
        np.testing.assert_array_equal(data, loaded)

    def test_load_single_page(self, tmp_path: Path) -> None:
        import tifffile

        data = np.random.randint(0, 255, (5, 16, 16), dtype=np.uint8)
        path = str(tmp_path / "pages.tiff")
        tifffile.imwrite(path, data)

        pp = TIFFPreprocessor()
        single = pp.load(path, key=0)
        np.testing.assert_array_equal(single, data[0])


# ---------------------------------------------------------------------------
# NFTYPreprocessor
# ---------------------------------------------------------------------------

class TestNFTYPreprocessor:
    """Tests for NFTYPreprocessor."""

    def test_supported_extensions(self) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor

        pp = NFTYPreprocessor()
        assert ".nii" in pp.supported_extensions
        assert ".nii.gz" in pp.supported_extensions

    def test_save_and_load(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor

        pp = NFTYPreprocessor()
        data = np.random.rand(10, 20, 30).astype(np.float32)
        path = str(tmp_path / "test.nii.gz")

        pp.save(data, path)
        loaded = pp.load(path)
        np.testing.assert_array_almost_equal(data, loaded, decimal=5)

    def test_save_and_load_with_affine(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor

        pp = NFTYPreprocessor()
        data = np.random.rand(5, 10, 10).astype(np.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        path = str(tmp_path / "affine.nii.gz")

        pp.save(data, path, affine=affine)
        loaded, loaded_affine = pp.load_with_affine(path)
        np.testing.assert_array_almost_equal(data, loaded, decimal=5)
        np.testing.assert_array_almost_equal(affine, loaded_affine)

    def test_validate_valid(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor
        import nibabel as nib

        data = np.zeros((5, 5, 5), dtype=np.float32)
        path = str(tmp_path / "valid.nii.gz")
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)

        pp = NFTYPreprocessor()
        assert pp.validate(path) is True

    def test_validate_nonexistent(self) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor

        pp = NFTYPreprocessor()
        assert pp.validate("/nonexistent.nii.gz") is False

    def test_validate_wrong_extension(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor

        path = str(tmp_path / "file.txt")
        (tmp_path / "file.txt").write_text("not nifti")
        pp = NFTYPreprocessor()
        assert pp.validate(path) is False

    def test_get_shape(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor
        import nibabel as nib

        data = np.zeros((8, 16, 32), dtype=np.float32)
        path = str(tmp_path / "shape.nii.gz")
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)

        pp = NFTYPreprocessor()
        assert pp.get_shape(path) == (8, 16, 32)

    def test_get_voxel_sizes(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor
        import nibabel as nib

        data = np.zeros((5, 5, 5), dtype=np.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        path = str(tmp_path / "voxel.nii.gz")
        nib.save(nib.Nifti1Image(data, affine), path)

        pp = NFTYPreprocessor()
        sizes = pp.get_voxel_sizes(path)
        assert sizes is not None
        assert pytest.approx(sizes[0], abs=0.1) == 2.0

    def test_get_metadata(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor
        import nibabel as nib

        data = np.zeros((5, 5, 5), dtype=np.float32)
        path = str(tmp_path / "meta.nii.gz")
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)

        pp = NFTYPreprocessor()
        meta = pp.get_metadata(path)
        assert meta["shape"] == (5, 5, 5)
        assert "voxel_sizes" in meta

    def test_load_file_not_found(self) -> None:
        from brainbow.preprocessors.nfty import NFTYPreprocessor

        pp = NFTYPreprocessor()
        with pytest.raises(FileNotFoundError):
            pp.load("/nonexistent.nii.gz")


# ---------------------------------------------------------------------------
# NRRDPreprocessor
# ---------------------------------------------------------------------------

class TestNRRDPreprocessor:
    """Tests for NRRDPreprocessor."""

    def test_supported_extensions(self) -> None:
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        pp = NRRDPreprocessor()
        assert ".nrrd" in pp.supported_extensions
        assert ".nhdr" in pp.supported_extensions

    def test_save_and_load(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        pp = NRRDPreprocessor()
        data = np.random.rand(8, 16, 16).astype(np.float32)
        path = str(tmp_path / "test.nrrd")

        pp.save(data, path)
        loaded = pp.load(path)
        np.testing.assert_array_almost_equal(data, loaded, decimal=5)

    def test_save_and_load_with_header(self, tmp_path: Path) -> None:
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        pp = NRRDPreprocessor()
        data = np.zeros((5, 5, 5), dtype=np.float32)
        path = str(tmp_path / "header.nrrd")

        pp.save(data, path, header={"encoding": "gzip"})
        loaded, header = pp.load_with_header(path)
        np.testing.assert_array_almost_equal(data, loaded)
        assert "encoding" in header

    def test_validate_valid(self, tmp_path: Path) -> None:
        import nrrd
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        data = np.zeros((5, 5, 5), dtype=np.float32)
        path = str(tmp_path / "valid.nrrd")
        nrrd.write(path, data)

        pp = NRRDPreprocessor()
        assert pp.validate(path) is True

    def test_validate_nonexistent(self) -> None:
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        pp = NRRDPreprocessor()
        assert pp.validate("/nonexistent.nrrd") is False

    def test_get_shape(self, tmp_path: Path) -> None:
        import nrrd
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        data = np.zeros((4, 8, 16), dtype=np.float32)
        path = str(tmp_path / "shape.nrrd")
        nrrd.write(path, data)

        pp = NRRDPreprocessor()
        shape = pp.get_shape(path)
        assert shape == (4, 8, 16)

    def test_get_metadata(self, tmp_path: Path) -> None:
        import nrrd
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        data = np.zeros((3, 3, 3), dtype=np.float32)
        path = str(tmp_path / "meta.nrrd")
        nrrd.write(path, data)

        pp = NRRDPreprocessor()
        meta = pp.get_metadata(path)
        assert "type" in meta
        assert "dimension" in meta

    def test_load_file_not_found(self) -> None:
        from brainbow.preprocessors.nrrd import NRRDPreprocessor

        pp = NRRDPreprocessor()
        with pytest.raises(FileNotFoundError):
            pp.load("/nonexistent.nrrd")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
