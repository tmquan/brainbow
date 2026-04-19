"""
Tests for DataModule classes and helper wrappers.

Covers:
- CircuitDataModule (base): hyperparameters, transforms, dataloaders
- SNEMI3DDataModule: dataset_class binding, kwargs forwarding
- MICRONSDataModule: dataset_class binding, kwargs forwarding
- NeuronsDataModule: dataset_class binding, kwargs forwarding
"""

from typing import Any, Dict, List, Union

import numpy as np
import pytest

from brainbow.datasets.base import CircuitDataset
from brainbow.datamodules.base import CircuitDataModule
from brainbow.datamodules.snemi3d import SNEMI3DDataModule
from brainbow.datamodules.microns import MICRONSDataModule
from brainbow.datamodules.neurons import NeuronsDataModule


# ---------------------------------------------------------------------------
# Helpers: minimal concrete implementations for testing
# ---------------------------------------------------------------------------

class _DummyDataset(CircuitDataset):
    """Minimal concrete dataset that yields synthetic samples."""

    def __init__(
        self,
        root_dir: str = ".",
        volumes: Any = None,
        transform: Any = None,
        cache_rate: float = 0.0,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        self.volumes = volumes
        self._transform = transform
        self._data = [
            {"image": np.random.rand(32, 32).astype(np.float32),
             "label": np.random.randint(0, 5, (32, 32)).astype(np.int64)}
            for _ in range(8)
        ]

    @property
    def paper(self) -> str:
        return "Dummy"

    @property
    def resolution(self) -> Dict[str, float]:
        return {"x": 1.0, "y": 1.0, "z": 1.0}

    @property
    def labels(self) -> List[str]:
        return ["bg", "fg"]

    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        return {"vol": "v.h5", "seg": "s.h5"}

    def _prepare_data(self) -> List[Dict[str, Any]]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._data[idx]
        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class _DummyDataModule(CircuitDataModule):
    """Concrete datamodule wired to _DummyDataset."""

    dataset_class = _DummyDataset  # type: ignore[assignment]

    def _get_spatial_dims(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Tests: CircuitDataModule (base)
# ---------------------------------------------------------------------------

class TestCircuitDataModule:

    def test_hyperparameters_stored(self) -> None:
        dm = _DummyDataModule(data_root="/tmp", batch_size=8, num_workers=2)
        assert dm.batch_size == 8
        assert dm.num_workers == 2
        assert dm.data_root == "/tmp"

    def test_persistent_workers_disabled_when_zero(self) -> None:
        dm = _DummyDataModule(data_root=".", num_workers=0, persistent_workers=True)
        assert dm.persistent_workers is False

    def test_setup_creates_datasets(self) -> None:
        dm = _DummyDataModule(data_root=".", batch_size=2, num_workers=0)
        dm.setup("fit")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_train_dataloader_returns_loader(self) -> None:
        dm = _DummyDataModule(data_root=".", batch_size=2, num_workers=0)
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape[0] == 2

    def test_val_dataloader_returns_loader(self) -> None:
        dm = _DummyDataModule(data_root=".", batch_size=2, num_workers=0)
        dm.setup("fit")
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert "image" in batch

    def test_get_train_transforms_returns_compose(self) -> None:
        from monai.transforms import Compose
        dm = _DummyDataModule(data_root=".")
        assert isinstance(dm.get_train_transforms(), Compose)

    def test_get_val_transforms_returns_compose(self) -> None:
        from monai.transforms import Compose
        dm = _DummyDataModule(data_root=".")
        assert isinstance(dm.get_val_transforms(), Compose)


# ---------------------------------------------------------------------------
# Concrete datamodule bindings
# ---------------------------------------------------------------------------

class TestSNEMI3DDataModule:

    def test_dataset_class_set(self) -> None:
        from brainbow.datasets import SNEMI3DDataset
        assert SNEMI3DDataModule.dataset_class is SNEMI3DDataset

    def test_kwargs_forwarded(self) -> None:
        dm = SNEMI3DDataModule(data_root=".", slice_mode=True, num_workers=0)
        assert dm._get_dataset_kwargs() == {"slice_mode": True}

    def test_patch_size_stored(self) -> None:
        dm = SNEMI3DDataModule(data_root=".", patch_size=[32, 128, 128])
        assert dm.patch_size == (32, 128, 128)


class TestMICRONSDataModule:

    def test_dataset_class_set(self) -> None:
        from brainbow.datasets import MICRONSDataset
        assert MICRONSDataModule.dataset_class is MICRONSDataset

    def test_kwargs_forwarded(self) -> None:
        dm = MICRONSDataModule(
            data_root=".",
            slice_mode=False,
            patch_size=(16, 64, 64),
            train_volumes=[{"vol": "train_volume", "seg": "train_seg"}],
        )
        kw = dm._get_dataset_kwargs()
        assert kw["slice_mode"] is False
        assert kw["patch_size"] == (16, 64, 64)


class TestNeuronsDataModule:

    def test_dataset_class_set(self) -> None:
        from brainbow.datasets import NeuronsDataset
        assert NeuronsDataModule.dataset_class is NeuronsDataset

    def test_kwargs_forwarded(self) -> None:
        dm = NeuronsDataModule(data_root=".", slice_mode=True, num_workers=0)
        kw = dm._get_dataset_kwargs()
        assert kw["slice_mode"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
