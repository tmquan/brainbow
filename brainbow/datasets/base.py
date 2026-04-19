"""
Base dataset class for connectomics research.

Provides abstract interface that all connectomics datasets must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from monai.data import CacheDataset
from monai.transforms import Randomizable


class CircuitDataset(CacheDataset, Randomizable, ABC):
    """
    Abstract base class for connectomics datasets.

    Datasets receive a ``volumes`` list — each item is a dict with at least
    ``vol`` and ``seg`` keys (basenames or paths).  The dataset loads
    everything in the list without any splitting logic.  The datamodule
    is responsible for choosing which volumes go to train / val / test.

    All connectomics datasets must implement the following properties:
    - paper: Reference or citation metadata (string)
    - resolution: Voxel/spatial resolution specification (dict)
    - labels: List of segmentation class labels (list)
    - data_files: Dictionary with 'vol' and 'seg' keys for data paths/arrays

    Args:
        root_dir: Root directory containing the dataset files.
        volumes: List of volume dicts, each with ``vol`` and ``seg`` keys.
            When ``None``, the dataset falls back to its own defaults.
        transform: Optional transforms to apply to each sample.
        cache_rate: Fraction of data to cache in memory (0.0 to 1.0).
        num_workers: Number of workers for data loading.
    """

    def __init__(
        self,
        root_dir: str,
        volumes: Optional[List[Dict[str, str]]] = None,
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.volumes = volumes
        self._virtual_len: Optional[int] = None

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found: {self.root_dir}\n"
                f"Please ensure the data is downloaded and extracted to this location."
            )

        data_dicts = self._prepare_data()

        if len(data_dicts) == 0:
            raise ValueError(
                f"No data found in {self.root_dir}.\n"
                f"Expected data files: {self.data_files}"
            )

        super().__init__(
            data=data_dicts,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        if self._virtual_len is not None:
            return self._virtual_len
        return super().__len__()

    def __getitem__(self, index: int) -> Any:
        real_index = index % super().__len__() if self._virtual_len is not None else index
        return super().__getitem__(real_index)

    def _get_volume_list(self) -> List[Dict[str, str]]:
        """Return the volume list, falling back to dataset-specific defaults."""
        if self.volumes is not None:
            return self.volumes
        return self._default_volumes()

    def _default_volumes(self) -> List[Dict[str, str]]:
        """Override in subclasses to provide default volumes when none specified."""
        return []

    @property
    @abstractmethod
    def paper(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def resolution(self) -> Dict[str, float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def _prepare_data(self) -> List[Dict[str, Any]]:
        """
        Prepare list of data dictionaries for each sample.

        Loops over ``self._get_volume_list()`` and loads each volume.

        Returns:
            List of dictionaries, each containing sample data.
        """
        raise NotImplementedError

    def get_resolution_tuple(self) -> Tuple[float, float, float]:
        res = self.resolution
        return (res["z"], res["y"], res["x"])

    def get_anisotropy_factor(self) -> float:
        res = self.resolution
        xy_res = (res["x"] + res["y"]) / 2.0
        return res["z"] / xy_res

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  root_dir={self.root_dir},\n"
            f"  volumes={len(self._get_volume_list())} entries,\n"
            f"  paper='{self.paper}',\n"
            f"  resolution={self.resolution},\n"
            f"  labels={self.labels},\n"
            f"  num_samples={len(self)}\n"
            f")"
        )
