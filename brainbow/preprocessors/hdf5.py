"""
HDF5 format preprocessor for handling HDF5 datasets and hierarchical structures.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from brainbow.preprocessors.base import BasePreprocessor


class HDF5Preprocessor(BasePreprocessor):
    """
    Preprocessor for HDF5 datasets and hierarchical data structures.

    Handles loading of:
    - Single datasets within HDF5 files
    - Nested hierarchical data structures
    - Chunked and compressed datasets

    Uses h5py library for efficient I/O operations.

    Args:
        default_key: Default dataset key to load. Common options:
            - 'main': pytorch_connectomics convention
            - 'data': neuroglancer/CloudVolume convention
            - 'raw': Some EM datasets
            Default: 'main'

    Example:
        >>> preprocessor = HDF5Preprocessor(default_key='main')
        >>> volume = preprocessor.load("volume.h5")
        >>>
        >>> # Load specific dataset
        >>> data = preprocessor.load("file.h5", key="volumes/raw")
        >>>
        >>> # List available datasets
        >>> datasets = preprocessor.list_datasets("file.h5")
    """

    def __init__(self, default_key: str = "main") -> None:
        self.default_key = default_key

    @property
    def supported_extensions(self) -> List[str]:
        return [".h5", ".hdf5", ".hdf", ".he5"]

    def load(
        self,
        path: str,
        key: Optional[str] = None,
        slices: Optional[Tuple[slice, ...]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Load data from HDF5 file.

        Args:
            path: Path to HDF5 file.
            key: Dataset key within the file. If None, uses default_key,
                or tries common keys ('main', 'data', 'raw', 'volume').
            slices: Optional tuple of slices for partial loading.
            **kwargs: Additional arguments (unused, for compatibility).

        Returns:
            Numpy array containing the dataset.

        Raises:
            FileNotFoundError: If file does not exist.
            KeyError: If dataset key is not found.
            ValueError: If file cannot be read as HDF5.
        """
        import h5py

        file_path = self._check_file_exists(path)

        try:
            with h5py.File(str(file_path), "r", locking=False) as f:
                if key is not None:
                    dataset_key = key
                elif self.default_key in f:
                    dataset_key = self.default_key
                else:
                    common_keys = ["main", "data", "raw", "volume", "image", "label"]
                    dataset_key = None
                    for k in common_keys:
                        if k in f:
                            dataset_key = k
                            break

                    if dataset_key is None:
                        datasets = self._find_datasets(f)
                        if len(datasets) > 0:
                            dataset_key = datasets[0]
                        else:
                            raise KeyError(
                                f"No datasets found in {path}. "
                                f"File structure: {list(f.keys())}"
                            )

                if dataset_key not in f:
                    raise KeyError(
                        f"Dataset '{dataset_key}' not found in {path}.\n"
                        f"Available keys: {list(f.keys())}\n"
                        f"Try using: preprocessor.list_datasets('{path}')"
                    )

                dataset = f[dataset_key]

                if slices is not None:
                    return dataset[slices]
                else:
                    return dataset[:]

        except KeyError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load HDF5 file: {path}\n"
                f"Error: {e}\n"
                f"Ensure the file is a valid HDF5 format."
            )

    def validate(self, path: str) -> bool:
        """
        Validate that file is a readable HDF5.

        Args:
            path: Path to file.

        Returns:
            True if file exists and is a valid HDF5.
        """
        import h5py

        file_path = Path(path)

        if not file_path.exists():
            return False

        if file_path.suffix.lower() not in [e.lower() for e in self.supported_extensions]:
            return False

        try:
            with h5py.File(str(file_path), "r", locking=False) as f:
                _ = list(f.keys())
            return True
        except Exception:
            return False

    def save(
        self,
        data: np.ndarray,
        path: str,
        key: Optional[str] = None,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
        chunks: Optional[Union[bool, Tuple[int, ...]]] = True,
        **kwargs: Any,
    ) -> None:
        """
        Save data to HDF5 file.

        Args:
            data: Numpy array to save.
            path: Output file path.
            key: Dataset key. If None, uses default_key.
            compression: Compression method ('gzip', 'lzf', None).
            compression_opts: Compression level (0-9 for gzip).
            chunks: Chunk shape for storage. True for auto-chunking.
            **kwargs: Additional arguments passed to create_dataset.
        """
        import h5py

        dataset_key = key if key is not None else self.default_key

        with h5py.File(path, "w") as f:
            f.create_dataset(
                dataset_key,
                data=data,
                compression=compression,
                compression_opts=compression_opts if compression else None,
                chunks=chunks,
                **kwargs,
            )

    def _find_datasets(self, group: Any, prefix: str = "") -> List[str]:
        """
        Recursively find all datasets in an HDF5 group.

        Args:
            group: h5py Group or File object.
            prefix: Current path prefix.

        Returns:
            List of dataset paths.
        """
        import h5py

        datasets: List[str] = []
        for key in group.keys():
            full_path = f"{prefix}/{key}" if prefix else key
            item = group[key]
            if isinstance(item, h5py.Dataset):
                datasets.append(full_path)
            elif isinstance(item, h5py.Group):
                datasets.extend(self._find_datasets(item, full_path))
        return datasets

    def list_datasets(self, path: str) -> List[str]:
        """
        List all datasets in an HDF5 file.

        Args:
            path: Path to HDF5 file.

        Returns:
            List of dataset paths within the file.
        """
        import h5py

        file_path = self._check_file_exists(path)

        with h5py.File(str(file_path), "r", locking=False) as f:
            return self._find_datasets(f)

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract HDF5 metadata without loading full data.

        Args:
            path: Path to HDF5 file.

        Returns:
            Dictionary with HDF5-specific metadata.
        """
        import h5py

        file_path = self._check_file_exists(path)
        metadata = super().get_metadata(path)

        try:
            with h5py.File(str(file_path), "r", locking=False) as f:
                datasets = self._find_datasets(f)
                metadata.update(
                    {
                        "num_datasets": len(datasets),
                        "datasets": datasets[:10],
                        "root_attrs": dict(f.attrs),
                    }
                )

                target_key: Optional[str] = None
                if self.default_key in f:
                    target_key = self.default_key
                elif len(datasets) > 0:
                    target_key = datasets[0]

                if target_key is not None:
                    ds = f[target_key]
                    metadata.update(
                        {
                            "primary_dataset": target_key,
                            "shape": ds.shape,
                            "dtype": str(ds.dtype),
                            "chunks": ds.chunks,
                            "compression": ds.compression,
                        }
                    )
        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def get_shape(self, path: str, key: Optional[str] = None) -> Optional[Tuple[int, ...]]:
        """
        Get shape of HDF5 dataset without loading full array.

        Args:
            path: Path to HDF5 file.
            key: Dataset key. If None, uses default_key.

        Returns:
            Shape tuple of the dataset.
        """
        import h5py

        try:
            with h5py.File(str(path), "r", locking=False) as f:
                if key is not None:
                    dataset_key = key
                elif self.default_key in f:
                    dataset_key = self.default_key
                else:
                    datasets = self._find_datasets(f)
                    if len(datasets) > 0:
                        dataset_key = datasets[0]
                    else:
                        return None

                return f[dataset_key].shape
        except Exception:
            return None
