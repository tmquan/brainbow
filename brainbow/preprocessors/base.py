"""
Abstract base for connectomics file-format preprocessors.

Why this file exists
--------------------
Each subclass isolates the I/O quirks of one format (HDF5 chunked
reads, NRRD headers, multi-page TIFF, NIfTI) so the rest of brainbow
can speak in plain :class:`numpy.ndarray`.

Public surface
--------------
* :class:`BasePreprocessor` -- the abstract class itself.

Required overrides
------------------
* :attr:`supported_extensions` -- tuple of suffixes (e.g.
  ``(".h5", ".hdf5")``).
* :meth:`load(path)` -- read a file into a :class:`numpy.ndarray`.
* :meth:`validate(path)` -- ``True`` if this preprocessor can read it.

Optional overrides
------------------
* :meth:`save(path, data)` -- writing back is optional.  The base class
  raises :class:`NotImplementedError` for read-only formats; that's a
  legitimate state, not a bug.
* :meth:`get_metadata`, :meth:`get_shape` -- the base implementations
  read the full file via :meth:`load`.  Override for cheap header-only
  paths on large files.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class BasePreprocessor(ABC):
    """
    Abstract base class for data format preprocessors.

    Preprocessors handle loading, validation, and conversion of data from
    specific file formats to a standard internal format (numpy arrays or
    torch tensors).

    All preprocessors must implement:
    - load(): Load data from file
    - validate(): Check if file is valid for this format
    - supported_extensions: Property listing supported file extensions

    Example:
        >>> class MyPreprocessor(BasePreprocessor):
        ...     @property
        ...     def supported_extensions(self) -> List[str]:
        ...         return [".my", ".mydata"]
        ...
        ...     def load(self, path: str) -> np.ndarray:
        ...         ...
        ...
        ...     def validate(self, path: str) -> bool:
        ...         return Path(path).suffix in self.supported_extensions
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        List of supported file extensions for this preprocessor.

        Returns:
            List of file extensions (including the dot), e.g., ['.h5', '.hdf5']
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> np.ndarray:
        """
        Load data from file.

        Args:
            path: Path to the data file.
            **kwargs: Additional format-specific options.

        Returns:
            Numpy array containing the loaded data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is invalid or unsupported.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, path: str) -> bool:
        """
        Validate that a file can be loaded by this preprocessor.

        Args:
            path: Path to the file to validate.

        Returns:
            True if file is valid and can be loaded, False otherwise.
        """
        raise NotImplementedError

    def to_tensor(
        self,
        data: np.ndarray,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Convert numpy array to torch tensor.

        Args:
            data: Numpy array to convert.
            dtype: Target torch dtype. If None, inferred from numpy dtype.
            device: Target device. If None, uses CPU.

        Returns:
            Torch tensor with the specified dtype and device.
        """
        tensor = torch.from_numpy(np.ascontiguousarray(data))

        if dtype is not None:
            tensor = tensor.to(dtype)

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def save(self, data: np.ndarray, path: str, **kwargs: Any) -> None:
        """
        Save data to file.

        Default implementation raises NotImplementedError. Subclasses should
        override this method to provide format-specific saving.

        Args:
            data: Numpy array to save.
            path: Output file path.
            **kwargs: Additional format-specific options.

        Raises:
            NotImplementedError: If saving is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support saving. "
            f"Override the save() method to add support."
        )

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract metadata from file without loading full data.

        Args:
            path: Path to the data file.

        Returns:
            Dictionary containing metadata.
        """
        file_path = Path(path)
        return {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": file_path.stat().st_size if file_path.exists() else None,
        }

    def get_shape(self, path: str) -> Optional[Tuple[int, ...]]:
        """
        Get shape of data without loading full array.

        Default implementation loads the data to get shape. Subclasses
        should override for more efficient implementations.

        Args:
            path: Path to the data file.

        Returns:
            Tuple of dimensions, or None if cannot be determined.
        """
        try:
            data = self.load(path)
            return data.shape
        except Exception:
            return None

    def _check_file_exists(self, path: str) -> Path:
        """
        Check that file exists and return Path object.

        Args:
            path: Path to check.

        Returns:
            Path object.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Expected a {', '.join(self.supported_extensions)} file."
            )
        return file_path

    def _check_extension(self, path: str) -> None:
        """
        Check that file has a supported extension.

        Args:
            path: Path to check.

        Raises:
            ValueError: If extension is not supported.
        """
        file_path = Path(path)
        if file_path.suffix.lower() not in [e.lower() for e in self.supported_extensions]:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}\n"
                f"Supported extensions: {self.supported_extensions}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(extensions={self.supported_extensions})"
