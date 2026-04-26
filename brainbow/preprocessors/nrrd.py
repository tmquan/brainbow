"""
NRRD preprocessor.

Reads / writes ``.nrrd`` (Nearly Raw Raster Data) files via
:mod:`pynrrd`.  Used primarily by MICRONS volumes, where the raw EM
ships as NRRD chunks.

See :class:`brainbow.preprocessors.BasePreprocessor` for the contract.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brainbow.preprocessors.base import BasePreprocessor


class NRRDPreprocessor(BasePreprocessor):
    """
    Preprocessor for NRRD (Nearly Raw Raster Data) format files.

    Handles loading of:
    - Standard NRRD files with embedded or detached data
    - Various encodings (raw, gzip, bzip2)
    - Rich header metadata including spatial information

    NRRD is commonly used in medical imaging and neuroimaging for its
    support of spatial metadata (origin, spacing, orientation).

    Uses pynrrd library for I/O operations.

    Example:
        >>> preprocessor = NRRDPreprocessor()
        >>> volume, header = preprocessor.load_with_header("volume.nrrd")
        >>> print(header['space directions'])  # Voxel spacing
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".nrrd", ".nhdr"]

    def load(
        self,
        path: str,
        index_order: str = "C",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Load data from NRRD file.

        Args:
            path: Path to NRRD file.
            index_order: Array index order ('C' for row-major, 'F' for column-major).
            **kwargs: Additional arguments passed to nrrd.read.

        Returns:
            Numpy array containing the volume data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file cannot be read as NRRD.
        """
        import nrrd

        file_path = self._check_file_exists(path)

        try:
            data, _ = nrrd.read(str(file_path), index_order=index_order, **kwargs)
            return data

        except Exception as e:
            raise ValueError(
                f"Failed to load NRRD file: {path}\n"
                f"Error: {e}\n"
                f"Ensure the file is a valid NRRD format."
            )

    def load_with_header(
        self,
        path: str,
        index_order: str = "C",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load data and header from NRRD file.

        Args:
            path: Path to NRRD file.
            index_order: Array index order. Default: 'C'.
            **kwargs: Additional arguments passed to nrrd.read.

        Returns:
            Tuple of (data array, header dictionary).
        """
        import nrrd

        file_path = self._check_file_exists(path)

        try:
            data, header = nrrd.read(str(file_path), index_order=index_order, **kwargs)
            return data, dict(header)

        except Exception as e:
            raise ValueError(
                f"Failed to load NRRD file: {path}\n"
                f"Error: {e}"
            )

    def validate(self, path: str) -> bool:
        """
        Validate that file is a readable NRRD.

        Args:
            path: Path to file.

        Returns:
            True if file exists and is a valid NRRD.
        """
        import nrrd

        file_path = Path(path)

        if not file_path.exists():
            return False

        if file_path.suffix.lower() not in [e.lower() for e in self.supported_extensions]:
            return False

        try:
            header = nrrd.read_header(str(file_path))
            return "type" in header and "dimension" in header
        except Exception:
            return False

    def save(
        self,
        data: np.ndarray,
        path: str,
        header: Optional[Dict[str, Any]] = None,
        compression_level: int = 9,
        index_order: str = "C",
        **kwargs: Any,
    ) -> None:
        """
        Save data to NRRD file.

        Args:
            data: Numpy array to save.
            path: Output file path.
            header: Optional header dictionary with metadata.
            compression_level: Gzip compression level (1-9).
            index_order: Array index order. Default: 'C'.
            **kwargs: Additional arguments passed to nrrd.write.
        """
        import nrrd

        if header is None:
            header = {}

        if "encoding" not in header:
            header["encoding"] = "gzip"

        nrrd.write(
            path,
            data,
            header=header,
            compression_level=compression_level,
            index_order=index_order,
            **kwargs,
        )

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract NRRD header metadata without loading full data.

        Args:
            path: Path to NRRD file.

        Returns:
            Dictionary with NRRD header fields.
        """
        import nrrd

        file_path = self._check_file_exists(path)
        metadata = super().get_metadata(path)

        try:
            header = nrrd.read_header(str(file_path))
            header_dict = dict(header)
            metadata.update(
                {
                    "nrrd_header": header_dict,
                    "type": header_dict.get("type"),
                    "dimension": header_dict.get("dimension"),
                    "sizes": header_dict.get("sizes"),
                    "encoding": header_dict.get("encoding"),
                    "space": header_dict.get("space"),
                    "space_directions": header_dict.get("space directions"),
                    "space_origin": header_dict.get("space origin"),
                }
            )
        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def get_shape(self, path: str) -> Optional[Tuple[int, ...]]:
        """
        Get shape of NRRD data without loading full array.

        Args:
            path: Path to NRRD file.

        Returns:
            Shape tuple from header 'sizes' field.
        """
        import nrrd

        try:
            header = nrrd.read_header(str(path))
            sizes = header.get("sizes")
            if sizes is not None:
                return tuple(sizes)
        except Exception:
            pass

        return None

    def get_spacing(self, path: str) -> Optional[Tuple[float, ...]]:
        """
        Get voxel spacing from NRRD header.

        Args:
            path: Path to NRRD file.

        Returns:
            Tuple of spacing values for each dimension, or None.
        """
        import nrrd

        try:
            header = nrrd.read_header(str(path))
            space_directions = header.get("space directions")

            if space_directions is not None:
                spacing = []
                for direction in space_directions:
                    if direction is not None:
                        magnitude = float(np.sqrt(np.sum(np.array(direction) ** 2)))
                        spacing.append(magnitude)
                return tuple(spacing) if spacing else None

            spacings = header.get("spacings")
            if spacings is not None:
                return tuple(spacings)

        except Exception:
            pass

        return None
