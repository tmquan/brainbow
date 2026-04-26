"""
TIFF preprocessor.

Reads / writes single-page and multi-page ``.tif`` / ``.tiff`` files
via :mod:`tifffile`.  Multi-page TIFFs are returned as ``[Z, Y, X]``.

See :class:`brainbow.preprocessors.BasePreprocessor` for the contract.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brainbow.preprocessors.base import BasePreprocessor


class TIFFPreprocessor(BasePreprocessor):
    """
    Preprocessor for TIFF stacks and multi-page TIFF files.

    Handles loading of:
    - Multi-page TIFF stacks (3D volumes stored as image sequences)
    - BigTIFF files for large datasets
    - Various bit depths (8-bit, 16-bit, 32-bit)

    Uses tifffile library for efficient I/O operations.

    Args:
        memmap: If True, use memory-mapped reading for large files.
            Default: False.

    Example:
        >>> preprocessor = TIFFPreprocessor()
        >>> volume = preprocessor.load("volume.tiff")
        >>> print(volume.shape)  # (100, 1024, 1024)
    """

    def __init__(self, memmap: bool = False) -> None:
        self.memmap = memmap

    @property
    def supported_extensions(self) -> List[str]:
        return [".tiff", ".tif"]

    def load(
        self,
        path: str,
        key: Optional[int] = None,
        series: int = 0,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Load data from TIFF file.

        Args:
            path: Path to TIFF file.
            key: Specific page/frame to load. If None, loads all pages.
            series: Series index for multi-series TIFF files.
            **kwargs: Additional arguments passed to tifffile.imread.

        Returns:
            Numpy array containing image data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file cannot be read as TIFF.
        """
        import tifffile

        file_path = self._check_file_exists(path)

        try:
            if self.memmap:
                if key is not None:
                    with tifffile.TiffFile(str(file_path)) as tif:
                        return tif.pages[key].asarray()
                else:
                    return tifffile.memmap(str(file_path))
            else:
                if key is not None:
                    return tifffile.imread(str(file_path), key=key, **kwargs)
                else:
                    return tifffile.imread(str(file_path), **kwargs)

        except Exception as e:
            raise ValueError(
                f"Failed to load TIFF file: {path}\n"
                f"Error: {e}\n"
                f"Ensure the file is a valid TIFF format."
            )

    def validate(self, path: str) -> bool:
        """
        Validate that file is a readable TIFF.

        Args:
            path: Path to file.

        Returns:
            True if file exists and is a valid TIFF.
        """
        import tifffile

        file_path = Path(path)

        if not file_path.exists():
            return False

        if file_path.suffix.lower() not in [e.lower() for e in self.supported_extensions]:
            return False

        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                _ = tif.pages[0].shape
            return True
        except Exception:
            return False

    def save(
        self,
        data: np.ndarray,
        path: str,
        compression: Optional[str] = None,
        bigtiff: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Save data to TIFF file.

        Args:
            data: Numpy array to save.
            path: Output file path.
            compression: Compression method ('zlib', 'lzw', None).
            bigtiff: Use BigTIFF format for files > 4GB.
            **kwargs: Additional arguments passed to tifffile.imwrite.
        """
        import tifffile

        tifffile.imwrite(
            path,
            data,
            compression=compression,
            bigtiff=bigtiff,
            **kwargs,
        )

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract TIFF metadata without loading full data.

        Args:
            path: Path to TIFF file.

        Returns:
            Dictionary with TIFF-specific metadata.
        """
        import tifffile

        file_path = self._check_file_exists(path)
        metadata = super().get_metadata(path)

        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                metadata.update(
                    {
                        "num_pages": len(tif.pages),
                        "num_series": len(tif.series),
                        "is_bigtiff": tif.is_bigtiff,
                        "byteorder": tif.byteorder,
                    }
                )

                if len(tif.pages) > 0:
                    page = tif.pages[0]
                    metadata.update(
                        {
                            "shape": page.shape,
                            "dtype": str(page.dtype),
                            "photometric": (
                                str(page.photometric) if hasattr(page, "photometric") else None
                            ),
                            "compression": (
                                str(page.compression) if hasattr(page, "compression") else None
                            ),
                        }
                    )
        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def get_shape(self, path: str) -> Optional[Tuple[int, ...]]:
        """
        Get shape of TIFF data without loading full array.

        Args:
            path: Path to TIFF file.

        Returns:
            Shape tuple (num_pages, height, width) or (height, width).
        """
        import tifffile

        try:
            with tifffile.TiffFile(str(path)) as tif:
                if len(tif.series) > 0:
                    return tif.series[0].shape
                elif len(tif.pages) > 0:
                    page_shape = tif.pages[0].shape
                    if len(tif.pages) > 1:
                        return (len(tif.pages),) + page_shape
                    return page_shape
        except Exception:
            pass

        return None
