"""
NIfTI format preprocessor for handling .nii and .nii.gz volumes.

Used for connectomics volumes stored as NIfTI, e.g. EM images and
segmentation labels in NIfTI format (nnU-Net convention).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brainbow.preprocessors.base import BasePreprocessor


class NFTYPreprocessor(BasePreprocessor):
    """
    Preprocessor for NIfTI (.nii, .nii.gz) volumes.

    Handles loading of:
    - Standard NIfTI-1 files (.nii)
    - Gzip-compressed NIfTI files (.nii.gz)
    - Affine and header metadata extraction

    Uses nibabel for I/O operations.

    Example:
        >>> preprocessor = NFTYPreprocessor()
        >>> volume = preprocessor.load("image.nii.gz")
        >>> print(volume.shape)
        >>>
        >>> volume, affine = preprocessor.load_with_affine("image.nii.gz")
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".nii", ".nii.gz"]

    def load(
        self,
        path: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Load data from NIfTI file.

        Args:
            path: Path to .nii or .nii.gz file.
            **kwargs: Additional arguments (unused, for compatibility).

        Returns:
            Numpy array containing volume data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file cannot be read as NIfTI.
        """
        import nibabel as nib

        file_path = self._check_file_exists(path)

        try:
            img = nib.load(str(file_path))
            return np.asarray(img.dataobj)
        except Exception as e:
            raise ValueError(
                f"Failed to load NIfTI file: {path}\n"
                f"Error: {e}\n"
                f"Ensure the file is a valid NIfTI format."
            )

    def load_with_affine(
        self,
        path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data and affine transform from NIfTI file.

        Args:
            path: Path to .nii or .nii.gz file.

        Returns:
            Tuple of (data array, 4x4 affine matrix).
        """
        import nibabel as nib

        file_path = self._check_file_exists(path)

        try:
            img = nib.load(str(file_path))
            return np.asarray(img.dataobj), img.affine
        except Exception as e:
            raise ValueError(
                f"Failed to load NIfTI file: {path}\n"
                f"Error: {e}"
            )

    def validate(self, path: str) -> bool:
        """
        Validate that file is a readable NIfTI.

        Args:
            path: Path to file.

        Returns:
            True if file exists and is a valid NIfTI.
        """
        import nibabel as nib

        file_path = Path(path)

        if not file_path.exists():
            return False

        # .nii.gz has suffix .gz but we need to check the full name
        name = file_path.name
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            return False

        try:
            img = nib.load(str(file_path))
            _ = img.shape
            return True
        except Exception:
            return False

    def save(
        self,
        data: np.ndarray,
        path: str,
        affine: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """
        Save data to NIfTI file.

        Args:
            data: Numpy array to save.
            path: Output file path (.nii or .nii.gz).
            affine: 4x4 affine matrix. Defaults to identity.
            **kwargs: Additional arguments (unused).
        """
        import nibabel as nib

        if affine is None:
            affine = np.eye(4)

        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(path))

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract NIfTI header metadata without loading full data.

        Args:
            path: Path to NIfTI file.

        Returns:
            Dictionary with NIfTI header fields.
        """
        import nibabel as nib

        file_path = self._check_file_exists(path)
        metadata = super().get_metadata(path)

        try:
            img = nib.load(str(file_path))
            hdr = img.header

            metadata.update({
                "shape": img.shape,
                "dtype": str(img.get_data_dtype()),
                "affine": img.affine.tolist(),
                "voxel_sizes": tuple(float(v) for v in hdr.get_zooms()),
            })
        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def get_shape(self, path: str) -> Optional[Tuple[int, ...]]:
        """
        Get shape of NIfTI data without loading full array.

        Args:
            path: Path to NIfTI file.

        Returns:
            Shape tuple.
        """
        import nibabel as nib

        try:
            img = nib.load(str(path))
            return img.shape
        except Exception:
            return None

    def get_voxel_sizes(self, path: str) -> Optional[Tuple[float, ...]]:
        """
        Get voxel sizes (spacing) from NIfTI header.

        Args:
            path: Path to NIfTI file.

        Returns:
            Tuple of voxel sizes per dimension, or None.
        """
        import nibabel as nib

        try:
            img = nib.load(str(path))
            return tuple(float(v) for v in img.header.get_zooms())
        except Exception:
            return None

    def _check_file_exists(self, path: str) -> Path:
        """Override to handle .nii.gz double extension."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Expected a .nii or .nii.gz file."
            )
        return file_path
