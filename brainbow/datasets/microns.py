"""
MICRONS Dataset for large-scale connectomics.

The MICrONS (Machine Intelligence from Cortical Networks) dataset.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from brainbow.transforms.find_boundaries import find_boundaries as _find_boundaries

from brainbow.datasets._patches import generate_patch_indices
from brainbow.datasets.base import CircuitDataset
from brainbow.preprocessors import HDF5Preprocessor, NRRDPreprocessor, TIFFPreprocessor
from brainbow.utils.io import find_folder


class MICRONSDataset(CircuitDataset):
    """
    MICRONS Dataset for large-scale cortical connectomics.

    Dataset from the MICrONS consortium containing petascale electron
    microscopy imaging of mouse visual cortex with dense neuron segmentation.

    Volume format: ``[{"vol": "volume_basename", "seg": "seg_basename"}]``

    Optional per-volume keys:
        - ``root``: override ``root_dir`` for this volume.
        - ``find_boundaries``: when > 0, boundary pixels between adjacent
          labels are zeroed out at load time via
          ``skimage.segmentation.find_boundaries``.

    Args:
        root_dir: Path to directory containing MICRONS data files.
        volumes: List of {vol, seg} dicts. Defaults to [{vol: "volume", seg: "segmentation"}].
        transform: Optional MONAI transforms to apply.
        cache_rate: Fraction of data to cache in memory (default: 1.0).
        slice_mode: If True, return individual 2D slices (default: True).
        patch_size: If not None, return 3D patches of this size (z, y, x).
        patch_overlap: Overlap between patches (default: 0.25).
    """

    _paper = (
        "MICrONS Consortium (2021). Functional connectomics spanning multiple "
        "areas of mouse visual cortex. bioRxiv. doi:10.1101/2021.07.28.454025"
    )
    _resolution: Dict[str, float] = {"x": 4.0, "y": 4.0, "z": 40.0}
    _labels_base: List[str] = ["background", "neuron"]

    def __init__(
        self,
        root_dir: str,
        volumes: Optional[List[Dict[str, str]]] = None,
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        slice_mode: bool = True,
        patch_size: Optional[Tuple[int, int, int]] = None,
        patch_overlap: float = 0.25,
        num_samples: Optional[int] = None,
        num_workers: int = 0,
    ) -> None:
        self.slice_mode = slice_mode
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self._num_samples = num_samples

        self._hdf5_preprocessor = HDF5Preprocessor()
        self._tiff_preprocessor = TIFFPreprocessor()
        self._nrrd_preprocessor = NRRDPreprocessor()

        super().__init__(
            root_dir=root_dir,
            volumes=volumes,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    @property
    def paper(self) -> str:
        return self._paper

    @property
    def resolution(self) -> Dict[str, float]:
        return self._resolution.copy()

    @property
    def labels(self) -> List[str]:
        return self._labels_base.copy()

    def _default_volumes(self) -> List[Dict[str, str]]:
        return [{"vol": "volume", "seg": "segmentation"}]

    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        vols = self._get_volume_list()
        if vols:
            return {"vol": vols[0]["vol"], "seg": vols[0]["seg"]}
        return {"vol": "volume", "seg": "segmentation"}

    def _load_volume(
        self,
        base_name: str,
        required: bool = True,
        root_dir: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Load volume data from file.

        Args:
            base_name: Base filename without extension.
            required: If True, raise error when not found.
            root_dir: Override the dataset root for this volume (used by
                ``vol_spec["root"]`` for parity with SNEMI3D / Neurons).

        Returns:
            Numpy array containing volume data, or None if not found.
        """
        search_dir = root_dir if root_dir is not None else self.root_dir
        path = find_folder(search_dir, base_name)

        if path is None:
            if required:
                raise FileNotFoundError(
                    f"Could not find data file '{base_name}' in {search_dir}.\n"
                    f"Expected one of: {base_name}.h5, {base_name}.tiff, {base_name}.nrrd"
                )
            return None

        suffix = path.suffix.lower()
        if suffix in [".h5", ".hdf5"]:
            return self._hdf5_preprocessor.load(str(path))
        elif suffix in [".tiff", ".tif"]:
            return self._tiff_preprocessor.load(str(path))
        else:
            return self._nrrd_preprocessor.load(str(path))

    def _generate_patch_indices(
        self,
        volume_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        overlap: float,
    ) -> List[Tuple[slice, slice, slice]]:
        return generate_patch_indices(volume_shape, patch_size, overlap)

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """Prepare data dictionaries from volume list."""
        data_list: List[Dict[str, Any]] = []
        total_slices = 0

        for vol_spec in self._get_volume_list():
            vol_root = Path(vol_spec["root"]) if "root" in vol_spec else None
            inputs = self._load_volume(str(vol_spec["vol"]), root_dir=vol_root)
            if inputs is None:
                raise FileNotFoundError(
                    f"Could not load volume '{vol_spec['vol']}' from "
                    f"{vol_root or self.root_dir}"
                )
            inputs = inputs.astype(np.float32)
            vmin, vmax = float(inputs.min()), float(inputs.max())
            if vmax > vmin:
                inputs = (inputs - vmin) / (vmax - vmin)

            labels: Optional[np.ndarray] = None
            labels = self._load_volume(
                str(vol_spec["seg"]), required=False, root_dir=vol_root,
            )
            if labels is not None:
                labels = labels.astype(np.int64)

            if labels is not None and float(vol_spec.get("find_boundaries", 0)) > 0:
                labels[_find_boundaries(labels, mode='inner')] = 0

            n_slices = inputs.shape[0]
            vol_name = vol_spec["vol"]

            if self.slice_mode:
                for si in range(n_slices):
                    entry: Dict[str, Any] = {
                        "image": inputs[si].copy(), "slice_idx": si,
                        "volume": vol_name, "idx": len(data_list),
                    }
                    if labels is not None:
                        entry["label"] = labels[si].copy()
                    data_list.append(entry)
                del inputs, labels

            elif self.patch_size is not None:
                patch_indices = self._generate_patch_indices(
                    inputs.shape, self.patch_size, self.patch_overlap
                )
                for pidx, (z_sl, y_sl, x_sl) in enumerate(patch_indices):
                    entry = {
                        "image": inputs[z_sl, y_sl, x_sl],
                        "patch_idx": pidx, "volume": vol_name,
                        "idx": len(data_list),
                    }
                    if labels is not None:
                        entry["label"] = labels[z_sl, y_sl, x_sl]
                    data_list.append(entry)

            else:
                entry = {"image": inputs, "volume": vol_name, "idx": len(data_list)}
                if labels is not None:
                    entry["label"] = labels
                data_list.append(entry)

            total_slices += n_slices

        if self._num_samples is not None:
            self._virtual_len = self._num_samples
        elif not self.slice_mode and self.patch_size is None:
            self._virtual_len = total_slices

        return data_list
