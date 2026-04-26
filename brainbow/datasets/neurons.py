"""
Neurons Dataset for neuron segmentation.

Kasthuri et al., Cell 2015 — mouse somatosensory cortex (S1), layer 4.
EM imagery at mip1 (2×2×10 nm), segmentation downloaded at mip0 then
downsampled 2× in XY to match.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from brainbow.transforms.find_boundaries import find_boundaries as _find_boundaries

from brainbow.datasets._patches import generate_patch_indices
from brainbow.datasets.base import CircuitDataset
from brainbow.preprocessors import HDF5Preprocessor, TIFFPreprocessor
from brainbow.utils.io import find_folder


class NeuronsDataset(CircuitDataset):
    """
    Neurons Dataset for neuron segmentation.

    Volume format: ``[{"vol": "volume_basename", "seg": "seg_basename"}]``

    Optional per-volume keys:
        - ``root``: override ``root_dir`` for this volume.
        - ``find_boundaries``: when > 0, boundary pixels between adjacent
          labels are zeroed out at load time.

    Args:
        root_dir: Path to directory containing Neurons data files.
        volumes: List of {vol, seg} dicts.
        transform: Optional MONAI transforms to apply.
        cache_rate: Fraction of data to cache in memory (default: 1.0).
        slice_mode: If True, return individual 2D slices (default: True).
        patch_size: If not None, return 3D patches of this size (z, y, x).
        patch_overlap: Overlap between patches (default: 0.25).
        num_samples: Number of samples per epoch.
    """

    _paper = (
        "Kasthuri, N., et al. (2015). Saturated Reconstruction of a Volume of "
        "Neocortex. Cell, 162(3), 648-661. doi:10.1016/j.cell.2015.06.054"
    )
    _resolution: Dict[str, float] = {"x": 2.0, "y": 2.0, "z": 10.0}
    _labels_base: List[str] = ["background", "neuron"]

    def __init__(
        self,
        root_dir: str,
        volumes: Optional[List[Dict[str, str]]] = None,
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        slice_mode: bool = True,
        patch_size: Optional[tuple] = None,
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

    def _load_volume(self, base_name: str, root_dir: Optional[Path] = None) -> Optional[np.ndarray]:
        search_dir = root_dir if root_dir is not None else self.root_dir
        path = find_folder(search_dir, base_name)
        if path is None:
            raise FileNotFoundError(
                f"Could not find data file '{base_name}' in {search_dir}.\n"
                f"Expected one of: {base_name}.h5, {base_name}.hdf5, "
                f"{base_name}.tiff, {base_name}.tif"
            )
        suffix = path.suffix.lower()
        if suffix in [".h5", ".hdf5"]:
            return self._hdf5_preprocessor.load(str(path))
        return self._tiff_preprocessor.load(str(path))

    def _generate_patch_indices(
        self,
        volume_shape: tuple,
        patch_size: tuple,
        overlap: float,
    ) -> List[Tuple[slice, slice, slice]]:
        return generate_patch_indices(volume_shape, patch_size, overlap)

    def _prepare_data(self) -> List[Dict[str, Any]]:
        data_list: List[Dict[str, Any]] = []
        total_slices = 0

        for vol_spec in self._get_volume_list():
            vol_root = Path(vol_spec["root"]) if "root" in vol_spec else None
            inputs = self._load_volume(vol_spec["vol"], root_dir=vol_root)
            if inputs is None:
                raise FileNotFoundError(
                    f"Could not load volume '{vol_spec['vol']}' from {self.root_dir}"
                )
            inputs = inputs.astype(np.float32)
            vmin, vmax = float(inputs.min()), float(inputs.max())
            if vmax > vmin:
                inputs = (inputs - vmin) / (vmax - vmin)

            labels: Optional[np.ndarray] = None
            try:
                labels = self._load_volume(
                    vol_spec["seg"], root_dir=vol_root
                ).astype(np.int64)
            except (FileNotFoundError, AttributeError):
                labels = None

            if labels is not None and float(vol_spec.get("find_boundaries", 0)) > 0:
                labels[_find_boundaries(labels, mode="inner")] = 0

            n_slices = inputs.shape[0]
            vol_name = vol_spec["vol"]

            if self.slice_mode:
                for si in range(n_slices):
                    entry: Dict[str, Any] = {
                        "image": inputs[si].copy(),
                        "slice_idx": si,
                        "volume": vol_name,
                        "idx": len(data_list),
                    }
                    if labels is not None:
                        entry["label"] = labels[si].copy()
                    data_list.append(entry)
                del inputs, labels

            elif self.patch_size is not None:
                patch_indices = self._generate_patch_indices(
                    inputs.shape, self.patch_size, self.patch_overlap,
                )
                for pidx, (z_sl, y_sl, x_sl) in enumerate(patch_indices):
                    entry = {
                        "image": inputs[z_sl, y_sl, x_sl],
                        "patch_idx": pidx,
                        "volume": vol_name,
                        "idx": len(data_list),
                    }
                    if labels is not None:
                        entry["label"] = labels[z_sl, y_sl, x_sl]
                    data_list.append(entry)

            else:
                entry = {
                    "image": inputs,
                    "volume": vol_name,
                    "idx": len(data_list),
                }
                if labels is not None:
                    entry["label"] = labels
                data_list.append(entry)

            total_slices += n_slices

        if self._num_samples is not None:
            self._virtual_len = self._num_samples
        elif not self.slice_mode and self.patch_size is None:
            self._virtual_len = total_slices

        return data_list
