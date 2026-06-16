"""
Neurons DataModule for PyTorch Lightning.

Uses :class:`LazyVolDataset` for 3D patch mode and
:class:`NeuronsDataset` for 2D slice mode.
"""

import logging
from typing import Dict, List, Optional, Tuple

from brainbow.datamodules.base import CircuitDataModule
from brainbow.datasets import NeuronsDataset

logger = logging.getLogger(__name__)


class NeuronsDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for Neurons dataset.

    In 3D patch mode (``slice_mode=False`` with ``patch_size``), uses
    :class:`LazyVolDataset` for on-demand disk reads.

    Args:
        train_volumes: e.g. ``[{"vol": "train_volume", "seg": "train_seg"}]``
        test_volumes: e.g. ``[{"vol": "test_volume", "seg": "test_seg"}]``
        slice_mode: Return 2D slices if True (default: True).
    """

    dataset_class = NeuronsDataset

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        slice_mode: bool = True,
        patch_size: Optional[Tuple[int, int, int]] = None,
        num_samples: Optional[int] = None,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 6,
        find_boundaries: float = 0.0,
        boundary_target: str = "both",
        pixel_size: Optional[Tuple[float, ...]] = None,
        min_foreground: float = 0.0,
        elastic_prob: float = 0.0,
        elastic_sigma_range: Tuple[float, float] = (35.0, 50.0),
        elastic_magnitude_range: Tuple[float, float] = (10.0, 40.0),
        resolution_zoom_prob: float = 0.0,
        resolution_zoom_range: Optional[Tuple[Tuple[float, float], ...]] = None,
        resolution_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        self.slice_mode = slice_mode
        self.num_samples = num_samples
        self.save_hyperparameters()
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            cache_rate=cache_rate,
            pin_memory=pin_memory,
            image_size=image_size,
            patch_size=patch_size,
            train_volumes=train_volumes,
            val_volumes=val_volumes,
            test_volumes=test_volumes,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            find_boundaries=find_boundaries,
            boundary_target=boundary_target,
            pixel_size=pixel_size,
            min_foreground=min_foreground,
            elastic_prob=elastic_prob,
            elastic_sigma_range=elastic_sigma_range,
            elastic_magnitude_range=elastic_magnitude_range,
            resolution_zoom_prob=resolution_zoom_prob,
            resolution_zoom_range=resolution_zoom_range,
            resolution_map=resolution_map,
        )

    @property
    def _use_lazy(self) -> bool:
        return not self.slice_mode and self.patch_size is not None

    def _get_dataset_kwargs(self) -> dict:
        kwargs: dict = {
            "slice_mode": self.slice_mode,
            "patch_size": self.patch_size,
        }
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        return kwargs

    def _get_spatial_dims(self) -> int:
        return 2 if self.slice_mode else 3

    def setup(self, stage: Optional[str] = None) -> None:
        if not self._use_lazy:
            return super().setup(stage)

        num_samples = self.num_samples or 16000
        read_size = self._effective_read_size()

        if stage == "fit" or stage is None:
            if not self.train_volumes:
                raise ValueError(
                    "NeuronsDataModule requires `train_volumes` to be a non-empty "
                    "list in lazy 3-D patch mode; got %r.  Set it in your YAML "
                    "(e.g. `data.train_volumes: [{vol: ..., seg: ...}]`) or "
                    "switch to slice mode (`slice_mode: true`)."
                    % (self.train_volumes,)
                )
            val_vols = self.val_volumes or self.train_volumes
            self.train_dataset = self._build_lazy_split(
                self.train_volumes, read_size,
                self.get_train_transforms(), num_samples,
            )
            self.val_dataset = self._build_lazy_split(
                val_vols, self.patch_size,
                self.get_val_transforms(), num_samples,
            )

        if stage == "test" or stage is None:
            test_vols = self.test_volumes or self.train_volumes
            self.test_dataset = self._build_lazy_split(
                test_vols, self.patch_size,
                self.get_val_transforms(), num_samples,
            )

        logger.info("NeuronsDataModule: using LazyVolDataset (~0 MB base RAM per rank)")
