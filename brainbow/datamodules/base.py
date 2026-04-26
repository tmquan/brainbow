"""
Base PyTorch Lightning DataModule for connectomics datasets.

Why this file exists
--------------------
Every dataset in brainbow shares the same MONAI augmentation pipeline,
the same train/val/test DataLoader plumbing, and the same hooks for
loss-target precomputation (instance relabel after crop, direction /
covariance fields, find-boundaries).  Keeping all of that in one place
means a new dataset is a 30-50 line subclass that only declares
``dataset_class``.

Public surface
--------------
* :class:`CircuitDataModule` -- shared base.

Required overrides for subclasses
---------------------------------
* :attr:`dataset_class` -- a :class:`CircuitDataset` subclass.

Optional overrides
------------------
* :meth:`_get_dataset_kwargs` -- add per-dataset kwargs to the
  ``__init__`` of :attr:`dataset_class`.
* :meth:`_instance_transforms`, :meth:`_semantic_transforms`,
  :meth:`_geometry_transforms` -- inject extra label-target transforms
  before the volume is handed to the loss.
* :meth:`_get_spatial_dims` -- 3 by default; override for 2-D datasets.
"""

from abc import ABC
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    CenterSpatialCropd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized,
    RandFlipd,
    SpatialPadd,
    RandRotate90d,
    Rand3DElasticd,
    RandSpatialCropd,
    RandAdjustContrastd,
    RandGaussianNoised,
)

from brainbow.datasets.base import CircuitDataset
from brainbow.transforms import (
    FindBoundariesd, 
    Labeld, 
    Directiond, 
    Covarianced,
    RandTransposeXYd,
    RandResolutionZoomd, 
    RandSpatialCropForegroundd, 
)


class CircuitDataModule(pl.LightningDataModule, ABC):
    """
    Base PyTorch Lightning DataModule for connectomics datasets.

    Subclasses set ``dataset_class``, override ``_get_dataset_kwargs``,
    and optionally override the three label-target hooks
    (``_instance_transforms``, ``_semantic_transforms``,
    ``_geometry_transforms``) or ``_get_spatial_dims`` for the
    appropriate dimensionality.

    Pipeline order (train)::

        EnsureChannelFirst → [FindBoundaries]
        → [Pad + Crop(safe_size)] → [ResolutionZoom] → [CenterCrop(patch_size)]
        → spatial augmentations (flip/rot90/elastic)
        → instance_transforms (CC-relabel) → intensity augmentations
        → geometry_transforms → EnsureType

    When resolution zoom can downsample (zoom < 1), an enlarged *safe*
    crop is taken first so the zoom's center-crop/pad never introduces
    zero-padded edges into the final patch.

    Args:
        data_root: Path to the data directory.
        batch_size: Batch size for training and validation.
        num_workers: Number of worker processes for data loading.
        cache_rate: Fraction of data to cache in memory (default: 0.5).
        pin_memory: Whether to pin memory for faster GPU transfer.
        image_size: Optional image size for resizing.
        patch_size: Spatial crop size (enables crop pipeline when set).
        train_volumes: Volume list for training (dataset-specific format).
        val_volumes: Volume list for validation (defaults to train_volumes).
        test_volumes: Volume list for testing (defaults to train_volumes).
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: How many batches each DataLoader worker pre-stages
            ahead of the trainer.  Higher values smooth out tail-latency
            spikes from slow HDF5 chunk decodes (lazy loaders) at the cost
            of ``num_workers * prefetch_factor * batch`` extra RAM.  Set
            via the ``data.prefetch_factor`` Hydra knob; default ``6``.
    """

    dataset_class: Type[CircuitDataset] = CircuitDataset  # type: ignore[type-abstract]

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        patch_size: Optional[Union[Tuple[int, ...], List[int]]] = None,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 6,
        compute_geometry: bool = True,
        find_boundaries: float = 0.0,
        min_foreground: float = 0.0,
        pixel_size: Optional[Tuple[float, ...]] = None,
        elastic_prob: float = 0.0,
        elastic_sigma_range: Tuple[float, float] = (35.0, 50.0),
        elastic_magnitude_range: Tuple[float, float] = (10.0, 40.0),
        resolution_zoom_prob: float = 0.0,
        resolution_zoom_range: Optional[Tuple[Tuple[float, float], ...]] = None,
        resolution_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        self.train_volumes = train_volumes
        self.val_volumes = val_volumes if val_volumes is not None else train_volumes
        self.test_volumes = test_volumes if test_volumes is not None else train_volumes
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = int(prefetch_factor)
        self.compute_geometry = compute_geometry
        self.find_boundaries = float(find_boundaries)
        self.min_foreground = float(min_foreground)
        self.pixel_size = tuple(pixel_size) if pixel_size is not None else None
        self.elastic_prob = float(elastic_prob)
        self.elastic_sigma_range = tuple(elastic_sigma_range)
        self.elastic_magnitude_range = tuple(elastic_magnitude_range)
        self.resolution_zoom_prob = float(resolution_zoom_prob)
        self.resolution_zoom_range = (
            tuple(tuple(r) for r in resolution_zoom_range)
            if resolution_zoom_range is not None
            else None
        )
        self.resolution_map = (
            {k: tuple(v) for k, v in resolution_map.items()}
            if resolution_map is not None
            else None
        )

        self.train_dataset: Optional[CircuitDataset] = None
        self.val_dataset: Optional[CircuitDataset] = None
        self.test_dataset: Optional[CircuitDataset] = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _get_dataset_kwargs(self) -> dict:
        """Override in subclasses to provide dataset-specific arguments."""
        return {}

    def _get_spatial_dims(self) -> int:
        """Return the number of spatial dimensions for the current config.

        Override in subclasses that support ``slice_mode`` or other
        dimension-switching logic.  Default is 3 (volumetric).
        """
        return 3

    # ------------------------------------------------------------------
    # Label-target transform hooks  (override to customise)
    # ------------------------------------------------------------------

    def _effective_read_size(self) -> Optional[tuple]:
        """Patch size that LazyVolDataset should read from disk.

        Returns ``_safe_patch_size()`` when extra margin is needed for
        zoom-out, otherwise ``self.patch_size``.  Callers that create
        ``LazyVolDataset`` should use this so the volume data entering
        the transform pipeline is large enough for the safe-crop /
        zoom / center-crop sequence to produce a zero-padding-free
        output.
        """
        return self._safe_patch_size() or self.patch_size

    def _safe_patch_size(self) -> Optional[tuple]:
        """Enlarged crop size that stays fully valid after worst-case zoom.

        When the resolution zoom can downsample (zoom < 1), the zoomed
        volume is smaller and ``_zoom_volume`` zero-pads the borders.
        By cropping to a larger *safe* size first, applying the zoom,
        then center-cropping to the final ``patch_size``, the output
        contains only valid data.

        Returns ``None`` when no extra margin is needed (zoom >= 1 for
        every axis, or zoom is disabled).
        """
        if (
            self.patch_size is None
            or self.resolution_zoom_prob <= 0
            or self.pixel_size is None
            or self._get_spatial_dims() != 3
        ):
            return None

        import math
        from brainbow.transforms.resolution_zoom import DEFAULT_TARGET_RANGE

        target_range = self.resolution_zoom_range or DEFAULT_TARGET_RANGE

        safe = []
        needs_margin = False
        for d in range(len(self.patch_size)):
            min_zoom = self.pixel_size[d] / target_range[d][1]
            if min_zoom < 1.0:
                safe.append(math.ceil(self.patch_size[d] / min_zoom))
                needs_margin = True
            else:
                safe.append(self.patch_size[d])

        return tuple(safe) if needs_margin else None

    def _resolution_zoom_transforms(self, spatial_dims: int) -> list:
        """Random resolution zoom (rescale to simulate different voxel size).

        Inserted between the safe crop and the final center crop so that
        only valid (non-padded) voxels reach the model.
        """
        if (
            spatial_dims != 3
            or self.resolution_zoom_prob <= 0
            or self.pixel_size is None
        ):
            return []
        target_range = self.resolution_zoom_range
        kwargs: dict = {
            "keys": ["image", "label"],
            "native_resolution": self.pixel_size,
            "prob": self.resolution_zoom_prob,
        }
        if target_range is not None:
            kwargs["target_range"] = target_range
        if self.resolution_map is not None:
            kwargs["resolution_map"] = self.resolution_map
        return [RandResolutionZoomd(**kwargs)]

    def _original_transforms(self, spatial_dims: int) -> list:
        """Spatial augmentations applied to both image and label.

        Flips, 90-degree rotations, and (3-D only) elastic deformation.
        Elastic uses a high sigma for a sparse, smooth displacement field.
        Override to customise augmentation strategy.
        """
        io_keys = ["image", "label"]
        rot_axes = (0, 1) if spatial_dims == 2 else (1, 2)
        xforms: list = [
            RandFlipd(keys=io_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=io_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=io_keys, prob=0.5, spatial_axis=2 if spatial_dims == 3 else 1),
            RandRotate90d(keys=io_keys, prob=0.5, spatial_axes=rot_axes),
            RandTransposeXYd(keys=io_keys, prob=0.5),
        ]
        if spatial_dims == 3 and self.elastic_prob > 0:
            xforms.append(
                Rand3DElasticd(
                    keys=io_keys,
                    sigma_range=self.elastic_sigma_range,
                    magnitude_range=self.elastic_magnitude_range,
                    prob=self.elastic_prob,
                    mode=("bilinear", "nearest"),
                    padding_mode="reflection",
                ),
            )
        return xforms

    def _semantic_transforms(self, spatial_dims: int) -> list:
        """Image intensity augmentations and semantic-level label transforms.

        Runs after spatial augmentations.  Override to add semantic
        targets (e.g. boundary maps, class maps).
        """
        return [
            RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.05),
            RandAdjustContrastd(keys=["image"], prob=1.0, gamma=(0.5, 2.0)),
        ]

    def _instance_transforms(self, spatial_dims: int) -> list:
        """Post-crop connected-component relabeling.

        Splits instances that became disconnected after cropping and
        renumbers labels sequentially.  Runs immediately after crop.
        """
        return [Labeld(keys=["label"], spatial_dims=spatial_dims)]

    def _geometry_transforms(self, spatial_dims: int) -> list:
        """Direction and covariance targets for the geometry loss head.

        Runs after spatial augmentations so targets are consistent with
        the augmented label layout.  Skipped when ``compute_geometry=False``.
        """
        if not self.compute_geometry:
            return []
        return [
            Directiond(keys=["label"], spatial_dims=spatial_dims),
            Covarianced(keys=["label"], spatial_dims=spatial_dims),
        ]

    # ------------------------------------------------------------------
    # Pipeline assembly
    # ------------------------------------------------------------------

    def _output_keys(self) -> list:
        """All keys that must pass through ``EnsureTyped``."""
        keys = ["image", "label"]
        if self.compute_geometry:
            keys.extend(["label_direction", "label_covariance"])
        return keys

    def get_train_transforms(self) -> Compose:
        io_keys = ["image", "label"]
        sd = self._get_spatial_dims()

        transforms: list = [
            EnsureChannelFirstd(keys=io_keys, channel_dim="no_channel"),
        ]

        if self.find_boundaries > 0:
            transforms.append(
                FindBoundariesd(
                    keys=["label"],
                    prob=self.find_boundaries,
                    pixel_size=self.pixel_size,
                ),
            )

        safe_size = self._safe_patch_size()

        if safe_size is not None:
            # Crop to a larger safe patch, relabel to compact IDs (avoids
            # float32 precision loss for large uint64 segment IDs during
            # zoom), zoom, then center-crop to final size.
            transforms.append(SpatialPadd(keys=io_keys, spatial_size=safe_size))
            if self.min_foreground > 0:
                transforms.append(
                    RandSpatialCropForegroundd(
                        keys=io_keys,
                        spatial_size=safe_size,
                        label_key="label",
                        min_foreground=self.min_foreground,
                    )
                )
            else:
                transforms.append(
                    RandSpatialCropd(keys=io_keys, roi_size=safe_size, random_size=False),
                )
            transforms.append(Labeld(keys=["label"], spatial_dims=sd))
            transforms.extend(self._resolution_zoom_transforms(sd))
            transforms.append(
                CenterSpatialCropd(keys=io_keys, roi_size=self.patch_size),
            )

        elif self.patch_size is not None:
            transforms.append(SpatialPadd(keys=io_keys, spatial_size=self.patch_size))
            if self.min_foreground > 0:
                transforms.append(
                    RandSpatialCropForegroundd(
                        keys=io_keys,
                        spatial_size=self.patch_size,
                        label_key="label",
                        min_foreground=self.min_foreground,
                    )
                )
            else:
                transforms.append(
                    RandSpatialCropd(keys=io_keys, roi_size=self.patch_size, random_size=False),
                )
            zoom_xforms = self._resolution_zoom_transforms(sd)
            if zoom_xforms:
                transforms.append(Labeld(keys=["label"], spatial_dims=sd))
                transforms.extend(zoom_xforms)

        elif self.image_size is not None:
            transforms.append(
                Resized(keys=io_keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
            )

        transforms.extend([
            *self._original_transforms(sd),
            *self._instance_transforms(sd),
            *self._semantic_transforms(sd),
            *self._geometry_transforms(sd),
            EnsureTyped(keys=self._output_keys()),
        ])

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        io_keys = ["image", "label"]
        sd = self._get_spatial_dims()

        transforms: list = [
            EnsureChannelFirstd(keys=io_keys, channel_dim="no_channel"),
        ]

        if self.find_boundaries > 0:
            transforms.append(
                FindBoundariesd(
                    keys=["label"],
                    prob=1.0,
                    pixel_size=self.pixel_size,
                ),
            )

        if self.patch_size is not None:
            transforms.extend([
                SpatialPadd(keys=io_keys, spatial_size=self.patch_size),
                CenterSpatialCropd(keys=io_keys, roi_size=self.patch_size),
            ])
        elif self.image_size is not None:
            transforms.append(
                Resized(keys=io_keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
            )

        transforms.extend([
            *self._original_transforms(sd),
            *self._semantic_transforms(sd),
            *self._instance_transforms(sd),
            *self._geometry_transforms(sd),
            EnsureTyped(keys=self._output_keys()),
        ])
        return Compose(transforms)

    # ------------------------------------------------------------------
    # Dataset / DataLoader wiring
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        extra = self._get_dataset_kwargs()

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.train_volumes,
                cache_rate=self.cache_rate,
                transform=self.get_train_transforms(),
                **extra,
            )
            self.val_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.val_volumes,
                cache_rate=1.0,
                transform=self.get_val_transforms(),
                **extra,
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.test_volumes,
                cache_rate=0.0,
                transform=self.get_val_transforms(),
                **extra,
            )

    # ------------------------------------------------------------------
    # Lazy 3-D patch mode (shared by SNEMI3D / MICRONS / Neurons leaves)
    # ------------------------------------------------------------------

    def _build_lazy_split(
        self,
        volumes: Optional[List[Dict[str, str]]],
        patch_size: Optional[Tuple[int, ...]],
        transform,
        num_samples: int,
    ):
        """Build a single :class:`LazyVolDataset` split.

        Returns ``None`` when ``volumes`` is empty or ``patch_size`` is
        ``None`` (the caller is then responsible for deciding whether
        that's an error or a no-op for this split).

        All three lazy-mode datamodules (:class:`SNEMI3DDataModule`,
        :class:`MICRONSDataModule`, :class:`NeuronsDataModule`) used to
        carry an identical ~30-line block per split; this helper is the
        single shared implementation.
        """
        if not volumes or patch_size is None:
            return None
        from brainbow.datasets.lazy import LazyVolDataset

        return LazyVolDataset(
            root_dir=self.data_root,
            volumes=volumes,
            patch_size=patch_size,
            transform=transform,
            num_samples=num_samples,
            min_foreground=self.min_foreground,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()
