"""
Base PyTorch Lightning DataModule for connectomics datasets.

Why this file exists
--------------------
Every dataset in brainbow shares the same MONAI augmentation pipeline,
the same train/val/test DataLoader plumbing, and the same hooks for
loss-target precomputation (instance relabel after crop,
find-boundaries).  Keeping all of that in one place
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
* :meth:`_instance_transforms`, :meth:`_semantic_transforms` -- inject
  extra label-target transforms before the volume is handed to the loss.
* :meth:`_get_spatial_dims` -- 3 by default; override for 2-D datasets.
"""

import logging
import os
from abc import ABC
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    CenterSpatialCropd,
    CopyItemsd,
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
    RandMissingSliced,
    RandTransposeXYd,
    RandResolutionZoomd,
    RandSpatialCropForegroundd,
)

logger = logging.getLogger(__name__)


def _parse_cpulist(spec: str) -> set:
    """Parse a Linux ``cpulist`` string (e.g. ``"0-55,112-167"``) to a set."""
    cpus: set = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus


def bind_local_numa(log: bool = False) -> None:
    """Pin the calling process to the NUMA node local to its GPU.

    Best-effort, by ``LOCAL_RANK``: on a multi-socket node the GPUs are
    split across NUMA nodes in contiguous blocks (e.g. 8 GPUs / 2 nodes ->
    ranks 0-3 on node0, 4-7 on node1).  Binding each rank's process and its
    DataLoader workers to the local socket stops the higher-numbered ranks
    from fetching/decoding data across the socket boundary (which starves
    their GPU and serialises DDP).  Silently no-ops on a single NUMA node,
    non-Linux, cgroup-restricted, or any error -- never breaks training.
    """
    if not hasattr(os, "sched_setaffinity"):
        return
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        node_root = "/sys/devices/system/node"
        nodes = sorted(
            int(n[4:]) for n in os.listdir(node_root)
            if n.startswith("node") and n[4:].isdigit()
        )
        if len(nodes) < 2:
            return
        n_gpu = int(os.environ.get("LOCAL_WORLD_SIZE") or 0) or (
            torch.cuda.device_count() or 1
        )
        per_node = max(1, n_gpu // len(nodes))
        node_idx = min(local_rank // per_node, len(nodes) - 1)
        with open(f"{node_root}/node{nodes[node_idx]}/cpulist") as fh:
            cpus = _parse_cpulist(fh.read())
        cpus &= set(os.sched_getaffinity(0))   # respect any cgroup/slurm cap
        if cpus:
            os.sched_setaffinity(0, cpus)
            if log:
                logger.info(
                    "NUMA-bound LOCAL_RANK=%d -> node%d (%d CPUs)",
                    local_rank, nodes[node_idx], len(cpus),
                )
    except Exception as exc:  # best-effort; never break training
        if log:
            logger.warning("bind_local_numa skipped: %s", exc)


def _numa_worker_init(_worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: bind each worker to its rank's socket."""
    bind_local_numa(log=False)


class CircuitDataModule(pl.LightningDataModule, ABC):
    """
    Base PyTorch Lightning DataModule for connectomics datasets.

    Subclasses set ``dataset_class``, override ``_get_dataset_kwargs``,
    and optionally override the label-target hooks
    (``_instance_transforms``, ``_semantic_transforms``) or
    ``_get_spatial_dims`` for the appropriate dimensionality.

    Pipeline order (train)::

        EnsureChannelFirst → [FindBoundaries]
        → [Pad + Crop(safe_size)] → [ResolutionZoom] → [CenterCrop(patch_size)]
        → spatial augmentations (flip/rot90/elastic)
        → instance_transforms (CC-relabel) → intensity augmentations
        → EnsureType

    When resolution zoom can downsample (zoom < 1), an enlarged *safe*
    crop is taken first so the zoom's center-crop/pad never introduces
    zero-padded edges into the final patch.

    Args:
        data_root: Path to the data directory.
        batch_size: Batch size for training (and validation when
            ``val_batch_size`` is unset).
        val_batch_size: Batch size for validation / test loaders.
            Defaults to ``batch_size``; set lower when the eval-time
            memory peak exceeds the train step.
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
        val_batch_size: Optional[int] = None,
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
        find_boundaries: float = 0.0,
        boundary_target: str = "both",
        min_foreground: float = 0.0,
        pixel_size: Optional[Tuple[float, ...]] = None,
        elastic_prob: float = 0.0,
        elastic_sigma_range: Tuple[float, float] = (35.0, 50.0),
        elastic_magnitude_range: Tuple[float, float] = (10.0, 40.0),
        resolution_zoom_prob: float = 0.0,
        resolution_zoom_range: Optional[Tuple[Tuple[float, float], ...]] = None,
        resolution_zoom_mode: str = "ratio",
        resolution_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
        missing_slice_prob: float = 0.0,
        missing_slice_max: int = 2,
        missing_slice_fill: str = "zero",
        missing_slice_consecutive: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.batch_size = batch_size
        # Validation / test batch size.  Defaults to ``batch_size``.  Set
        # lower (e.g. 1) when the eval-time forward + full-resolution decode
        # + Mutex Watershed / metrics peak exceeds the train-step memory --
        # common on large backbones where validation is the memory-binding path.
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
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
        self.find_boundaries = float(find_boundaries)
        # ``both``    -> erode boundary voxels in the shared instance ``label``
        #                (affects sem AND affinity targets + val instance GT).
        # ``semantic`` -> erode a separate ``sem_label`` copy used only by the
        #                foreground (sem) head; the instance ``label`` (affinity
        #                targets + val instance GT / fg_mask) stays pristine.
        self.boundary_target = str(boundary_target)
        if self.boundary_target not in ("both", "semantic"):
            raise ValueError(
                "boundary_target must be 'both' or 'semantic'; "
                f"got {boundary_target!r}."
            )
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
        # ``ratio`` (legacy, anisotropy-preserving) or ``union`` (resample every
        # anisotropic volume onto the shared resolution envelope; isotropic
        # volumes are passed through).
        self.resolution_zoom_mode = str(resolution_zoom_mode)
        self.resolution_map = (
            {k: tuple(v) for k, v in resolution_map.items()}
            if resolution_map is not None
            else None
        )
        # Missing-section (missing z-slice) augmentation -- simulates ssTEM
        # acquisition defects (CREMI).  Train-only; image-only (labels keep
        # supervising the true connectivity through the corrupted section).
        self.missing_slice_prob = float(missing_slice_prob)
        self.missing_slice_max = int(missing_slice_max)
        self.missing_slice_fill = str(missing_slice_fill)
        self.missing_slice_consecutive = bool(missing_slice_consecutive)

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

        # Worst-case (smallest) native resolution per axis across all
        # datasets: a finer native voxel size means a larger downsample
        # (zoom < 1) toward the target range, hence a bigger safe margin.
        # Mixing e.g. CREMI (4 nm XY) with MICrONS (8 nm XY) means the
        # margin must follow the 4 nm dataset, not just ``pixel_size``.
        # In ``union`` mode, isotropic volumes (FIB-25 8x8x8) are NOT resampled
        # (the transform skips them), so they must be excluded from the
        # native-min -- otherwise their fine 8 nm z would force a ~5x z margin.
        native = list(self.pixel_size)
        if self.resolution_map:
            for res in self.resolution_map.values():
                if (
                    self.resolution_zoom_mode == "union"
                    and len(res) == 3
                    and res[0] == res[1] == res[2]
                ):
                    continue
                for d in range(min(len(native), len(res))):
                    native[d] = min(native[d], float(res[d]))

        safe = []
        needs_margin = False
        for d in range(len(self.patch_size)):
            min_zoom = native[d] / target_range[d][1]
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
            "mode": self.resolution_zoom_mode,
        }
        if target_range is not None:
            kwargs["target_range"] = target_range
        if self.resolution_map is not None:
            kwargs["resolution_map"] = self.resolution_map
        return [RandResolutionZoomd(**kwargs)]

    def _missing_section_transforms(self, spatial_dims: int) -> list:
        """CREMI-style missing-section augmentation (train-only, image-only).

        Blanks whole z-sections of the image to mimic ssTEM acquisition
        defects; the label is left intact so the affinity / sem targets keep
        supervising the true connectivity through the corrupted section.
        No-op in 2-D slice mode or when ``missing_slice_prob <= 0``.
        """
        if spatial_dims != 3 or self.missing_slice_prob <= 0:
            return []
        return [
            RandMissingSliced(
                keys=["image"],
                prob=self.missing_slice_prob,
                max_slices=self.missing_slice_max,
                fill=self.missing_slice_fill,
                consecutive=self.missing_slice_consecutive,
            )
        ]

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
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.3)),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.05),
        ]

    def _instance_transforms(self, spatial_dims: int) -> list:
        """Post-crop connected-component relabeling.

        Splits instances that became disconnected after cropping and
        renumbers labels sequentially.  Runs immediately after crop.
        """
        return [Labeld(keys=["label"], spatial_dims=spatial_dims)]

    # ------------------------------------------------------------------
    # Pipeline assembly
    # ------------------------------------------------------------------

    def _semantic_only_boundaries(self) -> bool:
        """True when boundary erosion targets only the ``sem`` head.

        In this mode the instance ``label`` is left intact (so the affinity
        targets and the validation instance GT / foreground mask are
        unaffected) and a separate eroded ``sem_label`` is produced for the
        foreground (sem) supervision.
        """
        return self.find_boundaries > 0 and self.boundary_target == "semantic"

    def _boundary_semantic_transforms(self, prob: float) -> list:
        """Build the eroded ``sem_label`` (semantic-only boundary mode).

        Runs last in the pipeline so it copies the final (cropped, relabeled)
        ``label`` and erodes the boundary on the actual training patch; the
        instance ``label`` itself is never modified.
        """
        if not self._semantic_only_boundaries():
            return []
        return [
            CopyItemsd(keys=["label"], times=1, names=["sem_label"]),
            FindBoundariesd(
                keys=["sem_label"],
                prob=prob,
                pixel_size=self.pixel_size,
            ),
        ]

    def _output_keys(self) -> list:
        """All keys that must pass through ``EnsureTyped``."""
        keys = ["image", "label"]
        if self._semantic_only_boundaries():
            keys.append("sem_label")
        return keys

    def get_train_transforms(self) -> Compose:
        io_keys = ["image", "label"]
        sd = self._get_spatial_dims()

        transforms: list = [
            EnsureChannelFirstd(keys=io_keys, channel_dim="no_channel"),
        ]

        if self.find_boundaries > 0 and self.boundary_target == "both":
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
            # Train-only: corrupt whole z-sections AFTER intensity aug so the
            # blanked sections are not re-normalised away.  Not added to the
            # validation pipeline.
            *self._missing_section_transforms(sd),
            *self._boundary_semantic_transforms(self.find_boundaries),
            EnsureTyped(keys=self._output_keys()),
        ])

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        io_keys = ["image", "label"]
        sd = self._get_spatial_dims()

        transforms: list = [
            EnsureChannelFirstd(keys=io_keys, channel_dim="no_channel"),
        ]

        if self.find_boundaries > 0 and self.boundary_target == "both":
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
            *self._boundary_semantic_transforms(1.0),
            EnsureTyped(keys=self._output_keys()),
        ])
        return Compose(transforms)

    # ------------------------------------------------------------------
    # Dataset / DataLoader wiring
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        # Pin this rank's process to the NUMA node local to its GPU (and the
        # DataLoader workers via ``_numa_worker_init`` below) so cross-socket
        # data fetch doesn't starve the higher-numbered ranks.
        bind_local_numa(log=True)
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

        Every lazy-mode datamodule (:class:`SNEMI3DDataModule`,
        :class:`MICRONSDataModule`, :class:`NeuronsDataModule`, and the
        :class:`CREMI3DDataModule` / :class:`FIB253DDataModule` subclasses)
        used to carry an identical ~30-line block per split; this helper is
        the single shared implementation.
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
            worker_init_fn=_numa_worker_init if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
            worker_init_fn=_numa_worker_init if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
            worker_init_fn=_numa_worker_init if self.num_workers > 0 else None,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()
