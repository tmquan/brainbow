# Brainbow — File Structure

What lives where.  For the *why* behind the layout (design patterns,
conventions, adding-new-X checklists), see
[`ORGANIZATION.md`](./ORGANIZATION.md).

---

## Top-level

```
brainbow/
├── LICENSE
├── README.md
├── pyproject.toml          # package metadata
├── requirements.txt        # pinned runtime dependencies
├── configs/                # Hydra YAMLs (§ Configuration)
├── data/                   # (gitignored) raw volumes
├── outputs/                # (gitignored) training artefacts
├── logs/                   # (gitignored) TensorBoard / wandb logs
├── doc/                    # this folder
├── scripts/                # CLI entrypoints (§ Scripts)
├── tests/                  # pytest suite (§ Tests)
└── brainbow/               # importable Python package (§ Package)
```

---

## Configuration

`configs/` is composed via Hydra's `defaults:` list.  The inheritance
chain is `default → <dataset> → <project>`.

| File                  | Purpose                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| `default.yaml`        | Every knob with a sensible default.  Base for every experiment.             |
| `snemi3d.yaml`        | SNEMI3D dataset overrides + shared **model/loss** hyperparameters.          |
| `combine.yaml`        | Multi-dataset training (SNEMI3D + neurons + MICrONS).                       |

---

## Scripts

CLI entry points.  Run with `python -m brainbow.<module>` where
applicable or directly: `python scripts/<name>.py`.

| Script                         | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| `scripts/train.py`             | Hydra-driven training loop (Lightning Trainer).   |
| `scripts/download_snemi3d.py`  | Fetch SNEMI3D volumes via cloudvolume.            |
| `scripts/download_microns.py`  | Fetch MICrONS volumes + segmentations.            |
| `scripts/download_zenodo_582636.py` | Generic Zenodo downloader, currently pointing at record 582636 (X-ray uCT of an assembly of rice grains, used as a 3D instance-segmentation benchmark with densely touching objects). |

---

## Tests

| File                               | Covers                                                     |
| ---------------------------------- | ---------------------------------------------------------- |
| `tests/test_losses.py`             | `CombinedLoss` 2-D and 3-D end-to-end (forward, gradients). |
| `tests/test_boundary_loss.py`      | `BoundaryLoss` + `build_boundary_target` (CPU / CUDA agree, channel layout, edge cases). |
| `tests/test_datasets.py`           | `CircuitDataset` abstract contract (resolution, anisotropy, length virtualisation). |
| `tests/test_datamodules.py`        | `CircuitDataModule` augmentation pipeline (via a synthetic in-memory dataset). |
| `tests/test_preprocessors.py`      | HDF5 / NRRD / TIFF / NfTy converters.                      |
| `tests/test_utils.py`              | label / io / parallel helpers.                             |

Tests for `SemanticLoss`, `InstanceLoss`, `GeometryLoss` (each in
isolation), the freeze schedule, `build_clusterer`, and
`sliding_window_inference` are added in Phase 5 of the audit overhaul
(see [`doc/CONTRIBUTING.md`](./CONTRIBUTING.md) once it lands).

---

## Package: `brainbow/`

Top-level `brainbow/__init__.py` re-exports the most common symbols
(wrappers, Lightning modules, losses, datamodules).

Subpackages, in the order a new contributor would typically explore:

### `brainbow/transforms/` — deterministic ops

Pure functions and MONAI `Transformd` wrappers used by the datamodules
and the loss targets.  No learnable state.

| File                       | Purpose                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `direction.py`             | Per-voxel unit vectors pointing at the instance centroid.              |
| `covariance.py`            | Per-voxel upper-triangle covariance field for each instance.           |
| `edt.py`                   | Exact Euclidean distance transform (scipy-backed).                     |
| `find_boundaries.py`       | Connectivity-1 inner/outer boundary masks (cucim / skimage / torch).   |
| `label.py`                 | Relabel / remap / consolidate instance ids.                            |
| `defect.py`                | Simulated acquisition defects (scratches, fold-overs).                 |
| `missing_section.py`       | Drop random z-slices to simulate missing sections.                     |
| `rand_crop_foreground.py`  | Random crop biased toward foreground voxels.                           |
| `rand_transpose_xy.py`     | Random xy-transpose augmentation.                                      |
| `resolution_zoom.py`       | Per-axis resolution scaling for multi-resolution training.             |

### `brainbow/datasets/` — MONAI `CacheDataset`s

| File            | Purpose                                                                         |
| --------------- | ------------------------------------------------------------------------------- |
| `base.py`       | `CircuitDataset` abstract base — declares `paper`, `resolution`, `labels`, etc. |
| `snemi3d.py`    | SNEMI3D dataset leaf.                                                           |
| `microns.py`    | MICrONS dataset leaf.                                                           |
| `neurons.py`    | Internal "neurons" volume leaf.                                                 |
| `lazy.py`       | `LazyVolDataset` — on-demand loading for very large volumes.                    |

### `brainbow/datamodules/` — Lightning `DataModule`s

| File            | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `base.py`       | `CircuitDataModule` — MONAI augmentation pipeline + split logic.    |
| `snemi3d.py`    | SNEMI3D datamodule leaf.                                            |
| `microns.py`    | MICrONS datamodule leaf.                                            |
| `neurons.py`    | Internal neurons datamodule leaf.                                   |

### `brainbow/losses/` — task losses + combiner

All task losses follow a uniform skeleton (see `ORGANIZATION.md` §5).

| File            | Purpose                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| `_common.py`    | Shared regression-loss name resolution (`l1` / `mse` / `smooth_l1` + aliases). |
| `semantic.py`   | `SemanticLoss` — CE + IoU + Dice (sigmoid or softmax mode).                    |
| `instance.py`   | `InstanceLoss` — pull / push / norm discriminative embedding loss.             |
| `geometry.py`   | `GeometryLoss` — raw(1) + cov upper-tri(S·(S+1)/2) + dir(S) regression.        |
| `boundary.py`   | `BoundaryLoss` — 16-ch target: raw(1) + min/avg/max RGB(9) + face-affinity(6, soft-Dice on sigmoid). |
| `combined.py`   | `CombinedLoss` — weighted sum with head-oriented output key hierarchy.         |

### `brainbow/metrics/` — per-head eval metrics

| File            | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `semantic.py`   | Per-class IoU, Dice, pixel accuracy.                                |
| `instance.py`   | Adapted Rand Error, Variation of Information, optimal split/merge. |

### `brainbow/models/` — backbone wrappers

`models/base.py::BaseModel` is the abstract contract (forward →
dict with `logits`, `get_output_channels()`).

#### `models/cosmos_transfer_2_5/` — Cosmos-Transfer 2.5 3-D wrapper

Split into a package because the wrapper needs HF auto-pull, a
standalone-DiT fallback, and a variant registry.

| File                  | Purpose                                                               |
| --------------------- | --------------------------------------------------------------------- |
| `__init__.py`         | Re-exports `CosmosTransfer3DWrapper`.                                 |
| `wrapper.py`          | `CosmosTransfer3DWrapper` — the public class.                         |
| `variants.py`         | 2B / 14B variant registry (+ sanity for unreleased variants).         |
| `layers.py`           | Shared primitives (conv blocks, normalisation wrappers).              |
| `decoder.py`          | Feature projector + VAE decoder adapter.                              |
| `standalone_dit.py`   | Random-init DiT fallback for `pretrained=False`.                      |
| `hf_loader.py`        | Rank-aware HF snapshot download (ignores `text_encoder/*`).           |

#### `models/vista/` — Vista3D wrapper + head + prompt I/O

| File                       | Purpose                                                             |
| -------------------------- | ------------------------------------------------------------------- |
| `__init__.py`              | Re-exports `Vista3DWrapper`, `VistaTaskHead3D`, prompt utils.       |
| `wrapper.py`               | `Vista3DWrapper` — the public class.                                |
| `heads.py`                 | `VistaTaskHead3D` (MONAI `UnetrBasicBlock`).                        |
| `hf_loader.py`             | MONAI `VISTA3D-HF` encoder download + partial-load.                 |
| `point_prompt_encoder.py`  | `PointPromptEncoder` (proofread / interactive conditioning).        |
| `point_sampling.py`        | `sample_point_prompts` — GT mask → click-point dict.                |

### `brainbow/modules/` — Lightning modules

`modules/base.py::BaseCircuitModule` captures the full training /
validation / test loop shared by every architecture.  Each arch gets
its own package with a freeze-/optim-aware `base.py` and a
concrete `module.py`.

| Path                                  | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `modules/base.py`                     | `BaseCircuitModule` — loop + head-oriented scalar logging.   |
| `modules/cosmos_transfer_2_5/base.py` | `BaseCosmosModule` — freeze schedule + optim parameter split.|
| `modules/cosmos_transfer_2_5/module.py` | `CosmosTransfer3DModule` — the concrete Lightning class.   |
| `modules/vista/base.py`               | `BaseVistaModule` — Vista-specific freeze schedule.          |
| `modules/vista/module.py`             | `Vista3DModule` — the concrete Lightning class.              |

### `brainbow/callbacks/` — Lightning callbacks

| Path                              | Purpose                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| `callbacks/memory.py`             | Per-epoch GPU/CPU memory logger.                                        |
| `callbacks/tensorboard/`          | `ImageLogger` — hierarchical TB visualisation (package).                |
| `callbacks/tensorboard/image_logger.py` | `ImageLogger` callback (the public class).                        |
| `callbacks/tensorboard/tags.py`   | `TagContext` — single source of `{stage}/{mode}/[{head}/]{panel}`.      |
| `callbacks/tensorboard/heads.py`  | `_log_semantic` / `_log_instance` / `_log_geometry` / `_log_boundary`.  |
| `callbacks/tensorboard/geometry.py` | Direction-quiver + covariance-glyph rendering helpers.                |
| `callbacks/tensorboard/viz.py`    | Colour-map, overlay, tile builders.                                     |

### `brainbow/inference/` — sliding-window + clustering

| File                    | Purpose                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| `sliding_window.py`     | Blended sliding-window inference over arbitrarily large volumes.    |
| `clusterer.py`          | Discriminative-embedding → instance-id clustering (meanshift / BFS).|

### `brainbow/preprocessors/` — format converters

`preprocessors/base.py::BasePreprocessor` declares the `save` / `load`
/ `validate` / `get_shape` / `get_metadata` interface.

| File         | Purpose                                     |
| ------------ | ------------------------------------------- |
| `hdf5.py`    | HDF5 preprocessor (primary format).         |
| `nrrd.py`    | NRRD preprocessor (medical imaging format). |
| `tiff.py`    | Multi-page TIFF preprocessor.               |
| `nfty.py`    | NfTy / neurofitty volumetric format.        |

### `brainbow/utils/` — miscellaneous helpers

| File            | Purpose                                                 |
| --------------- | ------------------------------------------------------- |
| `io.py`         | Volume read/write + norm-stat caching.                  |
| `parallel.py`   | `pmap` — forkserver-based parallel map for CPU work.    |
| `clustering.py` | Scalar post-processing for inference (CC, watershed).   |
| `manifold.py`   | UMAP / t-SNE helpers for embedding-space diagnostics.   |

### `brainbow/visualizer/` — interactive web volume renderer

| Path                      | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `app.py`                  | FastAPI server exposing volume tiles.              |
| `__main__.py`             | `python -m brainbow.visualizer` entrypoint.        |
| `volume_loader.py`        | Lazy chunked HDF5 loader for the server.           |
| `static/index.html`       | Single-page UI.                                    |
| `static/app.js`           | UI wiring + camera controls.                       |
| `static/volume_renderer.js` | WebGL 3-D volume raymarcher.                     |
| `static/style.css`        | Dark-mode layout.                                  |

---

## File count per subsystem (informational)

| Subsystem                               | .py files  |
| --------------------------------------- | ---------- |
| `brainbow/transforms/`                  | 11         |
| `brainbow/models/cosmos_transfer_2_5/`  |  7         |
| `brainbow/models/vista/`                |  6         |
| `brainbow/callbacks/tensorboard/`       |  5 (+ pkg) |
| `brainbow/losses/`                      |  7         |
| `brainbow/datamodules/` + `datasets/`   |  5 + 6     |
| `brainbow/preprocessors/`               |  5         |
| `brainbow/modules/`                     |  7         |
| `brainbow/metrics/`                     |  3         |
| `brainbow/inference/`                   |  3         |
| `brainbow/utils/`                       |  5         |
| `brainbow/visualizer/`                  |  4 py + 4 static |

---

## See also

- [`ORGANIZATION.md`](./ORGANIZATION.md) — design patterns, conventions, and
  "how to add a new …" checklists.
- `configs/*.yaml` — every knob is documented inline.
- `brainbow/losses/combined.py` — the canonical consumer of every
  task-loss's head-oriented output dict.
