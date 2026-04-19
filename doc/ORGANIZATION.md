# Brainbow — Code Organization & Design Patterns

This document describes **how** the Brainbow codebase is organized and **why**
— the recurring patterns that every new file should follow.  For a plain
file-by-file tree, see [`STRUCTURE.md`](./STRUCTURE.md).

---

## 1. High-level philosophy

- **Minimalist.**  Every subpackage exposes the smallest public API that
  still gets the job done; everything else is a private helper.
- **Base + concrete.**  Subsystems that have multiple variants (datasets,
  datamodules, models, modules, preprocessors, losses) expose an abstract
  `base.py` and one file per concrete implementation.  The base captures
  the shared logic once; the leaf file only declares what's actually
  different.
- **Package-per-thing when it grows.**  Single-file modules that exceed
  ~300 LOC are decomposed into a package with an `__init__.py` that
  re-exports the public API.  Examples:
  `models/cosmos_transfer_2_5/`, `models/vista/`, `modules/*/`,
  `callbacks/tensorboard/`.
- **Hydra-first configuration.**  Nothing is hard-coded; every
  behaviour-changing parameter is a key in `configs/*.yaml` with
  documented defaults in `configs/default.yaml`.
- **einops-first tensor reshaping.**  `rearrange` / `reduce` / `repeat`
  are preferred over `.view` / `.permute` / `.reshape` / `.sum(dim=...)`.
  Any reshape that isn't a plain `squeeze`/`unsqueeze` should be an
  einops call.
- **Deterministic target construction.**  Losses and datamodules build
  supervision targets from first principles (no learnable params, no
  global state) so the pipeline is fully reproducible.

---

## 2. Directory layout

```
brainbow/
├── configs/              # Hydra YAML: default + per-dataset + combine.
├── data/                 # (gitignored) raw volumes.
├── doc/                  # STRUCTURE.md (tree), ORGANIZATION.md (this file).
├── scripts/              # CLI entrypoints (train.py, download_*.py).
├── tests/                # pytest: one test file per subsystem.
└── brainbow/             # importable package.
    ├── callbacks/        # TensorBoard + memory callbacks.
    ├── datamodules/      # Lightning DataModules (base + per-dataset).
    ├── datasets/         # MONAI CacheDatasets (base + per-dataset + lazy).
    ├── inference/        # sliding-window inference + instance clustering.
    ├── losses/           # SemanticLoss / InstanceLoss / GeometryLoss /
    │                     # BrainbowLoss / CombinedLoss.
    ├── metrics/          # per-head evaluation metrics.
    ├── models/           # model wrappers (BaseModel + per-arch packages).
    ├── modules/          # Lightning modules (BaseCircuitModule + per-arch).
    ├── preprocessors/    # format converters (base + per-format).
    ├── transforms/       # deterministic ops (direction, covariance, EDT, ...).
    ├── utils/            # io, parallel, clustering, manifold.
    └── visualizer/       # web volume renderer.
```

Rule of thumb: if you need to pick where a new file goes, answer
*"what is its input/output contract?"* and put it in the subpackage
whose base class matches.

---

## 3. The base-and-concrete pattern

Five subsystems instantiate this pattern.  Each has a single shared
`base.py` and one leaf file per implementation:

| Subsystem     | Base                                       | Concrete examples                                 |
| ------------- | ------------------------------------------ | ------------------------------------------------- |
| datasets      | `datasets/base.py::CircuitDataset`         | `snemi3d.py`, `microns.py`, `neurons.py`, `lazy.py` |
| datamodules   | `datamodules/base.py::CircuitDataModule`   | `snemi3d.py`, `microns.py`, `neurons.py`          |
| models        | `models/base.py::BaseModel`                | `cosmos_transfer_2_5/`, `vista/`                  |
| modules       | `modules/base.py::BaseCircuitModule`       | `cosmos_transfer_2_5/`, `vista/`                  |
| preprocessors | `preprocessors/base.py::BasePreprocessor`  | `hdf5.py`, `nrrd.py`, `tiff.py`, `nfty.py`        |

**Convention:** the concrete class overrides only:

1. class-level attributes that declare what's different
   (e.g. `_model_cls`, `_loss_cls` for modules; `paper`, `resolution`,
   `labels` for datasets);
2. methods that *genuinely* diverge from the base (e.g. custom
   `configure_optimizers` with a parameter-group split).

Everything else — the training loop, the logging hierarchy, the
augmentation pipeline, the metric aggregation — lives in the base so
that new variants cost ~50 lines.

---

## 4. Package-per-component

When a wrapper outgrows a single file it becomes a package.  The
`__init__.py` is the **sole** public surface; everything else is treated
as private.  Two fully-realized examples:

### `models/cosmos_transfer_2_5/`

```
__init__.py          # re-exports CosmosTransfer3DWrapper
hf_loader.py         # rank-aware HF snapshot download
variants.py          # 2B / 14B variant registry
standalone_dit.py    # random-init DiT fallback
layers.py            # shared primitives
decoder.py           # feature projector + VAE decoder adapter
wrapper.py           # CosmosTransfer3DWrapper (the public class)
```

### `models/vista/`

```
__init__.py              # re-exports Vista3DWrapper, VistaTaskHead3D,
                         # PointPromptEncoder, sample_point_prompts
wrapper.py               # Vista3DWrapper (the public class)
heads.py                 # VistaTaskHead3D (MONAI UnetrBasicBlock)
hf_loader.py             # MONAI/VISTA3D-HF encoder download + partial-load
point_prompt_encoder.py  # interactive proofread conditioning
point_sampling.py        # GT-mask -> click-point dict
```

### `callbacks/tensorboard/`

```
__init__.py      # re-exports ImageLogger
tags.py          # TagContext: {stage}/{mode}/[{head}/]{panel}
geometry.py      # geometry-head visual helpers
heads.py         # _log_semantic / _log_instance / _log_geometry / _log_brainbow
viz.py           # colour-map, overlay, tile builders
image_logger.py  # ImageLogger callback (the public class)
```

**Rules:**

- **No deep imports.**  Downstream code imports from the package root
  (`from brainbow.models.vista import Vista3DWrapper`), never from a
  sibling file.
- **`__init__.py` stays thin.**  It re-exports; it does not execute
  substantial logic.
- **Private modules carry a leading-topic naming scheme** (`layers`,
  `heads`, `hf_loader`) — never `utils.py` inside a package.

---

## 5. Unified loss skeleton

Every task loss in `losses/` follows the same template:

```python
class <Task>Loss(nn.Module):
    def __init__(...)                        # weights, config, sub-modules

    @property
    def task_channels                        # expected prediction width

    # --- Target construction -----------------------------------------
    @torch.no_grad() def _build_target_<component>(...)
    @torch.no_grad() def build_target(...)   # public dispatcher

    # --- Per-voxel weights (optional) --------------------------------
    def compute_weights(labels)              # None by default

    # --- Sub-losses --------------------------------------------------
    def _compute_loss_<component>(...)       # scalar per component

    # --- Forward -----------------------------------------------------
    def forward(...) -> Dict[str, Tensor]    # {"loss", <component>, ...}
    def __repr__()
```

| Loss          | Components (keys in returned dict)               | `task_channels` |
| ------------- | ------------------------------------------------ | --------------- |
| SemanticLoss  | `loss`, `ce`, `iou`, `dice`                      | `num_classes`   |
| InstanceLoss  | `loss`, `pull`, `push`, `norm`                   | embedding `E`   |
| GeometryLoss  | `loss`, `raw`, `cov`, `dir`                      | `1 + S*(S+1)//2 + S` |
| BrainbowLoss  | `loss`, `raw`, `min`, `avg`, `max`, `aff`        | `16`            |

**Why this matters:**

- **Symmetry.**  Given any loss, a reader already knows which methods
  to open and what each one returns.
- **Pluggability.**  `CombinedLoss` treats each task loss identically —
  it calls `compute_weights` once, then `forward`, then scatters the
  returned dict under `{head}/loss/{component}`.
- **Target caching.**  `build_target(...)` lets the datamodule or the
  image-logger precompute the supervision target and feed it back to
  the loss via `cached_target=...`, avoiding double-construction in DDP.

Shared regression-loss name resolution lives in `losses/_common.py`
(``canonical_regression_name``, ``regression_loss_fn``).  Individual
losses use it instead of each duplicating an `_LOSS_FN` dict.

---

## 6. CombinedLoss: head-oriented key hierarchy

`CombinedLoss` composes the four task losses into a single
dict-returning module.  Every scalar it emits uses the head-oriented
tag hierarchy that matches the TensorBoard image tags emitted by
`callbacks.tensorboard.ImageLogger`:

```
loss                              # global total
{head}/loss                       # per-head total
{head}/loss/{component}           # per-head loss breakdown
eff_w/{head}                      # effective task weights (learned mode)
```

This way, when TensorBoard alphabetically sorts tags, each head's
scalars cluster next to its images — e.g. `instance/loss/pull` sits
beside `train/automatic/instance/embed`.

Task losses whose weight is `0.0` are **not instantiated** (not just
zeroed) so training is faster and memory is smaller.

---

## 7. Lightning module pattern

All modules in `brainbow.modules.*` inherit `BaseCircuitModule`, which
captures the entire training/eval loop:

1. forward the volume through the wrapper (`self.model`),
2. apply `CombinedLoss`,
3. accumulate per-head metrics during validation/test,
4. all-reduce once per epoch and log under the scalar hierarchy.

Subclasses only declare:

```python
class MyModule(BaseCircuitModule):
    _model_cls = MyWrapper
    _loss_cls  = CombinedLoss
    # Optional: override configure_optimizers, freeze schedule hooks.
```

The per-architecture package (`modules/cosmos_transfer_2_5/`,
`modules/vista/`) holds its own `base.py` for arch-specific concerns
(parameter-group split for HF-pretrained backbones, freeze scheduling)
and a `module.py` for the concrete Lightning class.

---

## 8. Hierarchical TensorBoard tags

A single `TagContext` dataclass in
`callbacks/tensorboard/tags.py` enforces the layout::

    {stage}/{mode}/[{head}/]{panel}

where

- `stage` ∈ `{"train", "val", "test"}`,
- `mode`  ∈ `{"automatic", "prompted", ...}` (single-value today,
  structured so `prompted` can slot in later),
- `head`  ∈ `{"semantic", "instance", "geometry", "brainbow"}` or
  `None` for mode-level panels,
- `panel` is the concrete image / scalar name.

Every image logged in `heads.py` and every scalar logged in
`modules/base.py` is routed through `TagContext.tag(panel)`.  This is
the **only** place tag strings are assembled.

---

## 9. Hydra configuration layering

Configs compose via Hydra's `defaults:` list.  Each file's `defaults:`
pulls in one parent; the effective config is the parent's merged with
the child's overrides.  The real chain (parent → child) is::

    default.yaml  →  snemi3d.yaml  →  combine.yaml  →  brainbow.yaml

- `default.yaml`: every knob with a sensible default.  Also the
  canonical home for **shared model / loss hyperparameters**
  (e.g. `model.brainbow_channels`).
- `snemi3d.yaml`: SNEMI3D volume list + per-dataset training overrides
  (batch size, augmentation mix, dense `loss:` block whose comments
  document every head and sub-weight).
- `combine.yaml`: extends `snemi3d.yaml` with the full multi-dataset
  volume list (SNEMI3D + neurons + MICrONS) and the harmonising
  `resolution_zoom_*` knobs.
- `brainbow.yaml`: project-level entry point.  Inherits from
  `combine.yaml` and flips the loss weights so only the Brainbow head
  is active.

**Convention:** a parameter lives in the *most general* config where
it's meaningful.  Things that don't depend on the dataset go in
`default.yaml`; dataset-scoped overrides go in the dataset config;
project-scoped knobs (which heads to enable) go in `brainbow.yaml`.

Loss-weight blocks are densely commented (see `configs/snemi3d.yaml`
`loss:` block) so newcomers can learn the loss by reading the config.
Brainbow-head sub-weights are prefixed (`brainbow_weight_raw`,
`brainbow_weight_min|avg|max|aff`) to keep them disambiguated from
GeometryLoss's own `weight_raw` inside `CombinedLoss.__init__`.  The
``aff`` sub-loss (soft-Dice on sigmoid face-affinity logits for the 6
neighbours in Z-Y-X order T/B/U/D/L/R) is tuned via `brainbow_weight_aff` and
`brainbow_aff_eps`.

---

## 10. HuggingFace checkpoint auto-pull

Models that wrap third-party pretrained backbones follow one pattern:

- The wrapper takes a `pretrained: bool` flag (surfaced as a Hydra
  knob).  When `True`, it auto-pulls weights from the HF Hub on the
  first rank only; other ranks wait and then load from the local
  snapshot.
- The downloader lives in the model's own package
  (`<pkg>/hf_loader.py`).  It is rank-aware, retries on transient
  failures, and **ignores the text-encoder subtree** for models that
  feed null prompts (Cosmos).
- Partial loading is graceful: if some head shapes don't match (e.g.
  Vista output classes differ), the backbone still loads and the heads
  stay random-initialized with a warning.
- Variants that don't have released weights (e.g. Cosmos 14B) raise a
  clear error when `pretrained=True` — never a silent random-init
  fallback.

---

## 11. einops style

Every reshape should read like prose.  Examples from the refactored
losses:

```python
# Channel-first -> channel-last one-hot:
target = rearrange(F.one_hot(x, C).float(), "b ... c -> b c ...")

# Grouped mean across (min|avg|max) RGB triplets:
per_group = reduce(per_voxel, "b (g c) ... -> g", "mean", g=3)

# Pairwise centroid distance matrix:
diff = (
    rearrange(centers, "i e -> i 1 e")
    - rearrange(centers, "j e -> 1 j e")
)

# Broadcast a per-instance colour to every foreground voxel:
voxel_rgb = rearrange(rgb9[inverse], "m c -> c m")
```

Rules:

- Use `rearrange` for permutations/reshapes.
- Use `reduce` for `mean` / `sum` / `max` across a named axis.
- Use `repeat` for broadcasting when the axis pattern matters
  (`"m -> m c"` is clearer than `.unsqueeze(-1).expand(...)`).
- Use `einsum` for bilinear ops (matrix products, attention).
- Avoid raw `.view` / `.reshape` unless you are reshaping into a single
  unnamed axis (e.g. `x.flatten()`).

---

## 12. Testing conventions

- One `tests/test_<subsystem>.py` per subsystem.
- Each test module imports only its subsystem's public API.
- Fixtures live at module scope; no `conftest.py` magic.
- Loss and metric tests use tiny synthetic volumes so the whole suite
  finishes in under 30 seconds on CPU.
- New features come with both a positive test (correct output) and an
  edge case (empty-label, single-class, dimension mismatch).

---

## 13. Checklist for adding a new ...

### ... task loss

1. Create `losses/<name>.py` following the loss skeleton in §5.
2. Declare `task_channels`, `_build_target_*`, `compute_weights`
   (or default to `None`), `_compute_loss_*`, `forward`, `__repr__`.
3. Register in `losses/__init__.py` and wire into `CombinedLoss`
   with its own `weight_<name>` flag and result-dict section.
4. Add a `<head>/loss/{component}` block to `CombinedLoss.forward`'s
   output dict.
5. Tests in `tests/test_losses.py` (shape / gradients / edge cases).

### ... model architecture

1. If it fits in one file, add `models/<name>.py` inheriting `BaseModel`.
2. If it needs HF auto-pull or more than ~300 LOC, create
   `models/<name>/` as a package: `wrapper.py`, `heads.py`,
   `hf_loader.py`, `__init__.py`.
3. Add a matching `modules/<name>/` package with `base.py` and
   `module.py` inheriting `BaseCircuitModule`.
4. Surface `pretrained: bool` + any new knobs in `configs/default.yaml`.
5. Tag the module with its preferred logging hierarchy (see §8).

### ... dataset

1. Add `datasets/<name>.py` inheriting `CircuitDataset`.
2. Add `datamodules/<name>.py` inheriting `CircuitDataModule`.
3. Create `configs/<name>.yaml` listing the volumes + resolution.
4. Add a downloader to `scripts/download_<name>.py` if appropriate.

### ... transform

1. Add `transforms/<name>.py` as a plain function or a MONAI
   `Transformd` wrapper.
2. Re-export from `transforms/__init__.py` only if it's expected to
   appear in a datamodule's `Compose([...])`.

---

## 14. Non-goals

These are deliberately *not* in the codebase:

- A global plugin registry.  Composition is explicit via Hydra YAMLs.
- A multi-ghost inheritance tree.  We prefer one abstract base with
  leaf implementations; no intermediate mixins.
- Dynamic attribute discovery.  If a method should exist on every
  subclass, it's declared on the base (often `@abstractmethod`).
- Config-driven class instantiation beyond Hydra's `_target_`.

If you find yourself needing any of the above, prefer making the
existing structure more explicit over adding indirection.
