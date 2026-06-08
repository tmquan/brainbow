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
    ├── inference/        # sliding-window inference + Mutex Watershed.
    ├── losses/           # AffinityFGLoss (affinity + sem + raw) + helpers.
    ├── metrics/          # foreground + instance evaluation metrics.
    ├── models/           # model wrappers (BaseModel + per-arch packages).
    ├── modules/          # Lightning modules (BaseCircuitModule + per-arch).
    ├── preprocessors/    # format converters (base + per-format).
    ├── transforms/       # deterministic ops (boundaries, EDT, relabel, ...).
    ├── utils/            # io helpers.
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
__init__.py              # re-exports Vista3DWrapper, VistaTaskHead3D
wrapper.py               # Vista3DWrapper (the public class; affinity head)
heads.py                 # VistaTaskHead3D (MONAI UnetrBasicBlock)
hf_loader.py             # MONAI/VISTA3D-HF encoder download + partial-load
```

### `callbacks/tensorboard/`

```
__init__.py      # re-exports ImageLogger
tags.py          # TagContext: {stage}/{mode}/{panel}
heads.py         # panel logger (`_log_predictions`): true/{image,label,
                 # aff/*}, pred/{sem,raw,aff/*}, and the Mutex Watershed
                 # pred/label/{pre,mul} instance panels
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

## 5. Affinity + sem + raw loss

The loss package has one public loss, `AffinityFGLoss` (`affinity.py`),
the shared `DiceBCEFocalLoss` supervisor, and the layout/helper module
`_common.py`.  Instance segmentation at eval is produced by the Mutex
Watershed (`inference/mutex_watershed.py`) -- see
[`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md) for the head, loss, and eval
in depth.

`_common.py` owns the channel layout:

| Field | Slice | Channels | Head output |
| ----- | ----- | -------- | ----------- |
| `aff` | `[0, N_AFF)` | `N_AFF` (14) | logit |
| `sem` | `[N_AFF, N_AFF+1)` | 1 | logit |
| `raw` | `[N_AFF+1, N_AFF+2)` | 1 | linear (target in `[-1, 1]`) |

Activation policy: the head emits **raw logits / linear values** with no
activation in `forward`.  The loss supervises `aff` / `sem` with
logit-stable BCE (plus sigmoid for the Dice / focal terms), and every
other consumer (metrics, Mutex Watershed, TensorBoard) applies `sigmoid`
at its own boundary.  The `raw` channel is linear and reconstructs the
normalised EM in `[-1, 1]`.

`_common.py` also owns `AFFINITY_OFFSETS` / `N_PULL` (3 pull
nearest-neighbour + 11 push long-range offsets), the
`affinity_target_from_offsets` / `affinity_validity_mask` builders,
`slice_head`, and the fp32-clamped `stable_bce_on_probs`.

`AffinityFGLoss` consumes the head tensor directly:

```python
out = criterion(head, {"labels": labels, "raw_image": image})
# out -> {"loss", "loss/aff", "loss/sem", "loss/raw"}
```

| Scalar group | Meaning |
| ------------ | ------- |
| `loss/aff` | masked + offset-weighted (pull/push) affinity composite (BCE + soft-Dice + focal) |
| `loss/sem` | foreground (semantic) `DiceBCEFocalLoss` vs `labels > 0` |
| `loss/raw` | L1 reconstruction of the input EM intensity |

---

## 6. Tag hierarchy (scalars ↔ image panels)

Scalars and image panels share a `{stage}/{mode}/...` hierarchy so each
field's loss sits next to its visualisation in TensorBoard:

```
loss                          # global total
loss/{aff,sem,raw}            # per-field totals
{sem,ins}/metric/<name>       # eval metrics
```

| image tag (`heads.py`)                 | scalar tag(s)                         |
| -------------------------------------- | ------------------------------------- |
| `pred/aff/{offset}`, `true/aff/{offset}` | `loss/aff`                          |
| `pred/sem`                             | `loss/sem`, `sem/metric/{acc,iou,dice}` |
| `pred/raw`                             | `loss/raw`                            |
| `pred/label/{pre,mul}` (Mutex Watershed) | `ins/metric/{ari,ami,voi,ted}`      |

**Affinity tag ordering.**  Each affinity panel is named by its offset
(`brainbow.losses.AFF_NAMES`, e.g. `01_pull_z1`, `04_push_y3`) with a
1-based numeric prefix, so TensorBoard's alphabetical sort keeps the
panels in offset order.  A curated subset (or all `N_AFF`) is chosen by
`aff_panel_indices`.

**Visualisation-only mask.**  The `pred/aff` panels are multiplied by
the predicted `sem`, and `true/aff` by the GT foreground, before being
written to TB.  Display-only; the loss uses the unmasked tensors.

Task losses whose weight is `0.0` are **not instantiated** (not just
zeroed) so training is faster and memory is smaller.

---

## 7. Lightning module pattern

All modules in `brainbow.modules.*` inherit `BaseCircuitModule`, which
captures the entire training/eval loop:

1. forward the volume through the wrapper (`self.model`),
2. apply `AffinityFGLoss`,
3. accumulate foreground + Mutex Watershed instance metrics during validation/test,
4. all-reduce once per epoch and log under the scalar hierarchy.

Subclasses only declare:

```python
class MyModule(BaseCircuitModule):
    _model_cls = MyWrapper
    _loss_cls  = AffinityFGLoss
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
- `head`  ∈ `{"aff", "sem", "raw", "ins"}` or
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

    default.yaml  →  snemi3d.yaml  →  combine.yaml

- `default.yaml`: every knob with a sensible default.  Also the
  canonical home for **shared model / loss hyperparameters**
  (e.g. `model.head_channels`).
- `snemi3d.yaml`: SNEMI3D volume list + the bulk of the model / loss
  hyperparameters (batch size, augmentation mix, dense `loss:` block
  whose comments document every head and sub-weight, and the
  `resolution_zoom_*` knobs that harmonise resolutions across datasets
  once `combine.yaml` adds neurons / MICrONS).
- `combine.yaml`: inherits `snemi3d.yaml` and **replaces** its volume
  lists with a multi-dataset mix (SNEMI3D + neurons + MICrONS train,
  with SNEMI3D held out for val/test).  Drops AC4 from train so it can
  serve as the canonical SNEMI3D val volume.

**Convention:** a parameter lives in the *most general* config where
it's meaningful.  Things that don't depend on the dataset go in
`default.yaml`; dataset-scoped overrides go in the dataset config.
Per-experiment toggles (e.g. enabling only the boundary head) go on
the CLI as Hydra overrides.

Loss-weight blocks are densely commented (see `configs/snemi3d.yaml`
`loss:` block) so newcomers can learn the loss by reading the config.
Every head uses the **nested** loss schema (one mapping per head,
e.g. ``weight_sem: { weight: 1.0, ... }``) which keeps every
head-scoped knob next to its weight.  A bare scalar
(``weight_sem: 1.0``) is also accepted as shorthand for
``{weight: 1.0}`` with no sub-knobs; a nested mapping without
``weight:`` defaults to ``weight: 1.0``.  Set ``weight: 0`` to
disable a head -- the sub-loss module is then not instantiated and
the head's contribution is a cached zero scalar.

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

### ... task term or affinity offset

1. To add a supervised term, add a `weight_<name>` field + a
   `_loss_<name>` method to `AffinityFGLoss` (`losses/affinity.py`),
   emit `loss/<name>` in `forward`, and list it in
   `canonical_loss_keys()` (so the eval reducer pre-seeds it).
2. To change the affinity edge set, edit `AFFINITY_OFFSETS` / `N_PULL`
   in `losses/_common.py` — `HEAD_CHANNELS`, the target builders, and the
   Mutex Watershed all re-derive from it; bump `model.head_channels`.
3. Tests in `tests/test_losses.py` (shape / gradients / edge cases).

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
