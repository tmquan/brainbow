# Brainbow — Contributor Recipes

> Audience: anyone adding a new dataset, loss head, model backbone,
> transform, or callback.  Every recipe shows the **smallest reasonable
> diff** and tells you which existing file to copy.

Companion docs:
[`STRUCTURE.md`](./STRUCTURE.md),
[`ORGANIZATION.md`](./ORGANIZATION.md),
[`WALKTHROUGH.md`](./WALKTHROUGH.md).

---

## How to add a new ...

1. [Dataset](#1-add-a-new-dataset)
2. [Head field / loss term](#2-add-a-new-head-field--loss-term)
3. [Model backbone](#3-add-a-new-model-backbone)
4. [Transform](#4-add-a-new-transform)
5. [Callback](#5-add-a-new-callback)
6. [Agglomeration / Mutex Watershed](#6-tune-the-mutex-watershed-agglomeration)

Every recipe sticks to these conventions:

* **Base + concrete.**  A new variant inherits from a `base.py` class
  and overrides only what genuinely differs.
* **Public API in `__init__.py`.**  Importers should never reach into
  sibling files (`from brainbow.<sub>.<file>` is fine; deeper is not).
* **Hydra-first.**  Every behaviour-changing knob lives in a YAML key,
  not as a Python default.
* **Tests live next to consumers.**  `tests/test_<subsystem>.py`.

---

## 1. Add a new dataset

You will end up touching **four** places:

```
brainbow/preprocessors/<format>.py    # only if your format isn't supported yet
brainbow/datasets/<name>.py           # CircuitDataset leaf
brainbow/datamodules/<name>.py        # CircuitDataModule leaf
configs/<name>.yaml                   # Hydra config
scripts/download_<name>.py            # optional, for reproducibility
```

### 1.1 Preprocessor (skip if your format already has one)

Inherit from `BasePreprocessor` and implement the three required
overrides:

```python
# brainbow/preprocessors/myformat.py
"""MyFormat preprocessor.  See BasePreprocessor for the contract."""

from typing import List
from brainbow.preprocessors.base import BasePreprocessor

class MyFormatPreprocessor(BasePreprocessor):
    @property
    def supported_extensions(self) -> List[str]:
        return [".myfmt"]

    def load(self, path: str):
        ...   # return a numpy array

    def validate(self, path: str) -> bool:
        return str(path).endswith(".myfmt")
```

Re-export from `brainbow/preprocessors/__init__.py` so the suffix-
dispatch in `brainbow/utils/io.py` picks it up.

### 1.2 Dataset leaf

Subclass `CircuitDataset` (`brainbow/datasets/base.py`).  Use the
existing `SNEMI3DDataset` (`brainbow/datasets/snemi3d.py`) as a
template -- it's the closest to a clean copy-paste.

```python
# brainbow/datasets/myset.py
from brainbow.datasets.base import CircuitDataset
from brainbow.preprocessors import HDF5Preprocessor

class MySetDataset(CircuitDataset):
    paper = "Author et al., Year"
    resolution = {"z": 30.0, "y": 4.0, "x": 4.0}     # nanometres
    labels = ["background", "membrane", "mito", ...]

    def _prepare_data(self):
        # Build the list of MONAI data dicts the CacheDataset will see.
        # Each dict needs at least {"image": <ndarray>, "label": <ndarray>}.
        ...
```

Register it in `brainbow/datasets/__init__.py`.

### 1.3 DataModule leaf

```python
# brainbow/datamodules/myset.py
from brainbow.datamodules.base import CircuitDataModule
from brainbow.datasets import MySetDataset

class MySetDataModule(CircuitDataModule):
    dataset_class = MySetDataset
```

That's the entire file in the typical case.  Override
`_get_dataset_kwargs` only if your dataset takes per-leaf kwargs.

### 1.4 Hydra config

Create `configs/myset.yaml` extending `default.yaml`:

```yaml
# configs/myset.yaml
defaults:
  - default
  - _self_

data:
  dataset: myset
  data_root: data/myset
  batch_size: 4
  patch_size: [16, 256, 256]
  train_volumes:
    - { vol: vol01.h5, seg: seg01.h5 }
  val_volumes:   ${data.train_volumes}
  test_volumes:  ${data.train_volumes}
```

### 1.5 Wire-in the dispatch

`scripts/train.py:get_datamodule` (line 169) has a hard-coded mapping
from dataset name to datamodule class.  Add your entry:

```python
datamodule_classes = {
    "snemi3d": SNEMI3DDataModule,
    "microns": MICRONSDataModule,
    "myset":   MySetDataModule,         # <-- new
}
```

### 1.6 Test

Add a fixture in `tests/test_datamodules.py`.  Use a synthetic in-
memory dataset (see how the SNEMI3D test does it) so the test runs
in <1 s.

---

## 2. Add a new head field / loss term

Most changes happen in two places:

```
brainbow/losses/_common.py            # channel slice / constants if the head layout changes
brainbow/losses/affinity.py           # loss weights, target build, scalar keys
```

### 2.1 Add a new field

1. Add `<FIELD>_SLICE` (and any constants) to `losses/_common.py`.  The
   head emits raw logits / linear values (no activation in `forward`); a
   probabilistic field is supervised with logit-stable BCE in the loss
   and sigmoided at each consumer (metrics / MWS / TensorBoard).
2. Bump `HEAD_CHANNELS` and `model.head_channels` in the configs.
3. Update `slice_head()` tests in `tests/test_losses.py`.
4. In `AffinityFGLoss.__init__`, add `weight_<field>` parsing; add a
   `_loss_<field>` method.
5. In `AffinityFGLoss.forward`, emit `loss/<field>` and list it in
   `canonical_loss_keys()` so the eval reducer pre-seeds it.

To instead change the **affinity edge set**, edit `AFFINITY_OFFSETS` /
`N_PULL` in `_common.py` — `HEAD_CHANNELS`, the target builders, and the
Mutex Watershed all re-derive from it.

### 2.2 TensorBoard

If the field has a useful visualisation, add it in
`brainbow/callbacks/tensorboard/heads.py::_log_predictions` under
`pred/<field>...`, and keep scalar tags parallel under
`loss/<field>...`.

### 2.3 Test

Drop a synthetic 3-D test in `tests/test_losses.py`.  Verify the field
slice shape, finite scalar(s), and gradient flow.

---

## 3. Add a new model backbone

The pattern is "package-per-thing once it grows past ~300 LOC":

```
brainbow/models/<arch>/__init__.py        # re-exports the wrapper class
brainbow/models/<arch>/wrapper.py         # the public class
brainbow/models/<arch>/heads.py           # task heads (often shared)
brainbow/models/<arch>/hf_loader.py       # optional HF auto-pull
```

And a matching Lightning module:

```
brainbow/modules/<arch>/__init__.py
brainbow/modules/<arch>/base.py           # arch-specific concerns
brainbow/modules/<arch>/module.py         # concrete Lightning class
```

> **If the new backbone is a thin variant of an existing one** (e.g.
> Cosmos-Predict 2.5 vs. Cosmos-Transfer 2.5, which share the base DiT
> + Wan VAE and only differ in whether a ControlNet residual branch is
> loaded on top), prefer factoring the shared scaffolding into a
> `cosmos_<family>_common/` package and inheriting from it.  See
> [`brainbow/models/cosmos_2_5_common/`](../brainbow/models/cosmos_2_5_common/)
> and [`brainbow/modules/cosmos_2_5_common/`](../brainbow/modules/cosmos_2_5_common/)
> for the canonical example: `_BaseCosmos25Wrapper` exposes
> `_init_arch_state` / `_post_load_diffusers` /
> `_compute_controlnet_residuals` extension hooks so each backbone-
> specific package only owns its true delta.

### 3.1 The wrapper class

* Inherit from `torch.nn.Module` (or `BaseModel` if you want the type
  guarantees).
* `forward(x: Tensor) -> Tensor` returning the
  `[B, HEAD_CHANNELS, *spatial]` tensor of **raw logits / linear values**
  (no activation in `forward`): `aff` / `sem` are logits and `raw` is
  linear.  Each consumer applies its own activation (logit-stable BCE in
  the loss; sigmoid for metrics / MWS / TensorBoard).
* If your backbone has frozen modules under DDP, follow Cosmos's
  approach: `requires_grad_(False)` + `.eval()` + `.detach()` on the
  output of the frozen subgraph (see `cosmos_transfer_2_5/wrapper.py`).

### 3.2 The Lightning module

```python
# brainbow/modules/myarch/base.py
from typing import Any, Dict
import torch
from brainbow.modules.base import BaseCircuitModule

class BaseMyArchModule(BaseCircuitModule):
    def _build_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        return self._model_cls(**model_config)
```

```python
# brainbow/modules/myarch/module.py
from brainbow.losses import AffinityFGLoss
from brainbow.models.myarch import MyArchWrapper
from brainbow.modules.myarch.base import BaseMyArchModule

class MyArchModule(BaseMyArchModule):
    _SPATIAL_DIMS = 3
    _model_cls = MyArchWrapper
    _loss_cls = AffinityFGLoss
```

### 3.3 Wire-in the dispatch

`scripts/train.py:build_module`:

```python
module_classes = {
    "vista3d": Vista3DModule,
    "cosmostransfer3d": CosmosTransfer3DModule,
    "cosmospredict3d": CosmosPredict3DModule,
    "myarch": MyArchModule,                    # <-- new
}
```

### 3.4 Defaults in `configs/default.yaml`

Surface the new knobs (`feature_size`, `pretrained`, freeze flags ...)
under `model:` with sensible defaults.

---

## 4. Add a new transform

Three lines, basically.

```python
# brainbow/transforms/myaug.py
"""Domain-specific MyAug transform."""

from typing import Dict
from monai.config import KeysCollection
from monai.transforms import MapTransform

class MyAugd(MapTransform):
    def __init__(self, keys: KeysCollection, **kwargs):
        super().__init__(keys)
        ...

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._transform(d[key])
        return d

    def _transform(self, x):
        ...
```

Register in `brainbow/transforms/__init__.py`.

If the transform is randomised, also subclass `Randomizable` and call
`self.R.uniform(...)` rather than `numpy.random.uniform(...)` so MONAI
seeds it correctly.

If your transform should run during training, plug it into the pipeline
in `brainbow/datamodules/base.py::CircuitDataModule.get_train_transforms`.

---

## 5. Add a new callback

```python
# brainbow/callbacks/mycallback.py
import pytorch_lightning as pl

class MyCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ...
```

Re-export from `brainbow/callbacks/__init__.py` and add an `if cfg.callbacks.mycallback.enabled` guard in `scripts/train.py:setup_callbacks`.

---

## 6. Tune the Mutex Watershed agglomeration

Instances come from the parameter-free Mutex Watershed
(`brainbow/inference/mutex_watershed.py`), not a learned clusterer, so
there's no algorithm to register — you tune it through
`training.mutex_watershed` in the config:

```yaml
training:
  mutex_watershed:
    strides: [1, 4, 4]   # per-axis subsample of push edges (Z, Y, X)
    size_filter: 0       # drop components < N voxels -> background
    backend: auto        # GPU mws_cp on CUDA, else CPU mws_np
    buckets: 16          # GPU Boruvka priority buckets
    max_push_edges: null # cap mutex edges; null = full edges
    # offsets / n_pull default to AFFINITY_OFFSETS / N_PULL
```

`MutexWatershed` returns `[B, *spatial]` long instance ids
(`0` = background), the drop-in contract for the eval metric path.  It
dispatches to the GPU cupy Boruvka (`mws_cp`, default on CUDA, zero-copy
via DLPack) or the exact numpy/numba `mws_np` (CPU fallback / reference);
see [`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md) §4.  To change the edge
set, edit `AFFINITY_OFFSETS` / `N_PULL` in `losses/_common.py` (§2).

---

## 7. Style guidelines

* **Use einops** (`rearrange` / `reduce` / `repeat`) instead of `view`
  / `permute` / `reshape` / `sum(dim=)` for any non-trivial reshape.
* **Type-hint public surfaces** (function signatures, class attributes
  exported from `__init__.py`).  Keep private helpers untyped if it
  makes the diff smaller.
* **No mutable defaults.**  Use `None` and assign in the body.
* **No silent `except Exception`.**  Either narrow the exception
  class or re-raise after logging.  See
  [`GOTCHAS.md` #7](./GOTCHAS.md) for an example of how this bites us.
* **Comments explain why, not what.**  If the code is doing something
  surprising, leave a one-line comment with a citation.

---

## 8. Where to put the test

| Subsystem            | Test file                          |
| -------------------- | ---------------------------------- |
| Datasets             | `tests/test_datasets.py`           |
| DataModules          | `tests/test_datamodules.py`        |
| Preprocessors        | `tests/test_preprocessors.py`      |
| Losses               | `tests/test_losses.py`             |
| Utils (io)           | `tests/test_utils.py`              |
| Sliding window       | `tests/test_sliding_window.py`     |
| Mutex Watershed      | `tests/test_mutex_watershed.py`    |
| Modules / Trainer    | `tests/test_modules.py`            |

If the test needs CUDA, gate it on `pytest.importorskip("torch.cuda")`
or `@pytest.mark.skipif(not torch.cuda.is_available(), reason=...)`.
