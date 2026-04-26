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
2. [Loss head](#2-add-a-new-loss-head)
3. [Model backbone](#3-add-a-new-model-backbone)
4. [Transform](#4-add-a-new-transform)
5. [Callback](#5-add-a-new-callback)
6. [Clustering algorithm](#6-add-a-new-clustering-algorithm)

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

## 2. Add a new loss head

Three places to touch:

```
brainbow/losses/<name>.py             # the loss itself
brainbow/losses/__init__.py           # re-export
brainbow/losses/combined.py           # add a weight + result-dict section
```

### 2.1 Loss skeleton

Every loss in `brainbow/losses/` follows the same template -- see
`brainbow/losses/_common.py` for the contract.  Copy
`brainbow/losses/semantic.py` if your loss is per-voxel categorical or
`brainbow/losses/geometry.py` if it's per-voxel regression.

```python
# brainbow/losses/myhead.py
import torch
import torch.nn as nn

class MyHeadLoss(nn.Module):
    def __init__(self, *, weight_a: float = 1.0, weight_b: float = 1.0):
        super().__init__()
        self.weight_a = weight_a
        self.weight_b = weight_b

    @property
    def task_channels(self) -> int:
        return 4   # whatever your head emits

    @torch.no_grad()
    def build_target(self, labels, image=None, cached_target=None):
        # Build the deterministic supervision target.
        ...

    def forward(self, prediction, labels, image=None, *, cached_target=None):
        target = cached_target if cached_target is not None else self.build_target(...)
        a = self._compute_a(prediction, target)
        b = self._compute_b(prediction, target)
        loss = self.weight_a * a + self.weight_b * b
        return {"loss": loss, "a": a, "b": b}

    def __repr__(self):
        return f"MyHeadLoss(weight_a={self.weight_a}, weight_b={self.weight_b})"
```

### 2.2 Re-export and combine

```python
# brainbow/losses/__init__.py
from brainbow.losses.myhead import MyHeadLoss
__all__ += ["MyHeadLoss"]
```

In `brainbow/losses/combined.py::CombinedLoss.__init__`:

```python
weight_myhead = loss_kwargs.pop("weight_myhead", 0.0)
if isinstance(weight_myhead, Mapping):
    self.myhead_loss = MyHeadLoss(**weight_myhead)
elif weight_myhead > 0:
    self.myhead_loss = MyHeadLoss()
else:
    self.myhead_loss = None
```

In `forward`, after the existing per-head sections::

```python
if self.myhead_loss is not None and "myhead" in predictions:
    out = self.myhead_loss(predictions["myhead"], labels, images)
    result["myhead/loss"] = out["loss"]
    for k, v in out.items():
        if k != "loss":
            result[f"myhead/loss/{k}"] = v
```

### 2.3 Surface in the model wrapper

If your head is per-voxel, add a `head_myhead = VistaTaskHead3D(...)`
to the model wrapper(s) that should support it (see Cosmos or Vista
wrapper for the existing examples) and emit it from `forward` under
the key `"myhead"`.

### 2.4 Surface in TensorBoard

Add a `_log_myhead` function to
`brainbow/callbacks/tensorboard/heads.py` and add `"myhead"` to
`HEADS` in `brainbow/callbacks/tensorboard/tags.py`.

### 2.5 Test

Drop a synthetic 8x32x32 test in `tests/test_losses.py`.  Verify
shapes, that the loss is non-negative, and that gradients flow.

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

### 3.1 The wrapper class

* Inherit from `torch.nn.Module` (or `BaseModel` if you want the type
  guarantees).
* `forward(x: Tensor) -> Dict[str, Tensor]` keyed by head name.
  Logits stay raw; activations are applied in the loss.
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
from brainbow.losses import CombinedLoss
from brainbow.models.myarch import MyArchWrapper
from brainbow.modules.myarch.base import BaseMyArchModule

class MyArchModule(BaseMyArchModule):
    _SPATIAL_DIMS = 3
    _model_cls = MyArchWrapper
    _loss_cls = CombinedLoss
```

### 3.3 Wire-in the dispatch

`scripts/train.py:get_module` (line 198):

```python
module_classes = {
    "vista3d": Vista3DModule,
    "cosmostransfer3d": CosmosTransfer3DModule,
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

## 6. Add a new clustering algorithm

```python
# brainbow/inference/clusterer.py
class MyClusterer(_BaseUnsupervisedClusterer):
    def _fit_predict_one(self, embeddings, labels=None):
        ...   # returns (labels, soft_assign, centers)
```

Register in `build_clusterer` (same file, ~line 723):

```python
if name == "myclusterer":
    return MyClusterer(**kwargs)
```

The metric path expects a `(labels, soft_assign, centers)` triple.
Returning ``None`` for `soft_assign` / `centers` is OK; raising or
returning a single tensor is **not** -- see
[`GOTCHAS.md` #15](./GOTCHAS.md).

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
  [`GOTCHAS.md` #6, #7, #11](./GOTCHAS.md) for examples of how this
  bites us.
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
| BoundaryLoss         | `tests/test_boundary_loss.py`      |
| Utils (io / parallel)| `tests/test_utils.py`              |
| Sliding window       | `tests/test_sliding_window.py` (new) |
| Clustering           | `tests/test_clustering.py` (new)   |
| Modules / Trainer    | `tests/test_modules.py` (new, Phase 5) |

If the test needs CUDA, gate it on `pytest.importorskip("torch.cuda")`
or `@pytest.mark.skipif(not torch.cuda.is_available(), reason=...)`.
