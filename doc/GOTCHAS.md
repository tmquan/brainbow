# Brainbow — Gotchas and Silent Failure Modes

> Audience: anyone debugging an unexpected result, looking for a
> "this can't be right" moment, or onboarding a new contributor.

This file collects the **non-obvious** behaviours that make brainbow
look like it's running correctly when it isn't.  Every entry follows
the same shape:

* What you'll see (the symptom).
* Where it happens (file:line).
* Why the code is written that way (intent).
* Recommended remediation if it bites you.

Companion docs: [`WALKTHROUGH.md`](./WALKTHROUGH.md),
[`ORGANIZATION.md`](./ORGANIZATION.md).

---

## 1. `torch.load` is monkey-patched to `weights_only=False`

**Symptom.** "How is Lightning loading my checkpoint without a
`safe_globals` warning when I have new objects in callback state?"

**Where.** `scripts/train.py:65-86`.  At import time, `torch.load` is
rebound to a wrapper that forces `weights_only=False`.

**Why.** PyTorch >= 2.6 made `weights_only=True` the default.  Lightning
checkpoints pickle non-tensor objects (`collections.defaultdict` for
metric / callback state, `OmegaConf` containers for hparams, custom
optimiser state).  `add_safe_globals` whitelists the *types*, but the
weights-only `SETITEM` opcode is hardcoded to accept only `dict` /
`OrderedDict` / `Counter` as the SETITEM target -- so resume still
fails with `defaultdict` state.  Our checkpoints are local, so the
script trusts them.

**Remediation.** Don't `torch.load(<untrusted file>)` from a process
that imports `scripts/train.py` (in a notebook, import the
function-level helper rather than the script module).  Phase 3a moves
this patch out of import time so the side effect is opt-in.

---

## 2. Loss-config schema is silently both flat and nested

**Symptom.** Two YAML files mix `weight_semantic: 0.5` with
`weight_semantic: { weight: 0.5, dice: 0.3, ce: 0.2 }` and both work,
but you can't tell which one is in effect from the config alone.

**Where.**
[`brainbow/losses/combined.py`](../brainbow/losses/combined.py),
in particular the legacy-key migration block in `__init__`.
`scripts/train.py:_head_weight_scalar` (line 30) and
`brainbow/modules/base.py:_head_weight` (line 55) implement the same
read on both shapes.

**Why.** The codebase shipped with the flat schema; the nested form
was added later.  Both are accepted for back-compat with old configs
and old checkpoints' hparams.

**Remediation.** Phase 4 migrates `default.yaml` and `boundary.yaml`
to the nested form so all configs match.  Phase 3b extracts the
legacy-key migration into `_migrate_legacy_keys` so the back-compat
logic is in one place.

---

## 3. Heads with `weight: 0.0` are not constructed at all

**Symptom.** You set `weight_geometry: 0.0`, then later try to read
`module.loss_fn.geometry_loss` and hit `AttributeError`.  Or you
inspect `module.model.head_geometry` and find it's missing.

**Where.** `brainbow/losses/combined.py::CombinedLoss.__init__` skips
`weight=0` heads;
`brainbow/modules/base.py:151-156` propagates the same set as
`disabled_heads` to the model wrapper, which skips constructing the
head module.

**Why.** Memory + speed.  An unused head still costs ~0.7 M params
and a forward pass.

**Remediation.** **Intentional.**  But make sure your tests /
inspection code does `getattr(..., None)` rather than direct attribute
access, and that any clusterer / image-logger panel keyed on a head
checks the prediction dict for the key first.

---

## 4. `sliding_window_inference` head set is auto-detected from the dummy pass

**Symptom.** You enable a fourth head mid-run and the inference output
silently grows / shrinks accordingly.

**Where.** `brainbow/inference/sliding_window.py::_detect_heads`.

**Why.** The function takes one dummy forward to learn which of
``semantic`` / ``instance`` / ``geometry`` / ``boundary`` the model
returns and allocates accumulators for exactly that set.  An unknown
key (i.e. one not in ``_KNOWN_HEADS``) is ignored, not aggregated.

**Remediation.** If you add a new head, add its name to ``_KNOWN_HEADS``
in `sliding_window.py` so it is aggregated.  If a model returns a dict
with no recognised heads, the function raises ``ValueError`` rather
than silently returning a near-empty result.

---

## 5. `freeze_dit_backbone` integer vs boolean is *not* a typo

**Symptom.** `freeze_dit_backbone: 1` in YAML behaves *differently*
from `freeze_dit_backbone: true` and you can't tell why.

**Where.** `brainbow/modules/cosmos_transfer_2_5/base.py::on_train_epoch_start`.
* `bool(True)` -> permanently frozen.
* `bool(False)` -> permanently trainable.
* `int(N)` -> frozen during epochs `0 .. N-1`, unfrozen at epoch `N`
  (optimizer rebuilt at that hop so the new param group picks up
  `dit_backbone_lr`).

**Why.** A two-stage warm-up schedule: epoch 0 trains heads + adapter
only (~7 M params, fast convergence), then the DiT joins.  Encoding
"epochs frozen" as an int is the simplest knob that survives YAML
round-trips.

**Remediation.** **Intentional.**  See
[`ARCHITECT.md` §1.6](./ARCHITECT.md#16-freeze-flags--what-actually-moves)
for the parameter-budget consequences and remember to bump the
default `dit_backbone_lr` if you change `N`.

---

## 6. UMAP "auto" backend silently degrades to PCA

**Symptom.** Your TensorBoard `instance/pred` panel looks like a
straight 2-D PCA, even though you wrote `projection_algorithm: umap`
in the config.

**Where.** `brainbow/utils/manifold.py:295-297` -- when no UMAP
backend is installed and `backend="auto"`, the code prints nothing
and falls through to PCA.

**Why.** UMAP is an optional GPU-only dependency (cuML).  CPU UMAP is
slow enough that we'd rather show *something* than block image-logger
emission.

**Remediation.** Pass `projection_backend: "umap-cuml"` (or
`"umap-cpu"`) to fail loud when the backend is missing.  Phase 3f's
narrower-exception sweep ups this to a `logger.warning` so you at
least see it in the run log.

---

## 7. Lazy dataset normalisation cache swallows every error

**Symptom.** A read-only data root (or a corrupted `.norm.json`)
silently re-computes normalisation stats on every worker, slowing
startup and producing slightly different statistics across workers.

**Where.** `brainbow/datasets/lazy.py:_read_norm_cache` and
`_write_norm_cache` -- both wrap their I/O in `except Exception:
return None / pass`.

**Why.** The cache is an optimisation; we never want a permission
error or a stale file to crash a long DDP run.

**Remediation.** Phase 3f narrows these to
`(OSError, json.JSONDecodeError)` so a real bug surfaces.  Until
then, if your workers are slow on first epoch, check the data root
is writable.

---

## 8. Lazy dataset HDF5 file handles are never closed

**Symptom.** Long DDP runs exhausting the OS open-file limit on data
roots with thousands of HDF5 chunks.

**Where.** `brainbow/datasets/lazy.py` -- the thread-local
`_thread_local` cache opens an `h5py.File` per worker per volume and
never closes them.

**Why.** Closing on every read is much slower than caching; the
expectation was that workers are short-lived.  With
`persistent_workers=True` (default), they aren't.

**Remediation.** Either bump `ulimit -n`, set
`persistent_workers=false` for very-many-volume datasets, or close
the cache periodically.  Phase 3c's lazy-setup refactor adds a
`__del__` that closes handles.

---

## 9. Empty `train_volumes` -> `None` train dataset

**Symptom.** You comment out the entire `train_volumes:` block to do
a val-only run, and `trainer.fit` either crashes or trains on nothing.

**Where.**
[`brainbow/datamodules/snemi3d.py`, `microns.py`, `neurons.py`]
-- each `setup` only creates `self.train_dataset` if `train_volumes`
is non-empty; otherwise it stays `None`.

**Why.** Originally written so `trainer.test` could be invoked
without populating the train split.

**Remediation.** Phase 3c makes empty `train_volumes` raise during
`setup` rather than silently succeeding.  Until then, point
`train_volumes` at a single-volume placeholder when you really want
val-only.

---

## 10. `MICRONSDataset._load_volume` ignores `vol_spec["root"]`

**Symptom.** You set `vol_spec = {"vol": "...", "seg": "...", "root":
"/scratch/alt"}` to override the root for one volume, and it's still
loaded from the global `root_dir`.

**Where.** `brainbow/datasets/microns.py:_load_volume` (~line 101)
hard-codes `self.root_dir`.

**Why.** Oversight when `root` was added to SNEMI3D and Neurons; the
MICRONS leaf wasn't updated.

**Remediation.** Phase 3c honours `vol_spec["root"]` consistently in
all leaves.

---

## 11. `pmap` falls back to sequential on ANY exception

**Symptom.** A bug inside the worker function (e.g. an `IndexError`)
makes the loop run sequentially with no warning, ~10x slower.

**Where.** `brainbow/utils/parallel.py:66-73`.  `except Exception` ->
`logger.debug(...)` (which is hidden by default).

**Why.** Multiprocessing setup can fail in container environments
(no `/dev/shm`, broken `forkserver`).  The fall-back keeps training
alive.

**Remediation.** Phase 3f narrows the exception to the actual
multiprocessing setup errors and re-raises everything else.  Until
then, set `logging.getLogger("brainbow.utils.parallel").setLevel("DEBUG")`
when investigating slow target construction.

---

## 12. Image-logger autocast cast back to fp32

**Symptom.** Predictions in TensorBoard look subtly different from
the training fp16/bf16 forward (especially logits near 0).

**Where.** `brainbow/callbacks/tensorboard/image_logger.py::_run_visualization`
-- runs the forward under `autocast` then casts the output dict back
to fp32.

**Why.** Image rendering / matplotlib glyphs require fp32 precision.

**Remediation.** **Intentional.**  Don't read scalar values off the
ImageLogger's predictions for non-visualisation purposes.

---

## 13. `compile=true` does not compile the whole model

**Symptom.** `nvidia-smi` shows your DiT compiled, but the VAE decoder
and task heads do not have any `torch.compile` overhead -- and you
can't figure out why "compile" gives only a 10% speedup.

**Where.** `scripts/train.py:446-456`.  Only `module.model.dit` is
wrapped in `torch.compile`; the rest of the wrapper isn't.

**Why.** `torch.compile + DDP` runs frozen subgraphs in
`inference_mode`, producing tensors that can't be saved for backward.
Compiling only the trainable DiT avoids that.

**Remediation.** **Intentional.**  See the comment in `train.py:434-445`
for the full rationale.

---

## 14. `compile_fullgraph=true` conflicts with Cosmos under DDP

**Symptom.** Compile succeeds, runs once, then errors on the next step
with "tensor with version != 0 used in inference_mode".

**Where.** Same code path as #13, but with
`fullgraph=True`.  `default.yaml` says it's safe-to-leave-off; some
recipes (`snemi3d.yaml:208`) had it on.

**Why.** Same DDP + inference_mode interaction; `fullgraph` magnifies
it because graph breaks are no longer tolerated.

**Remediation.** Keep `compile_fullgraph: false` on multi-GPU runs.
Phase 4 reconciles the comment / default mismatch between
`default.yaml` and `snemi3d.yaml`.

---

## 15. `HoughVoting.forward` returns `(labels, None, None)`

**Symptom.** Reading the source, `HoughVoting` looks like it should be
a drop-in for `SoftMeanShift` -- and it is, but only after Phase 3e:
older revisions returned `labels` alone and crashed any caller that
unpacked the documented 3-tuple.

**Where.** `brainbow/inference/clusterer.py:HoughVoting.forward`.

**Why.** `HoughVoting` is bin-coordinate based, so there are no
embedding-space soft assignments / centers.  The contract is honoured
by padding with `None`s rather than synthesising fake values.

**Remediation.** Don't rely on the second / third tuple slots when the
clusterer is `hough_voting`; treat them as optional everywhere.

---

## 16. Stale `last.ckpt` example paths in configs

**Symptom.** You copy-paste a `resume_from_checkpoint:` example from
`configs/snemi3d.yaml` (lines 28-29) or `configs/boundary.yaml`
(lines 22-24), only to discover the path doesn't exist on your machine.

**Where.** `configs/snemi3d.yaml`, `configs/boundary.yaml`.  Stale
`outputs/checkpoints/last.ckpt` examples that predate the timestamped
run-dir convention introduced in `train.py:407-411`.

**Why.** The run-dir layout changed but the docs didn't.

**Remediation.** Phase 4 fixes the stale comments.  Until then,
checkpoint paths are inside `outputs/<timestamp>_<experiment>/checkpoints/`.

---

## 17. `combine.yaml` drops AC4 from train

**Symptom.** You expect "combine" to literally mean SNEMI3D-AC3 +
SNEMI3D-AC4 + neurons + MICrONS, but training only sees AC3.

**Where.** `configs/combine.yaml::data.train_volumes`.

**Why.** AC4 is held out as the canonical SNEMI3D val volume; combine
was designed to leave it out of training.

**Remediation.** **Intentional.**  Documented in
[`ORGANIZATION.md` §9](./ORGANIZATION.md#9-hydra-configuration-layering).

---

## 18. Hydra default chain has a hidden hop

**Symptom.** You `--config-name boundary` and see settings you didn't
write -- they came from `combine.yaml`, and ultimately from
`snemi3d.yaml` and `default.yaml`.

**Where.** Hydra's `defaults:` lists in each YAML.  The chain is::

    default.yaml -> snemi3d.yaml -> combine.yaml -> boundary.yaml

**Why.** Layered overrides keep individual files small and let
`boundary.yaml` only declare what's *different*.

**Remediation.** **Intentional.**  Phase 4 adds a per-file
inheritance-chain comment header so the chain is visible without
opening every parent.  See also `configs/example_annotated.yaml`
(Phase 2e) for the merged result.

---

## 19. Notebook / docstring claim "all preprocessors implement `save()`"

**Symptom.** You write a generic round-trip test using
`BasePreprocessor.save(...)` and one of the leaves raises
`NotImplementedError`.

**Where.** `brainbow/preprocessors/__init__.py` (now fixed in Phase 1)
and the older `BasePreprocessor` class docstring.

**Why.** Some formats are intentionally read-only.

**Remediation.** The `__init__.py` docstring now states `save()` is
optional.  When in doubt, check `hasattr(p, "save")` or wrap the call
in `try/except NotImplementedError`.
