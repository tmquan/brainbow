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

**Where.** `scripts/train.py::_install_runtime_patches` (line 75).
This helper rebinds `torch.load` to a wrapper that forces
`weights_only=False`.  It is **called from `main()` (line 538)**, not
at import time, so `import scripts.train` from a notebook or test no
longer mutates the global `torch` module silently.

**Why.** PyTorch >= 2.6 made `weights_only=True` the default.  Lightning
checkpoints pickle non-tensor objects (`collections.defaultdict` for
metric / callback state, `OmegaConf` containers for hparams, custom
optimiser state).  `add_safe_globals` whitelists the *types*, but the
weights-only `SETITEM` opcode is hardcoded to accept only `dict` /
`OrderedDict` / `Counter` as the SETITEM target -- so resume still
fails with `defaultdict` state.  Our checkpoints are local, so the
script trusts them.

**Remediation.** Don't call `_install_runtime_patches()` in a process
that subsequently `torch.load`s untrusted files.  In notebooks, import
function-level helpers (e.g. `build_module`, `build_datamodule`) rather
than calling `main()`.

---

## 2. Loss-config schema is silently both flat and nested

**Symptom.** Two YAML files mix `weight_sem: 0.5` with
`weight_sem: { weight: 0.5, lambda_bce: 1.0, lambda_dice: 1.0,
lambda_focal: 1.0, gamma: 2.0 }` and both work, but you can't tell
which one is in effect from the config alone.

A scalar (`weight_sem: 0.5`) sets only the field weight; the nested
mapping additionally configures the `sem` head's `DiceBCEFocalLoss`
sub-terms (`lambda_{bce,dice,focal}`, `gamma`).  `weight_aff` /
`weight_raw` are plain scalars.

**Where.**
[`brainbow/losses/affinity.py`](../brainbow/losses/affinity.py)
(`AffinityFGLoss.__init__` splits scalar vs nested for `weight_sem`),
read consistently by `brainbow/modules/base.py`.

**Why.** A scalar is the common case; the nested form exists so the
foreground supervision can be tuned without a separate knob block.

**Remediation.** Prefer the nested form for `weight_sem` when you
touch its sub-terms; leave it scalar otherwise.

---

## 3. A loss term with `weight: 0.0` is skipped entirely

**Symptom.** You set `weight_raw: 0.0`, then look for `loss/raw` in
TensorBoard and it's absent.

**Where.** `brainbow/losses/affinity.py::AffinityFGLoss.forward` skips
any field whose weight is `0`, so it never enters the returned loss
dict (and the `raw` channel gets no gradient).

**Why.** Memory + speed: a zero-weighted term contributes nothing to
the backward, so computing/logging it is wasted work.

**Remediation.** **Intentional.**  Don't key dashboards or assertions
on a loss term you've zeroed; check the loss dict for the key first.
The head still emits all `HEAD_CHANNELS` regardless — only the
supervision is dropped.

---

## 5. `freeze_dit_backbone` integer warm-up schedule

`freeze_dit_backbone` accepts a bool *or* a non-negative int:

* `freeze_dit_backbone: true`  -> permanently frozen.
* `freeze_dit_backbone: false` -> permanently trainable from step 0.
* `freeze_dit_backbone: N` (int, `N >= 0`) -> frozen for epochs
  `0..N-1`, thawed at the start of epoch `N`.  `N == 0` is equivalent
  to `false`.

Parsing happens in
[`brainbow/models/cosmos_2_5_common/wrapper_base.py::_resolve_freeze_dit_backbone`](../brainbow/models/cosmos_2_5_common/wrapper_base.py);
the thaw is wired up in
[`brainbow/modules/cosmos_2_5_common/base.py::BaseCosmosModule.on_train_epoch_start`](../brainbow/modules/cosmos_2_5_common/base.py).
The optimizer keeps the DiT param group up front (zero-grad no-op
steps while frozen), so the thaw flips `requires_grad` without
rebuilding the optimizer or resetting the LR scheduler.

Negative ints and non-bool / non-int values raise at construction
rather than being silently coerced to truthy.  See also [`ARCHITECT.md`
§1.7](./ARCHITECT.md#17-freeze-flags--what-actually-moves).

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

**Remediation.** The handlers narrow to
`(OSError, json.JSONDecodeError)` so a real bug surfaces; if your
workers are slow on the first epoch, check the data root is writable.

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
the cache periodically.  The lazy dataset's `__del__` closes handles
on teardown.

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

**Remediation.** An empty `train_volumes` yields a `None` train
dataset rather than an error; point `train_volumes` at a
single-volume placeholder when you really want a val-only run.

---

## 10. `MICRONSDataset._load_volume` ignores `vol_spec["root"]`

**Symptom.** You set `vol_spec = {"vol": "...", "seg": "...", "root":
"/scratch/alt"}` to override the root for one volume, and it's still
loaded from the global `root_dir`.

**Where.** `brainbow/datasets/microns.py:_load_volume` (~line 101)
hard-codes `self.root_dir`.

**Why.** Oversight when `root` was added to SNEMI3D and Neurons; the
MICRONS leaf wasn't updated.

**Remediation.** Set the data `root` globally for MICRONS, or use the
SNEMI3D / Neurons leaves, which honour the per-volume `root` key.

---

## 12. Image-logger autocast cast back to fp32

**Symptom.** Predictions in TensorBoard look subtly different from
the training fp16/bf16 forward (especially logits near 0).

**Where.** `brainbow/callbacks/tensorboard/image_logger.py::_run_visualization`
-- runs the forward under `autocast` then casts the output dict back
to fp32.

**Why.** Image rendering -- colour-map LUTs, overlay compositing, and
the affinity / segmentation panels -- expects fp32 precision.

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
recipes (`snemi3d.yaml:222`) had it on.

**Why.** Same DDP + inference_mode interaction; `fullgraph` magnifies
it because graph breaks are no longer tolerated.

**Remediation.** Keep `compile_fullgraph: false` on multi-GPU runs.

---

## 16. `combine.yaml` drops AC4 from train

**Symptom.** You expect "combine" to literally mean SNEMI3D-AC3 +
SNEMI3D-AC4 + neurons + MICrONS, but training only sees AC3.

**Where.** `configs/combine.yaml::data.train_volumes`.

**Why.** AC4 is held out as the canonical SNEMI3D val volume; combine
was designed to leave it out of training.

**Remediation.** **Intentional.**  Documented in
[`ORGANIZATION.md` §9](./ORGANIZATION.md#9-hydra-configuration-layering).

---

## 17. Hydra default chain has a hidden hop

**Symptom.** You `--config-name combine` and see settings you didn't
write -- they came from `snemi3d.yaml`, and ultimately from `default.yaml`.

**Where.** Hydra's `defaults:` lists in each YAML.  The chain is::

    default.yaml -> snemi3d.yaml -> combine.yaml

**Why.** Layered overrides keep individual files small and let
`combine.yaml` only declare what's *different*.

**Remediation.** **Intentional.**  Each config carries a per-file
inheritance-chain comment header so the chain is visible without
opening every parent.

---

## 18. Notebook / docstring claim "all preprocessors implement `save()`"

**Symptom.** You write a generic round-trip test using
`BasePreprocessor.save(...)` and one of the leaves raises
`NotImplementedError`.

**Where.** `brainbow/preprocessors/__init__.py` and the
`BasePreprocessor` class docstring.

**Why.** Some formats are intentionally read-only.

**Remediation.** The `__init__.py` docstring now states `save()` is
optional.  When in doubt, check `hasattr(p, "save")` or wrap the call
in `try/except NotImplementedError`.

---

## 20. Affinity targets keep boundary voxels (`background=-1`)

**Where.** `brainbow/losses/affinity.py` (`AffinityFGLoss(background=-1)`),
the explicit `loss.background: -1` in the configs, and the target
builder `affinity_target_from_offsets` in `_common.py`.

**Why.** `FindBoundariesd` sets the voxels between adjacent instances to
label `0`.  `background` is the label value masked out of the affinity
target across all offsets.  `-1` is a sentinel no voxel ever has, so
every voxel stays in the target (boundary-to-boundary face pairs become
`aff=1`, boundary-to-foreground `aff=0`) — denser supervision and no
checkerboard along instance edges.

**Remediation.** Set `loss.background: 0` to instead mask the boundary
voxels out of the affinity target, or `null` (YAML `~`) to disable
masking entirely.

---

## 22. VOI metric is *mean-of-per-volume*, not global

**Symptom.** Comparing your `val/automatic/ins/metric/voi` to a
literature number ("VOI on MICrONS = ...") gives a confusing offset.

**Where.**
[`brainbow/metrics/instance.py::compute_per_batch_voi`](../brainbow/metrics/instance.py).

**Why.** The implementation averages per-sample VOI split / merge
across the batch and reports their sum as `total`.  VOI on pooled
voxels (the literature definition) would compute a single global
contingency table across the entire dataset.  These are not the same
number — they coincide only when every batch element has identical
class distribution.

**Remediation.** Document expected delta when comparing to
literature; if you need the global value, dump per-batch contingency
tables and aggregate offline.

---

## 24. HuggingFace download failure on rank 0 produces noisy errors on other ranks

**Symptom.** You restart training without internet and rank 0 prints
"HuggingFace download failed: …", then ranks 1-7 print
"FileNotFoundError" or "EntryNotFoundError" and the run aborts with
mixed errors.

**Where.** [`brainbow/models/cosmos_transfer_2_5/hf_loader.py`](../brainbow/models/cosmos_transfer_2_5/hf_loader.py)
(`_download_from_hf`) and the parallel
[`brainbow/models/vista/hf_loader.py`](../brainbow/models/vista/hf_loader.py).

**Why.** When rank 0's download fails, it calls `dist.barrier()` then
re-raises.  The other ranks unblock at the barrier, then call
`snapshot_download(local_files_only=True)` against the (empty) cache,
which fails with a different error class.  Not a deadlock — but the
log noise can mask the real cause on rank 0.

**Remediation.** Ensure outbound network on at least rank 0.  If you
need to retrain offline, pre-populate `~/.cache/huggingface/hub` on
the launch host before starting the multi-node run.  A future fix
should broadcast a "download succeeded" flag from rank 0 before
non-zero ranks attempt their `local_files_only` load.

---

## 27. `freeze_dit_backbone: N` epoch warm-up

The integer-N epoch warm-up (frozen for epochs `0..N-1`, thawed at
epoch `N`) is supported alongside the bool form — see #5 for the full
spec.  Per-block layer freezing is not exposed via Hydra; walk
`self.dit.blocks[:N].requires_grad_(False)` yourself if you need it.

---

## 34. Per-volume `find_boundaries` keys are no-op in lazy 3-D mode

**Symptom.** A YAML volume entry like::

    train_volumes:
      - vol: foo_volume
        seg: foo_segmentation
        root: data/SNEMI3D
        find_boundaries: 0          # I want no boundary stripping for this volume

is silently ignored on the SNEMI3D recipe (which uses
``slice_mode: false`` ⇒ 3-D lazy reads); only the global
``data.find_boundaries`` knob applies via the
``FindBoundariesd`` MONAI transform in the train pipeline.

**Where.**
[`brainbow/datasets/lazy.py::LazyVolDataset._discover_volumes`](../brainbow/datasets/lazy.py)
strips per-volume entries to ``vol`` / ``seg`` / ``root`` only.  The
eager branches in ``brainbow/datasets/{snemi3d,microns,neurons}.py``
honour the per-volume key at load time.

**Why.** Lazy reads stream the raw volume on demand from disk; per-
volume label-stripping would require a second pre-processed copy of
each volume which isn't materialised today.  The global
``data.find_boundaries`` works because it's a probabilistic
transform that runs after the lazy read.

**Remediation.** **Open issue.** Document ``find_boundaries`` as
"global probability only on the lazy path" or thread the per-volume
override into ``LazyVolDataset`` -- requires either a sidecar mask
or a transform inserted into the pipeline that consults the volume
key.  Today's recipes (``snemi3d.yaml``, ``combine.yaml``) don't
exercise the per-volume override so the silent no-op hasn't bitten
in production.

---

## 35. Lazy train vs val/test patch read sizes diverge with resolution zoom

**Symptom.** With ``resolution_zoom_prob: 1.0`` and a downsampling
range, training crops are sometimes obtained from a
``_safe_patch_size()``-enlarged read (e.g. 96 × 320 × 320 for
target ``80 × 256 × 256``) while validation crops use the literal
``patch_size``.  Boundary voxels visible to the model differ between
train and val.

**Where.**
[`brainbow/datamodules/base.py::_safe_patch_size`](../brainbow/datamodules/base.py)
and the lazy split builders in each dataset's datamodule
(``snemi3d.py`` ~120, ``microns.py`` ~120, ``neurons.py`` ~120).

**Why.** Train uses ``_effective_read_size()`` to provision a margin
for the post-zoom center-crop; val / test always use ``patch_size``.
This is a deliberate train/eval asymmetry so the eval pipeline stays
deterministic, but it does mean ``ResolutionZoom`` artefacts at the
crop edge differ between train and val.

**Remediation.** **Intentional**, but document the asymmetry so
users don't compare a literature paper's eval-on-full-volume number
to a Brainbow eval-on-patches number directly.

---

## 36. `cache_rate` is silently ignored in lazy 3-D mode

**Symptom.** Setting ``data.cache_rate: 1.0`` to "fit everything in
RAM" doesn't change steady-state RAM usage on a 3-D SNEMI3D run.

**Where.** ``data.cache_rate`` is forwarded to the eager
``CircuitDataset`` constructor in
[`brainbow/datamodules/base.py::setup`](../brainbow/datamodules/base.py)
but not consumed by ``LazyVolDataset`` (the path used when
``slice_mode: false`` ⇒ default for SNEMI3D / MICrONS / neurons).

**Why.** Caching a ``LazyVolDataset`` would defeat its purpose --
it exists to avoid materialising whole volumes in worker memory.
The MONAI ``CacheDataset`` semantics ``cache_rate`` was designed
for don't apply.

**Remediation.** Document; raise a warning when
``cache_rate > 0`` and the lazy path is selected; or add a small
LRU patch cache on top of ``LazyVolDataset`` if you really need it.

---

## 37. There are no `include_clefts` / `include_mito` config keys

Multi-channel MICrONS supervision (clefts / mitochondria) is not
implemented in any datamodule, so there are no `include_clefts` /
`include_mito` knobs — adding them to a config does nothing.  If you
need cleft / mito supervision, add the channels to the relevant
``MICRONSDataModule`` constructor and introduce the keys alongside.

---

## 38. CUDA `empty_cache()` is called twice at val end

**Symptom.** A small extra latency at the end of every validation
epoch and a stronger-than-expected drop in
``cuda_memory/reserved_gb`` between val and the next train epoch.

**Where.** Two separate hooks both call ``torch.cuda.empty_cache()``
at ``on_validation_epoch_end``:

1. [`brainbow/modules/base.py`](../brainbow/modules/base.py) -- the
   Lightning module's own override.
2. [`brainbow/callbacks/memory.py::CudaEmptyCacheCallback`](../brainbow/callbacks/memory.py)
   -- the opt-in callback.

**Why.** The module-level hook predates the callback; the callback
was added later for finer-grained control.  Lightning runs callback
hooks first, then the module hook -- so when both are enabled we
flush twice.

**Remediation.** **Cosmetic only** (``empty_cache`` is idempotent),
but candidates for cleanup: drop the module-level call when
``CudaEmptyCacheCallback`` is enabled in the callback set, or pick
one canonical location.

---

## 45. Composite ``DiceBCEFocalLoss`` supervises the `sem` head

**Symptom.** The TensorBoard tag for the foreground head is just
``loss/sem`` — no per-sub-term ``/ce``, ``/dice``, or ``/focal``
breakdown — even though the composite mixes three terms.  The
foreground supervision is configured by a nested mapping with
``lambda_{bce,dice,focal}`` + ``gamma`` under `loss.weight_sem`.

**Where.**
[`brainbow/losses/dice_bce_focal.py`](../brainbow/losses/dice_bce_focal.py)
defines the composite; `AffinityFGLoss` (in
[`brainbow/losses/affinity.py`](../brainbow/losses/affinity.py))
instantiates one for the `sem` head.  Default schema in
[`configs/default.yaml`](../configs/default.yaml).

**Why.** Dice is imbalance-robust but its gradient collapses at
saturation (``∂Dice/∂p`` shrinks with foreground volume); BCE gives a
constant per-voxel signal that survives saturation, and Focal's
``(1 - p_t)^gamma`` term sharpens the rare hard positives without a
``pos_weight`` knob.  Schema::

```yaml
weight_sem:
  weight: 1.0
  lambda_dice: 1.0
  lambda_bce: 1.0
  lambda_focal: 1.0
  gamma: 2.0            # 0 collapses focal back to plain BCE
```

**Activation contract.** The composite expects **logits** — the head
emits raw logits (no activation in `forward`).  The BCE term uses the
logit-stable ``binary_cross_entropy_with_logits``; the Dice
(``DiceLoss(sigmoid=False)``) and focal terms run on ``sigmoid(logits)``
(computed once internally).  Feed it the raw `sem` / `aff` logits, not
probabilities.

**Ablations.** Set any ``lambda_*`` to ``0`` to disable that term;
``gamma: 0`` reduces the focal term exactly to per-voxel BCE.

**See also.** Entry #2 (loss-config schema).

---

## 46. `boundary_target` decides whether erosion touches the affinity head

**Symptom.** Turning on ``data.find_boundaries`` to teach the `sem`
head thin gaps also changes the affinity (instance) supervision and
the validation instance GT — unexpectedly shifting VOI numbers.

**Where.**
[`brainbow/datamodules/base.py`](../brainbow/datamodules/base.py)
(`_boundary_semantic_transforms`, `_output_keys`),
[`brainbow/modules/base.py::_prepare_targets`](../brainbow/modules/base.py),
and [`brainbow/losses/affinity.py::AffinityFGLoss.forward`](../brainbow/losses/affinity.py).

**Why.** ``FindBoundariesd`` zeros boundary voxels in whatever label it
is given. With ``boundary_target: both`` (legacy) it mutates the shared
instance ``label`` in place, so **both** the `sem` target (`label > 0`)
**and** the affinity target / validity mask / val instance GT derive
from the eroded label. The fix is ``boundary_target: semantic``: a
separate eroded ``sem_label`` is produced (added to ``_output_keys`` and
threaded into ``targets["sem_label"]``); the `sem` loss + metric use it
while the affinity target (`affinity_target_from_offsets`) and the
validation instance GT / fg-mask keep the **pristine** ``label``.

**Contract.** When no ``sem_label`` is present (legacy / `both`), the
loss and metric fall back to ``labels`` — byte-identical to before. The
`sem_label` must be forwarded by ``scripts/train.py``
(`_build_datamodule_kwargs`) for the config knob to take effect.

**See also.** Entry #34 (per-volume vs global `find_boundaries`).
