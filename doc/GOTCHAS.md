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

**Remediation.** All shipped configs now use the nested form; the
flat-kwarg path remains in `combined.py` only for back-compat with
old checkpoints' hparams.  Phase 3b extracts the legacy-key migration
into `_migrate_legacy_keys` so the back-compat logic is in one place.

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

**Remediation.** **Intentional.**  Phase 4 adds a per-file
inheritance-chain comment header so the chain is visible without
opening every parent.

---

## 18. Notebook / docstring claim "all preprocessors implement `save()`"

**Symptom.** You write a generic round-trip test using
`BasePreprocessor.save(...)` and one of the leaves raises
`NotImplementedError`.

**Where.** `brainbow/preprocessors/__init__.py` (now fixed in Phase 1)
and the older `BasePreprocessor` class docstring.

**Why.** Some formats are intentionally read-only.

**Remediation.** The `__init__.py` docstring now states `save()` is
optional.  When in doubt, check `hasattr(p, "save")` or wrap the call
in `try/except NotImplementedError`.

---

## 19. Geometry channel layout was swapped in April 2026

**Symptom.** Loading an older `head_geometry` checkpoint produces
nonsense direction / covariance predictions while ``raw`` (ch 0) looks
fine.  The total geometry loss starts ~10× higher than the same model
trained from a fresh init.

**Where.**
[`brainbow/losses/geometry.py`](../brainbow/losses/geometry.py) (forward slice),
[`brainbow/models/cosmos_transfer_2_5/decoder.py`](../brainbow/models/cosmos_transfer_2_5/decoder.py)
(activation policy), and
[`brainbow/callbacks/tensorboard/heads.py`](../brainbow/callbacks/tensorboard/heads.py)
(`_log_geometry`).

**Why.** Before April 2026 the geometry head was laid out as
``[raw(1) | cov(S*(S+1)/2) | dir(S)]``.  It was swapped to
``[raw(1) | dir(S) | cov(S*(S+1)/2)]`` so the (cheaper, more
visualisable) direction channels sit immediately after raw and the
larger covariance block lives at the tail.  The total channel count
(`10` in 3-D) is unchanged, and the activation policy still applies
sigmoid only to ch 0 — both `dir` and `cov` carry signed values, so
sigmoid would clip them.  But the saved weights for `head_geometry`
permute their output rows under the new layout.

**Remediation.** Either (a) re-train fresh under the new layout, or
(b) write a tiny `state_dict` shim that permutes
`model.head_geometry.weight` and `.bias` rows
``[0, 7, 8, 9, 1, 2, 3, 4, 5, 6]`` -> ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``
before `load_state_dict(strict=False)`.  Channel layout is owned by
:class:`brainbow.losses.GeometryLoss` — that's the single source of
truth, every other doc / wrapper / callback now mirrors it.

---

## 20. Affinity targets default to `background=-1` (no masking)

**Symptom.** A run trained with the old `background=0` mask suddenly
shows a denser supervised aff target after pulling main; the
TensorBoard `boundary/true/aff/{...}` panels are visibly less
"checkerboard-y" along instance edges.

**Where.**
[`brainbow/losses/boundary.py`](../brainbow/losses/boundary.py)
(`BoundaryLoss(__init__)` default `background: Optional[int] = -1`),
[`brainbow/losses/instance.py`](../brainbow/losses/instance.py)
(`InstanceLoss(__init__)` default `background: Optional[int] = -1`),
and the explicit `background: -1` for both heads in
[`configs/snemi3d.yaml`](../configs/snemi3d.yaml).

**Why.** The `FindBoundariesd` transform sets boundary voxels (the
ones between adjacent instances) to label `0`.  With `background=0`,
those voxels were masked out of the aff target, producing alternating
"aff=1, aff=0" pixels along every instance border — a visible
checkerboard in TB.  Setting `background=-1` (a sentinel that no
voxel ever has) keeps every voxel in the target: boundary-to-boundary
face pairs become `aff=1`, boundary-to-foreground stays `aff=0`.  No
checkerboard, denser supervision.

**Remediation.** **Intentional**, but if you resume an old checkpoint
trained with `background=0`, the supervised signal changes.  Loss
values will jump by a few percent for a few hundred steps while the
model adapts.  No state-dict fixup is needed; only the target
construction is affected.  Pass `null` (YAML `~`) to opt out of any
masking explicitly — semantically identical to `-1` here, but kept
in the public API for symmetry with `BoundaryLoss`.

---

## 21. `max_hard_pairs` does not bound the push-loss forward peak

**Symptom.** A MICrONS crop with thousands of instances spikes
`cuda_memory/max_allocated_gb_train` even though
`weight_instance.max_hard_pairs: 4096` is set.  It looks like the
knob is broken.

**Where.**
[`brainbow/losses/instance.py::_compute_loss_push`](../brainbow/losses/instance.py)
(``diff = rearrange(centers, "i e -> i 1 e") - rearrange(centers, "j e -> 1 j e")``).

**Why.** The full ``[K, K, E]`` pairwise difference tensor is
materialised first, then the upper triangle is taken, then ``topk``
filters down to ``max_hard_pairs``.  The forward peak therefore
scales as ``O(K² · E)`` regardless of the post-topk cap.  ``K`` is
the number of unique instance ids in a single crop, which on dense
MICrONS volumes can be 1k–4k.  At ``E = 10`` and bf16 batch 4, that's
``[K, K, 10]`` per batch element — 40 MB at K=1k, 2.5 GB at K=4k.

**Remediation.** ``max_hard_pairs`` still helps gradient memory and
concentrates supervision on the actually-touching pairs; it just
doesn't bound the forward.  If you really need to cap the forward
peak by ``K``, the loss has to compute ``pw`` row-blockwise (e.g. 256
rows at a time, pre-mining top-k within each block before
concatenating).  See the comment block in
[`configs/snemi3d.yaml`](../configs/snemi3d.yaml) `weight_instance`
for the wording.

---

## 22. VOI metric is *mean-of-per-volume*, not global

**Symptom.** Comparing your val/automatic/instance/metric/voi to a
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

## 23. `BaseModel.get_output_channels()` returns `int`, docstring says `Dict[str, int]`

**Symptom.** A multi-head consumer expects a dict ``{"semantic": 1,
"instance": 10, ...}`` and gets a single integer
(`semantic_channels`).  Code that walks the dict crashes with
"int has no attribute keys".

**Where.** `brainbow/models/base.py` declares the abstract method as
returning `int`; the package docstring at
`brainbow/models/__init__.py` claims `Dict[str, int]`.
`CosmosTransfer3DWrapper.get_output_channels` returns `num_classes`
only (semantic head); `Vista3DWrapper` doesn't implement it at all.

**Why.** API drift.  Originally a single-head abstraction; multi-head
support was bolted on at the wrapper level without retrofitting the
ABC.

**Remediation.** **Open issue.**  Don't rely on
`get_output_channels()` for anything other than `semantic_channels`
today.  If you need per-head widths, read them from the loss
modules (`<loss>.task_channels`) or from the config block
(`model.{semantic,instance,boundary}_channels`).

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

## 25. `instance/loss/norm` rises monotonically during training

**Symptom.** Embedding-norm regulariser scalar climbs steadily from
~3 to ~4 over 30 epochs even though `weight_norm: 0.001` is enabled.

**Where.**
[`brainbow/losses/instance.py::_compute_loss_norm`](../brainbow/losses/instance.py)
and the `weight_norm` knob in
[`configs/snemi3d.yaml`](../configs/snemi3d.yaml) `weight_instance`.

**Why.** The push term repels centroids out to a margin of
`2 * delta_d = 3.0`.  When a single crop contains many instances
(MICrONS minnie65 averages 100s-1000s), centroids are pushed in many
directions simultaneously, growing their norms.  The norm
regulariser only damps proportionally to its weight (`0.001`), so it
tames but doesn't reverse the growth.

**Remediation.** **Intentional but tunable.**  Either (a) raise
`weight_norm` (e.g. to `0.01`) for a stronger pull toward the origin,
(b) set `normalize_embeddings: true` to bound each centroid to the
unit hypersphere (eliminates the norm growth entirely; you'll also
want to lower `delta_d` since distances are now bounded by `2`), or
(c) lower `delta_d` so the push margin is reachable without
inflating norms.  Watch
`val/automatic/instance/metric/ari` to confirm the trade-off doesn't
hurt clustering.

---

## 26. Eval pipeline used to apply random augmentations (fixed April 2026)

**Symptom (historical).** Validation / test metrics
(`val/automatic/instance/metric/{ari,ami,voi,ted}`,
`val/automatic/semantic/metric/{acc,iou,dice}`) drift run-to-run with
the same checkpoint, and disagree with metrics measured by an
external evaluator on the same volumes.  Loss curves look noisier on
val than on train despite val being a smaller, fixed set.

**Where.**
[`brainbow/datamodules/base.py::CircuitDataModule.get_val_transforms`](../brainbow/datamodules/base.py).

**Why.** Before the fix, `get_val_transforms` chained
`*_original_transforms(sd)` and `*_semantic_transforms(sd)` -- the
exact same hooks that the train pipeline uses -- which meant every
val pass ran:

| Transform                | Probability                |
| ------------------------ | -------------------------- |
| ``RandFlipd`` × 3 axes   | 0.5 each                   |
| ``RandRotate90d``        | 0.5                        |
| ``RandTransposeXYd``     | 0.5                        |
| ``Rand3DElasticd``       | ``data.elastic_prob``      |
| ``RandGaussianNoised``   | **1.0**  (always applied)  |
| ``RandAdjustContrastd``  | **1.0**  (always applied)  |

Because these were random, validation metrics measured **model + a
fresh stochastic augmentation realisation per batch**, not the
model's response to the held-out volume.  ARI / AMI / VOI on
SNEMI3D's `val/automatic/instance/metric/*` curves up to commit
`<this fix>` are therefore lower-bounded by what the model would
score on clean inputs.

**Remediation.** Fixed: `get_val_transforms` now only runs
deterministic ops (`EnsureChannelFirstd` → `FindBoundariesd` →
pad+center-crop or resize → `_instance_transforms` (CC relabel) →
`_geometry_transforms` (direction / covariance)).  In-progress runs
will see the val curves shift on the **next** validation tick after
pulling main; the train curves are unaffected.

---

## 27. `freeze_dit_backbone: N` is an epoch count, not a layer count

**Symptom.** ``model.freeze_dit_backbone: 2`` in a config -- you read
it as "freeze the first 2 DiT blocks" because that's a familiar
fine-tuning idiom, but every DiT parameter shows
``requires_grad: True`` from epoch 2 onward.

**Where.**
[`brainbow/modules/cosmos_transfer_2_5/base.py::on_train_epoch_start`](../brainbow/modules/cosmos_transfer_2_5/base.py)
and the ``freeze_dit_backbone`` table in
[`doc/ARCHITECT.md`](./ARCHITECT.md) §1.6 .

**Why.** The integer form is **phased epoch-count semantics**: ``N``
means "freeze the *entire* DiT during epochs ``0 .. N-1``, then
unfreeze at the start of epoch ``N`` (and rebuild the optimiser so
the new param group picks up ``optimizer.dit_backbone_lr``)".  There
is **no** "freeze first N blocks" code path -- `freeze_dit_backbone()`
on the wrapper toggles `requires_grad` on every parameter under
``self.dit``.

**Remediation.** Read the SNEMI3D config comment block (now
corrected) or the `ARCHITECT.md` table.  If you genuinely need
per-block layer freezing, you'd have to walk
`self.dit.blocks[:N].requires_grad_(False)` yourself; not currently
exposed via Hydra.

---

## 28. `weight_geometry.dir_target: skeleton` raises at construction

**Symptom (historical).**  Setting
``weight_geometry.dir_target: skeleton`` in a YAML config didn't
actually change supervision -- the direction-field target was still
the centroid-pointing field.

**Where.**
[`brainbow/losses/geometry.py::GeometryLoss.__init__`](../brainbow/losses/geometry.py),
[`brainbow/transforms/direction.py::compute_direction_field`](../brainbow/transforms/direction.py),
[`brainbow/callbacks/tensorboard/heads.py::_log_geometry`](../brainbow/callbacks/tensorboard/heads.py).

**Why.**  Before the fix, ``GeometryLoss`` swallowed every unknown
kwarg via ``**kwargs`` (kept for ``CombinedLoss``-side flat-kwarg
forwarding flexibility), so ``dir_target=skeleton`` was silently
dropped.  ``compute_direction_field`` only implements the centroid
target; there is no skeleton path.  The TB callback read
``getattr(geom_loss, "dir_target", "centroid")`` and so always
labelled the panel ``geometry/pred/dir_centroid``.

**Remediation.** Fixed: ``GeometryLoss(dir_target=...)`` now stores
the value and **raises** at construction unless ``dir_target ==
"centroid"``.  When you actually want a skeleton target, implement
the skeleton branch in
``compute_direction_field`` and relax the guard.  The TB callback
keeps reading ``loss.dir_target`` so the new tag flows through
automatically once the implementation lands.

---

## 29. `dataset: neurons` used to crash at startup

**Symptom (historical).** Setting ``data.dataset: neurons`` in the
config raised
``ValueError: Unknown dataset type: 'neurons'`` from
``scripts/train.py::build_datamodule`` even though
``NeuronsDataModule`` was exported by ``brainbow.datamodules``.

**Where.** ``scripts/train.py::build_datamodule`` registry.

**Why.** The registry only listed ``snemi3d`` and ``microns``.

**Remediation.** Fixed: ``"neurons": NeuronsDataModule`` is now in
the registry.  Other datasets follow the same one-line edit pattern.

---

## 30. `dit_backbone_lr: 0` silently fell back to base LR

**Symptom (historical).** A config block::

    optimizer:
      lr: 2e-4
      dit_backbone_lr: 0

reported four parameter groups with the **same** LR (``2e-4``) on
all of them, not a frozen-rate backbone.

**Where.**
[`brainbow/modules/cosmos_transfer_2_5/base.py::configure_optimizers`](../brainbow/modules/cosmos_transfer_2_5/base.py).

**Why.** Before the fix the line was
``backbone_lr = self.optimizer_config.get("dit_backbone_lr") or lr`` --
``0 or lr`` is ``lr`` because Python ``or`` returns the first truthy
value and ``0`` is falsy.  The argument was effectively
"any non-zero number, or default".

**Remediation.** Fixed: explicit ``is None`` check -- a deliberate
``0`` is now honoured.

---

## 31. Resume-from-checkpoint path in YAML comment was wrong

**Symptom (historical).** The header comment in
[`configs/snemi3d.yaml`](../configs/snemi3d.yaml) suggested::

    training.resume_from_checkpoint=<output_dir>/<run_dir>/checkpoints/last.ckpt

but ``output_dir`` doesn't contain the run dir at runtime.

**Where.**
[`scripts/train.py`](../scripts/train.py) -- the timestamped run
directory **overwrites** ``cfg.output_dir`` via
``OmegaConf.update(cfg, "output_dir", str(run_dir), force_add=True)``
just after creating it.

**Why.** Two-segment templates (``<output_dir>/<run_dir>``) read
naturally but didn't survive that overwrite.  Every checkpoint write
afterward uses ``Path(output_dir) / "checkpoints"`` -- which is now
``<run_dir>/checkpoints``.

**Remediation.** Fixed: the YAML comment now reads
``training.resume_from_checkpoint=<run_dir>/checkpoints/last.ckpt``
and points at the literal "Run directory: ..." line printed at the
top of every training log.

---

## 32. Vista geometry head's `raw` channel is **not** sigmoided

**Symptom.** Training the **Vista3D** wrapper with
``weight_geometry.weight_raw > 0`` produces a high
``geometry/loss/raw`` that doesn't decay below ~0.5 even after many
epochs, while the same loss config drops to ~0.07 on the Cosmos
wrapper.

**Where.**
[`brainbow/models/vista/wrapper.py`](../brainbow/models/vista/wrapper.py)
(``forward`` returns ``geometry`` linearly) vs
[`brainbow/models/cosmos_transfer_2_5/decoder.py`](../brainbow/models/cosmos_transfer_2_5/decoder.py)
(``head_geometry`` output is composed as
``torch.cat([geom[:, :1].sigmoid(), geom[:, 1:]], dim=1)``).

**Why.** ``GeometryLoss._compute_loss_raw`` clamps the **target**
``raw_image`` to ``[0, 1]`` and then regresses ``pred_raw`` against
it.  On the Cosmos path, ``pred_raw`` is already in ``[0, 1]`` thanks
to the channel-0 sigmoid in the decoder; on the Vista path,
``pred_raw`` is unbounded logits, so the target / pred ranges don't
match and the L1 / MSE loss is artificially high.

**Remediation.** **Open issue.**  Either:

1. apply the same channel-0 sigmoid in the Vista wrapper's
   ``forward`` (preferred -- centralise the activation contract);
2. set ``weight_geometry.weight_raw: 0`` on Vista runs;
3. drop the target clamp inside ``GeometryLoss._compute_loss_raw``
   and rely on the wrapper to make the ranges match (less safe).

The Cosmos path is the production training target, so this only
bites Vista debug runs today.

---

## 33. `loss.weight_geometry: { weight_dir: 1.0 }` (no `weight:`) used to silently disable the head

**Symptom (historical).**  A nested mapping omits the head's
``weight:`` key, intending to inherit the default::

    weight_geometry:
      weight_dir: 1.0
      weight_cov: 1.0
      weight_raw: 1.0

The geometry head is then **not instantiated** and every geometry
sub-loss kwarg is silently dropped.  No warning, no zero-scalar, just
nothing.

**Where.**
[`brainbow/losses/combined.py::_split_head`](../brainbow/losses/combined.py).

**Why.** Before the fix, ``_split_head`` used a per-head
``default_weight`` argument (``0.0`` for geometry / boundary,
``1.0`` for semantic / instance).  In the nested-mapping branch
``d.pop("weight", default_weight)`` therefore returned ``0.0`` for
geometry / boundary, and the head was skipped.

**Remediation.** Fixed: a nested mapping without ``weight:`` is now
treated as ``weight: 1.0`` regardless of head -- a user who wrote a
nested block clearly intended to enable the head.  If you want the
head disabled, write ``weight_<head>: 0`` (scalar) or
``weight_<head>: { weight: 0 }`` (nested explicit).

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

## 37. `include_clefts` / `include_mito` config keys are dropped

**Symptom.** Multi-channel MICrONS supervision described in
``configs/default.yaml`` (``include_clefts``, ``include_mito``)
doesn't change the dataset's emitted batch keys; predictions still
come out as the standard 4-head set.

**Where.** ``scripts/train.py::_build_datamodule_kwargs`` populates
the keys, but ``inspect.signature(cls).parameters`` filters them
out at the call site (no datamodule constructor accepts them).

**Why.** Implementation gap -- the keys were added to the config
namespace ahead of the multi-channel datamodule that would consume
them.

**Remediation.** **Open issue.** Either implement a multi-channel
MICrONS datamodule that accepts the keys or drop them from
``configs/default.yaml`` and ``_build_datamodule_kwargs``.  Today
they're load-bearing only in the configs themselves; nothing in
``brainbow/datamodules/*.py`` references them.

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
