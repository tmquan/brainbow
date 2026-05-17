# Brainbow — Gotchas and Silent Failure Modes

> Audience: anyone debugging an unexpected result, looking for a
> "this can't be right" moment, or onboarding a new contributor.

> **Phase 1 audit note (April 2026).**  This file is being rewritten
> in Phase 3 of the audit overhaul.  Phase 1 has only:
> 1. Marked entries that referenced **deleted modules** or
>    **removed APIs** as `(REMOVED in Phase 1 cleanup)` so they no
>    longer mislead readers.
> 2. Refreshed the wording of a few entries whose source-line anchors
>    drifted.
>
> The full rewrite will dedupe overlap with `ORGANIZATION.md` /
> `WALKTHROUGH.md` and reorder by symptom-frequency.  Treat any entry
> dated "April 2026 cleanup" as historical context, not active risk.

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

## 4. (Removed in Phase 1 cleanup -- `sliding_window_inference` head auto-detect)

`sliding_window_inference` no longer detects head sets from a dummy
pass.  The unified 32-channel head means the model wrapper returns a
single tensor `[B, 32, ...]`; the sliding-window code aggregates it as
one tensor and slices into named fields downstream via
`brainbow.losses.slice_head`.  See
[brainbow/inference/sliding_window.py](../brainbow/inference/sliding_window.py).

---

## 5. (Removed in Phase 1 cleanup -- `freeze_dit_backbone` integer schedule)

The integer-epoch thaw branch in `on_train_epoch_start` was deleted.
`freeze_*` flags are now plain bools applied **once at construction**
by the model wrapper:

* `freeze_dit_backbone: true`  -> permanently frozen.
* `freeze_dit_backbone: false` -> permanently trainable.

Integer values are silently truthy (so `freeze_*: 0` means trainable
and any non-zero int means frozen), but the per-epoch state machine
that used to thaw at epoch `N` is gone.  See
[brainbow/models/cosmos_transfer_2_5/wrapper.py](../brainbow/models/cosmos_transfer_2_5/wrapper.py)
and [`ARCHITECT.md` §1.6](./ARCHITECT.md#16-freeze-flags--what-actually-moves).

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

## 11. (Removed in April-2026 cleanup -- ``pmap`` / ``utils.parallel``)

The forkserver-based parallel-map helper backed the boundary /
skeleton-weight CPU paths in :class:`InstanceLoss`.  Both that helper
(``brainbow.utils.parallel``) and those weight paths were dropped in
the lean-up: the production config disabled them anyway
(``weight_edge: 1.0`` / ``weight_bone: 1.0``), and removing the
machinery cuts ~150 LOC plus the silent-fallback footgun this gotcha
described.

---

## 12. Image-logger autocast cast back to fp32

**Symptom.** Predictions in TensorBoard look subtly different from
the training fp16/bf16 forward (especially logits near 0).

**Where.** `brainbow/callbacks/tensorboard/image_logger.py::_run_visualization`
-- runs the forward under `autocast` then casts the output dict back
to fp32.

**Why.** Image rendering -- HSV→RGB LUTs, eigendecomposition for the
`cov` overlay, manifold projection -- requires fp32 precision.

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
Phase 4 reconciles the comment / default mismatch between
`default.yaml` and `snemi3d.yaml`.

---

## 15. (Removed in April-2026 cleanup -- ``HoughVoting`` clusterer)

The Hough-voting clusterer (offset-head based) was dropped: no model
in the repo emits an offset head, and no shipped config selected
``name: hough_voting``.  The remaining clusterers are
``soft_meanshift`` (training-time, differentiable), ``hdbscan``, and
``spatial_cc`` -- see ``brainbow/inference/clusterer.py``.

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

## 19. (Removed in Phase 1 cleanup -- separate `GeometryLoss` head)

The standalone four-head structure (`semantic`, `instance`, `geometry`,
`boundary`) was replaced by a unified 32-channel head whose layout is
owned by `brainbow.losses._common.HEAD_LAYOUT`:

```
raw[0,1) | sem[1,2) | skl[2,3) | dir[3,6) | cov[6,12) | rad[12,13) | avg[13,16) | emb[16,32)
```

The previous geometry-head layout swap (`raw|cov|dir` → `raw|dir|cov`)
is a non-issue under the unified head — `dir` and `cov` are now fixed
slices of the single output tensor.  Old `head_geometry` checkpoints
are not loadable; train fresh under the unified head.

---

## 20. Affinity targets default to `background=-1` (no masking)

**Symptom.** A run trained with the old `background=0` mask suddenly
shows a denser supervised aff target after pulling main; the
TensorBoard `true/aff/{01_t1,...}` panels are visibly less
"checkerboard-y" along instance edges.

**Where.**
[`brainbow/losses/combined.py`](../brainbow/losses/combined.py)
(`CombinedLoss(__init__)` default `background: int = -1`) and
the explicit `background: -1` for the unified loss in
[`configs/default.yaml`](../configs/default.yaml).

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
masking explicitly — semantically identical to `-1` here.

---

## 21. `max_hard_pairs` does not bound the push-loss forward peak

**Symptom.** A MICrONS crop with thousands of instances spikes
`cuda_memory/max_allocated_gb_train` even though
`weight_emb.max_hard_pairs: 4096` is set.  It looks like the
knob is broken.

**Where.**
[`brainbow/losses/combined.py`](../brainbow/losses/combined.py) embedding
push branch
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
[`configs/snemi3d.yaml`](../configs/snemi3d.yaml) `weight_emb`
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

## 23. (Removed in Phase 1 cleanup -- multi-head `get_output_channels`)

The four-head dict has been replaced by a unified 32-channel head, so
`BaseModel.get_output_channels()` returns the single integer
`HEAD_CHANNELS = 32` (or whatever the wrapper was configured with).
Per-field widths are read from `brainbow.losses.HEAD_LAYOUT`, e.g.
`HEAD_LAYOUT["dir"].stop - HEAD_LAYOUT["dir"].start`.

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

## 25. `loss/emb/norm` rises monotonically during training

**Symptom.** Embedding-norm regulariser scalar climbs steadily from
~3 to ~4 over 30 epochs even though `weight_norm: 0.001` is enabled.

**Where.**
[`brainbow/losses/combined.py`](../brainbow/losses/combined.py)
embedding-norm branch and the `weight_norm` knob in
[`configs/default.yaml`](../configs/default.yaml) under `weight_emb`.

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

## 27. (Removed in Phase 1 cleanup -- `freeze_dit_backbone: N` epoch count)

The integer-epoch interpretation has been deleted along with the
per-epoch thaw machinery (see #5).  `freeze_dit_backbone` is now a
plain bool: `true` freezes the whole DiT for the run, `false` keeps
it trainable.  If you need per-block layer freezing, walk
`self.dit.blocks[:N].requires_grad_(False)` yourself outside the
Hydra surface.

---

## 28. (Removed in Phase 1 cleanup -- `weight_geometry.dir_target`)

`GeometryLoss` is gone; direction supervision is now a slice of the
unified head (`HEAD_LAYOUT["dir"]`).  There is currently no
configurable direction target -- centroid-pointing is the only
implementation in `brainbow/transforms/direction.py`.  Adding a
skeleton target is a clean follow-up but not exposed via Hydra today.

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

## 32. Vista regression channels were sigmoided in early versions (resolved by #39)

**Symptom (historical).** Training the **Vista3D** wrapper with
``weight_raw > 0`` (or any other regression head) produced a high
``loss/raw`` that didn't decay below ~0.5 even after many epochs,
while the same loss config dropped to ~0.07 on the Cosmos wrapper.

**Resolved by #39.**  The activation policy is now uniform across
both wrappers: every regression channel (raw, avg, dir, cov) is
linear; only `sem` (binary semantic) carries sigmoid; the derived
12-channel affinity heads compute their own sigmoids inside
`soft_aff_from_field`.

---

## 33. `loss.weight_geometry: { weight_dir: 1.0 }` (no `weight:`) used to silently disable the head

**Symptom (historical).**  A nested mapping omits the field's
``weight:`` key, intending to inherit the default::

    weight_dir:
      loss: l1

The field is then **not instantiated** and every sub-loss kwarg is
silently dropped.  No warning, no zero-scalar, just nothing.

**Where.**
[`brainbow/losses/combined.py::_split_field`](../brainbow/losses/combined.py).

**Why.** Before the fix, ``_split_field`` used a per-field
``default_weight`` argument (``0.0`` for some fields, ``1.0`` for
others).  In the nested-mapping branch
``d.pop("weight", default_weight)`` therefore returned ``0.0`` for
those fields, and they were skipped.

**Remediation.** Fixed: a nested mapping without ``weight:`` is now
treated as ``weight: 1.0`` regardless of field -- a user who wrote a
nested block clearly intended to enable the field.  If you want the
field disabled, write ``weight_<field>: 0`` (scalar) or
``weight_<field>: { weight: 0 }`` (nested explicit).

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

**Resolved by the April-2026 cleanup**: ``include_clefts`` /
``include_mito`` were dropped from ``configs/default.yaml`` and from
``scripts/train.py::_build_datamodule_kwargs``.  Multi-channel
MICrONS supervision was never implemented in any datamodule; the
config knobs were forward-looking placeholders and accumulated
config noise, so they're now gone.  If you do need cleft / mito
supervision later, add the channels to the relevant
``MICRONSDataModule`` constructor and re-introduce the keys then.

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

## 39. Activation policy migration April 2026: regression heads are now linear

**Symptom (historical).** Regression-supervised channels (raw, dir,
cov, avg) decayed slower than expected and stalled with a saturation
plateau.  Loading any pre-April 2026 checkpoint under the new code
produces nonsense on those channels.

**Where.**
[`brainbow/models/cosmos_transfer_2_5/decoder.py`](../brainbow/models/cosmos_transfer_2_5/decoder.py)
(``_DecoderAdapter3D.forward`` -- the activation contract),
[`brainbow/models/vista/wrapper.py`](../brainbow/models/vista/wrapper.py)
(forward -- mirrors the contract),
[`brainbow/losses/combined.py`](../brainbow/losses/combined.py)
(field docstrings document the linear regression contract).

**Why.** The previous policy was "sigmoid everywhere we could".  For
the **classification-supervised** channels (semantic) that's correct
-- BCE and Dice / IoU consume probabilities by construction.  For the
**regression-supervised** channels (raw, dir, cov, avg) sigmoid + L1
has a saturation problem: the chain-rule factor through
``sigmoid'(x) = p(1-p)`` collapses to ~0 whenever the target is near
``0`` or ``1``, so very dark / very bright voxels get effectively
zero gradient and the loss stalls.  Moving those heads to linear
gives the regression loss a constant gradient magnitude (``±1`` for
L1 everywhere except the kink) and lets the model actually reach the
extremes.

The current rule is **sigmoid only where the loss is BCE / Dice**:

| Field        | Slice         | Supervision     | Activation |
| ------------ | ------------- | --------------- | ---------- |
| `raw`        | `[0, 1)`      | L1 / MSE        | linear     |
| `sem`        | `[1, 2)`      | BCE / Dice      | sigmoid    |
| `dir`        | `[2, 5)`      | L1 / MSE        | linear     |
| `cov`        | `[5, 11)`     | L1 / MSE        | linear     |
| `avg`        | `[11, 14)`    | L1 / MSE        | linear     |
| `emb`        | `[14, 30)`    | discriminative  | linear     |
| `aff_avg`/`aff_emb` (derived) | n/a | BCE / Dice | applied internally by `soft_aff_from_field` |

**Remediation.** Old four-head checkpoints can't be loaded under the
unified head -- train fresh.  Inside the unified head, regression
fields are linear and visualisation panels apply a `clamp(0, 1)` for
display only (see
[`brainbow/callbacks/tensorboard/heads.py`](../brainbow/callbacks/tensorboard/heads.py)).

**Replaces gotcha #32** (Vista regression raw not sigmoided).  Under
the new uniform policy, both wrappers emit linear regression and the
mismatch is gone.

---

## 40. April-2026 cleanup -- features removed for leanness

A pass over the codebase dropped a handful of features that no
shipped config exercised and that contributed cognitive overhead
without paying their way in production.  If you're reading old
notebooks or external code that imported any of these, here's the
shortest path forward.

| Removed                                                  | Replacement                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------- |
| ``HoughVoting`` clusterer + ``cluster_offsets_hough``    | none -- no model emits offsets; use ``soft_meanshift`` / ``hdbscan`` / ``spatial_cc`` |
| ``MeanShiftClusterer`` + ``cluster_embeddings_meanshift`` | ``HDBSCANClusterer`` (cuML / CPU; auto-K) or ``SpatialCCClusterer`` (anisotropy-aware) |
| AXI metric (geometric mean of ARI and AMI)               | log ARI and AMI separately; plot the geometric mean offline if needed  |
| ``InstanceLoss(weight_edge=..., weight_bone=...)``       | per-voxel boundary / skeleton weighting was disabled in the production config (=1.0) and the cpu/torch kernels were 4 paths (~150 LOC) keeping a feature nobody used |
| ``InstanceLoss(anchor_to_centroid=..., centroid_scale=...)`` | sinusoidal-encoding centroid anchor; experimental, never enabled in any config |
| ``InstanceLoss(... semantic_ids=...)`` multi-class branch | no datamodule populated ``semantic_ids``; the loss is single-class only |
| ``CombinedLoss(... learned_task_weights=True)``          | Kendall-Gal uncertainty weighting; never enabled in any shipped config |
| ``SemanticLoss(label_smoothing=...)``                    | sigmoid BCE has no native smoothing knob; the param was stored but never used |
| Flat-form loss config schema (``weight_ce`` at top level, ``boundary_*`` prefix) | nested form (``weight_<head>: { weight: ..., ... }``) is now the only schema |
| ``data.include_clefts`` / ``data.include_mito``          | placeholders for a multi-channel MICrONS pipeline that was never wired in |
| Vista ``PointPromptEncoder`` + ``sample_point_prompts`` + ``forward(... point_prompts=...)`` | interactive proofreading was never wired into a training loop; the encoder was added then frozen on every step |
| ``CosmosTransfer3DWrapper._try_load_raw_checkpoint``     | the dead third loader path that loaded HF safetensors into a ``_StandaloneDiT3D`` (different architecture) -- the `_try_load_diffusers` and `_try_load_cosmos_package` paths cover the production path; if both fail the wrapper now uses the random-init standalone DiT explicitly |
| ``brainbow.utils.parallel`` (``pmap``)                   | only used by the boundary / skeleton-weight CPU path that's also gone |

Total: ~1,300 LOC removed, 8 files / modules deleted, two tests
classes (AXI, MeanShift) dropped.  185/185 remaining tests pass.

If you discover something you actually need that was cut, the
restoration recipe is one of:

1. ``git revert`` the cleanup commit's removal of the specific file
   / branch.
2. Re-apply the patch from a previous commit (``git log --all -- <path>``
   to find the last revision that had it).

---

## 41. cov-overlay eigendecomposition does **not** use `torch.linalg.eigh`

**Symptom.** Training crashes during sanity check (or the first few
ImageLogger panels) with::

    cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling
    `cusolverDnXsyevBatched(...)`. ... try linear algebra operators
    with other supported backends.

**Where.** `brainbow/callbacks/tensorboard/geometry.py::_eigh_2x2_sym`
and its caller `_render_cov_flow`.

**Why.** The `pred/cov` panel needs the principal eigenvector of a
2x2 symmetric structure-tensor matrix at **every foreground pixel**.
At a typical 4×512×512 panel that's ~1M matrices per epoch.
``torch.linalg.eigh`` dispatches to cuSOLVER's batched ``syevj``
kernel (``cusolverDnXsyevBatched``), which is iterative and unstable
on million-matrix batches when the inputs are still random or
contain NaN/Inf -- exactly the situation during PyTorch Lightning's
sanity-check pass and the first training epochs.  The crash
manifests as the cuSOLVER status code above and aborts training.

**Remediation.** **Intentional.**  We use a closed-form 2×2
analytical eigendecomposition (`_eigh_2x2_sym`) that is exact,
dispatch-free, fp32-safe, and requires no cuSOLVER call.  The input
is also `nan_to_num`-sanitised so a single bad pixel cannot poison
the whole panel.  Do **not** "simplify" this back to
``torch.linalg.eigh`` -- the previous implementation crashed every
fresh run before the first checkpoint.

If you ever extend the cov overlay to 3×3 (full ZYX submatrix) you
will need a closed-form 3×3 routine, **not** ``eigh``; do it on CPU
or via Cardano's formula rather than reintroducing the cuSOLVER
batched path.

---

## 42. kimimaro skeletonization runs inside DataLoader workers (CPU only)

**Symptom (debugging mode).** With `data.num_workers: 0`, every
training step blocks for ~100 ms on the main process while
`SkeletonGeometryd` runs `kimimaro.skeletonize` and the per-instance
Euclidean distance transform.  Throughput is much lower than expected.

**Where.**
[`brainbow/transforms/skeleton.py`](../brainbow/transforms/skeleton.py)
inside :class:`SkeletonGeometryd` and :func:`compute_skeleton_geometry`.

**Why.**  kimimaro is CPU-only and we call it with `parallel=1` (NOT
`0`) so it stays single-threaded under MONAI's `forkserver` DataLoader
workers.  kimimaro 5.x interprets `parallel <= 0` as "fork
`cpu_count()` subprocesses" via pathos, each backed by named POSIX
shm segments `kimimaro-shm-{dbf,cc-labels}-*` (see entry #43 below
for the original `/dev/shm` blow-up).  `parallel=1` short-circuits
that branch entirely.  With `num_workers > 0` the per-crop kimimaro
cost is hidden by the dataloader prefetch queue; with
`num_workers: 0` it lands directly on the training loop's critical
path.

**Remediation.**
- For real training, leave `data.num_workers >= 4` so kimimaro overlaps
  with the GPU forward.  At ~50 instances on a 64³ crop the geometry
  pass is ~100-200 ms, well under the typical prefetch budget.
- For step-through debugging (a profiler, `breakpoint()`, etc.), set
  `loss.weight_skl: 0`, `loss.weight_dir: 0`, `loss.weight_cov: 0`,
  `loss.weight_rad: 0` -- this disables `compute_geometry` end-to-end
  and skips `SkeletonGeometryd` entirely.  Or set `data.num_workers >= 4`
  even in single-step debugging and place breakpoints inside
  `training_step` only.

**Falls back gracefully.** When kimimaro fails to import (no wheel
for the platform, or fresh editable install without the dep), the
transform silently downgrades to
`skimage.morphology.skeletonize` per instance.  Topology is preserved
but the centerline can be 1-2 voxels off where the two algorithms
disagree (most prominent at branch points).

---

## 43. `/dev/shm` invariant: kimimaro must run with `parallel=1`

**Symptom (historical -- 2026-05-17 04:26:50).**  The user session was
OOM-killed after a multi-day combine.yaml run; `/dev/shm` was
filled with ~50 k orphaned `kimimaro-shm-{dbf,cc-labels}-*` POSIX
named-shm segments totalling ~984 GB.  Every DataLoader worker
restart leaked another ~2 segments per crop.

**Where.**
[`brainbow/transforms/skeleton.py::_skeletonize_all_kimimaro`](../brainbow/transforms/skeleton.py)
-- the multi-label kimimaro call inside MONAI's `forkserver`
DataLoader workers.

**Why.** kimimaro 5.x maps `parallel <= 0` to "spawn `cpu_count()`
pathos subprocesses" (see `kimimaro/intake.py` ~line 195).  Each
pathos child mmaps two named POSIX shm segments for its
distance-field and CC-labels scratch space.  Those segments are
only `shm_unlink()`-ed on the success / SIGINT / SIGTERM paths of
the **parent** kimimaro call.  Under MONAI's `forkserver` workers,
abnormal exits are common (DataLoader tear-down, SIGKILL from
systemd, unhandled exceptions in `compute_skeleton_geometry`, etc.)
and orphan the segments in `/dev/shm`, where nothing else cleans
them up.  At 8 workers x ~2 leaks/crop x 16 000 crops/epoch, a single
overnight run can fill a 1 TB `tmpfs`.

**Invariant.** Brainbow's `forkserver`-based DataLoader workers
**must not** spawn additional pathos / multiprocessing pools.  The
single source of truth is
`brainbow/transforms/skeleton.py::_skeletonize_all_kimimaro`, which
hard-codes `parallel=1` in the kimimaro call.  Anywhere else that
intends to parallelise CPU work inside a worker must use
`np.vectorize` / numpy ufuncs / threadpoolctl-bounded BLAS rather
than spawning subprocesses.

**Audit checklist (2026-05 sweep).**

| Vector                                     | Current state                               |
| ------------------------------------------ | ------------------------------------------- |
| `kimimaro.skeletonize`                     | `parallel=1` (skeleton.py L195) -- safe     |
| `pathos` / `multiprocessing.Pool`          | not imported anywhere in `brainbow/`        |
| `multiprocessing.shared_memory`            | not imported anywhere in `brainbow/`        |
| MONAI `CacheDataset` / `SmartCacheDataset` | not used (lazy path bypasses it; `cache_rate=0` in snemi3d.yaml is a no-op) |
| `torch.multiprocessing.set_sharing_strategy` | not called -- defaults to `file_descriptor` under `forkserver` |
| cuPy IPC handles                           | not used; cucim/cupy run only inside the GPU-bound path of `_use_gpu()` |
| DataLoader `multiprocessing_context`       | `forkserver` (datamodules/base.py L537/550/562) |

**Remediation.** Don't touch `parallel=` in `skeleton.py` without
re-doing the audit.  If you add a new transform that wants pathos /
multiprocessing parallelism, gate it behind a check that the worker
isn't already a forkserver child (e.g. test
`multiprocessing.parent_process() is None`) -- otherwise you will
re-create this leak.  And before merging anything that imports
`pathos` / `multiprocessing.Pool` / `multiprocessing.shared_memory`
into a hot DataLoader path, run the combine.yaml recipe overnight
and check `df -h /dev/shm`.
