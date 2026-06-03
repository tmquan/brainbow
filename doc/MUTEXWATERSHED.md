# Brainbow — Mutex Watershed Head & Agglomeration

How Brainbow turns a backbone's dense predictions into an instance
segmentation **without any threshold or seed parameter**, using the
Mutex Watershed (Wolf et al. 2018).

This doc covers the full path: the affinity + sem + raw head layout, the
loss that supervises it, the parameter-free agglomeration algorithm and
its from-scratch implementation, and where each piece is wired into
training / evaluation / visualisation.

- Channel layout & helpers: [`brainbow/losses/_common.py`](../brainbow/losses/_common.py)
- Loss: [`brainbow/losses/affinity.py`](../brainbow/losses/affinity.py)
- Algorithm: [`brainbow/inference/mutex_watershed.py`](../brainbow/inference/mutex_watershed.py)
- Eval wiring: [`brainbow/modules/base.py`](../brainbow/modules/base.py)
- Visualisation: [`brainbow/callbacks/tensorboard/heads.py`](../brainbow/callbacks/tensorboard/heads.py),
  [`brainbow/callbacks/tensorboard/image_logger.py`](../brainbow/callbacks/tensorboard/image_logger.py)

---

## 1. The head

Every backbone (`Cosmos3Nano3DWrapper`, `CosmosPredict3DWrapper`,
`CosmosTransfer3DWrapper`, `Vista3DWrapper`) emits **one** dense tensor
`[B, HEAD_CHANNELS, D, H, W]`.  The canonical layout is the single source
of truth in `brainbow/losses/_common.py`:

```
ch  0 .. N_AFF-1 : aff   (N_AFF=14)  sigmoid, per-offset affinity in (0, 1)
ch  N_AFF        : sem   (1)         sigmoid, foreground / boundary prob
ch  N_AFF + 1    : raw   (1)         linear,  L1 reconstruction of input EM
                                     -----------------------------------
                                     HEAD_CHANNELS = N_AFF + 2 = 16
```

- `AFF_SLICE = slice(0, 14)`, `SEM_SLICE = slice(14, 15)`, `RAW_SLICE = slice(15, 16)`.
- `apply_head_activations` sigmoids the contiguous `aff + sem` block
  (`SIGMOID_SLICE`) and leaves `raw` linear — one call at the end of each
  wrapper's `forward`.

### 1.1 Affinity offsets

`aff[o, v] = P(label[v] == label[v + offset_o])` — a **high** value means
"these two voxels belong to the same object" (the `+`/attractive
convention).  The offset list `AFFINITY_OFFSETS` (`(dz, dy, dx)`) is
anisotropy-aware for EM (long reach in-plane, short in Z):

| #  | offset (dz,dy,dx) | role                 |
|----|-------------------|----------------------|
| 1  | (-1, 0, 0)        | attractive nn (z)    |
| 2  | (0, -1, 0)        | attractive nn (y)    |
| 3  | (0, 0, -1)        | attractive nn (x)    |
| 4  | (0, -3, 0)        | repulsive in-plane   |
| 5  | (0, 0, -3)        | repulsive in-plane   |
| 6  | (0, -9, 0)        | repulsive in-plane   |
| 7  | (0, 0, -9)        | repulsive in-plane   |
| 8  | (0, -27, 0)       | repulsive in-plane   |
| 9  | (0, 0, -27)       | repulsive in-plane   |
| 10 | (0, -9, -9)       | repulsive diagonal   |
| 11 | (0, 9, -9)        | repulsive diagonal   |
| 12 | (-2, 0, 0)        | repulsive z          |
| 13 | (-3, 0, 0)        | repulsive z          |
| 14 | (-4, 0, 0)        | repulsive z          |

`N_ATTRACTIVE = 3` (the leading nearest-neighbour offsets).  The first
three are the **attractive** edges; the remaining eleven are **repulsive**
(long-range) edges that the Mutex Watershed uses as mutual-exclusion
constraints.

To change the offset set, edit `AFFINITY_OFFSETS` / `N_ATTRACTIVE` in
`_common.py` — `HEAD_CHANNELS` and every downstream consumer (head width,
loss, target builder, agglomerator) re-derive from it automatically.  If
you do, set `model.head_channels` and the `training.mutex_watershed.offsets`
override accordingly.

---

## 2. The loss (`AffinityFGLoss`)

`brainbow/losses/affinity.py` supervises all three fields and returns
`{"loss", "loss/aff", "loss/sem", "loss/raw"}`:

- **aff** — masked, offset-weighted composite (BCE + soft-Dice + optional
  focal) of `head[:, AFF_SLICE]` against
  `affinity_target_from_offsets(labels, offsets, background)`.
  - The target is `1` iff `label[v] == label[v + o]` (replicate-padded at
    the volume edge).
  - A **validity mask** (`affinity_validity_mask`) drops every edge with a
    non-foreground endpoint, so the loss only learns within-/across-object
    relations, never foreground-vs-background.
  - `attractive_weight` / `repulsive_weight` rebalance the nn vs
    long-range offset groups; the per-offset weight vector is a buffer so
    it follows device / dtype.
- **sem** — `DiceBCEFocalLoss` on `head[:, SEM_SLICE]` vs `labels > 0`
  (with `ignore_index` voxels masked out as background).
- **raw** — plain L1 / MSE of the linear `head[:, RAW_SLICE]` against the
  (normalised) input EM intensity; an auxiliary self-supervised signal
  that stabilises the shared decoder features.

`canonical_loss_keys()` enumerates the active keys (gated only by
`weight_* > 0`, never by batch content) so the eval loop can pre-seed a
rank-consistent accumulator and reduce metrics across ranks **without**
a fragile `all_gather_object`.

The loss is the **training** supervisor only — the Mutex Watershed never
runs in the training step.

---

## 3. The algorithm

The Mutex Watershed produces a segmentation in a single pass over the
edge set, with no threshold and no seeds (Wolf et al., *The Mutex
Watershed*, ECCV/CVPR 2018).

### 3.1 Edges

Each offset `o` and voxel `v` defines an edge `(v, v + o)`:

- **attractive** (the first `n_attractive` offsets): priority `= aff`
  (high affinity ⇒ strong "merge").
- **repulsive** (the rest): priority `= 1 - aff` (low affinity ⇒ strong
  "must separate"); these are **mutex** edges.

Both priorities live in `[0, 1]`, so attractive and repulsive edges sort
on one scale.

### 3.2 Single pass (Kruskal with mutex)

Process all edges in **descending priority** with a union-find:

```
for (u, v, is_mutex) in sorted_edges:        # priority desc
    ru, rv = find(u), find(v)
    if ru == rv:               continue       # already same cluster
    if mutex_exists(ru, rv):   continue       # blocked by a constraint
    if is_mutex:  add_mutex(ru, rv)           # record "must separate"
    else:         union(ru, rv)               # merge (inherit mutexes)
```

- An **attractive** edge merges its two clusters *unless* they are already
  separated by an active mutex.
- A **repulsive** edge adds a mutex between the two clusters *unless* they
  are already merged.

The result is the maximal set of merges consistent with the strongest
constraints — parameter-free.

---

## 4. Implementation

`brainbow/inference/mutex_watershed.py`.  No `affogato` / `elf`
dependency — the core loop is JIT-compiled with `numba` over flat numpy
arrays (with a pure-python fallback if numba is unavailable).

### 4.1 numba core (`_mws_core`)

- **Union-find** with path compression (`_find`) + union by rank.
- **Mutex constraints** are stored as per-root **singly-linked lists** in
  flat int64 arrays (`link_next`, `link_to`, plus `head` / `tail` /
  `count` per node):
  - A mutex partner is stored as a *node id* and resolved with `find(...)`
    at query time, so the structure never has to migrate stale
    representative ids.
  - On `union`, the smaller-cluster check walks the shorter chain; the two
    chains are **spliced in O(1)** (tail-to-head) so merges stay cheap.
  - Storage is pre-allocated to `2 * n_mutex_edges` (an accepted mutex
    appends two entries), so the whole pass runs in nopython mode with no
    Python-object growth.

This keeps the dominant costs to the edge sort (`np.argsort`) and the
union-find finds — both fast in numba — while the number of *accepted*
mutexes stays bounded by the number of neighbouring segments.

### 4.2 Edge construction (`_build_edges`)

Pure-numpy, vectorised per offset via paired source/target slices
(`_axis_slices`):

- voxel ids are flat indices into the `[D, H, W]` grid;
- a `mask` (foreground) keeps only edges with **both** endpoints inside it
  (and drops out-of-bounds edges, which fall out as empty slices);
- **`strides`** subsample the *repulsive* edges only (attractive nn edges
  are always dense) — the primary lever to bound the edge count on large
  crops.

### 4.3 Public API

```python
# Functional: one volume.
labels = mutex_watershed(
    affinities,        # [n_offsets, D, H, W] float in [0, 1]
    offsets,           # list of (dz, dy, dx)
    n_attractive,
    strides=(1, 4, 4), # repulsive-edge subsampling (Z, Y, X)
    mask=None,         # [D, H, W] bool foreground
    size_filter=0,     # drop components < N voxels -> background
)  # -> [D, H, W] int64, 0 = background, 1..K relabelled consecutively

# nn.Module: batched, drop-in for the validation agglomeration step.
agglomerator = MutexWatershed(offsets=None, n_attractive=None,
                              strides=(1, 4, 4), size_filter=0)
ins = agglomerator(aff[B, n_offsets, D, H, W], fg_mask[B, D, H, W])  # -> [B, D, H, W] long
```

`MutexWatershed` runs on CPU (numpy/numba) per crop and returns the same
`[B, *spatial]` long label contract as the old embedding clusterers, so it
is a drop-in for the metric path.  Its `offsets` / `n_attractive` default
to `brainbow.losses.AFFINITY_OFFSETS` / `N_ATTRACTIVE`.

---

## 5. Where it's wired

### 5.1 Validation metrics — every val crop

`BaseCircuitModule` builds `self.agglomerator = MutexWatershed(**training_config["mutex_watershed"])`
(offsets / n_attractive default to the criterion's so head, target and
agglomerator share one edge convention).  In
`_accumulate_instance_metrics` it runs MWS over `head[:, AFF_SLICE]`,
**restricted to the GT foreground** (isolating agglomeration quality from
the sem head), and scores the instance metrics:

```
{stage}/automatic/ins/metric/{ari, ami, voi, voi_split, voi_merge, ted}
```

The semantic (foreground) metrics come from `SEM_SLICE`:

```
{stage}/automatic/sem/metric/{acc, iou, dice}
```

### 5.2 Visualisation — viz batch, once per epoch

`ImageLogger._run_visualization` runs the **same** agglomerator on the
cached viz batch (3-D, restricted to predicted `sem > 0.5`) and logs the
segmentation panels:

```
{stage}/automatic/pred/label/pre    # Mutex Watershed instances
{stage}/automatic/pred/label/mul    # × predicted sem mask
{stage}/automatic/pred/sem          # foreground probability
{stage}/automatic/pred/raw          # linear reconstruction
{stage}/automatic/pred/aff/{offset} # all N_AFF affinity channels
{stage}/automatic/true/aff/{offset} # GT affinity (3-D)
{stage}/automatic/true/{image,label}
```

All `N_AFF` affinity channels are shown by default
(`aff_panel_indices(...)`); pass `max_repulsive=N` to log a curated
subset instead.

---

## 6. Config

`configs/default.yaml` / `configs/snemi3d.yaml`:

```yaml
model:
  head_channels: 16            # = N_AFF + 2; must match the layout

loss:                          # AffinityFGLoss
  background: -1
  ignore_index: -100
  weight_aff:
    weight: 1.0
    lambda_bce: 1.0
    lambda_dice: 1.0
    lambda_focal: 0.0
    gamma: 2.0
    attractive_weight: 1.0
    repulsive_weight: 1.0
    mask_to_foreground: true
  weight_sem: { weight: 1.0, lambda_bce: 1.0, lambda_dice: 1.0, lambda_focal: 1.0, gamma: 2.0 }
  weight_raw: { weight: 1.0, loss: l1 }

training:
  mutex_watershed:
    strides: [1, 4, 4]         # repulsive-edge subsampling (Z, Y, X)
    size_filter: 50            # min component size (voxels)
    # offsets / n_attractive default to the loss's; only set if you also
    # override AFFINITY_OFFSETS in brainbow/losses/_common.py
```

---

## 7. Performance notes

The Mutex Watershed runs on **CPU per crop**, so it is the slow part of
validation.  Two levers, both throughput-only (not accuracy knobs in the
usual regime):

- **`training.mutex_watershed.strides`** — coarser repulsive strides ⇒ far
  fewer edges ⇒ faster MWS.  Default `[1, 4, 4]` keeps Z dense (anisotropy)
  and takes every 4th voxel in-plane for the long-range edges.
- **`data.val_batch_size`** / `training.limit_val_batches` — fewer / smaller
  val crops per epoch.

The visualisation runs MWS once per epoch on `max_images` crops (rank 0
only); the metric path runs it on every validation crop.

---

## 8. Why parameter-free matters

Earlier heads produced a per-voxel **embedding** that a separate clusterer
(soft mean-shift / spatial-CC / HDBSCAN) turned into instances — each with
its own bandwidth / `delta_v` / `min_cluster_size` threshold to tune, and
each sensitive to it.  The affinity + Mutex Watershed path removes those
knobs entirely: the network predicts merge/split evidence directly, and
the agglomeration is a single deterministic pass with **no threshold**.
The only remaining knobs (`strides`, `size_filter`) trade compute for
fidelity, not segmentation quality.

## Reference

S. Wolf, C. Pape, A. Bailoni, N. Rahaman, A. Kreshuk, U. Köthe,
F. A. Hamprecht. *The Mutex Watershed: Efficient, Parameter-Free Image
Partitioning.* ECCV 2018 (extended: *The Mutex Watershed and its
Objective*, TPAMI 2020).
