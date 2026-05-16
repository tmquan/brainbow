# Brainbow — Model Architecture & Parameter Budget

Three end-to-end wrappers live under `brainbow/models/`:

1. [`CosmosTransfer3DWrapper`](#1-cosmostransfer3dwrapper) — EM → pretrained Wan VAE → Cosmos-Transfer 2.5 (base DiT **+ ControlNet residual branch**) → one VISTA-style **32-channel unified head**.
2. `CosmosPredict3DWrapper` — same data flow as Transfer **without** the ControlNet residual branch (Predict is the upstream base DiT in NVIDIA's Cosmos 2.5 stack).  Shares all scaffolding with Transfer via `brainbow/models/cosmos_2_5_common/`; the only difference is the variant registry (HF repo `nvidia/Cosmos-Predict2.5-2B`) and the absence of `controlnet_revision` / `freeze_controlnet`.  When evaluating the parameter budget, drop the ControlNet row from §1.2.
3. [`Vista3DWrapper`](#2-vista3dwrapper) — EM → SegResNetDS2 → the same **32-channel unified head**.

Every channel count below mirrors `configs/default.yaml`. Parameter counts are
approximate; use `model.get_num_parameters(trainable_only=…)` on a loaded
instance for exact numbers.

---

## 1. `CosmosTransfer3DWrapper`

`brainbow/models/cosmos_transfer_2_5/wrapper.py`

### 1.1 Data flow

```
[B, 1, D, H, W]  EM volume
   │
   │ _adapt_to_rgb:          channel repeat 1 → 3                     (0 params)
   │ pad spatial/temporal:   multiples of (4, 8, 8)                   (0 params)
   ▼
[B, 3, D,   H,    W   ]
   │
   │ vae_encoder  (Wan 3-D VAE encoder)                        ≈ 50 M params
   │   stride  (4, 8, 8)  in (D, H, W)
   ▼
[B, 16, D/4, H/8, W/8]   latent grid
   │
   │ ┌──── controlnet (CosmosControlNetModel, residual branch) ≈ 0.3 B params
   │ │       n_controlnet_blocks (typically 4) × hidden 2048
   │ │       same EM latent fed as both ``controls_latents`` and ``latents``
   │ │       outputs ``control_block_samples``: list of residual tensors
   │ ▼
   │ block_controlnet_hidden_states (list, len = n_controlnet_blocks)
   │ │
   │ │ summed inside CosmosTransformerBlock.forward:
   │ │   hidden_states += controlnet_residual
   │ │ (every ``controlnet_block_every_n`` blocks, see
   │ │  `diffusers.models.transformers.transformer_cosmos`)
   │ ▼
   │ dit  (CosmosTransformer3DModel, 2B base variant)          ≈ 2.3 B params
   │   token-domain transformer: 28 blocks × hidden 2048
   │   hooks extract features at layers {7, 14, 21, 27}
   ▼
[B, N, 2048] × 4   per-layer token sequences
   │
   │ feature_projector (_FeatureProjector3D)                   ≈ 1.1 M params
   │   concat 4 × 2048 → MLP (1×1×1 conv) → feature_size
   ▼
[B, 64, D/4, H/8, W/8]
   │
   │ decoder_adapter._DecoderAdapter3D:                        ≈ 80 M params
   │   to_latent:      1×1×1 conv, 64 → 16                     ≈ 1 K params
   │   decoder_body:   Wan VAE decoder (same weights as vae_decoder)
   │                   ≈ 73 M params (shared reference, see §1.4)
   │   trilinear upsample to input size if needed
   ▼
[B, 64, D, H, W]   decoded feature map
   │
   └─ head  (VistaTaskHead3D, 64 → 30)                         ≈ 0.7 M params
```

### 1.2 Channel map

| Stage                        | Channels | Spatial factor vs input |
|------------------------------|----------|-------------------------|
| Input (EM)                   | **1**    | 1                       |
| After RGB adapt              | 3        | 1                       |
| VAE latent (DiT input)       | **16**   | 1 / (4, 8, 8)           |
| DiT hidden dim               | **2048** | 1 / (4, 8, 8)           |
| Extracted feature layers     | 4 × 2048 | 1 / (4, 8, 8)           |
| `feature_projector` output   | **64**   | 1 / (4, 8, 8)           |
| `to_latent` back to VAE      | **16**   | 1 / (4, 8, 8)           |
| Decoder output (trilinear up)| **64**   | 1                       |
| `head`                       | `head_channels = 32` = raw(1) + sem(1) + skl(1) + dir(3) + cov(6) + rad(1) + avg(3) + emb(16). Sigmoid on the contiguous (sem, skl) block at ch [1, 3); all other channels linear. |

### 1.3 The DiT variant (2B)

From `brainbow/models/cosmos_transfer_2_5/variants.py` (`_VARIANT_CONFIGS["2B"]`):

| Key                    | Value |
|------------------------|-------|
| HF repo                | `nvidia/Cosmos-Transfer2.5-2B`   |
| Base DiT revision      | `diffusers/general`              |
| ControlNet revision    | `diffusers/controlnet/general/edge` (default; also `depth` / `seg` / `blur`. Override via `model.controlnet_revision` -- empty string disables the ControlNet load path) |
| `hidden_dim`           | **2048**                         |
| `num_layers`           | **28**                           |
| `num_heads`            | **16** (head_dim 128)            |
| `latent_channels`      | **16**                           |
| `spatial_compression`  | **8**                            |
| `temporal_compression` | **4**                            |
| `mlp_ratio`            | **4**                            |
| `patch_size`           | **2**                            |
| Default `feature_layers` | `{n/4, n/2, 3n/4, n-1}` → `{7, 14, 21, 27}` |

> **Note.** Cosmos-Transfer2.5 is upstream a **base + ControlNet** stack
> (see [the model card](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B):
> *"The control branch is formed by replicating a few transformer blocks of
> the base model … then injected into the corresponding transformer blocks
> of the base model"*). Both halves live in the same HF repo on
> different revisions; `_try_load_diffusers` and `_try_load_controlnet` in
> `wrapper.py` download both, and the residual branch is summed into the
> base every `controlnet_block_every_n` blocks at forward time.

Per-block parameter budget:

- QKV projections:         3 × 2048² ≈ 12.6 M
- Attention output proj:   2048²     ≈  4.2 M
- MLP up + down (×4):      2 × 2048 × 8192 ≈ 33.6 M
- AdaLN modulation + norms ≈ 25 M
- **~75 M per block × 28 blocks ≈ 2.1 B**, plus patch embed, final norm and
  timestep embed ≈ 0.1-0.2 B → **~2.3 B total**.

### 1.4 `_DecoderAdapter3D` — shared-weight gotcha

The `decoder_adapter` holds a **reference** to the same `vae_decoder` module
that lives on the wrapper (`wrapper.vae_decoder is
wrapper.decoder_adapter.decoder_body`). Lightning's
`ModelSummary` walks the module tree and **double-counts** those weights once
per registration site. Use `CosmosTransfer3DWrapper.get_num_parameters(...)`
(which iterates `self.parameters()` and deduplicates by `id`) for authoritative
totals.

The adapter adds, on top of the shared decoder:

- `to_latent` — 1×1×1 conv, `feature_size → latent_channels` (~1 K params).
- One `VistaTaskHead3D` unified 32-channel head (~0.7 M).
- Optional trilinear upsample to input resolution (0 params).

### 1.5 `VistaTaskHead3D`

`brainbow/models/vista/heads.py` — shared by both wrappers.

```
in         → (optional 1×1 conv to refine_channels)
           → UnetrBasicBlock (3×3×3, instance-norm, residual)
           → UnetrBasicBlock (3×3×3, instance-norm, residual)
           → (optional Dropout3d)
           → 1×1 conv  → out_channels
```

No internal upsampler — spatial resolution is preserved. At
`refine_channels=64` and `out_channels ≤ 16`, each head is **~0.7 M** params.

### 1.6 Skeleton-relative geometry (`skl` / `dir` / `cov` / `rad`)

Four of the eight head fields (`skl`, `dir`, `cov`, `rad`) are
supervised by **per-instance skeleton-relative** targets precomputed
inside the DataLoader by
[`SkeletonGeometryd`](../brainbow/transforms/skeleton.py) in a single
pass per crop:

1. **Skeletonize each foreground instance** using `kimimaro`
   (3-D TEASAR) or `skimage.morphology.skeletonize` as a fallback.
   Union the per-instance skeletons into a binary mask
   (`label_skl`).  Voxel-space, no anisotropy passed — we keep the
   skeleton in voxel coordinates so downstream per-voxel quantities
   line up with the model's grid.
2. **One per-instance Euclidean distance transform**
   (`distance_transform_edt(..., return_indices=True)` via cucim on
   GPU or scipy on CPU) seeded at the instance's skeleton voxels.
   For every foreground voxel `v` of instance `i` we get both the
   distance to the nearest skeleton voxel and the coordinates `s*(v)`
   of that voxel.
3. **Derive all four targets** from the per-voxel `(s*(v), v)` pair:
   - `dir(v) = (s*(v) − v) / ||s*(v) − v||` (unit centripetal vector;
     zero at skeleton voxels themselves).
   - `rad(v) = ||s*(v) − v||`, per-instance-normalised to `[0, 1]` by
     default (controlled by `data.radius_normalize`).
   - `cov(v) = M_{s*(v)}` — the upper-tri of the Voronoi-cell
     2nd-moment matrix at the nearest skeleton vertex, centered at
     the vertex.  Cells with `population < 2` carry the zero matrix.
     Trace-normalised by default.

The per-voxel **reconstruction identity** `rad(v) * dir(v) ==
(s*(v) − v) / R_i` holds by construction in both normalisation modes
and doubles as a unit test for the pipeline (see
`tests/test_skeleton_transform.py::test_rad_times_dir_recovers_offset`).

Eigendecomposing the Voronoi-cell covariance gives the geometric
interpretation of `cov`: for a roughly cylindrical neurite segment of
local radius `r` and inter-vertex spacing `Δℓ`, eigenvalues are
approximately `(r², r², Δℓ²)` with the **tangent** as the **smallest**
eigenvector (the Voronoi cell is "wide" across the cross-section and
"narrow" along the centerline because the next vertex's cell bounds
it).  At branch points and somas the cell deviates from rank-1 in a
way the 6-channel head can express.

### 1.7 Freeze flags → what actually moves

| Flag                       | Target module(s)                                    | Effect when `True` |
|----------------------------|-----------------------------------------------------|--------------------|
| `freeze_vae_encoder`       | `vae_encoder` (and any `_vae_ref[0].encode`)        | `requires_grad_(False)` + `eval()` + forward runs under `torch.no_grad()` |
| `freeze_dit_backbone`      | `dit` only (the base `CosmosTransformer3DModel`)    | `requires_grad_(False)`. The hook path only `.detach()`s block outputs when **both** the base DiT *and* the ControlNet are frozen — otherwise grad must flow through the block residual injection (`hidden_states += controlnet_residual`) back to the trainable ControlNet. |
| `freeze_controlnet`        | `controlnet` (the `CosmosControlNetModel` residual branch) | `requires_grad_(False)` + `eval()` + ControlNet forward runs under `torch.no_grad()`; ControlNet residuals are still summed into the base DiT but contribute zero gradient. |
| `freeze_vae_decoder`       | `decoder_adapter.decoder_body` (= `vae_decoder`)    | body frozen **except** the last up-block + `conv_norm_out`, which stay trainable as a fine-tuning shim |

The Cosmos-Transfer2.5 default in `configs/snemi3d.yaml` is to **freeze the
upper part (base DiT) and train the residual part (ControlNet)** — that's
the natural ControlNet pattern, and what NVIDIA's own training recipe uses.

`freeze_dit_backbone` accepts three forms in config:

| YAML value | Meaning |
|------------|---------|
| `true`     | permanently frozen |
| `false`    | permanently trainable (from step 0) |
| `1` (int)  | frozen during epoch 0, unfrozen at start of epoch 1 (optimizer rebuilt) |

The phased-schedule machinery lives in
`brainbow/modules/cosmos_transfer_2_5/base.py::on_train_epoch_start`.

### 1.8 Default parameter budget (snemi3d / combine recipes)

With `configs/snemi3d.yaml` (inherited by `configs/combine.yaml`):
`freeze_vae_encoder: true`, **`freeze_dit_backbone: true`**,
**`freeze_controlnet: false`**, `freeze_vae_decoder: true`, with the pretrained
HF Cosmos-Transfer 2B base + ControlNet-edge residual branch loaded.

| Component                                         | Total    | Trainable |
|---------------------------------------------------|---------:|----------:|
| VAE encoder                                       | ~50 M    | 0         |
| Base DiT (`self.dit`, frozen upper part)          | ~2.30 B  | 0         |
| ControlNet (`self.controlnet`, trainable residual)| ~0.30 B  | ~0.30 B   |
| `feature_projector`                               | ~1.1 M   | ~1.1 M    |
| `to_latent`                                       | ~1 K     | ~1 K      |
| VAE decoder body (frozen)                         | ~70 M    | 0         |
| VAE decoder shim (last up-block + norm)           | ~3 M     | ~3 M      |
| `VistaTaskHead3D` unified head                    | ~0.7 M   | ~0.7 M    |
| **Total**                                         | **~2.73 B** | **~0.31 B** |

ControlNet param count is approximate — `CosmosControlNetModel`'s
`n_controlnet_blocks` is checkpoint-specific and the model card for
`Cosmos-Transfer2.5-2B` quotes ~358 M parameters across the residual
branch. Use `model.get_num_parameters(trainable_only=True)` for the
exact number after construction.

The `_fallback_down` module (`brainbow/modules/cosmos_transfer_2_5/base.py`:
73-74) is only active when no HF VAE is loaded — in the pretrained path it is
frozen and contributes zero trainable params.

### 1.9 Practical training implications

- Default snemi3d trains **~0.31 B params** (ControlNet + heads + decoder
  shim). One forward still runs the full ~2.6 B-param base + control stack,
  but the AdamW state and grads are bounded by the trainable subset, so
  memory pressure is closer to a ~300 M-param fine-tune than a 2 B-param
  one. `model.gradient_checkpointing: true` is no longer mandatory.
- The optimizer has three LR groups
  (`brainbow/modules/cosmos_transfer_2_5/base.py::configure_optimizers`):
  `model.dit.*` → `optimizer.dit_backbone_lr`,
  `model.controlnet.*` → `optimizer.controlnet_lr` (defaults to
  `dit_backbone_lr`), and everything else → `optimizer.lr`. Keep both
  pretrained groups at 10× lower LR than the new heads.
- To unlock the base DiT (full fine-tune), flip
  `model.freeze_dit_backbone: false` (or set an integer epoch count for the
  warm-up schedule); to disable the ControlNet path entirely, set
  `model.controlnet_revision: ""`.
- Shared `vae_decoder` weights mean changes made via the "shim" (last up-block
  + out-norm) are visible to the main `wrapper.vae_decoder` too — no extra
  state-dict management needed when saving / loading.

---

## 2. `Vista3DWrapper`

`brainbow/models/vista/wrapper.py`

### 2.1 Data flow

```
[B, 1, D, H, W]  EM volume
   │
   │ backbone: SegResNetDS2 (or SegResNet fallback)     ≈ 30-50 M params
   │   blocks_down = (1, 2, 2, 4, 4)   init_filters=64
   │   norm="instance", dsdepth=1
   ▼
[B, 64, D, H, W]  full-resolution feature map
   │
   └─ head  (VistaTaskHead3D, 64 → 30)                     ≈ 0.7 M params
```

No VAE and no DiT — the SegResNetDS2 backbone does both downsampling and
upsampling internally.

### 2.2 Channel map

| Stage                        | Channels |
|------------------------------|----------|
| Input (EM)                   | **1**    |
| `init_filters`               | **64** (default; MONAI pretrained weights require **48**) |
| Backbone output / head input | **`feature_size` = 64** |
| `head`                       | `head_channels = 32` = raw(1) + sem(1) + skl(1) + dir(3) + cov(6) + rad(1) + avg(3) + emb(16). |

### 2.3 Pretraining

When `pretrained=True` **and** `feature_size == 48`, `load_pretrained_vista3d_encoder`
downloads MONAI's `VISTA3D-HF` encoder weights and loads them into the backbone
encoder (`strict=False`). With `feature_size == 64` (our default) the load is
skipped and the backbone starts random — trades pretrained init for a wider
feature channel throughout.

### 2.4 No freeze-flag API

`Vista3DWrapper` does **not** implement `freeze_vae_encoder` /
`freeze_dit_backbone` / `freeze_controlnet` / `freeze_vae_decoder`. Those are
Cosmos-specific. For Vista, the entire model is always trainable; freeze
individually if needed via `backbone.requires_grad_(False)` etc.

### 2.5 Same head as Cosmos

Vista and Cosmos now expose the same unified 32-channel head.  The only
difference is the backbone / decoder that produces the feature map before
the head.

### 2.6 Rough parameter budget

| Component         | Params    |
|-------------------|----------:|
| SegResNetDS2 (64) | ~30-45 M  |
| Unified task head | ~0.7 M    |
| **Total**         | **~35 M** |

Running Vista is roughly **70× cheaper per step** than full-unfrozen
Cosmos-2B and ~5× cheaper than the Cosmos warm-up (epoch 0). Reasonable for
local iteration and debugging when you don't need a boundary head.

---

## 3. Choosing between the two

| Use case                                               | Recommended wrapper |
|--------------------------------------------------------|---------------------|
| Full brainbow supervision (raw + avg + dual aff)       | Cosmos              |
| Fast local dev / debugging on a single GPU             | Vista               |
| Sufficient compute + data for 2.3 B parameter fine-tune | Cosmos (with warm-up schedule) |
| Transfer from MONAI VISTA3D pretrain (`feature_size=48`)| Vista               |

The `configs/default.yaml` and `configs/snemi3d.yaml` files default to
`model.type: cosmostransfer3d`; switch to `vista3d` in the config to train the
Vista wrapper instead.
