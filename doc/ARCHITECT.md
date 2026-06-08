# Brainbow — Model Architecture & Parameter Budget

Every backbone produces the same **affinity + sem + raw head**
(`HEAD_CHANNELS = N_AFF + 2`, 16 channels), supervised by
`AffinityFGLoss` and agglomerated into instances at eval by the Mutex
Watershed — see [`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md) for the head
layout, loss, and eval.  This doc covers the **backbones** that feed that
head: their data flow and parameter budgets.

The wrappers live under `brainbow/models/`:

1. `CosmosPredict3DWrapper` — EM → Wan VAE → Cosmos-Predict 2.5 base DiT
   (no ControlNet) → head.  The flattened 2B baseline recipe
   (`configs/cosmospredict3d.yaml`, variant `2B`, DDP).  Shares all
   scaffolding with Transfer via `brainbow/models/cosmos_2_5_common/`;
   for its budget take §1 and drop the ControlNet row.
2. [`CosmosTransfer3DWrapper`](#1-cosmostransfer3dwrapper) — Cosmos-Transfer 2.5: the same data flow **plus** a ControlNet residual branch (§1).
3. `Cosmos3Nano3DWrapper` — Cosmos 3 (Nano) 16B omni transformer + Wan2.2 VAE; trained under FSDP (`brainbow/models/cosmos_3_nano/`).  The shipped `snemi3d.yaml` / `default.yaml` default (`model.type: cosmos3nano3d`, variant `Nano`).
4. [`Vista3DWrapper`](#2-vista3dwrapper) — EM → SegResNetDS2 → head (fast local iteration).

Channel counts mirror `configs/default.yaml`. Parameter counts are
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
   └─ head  (VistaTaskHead3D, 64 → HEAD_CHANNELS=16)           ≈ 0.7 M params
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
| `head`                       | `head_channels = HEAD_CHANNELS = 16` = aff(N_AFF=14) + sem(1) + raw(1). Raw logits / linear values, no activation in `forward`: aff + sem are logits (logit-stable BCE in the loss; sigmoid at metrics / MWS / TensorBoard), raw is linear (target in `[-1, 1]`). |

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
- One `VistaTaskHead3D` head emitting `HEAD_CHANNELS` (~0.7 M).
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
`refine_channels=64` and `out_channels = HEAD_CHANNELS = 16`, each head
is **~0.7 M** params.

### 1.6 Freeze flags → what actually moves

| Flag                       | Target module(s)                                    | Effect when `True` |
|----------------------------|-----------------------------------------------------|--------------------|
| `freeze_vae_encoder`       | `vae_encoder` (and any `_vae_ref[0].encode`)        | `requires_grad_(False)` + `eval()` + forward runs under `torch.no_grad()` |
| `freeze_dit_backbone`      | `dit` only (the base `CosmosTransformer3DModel`)    | `requires_grad_(False)`. The hook path only `.detach()`s block outputs when **both** the base DiT *and* the ControlNet are frozen — otherwise grad must flow through the block residual injection (`hidden_states += controlnet_residual`) back to the trainable ControlNet. |
| `freeze_controlnet`        | `controlnet` (the `CosmosControlNetModel` residual branch) | `requires_grad_(False)` + `eval()` + ControlNet forward runs under `torch.no_grad()`; ControlNet residuals are still summed into the base DiT but contribute zero gradient. |
| `freeze_vae_decoder`       | `decoder_adapter.decoder_body` (= `vae_decoder`)    | body frozen **except** the last up-block + `conv_norm_out`, which stay trainable as a fine-tuning shim |

The natural ControlNet recipe for Transfer is to **freeze the base DiT
and train the residual ControlNet branch** (what NVIDIA's own recipe
uses).  Predict / Cosmos3-Nano have no ControlNet, so they instead train
the base DiT directly (`freeze_dit_backbone: false`) or run a frozen
warm-up; the shipped `snemi3d.yaml` (`cosmospredict3d`) full-fine-tunes
the 2B base DiT.

`freeze_dit_backbone` accepts three forms in config:

| YAML value     | Meaning |
|----------------|---------|
| `true`         | permanently frozen |
| `false`        | permanently trainable (from step 0) |
| `N` (int, ≥ 0) | frozen for epochs `0..N-1`, thawed at start of epoch `N` (`N == 0` ≡ `false`) |

Negative ints and non-bool / non-int values raise at construction.
Parsing lives in
`brainbow/models/cosmos_2_5_common/wrapper_base.py::_resolve_freeze_dit_backbone`;
the phased-schedule machinery lives in
`brainbow/modules/cosmos_2_5_common/base.py::BaseCosmosModule.on_train_epoch_start`.
The DiT param group is included in the optimizer up front (as
zero-grad no-ops while frozen) so the thaw only flips `requires_grad`
-- the LR scheduler state is preserved verbatim.

### 1.7 Example parameter budget (Transfer, frozen base + trainable ControlNet)

A representative Transfer config — `freeze_vae_encoder: true`,
`freeze_dit_backbone: true`, `freeze_controlnet: false`,
`freeze_vae_decoder: true` — with the pretrained HF Cosmos-Transfer 2B
base + ControlNet-edge residual branch loaded:

| Component                                         | Total    | Trainable |
|---------------------------------------------------|---------:|----------:|
| VAE encoder                                       | ~50 M    | 0         |
| Base DiT (`self.dit`, frozen upper part)          | ~2.30 B  | 0         |
| ControlNet (`self.controlnet`, trainable residual)| ~0.30 B  | ~0.30 B   |
| `feature_projector`                               | ~1.1 M   | ~1.1 M    |
| `to_latent`                                       | ~1 K     | ~1 K      |
| VAE decoder body (frozen)                         | ~70 M    | 0         |
| VAE decoder shim (last up-block + norm)           | ~3 M     | ~3 M      |
| `VistaTaskHead3D` head                            | ~0.7 M   | ~0.7 M    |
| **Total**                                         | **~2.73 B** | **~0.31 B** |

(The shipped `snemi3d.yaml` instead uses `cosmospredict3d` — drop the
ControlNet row and set the base DiT trainable for its budget.)

ControlNet param count is approximate — `CosmosControlNetModel`'s
`n_controlnet_blocks` is checkpoint-specific and the model card for
`Cosmos-Transfer2.5-2B` quotes ~358 M parameters across the residual
branch. Use `model.get_num_parameters(trainable_only=True)` for the
exact number after construction.

The `_fallback_down` module (`brainbow/modules/cosmos_transfer_2_5/base.py`:
73-74) is only active when no HF VAE is loaded — in the pretrained path it is
frozen and contributes zero trainable params.

### 1.8 Practical training implications

- The frozen-base + trainable-ControlNet recipe above trains **~0.31 B
  params**.  One forward still runs the full ~2.6 B-param base + control
  stack, but AdamW state and grads are bounded by the trainable subset,
  so memory pressure is closer to a ~300 M-param fine-tune.  The shipped
  `cosmospredict3d` recipe instead full-fine-tunes the 2B base DiT (no
  ControlNet) under DDP.
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
   └─ head  (VistaTaskHead3D, 64 → HEAD_CHANNELS=16)        ≈ 0.7 M params
```

No VAE and no DiT — the SegResNetDS2 backbone does both downsampling and
upsampling internally.

### 2.2 Channel map

| Stage                        | Channels |
|------------------------------|----------|
| Input (EM)                   | **1**    |
| `init_filters`               | **64** (default; MONAI pretrained weights require **48**) |
| Backbone output / head input | **`feature_size` = 64** |
| `head`                       | `head_channels = HEAD_CHANNELS = 16` = aff(14) + sem(1) + raw(1). |

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

Vista and Cosmos expose the same affinity + sem + raw head
(`HEAD_CHANNELS`).  The only difference is the backbone / decoder that
produces the feature map before the head.

### 2.6 Rough parameter budget

| Component         | Params    |
|-------------------|----------:|
| SegResNetDS2 (64) | ~30-45 M  |
| Affinity + sem + raw head | ~0.7 M |
| **Total**         | **~35 M** |

Running Vista is roughly **70× cheaper per step** than a full-unfrozen
Cosmos-2B and ~5× cheaper than a Cosmos frozen warm-up. Reasonable for
local iteration and debugging.

---

## 3. Choosing a backbone

| Use case                                               | Recommended wrapper |
|--------------------------------------------------------|---------------------|
| Shipped default (16B omni, FSDP)                       | Cosmos 3 (Nano)     |
| 2B affinity baseline, DDP                              | Cosmos-Predict      |
| ControlNet conditioning on top of the base DiT         | Cosmos-Transfer     |
| Fast local dev / debugging on a single GPU             | Vista               |

The `configs/default.yaml` and `configs/snemi3d.yaml` files default to
`model.type: cosmos3nano3d`; override to `cosmospredict3d` (or use the
flattened `cosmospredict3d.yaml`) or `vista3d` to train a different wrapper.
