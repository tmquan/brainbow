# brainbow

A PyTorch Lightning infrastructure for **spatially-coloured (brainbow-style)
instance segmentation** of 3-D connectomics volumes, adapted from the
`neurons` research codebase and built on top of NVIDIA's
Cosmos-Transfer2.5 video-diffusion backbone (DiT + VAE).

## What it does

For every connected-component label `> 0` in a volumetric segmentation,
`brainbow` builds a **16-channel per-voxel target** directly from the
label volume + raw EM image -- no learnable target parameters:

|   channels  | meaning                                                                   |
|-------------|---------------------------------------------------------------------------|
|    0        | **raw** := raw image intensity at the voxel                               |
|    1 - 3    | RGB := normalised (z, y, x) of the instance's **min** (bounding-box min)  |
|    4 - 6    | RGB := normalised (z, y, x) of the instance's **avg** (centroid)          |
|    7 - 9    | RGB := normalised (z, y, x) of the instance's **max** (bounding-box max)  |
|   10 - 15   | **aff** := binary face-affinity to 6 neighbours in Z-Y-X order (T, B, U, D, L, R) |

Each coordinate is divided by the patch dimensions `(D, H, W)` so the
nine localisation channels live in `[0, 1]` regardless of anisotropy
or patch size.  The six affinity channels use **SAME / replicate
padding** at the crop boundary (boundary voxels are self-connected,
`aff = 1`) so every voxel has a well-defined target without masking.
The target map is computed in a single vectorised pass (no Python
loops over voxels): SciPy's `find_objects` + `np.bincount` on CPU,
`torch.scatter_reduce_` on CUDA.

The model is a `CosmosTransfer3DWrapper` with a dedicated 16-channel
`"brainbow"` head attached after the shared VAE-decoder refinement
stack (the existing `semantic`, `instance`, and `geometry` heads remain
available and can be combined via weighted sums in `CombinedLoss`).

## Layout

```
brainbow/
├── configs/             # Hydra configs; see configs/boundary.yaml
├── brainbow/            # importable package
│   ├── losses/          # SemanticLoss, InstanceLoss, GeometryLoss,
│   │   └── boundary.py  #   BoundaryLoss + build_boundary_target
│   ├── models/          # CosmosTransfer3DWrapper, Vista3DWrapper
│   ├── modules/         # Lightning modules (Cosmos-Transfer3D, Vista3D)
│   ├── datasets/        # SNEMI3D, MICRONS, Neurons (base: CircuitDataset)
│   ├── datamodules/     # matching LightningDataModule wrappers
│   ├── transforms/      # MONAI-compatible 3-D volumetric augmentations
│   ├── inference/       # soft-meanshift / HDBSCAN / spatial_cc clusterers
│   ├── preprocessors/   # TIFF / HDF5 / NRRD / NIfTI loaders
│   ├── metrics/         # ARI, AMI, VOI, TED, Dice, IoU
│   └── visualizer/      # lightweight in-browser volume viewer
├── scripts/train.py
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Install

```bash
pip install -e ".[cosmos,dev]"
# optional: RAPIDS GPU clustering
pip install -e ".[gpu-cu13]" --extra-index-url https://pypi.nvidia.com
```

## Train

```bash
# Plain SNEMI3D run with the standard three-head recipe:
python scripts/train.py --config-name snemi3d

# Turn on the boundary head (16-channel: raw + instance-colour + face-affinity):
python scripts/train.py --config-name boundary

# DDP, custom batch size:
python scripts/train.py --config-name boundary data.batch_size=4 training.devices=4
```

### GPU memory: avoiding slow OOM drift on long runs

On long DDP runs (especially with `freeze_dit_backbone: <N>` phased
unfreeze, `compile: max-autotune`, or `max_hard_pairs: 0`) the PyTorch
caching allocator's reserved pool tends to creep upward over hours
even though live tensors are stable.  Two settings make the
difference between "stable at 90 %" and "OOM at epoch 30":

```bash
# 1.  Enable expandable allocator segments BEFORE launching python.
#     Mitigates fragmentation; near-zero runtime cost.  Read at CUDA
#     init, so it must be exported (cannot be applied in-process).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2.  Empty the cache around validation (callback already on by
#     default in snemi3d.yaml; turn on for custom configs):
#         callbacks.cuda_empty_cache_before_val: true
#     This now empties on BOTH sides of validation so the val-time
#     high-water mark does not stay reserved in the training pool.

python scripts/train.py --config-name snemi3d
```

Watch the trajectory in TensorBoard under the `cuda_memory/*` tags
(emitted by `CudaMemoryLoggerCallback`, on by default):

| Pattern                                                | Diagnosis                                                     |
|--------------------------------------------------------|---------------------------------------------------------------|
| `allocated_gb` flat, `reserved_gb` rising              | fragmentation — set `PYTORCH_CUDA_ALLOC_CONF` as above.       |
| `allocated_gb` and `reserved_gb` both rising           | tensor leak — inspect callbacks (image_logger, custom hooks). |
| sawtooth coupled to val epochs                         | val peak polluting train pool — enable the callback above.    |
| sudden step at the epoch boundary set by `freeze_dit_backbone` | DiT unfreeze added grads + AdamW state; expected. Enable `model.gradient_checkpointing: true` for headroom. |

## Loss

```python
from brainbow.losses import BoundaryLoss, build_boundary_target

loss_fn = BoundaryLoss(
    loss_loc="smooth_l1",    # regression loss on the 9 localisation channels
    loss_raw="l1",           # regression loss on channel 0 (raw intensity)
    weight_min=1.0,
    weight_avg=1.0,
    weight_max=1.0,
    weight_raw=1.0,
    weight_aff=1.0,          # soft-Dice on sigmoid(face-affinity logits)
    foreground_only_loc=True,
)

# prediction: [B, 16, D, H, W]  |  labels: [B, D, H, W]  |  image: [B, D, H, W]
out = loss_fn(prediction, labels, image)
# out -> {"loss", "raw", "min", "avg", "max", "aff"}
```

## Tests

```bash
pytest tests/ -q
```

## License

MIT.  See `LICENSE`.
