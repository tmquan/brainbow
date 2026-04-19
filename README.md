# brainbow

A PyTorch Lightning infrastructure for **spatially-coloured (brainbow-style)
instance segmentation** of 3-D connectomics volumes, adapted from the
`neurons` research codebase and built on top of NVIDIA's
Cosmos-Transfer2.5 video-diffusion backbone (DiT + VAE).

## What it does

For every connected-component label `> 0` in a volumetric segmentation,
`brainbow` builds a **10-channel per-voxel target** directly from the
label volume + raw EM image -- no learnable target parameters:

|   channels  | meaning                                                                |
|-------------|------------------------------------------------------------------------|
|    0        | **rawval** := raw image intensity at the voxel                         |
|    1 - 3    | RGB := normalised (z, y, x) of the instance's **minloc** (bounding-box min) |
|    4 - 6    | RGB := normalised (z, y, x) of the instance's **avgloc** (centroid)    |
|    7 - 9    | RGB := normalised (z, y, x) of the instance's **maxloc** (bounding-box max) |

Each coordinate is divided by the patch dimensions `(D, H, W)` so every
channel lives in `[0, 1]` regardless of anisotropy or patch size.  The
target map is computed in a single vectorised pass (no Python loops
over voxels): SciPy's `find_objects` + `np.bincount` on CPU,
`torch.scatter_reduce_` on CUDA.

The model is a `CosmosTransfer3DWrapper` with a dedicated 10-channel
`"brainbow"` head attached after the shared VAE-decoder refinement
stack (the existing `semantic`, `instance`, and `geometry` heads remain
available and can be combined via weighted sums in `CombinedLoss`).

## Layout

```
brainbow/
├── configs/             # Hydra configs; see configs/brainbow.yaml
├── brainbow/            # importable package
│   ├── losses/          # SemanticLoss, InstanceLoss, GeometryLoss,
│   │   └── brainbow.py  #   BrainbowLoss + build_brainbow_target
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

# Turn on the brainbow head (10-channel instance-colour + raw-value target):
python scripts/train.py --config-name brainbow

# DDP, custom batch size:
python scripts/train.py --config-name brainbow data.batch_size=4 training.devices=4
```

## Loss

```python
from brainbow.losses import BrainbowLoss, build_brainbow_target

loss_fn = BrainbowLoss(
    loss_loc="smooth_l1",    # regression loss on the 9 localisation channels
    loss_raw="l1",           # regression loss on channel 9 (raw intensity)
    weight_minloc=1.0,
    weight_avgloc=1.0,
    weight_maxloc=1.0,
    weight_rawval=1.0,
    foreground_only_loc=True,
)

# prediction: [B, 10, D, H, W]  |  labels: [B, D, H, W]  |  image: [B, D, H, W]
out = loss_fn(prediction, labels, image)
# out -> {"loss", "minloc", "avgloc", "maxloc", "rawval"}
```

## Tests

```bash
pytest tests/ -q
```

## License

MIT.  See `LICENSE`.
