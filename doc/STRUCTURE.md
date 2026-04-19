brainbow
├── LICENSE
├── README.md
├── configs
│   ├── brainbow.yaml
│   ├── default.yaml
│   └── snemi3d.yaml
├── data
│   ├── MICRONS
│   │   ├── minnie65_mip0_4096x4096x800_x110000_y60000_z18000_v1300_segmentation.h5
│   │   ├── minnie65_mip0_4096x4096x800_x110000_y60000_z18000_volume.h5
│   │   ├── minnie65_mip0_4096x4096x800_x110000_y60000_z18000_volume.h5.norm.json
│   │   ├── minnie65_mip0_4096x4096x800_x140000_y80000_z17500_v1300_segmentation.h5
│   │   ├── minnie65_mip0_4096x4096x800_x140000_y80000_z17500_volume.h5
│   │   ├── minnie65_mip0_4096x4096x800_x140000_y80000_z17500_volume.h5.norm.json
│   │   ├── minnie65_mip0_4096x4096x800_x50000_y60000_z16000_v1300_segmentation.h5
│   │   ├── minnie65_mip0_4096x4096x800_x50000_y60000_z16000_volume.h5
│   │   ├── minnie65_mip0_4096x4096x800_x50000_y60000_z16000_volume.h5.norm.json
│   │   ├── minnie65_mip0_4096x4096x800_x70000_y90000_z17000_v1300_segmentation.h5
│   │   ├── minnie65_mip0_4096x4096x800_x70000_y90000_z17000_volume.h5
│   │   ├── minnie65_mip0_4096x4096x800_x70000_y90000_z17000_volume.h5.norm.json
│   │   ├── minnie65_mip0_4096x4096x800_x80000_y70000_z16500_v1300_segmentation.h5
│   │   ├── minnie65_mip0_4096x4096x800_x80000_y70000_z16500_volume.h5
│   │   └── minnie65_mip0_4096x4096x800_x80000_y70000_z16500_volume.h5.norm.json
│   └── SNEMI3D
│       ├── AC3_inputs.h5
│       ├── AC4_inputs.h5
│       ├── AC4_inputs.h5.norm.json
│       ├── AC4_labels.h5
│       ├── neurons_5000x2900x300_x3000_y7200_z950_segmentation.h5
│       ├── neurons_5000x2900x300_x3000_y7200_z950_volume.h5
│       └── neurons_5000x2900x300_x3000_y7200_z950_volume.h5.norm.json
├── doc
│   └── STRUCTURE.md
├── brainbow
│   ├── __init__.py
│   ├── callbacks
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   └── tensorboard.py
│   ├── datamodules
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── microns.py
│   │   ├── neurons.py
│   │   └── snemi3d.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── lazy.py
│   │   ├── microns.py
│   │   ├── neurons.py
│   │   └── snemi3d.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── clusterer.py
│   │   └── sliding_window.py
│   ├── losses
│   │   ├── __init__.py
│   │   ├── brainbow.py        # 10-ch (min/avg/max)loc-RGB + rawval
│   │   ├── combined.py
│   │   ├── geometry.py
│   │   ├── instance.py
│   │   └── semantic.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── instance.py
│   │   └── semantic.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cosmos_transfer_2_5/        # Cosmos-Transfer 2.5 3-D wrapper (split)
│   │   │   ├── __init__.py             # re-exports CosmosTransfer3DWrapper
│   │   │   ├── decoder.py              # feature projector + VAE decoder adapter
│   │   │   ├── hf_loader.py            # rank-aware HF snapshot download
│   │   │   ├── layers.py               # shared primitives
│   │   │   ├── standalone_dit.py       # random-init DiT fallback
│   │   │   ├── variants.py             # 2B / 14B variant registry
│   │   │   └── wrapper.py              # CosmosTransfer3DWrapper
│   │   └── vista/                      # Vista3D wrapper + shared head + prompt I/O
│   │       ├── __init__.py             # re-exports Vista3DWrapper, VistaTaskHead3D, PointPromptEncoder, sample_point_prompts
│   │       ├── heads.py                # VistaTaskHead3D (MONAI UnetrBasicBlock)
│   │       ├── hf_loader.py            # MONAI/VISTA3D-HF encoder download + partial-load
│   │       ├── point_prompt_encoder.py # PointPromptEncoder (proofread/interactive conditioning)
│   │       ├── point_sampling.py       # sample_point_prompts (GT masks -> click-point dict)
│   │       └── wrapper.py              # Vista3DWrapper
│   ├── modules
│   │   ├── __init__.py
│   │   ├── base.py                          # BaseCircuitModule (shared loop + logging)
│   │   ├── cosmos_transfer_2_5/             # Cosmos-Transfer 2.5 Lightning modules
│   │   │   ├── __init__.py                  # re-exports Base/Concrete modules
│   │   │   ├── base.py                      # BaseCosmosModule (freeze + optim split)
│   │   │   └── module.py                    # CosmosTransfer3DModule
│   │   └── vista/                           # Vista3D Lightning modules
│   │       ├── __init__.py                  # re-exports Base/Concrete modules
│   │       ├── base.py                      # BaseVistaModule
│   │       └── module.py                    # Vista3DModule
│   ├── preprocessors
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hdf5.py
│   │   ├── nfty.py
│   │   ├── nrrd.py
│   │   └── tiff.py
│   ├── transforms
│   │   ├── __init__.py
│   │   ├── covariance.py
│   │   ├── defect.py
│   │   ├── direction.py
│   │   ├── edt.py
│   │   ├── find_boundaries.py
│   │   ├── label.py
│   │   ├── missing_section.py
│   │   ├── rand_crop_foreground.py
│   │   ├── rand_transpose_xy.py
│   │   └── resolution_zoom.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── clustering.py
│   │   ├── io.py
│   │   ├── manifold.py
│   │   └── parallel.py
│   └── visualizer
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── static
│       │   ├── app.js
│       │   ├── index.html
│       │   ├── style.css
│       │   └── volume_renderer.js
│       └── volume_loader.py
├── pyproject.toml
├── requirements.txt
├── scripts
│   ├── download_microns.py
│   ├── download_snemi3d.py
│   ├── download_zenodo_582636.py
│   └── train.py
└── tests
    ├── __init__.py
    ├── test_brainbow_loss.py
    ├── test_datamodules.py
    ├── test_datasets.py
    ├── test_losses.py
    ├── test_preprocessors.py
    └── test_utils.py
