# Datasets

Every dataset brainbow trains on is **3-D electron-microscopy (EM)** with
dense **instance** (neuron-id) labels, stored on disk in one shared
convention and pulled in by the datamodule. This doc covers what each
dataset is, its native resolution, and the exact script that downloads
it.

> Resolutions are quoted as the source reports them (usually `x × y × z`
> nm). The config's `resolution_map` uses **`(z, y, x)`** order — both are
> given below so there's no ambiguity.

## At a glance

| Dataset    | Tissue / modality                         | Native res (x,y,z nm) | `resolution_map` (z,y,x) | Download script             | Data root        | Config key (`data.dataset`) |
| ---------- | ----------------------------------------- | --------------------- | ------------------------ | --------------------------- | ---------------- | --------------------------- |
| SNEMI3D    | Mouse S1 cortex, ssEM (AC3/AC4)           | 6 × 6 × 30            | `AC: [30,6,6]`           | `download_snemi3d.py`       | `data/SNEMI3D`   | `snemi3d`                   |
| Neurons    | Mouse S1 cortex, ssEM (Kasthuri cylinder) | 6 × 6 × 30            | `neurons: [30,6,6]`      | `download_snemi3d.py`       | `data/SNEMI3D`   | `neurons`                   |
| MICrONS    | Mouse V1 cortex, ssEM (minnie65)          | 8 × 8 × 40            | `minnie65: [40,8,8]`     | `download_microns.py`       | `data/MICRONS`   | `microns`                   |
| CREMI3D    | *Drosophila* brain, ssTEM (A/B/C)         | 4 × 4 × 40            | `cremi3d: [40,4,4]`      | `download_cremi3d.py`       | `data/CREMI3D`   | `cremi3d`                   |
| FIB253D    | *Drosophila* medulla, FIB-SEM (FIB-25)    | 8 × 8 × 8 (isotropic) | `fib253d: [8,8,8]`       | `download_fib253d.py`       | `data/FIB253D`   | `fib253d`                   |
| *(aux)* Zenodo 582636 | Rice grains, X-ray µCT (smoke test) | n/a              | n/a                      | `download_zenodo_582636.py` | `data/zenodo582636` | n/a                      |

All scripts live in `scripts/` and write the on-disk convention described
in [On-disk convention](#on-disk-convention). The multi-dataset
"foundation" recipe (`configs/cosmos3nano3d.yaml`) mixes SNEMI3D,
Neurons, MICrONS, CREMI3D, and FIB253D in a single run by listing volumes
from several `data/<root>` directories under one datamodule.

---

## On-disk convention

The 3-D training path (`LazyVolDataset`, used whenever
`slice_mode: false` + `patch_size`) reads patches **on demand** from
HDF5, so every volume is stored as **two separate `.h5` files**:

```
<root>/<name>_volume.h5         # EM intensity   (uint8/float)
<root>/<name>_segmentation.h5   # instance ids   (int/uint, 0 = background)
```

- Dataset key inside each file: **`main`** (the loader also falls back to
  `data`/`raw`/`volume`/`image`/`label`).
- Axis order: **`[Z, Y, X]`** (z = section axis).
- A volume is referenced in YAML as `{vol: <name>_volume, seg: <name>_segmentation, root: <dir>}`.
- `LazyVolDataset` finds files by base name with extensions
  `.h5/.hdf5/.tif/.tiff` — so a packed/nested HDF5 (e.g. CREMI's native
  `.hdf` with `volumes/raw` + `volumes/labels/neuron_ids`) must be
  **converted** first; the CREMI/FIB scripts do this at download time.

The dataset/datamodule classes (`SNEMI3DDataset`, `MICRONSDataset`,
`CREMI3DDataset`, `FIB253DDataset`, `NeuronsDataset` and their
`*DataModule`s) only differ in metadata; the CREMI3D and FIB253D leaves
are thin metadata subclasses of `MICRONSDataset`/`MICRONSDataModule`.

---

## SNEMI3D (AC3 / AC4)

- **What:** The SNEMI3D challenge crops from Kasthuri et al. 2015 mouse
  somatosensory cortex (ssEM). `AC4` = train (1024 × 1024 × 100, EM +
  labels); `AC3` = test (1024 × 1024 × 100, **EM only** — labels were
  never publicly released).
- **Resolution:** 6 × 6 × 30 nm (anisotropic).
- **Source:** `snemi.zip` (rhoana / Zenodo). AC3/AC4 sit at Y ≈ 5440 in
  the `kasthuri11` volume, outside the GCS ground-truth cylinder.
- **Citation:** Kasthuri, N. et al. (2015), *Saturated Reconstruction of
  a Volume of Neocortex*, Cell 162(3):648-661.

```bash
python scripts/download_snemi3d.py --source snemi      # AC3 EM + AC4 EM/labels
python scripts/download_snemi3d.py --link /scratch/SNEMI3D   # or symlink existing
```

Files land in `data/SNEMI3D/` (e.g. `AC4_inputs`, `AC4_labels`).

---

## Neurons (Kasthuri annotated cylinder)

- **What:** The densely-annotated cylinder from the **same** Kasthuri
  2015 volume, fetched from the public `kasthuri2011` Google bucket. A
  single large crop `5000 × 2900 × 300` at start `(x=3000, y=7200, z=950)`
  inside the annotated region (X≈3000–8000, Y≈7200–10100, Z≈950–1250).
- **Resolution:** 6 × 6 × 30 nm (as downloaded / as used in the config).
- **Source:**
  `gs://neuroglancer-public-data/kasthuri2011/{image_color_corrected, ground_truth}`.

```bash
python scripts/download_snemi3d.py --source neurons
# custom crop:
python scripts/download_snemi3d.py --source neurons --start 3000 7200 950 --size 5000 2900 300
# everything (SNEMI3D + neurons):
python scripts/download_snemi3d.py --source all
```

Files land in `data/SNEMI3D/` (config volume
`neurons_5000x2900x300_x3000_y7200_z950_{volume,segmentation}`).

---

## MICrONS (minnie65)

- **What:** IARPA MICrONS mouse primary visual cortex (V1), ~1 mm³,
  ~120k neurons. We use representative sub-volumes for training.
- **Resolution:** **8 × 8 × 40 nm** — this is the EM *imagery*
  resolution (mip 0 of the released precomputed bucket). The often-quoted
  **4 × 4 × 40 nm** is the *annotation/coordinate frame*, not the image
  voxel size.
- **Segmentation versions:** `v117`, `v343`, `v943`, `v1300` (default,
  latest, Jan 2025).
- **Splits:** 12 pre-defined `4096 × 4096 × 800` crops (10 train + 2
  test) at disjoint XY positions / cortical depths; file names encode the
  origin, e.g.
  `minnie65_mip0_4096x4096x800_x50000_y60000_z16000_volume.h5`.
- **Source:** AWS / GCS public buckets via `cloud-volume`
  (`.../iarpa_microns/minnie/minnie65/em`).
- **Citation:** MICrONS Consortium (2021), bioRxiv 2021.07.28.454025.

```bash
python scripts/download_microns.py --split                       # 10 train + 2 test, v1300
python scripts/download_microns.py --size 4096 4096 800 --seg-version 1300   # custom
python scripts/download_microns.py --seg-version all             # all 4 seg versions
```

Files land in `data/MICRONS/`. Crop size guide (mip0, uint8 EM +
uint64 seg): `512³` ≈ 1.1 GB, `1024³` ≈ 9 GB, `2048³` ≈ 72 GB,
`4096×4096×800` ≈ tens of GB per crop.

---

## CREMI3D

- **What:** CREMI (MICCAI 2016), adult *Drosophila* brain ssTEM.
  - **A, B, C** — labelled TRAINING volumes (`1250 × 1250 × 125`, dense
    neuron ids) → `train_volumes`.
  - **A+, B+, C+** — padded TEST volumes; public EM only (challenge
    withholds the labels) → converted **image-only**.  They are **not**
    listed in any config's `test_volumes` (no GT = no metrics); run **blind
    inference** on them separately.
- **Resolution:** 4 × 4 × 40 nm (anisotropic; 10:1 z:xy).
- **Source:** `https://cremi.org/static/data/sample_{A,B,C}_20160501.hdf`
  (raw + labels packed in one nested `.hdf`).
- **Conversion:** the script downloads the official `.hdf` and writes the
  brainbow convention (`cremi3d_sample_<X>_{volume,segmentation}.h5`, key
  `main`, `[Z,Y,X]`), mapping CREMI's "no-data" marker to background.
- **Citation:** Funke, Saalfeld, Bock, Turaga, Perlman — CREMI Challenge,
  https://cremi.org/.

```bash
# downloads + converts all six (A,B,C labelled + A+,B+,C+ image-only test)
python scripts/download_cremi3d.py --out-dir data/CREMI3D
# training only:
python scripts/download_cremi3d.py --out-dir data/CREMI3D --samples A B C
# reuse already-downloaded .hdf (skip the network):
python scripts/download_cremi3d.py --out-dir data/CREMI3D --hdf-dir /scratch/CREMI3D
```

Files land in `data/CREMI3D/` (`cremi3d_sample_A_volume`, … and the
image-only `cremi3d_sample_A+_volume`, …).

---

## FIB253D (FIB-25)

- **What:** Janelia FlyEM 7-column *Drosophila* medulla FIB-SEM
  reconstruction. Dense neuron instance labels.
- **Resolution:** **8 × 8 × 8 nm isotropic** at mip 0 (doubles per mip,
  7 mip levels). Full volume `6446 × 6643 × 8090` voxels (x,y,z).
- **Segmentation coverage:** the ground truth covers only **~8.65%** of
  the full volume; the labeled bounding box is `x[1856:5024]
  y[1664:4288] z[1472:8000]` (≈ 25 × 21 × 52 µm) and is ~55% filled.
  Fully-dense crops top out around `1024³` (≈100% fg); the chosen
  **primary training core is `1536³` at origin `(2304, 2048, 6144)`**
  (≈12.3 µm, **~99.7% foreground**).
- **Source:** `gs://neuroglancer-public-data/flyem_fib-25/{image, ground_truth}`
  (Neuroglancer precomputed) via `cloud-volume`.
- **Citation:** Takemura, S. et al. (2015), PNAS 112(44):13711-13716,
  doi:10.1073/pnas.1509820112.

Recommended: fetch the native cube **once**, then generate all variants
locally with `--from-local` (no re-download).

```bash
# 1. primary 1536^3 dense core (native 8 nm isotropic)
python scripts/download_fib253d.py --out-dir data/FIB253D --name fib253d \
    --origin 2304 2048 6144 --size 1536 1536 1536 --mip 0

# 2. isotropic orientation variants (thin axis z / y / x) from the local cube
python scripts/download_fib253d.py --from-local \
    data/FIB253D/fib253d_8nm_x2304_y2048_z6144_volume.h5 \
    --name fib253d --orientations z y x

# 3. anisotropic 32 nm copies, all z-stride-4 phase offsets (p0..p3)
python scripts/download_fib253d.py --from-local \
    data/FIB253D/fib253d_8nm_x2304_y2048_z6144_volume.h5 \
    --name fib253d --z-stride 4
```

Files land in `data/FIB253D/`
(`fib253d_8nm_x2304_y2048_z6144_{volume,segmentation}`). `--size` is
clamped to the volume bounds; the script loads the whole crop into RAM,
so it is built for crops, not the full petavoxel volume (the full image
is ~346 GB / seg ~2.8 TB at mip 0).

---

## *(Auxiliary)* Zenodo 582636 — rice grains

Not connectomics: an X-ray µCT volume of packed rice grains (689 TIFF
slices, ~3.93 GB). Useful as a non-EM **3-D instance-segmentation smoke
test** for the data-loading + clustering pipeline (densely-packed,
touching objects easy to verify by eye). MD5-verified, resumable.

```bash
python scripts/download_zenodo_582636.py            # downloads + verifies
```

The script doubles as a template for any Zenodo record (change
`RECORD_ID`).

---

## How datasets enter a training run

1. **Download** with the script(s) above into the matching `data/<root>`.
2. **List volumes** in the config under `data.train_volumes` /
   `val_volumes` / `test_volumes` as `{vol, seg, root}` triples (the
   combine recipe mixes roots in one run).
3. **Per-dataset native resolution** goes in `data.resolution_map`
   (`(z, y, x)` nm), keyed by a **prefix of the volume name** (e.g.
   `fib253d`, `cremi3d`, `minnie65`, `neurons`, `AC`). It is consumed
   only by the `resolution_zoom` augmentation.
4. **Resolution policy** (see `configs/cosmos3nano3d.yaml`):
   `resolution_zoom_mode: union` with `resolution_zoom_prob: 0.5`. Each
   anisotropic patch is resampled to a **random target inside the shared
   union envelope** `z ∈ [30,40]`, `xy ∈ [4,8]` nm (the union of all native
   resolutions), with the `z` and `xy` targets sampled **independently**
   (log-uniform). Because the affinity offsets are defined in **voxels**,
   harmonising onto a common resolution space gives them a *consistent
   physical meaning across datasets*, and the random target doubles as
   scale/anisotropy augmentation.
   - **Isotropic volumes (FIB-25 `8×8×8`) are skipped** by the union
     resample (`z==y==x` in `resolution_map`): upsampling their fine 8 nm z
     to the 30–40 nm envelope would be a ~5× downsample, blowing up the
     pre-zoom safe-crop and destroying their isotropy. FIB's anisotropic
     contribution comes from the **z-strided `[32,8,8]` copy** (key
     `fib253d_z32`, `--z-stride 4`, all phase offsets), which participates
     normally. The isotropic FIB variants
     are augmented by their octahedral orientation copies instead (see the
     FIB253D section / `download_fib253d.py --orientations`).
   - Safe-crop (pre-zoom read) is bounded at ≈ `(107, 512, 512)`; the 512
     is driven by CREMI's 4 nm → 8 nm 2× downsample.
   - Legacy `resolution_zoom_mode: ratio` (anisotropy-preserving, single
     scale factor) is still available.

For the recipe to add a brand-new dataset (preprocessor → leaf dataset →
leaf datamodule → YAML), see [`CONTRIBUTING.md`](./CONTRIBUTING.md).
