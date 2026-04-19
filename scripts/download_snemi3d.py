#!/usr/bin/env python
"""
Unified download script for all Kasthuri et al. 2015 data at 6×6×30 nm.

All data originates from the same mouse somatosensory cortex volume
published in Kasthuri et al., Cell 2015.  This script can download:

  ac3+ac4   SNEMI3D challenge data from snemi.zip (rhoana/Zenodo)
  neurons   Full annotated cylinder from GCS (5000×2900×300)

SNEMI3D challenge data (AC3 / AC4)
-----------------------------------
Downloaded as snemi.zip from rhoana.rc.fas.harvard.edu (or Zenodo).
  AC4 = train  (1024×1024×100, EM + 401-ID labels)
  AC3 = test   (1024×1024×100, EM only — labels never released)

These ROIs sit at Y≈5440 in the kasthuri11 volume, *outside* the
ground_truth annotation cylinder, so GCS ground_truth is empty there.
The AC3/AC4 segmentations were separate OCP annotation tokens.

Verified coordinates (OCP scale 1 = GCS mip0, 6×6×30 nm):
  AC4: x=[4400,5424], y=[5440,6464], z=[1099,1199]  (Z reversed vs snemi.zip)
  Spearman correlation between GCS EM and snemi.zip: 0.96+

Neurons (GCS)
-------------
  EM:  gs://neuroglancer-public-data/kasthuri2011/image_color_corrected
  Seg: gs://neuroglancer-public-data/kasthuri2011/ground_truth

  Full volume: 10752 × 13312 × 1850 voxels at 6 × 6 × 30 nm.
  Annotated cylinder: X≈3000–8000, Y≈7200–10100, Z≈950–1250
  Single crop: start=(3000, 7200, 950) size=(5000, 2900, 300)

Usage
-----
    # Download SNEMI3D challenge data (AC3 EM + AC4 EM/labels)
    python scripts/download_snemi3d.py --source snemi

    # Download neurons full annotated cylinder from GCS
    python scripts/download_snemi3d.py --source neurons

    # Download everything
    python scripts/download_snemi3d.py --source all

    # Symlink from existing local directory
    python scripts/download_snemi3d.py --link /scratch/SNEMI3D

    # Probe GCS volume metadata
    python scripts/download_snemi3d.py --probe

    # Custom neurons crop
    python scripts/download_snemi3d.py --source neurons \\
        --start 3000 7200 950 --size 5000 2900 300
"""

import argparse
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# SNEMI3D challenge data (snemi.zip)
# ---------------------------------------------------------------------------
SNEMI_ZIP_URL = "http://rhoana.rc.fas.harvard.edu/dataset/snemi.zip"

# Mapping from snemi.zip TIF files -> output H5 files
SNEMI_FILE_MAP = {
    "image/train-input.tif": "AC4_inputs.h5",
    "seg/train-labels.tif": "AC4_labels.h5",
    "image/test-input.tif": "AC3_inputs.h5",
}

# ---------------------------------------------------------------------------
# GCS paths for Neurons
# ---------------------------------------------------------------------------
EM_PATH = "gs://neuroglancer-public-data/kasthuri2011/image_color_corrected"
SEG_PATH = "gs://neuroglancer-public-data/kasthuri2011/ground_truth"

# Neurons pre-defined splits
_NEURONS_SIZE = (5000, 2900, 300)
NEURONS_SPLITS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "train01": {"start": (3000, 7200, 950), "size": _NEURONS_SIZE},
}

VALID_SOURCES = ["snemi", "neurons", "all"]

# Files expected for --link mode
LINK_FILES = [
    "AC3_inputs.h5",
    "AC4_inputs.h5", "AC4_labels.h5",
    "AC4_thin_inputs.h5", "AC4_thin_labels.h5",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def probe_volume(cloud_path: str, label: str) -> None:
    """Print volume metadata (dimensions, mip levels, dtype)."""
    from cloudvolume import CloudVolume

    print(f"\n  {label}: {cloud_path}")
    try:
        for mip in range(6):
            try:
                vol = CloudVolume(cloud_path, mip=mip, use_https=True)
                res = vol.resolution
                shape = vol.shape[:3]
                dt = vol.dtype
                print(f"    mip{mip}: {shape[0]}×{shape[1]}×{shape[2]}  "
                      f"res={res[0]}×{res[1]}×{res[2]} nm  dtype={dt}")
            except Exception:
                break
    except Exception as e:
        print(f"    ERROR: {e}")


def download_subvolume(
    cloud_path: str,
    bbox_start: Tuple[int, int, int],
    bbox_size: Tuple[int, int, int],
    mip: int = 0,
) -> np.ndarray:
    """Download a sub-volume via CloudVolume.  Returns shape (Z, Y, X)."""
    from cloudvolume import CloudVolume

    vol = CloudVolume(cloud_path, mip=mip, use_https=True,
                      fill_missing=True, bounded=False)
    x0, y0, z0 = bbox_start
    sx, sy, sz = bbox_size
    data = vol[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz]
    arr = np.squeeze(data)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    return arr


def save_h5(arr: np.ndarray, path: Path) -> None:
    """Save array to gzip-compressed HDF5 with key 'main'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("main", data=arr, compression="gzip")


def save_tiff(arr: np.ndarray, path: Path) -> None:
    """Save array to TIFF."""
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def make_neurons_name(
    suffix: str,
    crop_size: Tuple[int, int, int],
    start: Tuple[int, int, int],
) -> str:
    """Build coordinate-based file stem (no extension) for neurons crops."""
    x, y, z = start
    sx, sy, sz = crop_size
    return f"neurons_{sx}x{sy}x{sz}_x{x}_y{y}_z{z}_{suffix}"


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------
def download_snemi(out_dir: Path) -> None:
    """Download SNEMI3D challenge data (AC3 EM + AC4 EM/labels) from snemi.zip."""
    existing = [out_dir / h5 for h5 in SNEMI_FILE_MAP.values() if (out_dir / h5).exists()]
    if len(existing) == len(SNEMI_FILE_MAP):
        print("\n--- SNEMI3D (AC3 + AC4) ---")
        print("  All files already exist:")
        for h5 in SNEMI_FILE_MAP.values():
            print(f"    {h5}")
        return

    print("\n--- SNEMI3D (AC3 + AC4) ---")
    print(f"  Source: {SNEMI_ZIP_URL}")

    zip_path = out_dir / "snemi.zip"
    if not zip_path.exists():
        print(f"  Downloading snemi.zip ...")
        urllib.request.urlretrieve(SNEMI_ZIP_URL, zip_path)
        size_mb = zip_path.stat().st_size / 1e6
        print(f"    saved: snemi.zip ({size_mb:.0f} MB)")
    else:
        print(f"  snemi.zip already exists")

    print(f"  Extracting and converting to H5 ...")
    import tifffile

    with zipfile.ZipFile(zip_path, "r") as zf:
        for tif_name, h5_name in SNEMI_FILE_MAP.items():
            h5_path = out_dir / h5_name
            if h5_path.exists():
                print(f"    {h5_name}: SKIP (exists)")
                continue

            with zf.open(tif_name) as f:
                with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
                    tmp.write(f.read())
                    tmp.flush()
                    arr = tifffile.imread(tmp.name)

            save_h5(arr, h5_path)
            n_ids = len(np.unique(arr))
            print(f"    {h5_name}: shape={arr.shape} dtype={arr.dtype} "
                  f"unique={n_ids}")

    # Create thin variants (AC4 is already 100 slices so thin = same)
    for src_name, thin_name in [("AC4_inputs.h5", "AC4_thin_inputs.h5"),
                                 ("AC4_labels.h5", "AC4_thin_labels.h5")]:
        src = out_dir / src_name
        dst = out_dir / thin_name
        if dst.exists() or not src.exists():
            continue
        with h5py.File(src, "r", locking=False) as f:
            data = f["main"][:]
        save_h5(data, dst)
        print(f"    {thin_name}: shape={data.shape} (thin variant)")


def download_neurons(
    out_dir: Path,
    crops: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]],
) -> None:
    """Download neurons crops from GCS."""
    print(f"\n--- Neurons ---")
    print(f"  EM:  {EM_PATH}")
    print(f"  Seg: {SEG_PATH}")
    print()

    for label, bbox_start, bbox_size in crops:
        stem = make_neurons_name("volume", bbox_size, bbox_start)
        seg_stem = make_neurons_name("segmentation", bbox_size, bbox_start)

        print(f"  [{label}] start={bbox_start}  size={bbox_size}")

        em_h5 = out_dir / f"{stem}.h5"
        em_tiff = out_dir / f"{stem}.tiff"
        if em_h5.exists():
            print(f"    EM: SKIP (already exists: {em_h5.name})")
        else:
            print(f"    Downloading EM ...")
            em_vol = download_subvolume(EM_PATH, bbox_start, bbox_size)
            print(f"      shape: {em_vol.shape}  dtype={em_vol.dtype}")
            save_h5(em_vol, em_h5)
            save_tiff(em_vol, em_tiff)
            print(f"      saved: {em_h5.name} + {em_tiff.name}")

        seg_h5 = out_dir / f"{seg_stem}.h5"
        seg_tiff = out_dir / f"{seg_stem}.tiff"
        if seg_h5.exists():
            print(f"    Seg: SKIP (already exists: {seg_h5.name})")
        else:
            print(f"    Downloading segmentation ...")
            seg_vol = download_subvolume(SEG_PATH, bbox_start, bbox_size)
            n_ids = len(np.unique(seg_vol))
            print(f"      shape: {seg_vol.shape}  dtype={seg_vol.dtype}  "
                  f"unique: {n_ids} IDs")
            save_h5(seg_vol, seg_h5)
            save_tiff(seg_vol, seg_tiff)
            print(f"      saved: {seg_h5.name} + {seg_tiff.name}")
        print()


def link_from_existing(output_dir: Path, source_dir: Path) -> None:
    """Create symlinks from an existing data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname in LINK_FILES:
        src = source_dir / fname
        dst = output_dir / fname
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except PermissionError:
                print(f"  SKIP {fname} (cannot overwrite)")
                continue
        if src.exists():
            os.symlink(src, dst)
            print(f"  {fname} -> {src}")

    for f in sorted(source_dir.glob("neurons_*.h5")):
        dst = output_dir / f.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(f, dst)
        print(f"  {f.name} -> {f}")


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def run_probe() -> None:
    """Print metadata for the kasthuri11 volume on GCS."""
    print("=" * 60)
    print("SNEMI3D / Kasthuri 2015 — Volume Probe")
    print("=" * 60)
    probe_volume(EM_PATH, "EM (color-corrected)")
    probe_volume(SEG_PATH, "Seg (ground_truth)")

    print()
    print("GCS layers available:")
    print("  gs://neuroglancer-public-data/kasthuri2011/image")
    print("  gs://neuroglancer-public-data/kasthuri2011/image_color_corrected")
    print("  gs://neuroglancer-public-data/kasthuri2011/ground_truth")
    print()
    print("Neurons annotated cylinder (mip0 coords):")
    print("  X: ~3000–8000  Y: ~7200–10100  Z: ~950–1250")
    for name, sp in NEURONS_SPLITS.items():
        print(f"  {name}: start={sp['start']}  size={sp['size']}")
    print()
    print("AC3/AC4 (SNEMI3D challenge, from snemi.zip):")
    print("  AC4 (train): 1024×1024×100  EM + 401-ID labels")
    print("  AC3 (test):  1024×1024×100  EM only (labels never released)")
    print("  GCS coords:  x=[4400,5424] y=[5440,6464] z=[1099,1199]")
    print("  NOTE: outside ground_truth cylinder — labels only in snemi.zip")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SNEMI3D / Kasthuri 2015 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", type=str, nargs="+", default=None,
        metavar="SRC",
        help=f"What to download: {', '.join(VALID_SOURCES)} (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default="data/SNEMI3D",
        help="Output directory (default: data/SNEMI3D)",
    )
    parser.add_argument(
        "--link", type=str, default=None,
        help="Symlink from existing local directory instead of downloading",
    )
    parser.add_argument(
        "--start", type=int, nargs=3, default=None,
        help="Custom neurons crop start (X Y Z)",
    )
    parser.add_argument(
        "--size", type=int, nargs=3, default=None,
        help="Custom neurons crop size (X Y Z)",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Print volume metadata and exit",
    )
    args = parser.parse_args()

    if args.probe:
        run_probe()
        return

    out_dir = Path(args.output)

    if args.link:
        print("=" * 60)
        print("SNEMI3D — Symlink from existing directory")
        print("=" * 60)
        source = Path(args.link)
        print(f"  Source: {source}")
        print(f"  Target: {out_dir}")
        link_from_existing(out_dir, source)
        print("=" * 60)
        return

    sources = args.source or ["all"]
    if "all" in sources:
        sources = ["snemi", "neurons"]

    for s in sources:
        if s not in VALID_SOURCES:
            parser.error(
                f"Unknown source '{s}'. Choose from: {', '.join(VALID_SOURCES)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SNEMI3D / Kasthuri 2015 — Download")
    print("=" * 60)
    print(f"  Output:  {out_dir}")
    print(f"  Sources: {', '.join(sources)}")

    for source_name in sources:
        if source_name == "snemi":
            download_snemi(out_dir)
        elif source_name == "neurons":
            if args.start and args.size:
                crops = [("custom", tuple(args.start), tuple(args.size))]
            else:
                crops = [
                    (name, sp["start"], sp["size"])
                    for name, sp in NEURONS_SPLITS.items()
                ]
            download_neurons(out_dir, crops)

    print()
    print("=" * 60)
    print("Download complete!")
    print(f"  Output: {out_dir}")
    print()
    print("  Files:")
    for f in sorted(out_dir.glob("*.h5")):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name}  ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
