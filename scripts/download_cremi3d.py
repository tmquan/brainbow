#!/usr/bin/env python
"""Download the CREMI training + test samples and convert them to brainbow HDF5.

CREMI (https://cremi.org) provides three ssTEM volumes of adult *Drosophila*
brain at **4 x 4 x 40 nm** (x, y, z):

* **A, B, C** -- labelled TRAINING volumes (125 x 1250 x 1250, dense neuron
  ids), used for ``train_volumes``.
* **A+, B+, C+** -- padded TEST volumes; public EM only (the challenge
  withholds the test labels), converted **image-only** for ``test_volumes``.

The official files pack raw + labels into a single nested ``.hdf``
(``volumes/raw`` + ``volumes/labels/neuron_ids``), which ``LazyVolDataset``
cannot read directly.

This script downloads the originals and converts each sample into the
brainbow convention -- two ``.h5`` files (dataset key ``main``, axis order
``[Z, Y, X]``) consumed by ``MICRONSDataset`` / ``CREMI3DDataset`` and the
lazy 3-D patch loader::

    cremi3d_sample_A_volume.h5         (uint8  EM intensity)
    cremi3d_sample_A_segmentation.h5   (int64  neuron ids, 0 = background)

Example
-------
    # download A, B, C from cremi.org and convert
    python scripts/download_cremi3d.py --out-dir data/CREMI3D

    # reuse already-downloaded .hdf files (skip the network)
    python scripts/download_cremi3d.py --out-dir data/CREMI3D --hdf-dir /scratch/CREMI3D
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

CREMI_BASE_URL = "https://cremi.org/static/data"
# Labelled (cropped) TRAINING volumes -- dense neuron ids.
TRAIN_FILES = {
    "A": "sample_A_20160501.hdf",
    "B": "sample_B_20160501.hdf",
    "C": "sample_C_20160501.hdf",
}
# Padded TEST volumes (A+/B+/C+).  Public EM only -- the challenge withholds
# the test neuron ids -- so these convert to image-only crops for inference /
# qualitative test.
TEST_FILES = {
    "A+": "sample_A+_20160601.hdf",
    "B+": "sample_B+_20160601.hdf",
    "C+": "sample_C+_20160601.hdf",
}
SAMPLE_FILES = {**TRAIN_FILES, **TEST_FILES}
# CREMI marks unlabeled voxels (padded volumes) with the max uint64 value.
_NO_DATA = int(np.iinfo(np.uint64).max)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="data/CREMI3D", help="Output directory for converted .h5.")
    p.add_argument(
        "--samples", nargs="+",
        default=["A", "B", "C", "A+", "B+", "C+"],
        choices=["A", "B", "C", "A+", "B+", "C+"],
        help="Which CREMI samples to fetch/convert.  A/B/C are labelled "
             "training volumes; A+/B+/C+ are the padded TEST volumes "
             "(public EM only, labels withheld -> image-only).",
    )
    p.add_argument(
        "--hdf-dir", default=None,
        help="Directory of already-downloaded sample_*.hdf (skip the download step).",
    )
    return p.parse_args()


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip download {dest.name} (exists, {dest.stat().st_size / 1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  ok {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def _save_h5(arr: np.ndarray, path: Path, resolution_zyx_nm) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "main", data=arr, compression="gzip", compression_opts=4, chunks=True,
        )
        ds.attrs["resolution_zyx_nm"] = np.asarray(resolution_zyx_nm, dtype=np.float64)
        ds.attrs["source"] = "CREMI Challenge (https://cremi.org)"


def _convert(hdf_path: Path, out_dir: Path, sample: str) -> None:
    import h5py

    res = (40.0, 4.0, 4.0)  # z, y, x nm
    with h5py.File(str(hdf_path), "r", locking=False) as f:
        raw = f["volumes/raw"][:]  # [Z, Y, X] uint8
        seg = (
            f["volumes/labels/neuron_ids"][:]
            if "volumes/labels/neuron_ids" in f else None
        )

    raw = np.ascontiguousarray(raw.astype(np.uint8))
    vol_path = out_dir / f"cremi3d_sample_{sample}_volume.h5"
    _save_h5(raw, vol_path, res)

    # Map CREMI's "no data" marker to background; keep ids otherwise.  Test
    # volumes (A+/B+/C+) have no public labels, so only write a segmentation
    # when real foreground ids are present.
    if seg is not None:
        seg = np.where(np.asarray(seg) == _NO_DATA, 0, seg).astype(np.int64)
    if seg is not None and bool((seg > 0).any()):
        seg_path = out_dir / f"cremi3d_sample_{sample}_segmentation.h5"
        _save_h5(seg, seg_path, res)
        n_ids = int(np.unique(seg).size)
        fg = float((seg > 0).mean()) * 100.0
        print(
            f"  converted sample {sample}: image {raw.shape} uint8, "
            f"seg {seg.shape} int64, {n_ids} ids, foreground {fg:.1f}%"
        )
    else:
        print(
            f"  converted sample {sample}: image {raw.shape} uint8, "
            f"NO public labels (test/withheld) -> image-only"
        )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    hdf_dir = Path(args.hdf_dir) if args.hdf_dir else out_dir

    print(f"CREMI3D -> {out_dir}  (samples: {', '.join(args.samples)})")
    print("Resolution: 4 x 4 x 40 nm (x, y, z) -> resolution_map key 'cremi3d': [40, 4, 4]\n")

    for s in args.samples:
        fname = SAMPLE_FILES[s]
        hdf_path = hdf_dir / fname
        if args.hdf_dir is None:
            _download(f"{CREMI_BASE_URL}/{fname}", hdf_path)
        if not hdf_path.exists():
            print(f"  MISSING {hdf_path} -- skipping sample {s}")
            continue
        _convert(hdf_path, out_dir, s)

    train = [s for s in args.samples if s in TRAIN_FILES]
    test = [s for s in args.samples if s in TEST_FILES]
    if train:
        print("\nConfig train_volumes (labelled A/B/C):")
        for s in train:
            print(f"  - vol: cremi3d_sample_{s}_volume")
            print(f"    seg: cremi3d_sample_{s}_segmentation")
            print(f"    root: {out_dir}")
    if test:
        print("\nConfig test_volumes (A+/B+/C+, image-only -- labels withheld):")
        for s in test:
            print(f"  - vol: cremi3d_sample_{s}_volume")
            print(f"    root: {out_dir}")


if __name__ == "__main__":
    main()
