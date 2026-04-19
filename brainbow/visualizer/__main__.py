"""
CLI entry point: ``python -m brainbow.visualizer --raw vol.h5 --seg seg.h5``

Loads the volumes, registers them with the FastAPI app, then starts uvicorn.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from typing import Tuple, Optional


def _parse_spacing(s: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("spacing must be sz,sy,sx  (3 comma-separated floats)")
    return (parts[0], parts[1], parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="brainbow.visualizer",
        description="Web-based neuroglancer-style volume viewer",
    )
    parser.add_argument("--raw", required=True, help="Path to raw volume (HDF5/TIFF/NRRD/NumPy)")
    parser.add_argument("--seg", default=None, help="Path to segmentation volume (optional)")
    parser.add_argument("--key-raw", default=None, help="HDF5 dataset key for raw volume")
    parser.add_argument("--key-seg", default=None, help="HDF5 dataset key for segmentation")
    parser.add_argument(
        "--spacing", default=None, type=_parse_spacing,
        help="Voxel spacing as sz,sy,sx (e.g. 30,6,6 for SNEMI3D). "
             "Auto-detected from file metadata when omitted.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8899, type=int)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    from brainbow.visualizer.volume_loader import load_volume
    from brainbow.visualizer.app import app, _store, _VolumeEntry, _build_palette

    print(f"Loading raw volume: {args.raw}")
    raw = load_volume(args.raw, key=args.key_raw, spacing=args.spacing)
    print(f"  shape={raw.shape}  dtype={raw.dtype}  spacing={raw.spacing}")

    seg = None
    if args.seg:
        print(f"Loading segmentation: {args.seg}")
        seg = load_volume(args.seg, key=args.key_seg, spacing=args.spacing)
        print(f"  shape={seg.shape}  dtype={seg.dtype}")
        if seg.shape != raw.shape:
            print(f"WARNING: shape mismatch raw {raw.shape} vs seg {seg.shape}", file=sys.stderr)

    pal = _build_palette(seg) if seg else None
    vid = "default"
    _store[vid] = _VolumeEntry(raw, seg, pal)
    print(f"\nVolume registered as id='{vid}'")

    url = f"http://{args.host}:{args.port}"
    print(f"Starting server at {url}")

    if not args.no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
