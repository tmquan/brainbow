"""
FastAPI application serving volume slices, metadata, and chunks.

Endpoints:
  /api/volumes          POST  – register raw + seg volumes
  /api/volumes/{id}/meta       – shape, spacing, dtype, label count
  /api/volumes/{id}/slice/…    – PNG slice with optional seg overlay
  /api/volumes/{id}/chunk      – downsampled uint8 raw for 3-D textures
  /api/volumes/{id}/seg_chunk  – RGBA palette-coloured seg for 3-D textures
  /api/volumes/{id}/seg_id_chunk – uint8 seg IDs (mod 256) for selection
  /api/volumes/{id}/labels     – unique label list + palette
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from brainbow.visualizer.volume_loader import VolumeData, load_volume

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Neurons Volume Visualizer")

# ── in-memory store ────────────────────────────────────────────────────

class _VolumeEntry:
    def __init__(
        self,
        raw: VolumeData,
        seg: Optional[VolumeData],
        palette: Optional[np.ndarray],
    ) -> None:
        self.raw = raw
        self.seg = seg
        self.raw_u8 = raw.to_uint8()
        self.palette = palette

_store: Dict[str, _VolumeEntry] = {}


def _build_palette(seg: VolumeData, seed: int = 0) -> np.ndarray:
    """Random RGBA palette for segment IDs.  ID 0 maps to transparent."""
    labels = np.unique(seg.data)
    max_id = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    pal = rng.integers(60, 256, size=(max_id, 3), dtype=np.uint8)
    pal[0] = 0
    return pal


# ── models ─────────────────────────────────────────────────────────────

class LoadRequest(BaseModel):
    raw: str
    seg: Optional[str] = None
    key_raw: Optional[str] = None
    key_seg: Optional[str] = None
    spacing: Optional[Tuple[float, float, float]] = None


class LoadResponse(BaseModel):
    id: str
    shape: Tuple[int, int, int]
    dtype: str
    spacing: Tuple[float, float, float]


# ── routes ─────────────────────────────────────────────────────────────

@app.post("/api/volumes", response_model=LoadResponse)
def load_volumes(req: LoadRequest):
    raw = load_volume(req.raw, key=req.key_raw, spacing=req.spacing)
    seg = load_volume(req.seg, key=req.key_seg, spacing=req.spacing) if req.seg else None
    if seg and seg.shape != raw.shape:
        raise HTTPException(
            400,
            f"Shape mismatch: raw {raw.shape} vs seg {seg.shape}",
        )
    pal = _build_palette(seg) if seg else None
    vid = uuid.uuid4().hex[:8]
    _store[vid] = _VolumeEntry(raw, seg, pal)
    return LoadResponse(
        id=vid,
        shape=raw.shape,
        dtype=str(raw.dtype),
        spacing=raw.spacing,
    )


@app.get("/api/volumes/{vid}/meta")
def get_meta(vid: str):
    e = _store.get(vid)
    if e is None:
        raise HTTPException(404, "Volume not found")
    resp = {
        "shape": e.raw.shape,
        "dtype": str(e.raw.dtype),
        "spacing": e.raw.spacing,
        "has_seg": e.seg is not None,
    }
    if e.seg is not None:
        labels = np.unique(e.seg.data)
        resp["num_labels"] = int(len(labels))
    return resp


@app.get("/api/volumes/{vid}/labels")
def get_labels(vid: str):
    e = _store.get(vid)
    if e is None:
        raise HTTPException(404, "Volume not found")
    if e.seg is None:
        return {"labels": [], "palette": []}
    labels = [int(v) for v in np.unique(e.seg.data) if v > 0]
    pal = {int(lid): [int(c) for c in e.palette[lid]] for lid in labels}
    return {"labels": labels, "palette": pal}


def _slice_to_png(
    raw_u8: np.ndarray,
    seg_slice: Optional[np.ndarray],
    palette: Optional[np.ndarray],
    opacity: float,
    selected: Optional[set] = None,
) -> bytes:
    """Composite raw grayscale + seg overlay into an RGBA PNG.

    When *selected* is non-empty, selected instances are drawn at full
    *opacity* while unselected foreground uses ``opacity * 0.15``.
    """
    from PIL import Image

    h, w = raw_u8.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = raw_u8
    rgba[..., 1] = raw_u8
    rgba[..., 2] = raw_u8
    rgba[..., 3] = 255

    if seg_slice is not None and palette is not None and opacity > 0:
        fg = seg_slice > 0
        if fg.any():
            seg_ids = seg_slice[fg].astype(np.int64)
            seg_ids_clipped = np.clip(seg_ids, 0, len(palette) - 1)
            colors = palette[seg_ids_clipped]  # (N, 3)

            if selected:
                sel_mask = np.isin(seg_ids, list(selected))
                alphas = np.where(sel_mask, opacity, opacity * 0.15).astype(np.float32)
            else:
                alphas = np.full(len(seg_ids), opacity, dtype=np.float32)

            for ch in range(3):
                channel = rgba[..., ch]
                channel[fg] = np.clip(
                    (1 - alphas) * channel[fg].astype(np.float32) + alphas * colors[:, ch],
                    0, 255,
                ).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


@app.get("/api/volumes/{vid}/slice/{axis}/{idx}")
def get_slice(
    vid: str,
    axis: int,
    idx: int,
    overlay: bool = Query(True),
    opacity: float = Query(0.4),
    selected: str = Query(""),
):
    e = _store.get(vid)
    if e is None:
        raise HTTPException(404, "Volume not found")
    if axis not in (0, 1, 2):
        raise HTTPException(400, "axis must be 0, 1, or 2")

    sel_set: Optional[set] = None
    if selected:
        try:
            sel_set = {int(x) for x in selected.split(",") if x.strip()}
        except ValueError:
            sel_set = None

    raw_slice = e.raw.slice_axis(axis, idx)
    raw_u8 = e.raw_u8[tuple(
        idx if d == axis else slice(None) for d in range(3)
    )]

    seg_slice = e.seg.slice_axis(axis, idx) if (overlay and e.seg) else None
    png = _slice_to_png(raw_u8, seg_slice, e.palette, opacity if overlay else 0, sel_set)

    aspect = e.raw.aspect_ratio(axis)

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Aspect-Ratio": f"{aspect:.6f}",
            "X-Slice-Shape": f"{raw_slice.shape[0]},{raw_slice.shape[1]}",
            "Access-Control-Expose-Headers": "X-Aspect-Ratio,X-Slice-Shape",
        },
    )


@app.get("/api/volumes/{vid}/label_at/{axis}/{idx}")
def get_label_at(vid: str, axis: int, idx: int, row: int = 0, col: int = 0):
    """Return the segment ID at a specific pixel in a slice."""
    e = _store.get(vid)
    if e is None:
        raise HTTPException(404, "Volume not found")
    if e.seg is None:
        return {"label": 0}
    seg_slice = e.seg.slice_axis(axis, idx)
    r = max(0, min(row, seg_slice.shape[0] - 1))
    c = max(0, min(col, seg_slice.shape[1] - 1))
    return {"label": int(seg_slice[r, c])}


@app.get("/api/volumes/{vid}/chunk")
def get_chunk(
    vid: str,
    downsample: int = Query(4, ge=1, le=16),
):
    """Return downsampled uint8 raw volume for the 3-D renderer."""
    e = _store.get(vid)
    if e is None:
        raise HTTPException(404, "Volume not found")

    vol = e.raw_u8[::downsample, ::downsample, ::downsample]
    vol = np.ascontiguousarray(vol)

    sz, sy, sx = e.raw.spacing
    ds_spacing = (sz * downsample, sy * downsample, sx * downsample)

    return Response(
        content=vol.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{vol.shape[0]},{vol.shape[1]},{vol.shape[2]}",
            "X-Spacing": f"{ds_spacing[0]:.4f},{ds_spacing[1]:.4f},{ds_spacing[2]:.4f}",
            "X-Dtype": "uint8",
            "Access-Control-Expose-Headers": "X-Shape,X-Spacing,X-Dtype",
        },
    )


@app.get("/api/volumes/{vid}/seg_chunk")
def get_seg_chunk(
    vid: str,
    downsample: int = Query(4, ge=1, le=16),
):
    """Return downsampled segmentation as RGBA volume (palette-colored)."""
    e = _store.get(vid)
    if e is None:
        raise HTTPException(404, "Volume not found")
    if e.seg is None:
        raise HTTPException(404, "No segmentation loaded")

    seg_ds = e.seg.data[::downsample, ::downsample, ::downsample]
    Z, Y, X = seg_ds.shape

    rgba = np.zeros((Z, Y, X, 4), dtype=np.uint8)
    fg = seg_ds > 0
    if fg.any():
        ids = seg_ds[fg].astype(np.int64)
        ids = np.clip(ids, 0, len(e.palette) - 1)
        rgba[..., 0][fg] = e.palette[ids, 0]
        rgba[..., 1][fg] = e.palette[ids, 1]
        rgba[..., 2][fg] = e.palette[ids, 2]
        rgba[..., 3][fg] = 255

    rgba = np.ascontiguousarray(rgba)
    sz, sy, sx = e.raw.spacing
    ds_spacing = (sz * downsample, sy * downsample, sx * downsample)

    return Response(
        content=rgba.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{Z},{Y},{X}",
            "X-Spacing": f"{ds_spacing[0]:.4f},{ds_spacing[1]:.4f},{ds_spacing[2]:.4f}",
            "X-Dtype": "rgba8",
            "Access-Control-Expose-Headers": "X-Shape,X-Spacing,X-Dtype",
        },
    )


@app.get("/api/volumes/{vid}/seg_id_chunk")
def get_seg_id_chunk(
    vid: str,
    downsample: int = Query(4, ge=1, le=16),
):
    """Return downsampled seg IDs as uint8 (mod 256) for shader selection lookup."""
    e = _store.get(vid)
    if e is None:
        raise HTTPException(404, "Volume not found")
    if e.seg is None:
        raise HTTPException(404, "No segmentation loaded")

    seg_ds = e.seg.data[::downsample, ::downsample, ::downsample]
    ids_u8 = (seg_ds % 256).astype(np.uint8)
    ids_u8 = np.ascontiguousarray(ids_u8)

    return Response(
        content=ids_u8.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{ids_u8.shape[0]},{ids_u8.shape[1]},{ids_u8.shape[2]}",
            "Access-Control-Expose-Headers": "X-Shape",
        },
    )


# ── static files (must be last so it doesn't shadow /api) ─────────────

app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
