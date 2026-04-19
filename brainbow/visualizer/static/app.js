/**
 * Neurons Volume Visualizer — client-side slice rendering for instant scrolling.
 *
 * On boot, the raw + seg volumes are downloaded as typed arrays at a moderate
 * downsample.  Slice extraction and overlay compositing happen entirely in
 * JavaScript (no server round-trip), giving neuroglancer-like scroll speed.
 *
 * Mouse controls (matching neuroglancer):
 *   Slice panels:
 *     Left-click          → set crosshair / toggle instance selection
 *     Scroll              → move through slices (instant)
 *     Ctrl+Scroll         → zoom in/out
 *     Right-drag          → pan
 *     Double-click        → reset zoom/pan
 *   3D panel:
 *     Left-drag           → rotate
 *     Right-drag          → pan
 *     Scroll              → zoom
 */

"use strict";

/* ── global state ──────────────────────────────────────────────── */

window.NV = {
    vid: "default",
    shape: [1, 1, 1],       // Z, Y, X  (full-res)
    spacing: [1, 1, 1],     // sz, sy, sx
    sliceIdx: [0, 0, 0],    // current slice per axis
    overlay: true,
    opacity: 0.4,
    ready: false,
    selected: new Set(),

    // client-side volume data (preloaded)
    rawVol: null,            // Uint8Array, shape = dsShape, ZYX C-order
    segVol: null,            // Uint8Array (RGBA×4), shape = dsShape
    segIdVol: null,          // Uint8Array, shape = dsShape (raw IDs mod 256)
    palette: null,           // {id: [r,g,b], ...}
    dsShape: [1, 1, 1],      // downsampled shape
    ds: 1,                   // isotropic downsample factor
    dsPerAxis: [1, 1, 1],    // per-axis downsample (accounts for anisotropy)
};

/* shared view state: one zoom + per-axis pan in pixels */
const globalView = { zoom: 1, panPixX: 0, panPixY: 0, panPixZ: 0 };

/* ── elements ──────────────────────────────────────────────────── */

const canvases = [
    document.getElementById("canvas-axial"),
    document.getElementById("canvas-coronal"),
    document.getElementById("canvas-sagittal"),
];
const ctxs = canvases.map(c => c.getContext("2d"));
const infoEls = [
    document.getElementById("info-axial"),
    document.getElementById("info-coronal"),
    document.getElementById("info-sagittal"),
];

const cbOverlay  = document.getElementById("cb-overlay");
const slOpacity  = document.getElementById("sl-opacity");
const valOpacity = document.getElementById("val-opacity");
const tbInstance  = document.getElementById("tb-instance");
const tbCoord     = document.getElementById("tb-coord");
const tbSpacing   = document.getElementById("tb-spacing");

/* per-panel cached ImageData */
const panelImageData = [null, null, null];

/* ── helpers ───────────────────────────────────────────────────── */

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ── client-side slice extraction + compositing ────────────────── */

/**
 * Extract a 2D RGBA ImageData from the preloaded volume arrays.
 * Runs entirely in JS — no server call.
 */
function extractSlice(axis) {
    const st = window.NV;
    if (!st.rawVol) return null;

    const [dZ, dY, dX] = st.dsShape;
    const dsIdx = Math.floor(st.sliceIdx[axis] / st.ds);

    let rows, cols, getOffset;
    if (axis === 0) {        // axial: rows=Y, cols=X
        rows = dY; cols = dX;
        getOffset = (r, c) => dsIdx * dY * dX + r * dX + c;
    } else if (axis === 1) { // coronal: rows=Z, cols=X
        rows = dZ; cols = dX;
        getOffset = (r, c) => r * dY * dX + dsIdx * dX + c;
    } else {                 // sagittal: rows=Y, cols=Z (transposed for co-axis alignment)
        rows = dY; cols = dZ;
        getOffset = (r, c) => c * dY * dX + r * dX + dsIdx;
    }

    const buf = new Uint8ClampedArray(rows * cols * 4);
    const raw = st.rawVol;
    const hasSeg = st.overlay && st.segVol;
    const segRGBA = st.segVol;
    const segId = st.segIdVol;
    const opacity = st.opacity;
    const hasSel = st.selected.size > 0;
    const selSet = st.selected;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const off = getOffset(r, c);
            const px = (r * cols + c) * 4;
            const v = raw[off];

            let R = v, G = v, B = v;

            if (hasSeg) {
                const sOff = off * 4;
                const sA = segRGBA[sOff + 3];
                if (sA > 0) {
                    const sR = segRGBA[sOff];
                    const sG = segRGBA[sOff + 1];
                    const sB = segRGBA[sOff + 2];

                    let a = opacity;
                    if (hasSel) {
                        const id = segId ? segId[off] : 0;
                        a = selSet.has(id) ? opacity : opacity * 0.15;
                    }

                    R = Math.round((1 - a) * v + a * sR);
                    G = Math.round((1 - a) * v + a * sG);
                    B = Math.round((1 - a) * v + a * sB);
                }
            }

            buf[px]     = R;
            buf[px + 1] = G;
            buf[px + 2] = B;
            buf[px + 3] = 255;
        }
    }

    return new ImageData(buf, cols, rows);
}

/* ── draw panel from client-side data ──────────────────────────── */

function drawSlice(axis) {
    const st = window.NV;
    const idx = st.sliceIdx[axis];
    infoEls[axis].textContent = `${idx} / ${st.shape[axis] - 1}`;

    const imgData = extractSlice(axis);
    if (!imgData) return;
    panelImageData[axis] = imgData;
    drawPanel(axis);
}

function getAxisSpacing(axis) {
    const [sz, sy, sx] = window.NV.spacing;
    if (axis === 0)      return { colSp: sx, rowSp: sy };
    else if (axis === 1) return { colSp: sx, rowSp: sz };
    else                 return { colSp: sz, rowSp: sy };
}

function getGlobalBaseScale(pw, ph) {
    const st = window.NV;
    if (!st.shape) return 1;
    const [Z, Y, X] = st.shape;
    const [sz, sy, sx] = st.spacing;
    return Math.min(
        pw / (X * sx), ph / (Y * sy),
        ph / (Z * sz), pw / (Z * sz),
    );
}

function getPanPixels(axis) {
    const v = globalView;
    if (axis === 0)      return { h: v.panPixX, v: v.panPixY };
    else if (axis === 1) return { h: v.panPixX, v: v.panPixZ };
    else                 return { h: v.panPixZ, v: v.panPixY };
}

function computeTransform(axis) {
    const imgData = panelImageData[axis];
    if (!imgData) return null;

    const canvas = canvases[axis];
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const st = window.NV;
    const imgW = imgData.width;
    const imgH = imgData.height;
    const { colSp, rowSp } = getAxisSpacing(axis);

    const base = getGlobalBaseScale(rect.width, rect.height);
    const s = base * globalView.zoom;

    const dw = imgW * colSp * st.ds * s;
    const dh = imgH * rowSp * st.ds * s;

    const pan = getPanPixels(axis);
    const dx = (rect.width - dw) / 2 + pan.h;
    const dy = (rect.height - dh) / 2 + pan.v;

    return { dx, dy, dw, dh, imgW, imgH, cw: rect.width, ch: rect.height };
}

// Persistent offscreen canvases to avoid GC churn
const offscreenCanvases = [null, null, null];

function drawPanel(axis) {
    const ctx = ctxs[axis];
    const t = computeTransform(axis);
    if (!t) return;
    const imgData = panelImageData[axis];
    if (!imgData) return;

    ctx.fillStyle = "#f0f0f4";
    ctx.fillRect(0, 0, t.cw, t.ch);
    ctx.imageSmoothingEnabled = false;

    // Reuse offscreen canvas for putImageData → drawImage scaling
    let off = offscreenCanvases[axis];
    if (!off || off.width !== t.imgW || off.height !== t.imgH) {
        off = document.createElement("canvas");
        off.width = t.imgW;
        off.height = t.imgH;
        offscreenCanvases[axis] = off;
    }
    off.getContext("2d").putImageData(imgData, 0, 0);

    // drawImage stretches the imgH-pixel image into dh canvas pixels (aspect-corrected)
    ctx.drawImage(off, 0, 0, t.imgW, t.imgH, t.dx, t.dy, t.dw, t.dh);

    canvases[axis]._vt = t;
    drawCrosshair(axis);
}

/* ── crosshairs ────────────────────────────────────────────────── */

function drawCrosshair(axis) {
    const ctx = ctxs[axis];
    const t = canvases[axis]._vt;
    if (!t) return;
    const st = window.NV;

    let crossRow, crossCol;
    if (axis === 0) {
        crossRow = Math.floor(st.sliceIdx[1] / st.ds);
        crossCol = Math.floor(st.sliceIdx[2] / st.ds);
    } else if (axis === 1) {
        crossRow = Math.floor(st.sliceIdx[0] / st.ds);
        crossCol = Math.floor(st.sliceIdx[2] / st.ds);
    } else {
        crossRow = Math.floor(st.sliceIdx[1] / st.ds);
        crossCol = Math.floor(st.sliceIdx[0] / st.ds);
    }

    const px = t.dx + ((crossCol + 0.5) / t.imgW) * t.dw;
    const py = t.dy + ((crossRow + 0.5) / t.imgH) * t.dh;

    ctx.save();
    ctx.strokeStyle = "rgba(30, 120, 200, 0.6)";
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, t.ch); ctx.stroke();

    ctx.strokeStyle = "rgba(200, 120, 30, 0.6)";
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(t.cw, py); ctx.stroke();
    ctx.restore();
}

/* ── mouse → voxel coord mapping ───────────────────────────────── */

function canvasToVoxel(axis, mx, my) {
    const t = canvases[axis]._vt;
    if (!t) return null;
    const col = Math.floor(((mx - t.dx) / t.dw) * t.imgW);
    const row = Math.floor(((my - t.dy) / t.dh) * t.imgH);
    if (col < 0 || row < 0 || col >= t.imgW || row >= t.imgH) return null;
    return { row, col };
}

function voxelToFull(axis, row, col) {
    const ds = window.NV.ds;
    const st = window.NV;
    let z, y, x;
    if (axis === 0)      { z = st.sliceIdx[0]; y = row * ds; x = col * ds; }
    else if (axis === 1) { z = row * ds; y = st.sliceIdx[1]; x = col * ds; }
    else                 { y = row * ds; z = col * ds; x = st.sliceIdx[2]; }
    return { z, y, x };
}

/* ── event handlers ────────────────────────────────────────────── */

const panState = { active: false, axis: -1, startX: 0, startY: 0, origPanH: 0, origPanV: 0 };

canvases.forEach((c, axis) => {
    c.addEventListener("contextmenu", e => e.preventDefault());

    c.addEventListener("wheel", e => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const newZoom = clamp(globalView.zoom * factor, 0.2, 30);
            const ratio = newZoom / globalView.zoom;
            globalView.panPixX *= ratio;
            globalView.panPixY *= ratio;
            globalView.panPixZ *= ratio;
            globalView.zoom = newZoom;
            [0, 1, 2].forEach(a => drawPanel(a));
        } else {
            const delta = e.deltaY > 0 ? 1 : -1;
            const st = window.NV;
            st.sliceIdx[axis] = clamp(st.sliceIdx[axis] + delta, 0, st.shape[axis] - 1);
            drawSlice(axis);
            [0, 1, 2].forEach(a => { if (a !== axis) drawPanel(a); });
            updateCoordDisplay();
            sync3dPlanes();
        }
    }, { passive: false });

    c.addEventListener("mousedown", e => {
        if (e.button === 0 && !e.ctrlKey && !e.shiftKey) {
            c._leftDown = { x: e.clientX, y: e.clientY };
        }
        if (e.button === 2 || (e.button === 0 && e.shiftKey)) {
            panState.active = true;
            panState.axis = axis;
            panState.startX = e.clientX;
            panState.startY = e.clientY;
            const pan = getPanPixels(axis);
            panState.origPanH = pan.h;
            panState.origPanV = pan.v;
            c.style.cursor = "grabbing";
        }
    });

    c.addEventListener("mousemove", e => {
        const rect = c.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const v = canvasToVoxel(axis, mx, my);
        if (v) {
            const full = voxelToFull(axis, v.row, v.col);
            tbCoord.textContent = `Z:${full.z} Y:${full.y} X:${full.x}`;
        }
    });

    c.addEventListener("dblclick", e => {
        e.preventDefault();
        globalView.zoom = 1;
        globalView.panPixX = 0;
        globalView.panPixY = 0;
        globalView.panPixZ = 0;
        [0, 1, 2].forEach(a => drawPanel(a));
    });
});

window.addEventListener("mousemove", e => {
    if (!panState.active) return;
    const ddx = e.clientX - panState.startX;
    const ddy = e.clientY - panState.startY;
    const a = panState.axis;
    if (a === 0)      { globalView.panPixX = panState.origPanH + ddx; globalView.panPixY = panState.origPanV + ddy; }
    else if (a === 1) { globalView.panPixX = panState.origPanH + ddx; globalView.panPixZ = panState.origPanV + ddy; }
    else              { globalView.panPixZ = panState.origPanH + ddx; globalView.panPixY = panState.origPanV + ddy; }
    [0, 1, 2].forEach(i => drawPanel(i));
});

window.addEventListener("mouseup", e => {
    if (panState.active) {
        canvases[panState.axis].style.cursor = "crosshair";
        panState.active = false;
    }
    canvases.forEach((c, axis) => {
        if (c._leftDown && e.button === 0) {
            const dx = Math.abs(e.clientX - c._leftDown.x);
            const dy = Math.abs(e.clientY - c._leftDown.y);
            if (dx < 3 && dy < 3) {
                const rect = c.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const v = canvasToVoxel(axis, mx, my);
                if (v) {
                    const st = window.NV;
                    const full = voxelToFull(axis, v.row, v.col);
                    if (axis === 0)      { st.sliceIdx[1] = clamp(full.y, 0, st.shape[1]-1); st.sliceIdx[2] = clamp(full.x, 0, st.shape[2]-1); }
                    else if (axis === 1) { st.sliceIdx[0] = clamp(full.z, 0, st.shape[0]-1); st.sliceIdx[2] = clamp(full.x, 0, st.shape[2]-1); }
                    else                 { st.sliceIdx[0] = clamp(full.z, 0, st.shape[0]-1); st.sliceIdx[1] = clamp(full.y, 0, st.shape[1]-1); }
                    drawAllSlices();
                    updateCoordDisplay();

                    // Look up segment ID from client-side data
                    if (st.segIdVol) {
                        const off = getVoxelOffset(axis, v.row, v.col);
                        const lid = st.segIdVol[off];
                        tbInstance.textContent = `Instance: #${lid}`;
                        if (lid > 0) toggleSelection(lid);
                    }
                }
            }
            c._leftDown = null;
        }
    });
});

function getVoxelOffset(axis, row, col) {
    const [dZ, dY, dX] = window.NV.dsShape;
    const dsIdx = Math.floor(window.NV.sliceIdx[axis] / window.NV.ds);
    if (axis === 0)      return dsIdx * dY * dX + row * dX + col;
    else if (axis === 1) return row * dY * dX + dsIdx * dX + col;
    else                 return col * dY * dX + row * dX + dsIdx;
}

/* ── selection management ───────────────────────────────────────── */

function toggleSelection(labelId) {
    if (labelId <= 0) return;
    const st = window.NV;
    if (st.selected.has(labelId)) {
        st.selected.delete(labelId);
    } else {
        st.selected.add(labelId);
    }
    renderSelectionChips();
    drawAllSlices();
    sync3dSelection();
}

function clearSelection() {
    window.NV.selected.clear();
    renderSelectionChips();
    drawAllSlices();
    sync3dSelection();
}

function renderSelectionChips() {
    const el = document.getElementById("tb-selected");
    const st = window.NV;
    if (st.selected.size === 0) { el.innerHTML = ""; return; }
    let html = "";
    for (const id of st.selected) {
        html += `<span class="sel-chip" data-id="${id}">#${id} <span class="sel-x">\u00d7</span></span>`;
    }
    html += `<span class="sel-clear" title="Clear all">clear</span>`;
    el.innerHTML = html;
    el.querySelectorAll(".sel-chip").forEach(chip => {
        chip.addEventListener("click", () => toggleSelection(parseInt(chip.dataset.id)));
    });
    el.querySelector(".sel-clear").addEventListener("click", clearSelection);
}

function sync3dSelection() {
    if (typeof window._set3dSelection === "function") {
        window._set3dSelection([...window.NV.selected]);
    }
}

/* ── draw all + sync ───────────────────────────────────────────── */

function drawAllSlices() {
    drawSlice(0);
    drawSlice(1);
    drawSlice(2);
    sync3dPlanes();
}

function sync3dPlanes() {
    if (typeof window._updateSlicePlanes === "function") {
        window._updateSlicePlanes();
    }
}

function updateCoordDisplay() {
    const s = window.NV.sliceIdx;
    tbCoord.textContent = `Z:${s[0]} Y:${s[1]} X:${s[2]}`;
}

/* ── toolbar controls ──────────────────────────────────────────── */

cbOverlay.addEventListener("change", () => {
    window.NV.overlay = cbOverlay.checked;
    drawAllSlices();
    if (typeof window._set3dSegOverlay === "function") {
        window._set3dSegOverlay(cbOverlay.checked, window.NV.opacity);
    }
});

slOpacity.addEventListener("input", () => {
    const v = parseInt(slOpacity.value) / 100;
    window.NV.opacity = v;
    valOpacity.textContent = v.toFixed(2);
    drawAllSlices();
    if (typeof window._set3dSegOverlay === "function") {
        window._set3dSegOverlay(window.NV.overlay, v);
    }
});

document.getElementById("cb-filled").addEventListener("change", (e) => {
    if (typeof window._set3dFilled === "function") {
        window._set3dFilled(e.target.checked);
    }
});

document.getElementById("cb-slicing").addEventListener("change", (e) => {
    if (typeof window._set3dSlicing === "function") {
        window._set3dSlicing(e.target.checked);
    }
});

document.getElementById("cb-cutting").addEventListener("change", (e) => {
    if (typeof window._set3dCutting === "function") {
        window._set3dCutting(e.target.checked);
    }
});

/* ── cache management ──────────────────────────────────────────── */

document.getElementById("btn-clear-cache").addEventListener("click", async () => {
    try {
        await caches.delete(CACHE_NAME);
        console.log("Cache cleared");
    } catch (_) {}
    location.reload();
});

/* ── resize ────────────────────────────────────────────────────── */

let resizeTimer;
window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { [0, 1, 2].forEach(a => drawPanel(a)); }, 80);
});

/* ── client-side Cache API for volume chunks ───────────────────── */

const CACHE_NAME = "brainbow-vol-v1";

async function cachedFetch(url, label, loadMsg) {
    let cache;
    try { cache = await caches.open(CACHE_NAME); } catch (_) { cache = null; }

    if (cache) {
        const cached = await cache.match(url);
        if (cached) {
            if (loadMsg) loadMsg.textContent = `${label} (cached)`;
            console.log(`Cache hit: ${label}`);
            return cached;
        }
    }

    if (loadMsg) loadMsg.textContent = `${label}...`;
    const resp = await fetch(url);
    if (resp.ok && cache) {
        // Clone before consuming the body — cache stores the clone, we return original
        cache.put(url, resp.clone());
    }
    return resp;
}

/* ── boot: preload volumes (with cache) then render ────────────── */

async function boot() {
    const st = window.NV;

    const loadingEl = document.createElement("div");
    loadingEl.id = "loading-overlay";
    loadingEl.style.cssText = "position:fixed;inset:0;background:rgba(240,240,244,0.95);display:flex;align-items:center;justify-content:center;z-index:999;color:#4080d0;font-size:16px;font-family:system-ui;flex-direction:column;gap:8px";
    loadingEl.innerHTML = '<div id="load-msg">Loading volume data...</div>';
    document.body.appendChild(loadingEl);
    const loadMsg = document.getElementById("load-msg");

    try {
        // 1. Fetch metadata (never cached — always fresh)
        loadMsg.textContent = "Fetching metadata...";
        const metaR = await fetch(`/api/volumes/${st.vid}/meta`);
        if (!metaR.ok) throw new Error("meta fetch failed");
        const meta = await metaR.json();
        st.shape = meta.shape;
        st.spacing = meta.spacing;
        st.sliceIdx = [
            meta.shape[0] - 1,
            meta.shape[1] - 1,
            meta.shape[2] - 1,
        ];

        // 2. Determine downsample
        const [sz, sy, sx] = st.spacing;
        const maxXY = Math.max(st.shape[1], st.shape[2]);
        st.ds = Math.max(1, Math.ceil(maxXY / 512));
        let totalVox = st.shape.reduce((a, s) => a * Math.ceil(s / st.ds), 1);
        while (totalVox > 40e6 && st.ds < 16) {
            st.ds++;
            totalVox = st.shape.reduce((a, s) => a * Math.ceil(s / st.ds), 1);
        }
        st.dsPerAxis = [st.ds, st.ds, st.ds];
        console.log(`Downsample: ${st.ds}, shape ${st.shape} → ${st.shape.map(s => Math.ceil(s/st.ds))} (${(totalVox/1e6).toFixed(1)}M vox)`);

        // 3. Spacing info (axis alignment handled by global base scale)
        console.log(`Spacing: sz=${sz}, sy=${sy}, sx=${sx}`);

        // 4. Preload raw volume (cached)
        const rawUrl = `/api/volumes/${st.vid}/chunk?downsample=${st.ds}`;
        const rawR = await cachedFetch(rawUrl, `Downloading raw (ds=${st.ds})`, loadMsg);
        const rawShape = rawR.headers.get("X-Shape").split(",").map(Number);
        st.rawVol = new Uint8Array(await rawR.arrayBuffer());
        st.dsShape = rawShape;
        console.log(`Raw loaded: [${rawShape}] (${(st.rawVol.length/1024).toFixed(0)} KB)`);

        // 5. Preload seg volumes (cached)
        if (meta.has_seg) {
            const segUrl = `/api/volumes/${st.vid}/seg_chunk?downsample=${st.ds}`;
            const segR = await cachedFetch(segUrl, "Downloading segmentation", loadMsg);
            if (segR.ok) {
                st.segVol = new Uint8Array(await segR.arrayBuffer());
                console.log(`Seg loaded: ${(st.segVol.length/1024).toFixed(0)} KB`);
            }
            const idUrl = `/api/volumes/${st.vid}/seg_id_chunk?downsample=${st.ds}`;
            const idR = await cachedFetch(idUrl, "Downloading seg IDs", loadMsg);
            if (idR.ok) {
                st.segIdVol = new Uint8Array(await idR.arrayBuffer());
                console.log(`Seg IDs loaded: ${(st.segIdVol.length/1024).toFixed(0)} KB`);
            }
        }

        // 6. Ready — render all panels
        loadMsg.textContent = "Rendering...";
        st.ready = true;
        tbSpacing.textContent = `Spacing: ${st.spacing.map(v => v.toFixed(1)).join(" \u00d7 ")}`;
        updateCoordDisplay();
        drawAllSlices();

        // 7. Start 3D renderer
        if (typeof initVolumeRenderer === "function") {
            loadMsg.textContent = "Initializing 3D renderer...";
            initVolumeRenderer();
        }

        // Remove loading overlay
        loadingEl.remove();

    } catch (err) {
        console.error("Boot failed:", err);
        if (loadingEl.parentNode) loadingEl.remove();
        document.getElementById("panel-3d").innerHTML =
            `<div class="panel-label" style="color:#e94560;top:40%">Failed to load: ${err.message}</div>`;
    }
}

boot();
