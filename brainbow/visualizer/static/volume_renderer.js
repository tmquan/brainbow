/**
 * Three.js 3-D panel: Gaussian-splatted segmentation rendering
 * + 3 orthogonal slice planes.
 *
 * Selected segments are rendered as point-cloud Gaussian splats —
 * each voxel becomes a soft, screen-aligned Gaussian blob, giving
 * a translucent volumetric appearance.
 */

"use strict";

/* ── Anisotropic 3D Gaussian splat shaders (GLSL 300 ES) ────── */

const SPLAT_VERT = `
in vec3 position;
in vec3 color;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 uSigma;
uniform float uScreenHeight;
uniform bool uCulling;
uniform vec3 uCullMin;
uniform vec3 uCullMax;
out vec3 vColor;
flat out vec3 vInvCov;
flat out float vPtSize;
flat out float vCulled;

void main() {
    vCulled = 0.0;
    if (uCulling) {
        if (position.x < uCullMin.x || position.x > uCullMax.x ||
            position.y < uCullMin.y || position.y > uCullMax.y ||
            position.z < uCullMin.z || position.z > uCullMax.z) {
            vCulled = 1.0;
        }
    }

    vColor = color;
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    float z = max(-mvPos.z, 0.01);

    mat3 R = mat3(modelViewMatrix);
    vec3 ax = uSigma.x * R[0];
    vec3 ay = uSigma.y * R[1];
    vec3 az = uSigma.z * R[2];

    float cxx = ax.x*ax.x + ay.x*ay.x + az.x*az.x;
    float cxy = ax.x*ax.y + ay.x*ay.y + az.x*az.y;
    float cyy = ax.y*ax.y + ay.y*ay.y + az.y*az.y;

    float focal = projectionMatrix[1][1] * uScreenHeight * 0.5;
    float s = focal / z;
    cxx *= s * s;  cxy *= s * s;  cyy *= s * s;

    float mid = 0.5 * (cxx + cyy);
    float det = cxx * cyy - cxy * cxy;
    float disc = max(mid * mid - det, 0.0);
    float lambda_max = mid + sqrt(disc);
    float radius = ceil(3.0 * sqrt(max(lambda_max, 0.1)));
    float ptSz = clamp(2.0 * radius, 1.0, 512.0);
    gl_PointSize = vCulled > 0.5 ? 0.0 : ptSz;
    vPtSize = ptSz;

    float invDet = 1.0 / max(det, 1e-6);
    vInvCov = vec3(cyy * invDet, -cxy * invDet, cxx * invDet);

    gl_Position = projectionMatrix * mvPos;
}
`;

const SPLAT_FRAG = `
precision highp float;
uniform float uOpacity;
in vec3 vColor;
flat in vec3 vInvCov;
flat in float vPtSize;
flat in float vCulled;
out vec4 fragColor;
void main() {
    if (vCulled > 0.5) discard;
    vec2 d = vec2(gl_PointCoord.x - 0.5, 0.5 - gl_PointCoord.y) * vPtSize;
    float maha = d.x * d.x * vInvCov.x + 2.0 * d.x * d.y * vInvCov.y + d.y * d.y * vInvCov.z;
    float gauss = exp(-0.5 * maha);
    if (gauss < 0.02) discard;
    float a = gauss * uOpacity;
    vec3 lit = vColor * (0.55 + 0.45 * gauss) + vec3(0.12 * gauss * gauss);
    fragColor = vec4(lit, a);
}
`;

/* ── GLSL 300 ES — slice plane ────────────────────────────── */

const PLANE_VERT = `
in vec3 position;
in vec2 uv;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
out vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const PLANE_FRAG = `
precision highp float;
precision highp sampler3D;
precision highp sampler2D;
in vec2 vUv;
out vec4 fragColor;

uniform sampler3D uVolume;
uniform sampler3D uSeg;
uniform sampler3D uSegId;
uniform sampler2D uSelMask;
uniform float uSelMaskSize;
uniform bool  uHasSelection;
uniform int   uAxis;
uniform float uSliceFrac;
uniform float uWinLo;
uniform float uWinHi;
uniform float uSegOpacity;
uniform bool  uShowSeg;

void main() {
    vec3 tc;
    if (uAxis == 0)      tc = vec3(vUv.x, vUv.y, uSliceFrac);
    else if (uAxis == 1) tc = vec3(vUv.x, uSliceFrac, vUv.y);
    else                 tc = vec3(uSliceFrac, vUv.x, vUv.y);

    float raw = texture(uVolume, tc).r;
    float v = clamp((raw - uWinLo) / max(uWinHi - uWinLo, 0.001), 0.0, 1.0);
    vec3 c = vec3(v);

    if (uShowSeg) {
        vec4 seg = texture(uSeg, tc);
        if (seg.a > 0.01) {
            float segIdVal = texture(uSegId, tc).r * 255.0;

            bool isSel = false;
            if (uHasSelection && uSelMaskSize > 0.0) {
                float u = (segIdVal + 0.5) / uSelMaskSize;
                isSel = texture(uSelMask, vec2(u, 0.5)).r > 0.5;
            }

            float strength = uHasSelection
                ? (isSel ? uSegOpacity : uSegOpacity * 0.12)
                : uSegOpacity;
            c = mix(c, seg.rgb, strength);
        }
    }

    fragColor = vec4(c, 1.0);
}
`;

/* ── GLSL 300 ES — ray-marched volume rendering ──────────── */

const VOL_VERT = `
in vec3 position;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
out vec3 vWorldPos;
void main() {
    vWorldPos = position + 0.5 * (uBoxMin + uBoxMax);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const VOL_FRAG = `
precision highp float;
precision highp sampler3D;
in vec3 vWorldPos;
out vec4 fragColor;

uniform sampler3D uVolume;
uniform sampler3D uSeg;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform vec3 uCutMin;
uniform vec3 uCutMax;
uniform bool uCutting;
uniform vec3 uCamPos;
uniform float uWinLo;
uniform float uWinHi;
uniform float uSegOpacity;
uniform bool uShowSeg;
uniform float uDensity;

void main() {
    vec3 ro = uCamPos;
    vec3 rd = normalize(vWorldPos - ro);
    vec3 boxSize = uBoxMax - uBoxMin;

    vec3 bmin = uCutting ? max(uCutMin, uBoxMin) : uBoxMin;
    vec3 bmax = uCutting ? min(uCutMax, uBoxMax) : uBoxMax;

    vec3 invRd = 1.0 / rd;
    vec3 t0 = (bmin - ro) * invRd;
    vec3 t1 = (bmax - ro) * invRd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    float tFar  = min(min(tmax.x, tmax.y), tmax.z);

    if (tNear >= tFar) discard;

    int NUM_STEPS = 192;
    float stepSize = (tFar - tNear) / float(NUM_STEPS);
    vec3 accum = vec3(0.0);
    float alphaAcc = 0.0;

    for (int i = 0; i < 192; i++) {
        float t = tNear + (float(i) + 0.5) * stepSize;
        vec3 p = ro + rd * t;
        vec3 tc = (p - uBoxMin) / boxSize;

        float raw = texture(uVolume, tc).r;
        float v = clamp((raw - uWinLo) / max(uWinHi - uWinLo, 0.001), 0.0, 1.0);
        vec3 color = vec3(v);

        if (uShowSeg) {
            vec4 seg = texture(uSeg, tc);
            if (seg.a > 0.01) {
                color = mix(color, seg.rgb, uSegOpacity);
            }
        }

        float alpha = v * uDensity;
        accum += color * alpha * (1.0 - alphaAcc);
        alphaAcc += alpha * (1.0 - alphaAcc);
        if (alphaAcc > 0.95) break;
    }

    if (alphaAcc < 0.01) discard;
    fragColor = vec4(accum, alphaAcc);
}
`;

/* ── helpers ───────────────────────────────────────────────── */

function reorderZYX(data, dz, dy, dx) {
    const out = new Uint8Array(dx * dy * dz);
    for (let z = 0; z < dz; z++)
        for (let y = 0; y < dy; y++)
            for (let x = 0; x < dx; x++)
                out[x + y * dx + z * dx * dy] = data[z * dy * dx + y * dx + x];
    return out;
}

function reorderZYX_RGBA(data, dz, dy, dx) {
    const out = new Uint8Array(dx * dy * dz * 4);
    for (let z = 0; z < dz; z++)
        for (let y = 0; y < dy; y++)
            for (let x = 0; x < dx; x++) {
                const srcOff = (z * dy * dx + y * dx + x) * 4;
                const dstOff = (x + y * dx + z * dx * dy) * 4;
                out[dstOff]     = data[srcOff];
                out[dstOff + 1] = data[srcOff + 1];
                out[dstOff + 2] = data[srcOff + 2];
                out[dstOff + 3] = data[srcOff + 3];
            }
    return out;
}

/* ── init ──────────────────────────────────────────────────── */

async function initVolumeRenderer() {
    const st = window.NV;
    if (!st.ready) return;

    const container = document.getElementById("panel-3d");
    const canvas3d  = document.getElementById("canvas-3d");

    const testCtx = canvas3d.getContext("webgl2");
    if (!testCtx) {
        container.querySelector(".panel-label").textContent = "3D (WebGL2 required)";
        return;
    }

    const maxDim = Math.max(...st.shape);
    const ds = Math.max(1, Math.round(maxDim / 256));

    const cf = typeof cachedFetch === "function" ? cachedFetch : async (u) => fetch(u);

    /* ── fetch raw chunk ──────────────────────────────────────── */
    let volData, volShape, volSpacing;
    try {
        const r = await cf(`/api/volumes/${st.vid}/chunk?downsample=${ds}`, "3D raw chunk");
        volShape   = r.headers.get("X-Shape").split(",").map(Number);
        volSpacing = r.headers.get("X-Spacing").split(",").map(Number);
        volData = new Uint8Array(await r.arrayBuffer());
    } catch (e) {
        console.error("3D chunk load failed", e);
        container.querySelector(".panel-label").textContent = "3D: load failed";
        return;
    }

    const [dz, dy, dx] = volShape;
    const [sz, sy, sx] = volSpacing;

    /* ── fetch seg chunks ─────────────────────────────────────── */
    let segData = null, segIdData = null;
    let hasSeg = false;
    try {
        const r2 = await cf(`/api/volumes/${st.vid}/seg_chunk?downsample=${ds}`, "3D seg chunk");
        if (r2.ok) {
            segData = new Uint8Array(await r2.arrayBuffer());
            hasSeg = true;
        }
    } catch (_) {}

    try {
        const r3 = await cf(`/api/volumes/${st.vid}/seg_id_chunk?downsample=${ds}`, "3D seg IDs");
        if (r3.ok) {
            segIdData = new Uint8Array(await r3.arrayBuffer());
        }
    } catch (_) {}

    /* ── reorder for Three.js textures ────────────────────────── */
    const rawTex = reorderZYX(volData, dz, dy, dx);
    const segTex = hasSeg ? reorderZYX_RGBA(segData, dz, dy, dx) : new Uint8Array(dx * dy * dz * 4);
    const segIdTex = segIdData ? reorderZYX(segIdData, dz, dy, dx) : new Uint8Array(dx * dy * dz);

    /* ── physical extents ─────────────────────────────────────── */
    const pX = dx * sx, pY = dy * sy, pZ = dz * sz;
    const M = Math.max(pX, pY, pZ);
    const nx = pX / M, ny = pY / M, nz = pZ / M;

    /* ── Three.js scene ───────────────────────────────────────── */
    const rect = container.getBoundingClientRect();
    const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(rect.width, rect.height);

    const scene  = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f4);
    const camera = new THREE.PerspectiveCamera(50, rect.width / rect.height, 0.001, 50);

    /* ── 3D textures (for slice planes) ───────────────────────── */
    const Tex3D = THREE.Data3DTexture || THREE.DataTexture3D;

    const tRaw = new Tex3D(rawTex, dx, dy, dz);
    tRaw.format = THREE.RedFormat; tRaw.type = THREE.UnsignedByteType;
    tRaw.minFilter = THREE.LinearFilter; tRaw.magFilter = THREE.LinearFilter;
    tRaw.unpackAlignment = 1; tRaw.needsUpdate = true;

    const tSeg = new Tex3D(segTex, dx, dy, dz);
    tSeg.format = THREE.RGBAFormat; tSeg.type = THREE.UnsignedByteType;
    tSeg.minFilter = THREE.NearestFilter; tSeg.magFilter = THREE.NearestFilter;
    tSeg.unpackAlignment = 1; tSeg.needsUpdate = true;

    const tSegId = new Tex3D(segIdTex, dx, dy, dz);
    tSegId.format = THREE.RedFormat; tSegId.type = THREE.UnsignedByteType;
    tSegId.minFilter = THREE.NearestFilter; tSegId.magFilter = THREE.NearestFilter;
    tSegId.unpackAlignment = 1; tSegId.needsUpdate = true;

    const SEL_SIZE = 256;
    const selMaskData = new Uint8Array(SEL_SIZE);
    const tSelMask = new THREE.DataTexture(selMaskData, SEL_SIZE, 1, THREE.RedFormat, THREE.UnsignedByteType);
    tSelMask.minFilter = THREE.NearestFilter; tSelMask.magFilter = THREE.NearestFilter;
    tSelMask.needsUpdate = true;

    /* ── auto-window ──────────────────────────────────────────── */
    const sorted = Array.from(volData).sort((a, b) => a - b);
    const wLo = sorted[Math.floor(sorted.length * 0.02)] / 255;
    const wHi = sorted[Math.floor(sorted.length * 0.98)] / 255;

    const selUniforms = {
        uSegId:       { value: tSegId },
        uSelMask:     { value: tSelMask },
        uSelMaskSize: { value: SEL_SIZE },
        uHasSelection:{ value: false },
    };

    /* ── lighting (soft studio style) ────────────────────────── */
    scene.add(new THREE.HemisphereLight(0xffffff, 0xd0d0d8, 0.7));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(2, 3, 2);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xe8e8f0, 0.4);
    fillLight.position.set(-2, 0.5, -1);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
    rimLight.position.set(0, -2, 1);
    scene.add(rimLight);

    /* ── Gaussian splat state ─────────────────────────────────── */
    let splatPoints = null;
    const sigmaX = (nx / dx) * 0.75;
    const sigmaY = (ny / dy) * 0.75;
    const sigmaZ = (nz / dz) * 0.75;

    let slicingMode = false;
    let cuttingMode = true;
    const cutDir = [1, 1, 1];  // +1 = keep [0..slice], -1 = keep [slice..max]
    const cutMin = new THREE.Vector3(0, 0, 0);
    const cutMax = new THREE.Vector3(nx, ny, nz);

    const splatMat = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: SPLAT_VERT,
        fragmentShader: SPLAT_FRAG,
        uniforms: {
            uSigma:        { value: new THREE.Vector3(sigmaX, sigmaY, sigmaZ) },
            uScreenHeight: { value: rect.height },
            uOpacity:      { value: 0.15 },
            uCulling:      { value: true },
            uCullMin:      { value: cutMin },
            uCullMax:      { value: cutMax },
        },
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.NormalBlending,
    });

    function isSurface(off, id, x, y, z) {
        if (x === 0 || x === dx - 1 || y === 0 || y === dy - 1 || z === 0 || z === dz - 1) return true;
        const s = dy * dx;
        return segIdData[off - 1]  !== id || segIdData[off + 1]  !== id ||
               segIdData[off - dx] !== id || segIdData[off + dx] !== id ||
               segIdData[off - s]  !== id || segIdData[off + s]  !== id;
    }

    let splatFilled = true;

    function buildSplats(selectedIdSet) {
        if (splatPoints) {
            scene.remove(splatPoints);
            splatPoints.geometry.dispose();
            splatPoints = null;
        }
        if (!segIdData || !segData || selectedIdSet.size === 0) return;

        const filled = splatFilled;
        let totalVox = 0;
        for (let z = 0; z < dz; z++)
            for (let y = 0; y < dy; y++)
                for (let x = 0; x < dx; x++) {
                    const off = z * dy * dx + y * dx + x;
                    const id = segIdData[off];
                    if (!selectedIdSet.has(id)) continue;
                    if (!filled && !isSurface(off, id, x, y, z)) continue;
                    totalVox++;
                }
        if (totalVox === 0) return;

        const MAX_SPLATS = 300000;
        const stride = Math.max(1, Math.ceil(totalVox / MAX_SPLATS));
        const estCount = Math.ceil(totalVox / stride);

        const positions = new Float32Array(estCount * 3);
        const colors    = new Float32Array(estCount * 3);
        let cnt = 0, skip = 0;

        for (let z = 0; z < dz; z++) {
            for (let y = 0; y < dy; y++) {
                for (let x = 0; x < dx; x++) {
                    const off = z * dy * dx + y * dx + x;
                    const id = segIdData[off];
                    if (!selectedIdSet.has(id)) continue;
                    if (!filled && !isSurface(off, id, x, y, z)) continue;

                    skip++;
                    if (stride > 1 && skip % stride !== 0) continue;

                    positions[cnt * 3]     = (x + 0.5) / dx * nx;
                    positions[cnt * 3 + 1] = (y + 0.5) / dy * ny;
                    positions[cnt * 3 + 2] = (z + 0.5) / dz * nz;

                    const rgbaOff = off * 4;
                    colors[cnt * 3]     = segData[rgbaOff]     / 255;
                    colors[cnt * 3 + 1] = segData[rgbaOff + 1] / 255;
                    colors[cnt * 3 + 2] = segData[rgbaOff + 2] / 255;

                    cnt++;
                    if (cnt >= estCount) break;
                }
                if (cnt >= estCount) break;
            }
            if (cnt >= estCount) break;
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions.slice(0, cnt * 3), 3));
        geom.setAttribute("color",    new THREE.BufferAttribute(colors.slice(0, cnt * 3), 3));

        splatMat.uniforms.uOpacity.value = filled ? 0.15 : 0.35;
        splatPoints = new THREE.Points(geom, splatMat);
        splatPoints.renderOrder = 20;
        scene.add(splatPoints);
        console.log(`Gaussian splats: ${cnt} points (${filled ? "filled" : "surface"}, stride ${stride})`);
    }

    /* ── 3 orthogonal slice planes ────────────────────────────── */
    const sharedPlane = {
        uVolume:     { value: tRaw },
        uSeg:        { value: tSeg },
        ...selUniforms,
        uWinLo:      { value: wLo },
        uWinHi:      { value: wHi },
        uSegOpacity: { value: 0.7 },
        uShowSeg:    { value: hasSeg },
    };

    function makePlane(axis) {
        const geom = new THREE.BufferGeometry();
        let positions;
        if (axis === 0) {
            positions = new Float32Array([
                -nx/2, -ny/2, 0,   nx/2, -ny/2, 0,
                -nx/2,  ny/2, 0,   nx/2,  ny/2, 0,
            ]);
        } else if (axis === 1) {
            positions = new Float32Array([
                -nx/2, 0, -nz/2,   nx/2, 0, -nz/2,
                -nx/2, 0,  nz/2,   nx/2, 0,  nz/2,
            ]);
        } else {
            positions = new Float32Array([
                0, -ny/2, -nz/2,   0, ny/2, -nz/2,
                0, -ny/2,  nz/2,   0, ny/2,  nz/2,
            ]);
        }
        const uvs = new Float32Array([0,0, 1,0, 0,1, 1,1]);
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geom.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
        geom.setIndex([0, 1, 2, 2, 1, 3]);

        const mat = new THREE.RawShaderMaterial({
            glslVersion: THREE.GLSL3,
            vertexShader: PLANE_VERT, fragmentShader: PLANE_FRAG,
            uniforms: {
                ...sharedPlane,
                uAxis:      { value: axis },
                uSliceFrac: { value: 0.5 },
            },
            side: THREE.DoubleSide, transparent: true,
        });

        const mesh = new THREE.Mesh(geom, mat);
        mesh.renderOrder = 10;
        scene.add(mesh);
        return { mesh, mat };
    }

    const planes = [makePlane(0), makePlane(1), makePlane(2)];

    /* ── ray-marched volume cube ────────────────────────────── */
    const volBoxGeom = new THREE.BoxGeometry(nx, ny, nz);
    const volUniforms = {
        uVolume:     { value: tRaw },
        uSeg:        { value: tSeg },
        uBoxMin:     { value: new THREE.Vector3(0, 0, 0) },
        uBoxMax:     { value: new THREE.Vector3(nx, ny, nz) },
        uCutMin:     { value: cutMin },
        uCutMax:     { value: cutMax },
        uCutting:    { value: cuttingMode },
        uCamPos:     { value: new THREE.Vector3() },
        uWinLo:      { value: wLo },
        uWinHi:      { value: wHi },
        uSegOpacity: { value: 0.0 },
        uShowSeg:    { value: false },
        uDensity:    { value: 3.0 },
    };
    const volMat = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: VOL_VERT,
        fragmentShader: VOL_FRAG,
        uniforms: volUniforms,
        transparent: true,
        depthWrite: false,
        side: THREE.BackSide,
    });
    const volMesh = new THREE.Mesh(volBoxGeom, volMat);
    volMesh.position.set(nx / 2, ny / 2, nz / 2);
    volMesh.renderOrder = 5;
    scene.add(volMesh);

    function updatePlanes() {
        const s = st.sliceIdx;
        const fZ = (s[0] + 0.5) / st.shape[0];
        const fY = (s[1] + 0.5) / st.shape[1];
        const fX = (s[2] + 0.5) / st.shape[2];

        planes[0].mat.uniforms.uSliceFrac.value = fZ;
        planes[0].mesh.position.set(nx / 2, ny / 2, fZ * nz);
        planes[1].mat.uniforms.uSliceFrac.value = fY;
        planes[1].mesh.position.set(nx / 2, fY * ny, nz / 2);
        planes[2].mat.uniforms.uSliceFrac.value = fX;
        planes[2].mesh.position.set(fX * nx, ny / 2, nz / 2);

        const posX = fX * nx, posY = fY * ny, posZ = fZ * nz;
        if (cuttingMode) {
            cutMin.set(
                cutDir[0] > 0 ? 0 : posX,
                cutDir[1] > 0 ? 0 : posY,
                cutDir[2] > 0 ? 0 : posZ,
            );
            cutMax.set(
                cutDir[0] > 0 ? posX : nx,
                cutDir[1] > 0 ? posY : ny,
                cutDir[2] > 0 ? posZ : nz,
            );
            volUniforms.uCutting.value = true;
        } else {
            volUniforms.uCutting.value = false;
        }
        splatMat.uniforms.uCulling.value = false;
    }
    updatePlanes();

    window._updateSlicePlanes = updatePlanes;

    window._set3dSegOverlay = function(show, opacity) {
        planes.forEach(p => {
            p.mat.uniforms.uShowSeg.value = show;
            p.mat.uniforms.uSegOpacity.value = opacity;
        });
        volUniforms.uShowSeg.value = show;
        volUniforms.uSegOpacity.value = opacity;
    };

    function updateVisibility() {
        planes.forEach(p => { p.mesh.visible = slicingMode; });
        volMesh.visible = cuttingMode;
    }
    updateVisibility();

    window._set3dSelection = function(selectedIds) {
        selMaskData.fill(0);
        for (const id of selectedIds) {
            const idx = id % SEL_SIZE;
            if (idx >= 0 && idx < SEL_SIZE) selMaskData[idx] = 255;
        }
        tSelMask.needsUpdate = true;

        const hasSel = selectedIds.length > 0;
        planes.forEach(p => {
            p.mat.uniforms.uHasSelection.value = hasSel;
        });

        buildSplats(new Set(selectedIds));
    };

    window._set3dFilled = function(filled) {
        splatFilled = filled;
        console.log("Filled mode:", filled);
        const st = window.NV;
        if (st.selected.size > 0) {
            buildSplats(new Set(st.selected));
        }
    };

    window._set3dSlicing = function(enabled) {
        slicingMode = enabled;
        console.log("Slicing planes:", enabled);
        updateVisibility();
    };

    window._set3dCutting = function(enabled) {
        cuttingMode = enabled;
        volUniforms.uCutting.value = enabled;
        console.log("Cutting mode:", enabled);
        updateVisibility();
        updatePlanes();
    };

    window._flipCutDir = function(axis) {
        cutDir[axis] *= -1;
        const labels = ["X", "Y", "Z"];
        console.log(`Cut direction ${labels[axis]}: ${cutDir[axis] > 0 ? "keep min→slice" : "keep slice→max"}`);
        updatePlanes();
    };

    /* ── keyboard: flip cut direction per axis (X / Y / Z) ───── */
    window.addEventListener("keydown", e => {
        if (e.key === "x" || e.key === "X") window._flipCutDir(2);
        else if (e.key === "y" || e.key === "Y") window._flipCutDir(1);
        else if (e.key === "z" || e.key === "Z") window._flipCutDir(0);
    });

    /* ── bounding box wireframe ───────────────────────────────── */
    const bbGeom = new THREE.BoxGeometry(nx, ny, nz);
    const bbEdge = new THREE.EdgesGeometry(bbGeom);
    const bbLine = new THREE.LineSegments(bbEdge, new THREE.LineBasicMaterial({ color: 0xb0b0b8 }));
    bbLine.position.set(nx / 2, ny / 2, nz / 2);
    scene.add(bbLine);

    /* ── axis indicator ───────────────────────────────────────── */
    const axLen = 0.12;
    const axHelper = new THREE.Group();
    axHelper.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(axLen,0,0)]),
        new THREE.LineBasicMaterial({ color: 0xcc3333 })
    ));
    axHelper.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,axLen,0)]),
        new THREE.LineBasicMaterial({ color: 0x33aa33 })
    ));
    axHelper.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,axLen)]),
        new THREE.LineBasicMaterial({ color: 0x3366cc })
    ));
    axHelper.position.set(-0.05, -0.05, -0.05);
    scene.add(axHelper);

    console.log("Splat renderer initialised. Box:", nx.toFixed(3), ny.toFixed(3), nz.toFixed(3));
    container.querySelector(".panel-label").textContent = "3D Volume";

    /* ── orbit / pan / zoom ───────────────────────────────────── */
    let dragMode = null, prevX = 0, prevY = 0;
    let theta = 0.8, phi = 1.0, radius = 1.8;
    const target = new THREE.Vector3(nx / 2, ny / 2, nz / 2);

    function updateCamera() {
        camera.position.set(
            target.x + radius * Math.sin(phi) * Math.cos(theta),
            target.y + radius * Math.cos(phi),
            target.z + radius * Math.sin(phi) * Math.sin(theta),
        );
        camera.lookAt(target);
    }
    updateCamera();

    canvas3d.addEventListener("contextmenu", e => e.preventDefault());
    canvas3d.addEventListener("mousedown", e => {
        prevX = e.clientX; prevY = e.clientY;
        if (e.button === 0)      dragMode = "rotate";
        else if (e.button === 2) dragMode = "pan";
    });
    window.addEventListener("mouseup", () => { dragMode = null; });
    canvas3d.addEventListener("mousemove", e => {
        if (!dragMode) return;
        const ddx = e.clientX - prevX, ddy = e.clientY - prevY;
        prevX = e.clientX; prevY = e.clientY;
        if (dragMode === "rotate") {
            theta += ddx * 0.008;
            phi   -= ddy * 0.008;
            phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi));
        } else {
            const sp = radius * 0.002;
            const right = new THREE.Vector3(), fwd = new THREE.Vector3();
            fwd.subVectors(target, camera.position).normalize();
            right.crossVectors(camera.up, fwd).normalize();
            const up = new THREE.Vector3().crossVectors(fwd, right).normalize();
            target.addScaledVector(right, -ddx * sp);
            target.addScaledVector(up, ddy * sp);
        }
        updateCamera();
    });
    canvas3d.addEventListener("wheel", e => {
        e.preventDefault();
        radius *= e.deltaY > 0 ? 1.1 : 0.91;
        radius = Math.max(0.1, Math.min(10, radius));
        updateCamera();
    }, { passive: false });

    /* ── render loop ──────────────────────────────────────────── */
    function animate() {
        requestAnimationFrame(animate);
        volUniforms.uCamPos.value.copy(camera.position);
        renderer.render(scene, camera);
    }
    animate();

    new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        if (width > 0 && height > 0) {
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            splatMat.uniforms.uScreenHeight.value = height;
        }
    }).observe(container);
}
