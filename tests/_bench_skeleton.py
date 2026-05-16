"""Throwaway micro-benchmark for the SkeletonGeometryd hotpath.

Times two slices of the per-crop work that dominates dataloader cost:

* ``geom``  -- full ``compute_skeleton_geometry`` (skl + dir + cov + rad).
* ``edt``   -- per-instance EDT-with-indices loop, full-volume vs bbox.
* ``--baseline`` -- inline pre-optimisation reference for the full
  pipeline (per-instance kimimaro on the full crop, full-crop EDT per
  instance, per-tri-pair bincount).  Required for a like-for-like
  before/after on the kimimaro path.

Synthetic 80x256x256 crop with cylinder-like instances matches the
SNEMI3D production patch shape.  Not part of the test suite.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from brainbow.transforms.edt import edt_with_indices
from brainbow.transforms.skeleton import (
    _bbox_slices,
    _kimimaro_available,
    _skeletonize_with_skimage,
    compute_skeleton_field,
    compute_skeleton_geometry,
)


def make_synthetic_crop(
    shape=(80, 256, 256), n_instances: int = 20, radius: int = 4,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D, H, W = shape
    label = np.zeros(shape, dtype=np.int64)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for uid in range(1, n_instances + 1):
        cy = int(rng.integers(radius + 2, H - radius - 2))
        cx = int(rng.integers(radius + 2, W - radius - 2))
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        label[:, disk] = uid
    return label


# ---------------------------------------------------------------------
# Pre-optimisation reference -- preserved inline so we can A/B without
# git-stashing.  Mirrors the OLD compute_skeleton_geometry shape:
# per-instance kimimaro on the full crop + full-crop EDT per instance.
# ---------------------------------------------------------------------
def _baseline_skeleton_geometry(
    label: np.ndarray,
    *,
    radius_normalize: bool = True,
    cov_normalized: bool = True,
) -> Dict[str, np.ndarray]:
    label_np = np.asarray(label, dtype=np.int64)
    S = label_np.ndim
    n_tri = S * (S + 1) // 2
    shape = label_np.shape

    skl_field = np.zeros((1,) + shape, dtype=np.float32)
    dir_field = np.zeros((S,) + shape, dtype=np.float32)
    cov_field = np.zeros((n_tri,) + shape, dtype=np.float32)
    rad_field = np.zeros((1,) + shape, dtype=np.float32)

    uids = np.unique(label_np)
    uids = uids[uids > 0]
    if len(uids) == 0:
        return {
            "skl": skl_field, "dir": dir_field,
            "cov": cov_field, "rad": rad_field,
        }

    use_kimimaro = _kimimaro_available() and S == 3
    tri_pairs = [(i, j) for i in range(S) for j in range(i, S)]

    for uid in uids:
        mask = (label_np == int(uid))
        if not mask.any():
            continue

        # Per-instance kimimaro on the FULL crop (the old shape).
        skel = None
        if use_kimimaro:
            try:
                import kimimaro
                labels_one = mask.astype(np.uint32, copy=False)
                skels = kimimaro.skeletonize(
                    labels_one,
                    dust_threshold=0,
                    parallel=0,
                    progress=False,
                )
                tmp = np.zeros(shape, dtype=np.float32)
                if skels:
                    for _, sk in skels.items():
                        verts = np.rint(sk.vertices).astype(np.int64)
                        verts = np.clip(
                            verts, 0, np.array(shape) - 1,
                        )
                        tmp[verts[:, 0], verts[:, 1], verts[:, 2]] = 1.0
                skel = tmp
            except Exception:
                skel = None
        if skel is None:
            skel = _skeletonize_with_skimage(mask)
        if skel.sum() == 0:
            skel = mask.astype(np.float32)
        skl_field[0] = np.maximum(skl_field[0], skel)

        # Full-crop EDT per instance (the other half of the old shape).
        s_i = mask & (skel > 0)
        if not s_i.any():
            continue
        dist, idx = edt_with_indices(~s_i)
        dist = np.asarray(dist, dtype=np.float32)

        flat_fg = np.flatnonzero(mask)
        v_coords = np.stack(
            np.unravel_index(flat_fg, shape), axis=0,
        ).astype(np.float32)
        s_coords = np.stack(
            [idx[k].ravel()[flat_fg] for k in range(S)],
            axis=0,
        ).astype(np.float32)
        offsets = s_coords - v_coords
        mags = np.sqrt(np.sum(offsets ** 2, axis=0))
        mags_safe = np.where(mags > 1e-8, mags, 1.0)

        unit = offsets / mags_safe
        unit = np.where(mags > 1e-8, unit, 0.0)
        for k in range(S):
            dir_field[k][mask] = unit[k]

        rad_fg = mags.copy()
        if radius_normalize and rad_fg.size:
            max_r = float(rad_fg.max())
            if max_r > 1e-8:
                rad_fg = rad_fg / max_r
        rad_field[0][mask] = rad_fg

        s_int = s_coords.astype(np.int64)
        cell_lin = np.ravel_multi_index(s_int, shape)
        _, inverse = np.unique(cell_lin, return_inverse=True)
        n_cells = int(inverse.max()) + 1 if inverse.size else 0
        if n_cells == 0:
            continue
        counts = np.bincount(
            inverse, minlength=n_cells,
        ).astype(np.float32)

        cov6 = np.zeros((n_tri, n_cells), dtype=np.float32)
        for c, (i, j) in enumerate(tri_pairs):
            p = (offsets[i] * offsets[j]).astype(np.float32)
            cov6[c] = np.bincount(
                inverse, weights=p, minlength=n_cells,
            ).astype(np.float32)
        cov6 = cov6 / np.maximum(counts, 1.0)

        if cov_normalized:
            diag_channels = [
                c for c, (i, j) in enumerate(tri_pairs) if i == j
            ]
            trace = cov6[diag_channels].sum(axis=0)
            trace_safe = np.where(trace > 1e-8, trace, 1.0)
            cov6 = cov6 / trace_safe[None, :]

        cov_per_voxel = cov6[:, inverse]
        for c in range(n_tri):
            cov_field[c][mask] = cov_per_voxel[c]

    return {
        "skl": skl_field, "dir": dir_field,
        "cov": cov_field, "rad": rad_field,
    }


def _edt_full_volume(label: np.ndarray, skl: np.ndarray) -> float:
    """Reference: per-instance EDT-with-indices on the FULL crop."""
    skl_bool = skl[0] > 0
    uids = np.unique(label)
    uids = uids[uids > 0]
    t0 = time.perf_counter()
    for uid in uids:
        m_i = label == int(uid)
        s_i = m_i & skl_bool
        if not s_i.any():
            continue
        edt_with_indices(~s_i)
    return (time.perf_counter() - t0) * 1000.0


def _edt_bbox(label: np.ndarray, skl: np.ndarray) -> float:
    """Optimised: per-instance EDT confined to the instance bbox."""
    skl_bool = skl[0] > 0
    uids = np.unique(label)
    uids = uids[uids > 0]
    t0 = time.perf_counter()
    for uid in uids:
        m_i = label == int(uid)
        slices = _bbox_slices(m_i)
        if slices is None:
            continue
        sub_m = m_i[slices]
        sub_skl = skl_bool[slices] & sub_m
        if not sub_skl.any():
            continue
        edt_with_indices(~sub_skl)
    return (time.perf_counter() - t0) * 1000.0


def _time(fn, label, n_runs: int = 3, warmup: bool = True) -> tuple[float, dict]:
    if warmup:
        fn(label)  # kimimaro lazily compiles on first call
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = fn(label)
    return (time.perf_counter() - t0) / n_runs * 1000.0, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="80,256,256")
    parser.add_argument("--n-instances", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also time the inline pre-optimisation reference.",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip the warm-up call (use when each run is very slow).",
    )
    args = parser.parse_args()

    shape = tuple(int(s) for s in args.shape.split(","))
    label = make_synthetic_crop(
        shape=shape, n_instances=args.n_instances,
    )
    n_uids = int(np.unique(label).size - 1)
    n_fg = int((label > 0).sum())
    crop_vox = int(np.prod(shape))
    print(
        f"crop shape={label.shape}  instances={n_uids}  "
        f"fg_voxels={n_fg}  crop_voxels={crop_vox}",
    )
    print(f"kimimaro available: {_kimimaro_available()}")

    print("\n--- compute_skeleton_geometry (full pipeline) ---")
    opt_ms, opt_out = _time(
        compute_skeleton_geometry, label,
        n_runs=args.n_runs, warmup=not args.no_warmup,
    )
    print(f"  optimised: {opt_ms:.1f} ms/crop  (avg of {args.n_runs} runs)")

    if args.baseline:
        base_ms, base_out = _time(
            _baseline_skeleton_geometry, label,
            n_runs=args.n_runs, warmup=not args.no_warmup,
        )
        print(
            f"  baseline : {base_ms:.1f} ms/crop  "
            f"(avg of {args.n_runs} runs)",
        )
        print(f"  speedup  : {base_ms / opt_ms:.1f}x")

        # Field-sum parity sanity check (exact correctness in the test
        # suite -- this is a coarse sentinel only).
        for k in opt_out:
            print(
                f"    field={k:>3}  opt_sum={float(opt_out[k].sum()):.3e}  "
                f"base_sum={float(base_out[k].sum()):.3e}",
            )

    print("\n--- per-instance EDT-with-indices loop ---")
    skl = compute_skeleton_field(label)
    print(
        f"  full-volume EDT (baseline): {_edt_full_volume(label, skl):.1f} ms",
    )
    print(
        f"  bbox EDT (optimised):       {_edt_bbox(label, skl):.1f} ms",
    )


if __name__ == "__main__":
    main()
