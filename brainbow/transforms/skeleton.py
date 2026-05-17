"""Skeleton-derived targets for the unified head.

For each foreground instance in a label map this module computes four
fields, all keyed off a single per-instance Euclidean distance
transform whose seeds are the instance's skeleton voxels:

* ``skl`` -- the 1-voxel-wide centerline ("skeleton"), unioned across
  instances into a single binary mask.  Supervised by BCE + Dice on
  the head's ``skl`` slot.
* ``dir`` -- per-voxel **unit centripetal vector** to the nearest
  same-instance skeleton voxel ``s*(v)``.  Zero at skeleton voxels
  themselves.  Supervised by foreground-only L1 on the head's ``dir``
  slot.  Sign convention: ``s* - v`` (vector points **toward** the
  centerline; same direction-of-travel convention as the centroid-
  based direction field this transform replaces).
* ``cov`` -- upper-tri of the per-Voronoi-cell second-moment matrix,
  centered at the skeleton vertex ``s_j``: ``M_j = Σ_v (v - s_j)(v -
  s_j)^T / |cell_j|`` over all voxels of instance ``i`` whose nearest
  skeleton voxel is ``s_j``.  Trace-normalised by default
  (scale-invariant).  Broadcast to every voxel in the cell.
  Supervised by foreground-only L1 on the head's ``cov`` slot.
* ``rad`` -- per-voxel ridge field encoding distance to the nearest
  same-instance skeleton voxel.  Supervised by foreground-only L1 on
  the head's ``rad`` slot.  Two orthogonal knobs:

    radius_normalize=True   (default): divide the per-instance EDT by
                                      its own max so the field is
                                      scale-invariant.
    radius_normalize=False             : keep raw EDT in voxel units.

    radius_invert=True      (default, only honoured when
                            ``radius_normalize=True``): emit the
                            **inverted** field ``rad(v) = 1 - dist/R_i``
                            so the skeleton is the peak (``rad = 1``),
                            the instance boundary tapers to ``rad ≈ 0``,
                            and background is also ``0`` -- continuous
                            across the foreground / background
                            interface, which is what the foreground-only
                            L1 actually wants to see.
    radius_invert=False                 : keep the raw distance field
                            (skeleton == 0, boundary == 1, background
                            == 0) -- the legacy behaviour.

In voxel units (``radius_normalize=False``) ``radius_invert`` is
ignored and the field is always the raw distance, since
``1 - dist`` only makes sense once you've divided by ``R_i``.

The four fields satisfy the per-voxel reconstruction identity::

    rad(v) * dir(v) == s*(v) - v             (un-normalised mode)
    rad(v) * dir(v) == (s*(v) - v) / R_i     (normalised, invert=False)

which doubles as a unit test for the per-instance pipeline.  The
inverted-normalised mode breaks this identity by construction
(``rad`` is now ``1 - dist/R_i``); in that mode the offset is
recovered via ``(1 - rad) * R_i * dir``.

Backends
--------
Two skeletonization backends are supported and tried in order:

1. ``kimimaro.skeletonize`` -- 3-D TEASAR.  Used when available and the
   array is 3-D.  Runs single-threaded (``parallel=0``) so it is safe
   inside MONAI's forked DataLoader workers.  Anisotropy is **not**
   passed in: we keep the skeleton in voxel coordinates so the rest of
   the pipeline (per-voxel direction, radius, etc.) stays consistent.
2. ``skimage.morphology.skeletonize`` -- pure-CPU, works in both 2-D and
   3-D.  Used as the fallback when kimimaro is unavailable, or for 2-D
   slice-mode inputs where kimimaro is not applicable.

Output shapes
-------------
* ``label_skl``        -- ``[1, *spatial]`` float32 binary mask.
* ``label_direction``  -- ``[S, *spatial]`` float32 unit-direction
  field (``S`` = spatial rank).
* ``label_covariance`` -- ``[S*(S+1)/2, *spatial]`` float32
  upper-tri Voronoi-cell moments.
* ``label_radius``     -- ``[1, *spatial]`` float32 distance field.

All four keys carry a leading channel dim so that
``EnsureChannelFirstd`` / ``EnsureTyped``-style downstream consumers
see a tensor-of-rank ``spatial_dims + 1`` matching ``label``.

Performance
-----------
Two optimisations keep the per-crop dataloader cost ~O(Σ|instance|)
instead of ``O(N_instances * |crop|)``:

1. **Single multi-label kimimaro call.**  ``kimimaro.skeletonize``
   accepts a multi-label volume and skeletonises every label > 0 in one
   pass; we no longer iterate per-instance with a single-id mask.
2. **Per-instance bounding box EDT.**  The Euclidean distance transform
   that drives ``rad``/``dir``/``cov`` runs on the cropped sub-volume
   bounding the instance (with a 1-voxel margin so the EDT sees the
   exterior boundary correctly), not on the full crop.  For
   SNEMI3D-shaped 80×256×256 crops with 20 small instances this is
   typically a 50–200x speedup on the EDT path.

Extending this module
---------------------
A soft (Gaussian-blurred) ``skl`` target can be added in a follow-up:
post-process the binary mask with
:func:`brainbow.transforms.edt.gaussian_filter` and switch the head's
loss to L1.  The plan keeps v1 strictly binary.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform

from brainbow.transforms.edt import distance_transform_edt, edt_with_indices


_KIMIMARO_AVAILABLE: Optional[bool] = None


def _kimimaro_available() -> bool:
    """Probe kimimaro once per process and cache the result."""
    global _KIMIMARO_AVAILABLE
    if _KIMIMARO_AVAILABLE is None:
        try:
            import kimimaro  # noqa: F401
            _KIMIMARO_AVAILABLE = True
        except ImportError:
            _KIMIMARO_AVAILABLE = False
    return _KIMIMARO_AVAILABLE


def _skeletonize_with_skimage(mask: np.ndarray) -> np.ndarray:
    """Per-instance skimage skeletonization (2-D and 3-D)."""
    from skimage.morphology import skeletonize
    return skeletonize(mask.astype(bool)).astype(np.float32)


def _skeletonize_all_kimimaro(
    label: np.ndarray,
    *,
    dust_threshold: int,
) -> Dict[int, np.ndarray]:
    """Run kimimaro on the full multi-label volume in a single call.

    Replaces the previous "loop over uids and skeletonize each binary
    mask separately" pattern: kimimaro internally bucketises by label
    so a single call is dramatically cheaper than ``len(uids)``
    independent calls on a copy of the same crop.

    Args:
        label: Integer instance label array ``[*spatial]``.  Background
            is ``0``.  Must be 3-D (kimimaro is volumetric).
        dust_threshold: Drop kimimaro components smaller than this many
            voxels.

    Returns:
        Dict ``{uid: voxel_coords}`` mapping each label > 0 that
        kimimaro returned a skeleton for to an ``[N, 3]`` int64 array
        of skeleton-vertex coordinates **in global crop space**, clipped
        to the volume bounds.  Labels with no skeleton (e.g. dropped by
        ``dust_threshold``) are absent from the dict; the caller is
        responsible for falling back to skimage / a degenerate
        single-voxel skeleton.
    """
    import kimimaro

    labels = np.ascontiguousarray(label.astype(np.uint32, copy=False))
    skels = kimimaro.skeletonize(
        labels,
        dust_threshold=int(dust_threshold),
        # ``fix_branching`` / ``fix_borders`` are quality-improving
        # post-processing passes that we don't need for our 1-voxel
        # rasterised skeleton mask -- we only consume the vertex set.
        # Disabling them roughly halves the kimimaro runtime.
        fix_branching=False,
        fix_borders=False,
        # IMPORTANT: ``parallel=1`` (NOT 0). kimimaro 5.x interprets
        # ``parallel <= 0`` as "fork ``cpu_count()`` subprocesses" (see
        # ``kimimaro/intake.py`` line ~195), which under DataLoader workers
        # spawns thousands of pathos children, each backed by named POSIX
        # shm segments ``kimimaro-shm-{dbf,cc-labels}-*``. Those segments
        # only get unlinked on the success/SIGINT/SIGTERM paths of the
        # parent; any abnormal exit (DataLoader worker tear-down, SIGKILL
        # from systemd, an exception in ``compute_skeleton_geometry``, etc.)
        # orphans them in ``/dev/shm``. Under MONAI's ``forkserver`` workers
        # this leaks at ~2 segments per crop and rapidly fills the
        # ``tmpfs``: a previous run accumulated ~50 k segments in /dev/shm
        # (~984 GB) before OOM-killing the user session at 2026-05-17
        # 04:26:50. ``parallel=1`` short-circuits the parallel codepath
        # entirely (line ~198: ``if parallel == 1: ... return``), no shm
        # ever allocated. Anisotropy intentionally omitted (skeleton stays
        # in voxel coordinates).
        parallel=1,
        progress=False,
    )

    out: Dict[int, np.ndarray] = {}
    if not skels:
        return out
    bounds = np.asarray(label.shape, dtype=np.int64) - 1
    for uid, sk in skels.items():
        verts = np.rint(sk.vertices).astype(np.int64)
        verts = np.clip(verts, 0, bounds)
        out[int(uid)] = verts
    return out


def _bbox_slices(mask: np.ndarray) -> Optional[Tuple[slice, ...]]:
    """Tight bounding-box slice tuple for ``mask`` (or ``None`` if empty).

    For each axis we project ``mask`` onto that axis with ``np.any``
    and read out the first / last True position.  The result is a
    tuple of ``slice(lo, hi+1)`` -- one per axis -- that crops the
    smallest axis-aligned sub-volume containing every True voxel.
    """
    coords_per_axis = []
    for axis in range(mask.ndim):
        reduced_axes = tuple(i for i in range(mask.ndim) if i != axis)
        proj = np.any(mask, axis=reduced_axes)
        nz = np.flatnonzero(proj)
        if nz.size == 0:
            return None
        coords_per_axis.append((int(nz[0]), int(nz[-1]) + 1))
    return tuple(slice(lo, hi) for lo, hi in coords_per_axis)


def _all_bbox_slices(
    label_np: np.ndarray,
    uids: np.ndarray,
) -> Dict[int, Tuple[slice, ...]]:
    """Compute tight bboxes for every label in one full-volume pass.

    Replaces the per-uid ``mask = label == uid`` + ``_bbox_slices(mask)``
    pattern, which is ``O(N_uids * crop_voxels)``.  ``scipy.ndimage.
    find_objects`` does a single linear scan over ``label_np`` and
    returns one slice tuple per consecutive integer id in
    ``[1, label_max]``.  We then filter to the uids actually present.
    """
    from scipy.ndimage import find_objects

    label_max = int(uids.max())
    objects = find_objects(label_np, max_label=label_max)
    out: Dict[int, Tuple[slice, ...]] = {}
    for uid in uids:
        uid_int = int(uid)
        sl = objects[uid_int - 1]
        if sl is not None:
            out[uid_int] = sl
    return out


def compute_skeleton_field(
    label: np.ndarray,
    *,
    dust_threshold: int = 0,
) -> np.ndarray:
    """Compute the union skeleton mask over all foreground instances.

    Args:
        label: Integer instance label array ``[*spatial]``.  Background
            is ``0``.  Both 2-D and 3-D inputs are accepted.
        dust_threshold: Drop kimimaro components smaller than this many
            voxels.  Has no effect on the skimage fallback (which has
            no equivalent knob).  ``0`` keeps every component.

    Returns:
        Binary skeleton mask ``[1, *spatial]`` float32 with the leading
        channel dim.  ``1.0`` at skeleton voxels, ``0.0`` elsewhere.
        Always confined to ``label > 0``.
    """
    label_np = np.asarray(label, dtype=np.int64)
    shape = label_np.shape
    out = np.zeros((1,) + shape, dtype=np.float32)

    uids = np.unique(label_np)
    uids = uids[uids > 0]
    if len(uids) == 0:
        return out

    use_kimimaro = _kimimaro_available() and label_np.ndim == 3

    # One multi-label kimimaro call covers the whole crop.  Falling
    # back instance-by-instance (skimage path) only when kimimaro is
    # unavailable / errors out keeps the slow path identical to before.
    skel_per_uid: Dict[int, np.ndarray] = {}
    if use_kimimaro:
        try:
            skel_per_uid = _skeletonize_all_kimimaro(
                label_np, dust_threshold=dust_threshold,
            )
        except Exception:
            skel_per_uid = {}
            use_kimimaro = False

    # One full-volume pass to get every instance's bbox, instead of
    # ``len(uids)`` independent ``label == uid`` + projection passes.
    bbox_per_uid = _all_bbox_slices(label_np, uids)

    for uid in uids:
        uid_int = int(uid)
        slices = bbox_per_uid.get(uid_int)
        if slices is None:
            continue

        verts = skel_per_uid.get(uid_int)
        if verts is not None and len(verts) > 0:
            # Rasterise the kimimaro vertex set inside the bbox and
            # clip to the per-instance mask so spurious vertices on
            # the instance boundary never bleed into a neighbour.
            origin = np.asarray(
                [s.start for s in slices], dtype=np.int64,
            )
            sub_shape = tuple(s.stop - s.start for s in slices)
            local = verts - origin
            in_bounds = np.all(
                (local >= 0) & (local < np.asarray(sub_shape)),
                axis=1,
            )
            local = local[in_bounds]
            if local.size:
                sub_m = (label_np[slices] == uid_int)
                tmp = np.zeros(sub_shape, dtype=bool)
                if sub_shape and len(local):
                    tmp[tuple(local.T)] = True
                tmp &= sub_m
                if tmp.any():
                    out_sub = out[(0,) + slices]
                    out_sub[tmp] = 1.0
                    continue

        # Fall back to skimage / single-voxel-instance handling on the
        # bbox-cropped mask -- avoids materialising a full-crop bool.
        sub_m = (label_np[slices] == uid_int)
        if not sub_m.any():
            continue
        n_vox = int(sub_m.sum())
        if n_vox == 1:
            out_sub = out[(0,) + slices]
            out_sub[sub_m] = 1.0
            continue

        skel_sub = _skeletonize_with_skimage(sub_m)
        if skel_sub.sum() == 0:
            # Degenerate instance with no skeleton -- fall back to the
            # foreground itself so the loss has a non-empty target.
            skel_sub = sub_m.astype(np.float32)
        out_sub = out[(0,) + slices]
        np.maximum(out_sub, skel_sub, out=out_sub)

    return out


def compute_radius_field(
    label: np.ndarray,
    skl: np.ndarray,
    *,
    normalize: bool = True,
    invert: bool = True,
) -> np.ndarray:
    """Compute the per-voxel distance to the nearest same-instance skeleton.

    For each foreground voxel ``v`` belonging to instance ``i``, the
    distance is to the nearest voxel of ``skl ∩ (label == i)``.
    Background voxels are ``0``.

    The implementation runs one Euclidean distance transform per
    instance with the per-instance skeleton mask as seeds.  Voxels
    outside the instance are not touched, so two touching instances
    can't leak radii into each other.

    Args:
        label: Instance label array ``[*spatial]``.  Background is 0.
        skl: Binary skeleton mask, either ``[*spatial]`` or
            ``[1, *spatial]``.  Skeleton voxels are 1, others 0.
        normalize: When ``True`` (default), each instance's radius
            field is divided by its own max so values land in ``[0, 1]``
            per-instance (scale-invariant).  When ``False``, the field
            stays in voxel units.
        invert: When ``True`` (default) and ``normalize=True``, emit
            the inverted ridge field ``1 - dist/R_i`` so the skeleton
            is the peak and the field decays continuously to ``0`` at
            the instance boundary -- matching the background's ``0``
            and removing the boundary discontinuity that the legacy
            distance-from-skeleton field had.  Ignored when
            ``normalize=False``.

    Returns:
        Radius field ``[1, *spatial]`` float32 with the leading channel
        dim, matching :func:`compute_skeleton_field`.
    """
    label_np = np.asarray(label, dtype=np.int64)
    shape = label_np.shape
    out = np.zeros((1,) + shape, dtype=np.float32)

    skl_np = np.asarray(skl)
    if skl_np.ndim == label_np.ndim + 1:
        skl_np = skl_np[0]
    skl_bool = skl_np > 0

    uids = np.unique(label_np)
    uids = uids[uids > 0]
    if len(uids) == 0:
        return out

    # Single full-volume pass for all bboxes (instead of N per-uid
    # ``label == uid`` + projection passes).
    bbox_per_uid = _all_bbox_slices(label_np, uids)

    for uid in uids:
        uid_int = int(uid)
        slices = bbox_per_uid.get(uid_int)
        if slices is None:
            continue

        # Crop into the instance bounding box and build the per-instance
        # mask there directly -- avoids materialising a full-crop bool.
        sub_m = (label_np[slices] == uid_int)
        if not sub_m.any():
            continue
        sub_skl = skl_bool[slices] & sub_m
        if not sub_skl.any():
            continue

        sub_dist = distance_transform_edt(~sub_skl)
        sub_dist = np.asarray(sub_dist, dtype=np.float32)

        if normalize:
            inside = sub_dist[sub_m]
            max_r = float(inside.max()) if inside.size else 0.0
            if max_r > 1e-8:
                sub_dist = sub_dist / max_r
            if invert:
                # 1 - dist/R_i clipped to [0, 1] -- skeleton == 1,
                # boundary == 0, background == 0 (continuous across
                # the foreground / background interface).
                sub_dist = np.clip(1.0 - sub_dist, 0.0, 1.0)

        out_sub = out[(0,) + slices]
        out_sub[sub_m] = sub_dist[sub_m]

    return out


def compute_skeleton_geometry(
    label: np.ndarray,
    *,
    dust_threshold: int = 0,
    radius_normalize: bool = True,
    radius_invert: bool = True,
    cov_normalized: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute the full skeleton-relative geometry quartet in one pass.

    Per instance ``i``:

    1. Skeletonize ``(label == i)`` and union into the global ``skl``
       mask (see :func:`compute_skeleton_field`).
    2. Run one Euclidean distance transform of ``~(skl ∩ (label == i))``
       with ``return_indices=True`` -- gives, for every voxel ``v`` of
       instance ``i``, both the distance to the nearest skeleton voxel
       and the integer coordinates of that voxel ``s*(v)``.
    3. Derive:
       * ``dir(v) = (s*(v) - v) / ||s*(v) - v||``  (zero at skeleton
         voxels);
       * ``rad(v) = ||s*(v) - v||``  (per-instance normalised to
         ``[0, 1]`` when ``radius_normalize=True``);
       * ``cov(v) = M_{s*(v)}`` where each skeleton vertex ``s_j``
         carries the upper-tri of its Voronoi-cell 2nd-moment matrix
         centered at ``s_j``.  Cells with ``population < 2`` get a
         zero matrix (degenerate; the centerline alone has no
         dispersion).

    Args:
        label: Instance label array ``[*spatial]``.  Background is 0.
        dust_threshold: Forwarded to :func:`compute_skeleton_field`.
        radius_normalize: When ``True`` (default), each instance's
            radius field is divided by its own max so values land in
            ``[0, 1]``.  When ``False`` the field stays in voxel
            units.
        radius_invert: When ``True`` (default) and ``radius_normalize
            =True``, emit ``rad = 1 - dist/R_i`` (skeleton-peaked
            ridge) so background and instance-boundary both sit at 0.
            See module docstring for the rationale.  Ignored when
            ``radius_normalize=False``.
        cov_normalized: When ``True`` (default), each Voronoi-cell
            covariance matrix is divided by its trace, making the
            field scale-invariant.

    Returns:
        Dict with four keys ``"skl"`` ``[1, *spatial]``, ``"dir"``
        ``[S, *spatial]``, ``"cov"`` ``[S*(S+1)/2, *spatial]``,
        ``"rad"`` ``[1, *spatial]`` -- all float32, all aligned to the
        same ``[*spatial]`` grid as ``label``.
    """
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
            "skl": skl_field,
            "dir": dir_field,
            "cov": cov_field,
            "rad": rad_field,
        }

    use_kimimaro = _kimimaro_available() and S == 3

    # Single multi-label kimimaro call instead of N per-instance ones.
    skel_per_uid: Dict[int, np.ndarray] = {}
    if use_kimimaro:
        try:
            skel_per_uid = _skeletonize_all_kimimaro(
                label_np, dust_threshold=dust_threshold,
            )
        except Exception:
            skel_per_uid = {}
            use_kimimaro = False

    # Upper-tri (i, j) pairs in the canonical order used by
    # :func:`brainbow.losses.upper_tri_to_matrix`: ``(0, 0), (0, 1),
    # (0, 2), (1, 1), (1, 2), (2, 2)`` for 3-D, i.e.
    # ``(Czz, Czy, Czx, Cyy, Cyx, Cxx)``.  Stored as 2 × n_tri int
    # arrays so the cov accumulator can do all upper-tri products in
    # one ``offsets[i_idx] * offsets[j_idx]`` call.
    tri_pairs = [(i, j) for i in range(S) for j in range(i, S)]
    tri_i = np.asarray([p[0] for p in tri_pairs], dtype=np.int64)
    tri_j = np.asarray([p[1] for p in tri_pairs], dtype=np.int64)
    diag_channels = np.flatnonzero(tri_i == tri_j)

    # One full-volume pass for every instance's bbox (rather than N
    # ``label == uid`` + axis-projection passes).
    bbox_per_uid = _all_bbox_slices(label_np, uids)

    for uid in uids:
        uid_int = int(uid)
        slices = bbox_per_uid.get(uid_int)
        if slices is None:
            continue
        origin = np.asarray(
            [s.start for s in slices], dtype=np.int64,
        )                                                   # [S]
        sub_shape = tuple(s.stop - s.start for s in slices)
        # Build the per-instance mask directly inside the bbox -- much
        # cheaper than a full-crop ``label == uid`` + slice.
        sub_m = (label_np[slices] == uid_int)
        if not sub_m.any():
            continue

        # ---- Build the per-instance skeleton mask in sub-volume ----
        sub_skel = np.zeros(sub_shape, dtype=bool)
        verts = skel_per_uid.get(uid_int)
        if verts is not None and len(verts) > 0:
            local = verts - origin                          # [N, S]
            # Discard kimimaro vertices that fall outside the bbox
            # (shouldn't happen, but be safe).
            in_bounds = np.all(
                (local >= 0) & (local < np.asarray(sub_shape)),
                axis=1,
            )
            local = local[in_bounds]
            if local.size:
                if S == 3:
                    sub_skel[local[:, 0], local[:, 1], local[:, 2]] = True
                elif S == 2:
                    sub_skel[local[:, 0], local[:, 1]] = True
                else:  # generic
                    sub_skel[tuple(local.T)] = True
                sub_skel &= sub_m

        if not sub_skel.any():
            # No usable kimimaro vertices for this instance -- fall
            # back to skimage skeletonize on the cropped sub-volume.
            n_vox = int(sub_m.sum())
            if n_vox == 1:
                sub_skel = sub_m.copy()
            else:
                fallback = _skeletonize_with_skimage(sub_m)
                sub_skel = fallback > 0
                if not sub_skel.any():
                    sub_skel = sub_m.copy()

        # Stamp the per-instance skeleton into the global mask.
        skl_sub = skl_field[(0,) + slices]
        np.maximum(skl_sub, sub_skel.astype(np.float32), out=skl_sub)

        # ---- Per-instance EDT with indices on the sub-volume ----
        # Seeds are the per-instance skeleton voxels; query voxels are
        # everything else (we only consume the result inside ``sub_m``).
        sub_dist, sub_idx = edt_with_indices(~sub_skel)
        sub_dist = np.asarray(sub_dist, dtype=np.float32)

        flat_fg = np.flatnonzero(sub_m)
        if flat_fg.size == 0:
            continue

        # Coordinates of foreground voxels in the sub-volume.
        v_coords = np.stack(
            np.unravel_index(flat_fg, sub_shape), axis=0,
        ).astype(np.float32)                              # [S, n_fg]

        # Coordinates of each fg voxel's nearest skeleton voxel
        # (still in sub-volume coordinates -- offsets are
        # translation-invariant so this is fine for dir/rad/cov).
        # Reshape ``sub_idx`` from ``[S, *sub_shape]`` to ``[S, prod]``
        # and gather all axes in one indexed read instead of S copies.
        s_coords = np.asarray(
            sub_idx, dtype=np.int64,
        ).reshape(S, -1)[:, flat_fg].astype(np.float32)    # [S, n_fg]

        offsets = s_coords - v_coords                      # [S, n_fg]
        mags = np.sqrt(np.sum(offsets ** 2, axis=0))       # [n_fg]
        mags_safe = np.where(mags > 1e-8, mags, 1.0)

        # ----- dir: per-voxel unit centripetal vector -----
        unit = offsets / mags_safe                         # [S, n_fg]
        unit = np.where(mags > 1e-8, unit, 0.0).astype(np.float32, copy=False)
        dir_sub = dir_field[(slice(None),) + slices]
        for k in range(S):
            dir_sub[k][sub_m] = unit[k]

        # ----- rad: per-voxel distance, optionally normalised + inverted -----
        rad_fg = mags.astype(np.float32, copy=True)
        if radius_normalize:
            max_r = float(rad_fg.max())
            if max_r > 1e-8:
                rad_fg = rad_fg / max_r
            if radius_invert:
                # Skeleton-peaked ridge: 1 at skeleton, 0 at boundary,
                # 0 at background -- continuous across fg/bg interface
                # so the foreground-only L1 isn't fighting a step.
                rad_fg = np.clip(1.0 - rad_fg, 0.0, 1.0)
        rad_sub = rad_field[(0,) + slices]
        rad_sub[sub_m] = rad_fg

        # ----- cov: Voronoi-cell 2nd-moment matrix at each vertex -----
        # Compact cell labelling: hash each fg voxel's nearest skeleton
        # vertex into a dense ``[0, n_cells)`` index via
        # ``np.unique(..., return_inverse=True)``.  Per-cell sums are
        # then ``np.bincount(inverse, weights=...)`` -- cheap because
        # ``n_cells <= len(skeleton voxels in instance)``, not the
        # crop volume.
        s_int = s_coords.astype(np.int64)
        cell_lin = np.ravel_multi_index(s_int, sub_shape)
        _, inverse = np.unique(cell_lin, return_inverse=True)
        n_cells = int(inverse.max()) + 1 if inverse.size else 0
        if n_cells == 0:
            continue
        counts = np.bincount(
            inverse, minlength=n_cells,
        ).astype(np.float32)

        # Vectorised per-tri-pair accumulation.  ``products`` has shape
        # ``[n_tri, n_fg]`` -- one row per upper-tri (i, j) pair.  We
        # bincount per row but with all rows in one call by stacking
        # the rows into a single 1-D array and offsetting their cell
        # ids by row * n_cells.  ``np.bincount`` then returns the
        # concatenated per-row sums in one shot.
        products = (offsets[tri_i] * offsets[tri_j]).astype(np.float32)
        flat_inv = (
            inverse[None, :]
            + np.arange(n_tri, dtype=np.int64)[:, None] * n_cells
        ).ravel()
        cov_flat = np.bincount(
            flat_inv,
            weights=products.ravel(),
            minlength=n_tri * n_cells,
        ).astype(np.float32)
        cov6 = cov_flat.reshape(n_tri, n_cells)
        cov6 = cov6 / np.maximum(counts, 1.0)

        if cov_normalized:
            trace = cov6[diag_channels].sum(axis=0)
            trace_safe = np.where(trace > 1e-8, trace, 1.0)
            cov6 = cov6 / trace_safe[None, :]

        # Broadcast per-cell M to per-voxel and write into the field.
        cov_per_voxel = cov6[:, inverse]                  # [n_tri, n_fg]
        cov_sub = cov_field[(slice(None),) + slices]
        for c in range(n_tri):
            cov_sub[c][sub_m] = cov_per_voxel[c]

    return {
        "skl": skl_field,
        "dir": dir_field,
        "cov": cov_field,
        "rad": rad_field,
    }


class SkeletonGeometryd(MapTransform):
    """Compute skeleton-derived targets for each instance label map.

    Reads instance labels from each key and stores:

    * ``{key}_skl``        -- ``[1, *spatial]`` binary skeleton mask.
    * ``{key}_direction``  -- ``[S, *spatial]`` unit centripetal vector
      to the nearest same-instance skeleton voxel.
    * ``{key}_covariance`` -- ``[S*(S+1)/2, *spatial]`` upper-tri
      Voronoi-cell 2nd-moment matrix at each fg voxel's nearest
      skeleton vertex (trace-normalised by default).
    * ``{key}_radius``     -- ``[1, *spatial]`` per-voxel distance to
      the nearest same-instance skeleton voxel (per-instance
      normalised to ``[0, 1]`` when ``radius_normalize=True``; voxel
      units when ``False``).

    Input labels are expected in ``[C, *spatial]`` format (post
    ``EnsureChannelFirstd``); the first channel is used.

    Args:
        keys: Keys of instance label maps.
        spatial_dims: Number of spatial dimensions (2 or 3).
        dust_threshold: Drop kimimaro components smaller than this many
            voxels.  ``0`` keeps every component.  Ignored on the
            skimage fallback path.
        radius_normalize: Per-instance-normalise the radius field to
            ``[0, 1]`` (default ``True``).  ``False`` keeps it in voxel
            units.
        radius_invert: When ``True`` (default) and ``radius_normalize
            =True``, emit ``1 - dist/R_i`` so the skeleton is the
            peak (``rad = 1``) and background / boundary are both
            ``0``.  Removes the boundary discontinuity that the raw
            distance-from-skeleton field had.  See the module
            docstring for the full rationale.
        cov_normalized: Trace-normalise each Voronoi-cell covariance
            matrix (default ``True``).
    """

    SKL_SUFFIX: str = "_skl"
    DIR_SUFFIX: str = "_direction"
    COV_SUFFIX: str = "_covariance"
    RAD_SUFFIX: str = "_radius"

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
        dust_threshold: int = 0,
        radius_normalize: bool = True,
        radius_invert: bool = True,
        cov_normalized: bool = True,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = int(spatial_dims)
        self.dust_threshold = int(dust_threshold)
        self.radius_normalize = bool(radius_normalize)
        self.radius_invert = bool(radius_invert)
        self.cov_normalized = bool(cov_normalized)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)

            if is_tensor:
                device = arr.device
                label_np = arr.cpu().numpy()
            else:
                device = None
                label_np = np.asarray(arr)

            # Strip leading non-spatial dims (channel) to get [*spatial].
            while label_np.ndim > self.spatial_dims:
                label_np = label_np[0]

            fields = compute_skeleton_geometry(
                label_np,
                dust_threshold=self.dust_threshold,
                radius_normalize=self.radius_normalize,
                radius_invert=self.radius_invert,
                cov_normalized=self.cov_normalized,
            )

            if is_tensor:
                fields = {
                    name: torch.from_numpy(arr).to(device)
                    for name, arr in fields.items()
                }

            d[f"{key}{self.SKL_SUFFIX}"] = fields["skl"]
            d[f"{key}{self.DIR_SUFFIX}"] = fields["dir"]
            d[f"{key}{self.COV_SUFFIX}"] = fields["cov"]
            d[f"{key}{self.RAD_SUFFIX}"] = fields["rad"]

        return d


# Backward-compatible alias for the PR-2 single-key class.  The
# upgraded ``SkeletonGeometryd`` emits the full geometry quartet;
# downstream consumers that only read ``label_skl`` continue to work.
Skeletond = SkeletonGeometryd


__all__ = [
    "SkeletonGeometryd",
    "Skeletond",
    "compute_skeleton_field",
    "compute_skeleton_geometry",
    "compute_radius_field",
]
