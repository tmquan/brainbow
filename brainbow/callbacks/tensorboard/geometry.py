"""Geometry overlays for the unified-head TB panels.

Two renderer families are available; the active one is chosen by the
``geometry_style`` knob plumbed through
:class:`brainbow.callbacks.tensorboard.image_logger.ImageLogger`:

* ``"glyph"`` (**default**) -- direct geometric depiction.

  - :func:`_render_dir_quiver` draws a matplotlib quiver of the
    predicted (Y, X) direction vectors at every foreground voxel on a
    subsampled grid; arrow length scales with vector magnitude.
  - :func:`_render_cov_glyphs` draws an ellipse glyph at every
    foreground voxel on a subsampled grid; ellipse size scales with
    the major eigenvalue, aspect ratio with anisotropy, orientation
    with the major eigenvector.

  These match the previous (pre-April-2026) panel style and are the
  most literal way to read predictions: an arrow is an arrow, an
  ellipse is an ellipse.  They use matplotlib internally so each panel
  costs ~50ms per epoch -- fine since the TB callback fires only at
  ``every_n_epochs``.

* ``"flow"`` -- vectorised optical-flow-style HSV colour map.

  - :func:`_render_dir_flow` -- Middlebury hue (angle) with a pastel
    palette mirroring
    :func:`brainbow.callbacks.tensorboard.viz._label_to_rgb`
    (constant low saturation, value lifted into ``[0.85, 1.0]``).
  - :func:`_render_cov_flow` -- HSV of the principal eigenvector
    (hue = orientation mod π, value = max-eigenvalue, saturation =
    anisotropy).

  No matplotlib; pure GPU-tensor ops.  Reads as a colour field (no
  obvious arrow / ellipse), useful when zoomed out at a glance.

Both styles composite the rendered overlay onto the raw EM using the
**soft** predicted sem probability as the per-pixel blend weight::

    composited = bg * (1 - sem) + fg * sem

so high-confidence foreground voxels show the rendered overlay,
low-confidence voxels keep the raw EM, and boundary regions fade
smoothly between the two -- same convention as ``pred/avg/val``,
``pred/emb/_{algo}``, and ``pred/label/mul``.
"""

import math
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat

from brainbow.callbacks.tensorboard.viz import _hsv_to_rgb, _normalise


def _to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Broadcast a 1-channel grayscale to 3-channel RGB."""
    if img.shape[1] == 1:
        return repeat(img, "b 1 h w -> b 3 h w")
    return img


def _vector_to_flow_rgb(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    angle_period: float = 2.0 * math.pi,
    max_mag: Optional[torch.Tensor] = None,
    sat: Optional[torch.Tensor] = None,
    val_lo: float = 0.0,
    val_hi: float = 1.0,
) -> torch.Tensor:
    """Map a 2-D vector field to optical-flow RGB.

    Args:
        u: ``[B, H, W]`` horizontal (X) component.
        v: ``[B, H, W]`` vertical (Y) component.
        angle_period: ``2π`` for full directional encoding (vector flow);
            ``π`` for axial encoding so ``+v`` and ``-v`` map to the same
            hue (line orientation, e.g. structure-tensor eigenvectors).
        max_mag: ``[B, 1, 1]`` per-image magnitude normaliser.  ``None``
            uses the per-image ``max(|uv|)`` so each panel auto-scales.
        sat: ``[B, H, W]`` (or scalar broadcastable) saturation in
            ``[0, 1]``.  ``None`` defaults to ones (fully saturated,
            classic optical-flow look).  Pastel palettes pass a small
            constant (e.g. ``0.35``) to match the ``_label_to_rgb``
            convention.
        val_lo: lower bound of the HSV value range.  Magnitude
            ``mag / max_mag`` is rescaled into ``[val_lo, val_hi]``;
            with ``val_lo == 0`` this collapses to the classic
            ``val = mag / max_mag`` Middlebury behaviour.  Set
            ``val_lo > 0`` to keep low-magnitude pixels visibly bright
            (matches the ``_label_to_rgb`` ``[0.75, 1.0]`` value range).
        val_hi: upper bound of the HSV value range.

    Returns:
        ``[B, 3, H, W]`` RGB in ``[0, 1]``.
    """
    B, H, W = u.shape
    angle = torch.atan2(v, u) + math.pi               # [0, 2π]
    if angle_period < 2.0 * math.pi - 1e-6:
        angle = angle % angle_period
    hue = (angle / angle_period) % 1.0

    mag = torch.sqrt(u * u + v * v)
    if max_mag is None:
        max_mag = rearrange(
            mag.amax(dim=(-2, -1)), "b -> b 1 1",
        ).clamp(min=1e-8)
    val_norm = (mag / max_mag).clamp(0.0, 1.0)
    val = val_lo + (val_hi - val_lo) * val_norm

    if sat is None:
        sat = torch.ones_like(val)
    elif not torch.is_tensor(sat):
        sat = torch.full_like(val, float(sat))

    h_flat = rearrange(hue, "b h w -> (b h w)")
    s_flat = rearrange(sat, "b h w -> (b h w)")
    v_flat = rearrange(val, "b h w -> (b h w)")
    rgb = _hsv_to_rgb(h_flat, s_flat, v_flat)         # [(b h w), 3]
    return rearrange(rgb, "(b h w) c -> b c h w", b=B, h=H, w=W)


def _overlay(
    fg_rgb: torch.Tensor,
    bg_rgb: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Composite ``fg_rgb`` onto ``bg_rgb`` using ``mask`` as the per-pixel
    blend weight.

    The formula is ``bg * (1 - mask) + fg * mask`` with ``mask`` clamped
    to ``[0, 1]``.  When ``mask`` is the predicted soft sem probability,
    high-confidence pixels show the pure ``fg`` colour, low-confidence
    pixels keep the raw ``bg`` intensity, and boundary pixels fade
    smoothly between the two.  A hard 0/1 mask still works -- it
    degenerates the formula to a strict gate.
    """
    if mask.dim() == 3:
        mask = rearrange(mask, "b h w -> b 1 h w")
    w = mask.float().clamp(0.0, 1.0)
    return bg_rgb * (1.0 - w) + fg_rgb * w


def _eigh_2x2_sym(
    mat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form eigendecomposition of batched 2x2 symmetric matrices.

    We **deliberately avoid** ``torch.linalg.eigh`` here because it
    dispatches to cuSOLVER's batched ``syevj`` kernel
    (``cusolverDnXsyevBatched``), which raises
    ``CUSOLVER_STATUS_INTERNAL_ERROR`` on the million-matrix batches we
    pass per panel -- especially during sanity check, when the model's
    predictions can contain large or NaN/Inf entries that destabilise
    the iterative solver.  See the GOTCHAS entry "cov-overlay
    eigendecomposition".

    The 2x2 closed-form is exact, dispatch-free, fp32-safe, and
    avoids any cuSOLVER call entirely.

    Args:
        mat: ``[..., 2, 2]`` symmetric matrix batch.  Off-diagonal is
            symmetrised internally; NaN/Inf entries are zeroed before
            the decomposition so a single bad pixel does not poison the
            whole panel.

    Returns:
        ``(lam_min, lam_max, vec_max)`` where ``lam_min`` / ``lam_max``
        are ``[...]`` ascending real eigenvalues and ``vec_max`` is the
        ``[..., 2]`` unit-norm principal eigenvector (rows match the
        input row order).
    """
    mat = torch.nan_to_num(mat.float(), nan=0.0, posinf=0.0, neginf=0.0)
    a = mat[..., 0, 0]
    c = mat[..., 1, 1]
    b = 0.5 * (mat[..., 0, 1] + mat[..., 1, 0])

    trace = a + c
    diff = a - c
    # disc = sqrt((a-c)^2 + 4 b^2); the +eps guards downstream divisions.
    disc = torch.sqrt(diff * diff + 4.0 * b * b + 1e-12)

    lam_max = 0.5 * (trace + disc)
    lam_min = 0.5 * (trace - disc)

    # Two algebraically equivalent forms of the principal eigenvector;
    # pick the better-conditioned one per pixel to avoid the degenerate
    # case where (a == lam_max) cancels v1 or (c == lam_max) cancels v2.
    v1x, v1y = b, lam_max - a
    v2x, v2y = lam_max - c, b
    n1 = v1x * v1x + v1y * v1y
    n2 = v2x * v2x + v2y * v2y
    use1 = n1 >= n2
    vx = torch.where(use1, v1x, v2x)
    vy = torch.where(use1, v1y, v2y)

    # Isotropic / scalar matrices have both forms == 0; fall back to
    # +x as an arbitrary-but-deterministic principal axis.
    norm = torch.sqrt(vx * vx + vy * vy)
    is_iso = norm < 1e-12
    safe_norm = norm.clamp(min=1e-12)
    vx = torch.where(is_iso, torch.ones_like(vx), vx / safe_norm)
    vy = torch.where(is_iso, torch.zeros_like(vy), vy / safe_norm)

    return lam_min, lam_max, torch.stack([vx, vy], dim=-1)


_DIR_PASTEL_SAT = 0.35     # midpoint of `_label_to_rgb`'s [0.20, 0.45] sat
_DIR_PASTEL_VAL_LO = 0.85  # bottom of `_label_to_rgb`'s [0.75, 1.00] value range
_DIR_PASTEL_VAL_HI = 1.00  # top of `_label_to_rgb`'s [0.75, 1.00] value range


def _render_dir_flow(
    dir_val: torch.Tensor,
    img_rgb: torch.Tensor,
    sem: torch.Tensor,
    S: int,
) -> torch.Tensor:
    """Direction overlay with the pastel ``pred/label`` palette.

    The in-plane (Y, X) components of the predicted direction field are
    encoded into HSV with the same soft palette family that
    :func:`brainbow.callbacks.tensorboard.viz._label_to_rgb` uses for
    the ``pred/label`` panels:

    * **Hue** -- ``atan2(v, u)`` mapped into ``[0, 1)`` (Middlebury
      convention).  Hue still encodes direction so that opposite
      vectors land on opposite sides of the colour wheel; this is what
      keeps the panel readable as a flow field.
    * **Saturation** -- constant pastel ``_DIR_PASTEL_SAT`` (``0.35``,
      midpoint of ``_label_to_rgb``'s ``[0.20, 0.45]`` range) so the
      colours match the soft instance-label look instead of the vivid
      Middlebury default.
    * **Value** -- magnitude rescaled into
      ``[_DIR_PASTEL_VAL_LO, _DIR_PASTEL_VAL_HI]`` (``[0.85, 1.0]``,
      matching ``_label_to_rgb``'s ``[0.75, 1.0]``) so even
      low-magnitude pixels stay visibly bright instead of fading to
      black.

    The HSV image is then composited onto the raw EM with the **soft**
    predicted sem probability as the per-pixel blend weight, same as
    every other ``pred/*/mul``-style panel.

    Args:
        dir_val: ``[B, S, H, W]`` predicted direction channels (already
            sliced to 2-D).  For 3-D the channels are ``(Z, Y, X)``; for
            2-D they are ``(Y, X)``.
        img_rgb: ``[B, C, H, W]`` grayscale EM (1- or 3-channel).
        sem: ``[B, H, W]`` predicted soft semantic probability in
            ``[0, 1]`` (clamped).  Used directly as the per-pixel blend
            weight in :func:`_overlay`.
        S: spatial_dims (2 or 3).

    Returns:
        ``[B, 3, H, W]`` RGB tensor with the flow overlay on the EM image.
    """
    bg = _to_rgb(_normalise(img_rgb).clamp(0.0, 1.0))
    if S == 3:
        v = dir_val[:, 1]                              # Y component
        u = dir_val[:, 2]                              # X component
    else:
        u = dir_val[:, 0]
        v = dir_val[:, 1]
    flow_rgb = _vector_to_flow_rgb(
        u, v,
        sat=_DIR_PASTEL_SAT,
        val_lo=_DIR_PASTEL_VAL_LO,
        val_hi=_DIR_PASTEL_VAL_HI,
    )
    return _overlay(flow_rgb, bg, sem)


def _render_cov_flow(
    cov_mat: torch.Tensor,
    img_rgb: torch.Tensor,
    sem: torch.Tensor,
    S: int,
) -> torch.Tensor:
    """Optical-flow-style covariance overlay.

    Each foreground pixel's structure-tensor matrix is eigendecomposed
    and rendered via HSV:

    * **Hue** -- orientation of the major eigenvector, folded into
      ``[0, π)`` so antipodal directions share a colour (the principal
      axis is a line, not an arrow).
    * **Value** -- max-eigenvalue magnitude, normalised by the per-image
      foreground max so high-curvature regions glow.
    * **Saturation** -- anisotropy ratio
      ``(λ_max - λ_min) / λ_max`` so isotropic (round) regions wash out
      to grey and elongated regions stay vivid.

    The HSV image is composited onto the raw EM with the **soft**
    predicted sem probability as the per-pixel blend weight (same
    convention as :func:`_render_dir_flow` and ``pred/avg/val``).

    Args:
        cov_mat: ``[B, H, W, S, S]`` predicted covariance / structure
            tensor matrices (already sliced to 2-D).  For 3-D the YX
            submatrix is used.
        img_rgb: ``[B, C, H, W]`` grayscale EM (1- or 3-channel).
        sem: ``[B, H, W]`` predicted soft semantic probability in
            ``[0, 1]`` (clamped).  Used directly as the per-pixel blend
            weight; a 0.5 hard-threshold derived from it gates the
            per-image magnitude normaliser so that uncertain pixels do
            not warp the colour scale.
        S: spatial_dims (2 or 3).

    Returns:
        ``[B, 3, H, W]`` RGB tensor with the cov overlay on the EM image.
    """
    bg = _to_rgb(_normalise(img_rgb).clamp(0.0, 1.0))

    mat = cov_mat
    if S == 3:
        mat = mat[..., 1:, 1:]                         # project YX submatrix

    # Closed-form 2x2 eigendecomposition.  We *deliberately* sidestep
    # `torch.linalg.eigh` here because cuSOLVER's batched syevj kernel
    # crashes on million-pixel batches with NaN/Inf-tinged predictions
    # early in training (see _eigh_2x2_sym docstring).
    lam_min, lam_max_signed, vec_max = _eigh_2x2_sym(mat)

    max_eig = lam_max_signed.abs()
    min_eig = lam_min.abs()
    anisotropy = (
        (max_eig - min_eig) / max_eig.clamp(min=1e-8)
    ).clamp(0.0, 1.0)

    # Rows of the YX submatrix correspond to (Y, X); the principal
    # eigenvector is returned in row order, so component 0 = Y, 1 = X.
    vec_y = vec_max[..., 0]
    vec_x = vec_max[..., 1]

    # Per-image normaliser uses a hard 0.5-threshold of the soft sem so
    # that uncertain pixels do not warp the colour scale -- otherwise a
    # noisy background patch with high `max_eig` and low confidence would
    # wash out the foreground colours.
    fg_hard = (sem > 0.5).float()
    fg_max = (max_eig * fg_hard).amax(dim=(-2, -1)).clamp(min=1e-8)
    fg_max = rearrange(fg_max, "b -> b 1 1")

    # _eigh_2x2_sym returns unit vectors; scale by max_eig so
    # |(u,v)| = max_eig and `_vector_to_flow_rgb` picks up the magnitude
    # as VAL.
    flow_rgb = _vector_to_flow_rgb(
        vec_x * max_eig, vec_y * max_eig,
        angle_period=math.pi,
        max_mag=fg_max,
        sat=anisotropy,
    )
    return _overlay(flow_rgb, bg, sem)


# ---------------------------------------------------------------------------
# Style 2: matplotlib quiver / ellipse-glyph renderers (default)
# ---------------------------------------------------------------------------
#
# These are the previous (pre-April-2026) panel renderers, restored so
# the user can flip back to a literal arrow / ellipse depiction via
# ``geometry_style="glyph"`` (the default).  They are matplotlib-based
# -- ~50ms / panel -- so we keep them in this dedicated section to make
# the matplotlib dependency easy to spot and to keep the ``flow``
# renderers above purely tensor-only.
#
# Both renderers follow the same two-step pattern as ``flow``:
#
#   1. Build an opaque RGB ``overlay`` (matplotlib draws bg + glyphs).
#   2. Composite onto the **original** raw EM with the soft sem as the
#      per-pixel blend weight, so glyphs that fall on uncertain pixels
#      fade out gracefully (matches the rest of the panel family).

_QUIVER_COLOR = (1.0, 0.4, 0.0, 0.9)       # warm orange, high alpha
_GLYPH_COLOR = (0.0, 0.8, 1.0)             # cyan; complements the orange quiver


def _matplotlib_render_to_tensor(
    draw_fn,
    bg_chw: torch.Tensor,
    *,
    dpi: int = 64,
) -> torch.Tensor:
    """Run ``draw_fn(ax)`` over a matplotlib figure seeded with ``bg_chw``.

    Returns the rasterised result as a ``[3, H, W]`` float tensor in
    ``[0, 1]``, resampled back to the input ``H, W`` if matplotlib's
    canvas size diverged from the input by a pixel (Agg renders at a
    rounded pixel size).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    bg_hwc = rearrange(
        bg_chw.detach().cpu().float(), "c h w -> h w c",
    ).numpy().copy()
    H, W, _ = bg_hwc.shape

    fig, ax = plt.subplots(1, 1, figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(bg_hwc, aspect="equal", interpolation="nearest")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    draw_fn(ax)

    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    rendered = rearrange(
        torch.from_numpy(arr).float() / 255.0, "h w c -> c h w",
    )
    if rendered.shape[-2:] != (H, W):
        rendered = rearrange(
            F.interpolate(
                rearrange(rendered, "c h w -> 1 c h w"),
                size=(H, W), mode="bilinear", align_corners=False,
            ),
            "1 c h w -> c h w",
        )
    return rendered


def _render_dir_quiver(
    dir_val: torch.Tensor,
    img_rgb: torch.Tensor,
    sem: torch.Tensor,
    S: int,
    *,
    step: int = 4,
) -> torch.Tensor:
    """Direction overlay as matplotlib quiver arrows on the raw EM.

    Arrows are placed on a ``step``-spaced subsampled grid; only voxels
    where the predicted soft sem exceeds 0.5 are drawn.  The rendered
    figure is then composited onto the **original** raw EM using the
    soft sem as the per-pixel blend weight (same convention as the
    optical-flow style and every other ``pred/*/mul`` panel) so arrows
    on uncertain pixels fade gracefully.

    Args:
        dir_val: ``[B, S, H, W]`` predicted direction channels (already
            sliced to 2-D).  For 3-D the channels are ``(Z, Y, X)``;
            for 2-D they are ``(Y, X)``.
        img_rgb: ``[B, C, H, W]`` grayscale EM (1- or 3-channel).
        sem: ``[B, H, W]`` predicted soft semantic probability in
            ``[0, 1]`` (clamped).
        S: spatial_dims (2 or 3).
        step: grid spacing for arrow placement.  Larger = sparser
            arrows.

    Returns:
        ``[B, 3, H, W]`` RGB tensor.
    """
    bg = _to_rgb(_normalise(img_rgb).clamp(0.0, 1.0))
    B, _, H, W = bg.shape

    rows_sub = np.arange(step // 2, H, step)
    cols_sub = np.arange(step // 2, W, step)
    CC, RR = np.meshgrid(cols_sub, rows_sub)

    sem_np = sem.detach().cpu().float().numpy()
    dir_np = dir_val.detach().cpu().float().numpy()

    rendered_per_b = []
    for b in range(B):
        d = dir_np[b]
        if S == 3:
            U = d[2][RR, CC]
            V = d[1][RR, CC]
        else:
            U = d[0][RR, CC]
            V = d[1][RR, CC]
        fg = sem_np[b][RR, CC] > 0.5
        cc_fg = CC.ravel()[fg.ravel()]
        rr_fg = RR.ravel()[fg.ravel()]
        u_fg = U.ravel()[fg.ravel()]
        v_fg = V.ravel()[fg.ravel()]

        def _draw(ax, _cc=cc_fg, _rr=rr_fg, _u=u_fg, _v=v_fg):
            if _cc.size == 0:
                return
            ax.quiver(
                _cc, _rr, _u, _v,
                color=_QUIVER_COLOR,
                angles="xy", scale_units="xy", scale=1.0 / (step * 2.0),
                width=0.014, headwidth=4.0, headlength=4.5,
            )

        rendered_per_b.append(_matplotlib_render_to_tensor(_draw, bg[b]))

    overlay = torch.stack(rendered_per_b).to(bg.device)
    return _overlay(overlay, bg, sem)


def _render_cov_glyphs(
    cov_mat: torch.Tensor,
    img_rgb: torch.Tensor,
    sem: torch.Tensor,
    S: int,
    *,
    step: int = 4,
) -> torch.Tensor:
    """Covariance overlay as matplotlib ellipse glyphs on the raw EM.

    Each foreground voxel on a ``step``-spaced subsampled grid gets an
    ellipse whose:

    * **size** -- scales with the major eigenvalue (rel. to the
      per-image foreground max), so high-curvature regions glow.
    * **aspect ratio** -- ``λ_min / λ_max`` (round near medial axis,
      thin near boundaries).
    * **angle** -- aligned with the major eigenvector.

    Eigendecomposition is done on CPU with NumPy on the per-pixel 2x2
    submatrix, so we never touch cuSOLVER's batched syevj kernel
    (see GOTCHAS §41 / :func:`_eigh_2x2_sym`).  ``nan_to_num`` is
    applied first so a single bad pixel does not poison the panel.

    The rendered figure is then composited onto the original raw EM
    using the soft sem as the per-pixel blend weight.

    Args:
        cov_mat: ``[B, H, W, S, S]`` predicted covariance / structure
            tensor matrices (already sliced to 2-D).  For 3-D the YX
            submatrix is used.
        img_rgb: ``[B, C, H, W]`` grayscale EM (1- or 3-channel).
        sem: ``[B, H, W]`` predicted soft semantic probability in
            ``[0, 1]`` (clamped).
        S: spatial_dims (2 or 3).
        step: grid spacing for glyph placement.

    Returns:
        ``[B, 3, H, W]`` RGB tensor.
    """
    from matplotlib.patches import Ellipse

    bg = _to_rgb(_normalise(img_rgb).clamp(0.0, 1.0))
    B, _, H, W = bg.shape
    max_glyph_radius = step * 1.2

    sem_np = sem.detach().cpu().float().numpy()
    cov_np = np.nan_to_num(
        cov_mat.detach().cpu().float().numpy(),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    rows_sub = np.arange(step // 2, H, step)
    cols_sub = np.arange(step // 2, W, step)

    rendered_per_b = []
    for b in range(B):
        sem_b = sem_np[b]
        mat_b = cov_np[b]

        # First pass: per-image normaliser for glyph size.
        max_eig_global = 0.0
        for r in rows_sub:
            for c in cols_sub:
                if sem_b[r, c] <= 0.5:
                    continue
                T = mat_b[r, c]
                if S == 3:
                    T = T[1:, 1:]
                e = float(np.abs(np.linalg.eigvalsh(T)).max())
                if e > max_eig_global:
                    max_eig_global = e
        if max_eig_global < 1e-8:
            max_eig_global = 1.0

        glyphs = []
        for r in rows_sub:
            for c in cols_sub:
                if sem_b[r, c] <= 0.5:
                    continue
                T = mat_b[r, c]
                if S == 3:
                    T = T[1:, 1:]
                eigvals, eigvecs = np.linalg.eigh(T)
                abs_eig = np.abs(eigvals)
                if abs_eig.max() < 1e-8:
                    continue
                scale = abs_eig.max() / max_eig_global
                glyph_radius = max_glyph_radius * float(np.clip(scale, 0.1, 1.0))
                ratio = float(abs_eig.min() / max(abs_eig.max(), 1e-8))
                idx_max = int(abs_eig.argmax())
                # Rows of YX submatrix are (Y, X); matplotlib ellipse
                # angle is measured CCW from +X in degrees.
                angle = float(np.degrees(np.arctan2(
                    eigvecs[0, idx_max], eigvecs[1, idx_max],
                )))
                glyphs.append((c, r, glyph_radius, ratio, angle))

        def _draw(ax, _glyphs=glyphs):
            for c, r, gr, ratio, angle in _glyphs:
                ax.add_patch(Ellipse(
                    xy=(c, r),
                    width=2 * gr,
                    height=2 * gr * ratio,
                    angle=angle,
                    fill=True, facecolor=_GLYPH_COLOR, edgecolor=_GLYPH_COLOR,
                    linewidth=1.2, alpha=0.8,
                ))

        rendered_per_b.append(_matplotlib_render_to_tensor(_draw, bg[b]))

    overlay = torch.stack(rendered_per_b).to(bg.device)
    return _overlay(overlay, bg, sem)


__all__ = [
    "_render_dir_flow",
    "_render_cov_flow",
    "_render_dir_quiver",
    "_render_cov_glyphs",
]
