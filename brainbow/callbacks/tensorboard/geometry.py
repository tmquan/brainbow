"""Matplotlib-based geometry renderers.

Ellipse glyph and direction quiver overlays used by the
``{stage}/{mode}/geometry/*`` panels.  These functions are the only
consumers of matplotlib in the TensorBoard logger, isolated here so
the rest of the package stays NumPy/PyTorch-only.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def _render_cov_glyphs(
    cov_mat: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    step: int = 4,
) -> torch.Tensor:
    """Render EDT structure-tensor ellipse glyphs on the EM image.

    Each foreground pixel on a subsampled grid gets an ellipse whose:
    - **size** reflects the maximum eigenvalue relative to the global max
      (large near instance centres where EDT is high, small near boundaries).
    - **aspect ratio** reflects the eigenvalue ratio (elongated near
      boundaries, round near the medial axis).
    - **angle** is aligned with the major eigenvector (boundary tangent
      direction for anisotropic regions).

    Args:
        cov_mat: [B, H, W, s1, s2] predicted covariance matrices (2D-sliced).
        img_rgb: [B, 3, H, W] grayscale EM repeated to 3 channels.
        labels: [B, H, W] fg/bg mask (any int tensor; 0 = background, >0 = foreground).
        S: spatial_dims (2 or 3).  For 3D the last 2x2 submatrix is used.
        step: grid spacing for glyph placement.

    Returns:
        [B, 3, H, W] tensor with ellipse glyphs overlaid on the EM image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    B, _, H, W = img_rgb.shape
    device = img_rgb.device
    max_glyph_radius = step * 1.2
    COLOR = (0.0, 0.8, 1.0)

    result = []
    for b in range(B):
        bg = rearrange(img_rgb[b].detach().cpu().float(), "c h w -> h w c").numpy().copy()
        lbl = labels[b].detach().cpu().numpy()
        mat = cov_mat[b].detach().cpu().float().numpy()

        rows_sub = np.arange(step // 2, H, step)
        cols_sub = np.arange(step // 2, W, step)

        # First pass: find global max eigenvalue for normalisation
        max_eig_global = 0.0
        for r in rows_sub:
            for c in cols_sub:
                if lbl[r, c] == 0:
                    continue
                T = mat[r, c]
                if S == 3:
                    T = T[1:, 1:]                                  # project 3x3 → 2x2 (YX plane)
                e = np.abs(np.linalg.eigvalsh(T)).max()
                if e > max_eig_global:
                    max_eig_global = e
        if max_eig_global < 1e-8:
            max_eig_global = 1.0

        fig, ax = plt.subplots(1, 1, figsize=(W / 64, H / 64), dpi=64)
        ax.imshow(bg, aspect="equal", interpolation="nearest")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Second pass: draw glyphs
        for r in rows_sub:
            for c in cols_sub:
                if lbl[r, c] == 0:
                    continue
                T = mat[r, c]
                if S == 3:
                    T = T[1:, 1:]

                eigvals, eigvecs = np.linalg.eigh(T)
                abs_eig = np.abs(eigvals)
                if abs_eig.max() < 1e-8:
                    continue

                scale = abs_eig.max() / max_eig_global
                glyph_radius = max_glyph_radius * np.clip(scale, 0.1, 1.0)

                ratio = abs_eig.min() / max(abs_eig.max(), 1e-8)

                idx_max = int(abs_eig.argmax())
                angle = np.degrees(np.arctan2(
                    eigvecs[1, idx_max], eigvecs[0, idx_max],
                ))

                ax.add_patch(Ellipse(
                    xy=(c, r),
                    width=2 * glyph_radius,
                    height=2 * glyph_radius * ratio,
                    angle=angle,
                    fill=True, facecolor=COLOR, edgecolor=COLOR,
                    linewidth=1.2, alpha=0.8,
                ))

        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        rendered = rearrange(
            torch.from_numpy(arr).float() / 255.0,
            "h w c -> c h w",
        )
        rendered = rearrange(
            F.interpolate(
                rearrange(rendered, "c h w -> 1 c h w"),
                size=(H, W), mode="bilinear", align_corners=False,
            ),
            "1 c h w -> c h w",
        )
        result.append(rendered)

    return torch.stack(result).to(device)


def _render_dir_quiver(
    dir_val: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    dir_target: str = "centroid",
    step: int = 4,
) -> torch.Tensor:
    """Render direction vectors as quiver arrows on the EM image.

    Arrow length reflects the global magnitude of each direction vector:
    boundary pixels (far from centroid/skeleton) produce long arrows,
    centre pixels produce short ones.

    Args:
        dir_val: [B, S, H, W] predicted direction channels (2D-sliced).
        img_rgb: [B, 3, H, W] grayscale EM repeated to 3 channels.
        labels: [B, H, W] fg/bg mask (any int tensor; 0 = background, >0 = foreground).
        S: spatial_dims (2 or 3).
        dir_target: ``"centroid"`` or ``"skeleton"`` (cosmetic only).
        step: grid spacing for arrow placement.

    Returns:
        [B, 3, H, W] tensor with quiver arrows overlaid on the EM image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B, _, H, W = img_rgb.shape
    device = img_rgb.device
    COLOR = (1.0, 0.4, 0.0, 0.9)

    rows_sub = np.arange(step // 2, H, step)
    cols_sub = np.arange(step // 2, W, step)
    CC, RR = np.meshgrid(cols_sub, rows_sub)

    result = []
    for b in range(B):
        bg = rearrange(img_rgb[b].detach().cpu().float(), "c h w -> h w c").numpy().copy()
        lbl = labels[b].detach().cpu().numpy()
        d = dir_val[b].detach().cpu().float().numpy()

        if S == 3:
            U = d[2][RR, CC]
            V = d[1][RR, CC]
        else:
            U = d[0][RR, CC]
            V = d[1][RR, CC]

        fg = lbl[RR, CC] > 0

        fig, ax = plt.subplots(1, 1, figsize=(W / 64, H / 64), dpi=64)
        ax.imshow(bg, aspect="equal", interpolation="nearest")
        m = fg.ravel()
        if m.any():
            ax.quiver(
                CC.ravel()[m], RR.ravel()[m],
                U.ravel()[m], V.ravel()[m],
                color=COLOR,
                angles="xy", scale_units="xy", scale=1.0 / (step * 2.0),
                width=0.014, headwidth=4.0, headlength=4.5,
            )
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        rendered = rearrange(
            torch.from_numpy(arr).float() / 255.0,
            "h w c -> c h w",
        )
        rendered = rearrange(
            F.interpolate(
                rearrange(rendered, "c h w -> 1 c h w"),
                size=(H, W), mode="bilinear", align_corners=False,
            ),
            "1 c h w -> c h w",
        )
        result.append(rendered)

    return torch.stack(result).to(device)


__all__ = ["_render_cov_glyphs", "_render_dir_quiver"]

_ = Optional  # keep forward-compat typing import available
