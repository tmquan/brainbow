"""Vectorised geometry overlays for the unified-head TB panels.

Optical-flow-style HSV encodings of the predicted ``dir`` and ``cov``
fields, masked by the predicted semantic foreground:

* :func:`_render_dir_flow` -- Middlebury HSV encoding of the in-plane
  (Y, X) direction vector.  Hue = angle, value = magnitude.
* :func:`_render_cov_flow` -- HSV encoding of the principal eigenvector
  of the structure tensor.  Hue = orientation (mod π since axes are
  bidirectional), value = max-eigenvalue magnitude, saturation =
  anisotropy ratio (round → washed out, elongated → vivid).

Foreground voxels (``mask > 0``) display the **pure** flow colour;
background voxels keep the raw EM intensity verbatim.  No soft alpha
blend -- it muddied the hue and made it hard to read direction at
glance.

Both renderers are pure NumPy / PyTorch -- no matplotlib pass per
batch -- so the TB callback now overlays at GPU-tensor speed instead of
spinning up a matplotlib figure per image.
"""

import math
from typing import Optional

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
        sat: ``[B, H, W]`` saturation in ``[0, 1]``.  ``None`` defaults to
            ones (fully saturated, classic optical-flow look).

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
    val = (mag / max_mag).clamp(0.0, 1.0)

    if sat is None:
        sat = torch.ones_like(val)

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
    """Composite ``fg_rgb`` onto ``bg_rgb`` using ``mask`` as a hard gate.

    Foreground pixels (``mask > 0``) show the pure flow colour; background
    pixels (``mask == 0``) keep the raw EM intensity verbatim.  No soft
    alpha blend -- the foreground hue is always at full strength so the
    direction / orientation reads at glance.
    """
    if mask.dim() == 3:
        mask = rearrange(mask, "b h w -> b 1 h w")
    w = mask.float().clamp(0.0, 1.0)
    return bg_rgb * (1.0 - w) + fg_rgb * w


def _render_dir_flow(
    dir_val: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
    S: int,
) -> torch.Tensor:
    """Optical-flow-style direction overlay.

    The in-plane (Y, X) components of the predicted direction field are
    encoded into HSV (hue = angle, value = magnitude) and composited
    onto the raw EM image using ``labels`` as a hard mask.  Foreground
    pixels show the pure flow colour; background pixels keep the raw EM
    intensity.

    Args:
        dir_val: ``[B, S, H, W]`` predicted direction channels (already
            sliced to 2-D).  For 3-D the channels are ``(Z, Y, X)``; for
            2-D they are ``(Y, X)``.
        img_rgb: ``[B, C, H, W]`` grayscale EM (1- or 3-channel).
        labels: ``[B, H, W]`` foreground mask (any int tensor; 0 =
            background, >0 = foreground).
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
    fg_mask = (labels > 0).float()
    flow_rgb = _vector_to_flow_rgb(u, v)
    return _overlay(flow_rgb, bg, fg_mask)


def _render_cov_flow(
    cov_mat: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
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

    Foreground pixels show the pure flow colour; background pixels keep
    the raw EM intensity (hard mask, no soft alpha blend).

    Args:
        cov_mat: ``[B, H, W, S, S]`` predicted covariance / structure
            tensor matrices (already sliced to 2-D).  For 3-D the YX
            submatrix is used.
        img_rgb: ``[B, C, H, W]`` grayscale EM (1- or 3-channel).
        labels: ``[B, H, W]`` foreground mask (any int tensor; 0 =
            background, >0 = foreground).
        S: spatial_dims (2 or 3).

    Returns:
        ``[B, 3, H, W]`` RGB tensor with the cov overlay on the EM image.
    """
    bg = _to_rgb(_normalise(img_rgb).clamp(0.0, 1.0))

    mat = cov_mat
    if S == 3:
        mat = mat[..., 1:, 1:]                         # project YX submatrix

    # Symmetrise for numerical safety -- eigh requires Hermitian input
    # and predictions can drift slightly off-symmetric.
    sym = 0.5 * (mat + rearrange(mat, "... a b -> ... b a"))
    eigvals, eigvecs = torch.linalg.eigh(sym.float())  # ascending order

    abs_eig = eigvals.abs()
    max_eig = abs_eig[..., 1]                          # [B, H, W]
    min_eig = abs_eig[..., 0]
    anisotropy = (
        (max_eig - min_eig) / max_eig.clamp(min=1e-8)
    ).clamp(0.0, 1.0)

    # Major eigenvector lives in the last column (eigh returns ascending
    # eigenvalues); rows 0/1 correspond to (Y, X) of the YX submatrix.
    vec_y = eigvecs[..., 0, 1]
    vec_x = eigvecs[..., 1, 1]

    # Per-image normaliser uses foreground only so background noise does
    # not wash out the colour scale.
    fg_mask = (labels > 0).float()
    fg_max = (max_eig * fg_mask).amax(dim=(-2, -1)).clamp(min=1e-8)
    fg_max = rearrange(fg_max, "b -> b 1 1")

    # eigh returns unit vectors; scale by max_eig so |(u,v)| = max_eig
    # and `_vector_to_flow_rgb` picks up the magnitude as VAL.
    flow_rgb = _vector_to_flow_rgb(
        vec_x * max_eig, vec_y * max_eig,
        angle_period=math.pi,
        max_mag=fg_max,
        sat=anisotropy,
    )
    return _overlay(flow_rgb, bg, fg_mask)


__all__ = ["_render_dir_flow", "_render_cov_flow"]
