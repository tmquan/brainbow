"""Low-level image utilities for TensorBoard visualisation.

Contains the small, dependency-light helpers used by the per-head
loggers: central-slice extraction, per-image min-max normalisation,
HSV colour LUT, integer-label → pastel-RGB mapping, and manifold
projection of instance embeddings.
"""

from typing import Optional

import torch
from einops import rearrange, reduce

from brainbow.utils.manifold import project_batch


def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Extract the central depth-slice from a 5-D tensor [B,C,D,H,W].

    Returns *t* unchanged if it is already 4-D [B,C,H,W].
    """
    if t.dim() == 5:
        return t[:, :, t.shape[2] // 2]
    return t


def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalisation to [0, 1].

    Each image in the batch is normalised independently so that its
    minimum becomes 0 and its maximum becomes 1.

    Note:
        This is a **contrast stretch**, not a simple clip.  Applying
        it to the ground-truth image panel (``true/image``) makes the
        panel visually comparable across samples with different
        intensity distributions, but it also means that
        ``true/image`` can look brighter / more contrasted than the
        boundary ``pred/raw`` panel, which is shown via
        ``clamp(0, 1)`` on the already-normalised reconstruction.
        Keep that in mind when comparing reconstruction quality
        visually; the loss scalars (``boundary/loss/raw``) are
        computed on the unstretched signal.
    """
    flat = rearrange(t, "b ... -> b (...)")                        # [B, N]
    lo = reduce(flat, "b n -> b 1", "min")
    hi = reduce(flat, "b n -> b 1", "max")
    denom = (hi - lo).clamp(min=1e-5)
    normed = (flat - lo) / denom                                   # [B, N]
    return rearrange(normed, "b (c h w) -> b c h w",
                     c=t.shape[1], h=t.shape[2], w=t.shape[3])


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorised HSV→RGB.  All inputs/outputs in [0, 1], shape [N]."""
    h6 = h * 6.0
    sector = h6.long() % 6
    f = h6 - h6.floor()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    lut = [(v, t, p), (q, v, p), (p, v, t),
           (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
    for i, (ri, gi, bi) in enumerate(lut):
        mask = sector == i
        r = torch.where(mask, ri, r)
        g = torch.where(mask, gi, g)
        b = torch.where(mask, bi, b)
    return torch.stack([r, g, b], dim=-1)


def _label_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """Map integer instance labels → pastel, deterministic RGB image.

    Background (0) is black.  Non-zero labels are coloured via a
    golden-ratio hue-spaced HSV palette with moderate saturation and
    high value for soft, PCA-like colours.

    Args:
        labels: [B, H, W] long tensor.

    Returns:
        [B, 3, H, W] float tensor in [0, 1].
    """
    B, H, W = labels.shape
    flat = rearrange(labels, "b h w -> (b h w)").long()
    n_labels = flat.max().item() + 1

    GOLDEN_RATIO = 0.618033988749895
    ids = torch.arange(n_labels, device=labels.device, dtype=torch.float32)
    hue = (ids * GOLDEN_RATIO) % 1.0

    gen = torch.Generator(device=labels.device).manual_seed(0)
    rand = torch.rand(n_labels, 2, device=labels.device, generator=gen)
    sat = 0.20 + 0.25 * rand[:, 0]                                # [0.20, 0.45]
    val = 0.75 + 0.25 * rand[:, 1]                                # [0.75, 1.00]

    palette = _hsv_to_rgb(hue, sat, val)                           # [n_labels, 3]
    palette[0] = 0.0                                               # background → black

    rgb = palette[flat]
    return rearrange(rgb, "(b h w) c -> b c h w", b=B, h=H, w=W)


def _project_embedding(
    emb: torch.Tensor,
    n_components: int = 3,
    algorithm: str = "pca",
    backend: str = "auto",
) -> torch.Tensor:
    """Project ``[B, E, H, W]`` embeddings → ``[B, 3, H, W]`` via per-image
    manifold reduction.

    The top ``n_components`` axes are used as RGB channels.  Each image in
    the batch is reduced independently so colours are locally meaningful
    (nearby pixels with similar embeddings receive similar colours).

    Args:
        emb: 4-D tensor of embeddings (after central-slice extraction for
            volumetric models).
        n_components: Number of output channels (typically 3 for RGB).
        algorithm: ``"pca"`` | ``"svd"`` | ``"umap"``.
        backend:   ``"auto"`` | ``"cuml"`` | ``"torch"`` | ``"umap-learn"``.

    Uses cuML GPU estimators when the input lives on CUDA and RAPIDS is
    installed; otherwise falls back to ``torch.linalg.svd`` (pca / svd)
    or ``umap-learn`` (umap).  Ill-conditioned cases degrade to the
    leading feature channels rather than crashing training.
    """
    _, _, H, W = emb.shape
    flat = rearrange(emb, "b e h w -> b e (h w)").float()         # [B, E, N]
    proj = project_batch(
        flat, n_components=n_components, algorithm=algorithm, backend=backend,
    )
    proj = rearrange(proj, "b c (h w) -> b c h w", h=H, w=W)
    return _normalise(proj)


__all__ = [
    "_hsv_to_rgb",
    "_label_to_rgb",
    "_normalise",
    "_project_embedding",
    "_to_2d",
]

# Silence "unused import" for Optional (kept for forward-compat type hints).
_ = Optional
