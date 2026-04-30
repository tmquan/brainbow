"""Gaussian-weighted sliding-window inference for unified-head models."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


def create_gaussian_weight(
    patch_size: Tuple[int, ...],
    sigma_scale: float = 0.125,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create an N-D Gaussian weight map for patch blending."""
    sigma = min(patch_size) * sigma_scale
    centers = [torch.arange(s, device=device).float() - s / 2 for s in patch_size]
    grids = torch.meshgrid(*centers, indexing="ij")
    sq_dist = sum(g ** 2 for g in grids)
    gaussian = torch.exp(-sq_dist / (2 * sigma ** 2))
    return gaussian / gaussian.max()


def sliding_window_inference(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    stride: Optional[Tuple[int, int, int]] = None,
    aggregation: str = "gaussian",
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
    sigma_scale: float = 0.125,
    progress: bool = True,
) -> torch.Tensor:
    """Run patch-wise inference and stitch a full-volume unified head.

    Args:
        model: Model returning ``[B, C, D, H, W]``.  For Brainbow, ``C=30``.
        volume: Input volume ``[C, D, H, W]`` or ``[D, H, W]``.
        patch_size: Patch size ``(D, H, W)``.
        stride: Patch stride.  Defaults to ``patch_size // 2``.
        aggregation: ``"gaussian"``, ``"average"``, or ``"max"``.
        batch_size: Number of patches per forward pass.
        device: Inference device.
        sigma_scale: Gaussian sigma as a fraction of the smallest patch dim.
        progress: Show a tqdm progress bar when available.

    Returns:
        Blended unified head tensor ``[C, D, H, W]``.  Split it with
        :func:`brainbow.losses.slice_head` if you need named fields.
    """
    if aggregation not in {"gaussian", "average", "max"}:
        raise ValueError("aggregation must be one of: gaussian, average, max")

    was_training = model.training
    model.eval()

    if volume.dim() == 3:
        volume = rearrange(volume, "d h w -> 1 d h w")
    volume = volume.to(device)
    _, D, H, W = volume.shape
    pd, ph, pw = patch_size

    if stride is None:
        stride = (pd // 2, ph // 2, pw // 2)
    sd, sh, sw = stride

    nd = max(1, (D - pd + sd) // sd)
    nh = max(1, (H - ph + sh) // sh)
    nw = max(1, (W - pw + sw) // sw)

    pad_d = max(0, (nd - 1) * sd + pd - D)
    pad_h = max(0, (nh - 1) * sh + ph - H)
    pad_w = max(0, (nw - 1) * sw + pw - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect")
    D_pad, H_pad, W_pad = volume.shape[1], volume.shape[2], volume.shape[3]

    with torch.no_grad():
        dummy = rearrange(volume[:, :pd, :ph, :pw], "c d h w -> 1 c d h w")
        dummy_out = model(dummy)
        if isinstance(dummy_out, dict):
            raise TypeError(
                "sliding_window_inference expects a model returning the "
                "unified head tensor, not a dict of separate heads."
            )
        n_channels = int(dummy_out.shape[1])

    output = torch.zeros((n_channels, D_pad, H_pad, W_pad), device=device)
    weight = torch.zeros((1, D_pad, H_pad, W_pad), device=device)
    patch_w = (
        create_gaussian_weight(patch_size, sigma_scale, device)
        if aggregation == "gaussian"
        else torch.ones(patch_size, device=device)
    )

    positions = []
    for i in range(nd):
        for j in range(nh):
            for k in range(nw):
                d_start = min(i * sd, D_pad - pd)
                h_start = min(j * sh, H_pad - ph)
                w_start = min(k * sw, W_pad - pw)
                positions.append((d_start, h_start, w_start))

    iterator = range(0, len(positions), batch_size)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                iterator,
                desc="Sliding window inference",
                total=(len(positions) + batch_size - 1) // batch_size,
            )
        except ImportError:
            pass

    with torch.no_grad():
        for batch_start in iterator:
            batch_pos = positions[batch_start:batch_start + batch_size]
            patches = torch.stack([
                volume[:, ds:ds + pd, hs:hs + ph, ws:ws + pw]
                for ds, hs, ws in batch_pos
            ], dim=0)

            pred = model(patches)
            if isinstance(pred, dict):
                raise TypeError("model returned a dict; expected unified tensor.")

            for idx, (ds, hs, ws) in enumerate(batch_pos):
                sl = (slice(None), slice(ds, ds + pd), slice(hs, hs + ph), slice(ws, ws + pw))
                weight_sl = (slice(None),) + sl[1:]
                if aggregation == "max":
                    output[sl] = torch.max(output[sl], pred[idx])
                else:
                    output[sl] += pred[idx] * patch_w
                    weight[weight_sl] += patch_w

    if aggregation != "max":
        output = output / (weight + 1e-8)
    output = output[:, :D, :H, :W]

    if was_training:
        model.train()
    return output


__all__ = ["sliding_window_inference"]
