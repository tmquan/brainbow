"""
Gaussian-weighted sliding window inference for volumetric segmentation.

Supports three aggregation strategies:
- **gaussian**: 3D Gaussian weighting for smooth blending (default)
- **average**: uniform weighting
- **max**: voxel-wise maximum probability

Handles the dual-head Vista architecture (semantic logits + instance
embeddings) with separate aggregation for each head.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, reduce


def create_gaussian_weight(
    patch_size: Tuple[int, ...],
    sigma_scale: float = 0.125,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create N-D Gaussian weight map for blending.

    Args:
        patch_size: Patch dimensions, e.g. (D, H, W) or (H, W).
        sigma_scale: Sigma as a fraction of the smallest patch dim.
        device: Target device.

    Returns:
        Gaussian weight tensor with shape ``patch_size``, peak-normalised to 1.
    """
    sigma = min(patch_size) * sigma_scale
    S = len(patch_size)

    centers = [torch.arange(s, device=device).float() - s / 2 for s in patch_size]
    grids = torch.meshgrid(*centers, indexing="ij")
    sq_dist = sum(g ** 2 for g in grids)

    gaussian = torch.exp(-sq_dist / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.max()
    return gaussian


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
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Perform sliding window inference on a 3D volume.

    If the model returns a dict with ``"semantic"`` and ``"instance"``
    keys (Vista architecture), both heads are aggregated and returned.
    Otherwise returns semantic probability maps only.

    Args:
        model: Segmentation model.
        volume: Input volume [C, D, H, W] or [D, H, W].
        patch_size: Size of patches (D, H, W).
        stride: Stride between patches.  Default: ``patch_size // 2``.
        aggregation: ``"gaussian"``, ``"average"``, or ``"max"``.
        batch_size: Patches per forward pass.
        device: Inference device.
        sigma_scale: Gaussian sigma as fraction of min patch dim.
        progress: Show tqdm progress bar.

    Returns:
        If model returns dict: ``{"semantic_probs": [C, D, H, W],
        "instance_embeddings": [E, D, H, W]}``.
        Otherwise: ``[num_classes, D, H, W]`` probability tensor.
    """
    was_training = model.training
    model.eval()

    if volume.dim() == 3:
        volume = rearrange(volume, "d h w -> 1 d h w")
    volume = volume.to(device)
    C, D, H, W = volume.shape
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

    is_dual = isinstance(dummy_out, dict) and "semantic" in dummy_out and "instance" in dummy_out

    if is_dual:
        num_classes = dummy_out["semantic"].shape[1]
        instance_channels = dummy_out["instance"].shape[1]
    else:
        logits = dummy_out if not isinstance(dummy_out, dict) else dummy_out.get("logits", dummy_out.get("semantic"))
        num_classes = logits.shape[1]
        instance_channels = 0

    sem_output = torch.zeros((num_classes, D_pad, H_pad, W_pad), device=device)
    sem_weight = torch.zeros((1, D_pad, H_pad, W_pad), device=device)

    if is_dual:
        emb_output = torch.zeros(
            (instance_channels, D_pad, H_pad, W_pad), device=device,
        )
        emb_weight = torch.zeros((1, D_pad, H_pad, W_pad), device=device)

    if aggregation == "gaussian":
        patch_w = create_gaussian_weight(patch_size, sigma_scale, device)
    else:
        patch_w = torch.ones(patch_size, device=device)

    positions = []
    for i in range(nd):
        for j in range(nh):
            for k in range(nw):
                d_start = min(i * sd, D_pad - pd)
                h_start = min(j * sh, H_pad - ph)
                w_start = min(k * sw, W_pad - pw)
                positions.append((d_start, h_start, w_start))

    total = len(positions)
    iterator = range(0, total, batch_size)

    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                iterator,
                desc="Sliding window inference",
                total=(total + batch_size - 1) // batch_size,
            )
        except ImportError:
            pass

    with torch.no_grad():
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, total)
            batch_pos = positions[batch_start:batch_end]

            patches = torch.stack([
                volume[:, ds:ds + pd, hs:hs + ph, ws:ws + pw]
                for ds, hs, ws in batch_pos
            ], dim=0)

            outputs = model(patches)

            # Brainbow model wrappers apply sigmoid to the semantic head
            # before returning, so ``outputs["semantic"]`` is already a
            # tensor of per-channel probabilities -- no extra activation
            # needed here.  External / legacy models that still return
            # raw logits under ``"logits"`` go through the softmax path
            # below.
            if is_dual:
                sem_probs = outputs["semantic"]
                ins_emb = outputs["instance"]
            elif isinstance(outputs, dict):
                if "semantic" in outputs:
                    sem_probs = outputs["semantic"]
                else:
                    sem_probs = F.softmax(outputs["logits"], dim=1)
                ins_emb = None
            else:
                sem_probs = F.softmax(outputs, dim=1)
                ins_emb = None

            for idx, (ds, hs, ws) in enumerate(batch_pos):
                sl = (slice(None), slice(ds, ds + pd), slice(hs, hs + ph), slice(ws, ws + pw))

                if aggregation == "max":
                    sem_output[sl] = torch.max(sem_output[sl], sem_probs[idx])
                else:
                    sem_output[sl] += sem_probs[idx] * patch_w
                    sem_weight[(slice(None),) + sl[1:]] += patch_w

                if is_dual and ins_emb is not None:
                    if aggregation == "max":
                        patch_max = reduce(sem_probs[idx], "c ... -> ...", "max")
                        out_max = reduce(
                            sem_output[:, ds:ds + pd, hs:hs + ph, ws:ws + pw],
                            "c ... -> ...", "max",
                        )
                        mask = patch_max > out_max
                        emb_output[:, ds:ds + pd, hs:hs + ph, ws:ws + pw] = torch.where(
                            rearrange(mask, "... -> 1 ..."),
                            ins_emb[idx],
                            emb_output[:, ds:ds + pd, hs:hs + ph, ws:ws + pw],
                        )
                    else:
                        emb_output[sl] += ins_emb[idx] * patch_w
                        emb_weight[(slice(None),) + sl[1:]] += patch_w

    if aggregation != "max":
        sem_output = sem_output / (sem_weight + 1e-8)
        if is_dual:
            emb_output = emb_output / (emb_weight + 1e-8)

    sem_output = sem_output[:, :D, :H, :W]

    if was_training:
        model.train()

    if is_dual:
        emb_output = emb_output[:, :D, :H, :W]
        return {
            "semantic_probs": sem_output,
            "instance_embeddings": emb_output,
            "_positions": positions,
            "_padding": (pad_d, pad_h, pad_w),
        }

    return sem_output
