"""
Gaussian-weighted sliding-window inference for volumetric segmentation.

Why this file exists
--------------------
Patch-based inference is the only practical way to evaluate a 3-D
volume that exceeds the GPU memory budget.  This module slides a
configurable window across the volume, runs the model on each patch,
and blends the per-patch predictions back into a full-volume output
using one of three weight strategies:

* ``gaussian`` -- 3-D Gaussian taper (default).  Smoothest blend at
  patch borders, cheapest enough to be the everyday choice.
* ``average``  -- uniform weight.  Cheaper still, slightly more visible
  patch borders.
* ``max``      -- voxel-wise maximum.  Useful for one-hot semantic
  logits where you want the most-confident patch to win.

Public surface
--------------
* :func:`sliding_window_inference` -- main entry point.  Accepts
  either a single-tensor model output (legacy semantic-only mode) or a
  ``Dict[str, Tensor]`` with any of the four standard heads
  (``semantic`` / ``instance`` / ``geometry`` / ``boundary``).

Head-aggregation rules
----------------------
On the dict path the function discovers every head present in a dummy
forward pass and allocates one accumulator per head.  All heads are
blended with the same weighting strategy (``gaussian`` / ``average``
/ ``max``).  Heads other than ``semantic`` are returned under their
own key (``instance_embeddings`` is preserved for back-compat); see
:func:`sliding_window_inference` for the exact return contract.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, reduce


# Names of head keys we know how to aggregate.  The first one ever
# detected becomes "the semantic head" for the back-compat
# ``semantic_probs`` return key; all others are returned under their
# own name (and ``instance`` keeps the legacy ``instance_embeddings``
# alias to avoid breaking older callers).
_KNOWN_HEADS: Tuple[str, ...] = ("semantic", "instance", "geometry", "boundary")


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


def _detect_heads(dummy_out) -> List[str]:
    """Return the list of standard heads present in the model output.

    Falls back to ``["semantic"]`` for legacy single-tensor outputs.
    Unknown extra dict keys are ignored.
    """
    if not isinstance(dummy_out, dict):
        return ["semantic"]
    heads = [h for h in _KNOWN_HEADS if h in dummy_out]
    if not heads:
        if "logits" in dummy_out:
            return ["semantic"]
        raise ValueError(
            "sliding_window_inference: model returned a dict with none of "
            f"the known heads {_KNOWN_HEADS}; got keys "
            f"{sorted(dummy_out.keys())}."
        )
    return heads


def _extract_head(
    head: str, outputs, *, is_dict: bool,
) -> torch.Tensor:
    """Pull a single head from a model output, applying softmax to legacy logits."""
    if not is_dict:
        return F.softmax(outputs, dim=1)
    if head in outputs:
        return outputs[head]
    if head == "semantic" and "logits" in outputs:
        return F.softmax(outputs["logits"], dim=1)
    raise KeyError(
        f"sliding_window_inference: head '{head}' missing from model output "
        f"(got {sorted(outputs.keys())})."
    )


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

    The function discovers which of the standard heads (``semantic``,
    ``instance``, ``geometry``, ``boundary``) the model returns and
    aggregates each one independently using the same weighting
    strategy.

    Args:
        model: Segmentation model.
        volume: Input volume ``[C, D, H, W]`` or ``[D, H, W]``.
        patch_size: Size of patches ``(D, H, W)``.
        stride: Stride between patches.  Default: ``patch_size // 2``.
        aggregation: ``"gaussian"``, ``"average"``, or ``"max"``.
        batch_size: Patches per forward pass.
        device: Inference device.
        sigma_scale: Gaussian sigma as fraction of min patch dim.
        progress: Show tqdm progress bar.

    Returns:
        Single-tensor model output (legacy mode):
            ``[num_classes, D, H, W]`` softmax probabilities.

        Dict model output (Vista / Cosmos):
            A dict containing one entry per detected head plus the
            ``_positions`` / ``_padding`` metadata used by the
            stitching code.  ``semantic_probs`` and
            ``instance_embeddings`` are kept as aliases for
            ``semantic`` / ``instance`` to preserve back-compat.
    """
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

    is_dict = isinstance(dummy_out, dict)
    heads = _detect_heads(dummy_out)

    head_channels = {
        h: _extract_head(h, dummy_out, is_dict=is_dict).shape[1] for h in heads
    }
    head_output = {
        h: torch.zeros((c, D_pad, H_pad, W_pad), device=device)
        for h, c in head_channels.items()
    }
    head_weight = {
        h: torch.zeros((1, D_pad, H_pad, W_pad), device=device)
        for h in heads
    }

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
            outputs_is_dict = isinstance(outputs, dict)

            # Brainbow model wrappers apply sigmoid to the semantic head
            # before returning, so ``outputs["semantic"]`` is already
            # per-channel probabilities -- no extra activation needed
            # for it.  Legacy models returning raw logits under
            # ``"logits"`` still go through the softmax path inside
            # ``_extract_head``.  Other heads (instance / geometry /
            # boundary) are written through unchanged.
            head_pred = {
                h: _extract_head(h, outputs, is_dict=outputs_is_dict)
                for h in heads
            }

            for idx, (ds, hs, ws) in enumerate(batch_pos):
                sl = (slice(None), slice(ds, ds + pd), slice(hs, hs + ph), slice(ws, ws + pw))
                weight_sl = (slice(None),) + sl[1:]
                for h in heads:
                    pred = head_pred[h][idx]
                    if aggregation == "max":
                        head_output[h][sl] = torch.max(head_output[h][sl], pred)
                    else:
                        head_output[h][sl] += pred * patch_w
                        head_weight[h][weight_sl] += patch_w

    if aggregation != "max":
        for h in heads:
            head_output[h] = head_output[h] / (head_weight[h] + 1e-8)

    for h in heads:
        head_output[h] = head_output[h][:, :D, :H, :W]

    if was_training:
        model.train()

    if not is_dict:
        return head_output["semantic"]

    out: Dict[str, torch.Tensor] = dict(head_output)
    # Back-compat aliases used by older callers / tests.
    if "semantic" in out:
        out["semantic_probs"] = out["semantic"]
    if "instance" in out:
        out["instance_embeddings"] = out["instance"]
    out["_positions"] = positions
    out["_padding"] = (pad_d, pad_h, pad_w)
    return out
