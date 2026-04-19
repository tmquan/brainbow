"""Rank-aware HuggingFace snapshot download + partial state-dict load for Vista3D.

The upstream MONAI VISTA3D checkpoint (`MONAI/VISTA3D-HF`) stores a full
``vista3d.VISTA3D`` network, of which we only need the SegResNetDS2
*encoder*.  This module:

1. Downloads the snapshot (rank-aware for DDP).
2. Filters keys that belong to the image encoder.
3. Strips the ``network.image_encoder.`` prefix so they align with
   ``Vista3DWrapper.backbone`` (a bare :class:`monai.networks.nets.segresnet_ds.SegResNetDS2`).
4. Loads them with ``strict=False`` and logs shape mismatches clearly.

The pretrained encoder was trained with ``init_filters=48``; users who
pass ``feature_size != 48`` will see shape-mismatch warnings and the
affected layers will stay randomly initialised.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Public HF release of the MONAI VISTA3D pretrained encoder (safetensors).
DEFAULT_VISTA3D_REPO = "MONAI/VISTA3D-HF"
DEFAULT_VISTA3D_REVISION = "main"
DEFAULT_VISTA3D_WEIGHT_FILE = "vista3d_pretrained_model/model.safetensors"

# The upstream VISTA3D network nests the SegResNetDS2 encoder under this
# prefix; everything else in the checkpoint (class head, MLP, etc.) is
# specific to the upstream architecture and is ignored.
_ENCODER_PREFIX = "network.image_encoder."


def _download_vista3d_snapshot(
    repo_id: str = DEFAULT_VISTA3D_REPO,
    revision: str = DEFAULT_VISTA3D_REVISION,
    weight_file: str = DEFAULT_VISTA3D_WEIGHT_FILE,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Download (or resolve-from-cache) the single safetensors weight file."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for Vista3D weight download.  "
            "Install with: pip install huggingface_hub"
        )

    import torch.distributed as dist

    cache_dir = cache_dir or str(
        Path.home() / ".cache" / "brainbow" / "vista3d"
    )
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if rank == 0:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=weight_file,
                cache_dir=cache_dir,
                token=token,
            )
            logger.info(
                "Downloaded Vista3D checkpoint %s@%s:%s -> %s",
                repo_id, revision, weight_file, local_path,
            )
        except Exception:
            if is_distributed:
                dist.barrier()
            raise

    if is_distributed:
        dist.barrier()

    if rank != 0:
        local_path = hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=weight_file,
            cache_dir=cache_dir,
            token=token,
            local_files_only=True,
        )

    return Path(local_path)


def _read_encoder_state_dict(path: Path) -> dict:
    """Read safetensors, keep only SegResNetDS2 encoder keys."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError(
            "safetensors is required to load the Vista3D checkpoint.  "
            "Install with: pip install safetensors"
        )

    encoder_sd: dict = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if not key.startswith(_ENCODER_PREFIX):
                continue
            short = key[len(_ENCODER_PREFIX):]  # "encoder.conv_init.weight", etc.
            encoder_sd[short] = f.get_tensor(key)
    return encoder_sd


def load_pretrained_vista3d_encoder(
    backbone: nn.Module,
    repo_id: str = DEFAULT_VISTA3D_REPO,
    revision: str = DEFAULT_VISTA3D_REVISION,
    weight_file: str = DEFAULT_VISTA3D_WEIGHT_FILE,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """Download the MONAI VISTA3D checkpoint and load its encoder into ``backbone``.

    Returns a dict with ``loaded`` / ``shape_mismatch`` / ``missing`` /
    ``unexpected`` counts so callers can surface actionable messages
    (e.g. "set `feature_size=48` to match the pretrained encoder").
    """
    local_path = _download_vista3d_snapshot(
        repo_id=repo_id,
        revision=revision,
        weight_file=weight_file,
        cache_dir=cache_dir,
        token=token,
    )
    src_sd = _read_encoder_state_dict(local_path)
    dst_sd = backbone.state_dict()

    filtered: dict = {}
    shape_mismatch: list = []
    for k, v in src_sd.items():
        if k not in dst_sd:
            continue
        if dst_sd[k].shape != v.shape:
            shape_mismatch.append((k, tuple(dst_sd[k].shape), tuple(v.shape)))
            continue
        filtered[k] = v.to(dtype=dst_sd[k].dtype)

    missing, unexpected = backbone.load_state_dict(
        {**dst_sd, **filtered}, strict=False,
    )
    # `missing`/`unexpected` reflect the merged dict (which covers every
    # dst key) so neither list will be populated; the informative counts
    # are `filtered` (loaded) and `shape_mismatch` (skipped).
    report = {
        "loaded": len(filtered),
        "shape_mismatch": len(shape_mismatch),
        "src_total": len(src_sd),
        "dst_total": len(dst_sd),
        "mismatch_samples": shape_mismatch[:5],
    }
    logger.info(
        "Vista3D pretrained encoder loaded: %d / %d upstream tensors "
        "copied into %d backbone tensors (shape-mismatches: %d).",
        report["loaded"], report["src_total"], report["dst_total"],
        report["shape_mismatch"],
    )
    if shape_mismatch:
        sample = ", ".join(
            f"{k}: dst={dst} vs src={src}" for k, dst, src in shape_mismatch[:3]
        )
        logger.warning(
            "Vista3D: %d encoder tensors skipped due to shape mismatch.  "
            "Upstream VISTA3D is trained with init_filters=48; set "
            "`feature_size=48` in your Vista3D config to fully load the "
            "pretrained encoder.  First mismatches: %s",
            len(shape_mismatch), sample,
        )
    return report


__all__ = [
    "DEFAULT_VISTA3D_REPO",
    "DEFAULT_VISTA3D_REVISION",
    "DEFAULT_VISTA3D_WEIGHT_FILE",
    "load_pretrained_vista3d_encoder",
]
