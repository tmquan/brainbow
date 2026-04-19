"""Rank-aware HuggingFace snapshot download for Cosmos-Transfer2.5."""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Files never read by CosmosTransfer3DWrapper.  The T5-XXL text encoder
# is upstream Cosmos baggage: our wrapper feeds null prompt embeddings,
# so skipping it saves ~15 GB per checkpoint snapshot.
_DEFAULT_IGNORE_PATTERNS: List[str] = [
    "*.md",
    "*.txt",
    "examples/*",
    "docs/*",
    "text_encoder/*",
    "tokenizer/*",
]


def _download_from_hf(
    repo_id: str,
    revision: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> Path:
    """Download model snapshot from HuggingFace Hub.

    In DDP training, rank 0 downloads first while other ranks wait at a
    barrier, then all ranks resolve the cached path without re-downloading.

    Args:
        repo_id: ``"<org>/<name>"`` HF Hub identifier.
        revision: Git ref (branch/tag/commit) to pin the download at.
        cache_dir: Where to cache the snapshot (default: Brainbow cache).
        token: HF access token for gated repositories.
        ignore_patterns: Override the default ignore list.  By default
            the text encoder and tokenizer are skipped because
            :class:`CosmosTransfer3DWrapper` feeds a null prompt
            embedding and never loads them.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for Cosmos-Transfer2.5 weight "
            "download.  Install with: pip install huggingface_hub"
        )

    import torch.distributed as dist

    cache_dir = cache_dir or str(
        Path.home() / ".cache" / "brainbow" / "cosmos_transfer25"
    )
    ignore = list(ignore_patterns) if ignore_patterns is not None else list(_DEFAULT_IGNORE_PATTERNS)

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if rank == 0:
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                ignore_patterns=ignore,
            )
            logger.info("Downloaded %s (rev=%s) -> %s", repo_id, revision, local_path)
        except Exception as exc:
            logger.warning(
                "HuggingFace download failed for %s (rev=%s): %s.  "
                "Falling back to random initialisation.",
                repo_id, revision, exc,
            )
            if is_distributed:
                dist.barrier()
            raise

    if is_distributed:
        dist.barrier()

    if rank != 0:
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=True,
            ignore_patterns=ignore,
        )
        logger.info("Downloaded %s (rev=%s) -> %s", repo_id, revision, local_path)

    return Path(local_path)


__all__ = ["_download_from_hf"]
