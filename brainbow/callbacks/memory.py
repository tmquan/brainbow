"""Callbacks that reduce peak memory between train/val phases."""

import torch
import pytorch_lightning as pl


class CudaEmptyCacheCallback(pl.Callback):
    """Clear the CUDA allocator cache before each validation epoch.

    Validation often allocates separate tensors; flushing the cache after
    training reduces peak VRAM at the epoch boundary for large 3D models.
    """

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
