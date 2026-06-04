"""Callbacks for CUDA memory management and observability.

Two callbacks live here:

- ``CudaEmptyCacheCallback`` — flushes the allocator cache at the train/val
  boundary (both before *and* after validation) so the val-time high-water
  mark does not stay reserved in the training pool across epochs.
- ``CudaMemoryLoggerCallback`` — logs ``allocated`` / ``reserved`` /
  per-phase peak to the experiment logger so steady-state drift is
  visible directly in TensorBoard.

Diagnosis cheat-sheet for ``cuda_memory/*`` curves:

- ``allocated_gb`` flat, ``reserved_gb`` rising → fragmentation;
  re-launch with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.
- both rising → real tensor leak; chase callback / image-logger refs.
- sawtooth tied to val epochs → val peak polluting train pool; enable
  ``callbacks.cuda_empty_cache_before_val`` (which now empties on both
  sides of validation).
"""

import torch
import pytorch_lightning as pl


class CudaEmptyCacheCallback(pl.Callback):
    """Clear the CUDA allocator cache around each validation epoch.

    Validation runs a different memory profile from training (sliding-window
    inference, full-resolution decode, image-logger snapshots), so its
    peak otherwise ends up reserved in the same allocator pool as training.
    Emptying on *both* sides of validation stops the val high-water mark
    from contaminating train memory across epochs.
    """

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


class CudaMemoryLoggerCallback(pl.Callback):
    """Log CUDA memory stats to the experiment logger.

    Emits four scalars under ``cuda_memory/`` (rank-0 device):

    - ``allocated_gb``         : live tensors right now
      (``torch.cuda.memory_allocated``).
    - ``reserved_gb``          : reserved-pool size
      (``torch.cuda.memory_reserved``).
    - ``max_allocated_gb_train`` : peak since last reset, captured at the
      end of every train epoch, then peak stats are reset.
    - ``max_allocated_gb_val``   : same, captured at the end of every val
      epoch.

    ``allocated`` / ``reserved`` are pushed every ``log_every_n_steps``
    training batches (default 50) so the trajectory is dense enough to
    spot slow drift but does not flood the event file.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every_n_steps = max(int(log_every_n_steps), 1)

    @staticmethod
    def _to_gb(num_bytes: int) -> float:
        return float(num_bytes) / (1024 ** 3)

    def _log_instantaneous(
        self, pl_module: pl.LightningModule, on_step: bool
    ) -> None:
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        pl_module.log(
            "cuda_memory/allocated_gb",
            self._to_gb(torch.cuda.memory_allocated(device)),
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        pl_module.log(
            "cuda_memory/reserved_gb",
            self._to_gb(torch.cuda.memory_reserved(device)),
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
        )

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not torch.cuda.is_available():
            return
        if (trainer.global_step % self.log_every_n_steps) != 0:
            return
        self._log_instantaneous(pl_module, on_step=True)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        pl_module.log(
            "cuda_memory/max_allocated_gb_train",
            self._to_gb(torch.cuda.max_memory_allocated(device)),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        torch.cuda.reset_peak_memory_stats(device)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        pl_module.log(
            "cuda_memory/max_allocated_gb_val",
            self._to_gb(torch.cuda.max_memory_allocated(device)),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        torch.cuda.reset_peak_memory_stats(device)
