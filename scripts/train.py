#!/usr/bin/env python
"""
Brainbow training entry point.

Loads a Hydra config, builds the datamodule + Lightning module + trainer,
then calls ``trainer.fit(...)``.

The default Hydra config name is ``default`` (see ``configs/default.yaml``)
which runs SNEMI3D-shaped data with all four loss heads.  Two recipes
are recommended for everyday use:

* ``--config-name snemi3d``   -- SNEMI3D only, three-head recipe.
* ``--config-name boundary``  -- multi-dataset (combine.yaml), 16-channel
  boundary head only.

Examples
--------
    python scripts/train.py --config-name snemi3d
    python scripts/train.py --config-name boundary
    python scripts/train.py --config-name snemi3d data.batch_size=8 training.max_epochs=200
    python scripts/train.py training.fast_dev_run=true
    python scripts/train.py --config-name snemi3d +ckpt_path=outputs/<run>/checkpoints/last.ckpt

Code layout
-----------
* :func:`_install_runtime_patches` -- side-effects deferred out of import.
* :func:`build_datamodule`         -- ``cfg`` -> :class:`LightningDataModule`.
* :func:`build_module`             -- ``cfg`` -> :class:`LightningModule`.
* :func:`build_trainer`            -- ``cfg`` -> :class:`pl.Trainer`.
* :func:`setup_callbacks` / :func:`setup_logger` / :func:`setup_profiler`
  -- the three plug-in lists composed into the Trainer.
* :func:`run_fit_with_recovery`    -- ``trainer.fit`` wrapped with a
  crash-recovery checkpoint on the rank-0 process.
* :func:`main`                     -- Hydra entry point that wires it
  all together.
"""

from __future__ import annotations

import collections
import datetime
import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from omegaconf.nodes import AnyNode, ValueNode
from rich.console import Console

console = Console()


# ----------------------------------------------------------------------
# Runtime patches (deferred out of import time)
# ----------------------------------------------------------------------


def _install_runtime_patches() -> None:
    """Install the global side-effects this script depends on.

    These were previously executed at import time, which made ``import
    scripts.train`` (e.g. from a notebook or a test) silently mutate the
    global ``torch`` module and the warning filters.  Calling this from
    :func:`main` keeps the import side-effect-free.

    The patches are:

    * Allow-list a handful of (Lightning-friendly) types for
      ``torch.load``'s weights-only unpickler.
    * Force ``weights_only=False`` on ``torch.load`` because Lightning
      checkpoints pickle ``defaultdict`` / ``DictConfig`` instances that
      the safe unpickler refuses even with the allow-list above.
    * Silence a handful of noisy warnings emitted by Lightning / MONAI
      that we cannot fix upstream.
    * Bump ``set_float32_matmul_precision`` so TF32 matmuls are allowed.
    """
    torch.serialization.add_safe_globals([
        Any,
        dict,
        collections.defaultdict,
        DictConfig, ListConfig, ContainerMetadata, ValueNode, AnyNode,
    ])

    _orig_torch_load = torch.load

    def _torch_load_trusted(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_trusted  # type: ignore[assignment]

    warnings.filterwarnings("ignore", message=r".*isinstance.*LeafSpec.*is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*AccumulateGrad.*stream.*mismatch.*")
    warnings.filterwarnings("ignore", message=".*lru_cache.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*sync_dist=True.*when logging on epoch level.*")
    warnings.filterwarnings("ignore", message=r".*module.*in eval mode at the start of training.*")

    torch.set_float32_matmul_precision("high")


# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------


def _head_weight_scalar(loss_cfg: Any, head: str, default: float = 0.0) -> float:
    """Extract the scalar head weight from a (nested or flat) loss config.

    Accepts both the new nested form (``weight_<head>`` is a mapping with
    a ``weight`` key) and the legacy flat form (``weight_<head>`` is a
    scalar).  OmegaConf's ``DictConfig`` / ``ListConfig`` are tolerated
    via :class:`collections.abc.Mapping`.
    """
    v = loss_cfg.get(f"weight_{head}", default) if hasattr(loss_cfg, "get") else default
    if isinstance(v, Mapping):
        return float(v.get("weight", default))
    return float(v)


def _to_vol_list(val):
    """Convert an OmegaConf volume list to a list of plain dicts."""
    if val is None:
        return None
    return [dict(v) if hasattr(v, "keys") else v for v in val]


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def _build_datamodule_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Collect the shared kwargs every DataModule understands."""
    data_cfg = cfg.data
    pixel_size = data_cfg.get("pixel_size")
    resolution_zoom_range = data_cfg.get("resolution_zoom_range")
    resolution_map = data_cfg.get("resolution_map")
    image_size = data_cfg.get("image_size")
    patch_size = data_cfg.get("patch_size")

    return {
        "data_root": data_cfg.get("data_root", "data"),
        "batch_size": data_cfg.get("batch_size", 4),
        "num_workers": data_cfg.get("num_workers", 4),
        "cache_rate": data_cfg.get("cache_rate", 0.5),
        "pin_memory": data_cfg.get("pin_memory", True),
        "persistent_workers": bool(data_cfg.get("persistent_workers", True)),
        "prefetch_factor": int(data_cfg.get("prefetch_factor", 6)),
        "train_volumes": _to_vol_list(data_cfg.get("train_volumes")),
        "val_volumes": _to_vol_list(data_cfg.get("val_volumes")),
        "test_volumes": _to_vol_list(data_cfg.get("test_volumes")),
        "find_boundaries": float(data_cfg.get("find_boundaries", 0.0)),
        "pixel_size": tuple(pixel_size) if pixel_size is not None else None,
        "min_foreground": float(data_cfg.get("min_foreground", 0.0)),
        "compute_geometry": _head_weight_scalar(
            cfg.get("loss", {}), "geometry", default=0.0,
        ) > 0,
        "elastic_prob": float(data_cfg.get("elastic_prob", 0.0)),
        "elastic_sigma_range": tuple(data_cfg.get("elastic_sigma_range", [35, 50])),
        "elastic_magnitude_range": tuple(data_cfg.get("elastic_magnitude_range", [10, 40])),
        "resolution_zoom_prob": float(data_cfg.get("resolution_zoom_prob", 0.0)),
        "resolution_zoom_range": (
            tuple(tuple(r) for r in resolution_zoom_range)
            if resolution_zoom_range is not None else None
        ),
        "resolution_map": (
            {str(k): tuple(v) for k, v in resolution_map.items()}
            if resolution_map is not None else None
        ),
        "image_size": tuple(image_size) if isinstance(image_size, list) else image_size,
        "patch_size": tuple(patch_size) if patch_size else None,
        "num_samples": data_cfg.get("num_samples"),
        "slice_mode": data_cfg.get("slice_mode", True),
        "include_clefts": data_cfg.get("include_clefts", True),
        "include_mito": data_cfg.get("include_mito", False),
    }


def build_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Instantiate the datamodule selected by ``cfg.data.dataset``."""
    from brainbow.datamodules import (
        MICRONSDataModule,
        SNEMI3DDataModule,
    )

    datamodule_classes = {
        "snemi3d": SNEMI3DDataModule,
        "microns": MICRONSDataModule,
    }

    dataset_type = cfg.data.get("dataset", "snemi3d").lower()
    cls = datamodule_classes.get(dataset_type)
    if cls is None:
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Choose from: {sorted(datamodule_classes)}"
        )

    kwargs = _build_datamodule_kwargs(cfg)
    accepted = set(inspect.signature(cls).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
    return cls(**kwargs)


def build_module(cfg: DictConfig) -> pl.LightningModule:
    """Instantiate the Lightning module selected by ``cfg.model.type``."""
    from brainbow.modules import (
        Vista3DModule,
        CosmosTransfer3DModule,
    )

    module_classes = {
        "vista3d": Vista3DModule,
        "cosmostransfer3d": CosmosTransfer3DModule,
        # Legacy alias.
        "cosmos_transfer25_3d": CosmosTransfer3DModule,
    }

    model_cfg = dict(cfg.get("model", {}))
    model_type = model_cfg.pop("type", "cosmostransfer3d").lower()

    cls = module_classes.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: {sorted(module_classes)}"
        )

    return cls(
        model_config=model_cfg,
        optimizer_config=dict(cfg.get("optimizer", {})),
        loss_config=dict(cfg.get("loss", {})),
        training_config=dict(cfg.get("training", {})),
    )


# Back-compat aliases for callers that imported the old names.
get_datamodule = build_datamodule
get_module = build_module


def _maybe_compile(module: pl.LightningModule, cfg: DictConfig) -> None:
    """Wrap the trainable DiT backbone with ``torch.compile`` when enabled.

    Only the trainable DiT backbone is compiled, not the whole wrapper:
    compiling frozen subgraphs under DDP runs them in ``inference_mode``,
    which produces tensors that cannot be saved for backward.

    ``training.compile`` accepts:
        false / null        -> no compile (fastest startup)
        true                -> mode="reduce-overhead"
        "max-autotune"      -> best runtime, 5-15 min first compile
        "reduce-overhead"   -> ~1 min compile
        "default"           -> minimal overhead, minimal speedup

    ``training.compile_fullgraph`` (default false) forces a single graph
    (no graph breaks); safer to leave off on DDP runs.
    """
    compile_cfg = cfg.get("training", {}).get("compile", False)
    if not compile_cfg:
        return

    mode = compile_cfg if isinstance(compile_cfg, str) else "reduce-overhead"
    fullgraph = bool(cfg.get("training", {}).get("compile_fullgraph", False))
    dit = getattr(getattr(module, "model", None), "dit", None)
    if dit is not None:
        module.model.dit = torch.compile(dit, mode=mode, fullgraph=fullgraph)
        console.log(f"torch.compile enabled on DiT backbone (mode={mode}, fullgraph={fullgraph})")
    else:
        module.model = torch.compile(module.model, mode=mode, fullgraph=fullgraph)
        console.log(f"torch.compile enabled on full model (mode={mode}, fullgraph={fullgraph})")


def setup_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Build the standard callback list from the ``callbacks`` config block."""
    output_dir = cfg.get("output_dir", "outputs")
    callback_cfg = cfg.get("callbacks", {})
    callbacks: List[pl.Callback] = []

    if callback_cfg.get("cuda_empty_cache_before_val", False):
        from brainbow.callbacks.memory import CudaEmptyCacheCallback
        callbacks.append(CudaEmptyCacheCallback())

    mem_cfg = callback_cfg.get("memory_logger", {})
    if mem_cfg.get("enabled", True):
        from brainbow.callbacks.memory import CudaMemoryLoggerCallback
        mem_every = mem_cfg.get("log_every_n_steps")
        if mem_every is None:
            mem_every = cfg.get("training", {}).get("log_every_n_steps", 50)
        callbacks.append(CudaMemoryLoggerCallback(
            log_every_n_steps=int(mem_every),
        ))

    ckpt_cfg = callback_cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True):
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_cfg.get("dirpath") or str(Path(output_dir) / "checkpoints"),
            filename=ckpt_cfg.get(
                "filename", "{epoch:02d}-{val/automatic/loss:.4f}",
            ),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            monitor=ckpt_cfg.get("monitor", "val/automatic/loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_last=ckpt_cfg.get("save_last", True),
            verbose=ckpt_cfg.get("verbose", True),
            auto_insert_metric_name=False,
        ))

    es_cfg = callback_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(EarlyStopping(
            monitor=es_cfg.get("monitor", "val/automatic/loss"),
            patience=es_cfg.get("patience", 20),
            mode=es_cfg.get("mode", "min"),
            verbose=es_cfg.get("verbose", True),
            min_delta=es_cfg.get("min_delta", 0.0),
        ))

    if callback_cfg.get("lr_monitor", {}).get("enabled", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    img_cfg = callback_cfg.get("image_logger", {})
    if img_cfg.get("enabled", True):
        from brainbow.callbacks import ImageLogger
        callbacks.append(ImageLogger(
            every_n_epochs=img_cfg.get("every_n_epochs", 1),
            max_images=img_cfg.get("max_images", 4),
            spatial_dims=3,
            projection_algorithm=img_cfg.get("projection_algorithm", "pca"),
            projection_backend=img_cfg.get("projection_backend", "auto"),
        ))

    callbacks.append(RichProgressBar())
    callbacks.append(ModelSummary(max_depth=2))
    return callbacks


def setup_logger(cfg: DictConfig):
    """Build the experiment logger (TensorBoard or Weights & Biases)."""
    output_dir = cfg.get("output_dir", "outputs")
    logger_type = cfg.get("logger", "tensorboard")

    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=str(Path(output_dir) / "logs"),
            name=cfg.get("experiment_name", "brainbow"),
            version=None,
        )
    if logger_type == "wandb":
        return WandbLogger(
            project=cfg.get("project_name", "brainbow"),
            name=f"{cfg.get('experiment_name', 'run')}_{cfg.get('seed', 42)}",
            save_dir=str(Path(output_dir) / "logs"),
        )
    return True


def setup_profiler(cfg: DictConfig):
    """Build the training profiler from ``cfg.training.profiler``, or None."""
    output_dir = cfg.get("output_dir", "outputs")
    profiler_type = cfg.get("training", {}).get("profiler")

    if profiler_type == "simple":
        return SimpleProfiler(dirpath=output_dir, filename="profile-simple")
    if profiler_type == "advanced":
        return AdvancedProfiler(dirpath=output_dir, filename="profile-advanced")
    if profiler_type == "pytorch":
        return PyTorchProfiler(
            dirpath=output_dir,
            filename="profile-pytorch",
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(Path(output_dir) / "profiler_traces"),
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    return None


def setup_strategy(cfg: DictConfig):
    """Resolve the Lightning distributed strategy from ``training.strategy``."""
    strategy_name = cfg.training.get("strategy", "auto")
    use_compile = cfg.get("training", {}).get("compile", False)

    if strategy_name == "ddp":
        return DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=not use_compile,
        )
    if strategy_name == "fsdp":
        return FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            activation_checkpointing_policy={torch.nn.modules.conv.Conv3d},
        )
    return strategy_name


def build_trainer(
    cfg: DictConfig,
    *,
    callbacks: List[pl.Callback],
    logger: Any,
    profiler: Any,
) -> pl.Trainer:
    """Construct the :class:`pl.Trainer` from ``cfg.training``."""
    training_cfg = cfg.training
    return pl.Trainer(
        max_epochs=training_cfg.get("max_epochs", 100),
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        strategy=setup_strategy(cfg),
        precision=training_cfg.get("precision", "32-true"),
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        log_every_n_steps=training_cfg.get("log_every_n_steps", 50),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        limit_val_batches=training_cfg.get("limit_val_batches", 1.0),
        val_check_interval=training_cfg.get("val_check_interval", 1.0),
        check_val_every_n_epoch=training_cfg.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=training_cfg.get("num_sanity_val_steps", 2),
        enable_progress_bar=training_cfg.get("enable_progress_bar", True),
        enable_model_summary=training_cfg.get("enable_model_summary", True),
        deterministic=training_cfg.get("deterministic", False),
        benchmark=training_cfg.get("benchmark", True),
        fast_dev_run=training_cfg.get("fast_dev_run", False),
    )


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------


def _resolve_checkpoint(cfg: DictConfig, module: pl.LightningModule) -> Optional[str]:
    """Pick up an existing checkpoint, either full-resume or weights-only.

    Returns the path to pass to ``trainer.fit(..., ckpt_path=...)`` for
    full-resume, or ``None`` after applying a weights-only load in-place.
    """
    training_cfg = cfg.training
    resume_ckpt = training_cfg.get("resume_from_checkpoint")
    weights_only_ckpt = cfg.get("ckpt_path")

    if resume_ckpt and weights_only_ckpt:
        raise ValueError(
            "Use either training.resume_from_checkpoint (full Lightning resume) "
            "or +ckpt_path= (weights-only warm start), not both."
        )

    if resume_ckpt:
        console.log(f"Full Lightning resume from: {resume_ckpt}")
        return str(resume_ckpt)

    if weights_only_ckpt:
        console.log(f"Loading model weights from checkpoint: {weights_only_ckpt}")
        ckpt = torch.load(weights_only_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing:
            console.log(f"  Missing keys: {len(missing)}")
        if unexpected:
            console.log(f"  Unexpected keys: {len(unexpected)}")
        console.log("Model weights loaded (optimiser state skipped).")

    return None


# ----------------------------------------------------------------------
# Fit + crash recovery
# ----------------------------------------------------------------------


def run_fit_with_recovery(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    *,
    ckpt_path: Optional[str],
    output_dir: Path,
) -> bool:
    """Call ``trainer.fit`` with a crash-recovery checkpoint on rank 0.

    Returns ``True`` if the run completed normally, ``False`` if it was
    interrupted by ``KeyboardInterrupt``.  Other exceptions propagate
    after the recovery checkpoint is written.
    """
    try:
        trainer.fit(module, datamodule, ckpt_path=ckpt_path)
        return True
    except KeyboardInterrupt:
        console.log("[yellow]Training interrupted by user.[/yellow]")
        return False
    except Exception as exc:
        console.log(f"[red]Training failed: {exc}[/red]")
        if trainer.global_rank == 0:
            recovery = output_dir / "checkpoints" / "crash_recovery.ckpt"
            recovery.parent.mkdir(parents=True, exist_ok=True)
            try:
                trainer.save_checkpoint(str(recovery))
                console.log(f"Recovery checkpoint written to {recovery}")
            except Exception as save_err:  # noqa: BLE001 -- best effort
                console.log(f"[red]Could not save recovery checkpoint: {save_err}[/red]")
        raise


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    _install_runtime_patches()

    console.rule("Brainbow - Connectomics Segmentation Training")
    console.print("\n[bold]Configuration:[/bold]")
    console.print(OmegaConf.to_yaml(cfg))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.get("output_dir", "outputs")) / f"{timestamp}_{cfg.get('experiment_name', 'run')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg, "output_dir", str(run_dir), force_add=True)
    console.log(f"Run directory: {run_dir}")

    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    console.log(f"Random seed: {seed}")

    datamodule = build_datamodule(cfg)
    console.log(f"DataModule: {datamodule.__class__.__name__}")
    console.log(f"  Dataset:    {cfg.data.get('dataset', 'snemi3d')}")
    console.log(f"  Data root:  {cfg.data.get('data_root', 'data')}")
    console.log(f"  Batch size: {cfg.data.get('batch_size', 4)}")

    module = build_module(cfg)
    console.log(f"Module: {module.__class__.__name__}")

    backbone_loaded = getattr(getattr(module, "model", None), "_backbone_loaded", None)
    if backbone_loaded is False:
        console.log(
            "[yellow]WARNING: pretrained backbone was NOT loaded - model will train "
            "from random init.  Check HuggingFace cache, network access, and "
            "diffusers version.[/yellow]"
        )

    _maybe_compile(module, cfg)

    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)
    profiler = setup_profiler(cfg)
    console.log(f"Callbacks: {len(callbacks)} registered")
    console.log(f"Logger:    {cfg.get('logger', 'tensorboard')}")
    if profiler is not None:
        console.log(f"Profiler:  {profiler.__class__.__name__}")

    trainer = build_trainer(cfg, callbacks=callbacks, logger=logger, profiler=profiler)

    training_cfg = cfg.training
    console.log("Trainer initialised:")
    console.log(f"  Max epochs:   {training_cfg.get('max_epochs', 100)}")
    console.log(f"  Accelerator:  {training_cfg.get('accelerator', 'auto')}")
    console.log(f"  Devices:      {training_cfg.get('devices', 1)}")
    console.log(f"  Precision:    {training_cfg.get('precision', '32-true')}")

    console.rule("Starting Training")

    fit_ckpt_path = _resolve_checkpoint(cfg, module)
    completed = run_fit_with_recovery(
        trainer,
        module,
        datamodule,
        ckpt_path=fit_ckpt_path,
        output_dir=Path(cfg.output_dir),
    )

    if trainer.global_rank == 0:
        final_path = Path(cfg.output_dir) / "checkpoints" / "final_model.ckpt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if not completed:
            console.log(
                "[yellow]WARNING: saving checkpoint from an interrupted run - "
                "weights may be partially updated.[/yellow]"
            )
        trainer.save_checkpoint(str(final_path))
        console.log(f"Final model saved: {final_path}")

    console.rule("Training Complete")


if __name__ == "__main__":
    main()
