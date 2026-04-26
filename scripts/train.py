#!/usr/bin/env python
"""
Brainbow training entry point.

Loads a Hydra config, builds the datamodule + Lightning module + trainer,
then calls ``trainer.fit(...)``.

Examples
--------
    # Train with the default config (brainbow loss on SNEMI3D)
    python scripts/train.py

    # Train with a specific config
    python scripts/train.py --config-name snemi3d

    # Override parameters via CLI
    python scripts/train.py --config-name snemi3d data.batch_size=8 training.max_epochs=200

    # Fast development run
    python scripts/train.py training.fast_dev_run=true
"""

import collections
import datetime
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


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

warnings.filterwarnings("ignore", message=r".*isinstance.*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", message=r".*AccumulateGrad.*stream.*mismatch.*")
warnings.filterwarnings("ignore", message=".*lru_cache.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*sync_dist=True.*when logging on epoch level.*")
warnings.filterwarnings("ignore", message=r".*module.*in eval mode at the start of training.*")

import hydra
import pytorch_lightning as pl
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from omegaconf.nodes import ValueNode, AnyNode

torch.serialization.add_safe_globals([
    Any,
    dict,
    collections.defaultdict,
    DictConfig, ListConfig, ContainerMetadata, ValueNode, AnyNode,
])

# PyTorch >= 2.6 made ``weights_only=True`` the default for ``torch.load``.
# Lightning checkpoints pickle non-tensor objects (``collections.defaultdict``
# in callback / metric state, ``DictConfig`` for hparams, optimizer state
# with custom types) that the weights-only unpickler refuses on top of the
# ``add_safe_globals`` allow-list above -- e.g. ``defaultdict`` is whitelisted
# as a *type*, but the ``SETITEM`` opcode is hardcoded to only accept
# ``dict / OrderedDict / Counter`` as the target, so resume still fails.
# Our checkpoints come from our own runs, so default to ``weights_only=False``
# at the entry point.  Callers can still pass ``weights_only=True`` explicitly.
_orig_torch_load = torch.load


def _torch_load_trusted(*args: Any, **kwargs: Any) -> Any:
    # Force-override: Lightning's checkpoint connector passes
    # ``weights_only=True`` explicitly in newer versions, so a plain
    # ``setdefault`` would be a no-op.  ``weights_only`` is keyword-only
    # in ``torch.load``, so this kwargs assignment is sufficient.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted  # type: ignore[assignment]

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

torch.set_float32_matmul_precision("high")


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------

_DATAMODULE_KWARGS = (
    "data_root", "batch_size", "num_workers", "cache_rate", "pin_memory",
    "persistent_workers", "train_volumes", "val_volumes", "test_volumes",
    "find_boundaries", "pixel_size", "min_foreground", "compute_geometry",
    "elastic_prob", "elastic_sigma_range", "elastic_magnitude_range",
    "resolution_zoom_prob", "resolution_zoom_range", "resolution_map",
    "image_size", "slice_mode", "num_samples", "patch_size",
    "include_clefts", "include_mito",
)


def _to_vol_list(val):
    """Convert an OmegaConf volume list to a list of plain dicts."""
    if val is None:
        return None
    return [dict(v) if hasattr(v, "keys") else v for v in val]


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


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
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
    # Each DataModule ignores kwargs it doesn't declare, so we filter to
    # avoid TypeError on older DataModule signatures.
    import inspect
    accepted = set(inspect.signature(cls).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
    return cls(**kwargs)


def get_module(cfg: DictConfig) -> pl.LightningModule:
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


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------

def _resolve_checkpoint(cfg: DictConfig, module: pl.LightningModule) -> Optional[str]:
    """Pick up an existing checkpoint, either full-resume or weights-only."""
    training_cfg = cfg.training
    resume_ckpt = training_cfg.get("resume_from_checkpoint")
    weights_only_ckpt = cfg.get("ckpt_path")

    if resume_ckpt and weights_only_ckpt:
        raise ValueError(
            "Use either training.resume_from_checkpoint (full Lightning resume) "
            "or +ckpt_path= (weights-only warm start), not both."
        )

    if resume_ckpt:
        print(f"\nFull Lightning resume from: {resume_ckpt}")
        return str(resume_ckpt)

    if weights_only_ckpt:
        print(f"Loading model weights from checkpoint: {weights_only_ckpt}")
        ckpt = torch.load(weights_only_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Model weights loaded (optimiser state skipped).")

    return None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("Brainbow - Connectomics Segmentation Training")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Unique per-run output directory.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.get("output_dir", "outputs")) / f"{timestamp}_{cfg.get('experiment_name', 'run')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg, "output_dir", str(run_dir), force_add=True)
    print(f"\nRun directory: {run_dir}")

    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    print(f"Random seed: {seed}")

    datamodule = get_datamodule(cfg)
    print(f"\nDataModule: {datamodule.__class__.__name__}")
    print(f"  Dataset:    {cfg.data.get('dataset', 'snemi3d')}")
    print(f"  Data root:  {cfg.data.get('data_root', 'data')}")
    print(f"  Batch size: {cfg.data.get('batch_size', 4)}")

    module = get_module(cfg)
    print(f"\nModule: {module.__class__.__name__}")

    backbone_loaded = getattr(getattr(module, "model", None), "_backbone_loaded", None)
    if backbone_loaded is False:
        print(
            "  WARNING: pretrained backbone was NOT loaded — model will train "
            "from random init.  Check HuggingFace cache, network access, and "
            "diffusers version."
        )

    # ``torch.compile``: only compile the trainable DiT backbone.  Compiling
    # the whole wrapper causes torch.compile+DDP to run frozen subgraphs in
    # inference_mode, producing tensors that cannot be saved for backward.
    #
    # ``training.compile`` accepts:
    #   false / null        -> no compile (fastest startup)
    #   true                -> mode="reduce-overhead" (fast compile, good runtime)
    #   "max-autotune"      -> best runtime, 5-15 min first compile
    #   "reduce-overhead"   -> ~1 min compile, ~10-20% slower than max-autotune
    #   "default"           -> minimal overhead, minimal speedup
    # ``training.compile_fullgraph`` (default false) forces a single graph
    # (no graph breaks); safer to leave off on DDP runs.
    compile_cfg = cfg.get("training", {}).get("compile", False)
    if compile_cfg:
        mode = compile_cfg if isinstance(compile_cfg, str) else "reduce-overhead"
        fullgraph = bool(cfg.get("training", {}).get("compile_fullgraph", False))
        dit = getattr(getattr(module, "model", None), "dit", None)
        if dit is not None:
            module.model.dit = torch.compile(dit, mode=mode, fullgraph=fullgraph)
            print(f"  torch.compile enabled on DiT backbone (mode={mode}, fullgraph={fullgraph})")
        else:
            module.model = torch.compile(module.model, mode=mode, fullgraph=fullgraph)
            print(f"  torch.compile enabled on full model (mode={mode}, fullgraph={fullgraph})")

    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)
    profiler = setup_profiler(cfg)
    print(f"\nCallbacks: {len(callbacks)} registered")
    print(f"Logger:    {cfg.get('logger', 'tensorboard')}")
    if profiler is not None:
        print(f"Profiler:  {profiler.__class__.__name__}")

    training_cfg = cfg.training
    trainer = pl.Trainer(
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

    print("\nTrainer initialised:")
    print(f"  Max epochs:   {training_cfg.get('max_epochs', 100)}")
    print(f"  Accelerator:  {training_cfg.get('accelerator', 'auto')}")
    print(f"  Devices:      {training_cfg.get('devices', 1)}")
    print(f"  Precision:    {training_cfg.get('precision', '32-true')}")

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    fit_ckpt_path = _resolve_checkpoint(cfg, module)

    interrupted = False
    try:
        trainer.fit(module, datamodule, ckpt_path=fit_ckpt_path)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted = True
    except Exception as exc:
        print(f"\n\nTraining failed: {exc}")
        if trainer.global_rank == 0:
            recovery = Path(cfg.output_dir) / "checkpoints" / "crash_recovery.ckpt"
            recovery.parent.mkdir(parents=True, exist_ok=True)
            try:
                trainer.save_checkpoint(str(recovery))
                print(f"Recovery checkpoint written to {recovery}")
            except Exception as save_err:
                print(f"Could not save recovery checkpoint: {save_err}")
        raise

    if trainer.global_rank == 0:
        final_path = Path(cfg.output_dir) / "checkpoints" / "final_model.ckpt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if interrupted:
            print("\nWARNING: saving checkpoint from an interrupted run — "
                  "weights may be partially updated.")
        trainer.save_checkpoint(str(final_path))
        print(f"\nFinal model saved: {final_path}")

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
