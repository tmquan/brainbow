"""
Cosmos Lightning module for 3-D volumetric segmentation.

Specialisation of :class:`~brainbow.modules.base.BaseCircuitModule` for
the Cosmos-Transfer2.5 backbone.  The base class owns training,
evaluation and logging; this module adds:

* HuggingFace token handling (kept out of ``save_hyperparameters``)
* phased freeze / unfreeze of the VAE encoder, DiT backbone and VAE
  decoder via :meth:`on_train_epoch_start`
* NaN/Inf gradient zeroing + backbone-specific AdamW learning rate
* param-group split (backbone vs heads) in :meth:`configure_optimizers`

Only the **automatic** training mode is supported (predict from the
volume alone).  Point-prompt / proofread training is a Vista-only path.
"""

import logging
from typing import Any, Dict, Optional

import torch

from brainbow.modules.base import BaseCircuitModule

logger = logging.getLogger(__name__)


class BaseCosmosModule(BaseCircuitModule):
    """Abstract base for Cosmos-Predict / Cosmos-Transfer 3-D modules.

    Subclasses **must** define :attr:`_model_cls` and :attr:`_loss_cls`.

    Args:
        model_config: Forwarded to ``_model_cls`` (see
            :class:`brainbow.models.cosmos_transfer_2_5.CosmosTransfer3DWrapper`).
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded as ``**loss_config`` to ``_loss_cls``.
        training_config: Training behaviour (clusterer, freeze schedule, ...).
    """

    _SPATIAL_DIMS = 3

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        model_config = dict(model_config or {})
        # ``hf_token`` is intentionally not persisted via save_hyperparameters.
        hf_token = model_config.pop("hf_token", None)
        self.save_hyperparameters()
        if hf_token is not None:
            model_config["hf_token"] = hf_token

        super().__init__(
            model_config=model_config,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            training_config=training_config,
            **kwargs,
        )

        if self._disabled_heads:
            logger.info("Heads disabled (weight=0): %s", sorted(self._disabled_heads))

        if self.model._backbone_loaded and self.model.vae_encoder is not None:
            self.model._fallback_down.requires_grad_(False)

        logger.info(
            "Validation clusterer: %s",
            type(self.clusterer).__name__,
        )

        # Phased freeze schedule: value is either a bool ("permanently
        # frozen / permanently trainable") or an int ("frozen for the
        # first N epochs, then unfreeze").
        self._freeze_schedule = {
            "vae_encoder": model_config.get("freeze_vae_encoder", True),
            "dit_backbone": model_config.get("freeze_dit_backbone", False),
            "vae_decoder": model_config.get("freeze_vae_decoder", False),
        }

    def _build_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        return self._model_cls(
            in_channels=model_config.get("in_channels", 1),
            num_classes=model_config.get(
                "semantic_channels",
                model_config.get("num_classes", 1),
            ),
            instance_channels=model_config.get("instance_channels", 10),
            boundary_channels=model_config.get("boundary_channels", 10),
            feature_size=model_config.get("feature_size", 64),
            variant=model_config.get("variant", "2B"),
            dtype=model_config.get("dtype", "bf16"),
            pretrained=model_config.get("pretrained", True),
            freeze_dit_backbone=model_config.get("freeze_dit_backbone", False),
            freeze_vae_decoder=model_config.get("freeze_vae_decoder", False),
            freeze_vae_encoder=model_config.get("freeze_vae_encoder", True),
            gradient_checkpointing=model_config.get("gradient_checkpointing", False),
            feature_layers=model_config.get("feature_layers"),
            cache_dir=model_config.get("cache_dir"),
            hf_token=model_config.get("hf_token"),
            dropout=model_config.get("dropout", 0.0),
            disabled_heads=self._disabled_heads or None,
        )

    # ------------------------------------------------------------------
    # Phased freeze / unfreeze
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        methods = {
            "vae_encoder": (
                self.model.freeze_vae_encoder,
                self.model.unfreeze_vae_encoder,
            ),
            "dit_backbone": (
                self.model.freeze_dit_backbone,
                self.model.unfreeze_dit_backbone,
            ),
            "vae_decoder": (
                self.model.freeze_vae_decoder,
                self.model.unfreeze_vae_decoder,
            ),
        }
        flags = {
            "vae_encoder": "_freeze_vae_encoder",
            "dit_backbone": "_freeze_dit_backbone",
            "vae_decoder": "_freeze_vae_decoder",
        }
        needs_rebuild = False
        for name, schedule in self._freeze_schedule.items():
            if isinstance(schedule, bool):
                continue  # permanently frozen / permanently trainable
            want_frozen = self.current_epoch < int(schedule)
            is_frozen = getattr(self.model, flags[name])
            if want_frozen and not is_frozen:
                methods[name][0]()
            elif not want_frozen and is_frozen:
                methods[name][1]()
                needs_rebuild = True
        if needs_rebuild and self.trainer is not None:
            self.trainer.strategy.setup_optimizers(self.trainer)

    # ------------------------------------------------------------------
    # Optimizer (backbone vs heads split + NaN/Inf gradient zeroing)
    # ------------------------------------------------------------------

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None,
    ) -> None:
        """Zero NaN/Inf gradients before clipping so bad batches don't poison weights."""
        bad = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                    p.grad.zero_()
                    bad += 1
        if bad:
            logger.warning(
                "Zeroed NaN/Inf gradients in %d parameters at step %d.",
                bad, self.global_step,
            )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self) -> Any:
        lr = self.optimizer_config.get("lr", 1e-4)
        wd = self.optimizer_config.get("weight_decay", 1e-5)
        # Use explicit ``is None`` so a deliberate ``dit_backbone_lr: 0``
        # (e.g. to keep the unfrozen DiT weights pinned via gradient-
        # only updates from learned LR schedulers) is honoured rather
        # than silently falling back to ``lr`` via ``or``-truthiness.
        backbone_lr = self.optimizer_config.get("dit_backbone_lr")
        if backbone_lr is None:
            backbone_lr = lr

        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("model.dit.")
            no_decay = param.dim() <= 1 or name.endswith(".bias")
            if is_backbone:
                (backbone_no_decay if no_decay else backbone_decay).append(param)
            else:
                (head_no_decay if no_decay else head_decay).append(param)

        param_groups = [
            {"params": backbone_decay,    "lr": backbone_lr, "weight_decay": wd},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay,        "lr": lr,          "weight_decay": wd},
            {"params": head_no_decay,     "lr": lr,          "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if g["params"]]

        clip_val = self.training_config.get("gradient_clip_val")
        use_fused = (
            not clip_val
            and torch.cuda.is_available()
            and all(p.is_cuda for g in param_groups for p in g["params"])
        )
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=wd, fused=use_fused,
        )

        return self._maybe_wrap_scheduler(optimizer)
