"""
Cosmos 2.5 Lightning-module base for 3-D volumetric segmentation.

Specialisation of :class:`~brainbow.modules.base.BaseCircuitModule`
shared by every Cosmos 2.5 backbone (Predict, Transfer, ...).  The
base ``BaseCircuitModule`` owns training, evaluation and logging;
this module adds:

* HuggingFace token handling (kept out of ``save_hyperparameters``)
* static freeze of the VAE encoder, DiT backbone and VAE decoder
  applied once at construction via the model wrapper's ``freeze_*``
  kwargs
* NaN/Inf gradient zeroing + backbone-specific AdamW learning rate
* param-group split (backbone vs ControlNet vs heads) in
  :meth:`configure_optimizers` -- the ``model.controlnet.*`` group
  is naturally empty for backbones without a ControlNet branch
  (e.g. Cosmos-Predict) and is filtered out before AdamW sees it.

Only the **automatic** training mode is supported (predict from the
volume alone).  Point-prompt / proofread training is a Vista-only path.

Subclasses with architecture-specific model kwargs (e.g.
Cosmos-Transfer's ``controlnet_revision`` / ``freeze_controlnet``)
override :meth:`_extra_model_kwargs` to add them; the base
``_build_model`` only forwards the kwargs the shared
``_BaseCosmos25Wrapper`` accepts.
"""

import logging
from typing import Any, Dict, Optional

import torch

from brainbow.losses import HEAD_CHANNELS
from brainbow.modules.base import BaseCircuitModule

logger = logging.getLogger(__name__)


class BaseCosmosModule(BaseCircuitModule):
    """Abstract base for Cosmos-Predict / Cosmos-Transfer 3-D modules.

    Subclasses **must** define :attr:`_model_cls` and :attr:`_loss_cls`.

    Args:
        model_config: Forwarded to ``_model_cls`` (see
            :class:`brainbow.models.cosmos_2_5_common.wrapper_base._BaseCosmos25Wrapper`).
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

        if self.model._backbone_loaded and self.model.vae_encoder is not None:
            self.model._fallback_down.requires_grad_(False)

        logger.info(
            "Validation clusterer: %s",
            type(self.clusterer).__name__,
        )

    def _build_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        return self._model_cls(
            in_channels=model_config.get("in_channels", 1),
            head_channels=model_config.get("head_channels", HEAD_CHANNELS),
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
            **self._extra_model_kwargs(model_config),
        )

    def _extra_model_kwargs(
        self, model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Subclass hook: forward arch-specific kwargs to ``_model_cls``.

        Cosmos-Transfer overrides this to add ``controlnet_revision``
        and ``freeze_controlnet``; Cosmos-Predict has no extras and
        inherits the empty-dict default.
        """
        return {}

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
        # ControlNet residual branch.  Defaults to the backbone LR
        # (it's also pretrained Cosmos-Transfer 2.5 weight, just a
        # smaller replicated stack); override via ``optimizer.controlnet_lr``.
        # Backbones without a ControlNet (Cosmos-Predict) simply have
        # no parameters in the controlnet groups; empty groups are
        # filtered out before AdamW sees them.
        controlnet_lr = self.optimizer_config.get("controlnet_lr")
        if controlnet_lr is None:
            controlnet_lr = backbone_lr

        backbone_decay, backbone_no_decay = [], []
        controlnet_decay, controlnet_no_decay = [], []
        head_decay, head_no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("model.dit.")
            is_controlnet = name.startswith("model.controlnet.")
            no_decay = param.dim() <= 1 or name.endswith(".bias")
            if is_controlnet:
                (controlnet_no_decay if no_decay else controlnet_decay).append(param)
            elif is_backbone:
                (backbone_no_decay if no_decay else backbone_decay).append(param)
            else:
                (head_no_decay if no_decay else head_decay).append(param)

        param_groups = [
            {"params": backbone_decay,      "lr": backbone_lr,   "weight_decay": wd},
            {"params": backbone_no_decay,   "lr": backbone_lr,   "weight_decay": 0.0},
            {"params": controlnet_decay,    "lr": controlnet_lr, "weight_decay": wd},
            {"params": controlnet_no_decay, "lr": controlnet_lr, "weight_decay": 0.0},
            {"params": head_decay,          "lr": lr,            "weight_decay": wd},
            {"params": head_no_decay,       "lr": lr,            "weight_decay": 0.0},
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


__all__ = ["BaseCosmosModule"]
