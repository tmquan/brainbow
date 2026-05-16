"""The public :class:`CosmosTransfer3DWrapper` model.

Cosmos-Transfer 2.5 is the "base DiT + ControlNet" variant of the
Cosmos 2.5 family.  All shared scaffolding (VAE encode, DiT feature
extraction, unified-head decoder adapter, freeze plumbing) lives in
:mod:`brainbow.models.cosmos_2_5_common`; this file only contributes
the ControlNet branch on top.

ControlNet split
----------------
Upstream Cosmos-Transfer 2.5 is a **base DiT + ControlNet** stack
(see :mod:`.variants`).  This wrapper loads both branches:

* ``self.dit``         -- ``CosmosTransformer3DModel`` (~2 B params),
                          the "upper" path; frozen by default.
* ``self.controlnet``  -- ``CosmosControlNetModel`` (a few replicated
                          transformer blocks), the trainable
                          residual path that injects
                          ``block_controlnet_hidden_states`` into
                          the base every
                          ``controlnet_block_every_n`` blocks.

For EM segmentation we feed the **same** EM VAE latent into both
paths -- the base sees the volume as ``hidden_states`` and the
ControlNet sees it as ``controls_latents``.
"""

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn

from brainbow.models.cosmos_2_5_common.hf_loader import _download_from_hf
from brainbow.models.cosmos_2_5_common.wrapper_base import _BaseCosmos25Wrapper
from brainbow.models.cosmos_transfer_2_5.variants import _VARIANT_CONFIGS

logger = logging.getLogger(__name__)


class CosmosTransfer3DWrapper(_BaseCosmos25Wrapper):
    """Cosmos-Transfer 2.5 adapted for **volumetric** connectomics segmentation.

    A single unified task head produces ``[B, 30, D, H, W]``.  Channel
    layout is owned by :mod:`brainbow.losses._common`: raw(1), sem(1),
    dir(3), cov(6), avg(3), emb(16).

    Because Cosmos-Transfer 2.5 is natively a video model, the depth
    axis of the EM volume is mapped to the temporal axis of the
    backbone, making the 3-D adaptation architecturally natural.

    Args:
        in_channels: Number of input channels (1 for EM volumes).
        head_channels: Unified head width (default 32).
        feature_size: Internal feature map channel count after projection.
        variant: ``"2B"`` or ``"14B"`` model variant.
        checkpoint_variant: HuggingFace revision string.
        dtype: Weight dtype (``"bf16"``, ``"fp16"``, ``"fp32"``).
        freeze_dit_backbone: Whether to freeze the pretrained base DiT
            (the "upper part").  Defaults to ``False`` (legacy behaviour
            -- the recipe config typically flips this to ``True`` so
            only the ControlNet trains).
        freeze_controlnet: Whether to freeze the ControlNet (the
            "residual part").  Defaults to ``False`` so the ControlNet
            adapts to the EM domain.
        controlnet_revision: HF revision for the ControlNet weights.
            ``None`` falls back to the variant default.  Pass an empty
            string ``""`` (or set the variant default to ``None``) to
            disable the ControlNet load path entirely.
        feature_layers: DiT block indices to extract features from.
        cache_dir: HuggingFace download cache directory.
        hf_token: HuggingFace authentication token.
        dropout: Dropout probability for heads.

    Example::

        >>> model = CosmosTransfer3DWrapper(in_channels=1, variant="2B")
        >>> x = torch.randn(1, 1, 32, 64, 64)
        >>> out = model(x)
        >>> out.shape   # [1, 30, 32, 64, 64]
    """

    _variant_configs = _VARIANT_CONFIGS

    def __init__(
        self,
        *args: Any,
        controlnet_revision: Optional[str] = None,
        freeze_controlnet: bool = False,
        **kwargs: Any,
    ) -> None:
        # Stash subclass-specific args BEFORE super().__init__() runs so
        # ``_init_arch_state`` (called from inside the base init) can
        # see them.  Plain attributes are safe to set on ``nn.Module``
        # before ``Module.__init__`` -- only Parameter / sub-Module
        # registration requires the parent init to have completed.
        self._pending_controlnet_revision = controlnet_revision
        self._freeze_controlnet = freeze_controlnet

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Subclass extension hooks
    # ------------------------------------------------------------------

    def _init_arch_state(self) -> None:
        """Resolve the ControlNet revision and reserve the ``controlnet`` slot."""
        cn_rev = self._pending_controlnet_revision
        # Empty string acts as an explicit "disable controlnet"
        # override; ``None`` means "fall back to variant default".
        self._controlnet_revision = (
            cn_rev if cn_rev is not None else self.cfg.hf_revision_controlnet
        ) or None
        self.controlnet: Optional[nn.Module] = None
        self._controlnet_loaded = False

    def _post_load_diffusers(
        self,
        local_path: Any,
        cache_dir: Optional[str],
        hf_token: Optional[str],
    ) -> None:
        """Load the ControlNet branch after the base DiT + VAE are in place."""
        self._try_load_controlnet(cache_dir, hf_token)

    def _post_init_freezes(self) -> None:
        """Apply freeze policy to the ControlNet branch (if loaded)."""
        if self.controlnet is None:
            return
        if self._freeze_controlnet:
            self.freeze_controlnet()
        else:
            self.controlnet.train()

    def _hook_should_detach(self) -> bool:
        # Only detach when there is no trainable path through the block.
        # When the ControlNet is trainable its residuals are summed into
        # the block output (see ``CosmosTransformerBlock.forward``:
        # ``hidden_states += controlnet_residual``), so detaching here
        # would break the gradient path back to the ControlNet weights.
        cn_trainable = self.controlnet is not None and not self._freeze_controlnet
        return self._freeze_dit_backbone and not cn_trainable

    def _any_trainable(self) -> bool:
        # Grad must be enabled whenever *any* trainable branch is in
        # play.  The ControlNet residuals are summed into base-DiT
        # block outputs, so even with a frozen base we still need
        # autograd enabled for the ControlNet update path.
        cn_trainable = self.controlnet is not None and not self._freeze_controlnet
        return (not self._freeze_dit_backbone) or cn_trainable

    def _compute_controlnet_residuals(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        enc_hidden: Any,
        padding_mask: torch.Tensor,
        null_condition: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        """Run the ControlNet residual branch over the same EM latent.

        Cosmos-Transfer 2.5 sums these per-block residuals into the
        base DiT every ``controlnet_block_every_n`` blocks (see
        ``CosmosTransformer3DModel.forward``).  We feed the *same* EM
        latent into both ``controls_latents`` and ``latents`` -- the
        EM volume is simultaneously the generative input and its own
        conditioning signal.
        """
        if self.controlnet is None:
            return None
        cn_ctx = (
            torch.no_grad() if self._freeze_controlnet
            else torch.enable_grad()
        )
        with cn_ctx:
            cn_out = self.controlnet(
                controls_latents=latent,
                latents=latent,
                timestep=timestep,
                encoder_hidden_states=enc_hidden,
                condition_mask=null_condition,
                conditioning_scale=1.0,
                padding_mask=padding_mask,
                return_dict=False,
            )
            return cn_out[0]

    # ------------------------------------------------------------------
    # ControlNet loader
    # ------------------------------------------------------------------

    def _try_load_controlnet(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
    ) -> None:
        """Load the ControlNet residual branch (``CosmosControlNetModel``).

        Failures here are non-fatal: the wrapper still works on the
        base DiT alone.  When the ControlNet is loaded its
        ``control_block_samples`` are summed into the base DiT's
        block hidden states inside :meth:`_compute_controlnet_residuals`.
        """
        if self._controlnet_revision is None:
            logger.info(
                "No ControlNet revision configured -- running on base DiT only.",
            )
            return

        try:
            from diffusers import (  # type: ignore[import-untyped]
                CosmosControlNetModel,
            )
        except ImportError:
            logger.warning(
                "diffusers does not expose CosmosControlNetModel "
                "-- skipping ControlNet load.",
            )
            return

        try:
            local_path = _download_from_hf(
                self.cfg.hf_repo_id,
                revision=self._controlnet_revision,
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception as exc:
            logger.warning(
                "ControlNet HuggingFace download failed (rev=%s): %s",
                self._controlnet_revision, exc,
            )
            return

        try:
            controlnet = CosmosControlNetModel.from_pretrained(
                str(local_path),
                torch_dtype=self._dtype,
            )
        except Exception:
            try:
                controlnet = CosmosControlNetModel.from_pretrained(
                    str(local_path),
                    subfolder="controlnet",
                    torch_dtype=self._dtype,
                )
            except Exception as exc:
                logger.warning("ControlNet load failed: %s", exc)
                return

        self.controlnet = controlnet.to(self._dtype)
        self._controlnet_loaded = True
        logger.info(
            "Loaded Cosmos ControlNet via diffusers (rev=%s, %d control blocks).",
            self._controlnet_revision,
            len(getattr(self.controlnet, "control_blocks", [])),
        )

    # ------------------------------------------------------------------
    # ControlNet freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_controlnet(self) -> None:
        if self.controlnet is None:
            return
        self.controlnet.requires_grad_(False)
        self.controlnet.eval()
        self._freeze_controlnet = True
        logger.info("ControlNet frozen (%s trainable params).",
                     f"{self.get_num_parameters(True):,}")

    def unfreeze_controlnet(self) -> None:
        if self.controlnet is None:
            return
        self.controlnet.requires_grad_(True)
        self.controlnet.train()
        self._freeze_controlnet = False
        logger.info("ControlNet unfrozen (%s trainable params).",
                     f"{self.get_num_parameters(True):,}")


__all__ = ["CosmosTransfer3DWrapper"]
