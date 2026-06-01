"""The public :class:`Cosmos3Nano3DWrapper` model.

Cosmos 3 (Nano) is NVIDIA's 16B omnimodal world model.  Its generator
is a Mixture-of-Transformers (``Cosmos3OmniTransformer``) that jointly
models text / image / video / audio / action; for volumetric EM
segmentation we drive only its diffusion (video) tower as a feature
extractor and feed null conditioning for every other modality.

Almost all scaffolding (HuggingFace download, VAE encode, multi-layer
feature extraction via persistent block hooks, the unified-head decoder
adapter, freeze / gradient-checkpointing plumbing) is inherited
verbatim from
:class:`brainbow.models.cosmos_2_5_common.wrapper_base._BaseCosmos25Wrapper`.
This file owns only:

* the Cosmos 3 variant registry (``variants.py``);
* the diffusers class names to load (``Cosmos3OmniTransformer`` +
  the shared ``AutoencoderKLWan``);
* the HF-snapshot ignore list (skip the unused omni sub-towers);
* the omni forward call (:meth:`_run_dit_forward`).

Because the base DiT is kept on the ``self.dit`` attribute -- exactly
as Predict / Transfer do -- every downstream convention that keys on
``model.dit.*`` keeps working unchanged: the optimiser param-group
split, the ``freeze_dit_backbone`` schedule, and the
``ckpt_path_skip_prefixes=[model.dit.]`` warm-start filter.

References:
    - HuggingFace: nvidia/Cosmos3-Nano
    - https://github.com/nvidia/cosmos
"""

import inspect
import logging
from typing import Any, List, Optional

import torch

from brainbow.models.cosmos_2_5_common.wrapper_base import _BaseCosmos25Wrapper
from brainbow.models.cosmos_3_nano.variants import _VARIANT_CONFIGS

logger = logging.getLogger(__name__)


class Cosmos3Nano3DWrapper(_BaseCosmos25Wrapper):
    """Cosmos 3 (Nano) omni transformer as a volumetric EM feature extractor.

    A single unified task head produces ``[B, head_channels, D, H, W]``
    (default 32); the channel layout is owned by
    :mod:`brainbow.losses._common`.

    The depth axis of the EM volume maps to the model's temporal (video)
    axis, exactly as for the Cosmos 2.5 wrappers::

        EM volume  [B, C, D, H, W]  <->  video  [B, C, T, H, W]

    The Wan2.2-TI2V VAE compresses 16x spatially and 4x temporally into
    a 48-channel latent grid, which the omni transformer's diffusion
    tower then processes.

    Args:
        in_channels: Number of input channels (1 for EM volumes).
        head_channels: Unified head width (default 32).
        feature_size: Internal feature map channel count after projection.
        variant: Cosmos 3 variant key (only ``"Nano"`` is published).
        dtype: Weight dtype.  Cosmos 3 is officially BF16-only; keep
            ``"bf16"`` unless you know what you are doing.
        pretrained: Auto-pull ``nvidia/Cosmos3-Nano`` from HF on first
            instantiation.
        freeze_dit_backbone: ``True`` / ``False`` / ``N`` (frozen for
            epochs ``0..N-1``, thawed at epoch ``N``).  Strongly
            recommended to keep frozen (or warm-up frozen) on a 16B
            backbone -- an end-to-end fine-tune needs a lot of memory.
        feature_layers: Omni-transformer block indices to extract
            features from.  Defaults (from the base class) to four
            evenly-spaced layers across the 36-layer stack.

    Example::

        >>> model = Cosmos3Nano3DWrapper(in_channels=1, variant="Nano")
        >>> x = torch.randn(1, 1, 16, 256, 256)
        >>> out = model(x)
        >>> out.shape   # [1, 32, 16, 256, 256]
    """

    _variant_configs = _VARIANT_CONFIGS

    def __init__(self, *args: Any, variant: str = "NANO", **kwargs: Any) -> None:
        super().__init__(*args, variant=variant, **kwargs)

    # ------------------------------------------------------------------
    # Backbone selection
    # ------------------------------------------------------------------

    def _diffusers_transformer_cls_name(self) -> str:
        # The Cosmos 3 omni generator (``model_index.json`` ->
        # ``transformer: ["diffusers", "Cosmos3OmniTransformer"]``).
        return "Cosmos3OmniTransformer"

    def _diffusers_vae_cls_name(self) -> str:
        # Cosmos 3 ships the Wan2.2-TI2V VAE under the same diffusers
        # class the Cosmos 2.5 stack already uses (just a different
        # config: z_dim=48, 16x spatial / 4x temporal).
        return "AutoencoderKLWan"

    def _hf_ignore_patterns(self) -> Optional[List[str]]:
        # We only load ``transformer/`` + ``vae/`` and feed null
        # conditioning for text / image / audio, so skip the heavy omni
        # sub-towers and tokenizers (saves tens of GB per snapshot).
        from brainbow.models.cosmos_2_5_common.hf_loader import (
            _DEFAULT_IGNORE_PATTERNS,
        )

        return list(_DEFAULT_IGNORE_PATTERNS) + [
            "vision_encoder/*",
            "sound_tokenizer/*",
            "text_tokenizer/*",
        ]

    # ------------------------------------------------------------------
    # Omni forward
    # ------------------------------------------------------------------

    def _run_dit_forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        """Drive the omni transformer's diffusion tower over ``latent``.

        Features are captured by the persistent block hooks the base
        class registers, so this only has to make ``self.dit`` *run*
        over the video latent with null conditioning.

        ``Cosmos3OmniTransformer`` (diffusers >= 0.37, bleeding-edge) has
        a materially different signature from the Cosmos 2.5
        ``CosmosTransformer3DModel`` -- it is an omni MoT whose forward
        accepts an interleaved text / vision / video token stream.
        Rather than hard-code a guessed signature, we introspect the
        *installed* ``forward`` and pass only the kwargs it actually
        declares, supplying null text conditioning sized from the model
        config.

        INTEGRATION NOTE: this is the one Cosmos-3-specific call that is
        not pinned to a frozen API.  If a future diffusers release
        renames these kwargs (or requires the video tokens to be packed
        into the joint sequence rather than passed as ``hidden_states``),
        update the ``candidate_kwargs`` map below.  Everything else in
        this package is keyed off the published config and is stable.
        """
        forward = self.dit.forward
        try:
            params = set(inspect.signature(forward).parameters)
        except (TypeError, ValueError):
            params = set()

        B = latent.shape[0]
        dit_cfg = getattr(self.dit, "config", None)
        text_dim = (
            getattr(dit_cfg, "hidden_size", None)
            or getattr(dit_cfg, "hidden_dim", None)
            or self.cfg.hidden_dim
        )
        null_text = torch.zeros(
            B, 1, text_dim, device=latent.device, dtype=latent.dtype,
        )

        # Superset of plausible diffusion-tower kwargs; filtered to the
        # ones the installed class actually accepts so unexpected names
        # never raise ``TypeError``.
        candidate_kwargs = {
            "hidden_states": latent,
            "timestep": timestep,
            "encoder_hidden_states": null_text,
            "return_dict": False,
        }
        call_kwargs = {k: v for k, v in candidate_kwargs.items() if k in params}

        with self._dit_forward_without_ckpt_when_eval():
            if "hidden_states" in call_kwargs:
                self.dit(**call_kwargs)
            else:
                # First positional arg is the latent under a different
                # name (e.g. ``inputs_embeds`` / ``sample``).
                extra = {
                    k: v for k, v in call_kwargs.items()
                    if k not in ("hidden_states",)
                }
                self.dit(latent, **extra)


__all__ = ["Cosmos3Nano3DWrapper"]
