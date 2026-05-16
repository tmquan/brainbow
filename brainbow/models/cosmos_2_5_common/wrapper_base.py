"""Shared backbone wrapper for the Cosmos 2.5 family.

:class:`_BaseCosmos25Wrapper` factors out everything Cosmos-Transfer 2.5
and Cosmos-Predict 2.5 have in common:

* HuggingFace snapshot download + diffusers-class instantiation (base
  DiT + Wan-style VAE)
* random-init ``_StandaloneDiT3D`` fallback for ``pretrained=False``
* multi-layer DiT feature extraction via persistent forward hooks
* Wan-VAE encode/decode + the unified 32-channel
  :class:`_DecoderAdapter3D` head
* freeze plumbing for the DiT, VAE encoder and VAE decoder
* gradient checkpointing on/off
* parameter-contiguity fix for DDP

Cosmos-Transfer 2.5 adds a ControlNet residual branch on top via the
two extension hooks (:meth:`_post_load_diffusers` and
:meth:`_compute_controlnet_residuals`); Cosmos-Predict 2.5 simply
inherits this base class without overrides.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from brainbow.losses import HEAD_CHANNELS
from brainbow.models.cosmos_2_5_common.decoder import (
    _DecoderAdapter3D,
    _FeatureProjector3D,
)
from brainbow.models.cosmos_2_5_common.hf_loader import _download_from_hf
from brainbow.models.cosmos_2_5_common.layers import (
    _NORM,
    _PointwiseLinear,
    _SPATIAL_DIMS,
    _adapt_to_rgb,
)
from brainbow.models.cosmos_2_5_common.standalone_dit import _StandaloneDiT3D
from brainbow.models.cosmos_2_5_common.variants import _VariantConfigBase

logger = logging.getLogger(__name__)


class _BaseCosmos25Wrapper(nn.Module):
    """Shared base for Cosmos 2.5 wrappers (Transfer / Predict).

    Subclasses set :attr:`_variant_configs` to their family's variant
    registry.  They may also override the following extension hooks:

    * :meth:`_init_arch_state` -- runs once between shared state setup
      and the backbone build.  Use it to initialise architecture-
      specific attributes (e.g. ``self.controlnet``) that downstream
      methods depend on.
    * :meth:`_post_load_diffusers` -- runs after the base DiT + VAE
      have been successfully loaded via diffusers.  Cosmos-Transfer
      uses this to load its ControlNet branch.
    * :meth:`_hook_should_detach` / :meth:`_any_trainable` -- adjust
      gradient policy when an extra trainable branch (e.g. ControlNet)
      is in play.
    * :meth:`_compute_controlnet_residuals` -- compute residual hidden
      states summed into the base DiT every ``controlnet_block_every_n``
      blocks; defaults to ``None`` (no residuals).
    """

    _variant_configs: Dict[str, _VariantConfigBase] = {}

    def __init__(
        self,
        in_channels: int = 1,
        head_channels: int = HEAD_CHANNELS,
        feature_size: int = 64,
        variant: str = "2B",
        checkpoint_variant: str = "post-trained",
        dtype: str = "bf16",
        pretrained: bool = True,
        freeze_dit_backbone: bool = False,
        freeze_vae_decoder: bool = False,
        freeze_vae_encoder: bool = True,
        gradient_checkpointing: bool = False,
        feature_layers: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not self._variant_configs:
            raise RuntimeError(
                f"{type(self).__name__} did not set ``_variant_configs``.  "
                "Concrete Cosmos 2.5 wrappers must populate this with a "
                "variant -> _VariantConfigBase mapping."
            )

        variant = variant.upper()
        if variant not in self._variant_configs:
            raise ValueError(
                f"Unknown variant '{variant}'.  "
                f"Choose from: {list(self._variant_configs)}"
            )

        self.variant = variant
        self.cfg: _VariantConfigBase = self._variant_configs[variant]
        self.in_channels = in_channels
        self.head_channels = int(head_channels)
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS
        self.dropout = dropout

        self._dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        self._freeze_dit_backbone = freeze_dit_backbone
        self._freeze_vae_decoder = freeze_vae_decoder
        self._freeze_vae_encoder = freeze_vae_encoder
        self._gradient_checkpointing = gradient_checkpointing

        if feature_layers is not None:
            self._feature_layers = sorted(feature_layers)
        else:
            n = self.cfg.num_layers
            self._feature_layers = sorted(
                {n // 4, n // 2, 3 * n // 4, n - 1}
            )

        s = self.cfg.spatial_compression
        t = self.cfg.temporal_compression
        lc = self.cfg.latent_channels
        self._fallback_down = nn.Sequential(
            nn.Conv3d(3, lc * 2, kernel_size=(t, s, s), stride=(t, s, s)),
            _NORM(lc * 2),
            nn.GELU(),
            _PointwiseLinear(lc * 2, lc),
        )

        self._backbone_loaded = False
        self._pretrained = pretrained

        self._init_arch_state()

        self._build_backbone(cache_dir, hf_token, checkpoint_variant)

        self.feature_projector = _FeatureProjector3D(
            hidden_dim=self.cfg.hidden_dim,
            num_feature_layers=len(self._feature_layers),
            out_dim=feature_size,
        ).float()

        if self._backend in ("diffusers", "cosmos_transfer2"):
            self._register_persistent_hooks()

        self.decoder_adapter = _DecoderAdapter3D(
            vae_decoder=self.vae_decoder,
            latent_channels=self.cfg.latent_channels,
            feature_size=feature_size,
            spatial_compression=self.cfg.spatial_compression,
            temporal_compression=self.cfg.temporal_compression,
            dropout=dropout,
            freeze_vae_decoder=freeze_vae_decoder,
            head_channels=self.head_channels,
        )
        if self.decoder_adapter.to_latent is not None:
            self.decoder_adapter.to_latent.float()
        self.decoder_adapter.head.float()

        if self.vae_encoder is not None and freeze_vae_encoder:
            self.vae_encoder.requires_grad_(False)
            self.vae_encoder.eval()

        if freeze_dit_backbone:
            self.freeze_dit_backbone()
        else:
            self.dit.train()

        self._post_init_freezes()

        self._make_params_contiguous()

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

        logger.info(
            "%s initialised: variant=%s, feature_layers=%s, "
            "backbone_loaded=%s, frozen_dit=%s, grad_ckpt=%s, "
            "params=%s (trainable=%s)",
            type(self).__name__,
            variant, self._feature_layers, self._backbone_loaded,
            freeze_dit_backbone, self._gradient_checkpointing,
            f"{self.get_num_parameters(trainable_only=False):,}",
            f"{self.get_num_parameters(trainable_only=True):,}",
        )

    # ------------------------------------------------------------------
    # Subclass extension hooks
    # ------------------------------------------------------------------

    def _init_arch_state(self) -> None:
        """Subclass hook: initialise arch-specific state BEFORE backbone build.

        Runs after the shared dataclass / config attributes are set but
        BEFORE :meth:`_build_backbone` (which may, via
        :meth:`_post_load_diffusers`, depend on attributes the subclass
        needs to expose -- e.g. Cosmos-Transfer's ``self.controlnet =
        None``).
        """
        return

    def _post_load_diffusers(
        self,
        local_path: Any,
        cache_dir: Optional[str],
        hf_token: Optional[str],
    ) -> None:
        """Subclass hook called after a successful diffusers DiT+VAE load.

        Cosmos-Transfer overrides this to load its ControlNet branch
        from a sibling revision of the same HF repo.
        """
        return

    def _post_init_freezes(self) -> None:
        """Subclass hook for additional freeze steps after the base init.

        Cosmos-Transfer uses this to freeze its ControlNet branch when
        configured to do so.
        """
        return

    def _hook_should_detach(self) -> bool:
        """Whether the persistent feature hook should detach captured outputs.

        Default: detach when the DiT backbone is frozen.  Subclasses
        with an additional trainable branch (e.g. ControlNet) override
        this to keep the gradient path alive.
        """
        return self._freeze_dit_backbone

    def _any_trainable(self) -> bool:
        """Whether *any* part of the DiT-side compute graph is trainable.

        Default: ``not self._freeze_dit_backbone``.  Subclasses with an
        additional trainable branch override this so autograd is
        enabled even when the base DiT is frozen.
        """
        return not self._freeze_dit_backbone

    def _compute_controlnet_residuals(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        enc_hidden: Any,
        padding_mask: torch.Tensor,
        null_condition: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        """Compute ControlNet residual hidden states (Transfer-only).

        Default: returns ``None`` (Predict has no ControlNet branch).
        Cosmos-Transfer overrides this to run its ``controlnet`` over
        the same EM latent and return the per-block residuals that
        ``CosmosTransformer3DModel.forward`` will sum into the base
        DiT every ``controlnet_block_every_n`` blocks.
        """
        return None

    # ------------------------------------------------------------------
    # Module placement
    # ------------------------------------------------------------------

    def _apply(self, fn):
        """Extend device/dtype placement to the untracked full-VAE reference.

        ``_vae_ref`` is a plain Python list (not ``nn.ModuleList``) to avoid
        double-registering encoder/decoder parameters in ``state_dict()``.
        This override ensures that auxiliary VAE components (e.g. quant_conv)
        are moved together with the rest of the model.
        """
        super()._apply(fn)
        if hasattr(self, "_vae_ref") and self._vae_ref:
            self._vae_ref[0]._apply(fn)
        return self

    def _make_params_contiguous(self) -> None:
        """Ensure all parameter data tensors are contiguous for DDP."""
        for p in self.parameters():
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _build_backbone(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> None:
        self.vae_encoder: Optional[nn.Module] = None
        self.vae_decoder: Optional[nn.Module] = None
        self.dit: nn.Module

        if not self._pretrained:
            logger.info(
                "pretrained=False -- skipping HuggingFace download; "
                "using randomly initialised 3-D DiT backbone (variant=%s).",
                self.variant,
            )
            self._build_standalone_backbone()
            return

        if self.cfg.hf_repo_id is None:
            raise ValueError(
                f"{type(self).__name__} variant '{self.variant}' has no "
                f"public HuggingFace checkpoint.  Either pass a variant "
                f"with a populated ``hf_repo_id`` or set "
                f"``pretrained=False`` to train from scratch."
            )

        _saved_dtype = torch.get_default_dtype()
        try:
            loaded = (
                self._try_load_diffusers(cache_dir, hf_token, checkpoint_variant)
                or self._try_load_cosmos_package(
                    cache_dir, hf_token, checkpoint_variant,
                )
            )
        finally:
            torch.set_default_dtype(_saved_dtype)

        if not loaded:
            logger.warning(
                "No pretrained weights loaded -- using randomly initialised "
                "3-D DiT backbone (%s architecture).",
                self.variant,
            )
            self._build_standalone_backbone()

    def _try_load_diffusers(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        try:
            from diffusers import (  # type: ignore[import-untyped]
                CosmosTransformer3DModel,
            )
            from diffusers import AutoencoderKLWan as _VAEClass  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("diffusers Cosmos classes not available.")
            return False

        try:
            local_path = _download_from_hf(
                self.cfg.hf_repo_id,
                revision=self.cfg.hf_revision,
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception as exc:
            logger.warning("HuggingFace download failed: %s", exc)
            return False

        try:
            transformer = CosmosTransformer3DModel.from_pretrained(
                str(local_path),
                subfolder="transformer",
                torch_dtype=self._dtype,
            )
            vae = _VAEClass.from_pretrained(
                str(local_path),
                subfolder="vae",
                torch_dtype=self._dtype,
            )

            vae = vae.to(self._dtype)
            self._vae_ref = [vae]
            self.vae_encoder = vae.encoder
            self.vae_decoder = vae.decoder

            self.dit = transformer.to(self._dtype)
            self._backbone_loaded = True
            self._backend = "diffusers"
            logger.info(
                "Loaded base 3-D DiT + VAE via diffusers (local snapshot, "
                "rev=%s).",
                self.cfg.hf_revision,
            )
        except Exception as exc:
            logger.warning("diffusers load from local snapshot failed: %s", exc)
            return False

        # Subclass hook -- Cosmos-Transfer uses this to load the
        # ControlNet residual branch from a sibling revision of the
        # same HF repo.  Predict is a no-op here.
        self._post_load_diffusers(local_path, cache_dir, hf_token)
        return True

    def _try_load_cosmos_package(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        # Try the upstream ``cosmos_transfer2`` / ``cosmos_predict2``
        # packages in turn.  Both expose a ``Pipeline`` class with the
        # same DiT/VAE attribute conventions; only the import path
        # differs.
        pipeline_cls = None
        for module_name, attr_name in (
            ("cosmos_transfer2.inference", "CosmosTransfer2Pipeline"),
            ("cosmos_predict2.inference", "CosmosPredict2Pipeline"),
        ):
            try:
                module = __import__(module_name, fromlist=[attr_name])
            except ImportError:
                continue
            pipeline_cls = getattr(module, attr_name, None)
            if pipeline_cls is not None:
                break

        if pipeline_cls is None:
            logger.debug("No upstream cosmos_* package available.")
            return False

        try:
            pipe = pipeline_cls.from_pretrained(
                self.cfg.hf_repo_id,
                cache_dir=cache_dir,
                token=hf_token,
            )
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "encoder"):
                self.vae_encoder = pipe.vae.encoder.to(self._dtype)

            if hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
                self.vae_decoder = pipe.vae.decoder.to(self._dtype)

            if hasattr(pipe, "dit"):
                self.dit = pipe.dit.to(self._dtype)
            elif hasattr(pipe, "transformer"):
                self.dit = pipe.transformer.to(self._dtype)
            else:
                logger.warning(
                    "Could not locate DiT module on %s pipeline.",
                    pipeline_cls.__name__,
                )
                return False

            self._backbone_loaded = True
            self._backend = "cosmos_transfer2"
            logger.info(
                "Loaded 3-D backbone via %s.", pipeline_cls.__name__,
            )
            return True
        except Exception as exc:
            logger.warning("Upstream cosmos_* pipeline load failed: %s", exc)
            return False

    def _build_standalone_backbone(self) -> None:
        self.dit = _StandaloneDiT3D(self.cfg)
        self._backend = "standalone"

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space volume ``[B, 3, D, H, W]`` to latent grid."""
        if hasattr(self, "_vae_ref") and self._vae_ref:
            vae = self._vae_ref[0]
            ctx = torch.no_grad() if self._freeze_vae_encoder else torch.enable_grad()
            with ctx:
                enc = vae.encode(x)
                if hasattr(enc, "latent_dist"):
                    latent = enc.latent_dist.mode()
                elif hasattr(enc, "sample"):
                    latent = enc.sample
                else:
                    latent = enc
                return latent.to(dtype=x.dtype)

        if self.vae_encoder is not None:
            ctx = torch.no_grad() if self._freeze_vae_encoder else torch.enable_grad()
            with ctx:
                enc = self.vae_encoder(x)
                if hasattr(enc, "latent_dist"):
                    latent = enc.latent_dist.mode()
                elif hasattr(enc, "sample"):
                    latent = enc.sample
                else:
                    latent = enc
                return latent.to(dtype=x.dtype)

        return self._conv_downsample(x)

    def _conv_downsample(self, x: torch.Tensor) -> torch.Tensor:
        return self._fallback_down(x)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Run 3-D DiT backbone and extract multi-layer features.

        Returns ``[B, feature_size, D_lat, H_lat, W_lat]``.
        """
        B, _C, D_lat, H_lat, W_lat = latent.shape

        dit_cfg = getattr(self.dit, "config", None)
        dit_ps = getattr(dit_cfg, "patch_size", None)
        if isinstance(dit_ps, (list, tuple)) and len(dit_ps) == 3:
            p_t, p_h, p_w = dit_ps
        else:
            p_t = p_h = p_w = self.cfg.patch_size

        pad_d = (p_t - D_lat % p_t) % p_t
        pad_h = (p_h - H_lat % p_h) % p_h
        pad_w = (p_w - W_lat % p_w) % p_w
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            latent = F.pad(
                latent, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate",
            )
        D_p = D_lat + pad_d
        H_p = H_lat + pad_h
        W_p = W_lat + pad_w

        d_tok, h_tok, w_tok = D_p // p_t, H_p // p_h, W_p // p_w

        # Strip MONAI MetaTensor wrapping before entering the compiled DiT.
        # MetaTensor.__torch_function__ causes dynamo to crash with '__objclass__'
        # when torch.compile traces through the DiT attention ops.
        if hasattr(latent, "as_tensor"):
            latent = latent.as_tensor()

        timestep = torch.zeros(B, device=latent.device, dtype=latent.dtype)

        if self._backend in ("diffusers", "cosmos_transfer2"):
            features = self._extract_features_hook(
                latent, timestep, d_tok, h_tok, w_tok,
            )
        else:
            with self._dit_forward_without_ckpt_when_eval():
                final, intermediates = self.dit(
                    latent, timestep=timestep,
                    feature_layers=self._feature_layers,
                )
            feat_list = [
                intermediates[i]
                for i in self._feature_layers
                if i in intermediates
            ]
            if not feat_list:
                feat_list = [final]
            feat_list = [f.float() for f in feat_list]
            features = self.feature_projector(
                feat_list, d_tok, h_tok, w_tok,
            )

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            features = features[:, :, :D_lat, :H_lat, :W_lat]

        return features

    def _register_persistent_hooks(self) -> None:
        """Register forward hooks on DiT blocks once (called from __init__)."""
        self._hook_buffer: List[torch.Tensor] = []
        self._hook_handles: List[Any] = []
        self._hook_block_container = None
        self._hooks_active = False

        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                self._hook_block_container = getattr(self.dit, attr)
                break

        if self._hook_block_container is None:
            return

        def _make_hook(_idx: int):
            def hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
                if not self._hooks_active:
                    return
                out = output[0] if isinstance(output, tuple) else output
                # Only detach when there is no trainable path through
                # this block.  Subclasses with trainable side branches
                # (e.g. Cosmos-Transfer's ControlNet, whose residuals
                # are summed into block outputs) override
                # ``_hook_should_detach`` to keep the gradient path
                # back to those branches alive.
                if self._hook_should_detach():
                    out = out.detach()
                if out.dim() == 3:
                    self._hook_buffer.append(out)
                else:
                    self._hook_buffer.append(rearrange(out, "b ... d -> b (...) d"))
            return hook_fn

        for idx in self._feature_layers:
            if idx < len(self._hook_block_container):
                h = self._hook_block_container[idx].register_forward_hook(
                    _make_hook(idx),
                )
                self._hook_handles.append(h)

    def _extract_features_hook(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        d_tok: int,
        h_tok: int,
        w_tok: int,
    ) -> torch.Tensor:
        """Extract intermediate features from diffusers / cosmos DiT."""
        if self._hook_block_container is None:
            logger.warning(
                "Cannot find block container on DiT (%s).  "
                "Returning conv-downsampled latent features.",
                type(self.dit).__name__,
            )
            fallback = rearrange(latent, "b c d h w -> b (d h w) c").float()
            return self.feature_projector(
                [fallback] * len(self._feature_layers),
                d_tok, h_tok, w_tok,
            )

        self._hook_buffer.clear()
        self._hooks_active = True

        # Grad must be enabled whenever *any* trainable branch is in
        # play.  Subclasses with side branches (e.g. Cosmos-Transfer's
        # ControlNet) override ``_any_trainable`` to keep autograd on
        # even when the base DiT is frozen.
        any_trainable = self._any_trainable()

        try:
            ctx = torch.enable_grad() if any_trainable else torch.no_grad()
            with ctx:
                B = latent.shape[0]
                dit_cfg = getattr(self.dit, "config", None)
                text_dim = getattr(dit_cfg, "crossattn_proj_in_channels", 1024)
                null_text = torch.zeros(B, 1, text_dim, device=latent.device, dtype=latent.dtype)

                img_dim_in = getattr(dit_cfg, "img_context_dim_in", None)
                img_tokens = getattr(dit_cfg, "img_context_num_tokens", 256)
                if img_dim_in:
                    null_img = torch.zeros(B, img_tokens, img_dim_in, device=latent.device, dtype=latent.dtype)
                    enc_hidden = (null_text, null_img)
                else:
                    enc_hidden = null_text

                padding_mask = torch.ones(1, 1, latent.shape[-2], latent.shape[-1], device=latent.device, dtype=latent.dtype)
                null_condition = torch.zeros(B, 1, *latent.shape[2:], device=latent.device, dtype=latent.dtype)

                # Subclass hook -- Cosmos-Transfer runs its ControlNet
                # over the *same* EM latent first so its residuals can
                # be summed into the base DiT every
                # ``controlnet_block_every_n`` blocks (see
                # ``CosmosTransformer3DModel.forward``).  Predict has
                # no ControlNet, so this returns ``None`` and the DiT
                # runs alone.
                block_controlnet_hidden_states = self._compute_controlnet_residuals(
                    latent, timestep, enc_hidden, padding_mask, null_condition,
                )

                with self._dit_forward_without_ckpt_when_eval():
                    self.dit(
                        hidden_states=latent,
                        timestep=timestep,
                        encoder_hidden_states=enc_hidden,
                        block_controlnet_hidden_states=block_controlnet_hidden_states,
                        condition_mask=null_condition,
                        padding_mask=padding_mask,
                    )
        finally:
            self._hooks_active = False

        collected = list(self._hook_buffer)
        self._hook_buffer.clear()

        expected = len(self._feature_layers)
        if len(collected) < expected:
            fallback = rearrange(latent, "b c d h w -> b (d h w) c")
            while len(collected) < expected:
                collected.append(fallback)

        collected = [f.float() for f in collected]
        return self.feature_projector(collected, d_tok, h_tok, w_tok)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode -> DiT features -> unified 30-ch head.

        Args:
            x: Input volume ``[B, C, D, H, W]``.

        Returns:
            Unified head tensor ``[B, 30, D, H, W]``.
        """
        features, target_size = self._encode_and_extract(x)
        return self.decoder_adapter(features, target_size=target_size)

    @torch.no_grad()
    def wan_decoder_output(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """RGB reconstruction from the **original (pretrained) Wan decoder**.

        Mirrors :meth:`forward` through encode + DiT + ``decoder_body``,
        then applies the pretrained ``conv_out`` preserved at
        construction in
        :attr:`decoder_adapter.original_conv_out` instead of the
        unified task head.

        Diagnostic only (``true/wan_decoder`` TensorBoard panel).  The
        pretrained ``conv_out`` is frozen and never optimised; it shows
        what the Wan VAE believes the model's learned latent should
        decode to in pixel space.

        Args:
            x: Input volume ``[B, C, D, H, W]`` (single-channel EM is
                tiled to 3-channel RGB internally).

        Returns:
            ``[B, 3, D, H, W]`` RGB reconstruction in roughly
            ``[-1, 1]``, or ``None`` if the wrapper was built without a
            pretrained VAE (random-init standalone DiT path).
        """
        if not getattr(self.decoder_adapter, "_has_pretrained", False):
            return None
        if getattr(self.decoder_adapter, "original_conv_out", None) is None:
            return None
        features, target_size = self._encode_and_extract(x)
        return self.decoder_adapter.wan_reconstruct(features, target_size=target_size)

    def _encode_and_extract(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Shared head of :meth:`forward` and :meth:`wan_decoder_output`.

        Adapts to RGB, pads to compression multiples, encodes to the
        VAE latent, runs the DiT to extract per-layer features, and
        casts back to the input dtype.  Returns the projected feature
        map plus the original ``(D, H, W)`` so the decoder side can
        crop / interpolate back.
        """
        original_dtype = x.dtype
        D_in, H_in, W_in = x.shape[-3], x.shape[-2], x.shape[-1]

        rgb = _adapt_to_rgb(x)

        s = self.cfg.spatial_compression
        t = self.cfg.temporal_compression
        pad_d = (t - D_in % t) % t
        pad_h = (s - H_in % s) % s
        pad_w = (s - W_in % s) % s
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")

        compute_dtype = self._dtype if self._backbone_loaded else original_dtype
        latent = self._encode_to_latent(rgb.to(dtype=compute_dtype))

        features = self._extract_features(latent)
        features = features.to(dtype=original_dtype)
        return features, (D_in, H_in, W_in)

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_dit_backbone(self) -> None:
        self.dit.requires_grad_(False)
        self._freeze_dit_backbone = True
        logger.info("DiT backbone frozen (%s trainable params).",
                     f"{self.get_num_parameters(True):,}")

    def unfreeze_dit_backbone(self) -> None:
        self.dit.requires_grad_(True)
        self._freeze_dit_backbone = False
        logger.info("DiT backbone unfrozen (%s trainable params).",
                     f"{self.get_num_parameters(True):,}")

    def freeze_vae_encoder(self) -> None:
        if self.vae_encoder is not None:
            self.vae_encoder.requires_grad_(False)
            self.vae_encoder.eval()
            self._freeze_vae_encoder = True
            logger.info("VAE encoder frozen.")

    def unfreeze_vae_encoder(self) -> None:
        if self.vae_encoder is not None:
            self.vae_encoder.requires_grad_(True)
            self.vae_encoder.train()
            self._freeze_vae_encoder = False
            logger.info("VAE encoder unfrozen.")

    def freeze_vae_decoder(self) -> None:
        self.decoder_adapter._freeze_body()
        self._freeze_vae_decoder = True
        logger.info("VAE decoder frozen.")

    def unfreeze_vae_decoder(self) -> None:
        self.decoder_adapter._unfreeze_body()
        self._freeze_vae_decoder = False
        logger.info("VAE decoder unfrozen.")

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    @contextmanager
    def _dit_forward_without_ckpt_when_eval(self):
        """Turn off DiT checkpointing during eval when it was enabled for training.

        PyTorch Lightning runs ``validation_step`` under ``torch.inference_mode()``.
        ``torch.utils.checkpoint`` cannot wrap inference tensors, so diffusers
        DiT forward fails with gradient checkpointing left on. Training is
        unaffected (``self.training`` is True).
        """
        if not self.training and self._gradient_checkpointing:
            self.disable_gradient_checkpointing(_log=False)
            try:
                yield
            finally:
                self.enable_gradient_checkpointing(_log=False)
        else:
            yield

    def enable_gradient_checkpointing(self, _log: bool = True) -> None:
        """Enable activation checkpointing on DiT transformer blocks.

        Trades ~20-30% slower forward for ~40% lower activation memory,
        allowing larger batch sizes or patch sizes.
        """
        if hasattr(self.dit, "enable_gradient_checkpointing"):
            self.dit.enable_gradient_checkpointing()
            self._gradient_checkpointing = True
            if _log:
                logger.info("Gradient checkpointing enabled (diffusers API).")
            return

        block_container = None
        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                block_container = getattr(self.dit, attr)
                break

        if block_container is None:
            logger.warning(
                "Cannot find transformer block container on %s -- "
                "gradient checkpointing not applied.",
                type(self.dit).__name__,
            )
            return

        for block in block_container:
            original_forward = block.forward

            def _make_ckpt_forward(fwd):
                def ckpt_forward(*args, **kwargs):
                    if not torch.is_grad_enabled():
                        return fwd(*args, **kwargs)
                    return torch.utils.checkpoint.checkpoint(
                        fwd, *args, use_reentrant=False, **kwargs,
                    )
                return ckpt_forward

            block.forward = _make_ckpt_forward(original_forward)
            block._original_forward = original_forward

        self._gradient_checkpointing = True
        if _log:
            logger.info(
                "Gradient checkpointing enabled (manual, %d blocks).",
                len(block_container),
            )

    def disable_gradient_checkpointing(self, _log: bool = True) -> None:
        """Disable activation checkpointing, restoring original block forwards."""
        if hasattr(self.dit, "disable_gradient_checkpointing"):
            self.dit.disable_gradient_checkpointing()
            self._gradient_checkpointing = False
            if _log:
                logger.info("Gradient checkpointing disabled (diffusers API).")
            return

        block_container = None
        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                block_container = getattr(self.dit, attr)
                break

        if block_container is not None:
            for block in block_container:
                if hasattr(block, "_original_forward"):
                    block.forward = block._original_forward
                    del block._original_forward

        self._gradient_checkpointing = False
        if _log:
            logger.info("Gradient checkpointing disabled.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_output_channels(self) -> int:
        return self.head_channels


__all__ = ["_BaseCosmos25Wrapper"]
