"""The public :class:`CosmosTransfer3DWrapper` model.

Ties together:

* the variant registry (:mod:`.variants`)
* the HuggingFace snapshot downloader (:mod:`.hf_loader`)
* the standalone random-init DiT fallback (:mod:`.standalone_dit`)
* the feature projector + VAE-aware decoder adapter (:mod:`.decoder`)

…and exposes them behind a single ``nn.Module`` producing four
co-registered task predictions (``semantic`` / ``instance`` /
``geometry`` / ``boundary``).
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from brainbow.models.cosmos_transfer_2_5.decoder import (
    _DecoderAdapter3D,
    _FeatureProjector3D,
)
from brainbow.models.cosmos_transfer_2_5.hf_loader import _download_from_hf
from brainbow.models.cosmos_transfer_2_5.layers import (
    _NORM,
    _PointwiseLinear,
    _SPATIAL_DIMS,
    _adapt_to_rgb,
)
from brainbow.models.cosmos_transfer_2_5.standalone_dit import _StandaloneDiT3D
from brainbow.models.cosmos_transfer_2_5.variants import (
    _VARIANT_CONFIGS,
    _VariantConfig,
)

logger = logging.getLogger(__name__)


class CosmosTransfer3DWrapper(nn.Module):
    """Cosmos-Transfer2.5 adapted for **volumetric** connectomics segmentation.

    Four parallel output heads produce:

    - ``semantic``  [B, num_classes, D, H, W]
    - ``instance``  [B, instance_channels, D, H, W]
    - ``geometry``  [B, G, D, H, W]  where
      ``G = 1 + S * (S + 1) // 2 + S = 1 (raw) + 6 (cov upper-tri) + 3 (dir) = 10``
      in 3-D.  Layout (must match :class:`brainbow.losses.GeometryLoss`):
      ch 0 = raw, ch 1..6 = covariance upper-triangle, ch 7..9 = direction.
    - ``boundary``  [B, boundary_channels, D, H, W]
      (1 raw + 3 avg RGB + 6 direct face-affinity = 10 by default).
      The boundary loss additionally derives a soft 6-face affinity
      from the predicted avgloc for dual supervision; see
      :class:`brainbow.losses.BoundaryLoss`.

    Because Cosmos-Transfer2.5 is natively a video model, the depth axis
    of the EM volume is mapped to the temporal axis of the backbone,
    making the 3-D adaptation architecturally natural.

    Args:
        in_channels: Number of input channels (1 for EM volumes).
        num_classes: Semantic segmentation classes.
        instance_channels: Per-voxel instance embedding dimensionality.
        feature_size: Internal feature map channel count after projection.
        variant: ``"2B"`` or ``"14B"`` model variant.
        checkpoint_variant: HuggingFace revision string.
        dtype: Weight dtype (``"bf16"``, ``"fp16"``, ``"fp32"``).
        freeze_dit_backbone: Whether to freeze the pretrained DiT backbone.
        feature_layers: DiT block indices to extract features from.
        cache_dir: HuggingFace download cache directory.
        hf_token: HuggingFace authentication token.
        dropout: Dropout probability for heads.

    Example::

        >>> model = CosmosTransfer3DWrapper(
        ...     in_channels=1, num_classes=1, variant="2B",
        ... )
        >>> x = torch.randn(1, 1, 32, 64, 64)
        >>> out = model(x)
        >>> out["semantic"].shape   # [1, 1, 32, 64, 64]
        >>> out["instance"].shape   # [1, 10, 32, 64, 64]
        >>> out["geometry"].shape   # [1, 10, 32, 64, 64]  (raw=1 + cov_tri=6 + dir=3)
        >>> out["boundary"].shape   # [1, 10, 32, 64, 64]  (raw=1 + avg=3 + aff=6)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        instance_channels: int = 10,
        boundary_channels: int = 10,
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
        disabled_heads: Optional[frozenset] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        variant = variant.upper()
        if variant not in _VARIANT_CONFIGS:
            raise ValueError(
                f"Unknown variant '{variant}'.  "
                f"Choose from: {list(_VARIANT_CONFIGS)}"
            )

        self.variant = variant
        self.cfg: _VariantConfig = _VARIANT_CONFIGS[variant]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.instance_channels = instance_channels
        self.boundary_channels = boundary_channels
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS
        self.dropout = dropout

        S = _SPATIAL_DIMS
        # Geometry head layout: raw (1) + cov upper-tri (S*(S+1)/2) + dir (S).
        # Matches BoundaryLoss channel convention with raw at ch 0.
        self.geometry_channels = 1 + S * (S + 1) // 2 + S

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
        self._build_backbone(cache_dir, hf_token, checkpoint_variant)

        self.feature_projector = _FeatureProjector3D(
            hidden_dim=self.cfg.hidden_dim,
            num_feature_layers=len(self._feature_layers),
            out_dim=feature_size,
        ).float()

        if self._backend in ("diffusers", "cosmos_transfer2"):
            self._register_persistent_hooks()

        self._disabled_heads: frozenset = disabled_heads or frozenset()
        self.decoder_adapter = _DecoderAdapter3D(
            vae_decoder=self.vae_decoder,
            latent_channels=self.cfg.latent_channels,
            feature_size=feature_size,
            num_classes=num_classes,
            instance_channels=instance_channels,
            geometry_channels=self.geometry_channels,
            boundary_channels=boundary_channels,
            spatial_compression=self.cfg.spatial_compression,
            temporal_compression=self.cfg.temporal_compression,
            dropout=dropout,
            freeze_vae_decoder=freeze_vae_decoder,
            disabled_heads=self._disabled_heads,
        )
        if self.decoder_adapter.to_latent is not None:
            self.decoder_adapter.to_latent.float()
        for head_name in ("semantic", "instance", "geometry", "boundary"):
            if head_name not in self._disabled_heads:
                getattr(self.decoder_adapter, f"head_{head_name}").float()

        if self.vae_encoder is not None and freeze_vae_encoder:
            self.vae_encoder.requires_grad_(False)
            self.vae_encoder.eval()

        if freeze_dit_backbone:
            self.freeze_dit_backbone()
        else:
            self.dit.train()

        self._make_params_contiguous()

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

        logger.info(
            "CosmosTransfer3DWrapper initialised: variant=%s, "
            "feature_layers=%s, backbone_loaded=%s, frozen=%s, "
            "grad_ckpt=%s, params=%s (trainable=%s)",
            variant, self._feature_layers, self._backbone_loaded,
            freeze_dit_backbone, self._gradient_checkpointing,
            f"{self.get_num_parameters(trainable_only=False):,}",
            f"{self.get_num_parameters(trainable_only=True):,}",
        )

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
                f"Cosmos-Transfer2.5 variant '{self.variant}' has no public "
                f"HuggingFace checkpoint.  Either pass `variant='2B'` (the "
                f"only publicly released v2.5 variant as of 2026-04) or set "
                f"`pretrained=False` to train from scratch."
            )

        _saved_dtype = torch.get_default_dtype()
        try:
            loaded = (
                self._try_load_diffusers(cache_dir, hf_token, checkpoint_variant)
                or self._try_load_cosmos_package(
                    cache_dir, hf_token, checkpoint_variant,
                )
                or self._try_load_raw_checkpoint(
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
                "Loaded 3-D backbone + VAE via diffusers (local snapshot).",
            )
            return True
        except Exception as exc:
            logger.warning("diffusers load from local snapshot failed: %s", exc)
            return False

    def _try_load_cosmos_package(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        try:
            # TODO: Exact import depends on installed cosmos_transfer2 version.
            from cosmos_transfer2.inference import CosmosTransfer2Pipeline  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("cosmos_transfer2 package not available.")
            return False

        try:
            pipe = CosmosTransfer2Pipeline.from_pretrained(
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
                    "Could not locate DiT module on cosmos_transfer2 pipeline."
                )
                return False

            self._backbone_loaded = True
            self._backend = "cosmos_transfer2"
            logger.info("Loaded 3-D backbone via cosmos_transfer2 package.")
            return True
        except Exception as exc:
            logger.warning("cosmos_transfer2 load failed: %s", exc)
            return False

    def _try_load_raw_checkpoint(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        try:
            local_path = _download_from_hf(
                self.cfg.hf_repo_id,
                revision=self.cfg.hf_revision,
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception:
            return False

        local_path = Path(local_path)

        # --- Load VAE from snapshot via diffusers ---
        try:
            from diffusers import AutoencoderKLWan  # type: ignore[import-untyped]

            vae = AutoencoderKLWan.from_pretrained(
                str(local_path), subfolder="vae",
                torch_dtype=self._dtype,
            )
            vae = vae.to(self._dtype)
            self._vae_ref = [vae]
            self.vae_encoder = vae.encoder
            self.vae_decoder = vae.decoder
            logger.info("Loaded VAE encoder + decoder from snapshot.")
        except Exception as exc:
            logger.warning("Could not load VAE from snapshot: %s", exc)

        # --- Load DiT: try diffusers first, then raw weights ---
        try:
            from diffusers import CosmosTransformer3DModel  # type: ignore[import-untyped]

            self.dit = CosmosTransformer3DModel.from_pretrained(
                str(local_path), subfolder="transformer",
                torch_dtype=self._dtype,
            ).to(self._dtype)
            self._backbone_loaded = True
            self._backend = "diffusers"
            logger.info("Loaded DiT transformer from snapshot via diffusers.")
            return True
        except Exception as exc:
            logger.warning("diffusers DiT load from snapshot failed: %s", exc)

        self._build_standalone_backbone()

        transformer_dir = local_path / "transformer"
        ckpt_files = (
            list(transformer_dir.glob("*.safetensors"))
            + list(transformer_dir.glob("*.pt"))
        ) if transformer_dir.is_dir() else (
            list(local_path.glob("**/*.safetensors"))
            + list(local_path.glob("**/*.pt"))
        )
        if not ckpt_files:
            logger.warning("No checkpoint files found in %s.", local_path)
            return False

        loaded_any = False
        for ckpt_file in ckpt_files:
            try:
                if ckpt_file.suffix == ".safetensors":
                    from safetensors.torch import load_file  # type: ignore[import-untyped]
                    state = load_file(str(ckpt_file), device="cpu")
                else:
                    state = torch.load(
                        str(ckpt_file), map_location="cpu", weights_only=True,
                    )
                missing, unexpected = self.dit.load_state_dict(
                    state, strict=False,
                )
                if missing:
                    logger.debug(
                        "Missing keys from %s: %d", ckpt_file.name, len(missing),
                    )
                if unexpected:
                    logger.debug(
                        "Unexpected keys in %s: %d",
                        ckpt_file.name, len(unexpected),
                    )
                loaded_any = True
            except Exception as exc:
                logger.warning("Failed to load %s: %s", ckpt_file.name, exc)

        if loaded_any:
            self._backbone_loaded = True
            self._backend = "raw_checkpoint"
            logger.info(
                "Loaded weights from raw checkpoint into standalone 3-D DiT."
            )
        return loaded_any

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
                if self._freeze_dit_backbone:
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

        try:
            ctx = torch.no_grad() if self._freeze_dit_backbone else torch.enable_grad()
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

                with self._dit_forward_without_ckpt_when_eval():
                    self.dit(
                        hidden_states=latent,
                        timestep=timestep,
                        encoder_hidden_states=enc_hidden,
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

    def forward(
        self,
        x: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode -> DiT features -> 3-D heads.

        Args:
            x: Input volume ``[B, C, D, H, W]``.
            semantic_ids: Optional per-voxel class labels ``[B, D, H, W]``.

        Returns:
            Dict with ``"semantic"``, ``"instance"``, ``"geometry"``
            and optionally ``"semantic_ids"``.
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

        out = self.decoder_adapter(features, target_size=(D_in, H_in, W_in))
        if semantic_ids is not None:
            out["semantic_ids"] = semantic_ids
        return out

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
                "Cannot find transformer block container on %s — "
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
        return self.num_classes


__all__ = ["CosmosTransfer3DWrapper"]
