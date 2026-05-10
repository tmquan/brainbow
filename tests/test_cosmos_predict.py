"""Smoke tests for the Cosmos-Predict 2.5 backbone.

These tests run with ``pretrained=False`` (random-init standalone DiT)
so they execute on CPU in a few seconds without touching HuggingFace.
The goal is to lock in the public contract -- forward returns
``[B, HEAD_CHANNELS, D, H, W]`` finite logits -- not to validate
numerical quality, which would require the real pretrained weights
and a non-trivial training loop.
"""

import pytest
import torch

from brainbow.losses import HEAD_CHANNELS
from brainbow.models import CosmosPredict3DWrapper
from brainbow.modules import CosmosPredict3DModule


@pytest.fixture(scope="module")
def small_predict_wrapper() -> CosmosPredict3DWrapper:
    """A tiny standalone (no HF download) Predict wrapper for fast smoke tests."""
    torch.manual_seed(0)
    return CosmosPredict3DWrapper(
        in_channels=1,
        head_channels=HEAD_CHANNELS,
        feature_size=16,
        variant="2B",
        pretrained=False,
        dtype="fp32",
    )


def test_wrapper_output_shape_and_finite(small_predict_wrapper):
    """Forward returns [B, 30, D, H, W] with finite values."""
    model = small_predict_wrapper
    x = torch.randn(1, 1, 16, 32, 32)
    out = model(x)
    assert out.shape == (1, HEAD_CHANNELS, 16, 32, 32)
    assert torch.isfinite(out).all().item()


def test_wrapper_has_no_controlnet(small_predict_wrapper):
    """Predict has no ControlNet branch (that's the Transfer-only delta)."""
    model = small_predict_wrapper
    assert not hasattr(model, "controlnet"), (
        "Cosmos-Predict 2.5 must not carry a ControlNet branch."
    )


def test_wrapper_unknown_variant_raises():
    """Unknown variant strings raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown variant"):
        CosmosPredict3DWrapper(variant="42B", pretrained=False)


def test_module_builds_with_inherited_transfer_keys():
    """Predict module silently filters out Transfer-only model_config keys.

    When the Hydra recipe inherits from ``snemi3d.yaml`` the model
    config carries ``controlnet_revision`` and ``freeze_controlnet``
    even on a Predict run -- ``BaseCosmosModule._build_model`` must
    forward only the kwargs the Predict wrapper accepts, via the
    :meth:`_extra_model_kwargs` hook.
    """
    module = CosmosPredict3DModule(
        model_config={
            "in_channels": 1,
            "head_channels": HEAD_CHANNELS,
            "feature_size": 16,
            "variant": "2B",
            "pretrained": False,
            "dtype": "fp32",
            # Transfer-only keys that Predict must ignore:
            "controlnet_revision": "diffusers/controlnet/general/edge",
            "freeze_controlnet": True,
        },
        optimizer_config={"lr": 1e-4, "weight_decay": 1e-5},
        loss_config={},
        training_config={},
    )
    assert isinstance(module.model, CosmosPredict3DWrapper)


def test_module_optimizer_groups_have_no_controlnet_group():
    """No ControlNet -> no controlnet param group in the optimiser."""
    module = CosmosPredict3DModule(
        model_config={
            "in_channels": 1,
            "head_channels": HEAD_CHANNELS,
            "feature_size": 16,
            "variant": "2B",
            "pretrained": False,
            "dtype": "fp32",
        },
        optimizer_config={"lr": 1e-4, "weight_decay": 1e-5},
        loss_config={},
        training_config={},
    )
    opt = module.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    # No model.controlnet.* parameters were collected, so the two
    # controlnet groups in BaseCosmosModule.configure_optimizers should
    # have been dropped by the empty-group filter.
    for group in opt.param_groups:
        for p in group["params"]:
            for name, candidate in module.named_parameters():
                if candidate is p:
                    assert not name.startswith("model.controlnet."), (
                        "Cosmos-Predict optimiser unexpectedly has "
                        f"controlnet parameter: {name}"
                    )
                    break
