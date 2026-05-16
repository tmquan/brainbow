"""Tests for unified-head sliding-window inference."""

import pytest
import torch

from brainbow.inference.sliding_window import (
    create_gaussian_weight,
    sliding_window_inference,
)
from brainbow.losses import HEAD_CHANNELS


class _UnifiedHeadModel(torch.nn.Module):
    def __init__(self, channels: int = HEAD_CHANNELS, value: float = 0.5) -> None:
        super().__init__()
        self.channels = channels
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        return torch.ones(
            b, self.channels, d, h, w, device=x.device,
        ) * self.value


class _DictModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return {"semantic": torch.zeros(x.shape[0], 1, *x.shape[2:])}


class TestCreateGaussianWeight:
    def test_shape_matches_patch(self) -> None:
        w = create_gaussian_weight((4, 4, 4))
        assert w.shape == (4, 4, 4)

    def test_peak_normalised(self) -> None:
        w = create_gaussian_weight((4, 4, 4))
        assert pytest.approx(w.max().item(), abs=1e-6) == 1.0

    def test_radial_decay(self) -> None:
        w = create_gaussian_weight((8, 8, 8))
        assert w[0, 0, 0] < w[4, 4, 4]


class TestSlidingWindowInference:
    def test_unified_tensor_output(self) -> None:
        model = _UnifiedHeadModel()
        vol = torch.randn(1, 8, 8, 8)
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(2, 2, 2),
            device=torch.device("cpu"), progress=False,
        )
        assert isinstance(out, torch.Tensor)
        assert out.shape == (HEAD_CHANNELS, 8, 8, 8)

    def test_average_aggregation_preserves_constant(self) -> None:
        model = _UnifiedHeadModel(value=0.25)
        vol = torch.randn(1, 8, 8, 8)
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(2, 2, 2),
            aggregation="average",
            device=torch.device("cpu"), progress=False,
        )
        assert torch.allclose(out, torch.full_like(out, 0.25), atol=1e-5)

    def test_3d_volume_without_channel(self) -> None:
        model = _UnifiedHeadModel()
        vol = torch.randn(8, 8, 8)
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(4, 4, 4),
            device=torch.device("cpu"), progress=False,
        )
        assert out.shape == (HEAD_CHANNELS, 8, 8, 8)

    def test_dict_model_raises(self) -> None:
        with pytest.raises(TypeError, match="unified head"):
            sliding_window_inference(
                _DictModel(), torch.randn(1, 4, 4, 4),
                patch_size=(4, 4, 4), stride=(4, 4, 4),
                device=torch.device("cpu"), progress=False,
            )

