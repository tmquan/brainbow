"""
Tests for :func:`sliding_window_inference`.

These tests exercise the head-detection / aggregation logic that was
generalised in Phase 3e.  They run on CPU with tiny tensors so the full
suite stays fast.
"""

from typing import Dict

import pytest
import torch

from brainbow.inference.sliding_window import (
    _detect_heads,
    _extract_head,
    create_gaussian_weight,
    sliding_window_inference,
)


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------

class _LegacySemanticModel(torch.nn.Module):
    """Returns a single tensor (legacy mode)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        return torch.ones(b, self.num_classes, d, h, w, device=x.device) * 0.5


class _MultiHeadModel(torch.nn.Module):
    """Returns a dict with the four standard heads."""

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, _, d, h, w = x.shape
        zero = lambda c: torch.zeros(b, c, d, h, w, device=x.device)
        return {
            "semantic": zero(2).fill_(0.7),
            "instance": zero(8),
            "geometry": zero(10),
            "boundary": zero(16),
        }


# ---------------------------------------------------------------------------
# create_gaussian_weight
# ---------------------------------------------------------------------------

class TestCreateGaussianWeight:

    def test_shape_matches_patch(self) -> None:
        w = create_gaussian_weight((4, 4, 4))
        assert w.shape == (4, 4, 4)

    def test_peak_normalised(self) -> None:
        w = create_gaussian_weight((4, 4, 4))
        assert pytest.approx(w.max().item(), abs=1e-6) == 1.0

    def test_radial_decay(self) -> None:
        w = create_gaussian_weight((8, 8, 8))
        # The corner is farther from the centre than the centre itself.
        assert w[0, 0, 0] < w[4, 4, 4]


# ---------------------------------------------------------------------------
# _detect_heads / _extract_head
# ---------------------------------------------------------------------------

class TestDetectHeads:

    def test_legacy_tensor_returns_semantic(self) -> None:
        assert _detect_heads(torch.zeros(1, 2, 4, 4, 4)) == ["semantic"]

    def test_dict_with_known_heads(self) -> None:
        out = {"semantic": None, "instance": None, "boundary": None}
        # Known-heads ordering preserved (semantic, instance, geometry, boundary).
        assert _detect_heads(out) == ["semantic", "instance", "boundary"]

    def test_logits_alias_maps_to_semantic(self) -> None:
        assert _detect_heads({"logits": None}) == ["semantic"]

    def test_unknown_keys_only_raises(self) -> None:
        with pytest.raises(ValueError, match="known heads"):
            _detect_heads({"foo": None})


class TestExtractHead:

    def test_softmax_applied_for_legacy(self) -> None:
        x = torch.randn(1, 3, 2, 2, 2)
        out = _extract_head("semantic", x, is_dict=False)
        # Softmax over channel dim => per-pixel sum is 1.
        assert torch.allclose(out.sum(dim=1), torch.ones(1, 2, 2, 2), atol=1e-5)

    def test_dict_passthrough(self) -> None:
        x = torch.randn(1, 5, 2, 2, 2)
        out = _extract_head("instance", {"instance": x}, is_dict=True)
        assert torch.equal(out, x)

    def test_dict_logits_softmax(self) -> None:
        x = torch.randn(1, 4, 2, 2, 2)
        out = _extract_head("semantic", {"logits": x}, is_dict=True)
        assert torch.allclose(out.sum(dim=1), torch.ones(1, 2, 2, 2), atol=1e-5)

    def test_missing_head_raises(self) -> None:
        with pytest.raises(KeyError, match="missing from model output"):
            _extract_head("semantic", {"foo": torch.zeros(1)}, is_dict=True)


# ---------------------------------------------------------------------------
# sliding_window_inference (end-to-end on tiny tensors)
# ---------------------------------------------------------------------------

class TestSlidingWindowInference:

    def test_legacy_tensor_output(self) -> None:
        model = _LegacySemanticModel(num_classes=2)
        vol = torch.randn(1, 8, 8, 8)
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(2, 2, 2),
            device=torch.device("cpu"), progress=False,
        )
        # Legacy mode returns a single tensor.
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, 8, 8)

    def test_multi_head_output_dict(self) -> None:
        model = _MultiHeadModel()
        vol = torch.randn(1, 8, 8, 8)
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(2, 2, 2),
            device=torch.device("cpu"), progress=False,
        )
        assert isinstance(out, dict)
        for head, ch in [("semantic", 2), ("instance", 8), ("geometry", 10), ("boundary", 16)]:
            assert head in out, f"head '{head}' missing from output"
            assert out[head].shape == (ch, 8, 8, 8)

    def test_back_compat_aliases(self) -> None:
        model = _MultiHeadModel()
        vol = torch.randn(1, 8, 8, 8)
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(2, 2, 2),
            device=torch.device("cpu"), progress=False,
        )
        assert torch.equal(out["semantic_probs"], out["semantic"])
        assert torch.equal(out["instance_embeddings"], out["instance"])

    def test_average_aggregation(self) -> None:
        model = _LegacySemanticModel(num_classes=2)
        vol = torch.randn(1, 8, 8, 8)
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(2, 2, 2),
            aggregation="average",
            device=torch.device("cpu"), progress=False,
        )
        # Constant 0.5 input => softmax => 0.5 / class everywhere.
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5)

    def test_3d_volume_without_channel(self) -> None:
        model = _LegacySemanticModel(num_classes=3)
        vol = torch.randn(8, 8, 8)  # no channel dim
        out = sliding_window_inference(
            model, vol,
            patch_size=(4, 4, 4), stride=(4, 4, 4),
            device=torch.device("cpu"), progress=False,
        )
        assert out.shape == (3, 8, 8, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
