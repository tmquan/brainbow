"""
Tests for the combined segmentation loss used by every Lightning module.
"""

import pytest
import torch

from brainbow.losses import CombinedLoss


# ---------------------------------------------------------------------------
# 2-D CombinedLoss
# ---------------------------------------------------------------------------

class TestCombinedLoss2D:
    """Sanity checks for ``CombinedLoss(spatial_dims=2)``."""

    @pytest.fixture()
    def loss_fn(self) -> CombinedLoss:
        return CombinedLoss(
            spatial_dims=2,
            weight_pull=1.0,
            weight_push=1.0,
            weight_norm=0.001,
            weight_edge=1.0,
            weight_bone=1.0,
            delta_v=0.5,
            delta_d=1.5,
        )

    @pytest.fixture()
    def sample_inputs(self):
        B, H, W = 2, 16, 16
        predictions = {
            "semantic": torch.randn(B, 16, H, W),
            "instance": torch.randn(B, 16, H, W),
        }
        labels = torch.zeros(B, H, W, dtype=torch.long)
        labels[:, :8, :8] = 1
        labels[:, :8, 8:] = 2
        targets = {
            "semantic_labels": (labels > 0).long(),
            "labels": labels,
        }
        return predictions, targets

    def test_forward_returns_required_keys(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        for key in ("loss", "semantic/loss", "semantic/loss/ce", "instance/loss"):
            assert key in result

    def test_total_loss_is_finite(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()

    def test_sub_losses_non_negative(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["semantic/loss"].item() >= 0.0

    def test_iou_loss_activates(self, sample_inputs) -> None:
        loss_fn = CombinedLoss(spatial_dims=2, weight_iou=1.0, weight_edge=1.0, weight_bone=1.0)
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["semantic/loss/iou"].item() > 0.0

    def test_backward_pass(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        for v in predictions.values():
            v.requires_grad_(True)
        result = loss_fn(predictions, targets)
        result["loss"].backward()
        for v in predictions.values():
            assert v.grad is not None

    def test_zero_instances_no_error(self, loss_fn) -> None:
        B, H, W = 1, 8, 8
        predictions = {
            "semantic": torch.randn(B, 16, H, W),
            "instance": torch.randn(B, 16, H, W),
        }
        targets = {
            "semantic_labels": torch.zeros(B, H, W, dtype=torch.long),
            "labels": torch.zeros(B, H, W, dtype=torch.long),
        }
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()

    def test_custom_hyperparameters(self) -> None:
        loss_fn = CombinedLoss(
            spatial_dims=2,
            weight_pull=2.0, weight_push=3.0, weight_norm=0.01,
            delta_v=0.3, delta_d=2.0,
        )
        assert loss_fn.instance_loss.weight_pull == 2.0
        assert loss_fn.instance_loss.weight_push == 3.0
        assert loss_fn.instance_loss.delta_v == 0.3
        assert loss_fn.instance_loss.delta_d == 2.0


# ---------------------------------------------------------------------------
# 3-D CombinedLoss
# ---------------------------------------------------------------------------

class TestCombinedLoss3D:
    """Sanity checks for ``CombinedLoss(spatial_dims=3)``."""

    @pytest.fixture()
    def loss_fn(self) -> CombinedLoss:
        return CombinedLoss(
            spatial_dims=3,
            weight_pull=1.0,
            weight_push=1.0,
            weight_norm=0.001,
            weight_edge=1.0,
            weight_bone=1.0,
            delta_v=0.5,
            delta_d=1.5,
        )

    @pytest.fixture()
    def sample_inputs(self):
        B, D, H, W = 1, 4, 8, 8
        predictions = {
            "semantic": torch.randn(B, 16, D, H, W),
            "instance": torch.randn(B, 16, D, H, W),
        }
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[:, :, :4, :4] = 1
        labels[:, :, :4, 4:] = 2
        targets = {
            "semantic_labels": (labels > 0).long(),
            "labels": labels,
        }
        return predictions, targets

    def test_forward_returns_required_keys(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        for key in ("loss", "semantic/loss", "semantic/loss/ce", "instance/loss"):
            assert key in result

    def test_total_loss_is_finite(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()

    def test_iou_loss_activates(self, sample_inputs) -> None:
        loss_fn = CombinedLoss(spatial_dims=3, weight_iou=1.0, weight_edge=1.0, weight_bone=1.0)
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["semantic/loss/iou"].item() > 0.0

    def test_backward_pass(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        for v in predictions.values():
            v.requires_grad_(True)
        result = loss_fn(predictions, targets)
        result["loss"].backward()
        for v in predictions.values():
            assert v.grad is not None

    def test_semantic_disabled_when_weight_zero(self) -> None:
        loss_fn = CombinedLoss(spatial_dims=3, weight_semantic=0.0)
        assert loss_fn.semantic_loss is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
