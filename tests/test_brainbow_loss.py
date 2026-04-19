"""
Tests for BrainbowLoss and its 10-channel target construction.

The target map encodes, per voxel:
  - channel  0   : raw image intensity at that voxel (dense, everywhere)
  - channels 1-3 : normalised minloc (z, y, x) of the instance
  - channels 4-6 : normalised avgloc (centroid) of the instance
  - channels 7-9 : normalised maxloc (z, y, x) of the instance
"""

import numpy as np
import pytest
import torch

from brainbow.losses.brainbow import (
    BrainbowLoss,
    build_brainbow_target,
)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


class TestBuildBrainbowTarget:
    """Unit tests for :func:`build_brainbow_target`."""

    def _simple_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = 1, 8, 16, 16
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :4, :8, :8] = 1
        labels[0, 4:, 8:, 8:] = 2
        image = torch.rand(B, D, H, W)
        return labels, image

    def test_target_shape(self) -> None:
        labels, image = self._simple_labels()
        target = build_brainbow_target(labels, image)
        assert target.shape == (1, 10, 8, 16, 16)
        assert target.dtype == torch.float32

    def test_rawval_equals_image(self) -> None:
        labels, image = self._simple_labels()
        target = build_brainbow_target(labels, image)
        assert torch.allclose(target[:, 0], image.float())

    def test_localisation_channels_in_unit_interval(self) -> None:
        labels, image = self._simple_labels()
        target = build_brainbow_target(labels, image)
        loc = target[:, 1:10]
        assert loc.min() >= 0.0
        assert loc.max() <= 1.0

    def test_background_localisation_is_zero(self) -> None:
        labels, image = self._simple_labels()
        target = build_brainbow_target(labels, image)
        bg_mask = labels == 0  # [B, D, H, W]
        loc_bg = target[:, 1:10][bg_mask.unsqueeze(1).expand(-1, 9, -1, -1, -1)]
        assert torch.all(loc_bg == 0.0)

    def test_known_instance_extrema(self) -> None:
        """Instance 1 occupies [0:4, 0:8, 0:8]; min=(0,0,0), max=(3,7,7)."""
        labels, image = self._simple_labels()
        D, H, W = 8, 16, 16
        target = build_brainbow_target(labels, image)

        fg1 = labels[0] == 1
        min_rgb = target[0, 1:4][:, fg1][:, 0]
        max_rgb = target[0, 7:10][:, fg1][:, 0]
        assert torch.allclose(min_rgb, torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(
            max_rgb, torch.tensor([3.0 / D, 7.0 / H, 7.0 / W]), atol=1e-6
        )

    def test_centroid_is_instance_mean(self) -> None:
        labels, image = self._simple_labels()
        D, H, W = 8, 16, 16
        target = build_brainbow_target(labels, image)

        for inst_id in (1, 2):
            idx = torch.nonzero(labels[0] == inst_id, as_tuple=False).float()
            expected = (idx.mean(0) / torch.tensor([D, H, W]))
            fg = labels[0] == inst_id
            got = target[0, 4:7][:, fg][:, 0]
            assert torch.allclose(got, expected, atol=1e-6)

    def test_empty_labels_only_rawval_contributes(self) -> None:
        B, D, H, W = 1, 4, 4, 4
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        image = torch.rand(B, D, H, W)
        target = build_brainbow_target(labels, image)
        assert torch.all(target[:, 1:10] == 0)
        assert torch.allclose(target[:, 0], image.float())

    def test_shape_mismatch_raises(self) -> None:
        labels = torch.zeros(1, 4, 8, 8, dtype=torch.long)
        image = torch.zeros(1, 4, 8, 16)
        with pytest.raises(ValueError):
            build_brainbow_target(labels, image)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cpu_cuda_agree(self) -> None:
        torch.manual_seed(0)
        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :2, :4, :4] = 1
        labels[0, 2:, 4:, 4:] = 2
        image = torch.rand(B, D, H, W)
        t_cpu = build_brainbow_target(labels, image)
        t_gpu = build_brainbow_target(labels.cuda(), image.cuda()).cpu()
        assert torch.allclose(t_cpu, t_gpu, atol=1e-6)


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


class TestBrainbowLoss:
    """Unit tests for :class:`BrainbowLoss`."""

    @pytest.fixture
    def batch(self):
        B, D, H, W = 2, 8, 16, 16
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :4, :8, :8] = 1
        labels[0, 4:, 8:, 8:] = 2
        labels[1, 2:6, 4:12, 4:12] = 3
        image = torch.rand(B, D, H, W)
        pred = torch.randn(B, 10, D, H, W, requires_grad=True)
        return pred, labels, image

    def test_num_channels_constant(self) -> None:
        assert BrainbowLoss.num_channels == 10

    def test_forward_returns_required_keys(self, batch) -> None:
        pred, labels, image = batch
        out = BrainbowLoss()(pred, labels, image)
        for k in ("loss", "minloc", "avgloc", "maxloc", "rawval"):
            assert k in out

    def test_loss_is_finite_and_non_negative(self, batch) -> None:
        pred, labels, image = batch
        out = BrainbowLoss()(pred, labels, image)
        for k, v in out.items():
            assert torch.isfinite(v), k
            assert v.item() >= 0.0, k

    def test_backward_flows(self, batch) -> None:
        pred, labels, image = batch
        out = BrainbowLoss()(pred, labels, image)
        out["loss"].backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_zero_instances_only_rawval(self, batch) -> None:
        pred, _, image = batch
        B, D, H, W = image.shape
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        out = BrainbowLoss()(pred, labels, image)
        assert out["minloc"].item() == 0.0
        assert out["avgloc"].item() == 0.0
        assert out["maxloc"].item() == 0.0
        assert out["rawval"].item() > 0.0

    def test_weights_applied_to_total(self, batch) -> None:
        pred, labels, image = batch
        uniform = BrainbowLoss(
            weight_minloc=1.0, weight_avgloc=1.0,
            weight_maxloc=1.0, weight_rawval=1.0,
        )(pred, labels, image)
        only_raw = BrainbowLoss(
            weight_minloc=0.0, weight_avgloc=0.0,
            weight_maxloc=0.0, weight_rawval=1.0,
        )(pred, labels, image)
        # "only_raw" has exactly loss == rawval; uniform has the extra
        # localisation terms, so it must be at least as large.
        assert only_raw["loss"].item() == pytest.approx(
            only_raw["rawval"].item(), rel=1e-6
        )
        assert uniform["loss"].item() >= only_raw["loss"].item() - 1e-6

    def test_wrong_prediction_channels_raises(self, batch) -> None:
        _, labels, image = batch
        bad = torch.randn(*labels.shape[:1], 9, *labels.shape[1:])
        with pytest.raises(ValueError):
            BrainbowLoss()(bad, labels, image)

    def test_cached_target_matches_recompute(self, batch) -> None:
        pred, labels, image = batch
        loss_fn = BrainbowLoss()
        cached = loss_fn.compute_target(labels, image)
        out_fresh = loss_fn(pred.detach().clone().requires_grad_(), labels, image)
        out_cached = loss_fn(
            pred.detach().clone().requires_grad_(),
            labels, image, cached_target=cached,
        )
        assert torch.isclose(out_fresh["loss"], out_cached["loss"], atol=1e-6)


# ---------------------------------------------------------------------------
# Integration with CombinedLoss
# ---------------------------------------------------------------------------


class TestBrainbowInCombinedLoss:
    """End-to-end sanity via :class:`brainbow.losses.CombinedLoss`."""

    def test_combined_loss_exposes_brainbow_keys(self) -> None:
        from brainbow.losses import CombinedLoss

        loss = CombinedLoss(
            spatial_dims=3,
            weight_semantic=0.0, weight_instance=0.0, weight_geometry=0.0,
            weight_brainbow=1.0,
        )
        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :2, :4, :4] = 1
        labels[0, 2:, 4:, 4:] = 2
        image = torch.rand(B, D, H, W)
        preds = {"brainbow": torch.randn(B, 10, D, H, W, requires_grad=True)}
        targets = {
            "labels": labels,
            "semantic_labels": (labels > 0).long(),
            "raw_image": image,
        }
        out = loss(preds, targets)
        for k in (
            "loss",
            "brainbow/loss",
            "brainbow/loss/minloc",
            "brainbow/loss/rawval",
        ):
            assert k in out
        out["loss"].backward()

    def test_combined_loss_disabled_brainbow_is_zero(self) -> None:
        from brainbow.losses import CombinedLoss

        loss = CombinedLoss(
            spatial_dims=3,
            weight_semantic=0.0, weight_instance=0.0, weight_geometry=0.0,
            weight_brainbow=0.0,
        )
        assert loss.brainbow_loss is None

    def test_combined_loss_raises_without_raw_image(self) -> None:
        from brainbow.losses import CombinedLoss

        loss = CombinedLoss(
            spatial_dims=3,
            weight_semantic=0.0, weight_instance=0.0, weight_geometry=0.0,
            weight_brainbow=1.0,
        )
        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :2, :4, :4] = 1
        preds = {"brainbow": torch.randn(B, 10, D, H, W)}
        with pytest.raises(KeyError):
            loss(preds, {"labels": labels, "semantic_labels": (labels > 0).long()})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
