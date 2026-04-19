"""
Tests for BrainbowLoss and its 16-channel target construction.

The target map encodes, per voxel:
  - channel  0     : raw image intensity at that voxel (dense, everywhere)
  - channels 1-3   : normalised min (bbox-min (z, y, x)) of the instance
  - channels 4-6   : normalised avg (centroid) of the instance
  - channels 7-9   : normalised max (bbox-max (z, y, x)) of the instance
  - channels 10-15 : binary face-affinity to 6 neighbours (U,D,L,R,T,B)
                    with SAME / replicate padding at the volume boundary.
"""

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
        assert target.shape == (1, 16, 8, 16, 16)
        assert target.dtype == torch.float32

    def test_raw_equals_image(self) -> None:
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
        bg_mask = labels == 0
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
            expected = idx.mean(0) / torch.tensor([D, H, W])
            fg = labels[0] == inst_id
            got = target[0, 4:7][:, fg][:, 0]
            assert torch.allclose(got, expected, atol=1e-6)

    def test_empty_labels_only_raw_and_aff_one(self) -> None:
        """No instances -> loc channels all zero, aff channels all one (bg==bg)."""
        B, D, H, W = 1, 4, 4, 4
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        image = torch.rand(B, D, H, W)
        target = build_brainbow_target(labels, image)
        assert torch.all(target[:, 1:10] == 0)
        assert torch.allclose(target[:, 0], image.float())
        # Every voxel sees the same background label as every neighbour.
        assert torch.all(target[:, 10:16] == 1.0)

    def test_affinity_shape_and_range(self) -> None:
        labels, image = self._simple_labels()
        target = build_brainbow_target(labels, image)
        aff = target[:, 10:16]
        assert aff.shape == (1, 6, 8, 16, 16)
        assert torch.all((aff == 0.0) | (aff == 1.0))

    def test_affinity_boundary_is_one(self) -> None:
        """SAME / replicate padding -> boundary voxels are self-connected."""
        labels, image = self._simple_labels()
        target = build_brainbow_target(labels, image)
        aff = target[0, 10:16]          # [6, D, H, W]

        # Direction layout: (U, D, L, R, T, B) with
        # U <-> (axis=H, shift=+1), D <-> (axis=H, shift=-1),
        # L <-> (axis=W, shift=+1), R <-> (axis=W, shift=-1),
        # T <-> (axis=D, shift=+1), B <-> (axis=D, shift=-1).
        assert torch.all(aff[0, :, 0, :] == 1.0)          # U: top row of H
        assert torch.all(aff[1, :, -1, :] == 1.0)         # D: bottom row of H
        assert torch.all(aff[2, :, :, 0] == 1.0)          # L: left col of W
        assert torch.all(aff[3, :, :, -1] == 1.0)         # R: right col of W
        assert torch.all(aff[4, 0, :, :] == 1.0)          # T: first slice of D
        assert torch.all(aff[5, -1, :, :] == 1.0)         # B: last slice of D

    def test_affinity_interior_matches_label_eq(self) -> None:
        """Interior aff[dir] == (labels == shift(labels, dir))."""
        labels, image = self._simple_labels()
        target = build_brainbow_target(labels, image)
        aff = target[0, 10:16]
        lbl = labels[0]

        # U: aff[0, z, y, x] == (lbl[z, y, x] == lbl[z, y-1, x])
        assert torch.all(
            aff[0, :, 1:, :] == (lbl[:, 1:, :] == lbl[:, :-1, :]).float()
        )
        # D: aff[1, z, y, x] == (lbl[z, y, x] == lbl[z, y+1, x])
        assert torch.all(
            aff[1, :, :-1, :] == (lbl[:, :-1, :] == lbl[:, 1:, :]).float()
        )
        # L / R mirror on the W axis.
        assert torch.all(
            aff[2, :, :, 1:] == (lbl[:, :, 1:] == lbl[:, :, :-1]).float()
        )
        assert torch.all(
            aff[3, :, :, :-1] == (lbl[:, :, :-1] == lbl[:, :, 1:]).float()
        )
        # T / B on the D axis.
        assert torch.all(
            aff[4, 1:, :, :] == (lbl[1:, :, :] == lbl[:-1, :, :]).float()
        )
        assert torch.all(
            aff[5, :-1, :, :] == (lbl[:-1, :, :] == lbl[1:, :, :]).float()
        )

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
        pred = torch.randn(B, 16, D, H, W, requires_grad=True)
        return pred, labels, image

    def test_num_channels_constant(self) -> None:
        assert BrainbowLoss.num_channels == 16

    def test_forward_returns_required_keys(self, batch) -> None:
        pred, labels, image = batch
        out = BrainbowLoss()(pred, labels, image)
        for k in ("loss", "min", "avg", "max", "raw", "aff"):
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

    def test_zero_instances_only_raw_and_aff(self, batch) -> None:
        """No instances -> aff target is all ones; loc sub-losses are zero."""
        pred, _, image = batch
        B, D, H, W = image.shape
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        out = BrainbowLoss()(pred, labels, image)
        assert out["min"].item() == 0.0
        assert out["avg"].item() == 0.0
        assert out["max"].item() == 0.0
        assert out["raw"].item() > 0.0
        # aff target is uniformly 1 -> soft-Dice loss should be finite.
        assert torch.isfinite(out["aff"])

    def test_weights_applied_to_total(self, batch) -> None:
        pred, labels, image = batch
        uniform = BrainbowLoss(
            weight_min=1.0, weight_avg=1.0, weight_max=1.0,
            weight_raw=1.0, weight_aff=1.0,
        )(pred, labels, image)
        only_raw = BrainbowLoss(
            weight_min=0.0, weight_avg=0.0, weight_max=0.0,
            weight_raw=1.0, weight_aff=0.0,
        )(pred, labels, image)
        # "only_raw" has exactly loss == raw; uniform has the extra
        # localisation + affinity terms, so it must be at least as large.
        assert only_raw["loss"].item() == pytest.approx(
            only_raw["raw"].item(), rel=1e-6
        )
        assert uniform["loss"].item() >= only_raw["loss"].item() - 1e-6

    def test_aff_loss_is_soft_dice_on_sigmoid(self, batch) -> None:
        """Perfect affinity logits (large +) give Dice ~ 1 -> loss ~ 0."""
        pred, labels, image = batch
        loss_fn = BrainbowLoss(weight_aff=1.0)
        target = loss_fn.build_target(labels, image)

        # Replace the affinity channels of `pred` with a near-perfect
        # prediction (large positive logit where target is 1, large
        # negative where target is 0).  Everything else random.
        ideal = pred.detach().clone()
        ideal[:, 10:16] = (target[:, 10:16] - 0.5) * 20.0    # +-10 logits
        out = loss_fn(ideal, labels, image)
        assert out["aff"].item() < 0.05

    def test_wrong_prediction_channels_raises(self, batch) -> None:
        _, labels, image = batch
        bad = torch.randn(*labels.shape[:1], 10, *labels.shape[1:])
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
        preds = {"brainbow": torch.randn(B, 16, D, H, W, requires_grad=True)}
        targets = {
            "labels": labels,
            "semantic_labels": (labels > 0).long(),
            "raw_image": image,
        }
        out = loss(preds, targets)
        for k in (
            "loss",
            "brainbow/loss",
            "brainbow/loss/min",
            "brainbow/loss/raw",
            "brainbow/loss/aff",
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
        preds = {"brainbow": torch.randn(B, 16, D, H, W)}
        with pytest.raises(KeyError):
            loss(preds, {"labels": labels, "semantic_labels": (labels > 0).long()})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
