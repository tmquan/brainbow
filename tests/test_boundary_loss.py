"""
Tests for BoundaryLoss and its 10-channel target construction.

The target map encodes, per voxel:
  - channel 0     : raw image intensity at that voxel (dense, everywhere)
  - channels 1-3  : normalised avg (centroid) of the instance
  - channels 4-9  : binary face-affinity to 6 neighbours in Z-Y-X order
                    (T, B, U, D, L, R) with SAME / replicate padding at
                    the volume boundary.

The BoundaryLoss additionally derives a soft 6-face affinity from the
predicted avgloc (ch 1-3) via ``soft_aff_from_avg`` and supervises that
derived signal against the same binary aff target -- the dual-aff
configuration is exercised in :class:`TestBoundaryLoss`.
"""

import pytest
import torch

from brainbow.losses.boundary import (
    BoundaryLoss,
    build_boundary_target,
    soft_aff_from_avg,
)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


class TestBuildBoundaryTarget:
    """Unit tests for :func:`build_boundary_target`."""

    def _simple_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = 1, 8, 16, 16
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :4, :8, :8] = 1
        labels[0, 4:, 8:, 8:] = 2
        image = torch.rand(B, D, H, W)
        return labels, image

    def test_target_shape(self) -> None:
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image)
        assert target.shape == (1, 10, 8, 16, 16)
        assert target.dtype == torch.float32

    def test_raw_equals_image(self) -> None:
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image)
        assert torch.allclose(target[:, 0], image.float())

    def test_avg_channels_in_unit_interval(self) -> None:
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image)
        avg = target[:, 1:4]
        assert avg.min() >= 0.0
        assert avg.max() <= 1.0

    def test_background_avg_is_zero(self) -> None:
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image)
        bg_mask = labels == 0
        avg_bg = target[:, 1:4][bg_mask.unsqueeze(1).expand(-1, 3, -1, -1, -1)]
        assert torch.all(avg_bg == 0.0)

    def test_centroid_is_instance_mean(self) -> None:
        labels, image = self._simple_labels()
        D, H, W = 8, 16, 16
        target = build_boundary_target(labels, image)

        for inst_id in (1, 2):
            idx = torch.nonzero(labels[0] == inst_id, as_tuple=False).float()
            expected = idx.mean(0) / torch.tensor([D, H, W])
            fg = labels[0] == inst_id
            got = target[0, 1:4][:, fg][:, 0]
            assert torch.allclose(got, expected, atol=1e-6)

    def test_empty_labels_aff_is_zero_with_default_background(self) -> None:
        """Default ``background=0`` masks out spurious bg-bg ``1``s.

        With no foreground voxels anywhere, ``aff = 0`` on every
        direction (instead of the legacy ``aff = 1`` from ``0 == 0``).
        """
        B, D, H, W = 1, 4, 4, 4
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        image = torch.rand(B, D, H, W)
        target = build_boundary_target(labels, image)
        assert torch.all(target[:, 1:4] == 0)
        assert torch.allclose(target[:, 0], image.float())
        assert torch.all(target[:, 4:10] == 0.0)

    def test_empty_labels_aff_is_one_with_background_none(self) -> None:
        """Opt-out via ``background=None`` reproduces the pre-fix targets."""
        B, D, H, W = 1, 4, 4, 4
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        image = torch.rand(B, D, H, W)
        target = build_boundary_target(labels, image, background=None)
        assert torch.all(target[:, 4:10] == 1.0)

    def test_affinity_shape_and_range(self) -> None:
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image)
        aff = target[:, 4:10]
        assert aff.shape == (1, 6, 8, 16, 16)
        assert torch.all((aff == 0.0) | (aff == 1.0))

    def test_affinity_boundary_is_one_for_foreground(self) -> None:
        """SAME / replicate padding -> foreground boundary voxels self-connect."""
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image)
        aff = target[0, 4:10]            # [6, D, H, W]
        fg = labels[0] > 0               # [D, H, W]

        # Channel layout is Z-Y-X:
        #   ch 0 T (z-1)  ch 1 B (z+1)
        #   ch 2 U (y-1)  ch 3 D (y+1)
        #   ch 4 L (x-1)  ch 5 R (x+1)
        assert torch.all(aff[0, 0, :, :][fg[0]] == 1.0)
        assert torch.all(aff[0, 0, :, :][~fg[0]] == 0.0)
        assert torch.all(aff[1, -1, :, :][fg[-1]] == 1.0)
        assert torch.all(aff[1, -1, :, :][~fg[-1]] == 0.0)
        assert torch.all(aff[2, :, 0, :][fg[:, 0]] == 1.0)
        assert torch.all(aff[2, :, 0, :][~fg[:, 0]] == 0.0)
        assert torch.all(aff[3, :, -1, :][fg[:, -1]] == 1.0)
        assert torch.all(aff[3, :, -1, :][~fg[:, -1]] == 0.0)
        assert torch.all(aff[4, :, :, 0][fg[:, :, 0]] == 1.0)
        assert torch.all(aff[4, :, :, 0][~fg[:, :, 0]] == 0.0)
        assert torch.all(aff[5, :, :, -1][fg[:, :, -1]] == 1.0)
        assert torch.all(aff[5, :, :, -1][~fg[:, :, -1]] == 0.0)

    def test_affinity_interior_matches_masked_label_eq(self) -> None:
        """Interior aff[dir] == ((labels == shift(labels, dir)) & (labels != 0))."""
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image)
        aff = target[0, 4:10]
        lbl = labels[0]

        c, s = lbl[1:, :, :], lbl[:-1, :, :]
        assert torch.all(aff[0, 1:, :, :] == ((c == s) & (c != 0)).float())
        c, s = lbl[:-1, :, :], lbl[1:, :, :]
        assert torch.all(aff[1, :-1, :, :] == ((c == s) & (c != 0)).float())
        c, s = lbl[:, 1:, :], lbl[:, :-1, :]
        assert torch.all(aff[2, :, 1:, :] == ((c == s) & (c != 0)).float())
        c, s = lbl[:, :-1, :], lbl[:, 1:, :]
        assert torch.all(aff[3, :, :-1, :] == ((c == s) & (c != 0)).float())
        c, s = lbl[:, :, 1:], lbl[:, :, :-1]
        assert torch.all(aff[4, :, :, 1:] == ((c == s) & (c != 0)).float())
        c, s = lbl[:, :, :-1], lbl[:, :, 1:]
        assert torch.all(aff[5, :, :, :-1] == ((c == s) & (c != 0)).float())

    def test_affinity_background_none_legacy(self) -> None:
        """``background=None`` reproduces the pre-fix unmasked target."""
        labels, image = self._simple_labels()
        target = build_boundary_target(labels, image, background=None)
        aff = target[0, 4:10]
        lbl = labels[0]

        assert torch.all(
            aff[0, 1:, :, :] == (lbl[1:, :, :] == lbl[:-1, :, :]).float()
        )
        assert torch.all(
            aff[2, :, 1:, :] == (lbl[:, 1:, :] == lbl[:, :-1, :]).float()
        )
        assert torch.all(
            aff[4, :, :, 1:] == (lbl[:, :, 1:] == lbl[:, :, :-1]).float()
        )
        assert torch.all(aff[0, 0, :, :] == 1.0)

    def test_affinity_background_custom_value(self) -> None:
        """``background=N`` masks where ``labels == N``, not where ``labels == 0``."""
        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :2, :4, :4] = 1
        labels[0, 2:, 4:, 4:] = 42
        image = torch.rand(B, D, H, W)

        target = build_boundary_target(labels, image, background=42)
        aff = target[0, 4:10]

        bg = labels[0] == 42
        for c in range(6):
            assert torch.all(aff[c][bg] == 0.0)

        zeros = labels[0] == 0
        assert torch.any(zeros)
        # ``aff[5, 0, 0, 4]`` corresponds to (z=0, y=0, x=4) vs (z=0, y=0, x=5):
        # both are 0 and neither equals the masked-out ``42``, so aff = 1.
        assert aff[5, 0, 0, 4] == 1.0

    def test_shape_mismatch_raises(self) -> None:
        labels = torch.zeros(1, 4, 8, 8, dtype=torch.long)
        image = torch.zeros(1, 4, 8, 16)
        with pytest.raises(ValueError):
            build_boundary_target(labels, image)

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
        t_cpu = build_boundary_target(labels, image)
        t_gpu = build_boundary_target(labels.cuda(), image.cuda()).cpu()
        assert torch.allclose(t_cpu, t_gpu, atol=1e-6)


# ---------------------------------------------------------------------------
# Soft affinity from avgloc
# ---------------------------------------------------------------------------


class TestSoftAffFromAvg:
    """Unit tests for :func:`soft_aff_from_avg`."""

    def test_shape_and_range(self) -> None:
        avg = torch.rand(2, 3, 4, 8, 8)
        aff = soft_aff_from_avg(avg, tau=1.0)
        assert aff.shape == (2, 6, 4, 8, 8)
        # exp(-tau * L1) is in (0, 1]; under SAME-pad the boundary
        # voxel compares to itself so the boundary slab is exactly 1.
        assert aff.min() > 0.0
        assert aff.max() <= 1.0 + 1e-6

    def test_constant_avg_is_one_everywhere(self) -> None:
        """If predicted avg is constant, every face-affinity is 1."""
        avg = torch.full((1, 3, 3, 5, 5), 0.7)
        aff = soft_aff_from_avg(avg, tau=2.0)
        assert torch.allclose(aff, torch.ones_like(aff))

    def test_known_step_l1_distance(self) -> None:
        """Two slabs with constant but different avg -> exp(-tau * L1) at boundary."""
        avg = torch.zeros(1, 3, 4, 1, 1)
        avg[0, :, 2:] = 1.0  # step at z=2 across all 3 channels: L1 = 3
        tau = 0.5
        aff = soft_aff_from_avg(avg, tau=tau)
        expected = torch.tensor([[[2.71828 ** (-tau * 3.0)]]])
        # T (z-1): boundary appears at z=2 (compares to z=1 where avg=0).
        assert torch.allclose(aff[0, 0, 2, 0, 0:1], expected, atol=1e-4)
        # B (z+1): boundary appears at z=1 (compares to z=2 where avg=1).
        assert torch.allclose(aff[0, 1, 1, 0, 0:1], expected, atol=1e-4)
        # Interior of each slab is 1.
        assert torch.allclose(aff[0, 0, 0, 0, 0], torch.tensor(1.0))
        assert torch.allclose(aff[0, 0, 3, 0, 0], torch.tensor(1.0))

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            soft_aff_from_avg(torch.rand(2, 4, 4, 8, 8))   # 4 channels != 3
        with pytest.raises(ValueError):
            soft_aff_from_avg(torch.rand(2, 3, 4, 8))      # 4-D not 5-D


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


class TestBoundaryLoss:
    """Unit tests for :class:`BoundaryLoss`."""

    @pytest.fixture
    def batch(self):
        B, D, H, W = 2, 8, 16, 16
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :4, :8, :8] = 1
        labels[0, 4:, 8:, 8:] = 2
        labels[1, 2:6, 4:12, 4:12] = 3
        image = torch.rand(B, D, H, W)
        # ``BoundaryLoss`` consumes post-sigmoid probabilities on every
        # boundary channel (raw + avg + direct aff); the model wrapper
        # applies sigmoid before the loss sees the prediction.
        raw = torch.rand(B, 1, D, H, W)
        avg = torch.rand(B, 3, D, H, W)
        aff = torch.rand(B, 6, D, H, W)
        pred = torch.cat([raw, avg, aff], dim=1).detach().requires_grad_(True)
        return pred, labels, image

    def test_num_channels_constant(self) -> None:
        assert BoundaryLoss.num_channels == 10

    def test_forward_returns_required_keys(self, batch) -> None:
        pred, labels, image = batch
        out = BoundaryLoss()(pred, labels, image)
        for k in (
            "loss", "raw", "avg", "aff",
            "aff_pred", "aff_avg",
            "aff_pred_ce", "aff_pred_dice", "aff_pred_iou",
            "aff_avg_ce", "aff_avg_dice", "aff_avg_iou",
        ):
            assert k in out

    def test_loss_is_finite_and_non_negative(self, batch) -> None:
        pred, labels, image = batch
        out = BoundaryLoss()(pred, labels, image)
        for k, v in out.items():
            assert torch.isfinite(v), k
            assert v.item() >= 0.0, k

    def test_backward_flows(self, batch) -> None:
        pred, labels, image = batch
        out = BoundaryLoss()(pred, labels, image)
        out["loss"].backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_zero_instances_only_raw_and_aff(self, batch) -> None:
        """No instances -> avg sub-loss is zero; raw + aff stay finite."""
        pred, _, image = batch
        B, D, H, W = image.shape
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        out = BoundaryLoss()(pred, labels, image)
        assert out["avg"].item() == 0.0
        assert out["raw"].item() > 0.0
        assert torch.isfinite(out["aff"])

    def test_weights_applied_to_total(self, batch) -> None:
        pred, labels, image = batch
        only_raw = BoundaryLoss(
            weight_avg=0.0, weight_raw=1.0,
            weight_aff_pred=0.0, weight_aff_avg=0.0,
        )(pred, labels, image)
        assert only_raw["loss"].item() == pytest.approx(
            only_raw["raw"].item(), rel=1e-6
        )

        uniform = BoundaryLoss(
            weight_avg=1.0, weight_raw=1.0,
            weight_aff_pred=1.0, weight_aff_avg=1.0,
            weight_dice=1.0,
        )(pred, labels, image)
        assert uniform["loss"].item() >= only_raw["loss"].item() - 1e-6

    def test_disabling_aff_avg_path_removes_its_term(self, batch) -> None:
        """``weight_aff_avg=0`` must zero out the derived path."""
        pred, labels, image = batch
        out = BoundaryLoss(weight_aff_avg=0.0)(pred, labels, image)
        assert out["aff_avg"].item() == 0.0
        assert out["aff_avg_ce"].item() == 0.0
        assert out["aff_avg_dice"].item() == 0.0

    def test_aff_loss_is_soft_dice_on_probs(self, batch) -> None:
        """Perfect direct-aff probs give Dice ~ 1 -> aff_pred ~ 0."""
        pred, labels, image = batch
        loss_fn = BoundaryLoss(
            weight_dice=1.0, weight_aff_avg=0.0,
        )
        target = loss_fn.build_target(labels, image)

        ideal = pred.detach().clone()
        ideal[:, 4:10] = target[:, 4:10]
        out = loss_fn(ideal, labels, image)
        assert out["aff_pred"].item() < 0.05

    def test_aff_avg_path_uses_predicted_avg(self, batch) -> None:
        """Changing predicted avg actually changes the derived-aff loss.

        End-to-end smoke test that the derived path is wired up
        through ``soft_aff_from_avg`` (avg shifts along the spatial
        axes, gradient propagates through ``exp(-tau * L1)``).  We
        deliberately don't compare ``ideal`` to ``random`` on the loss
        magnitude itself: bg→bg interfaces have constant avg=0 →
        derived aff=1 but the **target** at those voxels is 0 (masked
        by ``background=0``), so the bg-dominated mass of the
        derived-aff loss isn't strictly minimised by the ideal avg
        field.  See also :func:`soft_aff_from_avg` -- the kernel only
        provides a useful gradient signal where the predicted avg
        actually differs across faces.
        """
        pred, labels, image = batch
        loss_fn = BoundaryLoss(
            weight_dice=1.0, weight_aff_pred=0.0, weight_aff_avg=1.0,
            tau=2.0,
        )

        out_a = loss_fn(pred, labels, image)
        # Perturbing only the predicted avg (ch 1-3) must change the
        # aff_avg loss; the direct path is disabled so any change to
        # ``aff_avg`` is attributable to ``soft_aff_from_avg``.
        perturbed = pred.detach().clone()
        perturbed[:, 1:4] = (perturbed[:, 1:4] + 0.5).clamp(0.0, 1.0)
        out_b = loss_fn(perturbed, labels, image)

        assert torch.isfinite(out_a["aff_avg"])
        assert torch.isfinite(out_b["aff_avg"])
        assert out_a["aff_avg"].item() != pytest.approx(
            out_b["aff_avg"].item(), abs=1e-6,
        )
        # Perturbing only the avg channels (ch 1-3) must not affect the
        # direct aff sub-loss (ch 4-9 untouched).  Note: ``aff_pred`` in
        # the returned dict is the un-weighted sub-loss of the direct
        # path (dice on ch 4-9 vs the binary aff target); it does not
        # depend on ``weight_aff_pred``.
        assert out_a["aff_pred"].item() == pytest.approx(
            out_b["aff_pred"].item(), abs=1e-6,
        )

    def test_wrong_prediction_channels_raises(self, batch) -> None:
        _, labels, image = batch
        bad = torch.randn(*labels.shape[:1], 16, *labels.shape[1:])
        with pytest.raises(ValueError):
            BoundaryLoss()(bad, labels, image)

    def test_cached_target_matches_recompute(self, batch) -> None:
        pred, labels, image = batch
        loss_fn = BoundaryLoss()
        cached = loss_fn.build_target(labels, image)
        out_fresh = loss_fn(pred.detach().clone().requires_grad_(), labels, image)
        out_cached = loss_fn(
            pred.detach().clone().requires_grad_(),
            labels, image, cached_target=cached,
        )
        assert torch.isclose(out_fresh["loss"], out_cached["loss"], atol=1e-6)


# ---------------------------------------------------------------------------
# Integration with CombinedLoss
# ---------------------------------------------------------------------------


class TestBoundaryInCombinedLoss:
    """End-to-end sanity via :class:`brainbow.losses.CombinedLoss`."""

    def test_combined_loss_exposes_boundary_keys(self) -> None:
        from brainbow.losses import CombinedLoss

        loss = CombinedLoss(
            spatial_dims=3,
            weight_semantic=0.0, weight_instance=0.0, weight_geometry=0.0,
            weight_boundary=1.0,
        )
        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :2, :4, :4] = 1
        labels[0, 2:, 4:, 4:] = 2
        image = torch.rand(B, D, H, W)
        # Match the wrapper's output contract: every boundary channel
        # arrives in [0, 1] (sigmoid is applied by the model wrapper).
        bnd = torch.cat(
            [torch.rand(B, 4, D, H, W), torch.rand(B, 6, D, H, W)], dim=1,
        ).detach().requires_grad_(True)
        preds = {"boundary": bnd}
        targets = {
            "labels": labels,
            "semantic_labels": (labels > 0).long(),
            "raw_image": image,
        }
        out = loss(preds, targets)
        # Scalar tag layout mirrors the per-head image-tag layout in
        # ``brainbow.callbacks.tensorboard.image_logger``:
        #   boundary/pred/aff/{...}      <-> boundary/loss/aff
        #   boundary/pred/avg/aff/{...}  <-> boundary/loss/avg/aff
        for k in (
            "loss",
            "boundary/loss",
            "boundary/loss/raw",
            "boundary/loss/avg",
            "boundary/loss/aff",
            "boundary/loss/avg/aff",
        ):
            assert k in out
        out["loss"].backward()

    def test_combined_loss_disabled_boundary_is_none(self) -> None:
        from brainbow.losses import CombinedLoss

        loss = CombinedLoss(
            spatial_dims=3,
            weight_semantic=0.0, weight_instance=0.0, weight_geometry=0.0,
            weight_boundary=0.0,
        )
        assert loss.boundary_loss is None

    def test_combined_loss_raises_without_raw_image(self) -> None:
        from brainbow.losses import CombinedLoss

        loss = CombinedLoss(
            spatial_dims=3,
            weight_semantic=0.0, weight_instance=0.0, weight_geometry=0.0,
            weight_boundary=1.0,
        )
        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :2, :4, :4] = 1
        preds = {"boundary": torch.cat(
            [torch.rand(B, 4, D, H, W), torch.rand(B, 6, D, H, W)], dim=1,
        )}
        with pytest.raises(KeyError):
            loss(preds, {"labels": labels, "semantic_labels": (labels > 0).long()})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
