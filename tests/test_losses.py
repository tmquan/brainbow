"""
Tests for the combined segmentation loss + the standalone task losses
that feed into it.
"""

import pytest
import torch

from brainbow.losses import (
    CombinedLoss,
    GeometryLoss,
    InstanceLoss,
    SemanticLoss,
)


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

    def test_shared_aff_target_matches_per_head_rebuild(self) -> None:
        """Cached aff target shared with InstanceLoss matches the
        standalone rebuild used when the cache isn't pre-populated.

        Exercises the option-3b sharing path: when both ``BoundaryLoss``
        and ``InstanceLoss(weight_aff_emb > 0)`` are enabled,
        :meth:`CombinedLoss._build_targets` builds the boundary head's
        full 10-channel target once and slices ch 4-9 as the aff target
        for the instance head.  The total loss must equal what we get
        when the two heads each rebuild their own aff target.
        """
        torch.manual_seed(13)
        loss_fn = CombinedLoss(
            spatial_dims=3,
            weight_semantic=0.0,
            weight_instance=dict(
                weight=1.0,
                weight_pull=0.0, weight_push=0.0, weight_norm=0.0,
                weight_aff_emb=1.0, tau=1.0,
            ),
            weight_geometry=0.0,
            weight_boundary=1.0,
        )

        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[0, :2, :4, :4] = 1
        labels[0, 2:, 4:, 4:] = 2
        image = torch.rand(B, D, H, W)
        bnd = torch.cat(
            [torch.rand(B, 4, D, H, W), torch.rand(B, 6, D, H, W)], dim=1,
        )
        emb = torch.randn(B, 8, D, H, W)
        predictions = {"instance": emb, "boundary": bnd}
        targets = {
            "labels": labels,
            "semantic_labels": (labels > 0).long(),
            "raw_image": image,
        }

        out_lazy = loss_fn(predictions, targets)
        # Force the shared-cache path explicitly.
        targets["_cached_weights"] = loss_fn._build_targets(
            targets["labels"], targets,
        )
        out_cached = loss_fn(predictions, targets)

        assert torch.isclose(
            out_lazy["loss"], out_cached["loss"], atol=1e-5,
        )
        assert torch.isclose(
            out_lazy["instance/loss/emb/aff"],
            out_cached["instance/loss/emb/aff"],
            atol=1e-6,
        )
        assert torch.isclose(
            out_lazy["boundary/loss"],
            out_cached["boundary/loss"],
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# SemanticLoss (direct, no CombinedLoss orchestration)
# ---------------------------------------------------------------------------

class TestSemanticLoss:
    """Direct exercises of :class:`SemanticLoss`.

    Most of the existing coverage runs through :class:`CombinedLoss`
    with ``weight_geometry: 0`` -- which is fine for integration but
    leaves the direct-path return contract uncovered.
    """

    @pytest.fixture()
    def probs(self) -> torch.Tensor:
        # Already-sigmoided probabilities, [B, C, H, W].
        torch.manual_seed(0)
        return torch.sigmoid(torch.randn(2, 1, 8, 8))

    @pytest.fixture()
    def labels(self) -> torch.Tensor:
        labels = torch.zeros(2, 8, 8, dtype=torch.long)
        labels[:, :4, :4] = 1
        return labels

    def test_returns_loss_ce_iou_dice(self, probs, labels) -> None:
        loss_fn = SemanticLoss(weight_ce=1.0, weight_iou=1.0, weight_dice=1.0)
        out = loss_fn(probs, labels)
        for k in ("loss", "ce", "iou", "dice"):
            assert k in out

    def test_zero_weights_skip_subloss(self, probs, labels) -> None:
        loss_fn = SemanticLoss(weight_ce=1.0, weight_iou=0.0, weight_dice=0.0)
        out = loss_fn(probs, labels)
        # ``ce`` is unconditional; ``iou`` / ``dice`` short-circuit to 0.
        assert out["iou"].item() == pytest.approx(0.0)
        assert out["dice"].item() == pytest.approx(0.0)

    def test_finite_and_non_negative(self, probs, labels) -> None:
        loss_fn = SemanticLoss()
        out = loss_fn(probs, labels)
        assert out["loss"].isfinite()
        assert out["loss"].item() >= 0.0

    def test_active_classes_truncates(self) -> None:
        probs = torch.sigmoid(torch.randn(1, 4, 8, 8))
        labels = torch.zeros(1, 8, 8, dtype=torch.long)
        loss_fn = SemanticLoss(active_classes=2)
        out = loss_fn(probs, labels)
        assert out["loss"].isfinite()


# ---------------------------------------------------------------------------
# InstanceLoss (direct)
# ---------------------------------------------------------------------------

class TestInstanceLoss:
    @pytest.fixture()
    def embed_and_label(self):
        torch.manual_seed(1)
        embed = torch.randn(2, 8, 16, 16, requires_grad=True)
        label = torch.zeros(2, 16, 16, dtype=torch.long)
        label[:, :8, :8] = 1
        label[:, :8, 8:] = 2
        return embed, label

    def test_returns_loss_pull_push_norm(self, embed_and_label) -> None:
        embed, label = embed_and_label
        loss_fn = InstanceLoss(spatial_dims=2)
        out = loss_fn(embed, label)
        for k in ("loss", "pull", "push", "norm"):
            assert k in out
        assert out["loss"].isfinite()

    def test_zero_instances(self) -> None:
        loss_fn = InstanceLoss(spatial_dims=2)
        embed = torch.randn(1, 4, 8, 8)
        label = torch.zeros(1, 8, 8, dtype=torch.long)
        out = loss_fn(embed, label)
        assert out["loss"].isfinite()

    def test_backward(self, embed_and_label) -> None:
        embed, label = embed_and_label
        loss_fn = InstanceLoss(spatial_dims=2, weight_pull=1.0, weight_push=1.0)
        out = loss_fn(embed, label)
        out["loss"].backward()
        assert embed.grad is not None
        assert torch.isfinite(embed.grad).all()

    @pytest.fixture()
    def embed_and_label_3d(self):
        torch.manual_seed(7)
        B, E, D, H, W = 2, 8, 4, 16, 16
        embed = torch.randn(B, E, D, H, W, requires_grad=True)
        label = torch.zeros(B, D, H, W, dtype=torch.long)
        label[:, :2, :8, :8] = 1
        label[:, :2, :8, 8:] = 2
        label[:, 2:, :, :] = 3
        return embed, label

    def test_aff_emb_zero_weight_is_no_op(self, embed_and_label_3d) -> None:
        """Default ``weight_aff_emb=0`` zeroes the term + skips the kernel."""
        embed, label = embed_and_label_3d
        loss_fn = InstanceLoss(spatial_dims=3, weight_aff_emb=0.0)
        out = loss_fn(embed, label)
        assert "aff_emb" in out
        assert out["aff_emb"].item() == 0.0
        # Sanity: total loss equals weighted pull/push/norm only.
        expected = (
            loss_fn.weight_pull * out["pull"]
            + loss_fn.weight_push * out["push"]
            + loss_fn.weight_norm * out["norm"]
        )
        assert torch.isclose(out["loss"], expected, atol=1e-6)

    def test_aff_emb_active_changes_loss(self, embed_and_label_3d) -> None:
        """Enabling the aff_emb path adds a finite, positive term."""
        embed, label = embed_and_label_3d
        loss_fn = InstanceLoss(
            spatial_dims=3,
            weight_pull=0.0, weight_push=0.0, weight_norm=0.0,
            weight_aff_emb=1.0, tau=1.0,
        )
        out = loss_fn(embed, label)
        assert torch.isfinite(out["aff_emb"])
        assert out["aff_emb"].item() > 0.0
        # With pull/push/norm zeroed, total = aff_emb term.
        assert torch.isclose(out["loss"], out["aff_emb"], atol=1e-6)

    def test_aff_emb_backward_flows(self, embed_and_label_3d) -> None:
        embed, label = embed_and_label_3d
        loss_fn = InstanceLoss(
            spatial_dims=3,
            weight_pull=0.0, weight_push=0.0, weight_norm=0.0,
            weight_aff_emb=1.0, tau=1.0,
        )
        out = loss_fn(embed, label)
        out["loss"].backward()
        assert embed.grad is not None
        assert torch.isfinite(embed.grad).all()
        assert embed.grad.abs().sum() > 0

    def test_aff_emb_2d_raises(self) -> None:
        """The 6-face primitives assume BDHW labels; 2-D mode is rejected."""
        with pytest.raises(ValueError, match="spatial_dims"):
            InstanceLoss(spatial_dims=2, weight_aff_emb=1.0)

    def test_aff_emb_cached_target_matches_rebuild(
        self, embed_and_label_3d,
    ) -> None:
        """``cached_aff_target=`` skips the rebuild but yields the same loss."""
        from brainbow.losses.boundary import _affinity_target_torch

        embed, label = embed_and_label_3d
        loss_fn = InstanceLoss(
            spatial_dims=3,
            weight_pull=0.0, weight_push=0.0, weight_norm=0.0,
            weight_aff_emb=1.0, tau=1.0,
        )

        # Lazy path: the loss rebuilds the aff target itself.
        out_lazy = loss_fn(embed, label)

        # Cached path: pre-build with the same args the loss would use.
        cached = _affinity_target_torch(label.long(), background=loss_fn.background)
        out_cached = loss_fn(embed, label, cached_aff_target=cached)

        assert torch.isclose(
            out_lazy["aff_emb"], out_cached["aff_emb"], atol=1e-6,
        )
        assert torch.isclose(
            out_lazy["loss"], out_cached["loss"], atol=1e-6,
        )


# ---------------------------------------------------------------------------
# GeometryLoss (direct)
# ---------------------------------------------------------------------------

class TestGeometryLoss:
    @pytest.fixture()
    def geometry_inputs(self):
        torch.manual_seed(2)
        # 2-D geometry head: raw(1) + dir(2) + cov(3) = 6 channels.
        geom = torch.randn(1, 6, 16, 16, requires_grad=True)
        ins_label = torch.zeros(1, 16, 16, dtype=torch.long)
        ins_label[:, :8, :8] = 1
        ins_label[:, 8:, 8:] = 2
        raw = torch.randn(1, 1, 16, 16)
        return geom, ins_label, raw

    def test_returns_loss_dir_cov_raw(self, geometry_inputs) -> None:
        geom, ins_label, raw = geometry_inputs
        loss_fn = GeometryLoss(
            spatial_dims=2,
            weight_dir=1.0, weight_cov=1.0, weight_raw=1.0,
        )
        out = loss_fn(geom, ins_label, raw_image=raw)
        for k in ("loss", "dir", "cov", "raw"):
            assert k in out
        assert out["loss"].isfinite()

    def test_no_raw_image_skips_raw_subloss(self, geometry_inputs) -> None:
        geom, ins_label, _ = geometry_inputs
        loss_fn = GeometryLoss(
            spatial_dims=2,
            weight_dir=1.0, weight_cov=0.0, weight_raw=1.0,
        )
        out = loss_fn(geom, ins_label, raw_image=None)
        # ``raw`` is short-circuited to zero when no image is provided.
        assert out["raw"].item() == pytest.approx(0.0)

    def test_backward(self, geometry_inputs) -> None:
        geom, ins_label, raw = geometry_inputs
        loss_fn = GeometryLoss(spatial_dims=2)
        out = loss_fn(geom, ins_label, raw_image=raw)
        out["loss"].backward()
        assert geom.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
