"""Tests for the unified 32-channel ``CombinedLoss``."""

import pytest
import torch

from brainbow.losses import (
    AFF_CHANNELS,
    AFF_NAMES,
    EMB_SLICE,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    RAD_SLICE,
    SEM_SLICE,
    SIGMOID_SLICE,
    SKL_SLICE,
    CombinedLoss,
    affinity_target,
    apply_head_activations,
    slice_head,
    soft_aff_from_field,
)


def _sample_batch():
    torch.manual_seed(7)
    B, D, H, W = 2, 4, 8, 8
    head = torch.randn(B, HEAD_CHANNELS, D, H, W, requires_grad=True)
    # The wrappers apply sigmoid to the semantic + skeleton channels
    # before the loss; mimic that here so the Dice terms see
    # already-sigmoided probabilities.
    with torch.no_grad():
        head[:, SIGMOID_SLICE] = head[:, SIGMOID_SLICE].sigmoid()

    labels = torch.zeros(B, D, H, W, dtype=torch.long)
    labels[:, :2, :4, :4] = 1
    labels[:, :2, :4, 4:] = 2
    labels[:, 2:, 4:, :] = 3

    raw = torch.rand(B, 1, D, H, W)
    direction = torch.randn(B, 3, D, H, W)
    covariance = torch.randn(B, 6, D, H, W)
    skl = (torch.rand(B, D, H, W) > 0.95).float()
    radius = torch.rand(B, D, H, W)
    targets = {
        "labels": labels,
        "raw_image": raw,
        "label_direction": direction,
        "label_covariance": covariance,
        "label_skl": skl,
        "label_radius": radius,
    }
    return head, targets


def test_channel_layout_has_embedding_last() -> None:
    assert HEAD_CHANNELS == 32
    assert HEAD_LAYOUT["raw"] == slice(0, 1)
    assert HEAD_LAYOUT["sem"] == slice(1, 2)
    assert HEAD_LAYOUT["skl"] == slice(2, 3)
    assert HEAD_LAYOUT["dir"] == slice(3, 6)
    assert HEAD_LAYOUT["cov"] == slice(6, 12)
    assert HEAD_LAYOUT["rad"] == slice(12, 13)
    assert HEAD_LAYOUT["avg"] == slice(13, 16)
    assert HEAD_LAYOUT["emb"] == slice(16, 32)
    assert EMB_SLICE == slice(16, 32)
    assert SKL_SLICE == slice(2, 3)
    assert RAD_SLICE == slice(12, 13)
    # The two sigmoid slots are adjacent so the wrapper's activation
    # policy is a single contiguous slice.
    assert SIGMOID_SLICE == slice(1, 3)


def test_slice_head_returns_expected_shapes() -> None:
    head, _ = _sample_batch()
    fields = slice_head(head)
    assert {k: v.shape[1] for k, v in fields.items()} == {
        "raw": 1,
        "sem": 1,
        "skl": 1,
        "dir": 3,
        "cov": 6,
        "rad": 1,
        "avg": 3,
        "emb": 16,
    }


def test_apply_head_activations_sigmoids_sem_and_skl_only() -> None:
    """Only ``[SIGMOID_SLICE]`` should be passed through sigmoid; every
    other channel must come out bit-identical (linear pass-through)."""
    torch.manual_seed(0)
    B, D, H, W = 1, 2, 4, 4
    raw = torch.randn(B, HEAD_CHANNELS, D, H, W) * 5.0  # spread it out
    out = apply_head_activations(raw)

    # Sigmoid slots are bounded to [0, 1] (saturates to exactly 0 / 1
    # in fp32 once |x| >~ 17; sigmoid is monotone so equality at the
    # endpoints is fine).
    sig_slot = out[:, SIGMOID_SLICE]
    assert sig_slot.min().item() >= 0.0
    assert sig_slot.max().item() <= 1.0
    assert torch.allclose(sig_slot, raw[:, SIGMOID_SLICE].sigmoid())

    # Everything else is unchanged.
    assert torch.equal(out[:, :SIGMOID_SLICE.start], raw[:, :SIGMOID_SLICE.start])
    assert torch.equal(out[:, SIGMOID_SLICE.stop:], raw[:, SIGMOID_SLICE.stop:])
    # And there exist negative values in the linear region (i.e. the
    # sigmoid wasn't accidentally applied to a non-sigmoid slot).
    assert torch.any(out[:, SIGMOID_SLICE.stop:] < 0.0)


def test_affinity_target_is_12_channel() -> None:
    _, targets = _sample_batch()
    aff = affinity_target(targets["labels"], background=-1)
    assert aff.shape[1] == AFF_CHANNELS == 12
    assert AFF_NAMES == (
        "T1", "B1", "U1", "D1", "L1", "R1",
        "T2", "B2", "U2", "D2", "L2", "R2",
    )
    assert torch.all((aff == 0) | (aff == 1))


def test_soft_aff_from_field_is_12_channel() -> None:
    head, _ = _sample_batch()
    emb = head[:, EMB_SLICE]
    aff = soft_aff_from_field(emb, tau=1.0)
    assert aff.shape[:2] == (head.shape[0], 12)
    assert torch.isfinite(aff).all()
    assert float(aff.detach().min()) >= 0.0
    assert float(aff.detach().max()) <= 1.0


def test_combined_loss_forward_backward() -> None:
    head, targets = _sample_batch()
    loss_fn = CombinedLoss()
    targets["_cached_targets"] = loss_fn.build_targets(targets["labels"], targets)
    out = loss_fn(head, targets)

    required = {
        "loss",
        "loss/raw",
        "loss/sem",
        "loss/dir",
        "loss/cov",
        "loss/avg",
        "loss/emb",
        "loss/emb/pull",
        "loss/emb/push",
        "loss/emb/norm",
        "loss/aff_emb",
        "loss/aff_avg",
    }
    assert required.issubset(out)
    # Dice-only heads no longer emit a per-sub-loss breakdown.
    assert "loss/sem/ce" not in out
    assert "loss/sem/dice" not in out
    assert "loss/skl/ce" not in out
    assert "loss/aff_emb/ce" not in out
    assert "loss/aff_emb/dice" not in out
    assert "loss/aff_avg/ce" not in out
    assert "loss/aff_avg/dice" not in out
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert head.grad is not None
    assert torch.isfinite(head.grad).all()
    assert head.grad.abs().sum() > 0


def test_zero_weight_fields_are_omitted() -> None:
    head, targets = _sample_batch()
    loss_fn = CombinedLoss(
        weight_raw=0.0,
        weight_skl=0.0,
        weight_dir=0.0,
        weight_cov=0.0,
        weight_rad=0.0,
        weight_avg=0.0,
        weight_aff_emb=0.0,
        weight_aff_avg=0.0,
    )
    out = loss_fn(head, targets)
    assert "loss/raw" not in out
    assert "loss/skl" not in out
    assert "loss/dir" not in out
    assert "loss/cov" not in out
    assert "loss/rad" not in out
    assert "loss/avg" not in out
    assert "loss/aff_emb" not in out
    assert "loss/aff_avg" not in out
    assert "loss/sem" in out
    assert "loss/emb" in out


def test_skl_and_rad_run_when_enabled() -> None:
    """With both new fields enabled and their targets provided, the
    loss dict carries ``loss/skl`` / ``loss/rad`` and the backward pass
    routes gradient to those head slices."""
    head, targets = _sample_batch()
    loss_fn = CombinedLoss(weight_skl=1.0, weight_rad=1.0)
    targets["_cached_targets"] = loss_fn.build_targets(targets["labels"], targets)
    out = loss_fn(head, targets)
    assert "loss/skl" in out
    assert "loss/rad" in out
    assert torch.isfinite(out["loss/skl"])
    assert torch.isfinite(out["loss/rad"])
    out["loss"].backward()
    assert head.grad is not None
    assert head.grad[:, SKL_SLICE].abs().sum() > 0
    assert head.grad[:, RAD_SLICE].abs().sum() > 0


def test_nested_mapping_without_weight_defaults_to_enabled() -> None:
    loss_fn = CombinedLoss(weight_raw={"loss": "l1"})
    assert loss_fn.weight_raw == pytest.approx(1.0)


def test_missing_raw_image_raises_when_raw_enabled() -> None:
    head, targets = _sample_batch()
    targets.pop("raw_image")
    with pytest.raises(KeyError, match="raw_image"):
        CombinedLoss(weight_raw=1.0)(head, targets)


def test_direction_and_covariance_can_be_absent_when_disabled() -> None:
    head, targets = _sample_batch()
    targets.pop("label_direction")
    targets.pop("label_covariance")
    out = CombinedLoss(weight_dir=0.0, weight_cov=0.0)(head, targets)
    assert torch.isfinite(out["loss"])

