"""Tests for the unified 30-channel ``CombinedLoss``."""

import pytest
import torch

from brainbow.losses import (
    AFF_CHANNELS,
    AFF_NAMES,
    EMB_SLICE,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    SEM_SLICE,
    CombinedLoss,
    affinity_target,
    slice_head,
    soft_aff_from_field,
)


def _sample_batch():
    torch.manual_seed(7)
    B, D, H, W = 2, 4, 8, 8
    head = torch.randn(B, HEAD_CHANNELS, D, H, W, requires_grad=True)
    # The wrappers apply sigmoid to the semantic channel before the loss.
    with torch.no_grad():
        head[:, SEM_SLICE] = head[:, SEM_SLICE].sigmoid()

    labels = torch.zeros(B, D, H, W, dtype=torch.long)
    labels[:, :2, :4, :4] = 1
    labels[:, :2, :4, 4:] = 2
    labels[:, 2:, 4:, :] = 3

    raw = torch.rand(B, 1, D, H, W)
    direction = torch.randn(B, 3, D, H, W)
    covariance = torch.randn(B, 6, D, H, W)
    targets = {
        "labels": labels,
        "raw_image": raw,
        "label_direction": direction,
        "label_covariance": covariance,
    }
    return head, targets


def test_channel_layout_has_embedding_last() -> None:
    assert HEAD_CHANNELS == 30
    assert HEAD_LAYOUT["raw"] == slice(0, 1)
    assert HEAD_LAYOUT["sem"] == slice(1, 2)
    assert HEAD_LAYOUT["dir"] == slice(2, 5)
    assert HEAD_LAYOUT["cov"] == slice(5, 11)
    assert HEAD_LAYOUT["avg"] == slice(11, 14)
    assert HEAD_LAYOUT["emb"] == slice(14, 30)
    assert EMB_SLICE == slice(14, 30)


def test_slice_head_returns_expected_shapes() -> None:
    head, _ = _sample_batch()
    fields = slice_head(head)
    assert {k: v.shape[1] for k, v in fields.items()} == {
        "raw": 1,
        "sem": 1,
        "dir": 3,
        "cov": 6,
        "avg": 3,
        "emb": 16,
    }


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
        "loss/sem/ce",
        "loss/sem/dice",
        "loss/dir",
        "loss/cov",
        "loss/avg",
        "loss/emb",
        "loss/emb/pull",
        "loss/emb/push",
        "loss/emb/norm",
        "loss/aff_emb",
        "loss/aff_emb/dice",
        "loss/aff_avg",
        "loss/aff_avg/dice",
    }
    assert required.issubset(out)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert head.grad is not None
    assert torch.isfinite(head.grad).all()
    assert head.grad.abs().sum() > 0


def test_zero_weight_fields_are_omitted() -> None:
    head, targets = _sample_batch()
    loss_fn = CombinedLoss(
        weight_raw=0.0,
        weight_dir=0.0,
        weight_cov=0.0,
        weight_avg=0.0,
        weight_aff_emb=0.0,
        weight_aff_avg=0.0,
    )
    out = loss_fn(head, targets)
    assert "loss/raw" not in out
    assert "loss/dir" not in out
    assert "loss/cov" not in out
    assert "loss/avg" not in out
    assert "loss/aff_emb" not in out
    assert "loss/aff_avg" not in out
    assert "loss/sem" in out
    assert "loss/emb" in out


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

