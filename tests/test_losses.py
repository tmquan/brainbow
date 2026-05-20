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
    DiceBCEFocalLoss,
    affinity_target,
    apply_head_activations,
    slice_head,
    soft_aff_from_field,
    stable_bce_on_probs,
)


def _sample_batch():
    torch.manual_seed(7)
    B, D, H, W = 2, 4, 8, 8
    head = torch.randn(B, HEAD_CHANNELS, D, H, W, requires_grad=True)
    # The wrappers apply sigmoid to the semantic + skeleton channels
    # before the loss; mimic that here so the composite
    # Dice + BCE + Focal supervisor sees already-sigmoided
    # probabilities (per the activation contract documented in
    # ``apply_head_activations``).
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
    # Composite-loss heads emit only the field-level total -- their
    # three sub-terms are already weighted-in by ``lambda_*`` so
    # tracking them separately would re-introduce the per-config-knob
    # tuning loop the simplification was meant to eliminate.
    assert "loss/sem/ce" not in out
    assert "loss/sem/dice" not in out
    assert "loss/sem/focal" not in out
    assert "loss/skl/ce" not in out
    assert "loss/skl/dice" not in out
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


# ---------------------------------------------------------------------------
# DiceBCEFocalLoss (composite Dice + BCE + Focal on probabilities)
# ---------------------------------------------------------------------------


def _sample_probs_target():
    """A small binary-segmentation-style batch in probability space."""
    torch.manual_seed(11)
    B, C, D, H, W = 2, 1, 4, 8, 8
    logits = torch.randn(B, C, D, H, W) * 2.0
    probs = logits.sigmoid().requires_grad_(True)
    target = (torch.rand(B, C, D, H, W) > 0.6).float()
    return probs, target


def test_dice_bce_focal_forward_finite() -> None:
    probs, target = _sample_probs_target()
    loss_fn = DiceBCEFocalLoss(
        lambda_dice=1.0, lambda_bce=1.0, lambda_focal=1.0, gamma=2.0,
    )
    loss = loss_fn(probs, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_dice_bce_focal_backward_routes_gradient() -> None:
    probs, target = _sample_probs_target()
    loss_fn = DiceBCEFocalLoss()
    loss = loss_fn(probs, target)
    loss.backward()
    assert probs.grad is not None
    assert torch.isfinite(probs.grad).all()
    assert probs.grad.abs().sum() > 0


def test_dice_bce_focal_zero_lambdas_match_individual_terms() -> None:
    """Each lambda independently controls its term."""
    probs, target = _sample_probs_target()

    dice_only = DiceBCEFocalLoss(
        lambda_dice=1.0, lambda_bce=0.0, lambda_focal=0.0,
    )(probs, target)
    bce_only = DiceBCEFocalLoss(
        lambda_dice=0.0, lambda_bce=1.0, lambda_focal=0.0,
    )(probs, target)
    focal_only = DiceBCEFocalLoss(
        lambda_dice=0.0, lambda_bce=0.0, lambda_focal=1.0, gamma=2.0,
    )(probs, target)
    full = DiceBCEFocalLoss(
        lambda_dice=1.0, lambda_bce=1.0, lambda_focal=1.0, gamma=2.0,
    )(probs, target)

    # Linearity of the composite in its three lambdas.
    assert torch.allclose(full, dice_only + bce_only + focal_only, atol=1e-5)


def test_dice_bce_focal_gamma_zero_collapses_focal_to_bce() -> None:
    """With ``gamma=0`` the focal weight ``(1 - p_t)^0 == 1`` so focal
    reduces exactly to per-voxel BCE."""
    probs, target = _sample_probs_target()
    bce = DiceBCEFocalLoss(
        lambda_dice=0.0, lambda_bce=1.0, lambda_focal=0.0,
    )(probs, target)
    focal_g0 = DiceBCEFocalLoss(
        lambda_dice=0.0, lambda_bce=0.0, lambda_focal=1.0, gamma=0.0,
    )(probs, target)
    assert torch.allclose(bce, focal_g0, atol=1e-5)


def test_dice_bce_focal_gamma_upweights_hard_voxels() -> None:
    """Higher ``gamma`` puts more loss on misclassified (low ``p_t``)
    voxels, so for a perfectly-wrong prediction the focal loss grows
    while a perfectly-correct one stays at ~0."""
    # All-zero prediction against all-positive target -> every voxel
    # is "hard" (``p_t = eps``).  Increasing gamma increases the focal
    # weight, hence the loss.
    probs = torch.full((1, 1, 2, 4, 4), 0.05, requires_grad=True)
    target = torch.ones_like(probs)
    g1 = DiceBCEFocalLoss(
        lambda_dice=0.0, lambda_bce=0.0, lambda_focal=1.0, gamma=1.0,
    )(probs, target)
    g3 = DiceBCEFocalLoss(
        lambda_dice=0.0, lambda_bce=0.0, lambda_focal=1.0, gamma=3.0,
    )(probs, target)
    # ``g3`` weights the hard voxels by ``(1 - 0.05)^3`` vs ``g1``'s
    # ``(1 - 0.05)^1``, so it must be smaller (since ``0.95 < 1``).
    assert float(g3) < float(g1)


def test_stable_bce_on_probs_matches_torch_reference() -> None:
    """The fp32-clamped BCE on probs must agree with the canonical
    formula on safe inputs (away from 0 / 1)."""
    torch.manual_seed(0)
    probs = torch.rand(8).clamp(0.05, 0.95)
    target = (torch.rand(8) > 0.5).float()
    ours = stable_bce_on_probs(probs, target)
    ref = -(target * probs.log() + (1 - target) * (1 - probs).log())
    assert torch.allclose(ours, ref, atol=1e-6)


def test_combined_loss_threads_composite_lambdas_to_sem_skl_aff() -> None:
    """All four composite-loss heads accept the lambda / gamma sub-keys
    and surface them on the per-head ``_*_loss`` attribute."""
    loss_fn = CombinedLoss(
        weight_sem={"weight": 1.0, "lambda_dice": 2.0, "lambda_bce": 0.5, "lambda_focal": 0.1, "gamma": 1.0},
        weight_skl={"weight": 1.0, "lambda_dice": 0.5, "lambda_bce": 2.0, "lambda_focal": 0.0, "gamma": 3.0},
        weight_aff_emb={"weight": 1.0, "tau": 0.5, "lambda_focal": 4.0},
        weight_aff_avg={"weight": 1.0, "tau": 2.0, "lambda_bce": 0.0},
    )
    assert loss_fn._sem_loss.lambda_dice == pytest.approx(2.0)
    assert loss_fn._sem_loss.lambda_bce == pytest.approx(0.5)
    assert loss_fn._sem_loss.lambda_focal == pytest.approx(0.1)
    assert loss_fn._sem_loss.gamma == pytest.approx(1.0)
    assert loss_fn._skl_loss.lambda_dice == pytest.approx(0.5)
    assert loss_fn._skl_loss.lambda_focal == pytest.approx(0.0)
    assert loss_fn._skl_loss.gamma == pytest.approx(3.0)
    assert loss_fn.aff_emb_tau == pytest.approx(0.5)
    assert loss_fn._aff_emb_loss.lambda_focal == pytest.approx(4.0)
    assert loss_fn.aff_avg_tau == pytest.approx(2.0)
    assert loss_fn._aff_avg_loss.lambda_bce == pytest.approx(0.0)

