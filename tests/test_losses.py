"""Tests for the affinity + sem + raw head (``AffinityFGLoss``)."""

import pytest
import torch

from brainbow.losses import (
    AFF_CHANNELS,
    AFF_NAMES,
    AFF_SLICE,
    AFFINITY_OFFSETS,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    N_AFF,
    N_PULL,
    RAW_SLICE,
    SEM_SLICE,
    AffinityFGLoss,
    DiceBCEFocalLoss,
    affinity_target_from_offsets,
    affinity_validity_mask,
    slice_head,
    stable_bce_on_probs,
)


def _sample_batch(requires_grad: bool = True):
    """A small head + targets batch.

    The head emits raw logits / linear values directly (no activation in
    the forward pass), so the same tensor is both the differentiable head
    and the loss input; gradients flow back to it.
    """
    torch.manual_seed(7)
    # H/W >= 28 and D >= 5 so the longest affinity offset (in-plane 27,
    # axial 4) stays within the volume (``shift_replicate`` requires
    # ``|offset| < axis``).
    B, D, H, W = 2, 6, 32, 32
    raw_head = torch.randn(B, HEAD_CHANNELS, D, H, W, requires_grad=requires_grad)
    head = raw_head

    labels = torch.zeros(B, D, H, W, dtype=torch.long)
    labels[:, :, :16, :16] = 1
    labels[:, :, :16, 16:] = 2
    labels[:, :, 16:, :] = 3

    targets = {"labels": labels, "raw_image": torch.rand(B, 1, D, H, W)}
    return raw_head, head, targets


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------

def test_channel_layout() -> None:
    assert HEAD_CHANNELS == N_AFF + 2 == 16
    assert N_PULL == 3
    assert AFF_SLICE == slice(0, N_AFF)
    assert SEM_SLICE == slice(N_AFF, N_AFF + 1)
    assert RAW_SLICE == slice(N_AFF + 1, N_AFF + 2)
    assert HEAD_LAYOUT["aff"] == AFF_SLICE
    assert HEAD_LAYOUT["sem"] == SEM_SLICE
    assert HEAD_LAYOUT["raw"] == RAW_SLICE


def test_slice_head_returns_expected_shapes() -> None:
    _, head, _ = _sample_batch()
    fields = slice_head(head)
    assert {k: v.shape[1] for k, v in fields.items()} == {
        "aff": N_AFF,
        "sem": 1,
        "raw": 1,
    }


def test_head_is_raw_logits_unbounded() -> None:
    """The head emits raw logits / linear values (no activation applied).

    The loss must accept unbounded inputs and remain finite, since the
    head no longer sigmoids the aff + sem block in its forward pass.
    """
    torch.manual_seed(0)
    raw = (torch.randn(1, HEAD_CHANNELS, 6, 32, 32) * 5.0).requires_grad_(True)
    # Unbounded on every channel -- nothing is clamped to [0, 1].
    assert raw[:, AFF_SLICE].min().item() < 0.0
    assert raw[:, SEM_SLICE].max().item() > 1.0

    labels = torch.zeros(1, 6, 32, 32, dtype=torch.long)
    labels[:, :, :16] = 1
    targets = {"labels": labels, "raw_image": torch.rand(1, 1, 6, 32, 32) * 2 - 1}
    loss_fn = AffinityFGLoss(weight_aff={"weight": 1.0, "lambda_focal": 1.0})
    out = loss_fn(raw, targets)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert torch.isfinite(raw.grad).all()


# ---------------------------------------------------------------------------
# Affinity targets
# ---------------------------------------------------------------------------

def test_affinity_target_from_offsets() -> None:
    _, _, targets = _sample_batch()
    aff = affinity_target_from_offsets(targets["labels"], AFFINITY_OFFSETS, background=-1)
    assert aff.shape[1] == N_AFF == AFF_CHANNELS == 14
    assert aff.dtype == torch.uint8
    assert torch.all((aff == 0) | (aff == 1))
    # Names: pull (nearest-neighbour) first, then push (long-range).
    assert len(AFF_NAMES) == N_AFF
    assert AFF_NAMES[0].split("_")[1] == "pull"
    assert AFF_NAMES[N_PULL].split("_")[1] == "push"


def test_affinity_validity_mask() -> None:
    _, _, targets = _sample_batch()
    mask = affinity_validity_mask(targets["labels"] > 0, AFFINITY_OFFSETS)
    assert mask.shape[1] == N_AFF
    assert mask.dtype == torch.uint8
    assert torch.all((mask == 0) | (mask == 1))


# ---------------------------------------------------------------------------
# AffinityFGLoss
# ---------------------------------------------------------------------------

def test_affinity_fg_loss_forward_backward() -> None:
    raw_head, head, targets = _sample_batch()
    loss_fn = AffinityFGLoss(weight_aff={"weight": 1.0, "lambda_focal": 1.0})
    targets["_cached_targets"] = loss_fn.build_targets(targets["labels"], targets)
    out = loss_fn(head, targets)

    assert {"loss", "loss/aff", "loss/sem", "loss/raw"}.issubset(out)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert raw_head.grad is not None
    assert torch.isfinite(raw_head.grad).all()
    assert raw_head.grad.abs().sum() > 0


def test_canonical_loss_keys_match_forward() -> None:
    _, head, targets = _sample_batch(requires_grad=False)
    loss_fn = AffinityFGLoss()
    out = loss_fn(head, dict(targets))
    keys = set(loss_fn.canonical_loss_keys())
    assert keys == {"loss", "loss/aff", "loss/sem", "loss/raw"}
    assert keys == set(out)


def test_zero_weight_fields_are_omitted() -> None:
    _, head, targets = _sample_batch(requires_grad=False)
    out = AffinityFGLoss(weight_sem=0.0, weight_raw=0.0)(head, dict(targets))
    assert "loss/aff" in out
    assert "loss/sem" not in out
    assert "loss/raw" not in out


def test_chunked_affinity_loss_matches_unchunked() -> None:
    """Offset-axis chunking is a memory optimisation, not a numeric change."""
    _, head, targets = _sample_batch(requires_grad=False)
    cfg = {"weight": 1.0, "lambda_focal": 1.0, "push_weight": 3.0}
    l1 = AffinityFGLoss(weight_aff=dict(cfg), aff_chunk_size=1)(head, dict(targets))["loss/aff"]
    lN = AffinityFGLoss(weight_aff=dict(cfg), aff_chunk_size=N_AFF)(head, dict(targets))["loss/aff"]
    assert torch.allclose(l1, lN, atol=1e-5)


def test_pull_push_offset_weights() -> None:
    loss_fn = AffinityFGLoss(weight_aff={"pull_weight": 2.0, "push_weight": 7.0})
    w = loss_fn._offset_weights
    assert torch.all(w[:N_PULL] == 2.0)
    assert torch.all(w[N_PULL:] == 7.0)


def test_missing_raw_image_raises_when_raw_enabled() -> None:
    _, head, targets = _sample_batch(requires_grad=False)
    targets.pop("raw_image")
    with pytest.raises(KeyError, match="raw_image"):
        AffinityFGLoss(weight_raw=1.0)(head, targets)


# ---------------------------------------------------------------------------
# DiceBCEFocalLoss (composite Dice + BCE + Focal on logits)
# ---------------------------------------------------------------------------

def _sample_logits_target():
    torch.manual_seed(11)
    B, C, D, H, W = 2, 1, 4, 8, 8
    logits = (torch.randn(B, C, D, H, W) * 2.0).requires_grad_(True)
    target = (torch.rand(B, C, D, H, W) > 0.6).float()
    return logits, target


def test_dice_bce_focal_forward_finite() -> None:
    logits, target = _sample_logits_target()
    loss = DiceBCEFocalLoss(
        lambda_dice=1.0, lambda_bce=1.0, lambda_focal=1.0, gamma=2.0,
    )(logits, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_dice_bce_focal_backward_routes_gradient() -> None:
    logits, target = _sample_logits_target()
    loss = DiceBCEFocalLoss()(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_dice_bce_focal_lambdas_are_linear() -> None:
    logits, target = _sample_logits_target()
    dice = DiceBCEFocalLoss(lambda_dice=1.0, lambda_bce=0.0, lambda_focal=0.0)(logits, target)
    bce = DiceBCEFocalLoss(lambda_dice=0.0, lambda_bce=1.0, lambda_focal=0.0)(logits, target)
    focal = DiceBCEFocalLoss(lambda_dice=0.0, lambda_bce=0.0, lambda_focal=1.0, gamma=2.0)(logits, target)
    full = DiceBCEFocalLoss(lambda_dice=1.0, lambda_bce=1.0, lambda_focal=1.0, gamma=2.0)(logits, target)
    assert torch.allclose(full, dice + bce + focal, atol=1e-5)


def test_dice_bce_focal_gamma_zero_collapses_focal_to_bce() -> None:
    logits, target = _sample_logits_target()
    bce = DiceBCEFocalLoss(lambda_dice=0.0, lambda_bce=1.0, lambda_focal=0.0)(logits, target)
    focal_g0 = DiceBCEFocalLoss(lambda_dice=0.0, lambda_bce=0.0, lambda_focal=1.0, gamma=0.0)(logits, target)
    assert torch.allclose(bce, focal_g0, atol=1e-5)


def test_stable_bce_on_probs_matches_torch_reference() -> None:
    torch.manual_seed(0)
    probs = torch.rand(8).clamp(0.05, 0.95)
    target = (torch.rand(8) > 0.5).float()
    ours = stable_bce_on_probs(probs, target)
    ref = -(target * probs.log() + (1 - target) * (1 - probs).log())
    assert torch.allclose(ours, ref, atol=1e-6)


def test_affinity_fg_loss_threads_sem_composite_lambdas() -> None:
    loss_fn = AffinityFGLoss(
        weight_sem={"weight": 1.0, "lambda_dice": 2.0, "lambda_bce": 0.5,
                    "lambda_focal": 0.1, "gamma": 1.0},
    )
    assert loss_fn._sem_loss.lambda_dice == pytest.approx(2.0)
    assert loss_fn._sem_loss.lambda_bce == pytest.approx(0.5)
    assert loss_fn._sem_loss.lambda_focal == pytest.approx(0.1)
    assert loss_fn._sem_loss.gamma == pytest.approx(1.0)
