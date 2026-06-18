"""Tests for the affinity + sem + raw TensorBoard panel set.

Exercises :func:`brainbow.callbacks.tensorboard.heads._log_predictions`
with a recording mock writer so we lock down the exact tag set the
ImageLogger emits for the affinity head -- with and without the Mutex
Watershed instance segmentation threaded through.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest
import torch

from brainbow.callbacks.tensorboard.heads import _log_predictions, aff_panel_indices
from brainbow.callbacks.tensorboard.tags import TagContext
from brainbow.losses import AFF_NAMES, HEAD_CHANNELS, N_AFF, N_PULL


class _RecordingTB:
    """Minimal stand-in for a TB ``SummaryWriter`` that records tags."""

    def __init__(self) -> None:
        self.tags: List[str] = []
        self.payloads: Dict[str, torch.Tensor] = {}

    def add_images(self, tag: str, images: torch.Tensor, global_step: int = 0) -> None:
        self.tags.append(tag)
        self.payloads[tag] = images.detach().cpu()


# H/W >= 28 and D >= 5 so the longest affinity offset (in-plane 27,
# axial 4) stays within the volume.
_D, _H, _W = 6, 32, 32


def _make_head_pred(B: int = 2, D: int = _D, H: int = _H, W: int = _W) -> torch.Tensor:
    """Raw-logit head (no activation): the panels sigmoid aff / sem and
    rescale the linear raw channel for display themselves."""
    torch.manual_seed(0)
    return torch.randn(B, HEAD_CHANNELS, D, H, W) * 2.0


def _make_labels(B: int = 2, D: int = _D, H: int = _H, W: int = _W) -> torch.Tensor:
    labels = torch.zeros(B, D, H, W, dtype=torch.long)
    labels[:, :, :16, :16] = 1
    labels[:, :, 16:, 16:] = 2
    return labels


# By default all N_AFF offsets are visualised.
_AFF_PANELS = [AFF_NAMES[i] for i in aff_panel_indices(N_AFF, N_PULL)]
# Affinity panels live under a single ``aff/`` group (so the core
# image/label/sem/raw panels stay clustered); core panels keep true/pred.
_EXPECTED_PRED_TAGS = {
    "train/automatic/pred/sem",
    "train/automatic/pred/raw",
    *(f"train/automatic/aff/pred/{n}" for n in _AFF_PANELS),
}
_EXPECTED_TRUE_TAGS = {
    "train/automatic/true/image",
    "train/automatic/true/label",
    "train/automatic/true/sem",
    *(f"train/automatic/aff/true/{n}" for n in _AFF_PANELS),
}
_SEG_TAGS = {
    "train/automatic/pred/label/pre",
    "train/automatic/pred/label/mul",
}


def _run(seg_pred_2d: Optional[torch.Tensor] = None) -> _RecordingTB:
    B, D, H, W = 2, _D, _H, _W
    head_pred = _make_head_pred(B, D, H, W)
    labels_3d = _make_labels(B, D, H, W)
    images_2d = torch.randn(B, 1, H, W).clamp(-1.0, 1.0)
    labels_2d = labels_3d[:, D // 2]

    tb = _RecordingTB()
    _log_predictions(
        tb,
        TagContext(stage="train", mode="automatic"),
        images_2d,
        labels_2d,
        head_pred,
        spatial_dims=3,
        n=B,
        epoch=0,
        labels_3d=labels_3d,
        seg_pred_2d=seg_pred_2d,
    )
    return tb


class TestLogPredictions:
    def test_emits_all_pred_panels(self) -> None:
        emitted = set(_run().tags)
        missing = _EXPECTED_PRED_TAGS - emitted
        assert not missing, f"missing prediction panels: {sorted(missing)}"

    def test_emits_true_panels(self) -> None:
        emitted = set(_run().tags)
        missing = _EXPECTED_TRUE_TAGS - emitted
        assert not missing, f"missing true panels: {sorted(missing)}"

    def test_all_affinity_offsets_are_shown(self) -> None:
        emitted = set(_run().tags)
        assert sum(t.startswith("train/automatic/aff/pred/") for t in emitted) == N_AFF
        assert sum(t.startswith("train/automatic/aff/true/") for t in emitted) == N_AFF

    def test_no_seg_panels_without_seg_input(self) -> None:
        emitted = set(_run(seg_pred_2d=None).tags)
        assert emitted.isdisjoint(_SEG_TAGS)

    def test_seg_panels_emitted_when_seg_passed(self) -> None:
        seg = _make_labels()[:, 2]  # [B, H, W] central-slice label map
        emitted = set(_run(seg_pred_2d=seg).tags)
        assert _SEG_TAGS.issubset(emitted)

    def test_rejects_wrong_head_channel_count(self) -> None:
        bad_head = torch.randn(2, HEAD_CHANNELS - 1, 4, 8, 8)
        labels_3d = _make_labels()
        with pytest.raises(ValueError, match=str(HEAD_CHANNELS)):
            _log_predictions(
                _RecordingTB(),
                TagContext(stage="train", mode="automatic"),
                torch.randn(2, 1, 8, 8),
                labels_3d[:, 2],
                bad_head,
                spatial_dims=3,
                n=2,
                epoch=0,
            )

    def test_panels_are_rgb(self) -> None:
        tb = _run(seg_pred_2d=_make_labels()[:, 2])
        for tag in (
            "train/automatic/pred/sem",
            "train/automatic/pred/raw",
            f"train/automatic/aff/pred/{_AFF_PANELS[0]}",
            "train/automatic/pred/label/pre",
        ):
            assert tb.payloads[tag].shape[1] == 3, f"{tag} should be RGB"
