"""Tests for the unified 32-channel TensorBoard panel set.

Exercises :func:`brainbow.callbacks.tensorboard.heads._log_predictions`
with a recording mock writer so we lock down the exact tag set the
ImageLogger emits for the unified head -- with and without the
precomputed skeleton-geometry GT fields threaded through.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from brainbow.callbacks.tensorboard.heads import _log_predictions
from brainbow.callbacks.tensorboard.tags import TagContext
from brainbow.losses import HEAD_CHANNELS, SIGMOID_SLICE


class _RecordingTB:
    """Minimal stand-in for a TB ``SummaryWriter``.

    Captures every ``add_images`` tag in :attr:`tags` so the test can
    assert which panels were emitted; ignores the actual image
    payloads.
    """

    def __init__(self) -> None:
        self.tags: List[str] = []
        self.payloads: Dict[str, torch.Tensor] = {}

    def add_images(
        self,
        tag: str,
        images: torch.Tensor,
        global_step: int = 0,
    ) -> None:
        self.tags.append(tag)
        self.payloads[tag] = images.detach().cpu()


def _make_head_pred(B: int = 2, D: int = 4, H: int = 8, W: int = 8) -> torch.Tensor:
    """Build a HEAD_CHANNELS-wide head tensor with the sigmoid slot
    actually in ``(0, 1)``."""
    torch.manual_seed(0)
    head = torch.randn(B, HEAD_CHANNELS, D, H, W) * 0.5
    head[:, SIGMOID_SLICE] = head[:, SIGMOID_SLICE].sigmoid()
    return head


def _make_labels(B: int = 2, D: int = 4, H: int = 8, W: int = 8) -> torch.Tensor:
    """Two non-overlapping cuboid instances per batch element."""
    labels = torch.zeros(B, D, H, W, dtype=torch.long)
    labels[:, :, :4, :4] = 1
    labels[:, :, 4:, 4:] = 2
    return labels


# Tags every successful call should emit, regardless of the GT
# pass-through.  Affinity directions are spelled out so the alphabetic
# TB sort lands in their physical pairing (see
# ``brainbow/callbacks/tensorboard/heads.py``).
_AFF_NAMES = (
    "01_t1", "02_b1", "03_u1", "04_d1", "05_l1", "06_r1",
    "07_t2", "08_b2", "09_u2", "10_d2", "11_l2", "12_r2",
)
_EXPECTED_PRED_TAGS = {
    "train/automatic/pred/raw",
    "train/automatic/pred/sem",
    "train/automatic/pred/skl",
    "train/automatic/pred/dir",
    "train/automatic/pred/cov",
    "train/automatic/pred/rad",
    "train/automatic/pred/avg/val",
    "train/automatic/pred/emb/_pca",
    *(f"train/automatic/pred/avg/aff/{n}" for n in _AFF_NAMES),
    *(f"train/automatic/pred/emb/aff/{n}" for n in _AFF_NAMES),
}
_EXPECTED_TRUE_TAGS = {
    "train/automatic/true/image",
    "train/automatic/true/label",
    "train/automatic/true/avg/val",
    *(f"train/automatic/true/aff/{n}" for n in _AFF_NAMES),
}
_EXPECTED_TRUE_GEOMETRY_TAGS = {
    "train/automatic/true/skl",
    "train/automatic/true/dir",
    "train/automatic/true/cov",
    "train/automatic/true/rad",
}


def _run_log_predictions(
    *,
    pass_gt_geometry: bool,
    geometry_style: str = "flow",
) -> _RecordingTB:
    """Invoke ``_log_predictions`` with a recording writer.

    Uses the ``flow`` renderer by default so the test stays
    matplotlib-free (the ``glyph`` renderer pulls in matplotlib at
    import time and the test environment may not have a usable
    backend).
    """
    B, D, H, W = 2, 4, 8, 8
    head_pred = _make_head_pred(B, D, H, W)
    labels_3d = _make_labels(B, D, H, W)
    # 2-D-slice versions (central slice along the depth axis).
    images_2d = torch.randn(B, 1, H, W).clamp(-1.0, 1.0)
    labels_2d = labels_3d[:, D // 2]

    extra: Dict[str, Any] = {}
    if pass_gt_geometry:
        extra["gt_skl_2d"] = (torch.rand(B, 1, H, W) > 0.9).float()
        extra["gt_dir_2d"] = torch.randn(B, 3, H, W)
        extra["gt_cov_2d"] = torch.randn(B, 6, H, W).abs()
        extra["gt_rad_2d"] = torch.rand(B, 1, H, W)

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
        clusterer=None,
        labels_3d=labels_3d,
        projection_algorithm="pca",
        projection_backend="auto",
        aff_emb_tau=1.0,
        aff_avg_tau=1.0,
        normalize_embeddings=False,
        wan_decoder_2d=None,
        geometry_style=geometry_style,
        **extra,
    )
    return tb


class TestLogPredictions:
    def test_emits_all_pred_panels_for_32ch_head(self) -> None:
        tb = _run_log_predictions(pass_gt_geometry=False)
        emitted = set(tb.tags)
        missing = _EXPECTED_PRED_TAGS - emitted
        assert not missing, f"missing prediction panels: {sorted(missing)}"

    def test_emits_baseline_true_panels(self) -> None:
        tb = _run_log_predictions(pass_gt_geometry=False)
        emitted = set(tb.tags)
        missing = _EXPECTED_TRUE_TAGS - emitted
        assert not missing, f"missing baseline true panels: {sorted(missing)}"

    def test_skips_gt_geometry_panels_when_targets_absent(self) -> None:
        """Without ``gt_*_2d`` kwargs, the new ``true/skl`` /
        ``true/dir`` / ``true/cov`` / ``true/rad`` panels must not be
        emitted -- the ImageLogger only threads them through when the
        datamodule's ``compute_geometry`` path populated them."""
        tb = _run_log_predictions(pass_gt_geometry=False)
        emitted = set(tb.tags)
        assert emitted.isdisjoint(_EXPECTED_TRUE_GEOMETRY_TAGS), (
            f"unexpected GT panels with no targets: "
            f"{sorted(emitted & _EXPECTED_TRUE_GEOMETRY_TAGS)}"
        )

    def test_emits_gt_geometry_panels_when_targets_passed(self) -> None:
        tb = _run_log_predictions(pass_gt_geometry=True)
        emitted = set(tb.tags)
        missing = _EXPECTED_TRUE_GEOMETRY_TAGS - emitted
        assert not missing, (
            f"missing GT geometry panels with targets passed: "
            f"{sorted(missing)}"
        )

    def test_rejects_wrong_head_channel_count(self) -> None:
        bad_head = torch.randn(2, HEAD_CHANNELS - 1, 4, 8, 8)
        labels_3d = _make_labels()
        tb = _RecordingTB()
        with pytest.raises(ValueError, match=str(HEAD_CHANNELS)):
            _log_predictions(
                tb,
                TagContext(stage="train", mode="automatic"),
                torch.randn(2, 1, 8, 8),
                labels_3d[:, 2],
                bad_head,
                spatial_dims=3,
                n=2,
                epoch=0,
                geometry_style="flow",
            )

    def test_pred_skl_panel_has_three_channels(self) -> None:
        """All ``pred/*`` and ``true/*`` panels are written as RGB
        images, so each payload must have channel dim 3."""
        tb = _run_log_predictions(pass_gt_geometry=True)
        for tag in (
            "train/automatic/pred/skl",
            "train/automatic/pred/rad",
            "train/automatic/true/skl",
            "train/automatic/true/rad",
        ):
            payload = tb.payloads[tag]
            assert payload.shape[1] == 3, (
                f"{tag} should be RGB; got channel dim {payload.shape[1]}"
            )
