"""
Composite Dice + BCE + Focal loss for the (sem, skl, aff_*) heads.

This is the supervisor for every classification-style head in the
unified 32-channel layout owned by :mod:`brainbow.losses._common`:

* ``sem`` (ch 1, post-sigmoid) and ``skl`` (ch 2, post-sigmoid)
  receive a sigmoid in :func:`~brainbow.losses.apply_head_activations`
  before the loss is called.
* ``aff_emb`` and ``aff_avg`` receive their 12-channel face-affinity
  scores from :func:`~brainbow.losses.soft_aff_from_field`, whose
  kernel ``exp(-tau * sum |diff|)`` already lives in ``(0, 1]``.

All four arrive at the loss as **probabilities**, which is why this
module composes Dice (the imbalance-robust term) with a numerically-
stable BCE-on-probs and a focal-loss-on-probs rather than calling
MONAI's :class:`monai.losses.DiceCELoss` / :class:`monai.losses.FocalLoss`
directly -- those use ``BCEWithLogitsLoss`` / softmax-on-logits paths
internally and would double-activate a probability input.

The composite total is::

    L = lambda_dice  * Dice(p, t)
      + lambda_bce   * BCE(p, t)
      + lambda_focal * Focal(p, t; gamma)

where each lambda defaults to ``1.0`` so the composite has all three
terms active out of the box.  Set any lambda to ``0`` to disable that
term -- e.g. ``lambda_bce=0, lambda_focal=0`` recovers the prior
Dice-only behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.losses import DiceLoss

from brainbow.losses._common import stable_bce_on_probs


class DiceBCEFocalLoss(nn.Module):
    """Probability-input composite of Dice + BCE + Focal.

    Args:
        lambda_dice: Scalar multiplier on the Dice term.
        lambda_bce:  Scalar multiplier on the per-voxel BCE term.
        lambda_focal: Scalar multiplier on the focal-loss term.
        gamma: Focal focusing parameter ``gamma`` -- ``gamma=0`` makes
            focal reduce to plain BCE (so the focal channel becomes
            redundant with the BCE channel).  Default ``2.0`` matches
            the canonical Lin et al. setting.
        include_background: Forwarded to MONAI's :class:`DiceLoss`.
            For binary single-channel heads this is essentially a
            no-op; kept here so the constructor signature stays
            transparent.
        smooth_nr: Numerator smoothing in MONAI's Dice formula
            (forwarded to :class:`DiceLoss`).  Defaults to ``1e-5`` to
            match the aff Dice that this loss replaces.
        smooth_dr: Denominator smoothing in MONAI's Dice formula.
        eps: Clamp used by the BCE / Focal log math for fp32 stability
            under bf16-mixed autocast (forwarded to
            :func:`stable_bce_on_probs`).
        batch: Whether MONAI's Dice reduces per-batch (``True``) or
            per-sample (``False``).  ``True`` matches the previous
            ``_sem_dice`` / ``_skl_dice`` / ``_aff_dice`` config.

    Shapes:
        Both ``probs`` and ``target`` are ``[B, C, *spatial]``;
        ``probs`` is expected to be already in ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_focal: float = 1.0,
        gamma: float = 2.0,
        include_background: bool = True,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        eps: float = 1e-7,
        batch: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.lambda_focal = float(lambda_focal)
        self.gamma = float(gamma)
        self.eps = float(eps)

        # MONAI's DiceLoss with sigmoid=False / softmax=False is the
        # plain probability-input Dice -- no extra activation, no
        # cross-entropy branch.  We keep ``include_background=True``
        # since the binary single-channel heads only have one channel
        # and excluding it would zero out the loss.
        self._dice = DiceLoss(
            sigmoid=False,
            softmax=False,
            include_background=include_background,
            reduction="mean",
            batch=batch,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

    @staticmethod
    def _zero_like(t: torch.Tensor) -> torch.Tensor:
        return t.new_zeros(())

    def _focal_on_probs(
        self, probs: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-voxel focal loss on ``[0, 1]`` probabilities.

        Standard form
        ``focal = (1 - p_t)^gamma * (-log p_t)``
        with ``p_t = t * p + (1 - t) * (1 - p)`` (the probability the
        model assigns to the true class at this voxel).  ``gamma=0``
        reduces to ordinary BCE.
        """
        p = probs.float().clamp(self.eps, 1.0 - self.eps)
        t = target.float()
        p_t = t * p + (1.0 - t) * (1.0 - p)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        # Stable per-voxel BCE expressed in terms of ``p_t`` so the
        # `log` argument is the true-class probability (avoids a
        # branch on ``t`` and keeps everything in fp32).
        bce = -p_t.log()
        return focal_weight * bce

    def forward(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``lambda_bce * BCE + lambda_dice * Dice + lambda_focal * Focal``."""
        total = self._zero_like(probs)

        if self.lambda_dice > 0:
            total = total + self.lambda_dice * self._dice(probs, target)

        if self.lambda_bce > 0:
            per_voxel = stable_bce_on_probs(probs, target, eps=self.eps)
            total = total + self.lambda_bce * per_voxel.mean()

        if self.lambda_focal > 0:
            per_voxel = self._focal_on_probs(probs, target)
            total = total + self.lambda_focal * per_voxel.mean()

        return total

    def extra_repr(self) -> str:
        return (
            f"lambda_dice={self.lambda_dice}, "
            f"lambda_bce={self.lambda_bce}, "
            f"lambda_focal={self.lambda_focal}, "
            f"gamma={self.gamma}"
        )
