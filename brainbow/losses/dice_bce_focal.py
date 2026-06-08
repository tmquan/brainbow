"""
Composite Dice + BCE + Focal loss on **logit** inputs.

Used by :class:`brainbow.losses.AffinityFGLoss` to supervise the
foreground (``sem``) head.  The head emits raw logits (no activation in
``forward``), so this loss takes the logits directly:

* the BCE term uses :func:`torch.nn.functional.binary_cross_entropy_with_logits`
  (the log-sum-exp-stable logit form);
* the Dice and focal terms operate on ``sigmoid(logits)`` (computed once
  internally), since both are naturally defined on probabilities.

The composite total is::

    L = lambda_dice  * Dice(sigmoid(z), t)
      + lambda_bce   * BCEWithLogits(z, t)
      + lambda_focal * Focal(sigmoid(z), t; gamma)

where each lambda defaults to ``1.0`` so the composite has all three
terms active out of the box.  Set any lambda to ``0`` to disable that
term -- e.g. ``lambda_bce=0, lambda_focal=0`` recovers the prior
Dice-only behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class DiceBCEFocalLoss(nn.Module):
    """Logit-input composite of Dice + BCE + Focal.

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
        eps: Clamp used by the focal log math for fp32 stability under
            bf16-mixed autocast.
        batch: Whether MONAI's Dice reduces per-batch (``True``) or
            per-sample (``False``).  ``True`` is the setting used by
            ``AffinityFGLoss`` for the ``sem`` head.

    Shapes:
        Both ``logits`` and ``target`` are ``[B, C, *spatial]``;
        ``logits`` are raw (pre-sigmoid) head outputs.
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
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``lambda_bce * BCE + lambda_dice * Dice + lambda_focal * Focal``.

        ``logits`` are raw (pre-sigmoid) head outputs.  The BCE term uses
        the logit-stable ``binary_cross_entropy_with_logits``; the Dice
        and focal terms use ``sigmoid(logits)`` (computed once).
        """
        total = self._zero_like(logits)

        # Sigmoid once for the probability-defined terms (Dice / focal).
        need_probs = self.lambda_dice > 0 or self.lambda_focal > 0
        probs = logits.sigmoid() if need_probs else None

        if self.lambda_dice > 0:
            total = total + self.lambda_dice * self._dice(probs, target)

        if self.lambda_bce > 0:
            per_voxel = F.binary_cross_entropy_with_logits(
                logits.float(), target.float(), reduction="none",
            )
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
