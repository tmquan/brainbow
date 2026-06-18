"""
Composite Dice + BCE + Focal loss on **logit** inputs, with optional
class rebalancing.

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

**Class rebalancing (mirrors the affinity head).**  The foreground class
is typically the majority (``min_foreground`` crops, plus the thin
background "gaps" of a boundary-eroded ``sem_label``), which biases the
head toward predicting foreground everywhere.  The same knobs the
affinity head uses are available here:

* ``class_balance`` -- ``None`` (no rebalancing), ``"auto"`` (per-batch
  inverse-frequency positive/negative weighting, clamped by
  ``class_balance_clip``), or a fixed ``float`` multiplier on the
  positive (foreground) class;
* ``focal_alpha`` -- optional alpha-balanced focal up-weighting the rare
  *negative* (background) class;
* ``dice_two_sided`` -- also score the complementary background-class
  soft-Dice (averaged with the foreground Dice).

All of these default to off, so the loss is byte-identical to the prior
Dice + BCE + Focal composite when they are unset.  The BCE / focal terms
are reduced as a masked, weighted mean (``sum(term * w) / sum(w)``) and
the Dice term is a global weighted soft-Dice -- with unit weights and no
mask this equals the previous ``mean`` reduction and MONAI ``DiceLoss``
(``batch=True``) exactly.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCEFocalLoss(nn.Module):
    """Logit-input composite of Dice + BCE + Focal with optional rebalancing.

    Args:
        lambda_dice: Scalar multiplier on the Dice term.
        lambda_bce:  Scalar multiplier on the per-voxel BCE term.
        lambda_focal: Scalar multiplier on the focal-loss term.
        gamma: Focal focusing parameter ``gamma`` -- ``gamma=0`` makes
            focal reduce to plain BCE (so the focal channel becomes
            redundant with the BCE channel).  Default ``2.0`` matches
            the canonical Lin et al. setting.
        class_balance: Positive/negative rebalancing.  ``None`` (off,
            default), ``"auto"`` (per-batch inverse-frequency weighting
            of foreground vs background, clamped by ``class_balance_clip``),
            or a ``float`` multiplier on the positive (foreground) class.
            Must already be normalised (the caller parses ``"none"`` etc.).
        class_balance_clip: Upper clamp on the ``"auto"`` weights.
        focal_alpha: Optional alpha-balanced focal weight on the rare
            *negative* (background) class, matching the affinity head's
            ``focal_alpha`` convention.  ``None`` disables it.
        dice_two_sided: When ``True`` also score the complementary
            background-class soft-Dice and average it with the foreground
            Dice (direct region supervision on background / boundary).
        smooth_nr: Numerator smoothing in the soft-Dice formula.
        smooth_dr: Denominator smoothing in the soft-Dice formula.
        eps: Clamp used by the focal log math (and the ``"auto"`` weight
            frequency) for fp32 stability under bf16-mixed autocast.

    Shapes:
        ``logits`` / ``target`` (and the optional ``mask``) are
        ``[B, C, *spatial]``; ``logits`` are raw (pre-sigmoid) outputs.
    """

    def __init__(
        self,
        *,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_focal: float = 1.0,
        gamma: float = 2.0,
        class_balance: Union[None, str, float] = None,
        class_balance_clip: float = 10.0,
        focal_alpha: Optional[float] = None,
        dice_two_sided: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.lambda_focal = float(lambda_focal)
        self.gamma = float(gamma)
        # ``class_balance`` is expected pre-normalised to None | "auto" |
        # float by the caller (see ``_parse_class_balance``).
        self.class_balance = class_balance
        self.class_balance_clip = float(class_balance_clip)
        self.focal_alpha = float(focal_alpha) if focal_alpha is not None else None
        self.dice_two_sided = bool(dice_two_sided)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.eps = float(eps)

    def _class_weight(
        self, t: torch.Tensor, vm: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Per-voxel positive/negative class weight, or ``None`` if off."""
        cb = self.class_balance
        if cb is None:
            return None
        if cb == "auto":
            tot = vm.sum().clamp_min(1.0)
            f = ((t * vm).sum() / tot).clamp(self.eps, 1.0 - self.eps)
            w_pos = (0.5 / f).clamp(max=self.class_balance_clip)
            w_neg = (0.5 / (1.0 - f)).clamp(max=self.class_balance_clip)
            return t * w_pos + (1.0 - t) * w_neg
        # Fixed multiplier on the positive (foreground) class.
        return t * float(cb) + (1.0 - t)

    @staticmethod
    def _zero_like(t: torch.Tensor) -> torch.Tensor:
        return t.new_zeros(())

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ``lambda_bce * BCE + lambda_dice * Dice + lambda_focal * Focal``.

        ``logits`` are raw (pre-sigmoid) head outputs.  ``mask`` is an
        optional ``[B, C, *spatial]`` validity mask (1 = supervised); the
        BCE / focal terms reduce as ``sum(term * w) / sum(w)`` and the
        Dice term is a global weighted soft-Dice over the same weights
        ``w = mask * class_weight``.
        """
        z = logits.float()
        t = target.float()
        vm = torch.ones_like(z) if mask is None else mask.float()
        cw = self._class_weight(t, vm)
        wm = vm if cw is None else vm * cw
        denom = wm.sum().clamp_min(1.0)

        need_probs = self.lambda_dice > 0 or self.lambda_focal > 0
        p = z.sigmoid() if need_probs else None

        total = self._zero_like(z)

        if self.lambda_bce > 0:
            bce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
            total = total + self.lambda_bce * (bce * wm).sum() / denom

        if self.lambda_focal > 0:
            pc = p.clamp(self.eps, 1.0 - self.eps)
            p_t = t * pc + (1.0 - t) * (1.0 - pc)
            focal = (1.0 - p_t).pow(self.gamma) * (-p_t.log())
            if self.focal_alpha is not None:
                a = self.focal_alpha
                focal = focal * ((1.0 - t) * a + t * (1.0 - a))
            total = total + self.lambda_focal * (focal * wm).sum() / denom

        if self.lambda_dice > 0:
            pm = p * wm
            inter = (pm * t).sum()
            pos_dice = 1.0 - (2.0 * inter + self.smooth_nr) / (
                pm.sum() + (t * wm).sum() + self.smooth_dr
            )
            if self.dice_two_sided:
                qm = (1.0 - p) * wm
                s = 1.0 - t
                neg_dice = 1.0 - (2.0 * (qm * s).sum() + self.smooth_nr) / (
                    qm.sum() + (s * wm).sum() + self.smooth_dr
                )
                dice = 0.5 * (pos_dice + neg_dice)
            else:
                dice = pos_dice
            total = total + self.lambda_dice * dice

        return total

    def extra_repr(self) -> str:
        return (
            f"lambda_dice={self.lambda_dice}, "
            f"lambda_bce={self.lambda_bce}, "
            f"lambda_focal={self.lambda_focal}, "
            f"gamma={self.gamma}, "
            f"class_balance={self.class_balance}, "
            f"dice_two_sided={self.dice_two_sided}"
        )
