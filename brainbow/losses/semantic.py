"""
Semantic segmentation loss: CE + IoU + Dice.

Dimension-agnostic -- works for both 2-D and 3-D inputs.

This loss is **sigmoid-only**: every channel is treated as an independent
binary indicator (multi-label).  If you need softmax CE for a future
multi-class head, fork :class:`SemanticLoss` rather than reviving the
dead branch.

Activation contract
-------------------

The model wrapper applies ``torch.sigmoid`` to the semantic head **before**
this loss (and the TensorBoard callbacks) see it -- so ``prediction``
arriving at :meth:`forward` is already in ``[0, 1]``.  BCE uses
:func:`stable_bce_on_probs` (probability inputs, not logits, with
fp32 log math under bf16 autocast) and the Dice / IoU sub-losses are
:class:`monai.losses.DiceLoss` instances configured with
``sigmoid=False`` so they do **not** re-apply any activation.

Channel layout (what ``prediction`` looks like)::

    prediction: [B, num_classes, *spatial]   per-channel sigmoid probabilities
    labels:     [B, *spatial]  (int ids)
                or [B, num_classes, *spatial]  (dense float target)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.losses import DiceLoss

from brainbow.losses._common import stable_bce_on_probs


class SemanticLoss(nn.Module):
    """Weighted sum of CE + IoU + Dice on per-voxel sigmoid probabilities.

    ``loss = w_ce * BCE + w_iou * (1 - SoftIoU) + w_dice * (1 - SoftDice)``

    All three sub-losses operate on **probabilities** (``[0, 1]``) -- the
    sigmoid is applied once, in the model wrapper, before anything in this
    module runs.  Background indices (``ignore_index``) are masked out of
    BCE; Dice / IoU consume the full one-hot target produced by
    :meth:`build_target`.

    Args:
        weight_ce:   Weight for the cross-entropy (BCE) term.
        weight_iou:  Weight for the IoU term (0 to disable).
        weight_dice: Weight for the Dice term (0 to disable).
        class_weights: Per-class ``pos_weight`` for the BCE loss
            (broadcasted over the spatial dims).
        ignore_index: Label value to exclude from all loss terms.
        active_classes: Number of leading channels to include in the loss.
            ``None`` means use all channels.  Set to e.g. 2 when the model
            outputs 16 channels but only classes 0-1 have labels today;
            channels beyond ``active_classes`` receive zero gradient.
    """

    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_iou: float = 0.0,
        weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        active_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.weight_ce = float(weight_ce)
        self.weight_iou = float(weight_iou)
        self.weight_dice = float(weight_dice)
        self.ignore_index = ignore_index
        self.active_classes = active_classes
        self.class_weights = (
            list(map(float, class_weights)) if class_weights is not None else None
        )

        # ``class_weights`` is plumbed into BCE as a per-class multiplicative
        # weight on the positive (target == 1) term.  Reshape to broadcast
        # over the spatial dims of the prediction tensor, registered as a
        # buffer so device moves carry it along.
        if self.class_weights is not None:
            cw = torch.tensor(self.class_weights, dtype=torch.float32)
            self.register_buffer("_pos_weight", cw, persistent=False)
        else:
            self._pos_weight = None

        # Sigmoid is applied externally (in the model wrapper); both sub-
        # losses below therefore consume probabilities directly.
        self.dice_loss = DiceLoss(
            sigmoid=False, softmax=False,
            include_background=True, reduction="mean",
        )
        self.iou_loss = DiceLoss(
            sigmoid=False, softmax=False,
            include_background=True, reduction="mean", jaccard=True,
        )

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _slice_active(
        self,
        probs: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restrict ``probs`` to the first ``active_classes`` channels.

        Labels pass through unchanged; out-of-range ids get zeroed by
        :meth:`_build_target_onehot`.
        """
        if self.active_classes is None or self.active_classes >= probs.shape[1]:
            return probs, class_labels
        return probs[:, :self.active_classes], class_labels

    @torch.no_grad()
    def _build_target_onehot(
        self,
        probs: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One-hot target matching ``probs`` shape, plus a valid mask.

        Returns:
            target:     ``[B, C, *spatial]`` float one-hot (or binary FG
                        mask when ``C == 1``).
            valid_mask: ``[B, 1, *spatial]`` float, or ``None`` if the
                        caller already passed a dense float target.
        """
        C = probs.shape[1]

        # Caller may already have passed a dense [B, C, *] float target.
        if class_labels.dim() == probs.dim():
            return class_labels.float(), None

        valid = class_labels != self.ignore_index
        valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")

        # Binary foreground case -- most common for 1-class semantic heads.
        if C == 1:
            target = rearrange((class_labels > 0).float(), "b ... -> b 1 ...")
            return target * valid_mask, valid_mask

        # Multi-class sigmoid (multi-label).  Negative / out-of-range ids
        # get zeroed across all channels (they contribute no positive
        # gradient).
        safe = class_labels.clone().long()
        neg = safe < 0
        safe[neg] = 0
        safe = safe.clamp(0, C - 1)
        target = rearrange(F.one_hot(safe, C).float(), "b ... c -> b c ...")
        neg_br = repeat(neg, "b ... -> b c ...", c=target.shape[1])
        target[neg_br] = 0.0
        return target * valid_mask, valid_mask

    @torch.no_grad()
    def build_target(
        self,
        probs: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Construct the dense one-hot target used by Dice / IoU.

        Returns a dict with::

            probs        : [B, C_active, *spatial]  (possibly sliced)
            class_labels : [B, *spatial]
            target       : [B, C_active, *spatial]  dense one-hot
            valid_mask   : [B, 1,        *spatial]  or None
        """
        sl, cl = self._slice_active(probs, class_labels)
        tgt, valid = self._build_target_onehot(sl, cl)
        return {"probs": sl, "class_labels": cl, "target": tgt, "valid_mask": valid}

    # ------------------------------------------------------------------
    # Per-voxel weights
    # ------------------------------------------------------------------

    def compute_weights(self, labels: torch.Tensor) -> None:
        """SemanticLoss does not use per-voxel weights."""
        return None

    # ------------------------------------------------------------------
    # Sub-losses
    # ------------------------------------------------------------------

    def _compute_loss_ce(
        self,
        probs: torch.Tensor,
        class_labels: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked binary cross-entropy on per-voxel probabilities.

        Operates on already-sliced inputs -- caller must have run
        :meth:`_slice_active` first.  Computes BCE on probabilities (the
        model wrapper has already applied sigmoid) and clamps the input
        to ``[eps, 1 - eps]`` so ``log(p)`` stays finite at the
        boundaries.

        Numerical note: under ``bf16-mixed`` autocast, ``1 - p`` rounds
        to exactly ``0`` whenever ``p > ~0.992`` (bf16 has ~3 decimal
        digits of mantissa near 1), which would make ``log(1 - p)``
        return ``-inf`` and nuke the whole step as NaN.  We therefore
        do the log math in fp32 -- the upcast happens on the clamped
        probabilities, so the clamp actually has teeth and the loss
        stays bounded by ``log(eps)`` instead of blowing up.

        Args:
            probs:        Already-sliced probability tensor.
            class_labels: Class-id labels (or pre-built dense target).
            target:       Optional precomputed dense one-hot target
                from :meth:`build_target` -- pass it through and we
                skip the rebuild (``forward`` already calls
                ``build_target`` once for Dice / IoU).
            valid_mask:   Companion ``valid_mask`` from
                :meth:`build_target`.  Pass alongside ``target``.
        """
        if target is None:
            target, valid_mask = self._build_target_onehot(probs, class_labels)

        if self._pos_weight is not None:
            shape = [1, probs.shape[1]] + [1] * (probs.dim() - 2)
            pw = self._pos_weight.view(*shape)
        else:
            pw = None
        per_voxel = stable_bce_on_probs(probs, target, pos_weight=pw)

        if valid_mask is not None:
            per_voxel = per_voxel * valid_mask.to(per_voxel.dtype)
            # Denominator mirrors nn.BCELoss(reduction="mean") but excludes
            # ignore-index voxels.  ``* C`` accounts for the per-voxel sum
            # over channels that BCE reports.
            denom = valid_mask.sum().clamp(min=1.0) * per_voxel.shape[1]
        else:
            denom = max(per_voxel.numel(), 1)
        return per_voxel.sum() / denom

    def _compute_loss_dice(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.dice_loss(probs, target)

    def _compute_loss_iou(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.iou_loss(probs, target)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        probs: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute CE + IoU + Dice on per-voxel sigmoid probabilities.

        Args:
            probs:        ``[B, C, *spatial]`` model output, **already**
                          passed through ``sigmoid`` by the wrapper.
            class_labels: ``[B, *spatial]`` integer ids OR
                          ``[B, C, *spatial]`` dense float target.
        """
        zero = torch.zeros((), device=probs.device)
        ctx = self.build_target(probs, class_labels)

        # ``ctx["target"]`` is the dense one-hot built once here; we
        # forward it into BCE so the per-voxel target tensor is built
        # exactly once per step (instead of twice -- once here and
        # once inside ``_compute_loss_ce``'s old fallback).
        ce = self._compute_loss_ce(
            ctx["probs"], ctx["class_labels"],
            target=ctx["target"], valid_mask=ctx["valid_mask"],
        )
        dice = (
            self._compute_loss_dice(ctx["probs"], ctx["target"])
            if self.weight_dice > 0 else zero
        )
        iou = (
            self._compute_loss_iou(ctx["probs"], ctx["target"])
            if self.weight_iou > 0 else zero
        )

        total = self.weight_ce * ce + self.weight_iou * iou + self.weight_dice * dice
        return {"loss": total, "ce": ce, "iou": iou, "dice": dice}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"active_classes={self.active_classes}, "
            f"weight_ce={self.weight_ce}, "
            f"weight_iou={self.weight_iou}, "
            f"weight_dice={self.weight_dice}, "
            f"class_weights={self.class_weights}, "
            f"ignore_index={self.ignore_index})"
        )
