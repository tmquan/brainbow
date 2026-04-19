"""
Semantic segmentation loss: CE + IoU + Dice.

Dimension-agnostic -- works for both 2-D and 3-D inputs.

Supports two activation modes:

- **sigmoid**: independent per-channel binary CE (multi-label).
- **softmax**: mutually-exclusive CE via ``nn.CrossEntropyLoss``.

Uses MONAI's ``DiceLoss`` for both Dice and IoU (Jaccard) sub-losses.

Channel layout (what ``prediction`` looks like)::

    prediction: [B, num_classes, *spatial]       class logits
    labels:     [B, *spatial]  (int ids)  OR     [B, num_classes, *spatial]  (float target)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.losses import DiceLoss


class SemanticLoss(nn.Module):
    """Weighted sum of CE + IoU + Dice on per-voxel class logits.

    ``loss = w_ce * CE + w_iou * (1 - SoftIoU) + w_dice * (1 - SoftDice)``

    Args:
        mode: ``"sigmoid"`` for multi-label or ``"softmax"`` for exclusive.
        weight_ce:   Weight for the cross-entropy term.
        weight_iou:  Weight for the IoU term (0 to disable).
        weight_dice: Weight for the Dice term (0 to disable).
        class_weights: Per-class weights for the CE loss.
        ignore_index:  Label value to exclude from all loss terms.
        active_classes: Number of leading channels to include in the loss.
            ``None`` means use all channels.  Set to e.g. 2 when the model
            outputs 16 channels but only classes 0-1 have labels today;
            channels beyond ``active_classes`` receive zero gradient.
        label_smoothing: Passed to ``nn.CrossEntropyLoss`` in softmax mode.
    """

    def __init__(
        self,
        mode: str = "sigmoid",
        weight_ce: float = 1.0,
        weight_iou: float = 0.0,
        weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        active_classes: Optional[int] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("sigmoid", "softmax"):
            raise ValueError(f"mode must be 'sigmoid' or 'softmax', got '{mode}'")

        self.mode = mode
        self.weight_ce = weight_ce
        self.weight_iou = weight_iou
        self.weight_dice = weight_dice
        self.ignore_index = ignore_index
        self.active_classes = active_classes

        cw = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None
        )
        if mode == "softmax":
            self.ce_loss = nn.CrossEntropyLoss(
                weight=cw, ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
        else:
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=cw, reduction="none")

        shared = dict(
            sigmoid=(mode == "sigmoid"),
            softmax=(mode == "softmax"),
            include_background=True,
            reduction="mean",
        )
        self.dice_loss = DiceLoss(**shared)
        self.iou_loss = DiceLoss(**shared, jaccard=True)

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _slice_active(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restrict logits to the first ``active_classes`` channels.

        In softmax mode label values are clamped to ``[0, active_classes)``
        (keeping ignore_index untouched).  In sigmoid mode the labels
        pass through unchanged; out-of-range ids get zeroed by
        :meth:`_build_target_onehot`.
        """
        if self.active_classes is None or self.active_classes >= logits.shape[1]:
            return logits, class_labels
        C = self.active_classes
        logits = logits[:, :C]
        if self.mode == "softmax":
            class_labels = class_labels.clone()
            valid = class_labels != self.ignore_index
            class_labels[valid] = class_labels[valid].clamp(0, C - 1)
        return logits, class_labels

    @torch.no_grad()
    def _build_target_onehot(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One-hot target matching ``logits`` shape, plus a valid mask.

        Returns:
            target:     ``[B, C, *spatial]`` float one-hot (or binary FG
                        mask when ``mode='sigmoid'`` and ``C == 1``).
            valid_mask: ``[B, 1, *spatial]`` float, or ``None`` if the
                        caller already has a dense float target.
        """
        C = logits.shape[1]

        # Softmax: straight one-hot from integer class ids, zeroing
        # ignore_index positions.
        if self.mode == "softmax":
            valid = class_labels != self.ignore_index
            safe = torch.where(valid, class_labels, torch.zeros_like(class_labels))
            one_hot = rearrange(
                F.one_hot(safe.long(), C).float(),
                "b ... c -> b c ...",
            )
            valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")
            return one_hot * valid_mask, valid_mask

        # Sigmoid: caller may already pass a dense [B, C, *] float target.
        if class_labels.dim() == logits.dim():
            return class_labels.float(), None

        valid = class_labels != self.ignore_index
        valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")

        # Binary foreground case -- most common for 1-class semantic heads.
        if C == 1:
            target = rearrange((class_labels > 0).float(), "b ... -> b 1 ...")
            return target * valid_mask, valid_mask

        # Multi-class sigmoid.  Negative / out-of-range ids get zeroed
        # across all channels (they contribute no positive gradient).
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
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Construct the dense one-hot target used by Dice / IoU.

        Returns a dict with::

            logits       : [B, C_active, *spatial]  (possibly sliced)
            class_labels : [B, *spatial]            (possibly clamped)
            target       : [B, C_active, *spatial]  dense one-hot
            valid_mask   : [B, 1,        *spatial]  or None
        """
        sl, cl = self._slice_active(logits, class_labels)
        tgt, valid = self._build_target_onehot(sl, cl)
        return {"logits": sl, "class_labels": cl, "target": tgt, "valid_mask": valid}

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
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy (softmax) or masked BCE (sigmoid).

        Operates on already-sliced inputs -- caller must have run
        :meth:`_slice_active` first.
        """
        if self.mode == "softmax":
            return self.ce_loss(logits, class_labels)

        target, valid_mask = self._build_target_onehot(logits, class_labels)
        per_voxel = self.ce_loss(logits, target)
        if valid_mask is not None:
            per_voxel = per_voxel * valid_mask
            # Denominator mirrors nn.BCEWithLogitsLoss(reduction="mean") but
            # excludes ignore-index voxels.  ``* C`` accounts for the
            # per-voxel sum over channels that BCE reports.
            denom = valid_mask.sum().clamp(min=1.0) * per_voxel.shape[1]
        else:
            denom = max(per_voxel.numel(), 1)
        return per_voxel.sum() / denom

    def _compute_loss_dice(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.dice_loss(logits, target)

    def _compute_loss_iou(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.iou_loss(logits, target)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=logits.device)
        ctx = self.build_target(logits, class_labels)

        ce = self._compute_loss_ce(ctx["logits"], ctx["class_labels"])
        dice = (
            self._compute_loss_dice(ctx["logits"], ctx["target"])
            if self.weight_dice > 0 else zero
        )
        iou = (
            self._compute_loss_iou(ctx["logits"], ctx["target"])
            if self.weight_iou > 0 else zero
        )

        total = self.weight_ce * ce + self.weight_iou * iou + self.weight_dice * dice
        return {"loss": total, "ce": ce, "iou": iou, "dice": dice}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mode='{self.mode}', "
            f"active_classes={self.active_classes}, "
            f"weight_ce={self.weight_ce}, weight_iou={self.weight_iou}, "
            f"weight_dice={self.weight_dice})"
        )
