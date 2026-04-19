"""
Semantic segmentation loss: CE + IoU + Dice.

Dimension-agnostic — works for both 2-D and 3-D inputs.

Supports two activation modes:

- **sigmoid**: independent per-channel binary CE (multi-label).
- **softmax**: mutually-exclusive CE via ``nn.CrossEntropyLoss``.

Uses MONAI's ``DiceLoss`` for both Dice and IoU (Jaccard) sub-losses.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.losses import DiceLoss


class SemanticLoss(nn.Module):
    """Semantic segmentation loss.

    ``loss = w_ce * CE + w_iou * (1 - SoftIoU) + w_dice * (1 - SoftDice)``

    Args:
        mode: ``"sigmoid"`` for multi-label or ``"softmax"`` for exclusive.
        weight_ce: Weight for the cross-entropy term.
        weight_iou: Weight for the IoU term (0 to disable).
        weight_dice: Weight for the Dice term (0 to disable).
        class_weights: Per-class weights for the CE loss.
        ignore_index: Label value to exclude from all loss terms.
        active_classes: Number of leading channels to include in the loss.
            ``None`` means use all channels.  Set to e.g. 2 when the model
            outputs 16 channels but only classes 0-1 have labels today;
            channels beyond ``active_classes`` receive zero gradient.
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

        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        if mode == "softmax":
            self.ce_loss = nn.CrossEntropyLoss(
                weight=cw, ignore_index=ignore_index, label_smoothing=label_smoothing,
            )
        else:
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=cw, reduction="none")

        use_sigmoid = (mode == "sigmoid")
        use_softmax = (mode == "softmax")
        self.dice_loss = DiceLoss(
            sigmoid=use_sigmoid, softmax=use_softmax,
            include_background=True, reduction="mean",
        )
        self.iou_loss = DiceLoss(
            sigmoid=use_sigmoid, softmax=use_softmax,
            include_background=True, jaccard=True, reduction="mean",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slice_active(self, logits, class_labels):
        """Restrict logits to the first ``active_classes`` channels.

        In softmax mode the label values are clamped to [0, active_classes).
        Returns (logits_sliced, class_labels_adjusted).
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

    def _make_target(self, logits, class_labels):
        """Build a channel-wise target matching logits shape for Dice/IoU.

        Returns (target [B, C, *], valid_mask [B, 1, *]).
        """
        C = logits.shape[1]

        if self.mode == "softmax":
            valid = class_labels != self.ignore_index
            safe = torch.where(valid, class_labels, 0)
            one_hot = rearrange(
                F.one_hot(safe.long(), C).float(), "b ... c -> b c ...",
            )
            valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")
            return one_hot * valid_mask, valid_mask

        # sigmoid mode
        if class_labels.dim() == logits.dim():
            return class_labels.float(), None

        valid = (class_labels != self.ignore_index)
        valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")

        if C == 1:
            target = rearrange((class_labels > 0).float(), "b ... -> b 1 ...")
            return target * valid_mask, valid_mask

        safe = class_labels.clone().long()
        neg = safe < 0
        safe[neg] = 0
        safe = safe.clamp(0, C - 1)
        target = rearrange(
            F.one_hot(safe, C).float(), "b ... c -> b c ...",
        )
        neg_mask = repeat(neg, "b ... -> b c ...", c=target.shape[1])
        target[neg_mask] = 0.0

        return target * valid_mask, valid_mask

    def _compute_ce(self, logits, class_labels):
        logits, class_labels = self._slice_active(logits, class_labels)
        if self.mode == "softmax":
            return self.ce_loss(logits, class_labels)

        target, valid_mask = self._make_target(logits, class_labels)
        per_pixel = self.ce_loss(logits, target)
        if valid_mask is not None:
            per_pixel = per_pixel * valid_mask
            n_valid = valid_mask.sum().clamp(min=1.0) * per_pixel.shape[1]
        else:
            n_valid = max(per_pixel.numel(), 1)
        return per_pixel.sum() / n_valid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, logits, class_labels) -> Dict[str, torch.Tensor]:
        dev = logits.device
        ce = self._compute_ce(logits, class_labels)

        need_iou = self.weight_iou > 0
        need_dice = self.weight_dice > 0

        if need_iou or need_dice:
            sliced_logits, sliced_labels = self._slice_active(logits, class_labels)
            target, _ = self._make_target(sliced_logits, sliced_labels)
            dice = self.dice_loss(sliced_logits, target) if need_dice else torch.tensor(0.0, device=dev)
            iou = self.iou_loss(sliced_logits, target) if need_iou else torch.tensor(0.0, device=dev)
        else:
            iou = torch.tensor(0.0, device=dev)
            dice = torch.tensor(0.0, device=dev)

        loss = self.weight_ce * ce + self.weight_iou * iou + self.weight_dice * dice
        return {"loss": loss, "ce": ce, "iou": iou, "dice": dice}
