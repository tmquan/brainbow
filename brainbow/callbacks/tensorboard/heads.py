"""TensorBoard panels for the affinity + sem + raw head.

Emits the ``true/*`` ground-truth panels and the ``pred/*`` prediction
panels that mirror the loss scalar paths.  The head layout is owned by
:mod:`brainbow.losses._common`:

    aff: per-offset affinity logits (sigmoided for display; a curated
         subset is shown)
    sem: foreground / boundary logit (sigmoided for display)
    raw: linear reconstruction of the input EM intensity (target in
         [-1, 1]; rescaled to [0, 1] for display)

The instance segmentation (``pred/label``) is the Mutex Watershed
agglomeration of the predicted affinities, computed by the caller (see
:mod:`brainbow.callbacks.tensorboard.image_logger`) on the full 3-D head
and passed in as a central-slice label map.
"""

from typing import Any, List, Optional, Sequence

import torch
from einops import rearrange, reduce, repeat

from brainbow.callbacks.tensorboard.tags import TagContext
from brainbow.callbacks.tensorboard.viz import _label_to_rgb, _normalise, _to_2d
from brainbow.losses import (
    AFFINITY_OFFSETS,
    N_PULL,
    affinity_target_from_offsets,
    offset_names,
    slice_head,
)


def aff_panel_indices(
    n_offsets: int,
    n_pull: int,
    max_push: Optional[int] = None,
) -> List[int]:
    """Affinity channels to visualise.

    By default every offset is shown (``max_push=None`` -> all
    ``n_offsets`` channels: the pull nearest-neighbours followed by
    every long-range push offset).  Pass an integer ``max_push``
    to instead show all pull offsets plus that many evenly-spaced
    push ones.
    """
    if max_push is None:
        return list(range(n_offsets))
    idxs = list(range(min(n_pull, n_offsets)))
    push = list(range(n_pull, n_offsets))
    if push and max_push > 0:
        step = max(1, len(push) // max_push)
        idxs += push[::step][:max_push]
    return idxs


def _add_aff_panels(
    tb: Any,
    head: TagContext,
    aff_3d: torch.Tensor,
    indices: Sequence[int],
    *,
    names: Sequence[str],
    mask_2d: torch.Tensor,
    epoch: int,
    tag_prefix: str,
) -> None:
    """Central-slice affinity panels (a curated channel subset)."""
    aff_2d = _to_2d(aff_3d).clamp(0.0, 1.0)
    for k in indices:
        panel = repeat(aff_2d[:, k:k + 1] * mask_2d, "b 1 h w -> b 3 h w")
        tb.add_images(
            head.tag(f"{tag_prefix}/{names[k]}"), panel, global_step=epoch,
        )


def _log_predictions(
    tb: Any,
    ctx: TagContext,
    images: torch.Tensor,
    labels: torch.Tensor,
    head_pred: torch.Tensor,
    spatial_dims: int,
    n: int,
    epoch: int,
    *,
    offsets: Sequence[Sequence[int]] = AFFINITY_OFFSETS,
    n_pull: int = N_PULL,
    labels_3d: Optional[torch.Tensor] = None,
    sem_labels: Optional[torch.Tensor] = None,
    seg_pred_2d: Optional[torch.Tensor] = None,
    wan_decoder_2d: Optional[torch.Tensor] = None,
) -> None:
    """Log the affinity + sem + raw panels.

    Tags (under ``{stage}/{mode}/``):

    * ``true/image``, ``true/label``
    * ``true/sem``  -- ground-truth foreground target the sem head is trained
      on (``sem_labels > 0`` when a boundary-eroded ``sem_label`` is supplied,
      else the instance ``labels > 0``)
    * ``true/wan_decoder`` (Cosmos + VAE only, passed in)
    * ``pred/sem``  -- foreground probability
    * ``pred/raw``  -- linear reconstruction
    * ``pred/label/pre`` / ``pred/label/mul`` -- Mutex Watershed instances
      (raw, and multiplied by the predicted sem mask), when ``seg_pred_2d``
      is supplied.
    * ``aff/true/{offset}`` / ``aff/pred/{offset}`` (3-D only; curated
      subset) -- all affinity panels live under one ``aff/`` group so the
      core ``image / label / sem / raw`` panels stay clustered together
      instead of being split apart by the (potentially many) offsets.

    The affinity layout is inferred from the head's channel count, so any
    config-driven offset set works (the offset names come from ``offsets``
    / ``n_pull``).
    """
    expected = len(offsets) + 2
    if head_pred.shape[1] != expected:
        raise ValueError(
            f"_log_predictions expects {expected} channels "
            f"(len(offsets)={len(offsets)} + sem + raw); "
            f"got {head_pred.shape[1]}."
        )

    head = ctx
    fields = slice_head(head_pred[:n])
    indices = aff_panel_indices(len(offsets), n_pull)
    names = offset_names(offsets, n_pull)

    # ----- true panels -----
    gt_fg_2d = rearrange((labels[:n] > 0).float(), "b ... -> b 1 ...")
    if spatial_dims == 3 and labels_3d is not None:
        aff_true = affinity_target_from_offsets(
            labels_3d[:n].long(), offsets, background=-1,
        )
        _add_aff_panels(
            tb, head, aff_true, indices,
            names=names, mask_2d=gt_fg_2d, epoch=epoch, tag_prefix="aff/true",
        )

    true_img = _normalise(images[:n])
    if true_img.shape[1] == 1:
        true_img = repeat(true_img, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("true/image"), true_img, global_step=epoch)
    tb.add_images(
        head.tag("true/label"), _label_to_rgb(labels[:n]), global_step=epoch,
    )

    # Ground-truth foreground the sem head is supervised against.  Uses the
    # (boundary-eroded) ``sem_labels`` when provided so the panel matches the
    # actual sem target; otherwise falls back to the instance foreground.
    sem_src = sem_labels if sem_labels is not None else labels
    gt_sem_2d = rearrange((sem_src[:n] > 0).float(), "b ... -> b 1 ...")
    tb.add_images(
        head.tag("true/sem"),
        repeat(gt_sem_2d, "b 1 h w -> b 3 h w"),
        global_step=epoch,
    )

    if wan_decoder_2d is not None:
        # Collapse the VAE reconstruction to a single grayscale channel
        # BEFORE normalising so the panel looks identical regardless of the
        # backbone VAE's ``conv_out`` width (Cosmos-Predict decodes to
        # 3-channel RGB, Cosmos-3's residual VAE to a different width).  EM
        # is grayscale, so the channel-mean is the faithful display and also
        # drops the misleading per-channel colour cast (the reddish tint the
        # 3-channel Wan decoder otherwise shows).  Repeat back to 3 channels
        # only because TensorBoard's make_grid requires exactly 3.
        wan = reduce(wan_decoder_2d[:n], "b c h w -> b 1 h w", "mean")
        wan = repeat(_normalise(wan), "b 1 h w -> b 3 h w")
        tb.add_images(head.tag("true/wan_decoder"), wan, global_step=epoch)

    # ----- pred panels -----
    # The head emits raw logits / linear values: sigmoid the aff / sem
    # channels for display, and rescale the linear raw channel from its
    # [-1, 1] target range back to [0, 1].
    sem = _to_2d(fields["sem"].sigmoid()).clamp(0.0, 1.0)
    sem_rgb = repeat(sem, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("pred/sem"), sem_rgb, global_step=epoch)

    raw = repeat(
        ((_to_2d(fields["raw"]) + 1.0) / 2.0).clamp(0.0, 1.0),
        "b 1 h w -> b 3 h w",
    )
    tb.add_images(head.tag("pred/raw"), raw, global_step=epoch)

    _add_aff_panels(
        tb, head, fields["aff"].sigmoid(), indices,
        names=names, mask_2d=sem, epoch=epoch, tag_prefix="aff/pred",
    )

    # ----- Mutex Watershed instance segmentation -----
    if seg_pred_2d is not None:
        seg_rgb = _label_to_rgb(seg_pred_2d[:n])
        tb.add_images(head.tag("pred/label/pre"), seg_rgb, global_step=epoch)
        # Multiply by predicted sem so masked-out voxels fade to black --
        # easier to read next to the GT label panel.
        tb.add_images(
            head.tag("pred/label/mul"), seg_rgb * sem_rgb, global_step=epoch,
        )


__all__ = ["_log_predictions", "aff_panel_indices"]
