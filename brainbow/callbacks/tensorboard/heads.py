"""TensorBoard panels for the affinity + sem + raw head.

Emits the ``true/*`` ground-truth panels and the ``pred/*`` prediction
panels that mirror the loss scalar paths.  The head layout is owned by
:mod:`brainbow.losses._common`:

    aff: per-offset affinity probabilities (a curated subset is shown)
    sem: foreground / boundary probability
    raw: linear reconstruction of the input EM intensity

The instance segmentation (``pred/label``) is the Mutex Watershed
agglomeration of the predicted affinities, computed by the caller (see
:mod:`brainbow.callbacks.tensorboard.image_logger`) on the full 3-D head
and passed in as a central-slice label map.
"""

from typing import Any, List, Optional, Sequence

import torch
from einops import rearrange, repeat

from brainbow.callbacks.tensorboard.tags import TagContext
from brainbow.callbacks.tensorboard.viz import _label_to_rgb, _normalise, _to_2d
from brainbow.losses import (
    AFF_NAMES,
    AFFINITY_OFFSETS,
    HEAD_CHANNELS,
    N_ATTRACTIVE,
    affinity_target_from_offsets,
    slice_head,
)


def aff_panel_indices(
    n_offsets: int,
    n_attractive: int,
    max_repulsive: Optional[int] = None,
) -> List[int]:
    """Affinity channels to visualise.

    By default every offset is shown (``max_repulsive=None`` -> all
    ``n_offsets`` channels: the attractive nearest-neighbours followed by
    every long-range repulsive offset).  Pass an integer ``max_repulsive``
    to instead show all attractive offsets plus that many evenly-spaced
    repulsive ones.
    """
    if max_repulsive is None:
        return list(range(n_offsets))
    idxs = list(range(min(n_attractive, n_offsets)))
    repulsive = list(range(n_attractive, n_offsets))
    if repulsive and max_repulsive > 0:
        step = max(1, len(repulsive) // max_repulsive)
        idxs += repulsive[::step][:max_repulsive]
    return idxs


def _add_aff_panels(
    tb: Any,
    head: TagContext,
    aff_3d: torch.Tensor,
    indices: Sequence[int],
    *,
    mask_2d: torch.Tensor,
    epoch: int,
    tag_prefix: str,
) -> None:
    """Central-slice affinity panels (a curated channel subset)."""
    aff_2d = _to_2d(aff_3d).clamp(0.0, 1.0)
    for k in indices:
        panel = repeat(aff_2d[:, k:k + 1] * mask_2d, "b 1 h w -> b 3 h w")
        tb.add_images(
            head.tag(f"{tag_prefix}/{AFF_NAMES[k]}"), panel, global_step=epoch,
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
    n_attractive: int = N_ATTRACTIVE,
    labels_3d: Optional[torch.Tensor] = None,
    seg_pred_2d: Optional[torch.Tensor] = None,
    wan_decoder_2d: Optional[torch.Tensor] = None,
) -> None:
    """Log the affinity + sem + raw panels.

    Tags (under ``{stage}/{mode}/``):

    * ``true/image``, ``true/label``
    * ``true/aff/{offset}`` (3-D only; curated subset)
    * ``true/wan_decoder`` (Cosmos + VAE only, passed in)
    * ``pred/sem``  -- foreground probability
    * ``pred/raw``  -- linear reconstruction
    * ``pred/aff/{offset}`` -- predicted affinities (same subset)
    * ``pred/label/pre`` / ``pred/label/mul`` -- Mutex Watershed instances
      (raw, and multiplied by the predicted sem mask), when ``seg_pred_2d``
      is supplied.
    """
    if head_pred.shape[1] != HEAD_CHANNELS:
        raise ValueError(
            f"_log_predictions expects {HEAD_CHANNELS} channels; "
            f"got {head_pred.shape[1]}."
        )

    head = ctx
    fields = slice_head(head_pred[:n])
    indices = aff_panel_indices(len(offsets), n_attractive)

    # ----- true panels -----
    gt_fg_2d = rearrange((labels[:n] > 0).float(), "b ... -> b 1 ...")
    if spatial_dims == 3 and labels_3d is not None:
        aff_true = affinity_target_from_offsets(
            labels_3d[:n].long(), offsets, background=-1,
        )
        _add_aff_panels(
            tb, head, aff_true, indices,
            mask_2d=gt_fg_2d, epoch=epoch, tag_prefix="true/aff",
        )

    true_img = _normalise(images[:n])
    if true_img.shape[1] == 1:
        true_img = repeat(true_img, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("true/image"), true_img, global_step=epoch)
    tb.add_images(
        head.tag("true/label"), _label_to_rgb(labels[:n]), global_step=epoch,
    )

    if wan_decoder_2d is not None:
        tb.add_images(
            head.tag("true/wan_decoder"),
            _normalise(wan_decoder_2d[:n]),
            global_step=epoch,
        )

    # ----- pred panels -----
    sem = _to_2d(fields["sem"]).clamp(0.0, 1.0)
    sem_rgb = repeat(sem, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("pred/sem"), sem_rgb, global_step=epoch)

    raw = repeat(_to_2d(fields["raw"]).clamp(0.0, 1.0), "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("pred/raw"), raw, global_step=epoch)

    _add_aff_panels(
        tb, head, fields["aff"], indices,
        mask_2d=sem, epoch=epoch, tag_prefix="pred/aff",
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
