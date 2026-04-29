"""TensorBoard panels for the unified 30-channel head."""

from typing import Any

import torch
from einops import rearrange, repeat

from brainbow.callbacks.tensorboard.geometry import (
    _render_cov_glyphs,
    _render_dir_quiver,
)
from brainbow.callbacks.tensorboard.tags import TagContext
from brainbow.callbacks.tensorboard.viz import (
    _label_to_rgb,
    _normalise,
    _project_embedding,
    _to_2d,
)
from brainbow.losses import (
    AFF_NAMES,
    AVG_SLICE,
    COV_SLICE,
    DIR_SLICE,
    EMB_SLICE,
    HEAD_CHANNELS,
    RAW_SLICE,
    SEM_SLICE,
    affinity_target,
    slice_head,
    soft_aff_from_field,
    upper_tri_to_matrix,
)

_AFF_TAG_NAMES = tuple(name.lower() for name in AFF_NAMES)


def _aff_fg_mask_2d(sem_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Foreground mask for affinity-panel display only."""
    src = sem_ids if sem_ids is not None else labels
    return rearrange((src > 0).float(), "b ... -> b 1 ...")


def _add_aff_panels(
    tb: Any,
    head: TagContext,
    aff_3d: torch.Tensor,
    *,
    labels_2d: torch.Tensor,
    sem_ids: torch.Tensor,
    epoch: int,
    tag_prefix: str,
) -> None:
    """Central-slice 12-channel affinity panels under ``tag_prefix``."""
    aff_2d = _to_2d(aff_3d).clamp(0.0, 1.0)
    mask = _aff_fg_mask_2d(sem_ids, labels_2d)
    for k, name in enumerate(_AFF_TAG_NAMES):
        panel = repeat(aff_2d[:, k:k + 1] * mask, "b 1 h w -> b 3 h w")
        tb.add_images(head.tag(f"{tag_prefix}/{name}"), panel, global_step=epoch)


def _log_predictions(
    tb: Any,
    ctx: TagContext,
    images: torch.Tensor,
    labels: torch.Tensor,
    head_pred: torch.Tensor,
    spatial_dims: int,
    n: int,
    epoch: int,
    clusterer: Any = None,
    labels_3d: torch.Tensor | None = None,
    projection_algorithm: str = "pca",
    projection_backend: str = "auto",
    aff_emb_tau: float = 1.0,
    aff_avg_tau: float = 1.0,
    normalize_embeddings: bool = False,
) -> None:
    """Log true panels plus the unified-head prediction panels.

    Tags intentionally mirror the loss scalar paths:

    * ``pred/raw``
    * ``pred/sem``
    * ``pred/dir``
    * ``pred/cov``
    * ``pred/avg`` and ``pred/avg/aff/{t1,...,r2}``
    * ``pred/emb/{pca|svd|umap}``, ``pred/emb/aff/{...}``, ``pred/label``
    * ``true/aff/{...}``
    """
    if head_pred.shape[1] != HEAD_CHANNELS:
        raise ValueError(
            f"_log_predictions expects {HEAD_CHANNELS} channels; "
            f"got {head_pred.shape[1]}."
        )

    head = ctx
    fields = slice_head(head_pred[:n])

    # ----- true panels -----
    true_img = _normalise(images[:n])
    if true_img.shape[1] == 1:
        true_img = repeat(true_img, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("true/image"), true_img, global_step=epoch)
    tb.add_images(
        head.tag("true/label"),
        _label_to_rgb(labels[:n]),
        global_step=epoch,
    )
    if spatial_dims == 3 and labels_3d is not None:
        aff_true = affinity_target(labels_3d[:n].long(), background=-1)
        _add_aff_panels(
            tb, head, aff_true,
            labels_2d=labels[:n], sem_ids=labels[:n],
            epoch=epoch, tag_prefix="true/aff",
        )

    # ----- raw / sem -----
    raw = repeat(
        _to_2d(fields["raw"]).clamp(0.0, 1.0),
        "b 1 h w -> b 3 h w",
    )
    tb.add_images(head.tag("pred/raw"), raw, global_step=epoch)

    sem = _to_2d(fields["sem"]).clamp(0.0, 1.0)
    sem_ids = (sem[:, 0] > 0.5).long()
    tb.add_images(
        head.tag("pred/sem"),
        repeat(sem, "b 1 h w -> b 3 h w"),
        global_step=epoch,
    )

    # ----- dir / cov -----
    dir_rgb = _render_dir_quiver(
        _to_2d(fields["dir"]), images[:n], sem_ids, spatial_dims,
        dir_target="centroid",
    )
    tb.add_images(head.tag("pred/dir"), dir_rgb, global_step=epoch)

    cov_tri = _to_2d(fields["cov"])
    cov_mat = upper_tri_to_matrix(cov_tri, spatial_dims)
    cov_rgb = _render_cov_glyphs(cov_mat, images[:n], sem_ids, spatial_dims)
    tb.add_images(head.tag("pred/cov"), cov_rgb, global_step=epoch)

    # ----- avg + avg-aff -----
    avg_rgb = _to_2d(fields["avg"]).clamp(0.0, 1.0)
    tb.add_images(head.tag("pred/avg"), avg_rgb, global_step=epoch)

    if spatial_dims == 3:
        aff_avg = soft_aff_from_field(fields["avg"], tau=aff_avg_tau)
        _add_aff_panels(
            tb, head, aff_avg,
            labels_2d=labels[:n], sem_ids=sem_ids,
            epoch=epoch, tag_prefix="pred/avg/aff",
        )

    # ----- embedding projection + emb-aff + clustered labels -----
    emb_2d = _to_2d(fields["emb"])
    emb_rgb = _project_embedding(
        emb_2d, n_components=3,
        algorithm=projection_algorithm, backend=projection_backend,
    )
    tb.add_images(
        head.tag(f"pred/emb/{projection_algorithm}"),
        emb_rgb,
        global_step=epoch,
    )

    if spatial_dims == 3:
        emb = fields["emb"]
        if normalize_embeddings:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1, eps=1e-6)
        aff_emb = soft_aff_from_field(emb, tau=aff_emb_tau)
        _add_aff_panels(
            tb, head, aff_emb,
            labels_2d=labels[:n], sem_ids=sem_ids,
            epoch=epoch, tag_prefix="pred/emb/aff",
        )

    if clusterer is not None:
        fg_mask_pred = sem_ids > 0
        labels_pred, _, _ = clusterer(emb_2d.float(), fg_mask_pred)
        label_rgb = _label_to_rgb(labels_pred)
        # Mask by semantic probability for readability.
        label_rgb = label_rgb * repeat(sem, "b 1 h w -> b 3 h w")
        tb.add_images(head.tag("pred/label"), label_rgb, global_step=epoch)


__all__ = ["_log_predictions"]
