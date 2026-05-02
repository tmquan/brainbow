"""TensorBoard panels for the unified 30-channel head."""

from typing import Any

import torch
from einops import rearrange, repeat

from brainbow.callbacks.tensorboard.geometry import (
    _render_cov_flow,
    _render_dir_flow,
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
    build_avg_target,
    slice_head,
    soft_aff_from_field,
    upper_tri_to_matrix,
)

# Index each direction so the alphabetical TB sort respects the
# physical pairing in :data:`brainbow.losses.DIRECTIONS` -- each
# axis-aligned pair (T/B, U/D, L/R) for stride 1 then stride 2 lands at
# consecutive (even, odd) panel positions::
#
#     01_t1, 02_b1   # z stride 1
#     03_u1, 04_d1   # y stride 1
#     05_l1, 06_r1   # x stride 1
#     07_t2, 08_b2   # z stride 2
#     09_u2, 10_d2   # y stride 2
#     11_l2, 12_r2   # x stride 2
#
# Zero-padding to two digits keeps lexicographic order numeric (so
# "10_d2" sorts after "09_u2", not between "01_t1" and "02_b1").
_AFF_TAG_NAMES: tuple[str, ...] = tuple(
    f"{i + 1:02d}_{name.lower()}" for i, name in enumerate(AFF_NAMES)
)


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
    wan_decoder_2d: torch.Tensor | None = None,
) -> None:
    """Log true panels plus the unified-head prediction panels.

    Tags intentionally mirror the loss scalar paths.  Each affinity
    direction is prefixed with its position in
    :data:`brainbow.losses.DIRECTIONS` (``01_t1`` ... ``12_r2``) so the
    alphabetical TB sort places each axis-aligned pair on consecutive
    panels (T/B, U/D, L/R for stride 1 then stride 2).

    * ``pred/raw``
    * ``pred/sem``
    * ``pred/dir``
    * ``pred/cov``
    * ``pred/avg/val`` and ``pred/avg/aff/{01_t1,02_b1,...,12_r2}``
    * ``pred/emb/_{pca|svd|umap}``, ``pred/emb/aff/{01_t1,...,12_r2}``,
      ``pred/label/{pre,mul}`` (``pre`` = raw clustering output;
      ``mul`` = same panel multiplied by the predicted sem mask)
    * ``true/image``, ``true/label``
    * ``true/avg/val`` and ``true/aff/{01_t1,...,12_r2}`` (3-D only)
    * ``true/wan_decoder`` (RGB pixel reconstruction from the original
      pretrained Wan decoder; only emitted when ``wan_decoder_2d`` is
      passed in -- i.e. on the Cosmos wrapper with a loaded VAE).
    """
    if head_pred.shape[1] != HEAD_CHANNELS:
        raise ValueError(
            f"_log_predictions expects {HEAD_CHANNELS} channels; "
            f"got {head_pred.shape[1]}."
        )

    head = ctx
    fields = slice_head(head_pred[:n])

    # ----- true panels -----
    if spatial_dims == 3 and labels_3d is not None:
        aff_true = affinity_target(labels_3d[:n].long(), background=-1)
        _add_aff_panels(
            tb, head, aff_true,
            labels_2d=labels[:n], sem_ids=labels[:n],
            epoch=epoch, tag_prefix="true/aff",
        )
        avg_true = _to_2d(build_avg_target(labels_3d[:n].long())).clamp(0.0, 1.0)
        tb.add_images(head.tag("true/avg/val"), avg_true, global_step=epoch)

    true_img = _normalise(images[:n])
    if true_img.shape[1] == 1:
        true_img = repeat(true_img, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("true/image"), true_img, global_step=epoch)
    tb.add_images(
        head.tag("true/label"),
        _label_to_rgb(labels[:n]),
        global_step=epoch,
    )

    # Pretrained Wan decoder reconstruction (Cosmos only).  Wan emits
    # values in roughly ``[-1, 1]``; per-image min/max normalise to
    # ``[0, 1]`` for display so the panel reads naturally next to
    # ``true/image``.  Suppressed when no VAE was loaded
    # (``wan_decoder_2d is None``).
    if wan_decoder_2d is not None:
        tb.add_images(
            head.tag("true/wan_decoder"),
            _normalise(wan_decoder_2d[:n]),
            global_step=epoch,
        )

    # ----- raw / sem -----
    raw = repeat(
        _to_2d(fields["raw"]).clamp(0.0, 1.0),
        "b 1 h w -> b 3 h w",
    )
    tb.add_images(head.tag("pred/raw"), raw, global_step=epoch)

    sem = _to_2d(fields["sem"]).clamp(0.0, 1.0)
    sem_ids = (sem[:, 0] > 0.5).long()
    sem_rgb = repeat(sem, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("pred/sem"), sem_rgb, global_step=epoch)

    # ----- dir / cov -----
    # Optical-flow-style HSV overlays on the raw EM (see geometry.py):
    # `dir` colours each foreground pixel by its centroid-direction
    # angle (hue) and magnitude (value); `cov` colours by the principal
    # eigenvector orientation (hue) with anisotropy → saturation and
    # max-eigenvalue → value.
    dir_rgb = _render_dir_flow(
        _to_2d(fields["dir"]), images[:n], sem_ids, spatial_dims,
    )
    tb.add_images(head.tag("pred/dir"), dir_rgb, global_step=epoch)

    cov_tri = _to_2d(fields["cov"])
    cov_mat = upper_tri_to_matrix(cov_tri, spatial_dims)
    cov_rgb = _render_cov_flow(cov_mat, images[:n], sem_ids, spatial_dims)
    tb.add_images(head.tag("pred/cov"), cov_rgb, global_step=epoch)

    # ----- avg + avg-aff -----
    # Multiply by the predicted sem so background voxels fade to black
    # and the per-instance RGB triplet reads cleanly against the dark
    # backdrop, matching the `pred/label/mul` and `pred/emb/_{algo}`
    # convention.
    avg_rgb = _to_2d(fields["avg"]).clamp(0.0, 1.0) * sem_rgb
    tb.add_images(head.tag("pred/avg/val"), avg_rgb, global_step=epoch)

    if spatial_dims == 3:
        aff_avg = soft_aff_from_field(fields["avg"], tau=aff_avg_tau)
        _add_aff_panels(
            tb, head, aff_avg,
            labels_2d=labels[:n], sem_ids=sem_ids,
            epoch=epoch, tag_prefix="pred/avg/aff",
        )

    # ----- embedding projection + emb-aff + clustered labels -----
    # Multiply the manifold projection by the predicted sem so the
    # background fades to black, same convention as `pred/avg/val` and
    # `pred/label/mul`.
    emb_2d = _to_2d(fields["emb"])
    emb_rgb = _project_embedding(
        emb_2d, n_components=3,
        algorithm=projection_algorithm, backend=projection_backend,
    ) * sem_rgb
    tb.add_images(
        head.tag(f"pred/emb/_{projection_algorithm}"),
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
        tb.add_images(head.tag("pred/label/pre"), label_rgb, global_step=epoch)
        # Multiply by predicted semantic probability so masked-out
        # voxels fade to black -- easier to read in TB next to the GT
        # label panel.
        label_rgb_mul = label_rgb * sem_rgb
        tb.add_images(head.tag("pred/label/mul"), label_rgb_mul, global_step=epoch)


__all__ = ["_log_predictions"]
