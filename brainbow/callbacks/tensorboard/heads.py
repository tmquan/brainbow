"""Per-head TensorBoard loggers and the :func:`_log_predictions` orchestrator.

Each ``_log_<head>`` function writes a small, well-defined set of
panels under ``{stage}/{mode}/{head}/...`` and is a no-op when its
head key is missing from the model prediction dict.  The orchestrator
stitches them together behind a single call.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange, repeat

from brainbow.callbacks.tensorboard.geometry import (
    _render_cov_glyphs,
    _render_dir_quiver,
)
from brainbow.callbacks.tensorboard.tags import TagContext
from brainbow.losses.geometry import upper_tri_channels_to_matrix
from brainbow.callbacks.tensorboard.viz import (
    _label_to_rgb,
    _normalise,
    _project_embedding,
    _to_2d,
)


def _log_semantic(
    tb: Any,
    ctx: TagContext,
    preds: Dict[str, torch.Tensor],
    n: int,
    epoch: int,
    *,
    active_classes: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Log ``{stage}/{mode}/semantic/prediction`` and return fg tensors.

    Returns:
        ``(sem_fg, sem_ids)``:
          - ``sem_fg``  : ``[n, 1, H, W]`` foreground probability, or
            ``None`` if the head is absent.
          - ``sem_ids`` : ``[n, H, W]`` argmax / thresholded class map,
            or ``None`` if the head is absent.
        Both values are consumed by :func:`_log_instance` and
        :func:`_log_geometry` to build a foreground mask consistent
        with what the deployed model would see at inference time.
    """
    if "semantic" not in preds:
        return None, None
    head = ctx.for_head("semantic")
    sem = _to_2d(preds["semantic"][:n])
    if active_classes is not None and active_classes < sem.shape[1]:
        sem = sem[:, :active_classes]
    if sem.shape[1] == 1:
        sem_fg = sem[:, :1].sigmoid()
        sem_ids = (sem_fg[:, 0] > 0.5).long()
    else:
        sem_ids = sem.argmax(dim=1)
        sem_fg = sem[:, :2].softmax(dim=1)[:, 1:]
    sem_gray = repeat(sem_fg, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("prediction"), sem_gray, global_step=epoch)
    return sem_fg, sem_ids


def _log_instance(
    tb: Any,
    ctx: TagContext,
    preds: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    n: int,
    epoch: int,
    *,
    sem_fg: Optional[torch.Tensor],
    sem_ids: Optional[torch.Tensor],
    clusterer: Any = None,
    projection_algorithm: str = "pca",
    projection_backend: str = "auto",
) -> None:
    """Log the manifold-projected embedding (``instance/{pca|svd|umap}``)
    and, when a clusterer is provided, the clustered ``instance/prediction``."""
    if "instance" not in preds:
        return
    head = ctx.for_head("instance")
    inst = _to_2d(preds["instance"][:n])
    inst_rgb = _project_embedding(
        inst, n_components=3,
        algorithm=projection_algorithm, backend=projection_backend,
    )
    tb.add_images(head.tag(projection_algorithm), inst_rgb, global_step=epoch)

    if clusterer is None:
        return
    if sem_fg is not None:
        fg_mask_pred = sem_ids > 0
        fg_alpha = sem_fg
    else:
        # Semantic head disabled (e.g. ``weight_semantic=0``):
        # there is no predicted foreground at inference time.
        # Use an all-ones mask so the panel honestly reflects
        # what will happen when the model is deployed — every
        # voxel is clustered — instead of silently pulling
        # information from GT labels and flattering the viz.
        fg_mask_pred = torch.ones_like(labels, dtype=torch.bool)
        fg_alpha = rearrange(fg_mask_pred.float(), "b ... -> b 1 ...")
    # `inst` has been sliced to 2-D by `_to_2d` above; the fg mask
    # must carry the same spatial rank, i.e. [B, H, W].  (Historically
    # an extra channel dim was added here which the flattening
    # clusterers tolerated but `spatial_cc` does not — keep shapes
    # honest.)
    if inst.dim() == 5:
        fg_mask_full = rearrange(
            _to_2d(rearrange(fg_mask_pred, "b ... -> b 1 ...")),
            "b 1 ... -> b ...",
        )
    else:
        fg_mask_full = fg_mask_pred
    ins_pred, _, _ = clusterer(inst, fg_mask_full)
    if ins_pred.dim() > 3:
        ins_pred = rearrange(
            _to_2d(rearrange(ins_pred, "b ... -> b 1 ...")),
            "b 1 ... -> b ...",
        )
    ins_rgb = _label_to_rgb(ins_pred.long()) * fg_alpha
    tb.add_images(head.tag("prediction"), ins_rgb, global_step=epoch)


def _log_geometry(
    tb: Any,
    ctx: TagContext,
    preds: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    img_gray: torch.Tensor,
    n: int,
    epoch: int,
    *,
    spatial_dims: int,
    sem_ids: Optional[torch.Tensor],
    dir_target: str = "centroid",
) -> None:
    """Log the three geometry panels under ``{stage}/{mode}/geometry/``:

    * ``raw``                     -- raw-intensity reconstruction (ch 0, grayscale)
    * ``dir_{centroid|skeleton}`` -- direction quiver overlay
    * ``cov``                     -- covariance ellipse glyphs (from
      the upper-triangle channels, expanded to full matrices here
      only for rendering)

    Uses predicted semantic foreground (``sem_ids``) when available,
    otherwise falls back to an all-ones mask so the visualisation
    reflects what the deployed model would see at inference time.

    Channel layout matches :class:`brainbow.losses.geometry.GeometryLoss`::

        ch 0                         := raw   (1 channel, grayscale)
        ch 1 .. 1+S                  := dir   (S channels)
        ch 1+S .. 1+S + S*(S+1)//2   := cov   (upper-triangle covariance)
    """
    if "geometry" not in preds:
        return
    head = ctx.for_head("geometry")
    S = spatial_dims
    ch_raw = 1
    ch_dir = S
    ch_cov_tri = S * (S + 1) // 2
    geom = _to_2d(preds["geometry"][:n])
    if sem_ids is not None:
        fg_mask_pred = (sem_ids > 0).long()
    else:
        fg_mask_pred = torch.ones_like(labels, dtype=torch.long)

    g_raw = geom[:, :ch_raw].clamp(0.0, 1.0)
    g_raw_rgb = repeat(g_raw, "b 1 h w -> b 3 h w")

    g_dir_rgb = _render_dir_quiver(
        geom[:, ch_raw:ch_raw + ch_dir], img_gray, fg_mask_pred, S,
        dir_target=dir_target,
    )

    cov_tri = geom[:, ch_raw + ch_dir:ch_raw + ch_dir + ch_cov_tri]
    cov_mat = upper_tri_channels_to_matrix(cov_tri, S)
    g_cov_rgb = _render_cov_glyphs(cov_mat, img_gray, fg_mask_pred, S)

    tb.add_images(head.tag("raw"), g_raw_rgb, global_step=epoch)
    tb.add_images(head.tag(f"dir_{dir_target}"), g_dir_rgb, global_step=epoch)
    tb.add_images(head.tag("cov"), g_cov_rgb, global_step=epoch)


def _log_brainbow(
    tb: Any,
    ctx: TagContext,
    preds: Dict[str, torch.Tensor],
    n: int,
    epoch: int,
    *,
    brainbow_target: Optional[torch.Tensor] = None,
) -> None:
    """Log the 4 brainbow panels under ``{stage}/{mode}/brainbow/``.

    Layout of the 10-channel brainbow prediction / target:
      - ch 0    : ``rawval`` (dense, fg + bg; logged as grayscale)
      - ch 1-3  : ``minloc`` RGB (foreground-only; zero on background)
      - ch 4-6  : ``avgloc`` RGB
      - ch 7-9  : ``maxloc`` RGB

    Panels are written under ``brainbow/pred/*`` and, when
    ``brainbow_target`` is supplied, also under ``brainbow/gt/*`` so
    the model output and its supervision signal can be compared
    side-by-side in TensorBoard.
    """
    if "brainbow" not in preds:
        return
    head = ctx.for_head("brainbow")
    bb_pred = _to_2d(preds["brainbow"][:n]).clamp(0.0, 1.0)
    _add_brainbow_panels(tb, head, "pred", bb_pred, epoch)
    if brainbow_target is not None:
        bb_gt = _to_2d(brainbow_target[:n]).clamp(0.0, 1.0)
        _add_brainbow_panels(tb, head, "gt", bb_gt, epoch)


def _add_brainbow_panels(
    tb: Any,
    head: TagContext,
    variant: str,
    bb: torch.Tensor,
    epoch: int,
) -> None:
    """Split a ``[n, 10, H, W]`` brainbow tensor into its 4 sub-panels.

    Args:
        tb: TensorBoard SummaryWriter.
        head: ``TagContext`` for the brainbow head (i.e. ``ctx.for_head("brainbow")``).
        variant: ``"pred"`` or ``"gt"``.  Becomes the next tag segment.
        bb: ``[n, 10, H, W]`` brainbow map (already 2-D sliced and
            clamped to ``[0, 1]``).
        epoch: global step for TensorBoard.
    """
    rawval = repeat(bb[:, 0:1], "b 1 h w -> b 3 h w")
    minloc = bb[:, 1:4]
    avgloc = bb[:, 4:7]
    maxloc = bb[:, 7:10]
    tb.add_images(head.tag(f"{variant}/rawval"), rawval, global_step=epoch)
    tb.add_images(head.tag(f"{variant}/minloc"), minloc, global_step=epoch)
    tb.add_images(head.tag(f"{variant}/avgloc"), avgloc, global_step=epoch)
    tb.add_images(head.tag(f"{variant}/maxloc"), maxloc, global_step=epoch)


def _log_predictions(
    tb: Any,
    ctx: TagContext,
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    spatial_dims: int,
    n: int,
    epoch: int,
    clusterer: Any = None,
    dir_target: str = "centroid",
    active_classes: Optional[int] = None,
    projection_algorithm: str = "pca",
    projection_backend: str = "auto",
    brainbow_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Log a standard set of prediction visualisations to TensorBoard.

    Emits, under ``ctx.prefix`` = ``{stage}/{mode}``:

    * mode level ::

        {ctx.prefix}/image
        {ctx.prefix}/label

    * per-head (only when the corresponding output is present in
      ``preds``) ::

        {ctx.prefix}/semantic/prediction
        {ctx.prefix}/instance/{pca|svd|umap}
        {ctx.prefix}/instance/prediction                  (if clusterer)
        {ctx.prefix}/geometry/dir_{centroid|skeleton}
        {ctx.prefix}/geometry/cov
        {ctx.prefix}/geometry/raw
        {ctx.prefix}/brainbow/pred/{rawval,minloc,avgloc,maxloc}
        {ctx.prefix}/brainbow/gt/{rawval,minloc,avgloc,maxloc}  (if target)

    Args:
        tb: TensorBoard SummaryWriter.
        ctx: mode-level :class:`TagContext` (``head`` must be ``None``).
        images: ``[n, 1, H, W]`` input images (already 2-D sliced).
        labels: ``[n, H, W]`` instance labels (already 2-D sliced).
        preds: model output dict with any subset of ``semantic``,
            ``instance``, ``geometry``, ``brainbow``.
        spatial_dims: 2 or 3 (controls geometry channel layout).
        n: number of images.
        epoch: global step for TensorBoard.
        clusterer: optional clusterer (SoftMeanShift / HDBSCAN / MeanShift)
            for producing the ``instance/prediction`` panel.
        dir_target: ``"centroid"`` or ``"skeleton"`` (geometry head).
        active_classes: number of active semantic channels (from config).
        projection_algorithm: Manifold algorithm for the ``instance/*``
            panel.  One of ``"pca"`` (default), ``"svd"``, ``"umap"``.
        projection_backend: Backend for the projection.  ``"auto"`` picks
            cuML on CUDA, else a CPU fallback.  ``"cuml"`` forces GPU.
        brainbow_target: optional ``[n, 10, D, H, W]`` ground-truth map
            to log alongside the brainbow prediction.

    Returns:
        ``[n, 3, H, W]`` grayscale image repeated to RGB (reused by
        geometry overlays and available for future prompt overlays).
    """
    if ctx.head is not None:
        raise ValueError(
            f"_log_predictions expects a mode-level TagContext "
            f"(head=None); got head={ctx.head!r}."
        )

    img_gray = repeat(_normalise(images), "b 1 h w -> b 3 h w").contiguous()
    lbl_rgb = _label_to_rgb(labels.long())
    tb.add_images(ctx.tag("image"), img_gray, global_step=epoch)
    tb.add_images(ctx.tag("label"), lbl_rgb, global_step=epoch)

    sem_fg, sem_ids = _log_semantic(
        tb, ctx, preds, n, epoch, active_classes=active_classes,
    )
    _log_instance(
        tb, ctx, preds, labels, n, epoch,
        sem_fg=sem_fg, sem_ids=sem_ids, clusterer=clusterer,
        projection_algorithm=projection_algorithm,
        projection_backend=projection_backend,
    )
    _log_geometry(
        tb, ctx, preds, labels, img_gray, n, epoch,
        spatial_dims=spatial_dims, sem_ids=sem_ids, dir_target=dir_target,
    )
    _log_brainbow(
        tb, ctx, preds, n, epoch, brainbow_target=brainbow_target,
    )

    return img_gray


__all__ = [
    "_add_brainbow_panels",
    "_log_brainbow",
    "_log_geometry",
    "_log_instance",
    "_log_predictions",
    "_log_semantic",
]
