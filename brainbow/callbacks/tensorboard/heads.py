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
# Channel layout constants for the boundary head live in the loss
# module -- single source of truth.  Re-importing them here (rather
# than re-declaring the magic numbers) guarantees that any future
# change to the boundary channel layout (e.g. extra affinity
# directions) propagates into the visualiser automatically.
from brainbow.losses.boundary import (
    _AFF_END,
    _AFF_NAMES as _BND_AFF_NAMES,
    _AFF_START,
    _AVG_END,
    _AVG_START,
    _BOUNDARY_CHANNELS,
    _N_AFF,
    soft_aff_from_avg,
)
from brainbow.losses.geometry import upper_tri_channels_to_matrix
from brainbow.callbacks.tensorboard.viz import (
    _label_to_rgb,
    _normalise,
    _project_embedding,
    _to_2d,
)

# Lowercase affinity tag segments (T/B/U/D/L/R -> t/b/u/d/l/r) derived
# from the canonical Z-Y-X-ordered name tuple in ``brainbow.losses.
# boundary._AFF_NAMES``.
_AFF_TAG_NAMES: Tuple[str, ...] = tuple(name.lower() for name in _BND_AFF_NAMES)


def _log_semantic(
    tb: Any,
    ctx: TagContext,
    preds: Dict[str, torch.Tensor],
    n: int,
    epoch: int,
    *,
    active_classes: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Log ``{stage}/{mode}/semantic/pred`` and return fg tensors.

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
    # ``preds["semantic"]`` is already a tensor of per-channel sigmoid
    # probabilities -- the model wrapper applies sigmoid to the semantic
    # head before anything downstream (loss, metrics, this callback)
    # sees it.  No activation is re-applied here.
    sem = _to_2d(preds["semantic"][:n])
    if active_classes is not None and active_classes < sem.shape[1]:
        sem = sem[:, :active_classes]
    if sem.shape[1] == 1:
        sem_fg = sem[:, :1]
        sem_ids = (sem_fg[:, 0] > 0.5).long()
    else:
        # Multi-channel sigmoid (multi-label).  ``argmax`` is monotone
        # under sigmoid so the class ranking matches the old logits path;
        # use the second channel's probability as a foreground heatmap.
        sem_ids = sem.argmax(dim=1)
        sem_fg = sem[:, 1:2]
    sem_gray = repeat(sem_fg, "b 1 h w -> b 3 h w")
    tb.add_images(head.tag("pred"), sem_gray, global_step=epoch)
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
    """Log the manifold-projected embedding
    (``instance/pred/{pca|svd|umap}``) and, when a clusterer is
    provided, the clustered label map under ``instance/pred/label``."""
    if "instance" not in preds:
        return
    head = ctx.for_head("instance")
    inst = _to_2d(preds["instance"][:n])
    inst_rgb = _project_embedding(
        inst, n_components=3,
        algorithm=projection_algorithm, backend=projection_backend,
    )
    tb.add_images(
        head.tag(f"pred/{projection_algorithm}"), inst_rgb, global_step=epoch,
    )

    if clusterer is None:
        return
    if sem_fg is not None:
        # ``sem_ids`` is already 2-D (the central slice was taken inside
        # ``_log_semantic`` via ``_to_2d``), so ``fg_mask_pred`` is
        # [B, H, W] -- same spatial rank as ``inst`` above.
        fg_mask_pred = sem_ids > 0
        fg_alpha = sem_fg
    else:
        # Semantic head disabled (e.g. ``weight_semantic=0``):
        # there is no predicted foreground at inference time.
        # Use an all-ones mask so the panel honestly reflects what
        # will happen when the model is deployed -- every voxel is
        # clustered -- instead of silently pulling information from
        # GT labels and flattering the viz.
        fg_mask_pred = torch.ones_like(labels, dtype=torch.bool)
        fg_alpha = rearrange(fg_mask_pred.float(), "b ... -> b 1 ...")

    # Both ``inst`` and ``fg_mask_pred`` are 2-D here (``_to_2d`` was
    # applied above and ``sem_ids`` / ``labels`` arrive 2-D from the
    # orchestrator), so the clusterer sees matching ranks [B, H, W].
    ins_pred, _, _ = clusterer(inst, fg_mask_pred)
    if ins_pred.dim() > 3:
        # Defensive: a clusterer that returns a 4-D tensor (B, 1, H, W)
        # is still acceptable -- squeeze the channel axis so
        # ``_label_to_rgb`` receives the expected [B, H, W] long map.
        ins_pred = rearrange(
            _to_2d(rearrange(ins_pred, "b ... -> b 1 ...")),
            "b 1 ... -> b ...",
        )
    ins_rgb = _label_to_rgb(ins_pred.long()) * fg_alpha
    tb.add_images(head.tag("pred/label"), ins_rgb, global_step=epoch)


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
    """Log the three geometry panels under ``{stage}/{mode}/geometry/pred/``:

    * ``pred/raw``                     -- raw-intensity reconstruction (ch 0, grayscale)
    * ``pred/dir_{centroid|skeleton}`` -- direction quiver overlay
    * ``pred/cov``                     -- covariance ellipse glyphs (from
      the upper-triangle channels, expanded to full matrices here
      only for rendering)

    Uses predicted semantic foreground (``sem_ids``) when available,
    otherwise falls back to an all-ones mask so the visualisation
    reflects what the deployed model would see at inference time.

    Channel layout matches :class:`brainbow.losses.geometry.GeometryLoss`::

        ch 0                          := raw   (1 channel, grayscale)
        ch 1 .. 1 + S*(S+1)//2        := cov   (upper-triangle covariance)
        ch 1 + S*(S+1)//2 .. channels := dir   (S channels)
    """
    if "geometry" not in preds:
        return
    head = ctx.for_head("geometry")
    S = spatial_dims
    ch_raw = 1
    ch_cov_tri = S * (S + 1) // 2
    ch_dir = S
    c_cov_end = ch_raw + ch_cov_tri
    c_dir_end = c_cov_end + ch_dir
    geom = _to_2d(preds["geometry"][:n])
    if sem_ids is not None:
        fg_mask_pred = (sem_ids > 0).long()
    else:
        fg_mask_pred = torch.ones_like(labels, dtype=torch.long)

    g_raw = geom[:, :ch_raw].clamp(0.0, 1.0)
    g_raw_rgb = repeat(g_raw, "b 1 h w -> b 3 h w")

    cov_tri = geom[:, ch_raw:c_cov_end]
    cov_mat = upper_tri_channels_to_matrix(cov_tri, S)
    g_cov_rgb = _render_cov_glyphs(cov_mat, img_gray, fg_mask_pred, S)

    g_dir_rgb = _render_dir_quiver(
        geom[:, c_cov_end:c_dir_end], img_gray, fg_mask_pred, S,
        dir_target=dir_target,
    )

    tb.add_images(head.tag("pred/raw"), g_raw_rgb, global_step=epoch)
    tb.add_images(
        head.tag(f"pred/dir_{dir_target}"), g_dir_rgb, global_step=epoch,
    )
    tb.add_images(head.tag("pred/cov"), g_cov_rgb, global_step=epoch)


def _log_boundary(
    tb: Any,
    ctx: TagContext,
    preds: Dict[str, torch.Tensor],
    n: int,
    epoch: int,
    *,
    boundary_target: Optional[torch.Tensor] = None,
    tau: float = 1.0,
) -> None:
    """Log the boundary panels under ``{stage}/{mode}/boundary/``.

    Layout of the 10-channel boundary prediction / target:
      - ch 0    : ``raw`` (dense, fg + bg; logged as grayscale)
      - ch 1-3  : ``avg`` RGB (foreground-only; zero on background)
      - ch 4-9  : ``aff`` (T / B / U / D / L / R face affinities, Z-Y-X
                  order; grayscale)

    Predictions arriving here are post-activation: the model wrapper
    applies a single sigmoid to all 10 channels of the boundary head
    (every target lives in ``[0, 1]``) so this callback does **not**
    re-apply any activation; the ``clamp`` below is purely a guard for
    float-rounding drift.

    A second 6-face affinity is derived from the predicted avg (ch 1-3)
    via :func:`soft_aff_from_avg(avg, tau)` and logged under
    ``boundary/pred/avg/aff/{t,b,u,d,l,r}`` so the dual-aff supervision
    is visible in TensorBoard alongside the direct prediction.

    Panels are written under ``boundary/pred/*`` and, when
    ``boundary_target`` is supplied, also under ``boundary/true/*`` so
    the model output and its supervision signal can be compared
    side-by-side in TensorBoard.

    Args:
        tau: Bandwidth used for the derived ``aff_avg`` panels; should
            match :class:`brainbow.losses.BoundaryLoss.tau` for the
            visualisation to mirror the supervision signal.
    """
    if "boundary" not in preds:
        return
    head = ctx.for_head("boundary")
    bnd_pred_3d = preds["boundary"][:n]
    bnd_pred = _to_2d(bnd_pred_3d)
    _add_boundary_panels(tb, head, "pred", bnd_pred, epoch, is_pred=True)

    # Derived aff_avg = soft_aff_from_avg(predicted avg, tau).  Compute
    # on the 3-D prediction so the Z-direction shifts are meaningful,
    # then _to_2d the 6-channel result.
    if bnd_pred_3d.dim() == 5:
        avg_3d = bnd_pred_3d[:, _AVG_START:_AVG_END]
        derived_aff = soft_aff_from_avg(avg_3d, tau=tau)
        derived_aff_2d = _to_2d(derived_aff)
        for k, name in enumerate(_AFF_TAG_NAMES):
            panel = repeat(
                derived_aff_2d[:, k:k + 1].clamp(0.0, 1.0),
                "b 1 h w -> b 3 h w",
            )
            tb.add_images(
                head.tag(f"pred/avg/aff/{name}"), panel, global_step=epoch,
            )

    if boundary_target is not None:
        bnd_true = _to_2d(boundary_target[:n])
        _add_boundary_panels(tb, head, "true", bnd_true, epoch, is_pred=False)


def _add_boundary_panels(
    tb: Any,
    head: TagContext,
    variant: str,
    bnd: torch.Tensor,
    epoch: int,
    *,
    is_pred: bool,
) -> None:
    """Split a ``[n, 10, H, W]`` boundary tensor into its sub-panels.

    Channel layout (imported from
    :mod:`brainbow.losses.boundary` -- single source of truth)::

        ch 0                       : raw   (dense intensity; grayscale)
        ch ``_AVG_START``..         : avg   (centroid xyz colour)
              ``_AVG_END``
        ch ``_AFF_START``..         : aff   (``_N_AFF`` face affinities,
              ``_AFF_END``               Z-Y-X order: T B U D L R)

    Emits, under ``{variant}`` ∈ {``pred``, ``true``}:

    * ``pred/raw``  (only on the prediction side -- ``true/raw``
      duplicates ``{ctx.prefix}/true/image`` pixel-for-pixel by
      construction, so we skip it.)
    * ``{variant}/avg``
    * ``{variant}/aff/{t,b,u,d,l,r}``  (direction subtags nested under
      ``aff/``)

    Args:
        tb: TensorBoard SummaryWriter.
        head: ``TagContext`` for the boundary head (i.e.
            ``ctx.for_head("boundary")``).
        variant: ``"pred"`` or ``"true"``.  Becomes the next tag segment.
        bnd: ``[n, _BOUNDARY_CHANNELS, H, W]`` boundary map (already 2-D
            sliced).  For predictions this is the post-activation model
            output (the wrapper has applied sigmoid to **every**
            channel); for the ground-truth target it is the
            ``[0, 1]``-valued target tensor.  Both cases are already in
            display range and the ``clamp`` below is just a rounding
            guard.
        epoch: global step for TensorBoard.
        is_pred: ``True`` when ``bnd`` comes from the model (post-sigmoid
            on every channel), ``False`` when it is the ground-truth
            target tensor.  Gates the ``raw`` panel (true/raw is
            skipped because it duplicates ``{ctx.prefix}/true/image``).
    """
    if bnd.shape[1] != _BOUNDARY_CHANNELS:
        raise ValueError(
            f"_add_boundary_panels expects {_BOUNDARY_CHANNELS} channels, "
            f"got {bnd.shape[1]} (variant={variant!r})."
        )

    av = bnd[:, _AVG_START:_AVG_END].clamp(0.0, 1.0)
    if is_pred:
        raw = repeat(bnd[:, 0:1].clamp(0.0, 1.0), "b 1 h w -> b 3 h w")
        tb.add_images(head.tag(f"{variant}/raw"), raw, global_step=epoch)
    tb.add_images(head.tag(f"{variant}/avg"), av, global_step=epoch)

    # Affinity channels: the wrapper has already applied sigmoid on the
    # prediction side (the same sigmoid that covers ch 0-9), so for both
    # pred and true we just clamp into the valid display range.  The
    # 6 directions live under ``aff/{t,b,u,d,l,r}`` so TensorBoard
    # collapses them into a single sub-group next to ``avg`` and
    # ``raw``.
    aff = bnd[:, _AFF_START:_AFF_END].clamp(0.0, 1.0)
    assert aff.shape[1] == _N_AFF == len(_AFF_TAG_NAMES), (
        f"affinity channel / tag count mismatch: tensor has {aff.shape[1]} "
        f"channels, expected {_N_AFF} (tags={_AFF_TAG_NAMES})."
    )
    for k, name in enumerate(_AFF_TAG_NAMES):
        panel = repeat(aff[:, k:k + 1], "b 1 h w -> b 3 h w")
        tb.add_images(
            head.tag(f"{variant}/aff/{name}"), panel, global_step=epoch,
        )


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
    boundary_target: Optional[torch.Tensor] = None,
    boundary_tau: float = 1.0,
) -> None:
    """Log a standard set of prediction visualisations to TensorBoard.

    Emits, under ``ctx.prefix`` = ``{stage}/{mode}``:

    * mode level ::

        {ctx.prefix}/true/image
        {ctx.prefix}/true/label

    * per-head (only when the corresponding output is present in
      ``preds``) ::

        {ctx.prefix}/semantic/pred
        {ctx.prefix}/instance/pred/{pca|svd|umap}
        {ctx.prefix}/instance/pred/label                 (if clusterer)
        {ctx.prefix}/geometry/pred/dir_{centroid|skeleton}
        {ctx.prefix}/geometry/pred/cov
        {ctx.prefix}/geometry/pred/raw
        {ctx.prefix}/boundary/pred/raw
        {ctx.prefix}/boundary/pred/avg
        {ctx.prefix}/boundary/pred/aff/{t,b,u,d,l,r}     (direct aff;
            Z-Y-X order)
        {ctx.prefix}/boundary/pred/avg/aff/{t,b,u,d,l,r} (derived from
            predicted avgloc via soft_aff_from_avg)
        {ctx.prefix}/boundary/true/avg                   (if target;
            ``true/raw`` is omitted because it duplicates
            ``{ctx.prefix}/true/image``)
        {ctx.prefix}/boundary/true/aff/{t,b,u,d,l,r}     (if target)

    Args:
        tb: TensorBoard SummaryWriter.
        ctx: mode-level :class:`TagContext` (``head`` must be ``None``).
        images: ``[n, 1, H, W]`` input images (already 2-D sliced).
        labels: ``[n, H, W]`` instance labels (already 2-D sliced).
        preds: model output dict with any subset of ``semantic``,
            ``instance``, ``geometry``, ``boundary``.
        spatial_dims: 2 or 3 (controls geometry channel layout).
        n: number of images.
        epoch: global step for TensorBoard.
        clusterer: optional clusterer (SoftMeanShift / HDBSCAN / MeanShift)
            for producing the ``instance/pred/label`` panel.
        dir_target: ``"centroid"`` or ``"skeleton"`` (geometry head).
        active_classes: number of active semantic channels (from config).
        projection_algorithm: Manifold algorithm for the ``instance/*``
            panel.  One of ``"pca"`` (default), ``"svd"``, ``"umap"``.
        projection_backend: Backend for the projection.  ``"auto"`` picks
            cuML on CUDA, else a CPU fallback.  ``"cuml"`` forces GPU.
        boundary_target: optional ``[n, 10, D, H, W]`` ground-truth map
            to log alongside the boundary prediction.
        boundary_tau: bandwidth for the derived ``aff_avg`` panels;
            should match :class:`brainbow.losses.BoundaryLoss.tau`.
    """
    if ctx.head is not None:
        raise ValueError(
            f"_log_predictions expects a mode-level TagContext "
            f"(head=None); got head={ctx.head!r}."
        )

    img_gray = repeat(_normalise(images), "b 1 h w -> b 3 h w").contiguous()
    lbl_rgb = _label_to_rgb(labels.long())
    tb.add_images(ctx.tag("true/image"), img_gray, global_step=epoch)
    tb.add_images(ctx.tag("true/label"), lbl_rgb, global_step=epoch)

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
    _log_boundary(
        tb, ctx, preds, n, epoch,
        boundary_target=boundary_target, tau=boundary_tau,
    )


__all__ = [
    "_add_boundary_panels",
    "_log_boundary",
    "_log_geometry",
    "_log_instance",
    "_log_predictions",
    "_log_semantic",
]
