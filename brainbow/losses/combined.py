"""
Combined segmentation loss.

Composes :class:`SemanticLoss`, :class:`InstanceLoss`,
:class:`GeometryLoss` and :class:`BoundaryLoss` into a single
dict-returning module shared by every Lightning module in the project
(Vista3D, Cosmos-Transfer 3D).  Any task loss whose head weight is
``0.0`` is **not instantiated** and contributes a cached zero scalar
to the total.

Configuration schema
--------------------
Each ``weight_<head>`` argument is a mapping::

    weight_semantic:
      weight: 1.0          # head scalar (0 disables the head entirely)
      weight_ce: 1.0
      weight_dice: 1.0
      ...                  # forwarded 1:1 to SemanticLoss(__init__)

The mapping's ``weight`` key is the head's scalar multiplier; every
other key is forwarded to the corresponding sub-loss constructor.
A bare scalar (``weight_semantic: 1.0``) is also accepted -- shorthand
for ``{weight: 1.0}`` with no sub-knobs.

When ``weight`` is omitted from the mapping it defaults to ``1.0`` --
a user who wrote a nested block clearly intended to enable the head;
silent disablement on a missing key would be a footgun.

Output dict
-----------
The returned dict uses a head-oriented key hierarchy that mirrors the
image-tag layout in
:class:`brainbow.callbacks.tensorboard.image_logger.ImageLogger`, so
each head's scalars cluster next to its images in TensorBoard::

    loss                                       # global total
    {head}/loss                                # per-head total
    {head}/loss/{component}                    # per-head breakdown
    {head}/loss/<field>[/<component>]          # per-field breakdown
                                               # (parallels image tag
                                               # {head}/pred/<field>/<panel>)

Concretely::

    instance/pred/emb/aff/{t,b,u,d,l,r}   <->  instance/loss/emb/aff
    boundary/pred/aff/{t,b,u,d,l,r}       <->  boundary/loss/aff
    boundary/pred/avg/aff/{t,b,u,d,l,r}   <->  boundary/loss/avg/aff
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from brainbow.losses.semantic import SemanticLoss
from brainbow.losses.instance import InstanceLoss
from brainbow.losses.geometry import GeometryLoss
from brainbow.losses.boundary import (
    _AFF_END,
    _AFF_START,
    _affinity_target_torch,
    BoundaryLoss,
)


HeadConfig = Union[float, int, Mapping[str, Any]]


def _split_head(cfg: HeadConfig) -> Tuple[float, Dict[str, Any]]:
    """Split a head config into ``(head_weight, sub_kwargs)``.

    Accepts either a scalar (only the head weight) or a mapping
    ``{weight: float, **sub_kwargs}``.  A nested mapping without
    ``weight`` defaults to ``weight=1.0`` -- assume the user intended
    to enable the head.
    """
    if isinstance(cfg, Mapping):
        d = dict(cfg)
        return float(d.pop("weight", 1.0)), d
    return float(cfg), {}


class CombinedLoss(nn.Module):
    """Weighted sum of the four task losses.

    Each task loss is constructed only when its head weight is
    strictly positive; disabled heads contribute a cached zero scalar
    to the total and are absent from the output dict.

    Args:
        spatial_dims: 2 or 3.  Forwarded to InstanceLoss / GeometryLoss.
        weight_semantic / weight_instance / weight_geometry /
            weight_boundary: head config (scalar or nested mapping;
            see module docstring).
        ignore_index: label value excluded from every loss term.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        weight_semantic: HeadConfig = 1.0,
        weight_instance: HeadConfig = 1.0,
        weight_geometry: HeadConfig = 0.0,
        weight_boundary: HeadConfig = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims

        w_sem, sem_kwargs = _split_head(weight_semantic)
        w_ins, ins_kwargs = _split_head(weight_instance)
        w_geom, geom_kwargs = _split_head(weight_geometry)
        w_bnd, bnd_kwargs = _split_head(weight_boundary)

        self.weight_semantic = w_sem
        self.weight_instance = w_ins
        self.weight_geometry = w_geom
        self.weight_boundary = w_bnd

        self.semantic_loss: Optional[SemanticLoss] = (
            SemanticLoss(ignore_index=ignore_index, **sem_kwargs)
            if w_sem > 0 else None
        )
        self.instance_loss: Optional[InstanceLoss] = (
            InstanceLoss(spatial_dims=spatial_dims, **ins_kwargs)
            if w_ins > 0 else None
        )
        self.geometry_loss: Optional[GeometryLoss] = (
            GeometryLoss(spatial_dims=spatial_dims, **geom_kwargs)
            if w_geom > 0 else None
        )
        self.boundary_loss: Optional[BoundaryLoss] = (
            BoundaryLoss(**bnd_kwargs) if w_bnd > 0 else None
        )

        # Cached zero scalar reused by every disabled-head branch in
        # ``forward`` -- avoids a fresh ``torch.tensor(0.0, device=...)``
        # allocation per step.  Lazily migrates to the active device
        # (DDP can move the criterion at any time).
        self.register_buffer(
            "_zero_scalar", torch.tensor(0.0), persistent=False,
        )

    def _zero(self, device: torch.device) -> torch.Tensor:
        z = self._zero_scalar
        if z.device != device:
            self._zero_scalar = z = torch.zeros((), device=device)
        return z

    # ------------------------------------------------------------------
    # Target precomputation (called once per step by the Lightning module)
    # ------------------------------------------------------------------

    def _build_targets(
        self,
        labels: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Optional[Dict[str, list]],          # geometry per-batch targets
        Optional[torch.Tensor],              # full 10-channel boundary target
        Optional[torch.Tensor],              # 6-channel aff target for InstanceLoss
    ]:
        """Precompute per-head shared targets in a single pass.

        The shared affinity target avoids rebuilding the same 6-channel
        ``_affinity_target_torch(labels)`` tensor twice when both
        :class:`BoundaryLoss` and :class:`InstanceLoss` (with
        ``weight_aff_emb > 0``) are active.  When boundary is on and
        the two heads agree on the ``background`` value, the boundary
        target's ch 4-9 slice is the aff target -- one
        ``_affinity_target_torch`` call covers both heads.  Otherwise
        each head receives a target built with its own ``background``.
        """
        geom_targets = None
        if self.geometry_loss is not None:
            geom_targets = self.geometry_loss.build_target(
                labels,
                direction=targets.get("label_direction") if targets else None,
                covariance=targets.get("label_covariance") if targets else None,
            )

        boundary_target: Optional[torch.Tensor] = None
        aff_target: Optional[torch.Tensor] = None

        if self.boundary_loss is not None and targets is not None:
            raw_image = targets.get("raw_image")
            if raw_image is not None:
                if raw_image.dim() == 5 and raw_image.shape[1] == 1:
                    raw_image = raw_image[:, 0]
                boundary_target = self.boundary_loss.build_target(
                    labels, raw_image,
                )

        if (
            self.instance_loss is not None
            and self.instance_loss.weight_aff_emb > 0
        ):
            ins_bg = self.instance_loss.background
            if (
                boundary_target is not None
                and self.boundary_loss is not None
                and self.boundary_loss.background == ins_bg
            ):
                aff_target = boundary_target[:, _AFF_START:_AFF_END]
            else:
                aff_target = _affinity_target_torch(
                    labels.long(), background=ins_bg,
                )

        return geom_targets, boundary_target, aff_target

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        labels = targets["labels"]
        zero = self._zero(labels.device)

        cached = targets.get("_cached_targets")
        if cached is not None:
            geom_targets, boundary_target, aff_target = cached
        else:
            geom_targets, boundary_target, aff_target = (
                self._build_targets(labels, targets)
            )

        # --- per-head sub-losses ---

        if self.semantic_loss is not None and "semantic" in predictions:
            sem = self.semantic_loss(
                predictions["semantic"], targets["semantic_labels"],
            )
        else:
            sem = {"loss": zero, "ce": zero, "iou": zero, "dice": zero}

        if self.instance_loss is not None and "instance" in predictions:
            ins = self.instance_loss(
                predictions["instance"], labels,
                cached_aff_target=aff_target,
            )
        else:
            ins = {
                "loss": zero, "pull": zero, "push": zero, "norm": zero,
                "aff_emb": zero,
            }

        if self.geometry_loss is not None and "geometry" in predictions:
            geom = self.geometry_loss(
                predictions["geometry"], labels,
                raw_image=targets.get("raw_image"),
                cached_targets=geom_targets,
            )
        else:
            geom = {"loss": zero, "dir": zero, "cov": zero, "raw": zero}

        if self.boundary_loss is not None and "boundary" in predictions:
            raw_image = targets.get("raw_image")
            if raw_image is None:
                raise KeyError(
                    "BoundaryLoss requires `targets['raw_image']` "
                    "(normalised [B, D, H, W] image) to build its target."
                )
            if raw_image.dim() == 5 and raw_image.shape[1] == 1:
                raw_image = raw_image[:, 0]
            bnd = self.boundary_loss(
                predictions["boundary"], labels, raw_image,
                cached_target=boundary_target,
            )
        else:
            bnd = {
                "loss": zero, "avg": zero, "raw": zero, "aff": zero,
                "aff_pred": zero, "aff_avg": zero,
                "aff_pred_ce": zero, "aff_pred_dice": zero, "aff_pred_iou": zero,
                "aff_avg_ce": zero, "aff_avg_dice": zero, "aff_avg_iou": zero,
            }

        # --- weighted sum ---

        total = (
            self.weight_semantic * sem["loss"]
            + self.weight_instance * ins["loss"]
            + self.weight_geometry * geom["loss"]
            + self.weight_boundary * bnd["loss"]
        )

        out: Dict[str, torch.Tensor] = {"loss": total}

        if self.semantic_loss is not None:
            out["semantic/loss"] = sem["loss"]
            out["semantic/loss/ce"] = sem["ce"]
            if self.semantic_loss.weight_iou > 0:
                out["semantic/loss/iou"] = sem["iou"]
            if self.semantic_loss.weight_dice > 0:
                out["semantic/loss/dice"] = sem["dice"]

        if self.instance_loss is not None:
            out["instance/loss"] = ins["loss"]
            out["instance/loss/pull"] = ins["pull"]
            out["instance/loss/push"] = ins["push"]
            out["instance/loss/norm"] = ins["norm"]
            # ``emb/aff`` mirrors the image tag
            # ``instance/pred/emb/aff/{...}``: kernel-derived 6-aff from
            # the predicted embedding.  Always emitted (zero scalar
            # when ``weight_aff_emb == 0``) so the scalar set is
            # stable.
            out["instance/loss/emb/aff"] = ins["aff_emb"]

        if self.geometry_loss is not None:
            out["geometry/loss"] = geom["loss"]
            out["geometry/loss/dir"] = geom["dir"]
            out["geometry/loss/cov"] = geom["cov"]
            out["geometry/loss/raw"] = geom["raw"]

        if self.boundary_loss is not None:
            out["boundary/loss"] = bnd["loss"]
            out["boundary/loss/raw"] = bnd["raw"]
            out["boundary/loss/avg"] = bnd["avg"]
            # Two parallel aff supervision paths.  Tag layout mirrors
            # the image tags::
            #     boundary/pred/aff/{...}      <-> boundary/loss/aff
            #     boundary/pred/avg/aff/{...}  <-> boundary/loss/avg/aff
            # Path totals are always emitted (stable scalar set);
            # per-sub CE / Dice / IoU breakdowns are surfaced only when
            # their weight is non-zero, mirroring the semantic head's
            # pattern above.
            out["boundary/loss/aff"] = bnd["aff_pred"]
            out["boundary/loss/avg/aff"] = bnd["aff_avg"]
            if self.boundary_loss.weight_ce > 0:
                out["boundary/loss/aff/ce"] = bnd["aff_pred_ce"]
                out["boundary/loss/avg/aff/ce"] = bnd["aff_avg_ce"]
            if self.boundary_loss.weight_dice > 0:
                out["boundary/loss/aff/dice"] = bnd["aff_pred_dice"]
                out["boundary/loss/avg/aff/dice"] = bnd["aff_avg_dice"]
            if self.boundary_loss.weight_iou > 0:
                out["boundary/loss/aff/iou"] = bnd["aff_pred_iou"]
                out["boundary/loss/avg/aff/iou"] = bnd["aff_avg_iou"]

        return out
