"""
Combined segmentation loss.

Composes :class:`SemanticLoss`, :class:`InstanceLoss`,
:class:`GeometryLoss` and :class:`BrainbowLoss` into a single dict-
returning module shared by every Lightning module in the project
(Vista3D, Cosmos-Transfer 3D).  Any task loss whose weight is
``0.0`` is not instantiated and contributes zero to the total.

The returned dict uses a **head-oriented** key hierarchy so that
TensorBoard clusters each head's scalars together next to the images
emitted by :mod:`brainbow.callbacks.tensorboard`::

    loss                              # global total
    {head}/loss                       # per-head total (semantic, instance, ...)
    {head}/loss/{component}           # per-head loss breakdown
    eff_w/{head}                      # effective task weights (learned mode)
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from brainbow.losses.semantic import SemanticLoss
from brainbow.losses.instance import InstanceLoss
from brainbow.losses.geometry import GeometryLoss
from brainbow.losses.brainbow import BrainbowLoss


class CombinedLoss(nn.Module):
    """Weighted sum of SemanticLoss + InstanceLoss + GeometryLoss + BrainbowLoss.

    Each task loss is only constructed when its weight is strictly
    positive.  With ``learned_task_weights=True`` the scalar weights
    become learned log-variances (Kendall & Gal, 2018).

    Args:
        spatial_dims: 2 or 3 (controls InstanceLoss / GeometryLoss).
        weight_semantic / weight_instance / weight_geometry /
            weight_brainbow: task weights.
        learned_task_weights: if ``True``, weights are learned via
            log-variances and the static weights become initialisations.
        **sub_kwargs: forwarded to the four sub-losses.  See
            :class:`SemanticLoss`, :class:`InstanceLoss`, :class:`GeometryLoss`,
            :class:`BrainbowLoss`.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        weight_semantic: float = 1.0,
        weight_instance: float = 1.0,
        weight_geometry: float = 0.0,
        weight_brainbow: float = 0.0,
        learned_task_weights: bool = False,
        # SemanticLoss kwargs
        semantic_mode: str = "sigmoid",
        weight_ce: float = 1.0,
        weight_iou: float = 0.0,
        weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        active_classes: Optional[int] = None,
        label_smoothing: float = 0.0,
        # InstanceLoss kwargs
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        normalize_embeddings: bool = False,
        max_hard_pairs: int = 0,
        anchor_to_centroid: bool = False,
        centroid_scale: float = 5.0,
        # BrainbowLoss kwargs (prefixed with ``brainbow_`` because
        # ``weight_raw`` would otherwise collide with GeometryLoss's
        # ``weight_raw`` when both heads are active).
        brainbow_loss_loc: str = "smooth_l1",
        brainbow_loss_raw: str = "l1",
        brainbow_weight_min: float = 1.0,
        brainbow_weight_avg: float = 1.0,
        brainbow_weight_max: float = 1.0,
        brainbow_weight_raw: float = 1.0,
        brainbow_foreground_only_loc: bool = True,
        # GeometryLoss kwargs (forwarded via **geom_kwargs)
        **geom_kwargs,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.weight_semantic = weight_semantic
        self.weight_instance = weight_instance
        self.weight_geometry = weight_geometry
        self.weight_brainbow = weight_brainbow
        self.learned_task_weights = learned_task_weights

        if learned_task_weights:
            def _logvar(w: float) -> nn.Parameter:
                return nn.Parameter(
                    torch.tensor(math.log(1.0 / max(w, 1e-8))),
                    requires_grad=w > 0,
                )
            self.log_var_sem = _logvar(weight_semantic)
            self.log_var_ins = _logvar(weight_instance)
            self.log_var_geom = _logvar(weight_geometry)
            self.log_var_bbow = _logvar(weight_brainbow)

        self.semantic_loss: Optional[SemanticLoss] = (
            SemanticLoss(
                mode=semantic_mode,
                weight_ce=weight_ce,
                weight_iou=weight_iou,
                weight_dice=weight_dice,
                class_weights=class_weights,
                ignore_index=ignore_index,
                active_classes=active_classes,
                label_smoothing=label_smoothing,
            )
            if weight_semantic > 0 else None
        )
        self.instance_loss: Optional[InstanceLoss] = (
            InstanceLoss(
                spatial_dims=spatial_dims,
                weight_pull=weight_pull,
                weight_push=weight_push,
                weight_norm=weight_norm,
                weight_edge=weight_edge,
                weight_bone=weight_bone,
                delta_v=delta_v,
                delta_d=delta_d,
                normalize_embeddings=normalize_embeddings,
                max_hard_pairs=max_hard_pairs,
                anchor_to_centroid=anchor_to_centroid,
                centroid_scale=centroid_scale,
            )
            if weight_instance > 0 else None
        )
        self.geometry_loss: Optional[GeometryLoss] = (
            GeometryLoss(spatial_dims=spatial_dims, **geom_kwargs)
            if weight_geometry > 0 else None
        )
        self.brainbow_loss: Optional[BrainbowLoss] = (
            BrainbowLoss(
                loss_loc=brainbow_loss_loc,
                loss_raw=brainbow_loss_raw,
                weight_min=brainbow_weight_min,
                weight_avg=brainbow_weight_avg,
                weight_max=brainbow_weight_max,
                weight_raw=brainbow_weight_raw,
                foreground_only_loc=brainbow_foreground_only_loc,
            )
            if weight_brainbow > 0 else None
        )

    # ------------------------------------------------------------------
    # Target precomputation (called once per step by the Lightning module)
    # ------------------------------------------------------------------

    def _compute_targets(
        self,
        labels: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Precompute instance pixel weights and geometry targets."""
        ins_weights = (
            self.instance_loss.compute_weights(labels)
            if self.instance_loss is not None else (None, None)
        )

        geom_targets = None
        if self.geometry_loss is not None:
            if (
                targets is not None
                and "label_direction" in targets
                and "label_covariance" in targets
            ):
                geom_targets = self.geometry_loss.targets_from_pipeline(
                    targets["label_direction"],
                    targets["label_covariance"],
                )
            else:
                geom_targets = self.geometry_loss.compute_targets(labels)

        return ins_weights, geom_targets

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        labels = targets["labels"]
        zero = torch.tensor(0.0, device=labels.device)

        cached = targets.get("_cached_weights")
        if cached is not None:
            (w_edge, w_bone), geom_targets = cached
        else:
            (w_edge, w_bone), geom_targets = self._compute_targets(labels, targets)

        # Semantic
        if self.semantic_loss is not None and "semantic" in predictions:
            sem = self.semantic_loss(
                predictions["semantic"], targets["semantic_labels"],
            )
        else:
            sem = {"loss": zero, "ce": zero, "iou": zero, "dice": zero}

        # Instance
        if self.instance_loss is not None and "instance" in predictions:
            sem_ids = targets.get("semantic_ids") or predictions.get("semantic_ids")
            ins = self.instance_loss(
                predictions["instance"],
                labels,
                semantic_ids=sem_ids,
                weight_edge=w_edge,
                weight_bone=w_bone,
            )
        else:
            ins = {"loss": zero, "pull": zero, "push": zero, "norm": zero}

        # Geometry
        if self.geometry_loss is not None and "geometry" in predictions:
            geom = self.geometry_loss(
                predictions["geometry"],
                labels,
                raw_image=targets.get("raw_image"),
                cached_targets=geom_targets,
            )
        else:
            geom = {"loss": zero, "dir": zero, "cov": zero, "raw": zero}

        # Brainbow
        if self.brainbow_loss is not None and "brainbow" in predictions:
            raw_image = targets.get("raw_image")
            if raw_image is None:
                raise KeyError(
                    "BrainbowLoss requires `targets['raw_image']` "
                    "(normalised [B,D,H,W] image) to build its target."
                )
            if raw_image.dim() == 5 and raw_image.shape[1] == 1:
                raw_image = raw_image[:, 0]
            bbow = self.brainbow_loss(predictions["brainbow"], labels, raw_image)
        else:
            bbow = {
                "loss": zero, "min": zero, "avg": zero,
                "max": zero, "raw": zero,
            }

        # Total
        if self.learned_task_weights:
            total = zero.clone()
            if self.weight_semantic > 0:
                total = total + torch.exp(-self.log_var_sem) * sem["loss"] + self.log_var_sem
            if self.weight_instance > 0:
                total = total + torch.exp(-self.log_var_ins) * ins["loss"] + self.log_var_ins
            if self.weight_geometry > 0:
                total = total + torch.exp(-self.log_var_geom) * geom["loss"] + self.log_var_geom
            if self.weight_brainbow > 0:
                total = total + torch.exp(-self.log_var_bbow) * bbow["loss"] + self.log_var_bbow
        else:
            total = (
                self.weight_semantic * sem["loss"]
                + self.weight_instance * ins["loss"]
                + self.weight_geometry * geom["loss"]
                + self.weight_brainbow * bbow["loss"]
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

        if self.geometry_loss is not None:
            out["geometry/loss"] = geom["loss"]
            out["geometry/loss/dir"] = geom["dir"]
            out["geometry/loss/cov"] = geom["cov"]
            out["geometry/loss/raw"] = geom["raw"]

        if self.brainbow_loss is not None:
            out["brainbow/loss"] = bbow["loss"]
            out["brainbow/loss/min"] = bbow["min"]
            out["brainbow/loss/avg"] = bbow["avg"]
            out["brainbow/loss/max"] = bbow["max"]
            out["brainbow/loss/raw"] = bbow["raw"]

        if self.learned_task_weights:
            if self.weight_semantic > 0:
                out["eff_w/semantic"] = torch.exp(-self.log_var_sem).detach()
            if self.weight_instance > 0:
                out["eff_w/instance"] = torch.exp(-self.log_var_ins).detach()
            if self.weight_geometry > 0:
                out["eff_w/geometry"] = torch.exp(-self.log_var_geom).detach()
            if self.weight_brainbow > 0:
                out["eff_w/brainbow"] = torch.exp(-self.log_var_bbow).detach()

        return out
