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
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from brainbow.losses.semantic import SemanticLoss
from brainbow.losses.instance import InstanceLoss
from brainbow.losses.geometry import GeometryLoss
from brainbow.losses.brainbow import BrainbowLoss


HeadConfig = Union[float, int, Mapping[str, Any]]


def _split_head(
    cfg: HeadConfig, default_weight: float,
) -> Tuple[float, Dict[str, Any]]:
    """Split a head config into ``(head_weight, sub_kwargs)``.

    Accepts either a scalar (legacy flat form -- only the head weight is
    set, sub-kwargs come from the outer flat kwargs) or a mapping of the
    form ``{weight: float, **sub_kwargs}`` (new nested form).
    """
    if isinstance(cfg, Mapping):
        d = dict(cfg)
        w = float(d.pop("weight", default_weight))
        return w, d
    return float(cfg), {}


# Legacy flat-kwarg groupings, used as a fallback when a head is passed
# as a scalar (or omits a given key from its nested dict).  Keys map
# 1:1 onto the sub-loss ``__init__`` parameter names.
_SEMANTIC_FLAT_KEYS: Tuple[str, ...] = (
    "weight_ce", "weight_iou", "weight_dice",
    "class_weights", "active_classes", "label_smoothing",
)
_INSTANCE_FLAT_KEYS: Tuple[str, ...] = (
    "weight_pull", "weight_push", "weight_norm", "weight_edge", "weight_bone",
    "delta_v", "delta_d", "normalize_embeddings", "max_hard_pairs",
    "anchor_to_centroid", "centroid_scale",
)
_BRAINBOW_FLAT_PREFIX: str = "brainbow_"


class CombinedLoss(nn.Module):
    """Weighted sum of SemanticLoss + InstanceLoss + GeometryLoss + BrainbowLoss.

    Each task loss is only constructed when its weight is strictly
    positive.  With ``learned_task_weights=True`` the scalar weights
    become learned log-variances (Kendall & Gal, 2018).

    Each ``weight_<head>`` parameter accepts either of two shapes:

    - **Nested mapping** (preferred, recommended for new configs)::

          weight_semantic:
            weight: 1.0                 # head scalar (0 disables the head)
            weight_ce: 1.0
            weight_dice: 1.0
            ...

      All keys except ``weight`` are forwarded 1:1 to the corresponding
      sub-loss constructor.  ``SemanticLoss`` is sigmoid-only (the model
      wrapper applies sigmoid to the semantic head before the loss sees
      it) so there is no ``semantic_mode`` knob; the same applies to the
      6 brainbow affinity channels (``BrainbowLoss``).

    - **Scalar float** (legacy flat form).  In this case the sub-loss
      kwargs are read from the matching top-level flat kwargs
      (e.g. ``weight_ce``, ``delta_v``, ``brainbow_weight_raw``).  This
      preserves compatibility with configs written against the pre-nest
      signature.

    For the ``brainbow`` head the legacy flat kwargs are prefixed with
    ``brainbow_`` to avoid colliding with ``GeometryLoss``'s
    ``weight_raw``; inside the nested mapping the prefix is dropped
    (``weight_raw``, ``weight_min``, ...).

    Args:
        spatial_dims: 2 or 3 (controls InstanceLoss / GeometryLoss).
        weight_semantic / weight_instance / weight_geometry /
            weight_brainbow: head config (float or nested mapping; see above).
        learned_task_weights: if ``True``, weights are learned via
            log-variances and the static weights become initialisations.
        ignore_index: label value excluded from every loss term.
        **legacy_kwargs: flat sub-loss kwargs for the legacy form;
            unknown keys that don't match any of the four sub-losses are
            forwarded to :class:`GeometryLoss` (which ignores unknowns
            via its own ``**kwargs``).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        weight_semantic: HeadConfig = 1.0,
        weight_instance: HeadConfig = 1.0,
        weight_geometry: HeadConfig = 0.0,
        weight_brainbow: HeadConfig = 0.0,
        learned_task_weights: bool = False,
        ignore_index: int = -100,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims

        # Split head configs into (scalar_weight, nested_kwargs).
        w_sem, sem_nested = _split_head(weight_semantic, 1.0)
        w_ins, ins_nested = _split_head(weight_instance, 1.0)
        w_geom, geom_nested = _split_head(weight_geometry, 0.0)
        w_bbow, bbow_nested = _split_head(weight_brainbow, 0.0)

        self.weight_semantic = w_sem
        self.weight_instance = w_ins
        self.weight_geometry = w_geom
        self.weight_brainbow = w_bbow
        self.learned_task_weights = learned_task_weights

        # Partition legacy flat kwargs by target head.  Nested entries
        # take priority; anything in ``legacy_kwargs`` only fills gaps.
        sem_legacy = {
            k: legacy_kwargs.pop(k) for k in _SEMANTIC_FLAT_KEYS
            if k in legacy_kwargs
        }
        ins_legacy = {
            k: legacy_kwargs.pop(k) for k in _INSTANCE_FLAT_KEYS
            if k in legacy_kwargs
        }
        bbow_legacy: Dict[str, Any] = {}
        for k in list(legacy_kwargs.keys()):
            if k.startswith(_BRAINBOW_FLAT_PREFIX):
                bbow_legacy[k[len(_BRAINBOW_FLAT_PREFIX):]] = legacy_kwargs.pop(k)

        # Legacy alias: ``brainbow_weight_aff`` was the single Dice-on-
        # affinities sub-weight before the CE/Dice/IoU split.  Map it to
        # ``weight_dice`` (its direct successor) unless the caller has
        # already set ``weight_dice`` explicitly.
        if "weight_aff" in bbow_legacy:
            legacy_aff = bbow_legacy.pop("weight_aff")
            bbow_legacy.setdefault("weight_dice", legacy_aff)

        # ``affinity_mode`` was the dead-end mode flag on BrainbowLoss
        # (sigmoid-only in practice); silently drop it from both legacy
        # flat kwargs and nested dicts so old configs keep loading.
        bbow_legacy.pop("affinity_mode", None)

        # Whatever remains (``dir_target``, ``weight_dir``, ``loss_dir``,
        # etc.) is GeometryLoss-bound in the legacy flat form.
        geom_legacy = legacy_kwargs

        def _merge(legacy: Dict[str, Any], nested: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(legacy)
            merged.update(nested)
            return merged

        sem_kwargs = _merge(sem_legacy, sem_nested)
        ins_kwargs = _merge(ins_legacy, ins_nested)
        geom_kwargs = _merge(geom_legacy, geom_nested)
        bbow_kwargs = _merge(bbow_legacy, bbow_nested)
        # Drop dead-end mode flag whether it came in via flat or nested.
        bbow_kwargs.pop("affinity_mode", None)

        if learned_task_weights:
            def _logvar(w: float) -> nn.Parameter:
                return nn.Parameter(
                    torch.tensor(math.log(1.0 / max(w, 1e-8))),
                    requires_grad=w > 0,
                )
            self.log_var_sem = _logvar(w_sem)
            self.log_var_ins = _logvar(w_ins)
            self.log_var_geom = _logvar(w_geom)
            self.log_var_bbow = _logvar(w_bbow)

        # SemanticLoss is sigmoid-only; silently drop any leftover
        # ``semantic_mode`` / ``mode`` keys from old configs so that
        # passing the legacy YAML doesn't raise an unexpected-kwarg error.
        for dead in ("semantic_mode", "mode"):
            sem_kwargs.pop(dead, None)

        self.semantic_loss: Optional[SemanticLoss] = (
            SemanticLoss(
                ignore_index=ignore_index,
                **sem_kwargs,
            )
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
        self.brainbow_loss: Optional[BrainbowLoss] = (
            BrainbowLoss(**bbow_kwargs) if w_bbow > 0 else None
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
                "max": zero, "raw": zero, "aff": zero,
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
            out["brainbow/loss/aff"] = bbow["aff"]

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
