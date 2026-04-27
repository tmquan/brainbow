"""
Combined segmentation loss.

Composes :class:`SemanticLoss`, :class:`InstanceLoss`,
:class:`GeometryLoss` and :class:`BoundaryLoss` into a single dict-
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

from __future__ import annotations

import math
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


def _split_head(
    cfg: HeadConfig, default_weight: float,
) -> Tuple[float, Dict[str, Any]]:
    """Split a head config into ``(head_weight, sub_kwargs)``.

    Accepts either a scalar (flat form -- only the head weight is set,
    sub-kwargs come from the outer flat kwargs) or a mapping of the
    form ``{weight: float, **sub_kwargs}`` (nested form, recommended).
    """
    if isinstance(cfg, Mapping):
        d = dict(cfg)
        w = float(d.pop("weight", default_weight))
        return w, d
    return float(cfg), {}


# Flat-kwarg groupings, used as a fallback when a head is passed as a
# scalar (or omits a given key from its nested dict).  Keys map 1:1 onto
# the sub-loss ``__init__`` parameter names.
_SEMANTIC_FLAT_KEYS: Tuple[str, ...] = (
    "weight_ce", "weight_iou", "weight_dice",
    "class_weights", "active_classes", "label_smoothing",
)
_INSTANCE_FLAT_KEYS: Tuple[str, ...] = (
    "weight_pull", "weight_push", "weight_norm", "weight_edge", "weight_bone",
    "weight_aff_emb", "tau",
    "delta_v", "delta_d", "normalize_embeddings", "max_hard_pairs",
    "anchor_to_centroid", "centroid_scale", "aff_eps", "background",
)
# Flat-kwarg prefix for the boundary head, used to disambiguate from
# ``GeometryLoss``'s ``weight_raw`` etc.  Inside the nested mapping the
# prefix is dropped (``weight_raw``, ``weight_avg``, ``weight_aff_pred``,
# ``weight_aff_avg``, ...).
_BOUNDARY_FLAT_PREFIX: str = "boundary_"


class CombinedLoss(nn.Module):
    """Weighted sum of SemanticLoss + InstanceLoss + GeometryLoss + BoundaryLoss.

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
      6 boundary affinity channels (``BoundaryLoss``).

    - **Scalar float** (flat form).  In this case the sub-loss kwargs
      are read from the matching top-level flat kwargs
      (e.g. ``weight_ce``, ``delta_v``, ``boundary_weight_raw``).
      Boundary-head flat kwargs are prefixed with ``boundary_`` to
      avoid colliding with ``GeometryLoss``'s ``weight_raw``; inside
      the nested mapping the prefix is dropped (``weight_raw``,
      ``weight_avg``, ``weight_aff_pred``, ``weight_aff_avg``, ...).

    Args:
        spatial_dims: 2 or 3 (controls InstanceLoss / GeometryLoss).
        weight_semantic / weight_instance / weight_geometry /
            weight_boundary: head config (float or nested mapping; see above).
        learned_task_weights: if ``True``, weights are learned via
            log-variances and the static weights become initialisations.
        ignore_index: label value excluded from every loss term.
        **flat_kwargs: flat sub-loss kwargs for the scalar form;
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
        weight_boundary: HeadConfig = 0.0,
        learned_task_weights: bool = False,
        ignore_index: int = -100,
        **flat_kwargs: Any,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims

        # Split head configs into (scalar_weight, nested_kwargs).
        w_sem, sem_nested = _split_head(weight_semantic, 1.0)
        w_ins, ins_nested = _split_head(weight_instance, 1.0)
        w_geom, geom_nested = _split_head(weight_geometry, 0.0)
        w_bnd, bnd_nested = _split_head(weight_boundary, 0.0)

        self.weight_semantic = w_sem
        self.weight_instance = w_ins
        self.weight_geometry = w_geom
        self.weight_boundary = w_bnd
        self.learned_task_weights = learned_task_weights

        # Partition flat kwargs by target head.  Nested entries take
        # priority; ``flat_kwargs`` only fills gaps.  Whatever remains
        # after the semantic / instance / boundary partitions is
        # interpreted as :class:`GeometryLoss`-bound (it is the only
        # sub-loss that accepts ``**kwargs``-style overflow).
        sem_flat = {
            k: flat_kwargs.pop(k) for k in _SEMANTIC_FLAT_KEYS if k in flat_kwargs
        }
        ins_flat = {
            k: flat_kwargs.pop(k) for k in _INSTANCE_FLAT_KEYS if k in flat_kwargs
        }
        bnd_flat: Dict[str, Any] = {}
        for k in list(flat_kwargs.keys()):
            if k.startswith(_BOUNDARY_FLAT_PREFIX):
                bnd_flat[k[len(_BOUNDARY_FLAT_PREFIX):]] = flat_kwargs.pop(k)
        geom_flat = flat_kwargs

        def _merge(flat: Dict[str, Any], nested: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(flat)
            merged.update(nested)
            return merged

        sem_kwargs = _merge(sem_flat, sem_nested)
        ins_kwargs = _merge(ins_flat, ins_nested)
        geom_kwargs = _merge(geom_flat, geom_nested)
        bnd_kwargs = _merge(bnd_flat, bnd_nested)

        if learned_task_weights:
            def _logvar(w: float) -> nn.Parameter:
                return nn.Parameter(
                    torch.tensor(math.log(1.0 / max(w, 1e-8))),
                    requires_grad=w > 0,
                )
            self.log_var_sem = _logvar(w_sem)
            self.log_var_ins = _logvar(w_ins)
            self.log_var_geom = _logvar(w_geom)
            self.log_var_bnd = _logvar(w_bnd)

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
        self.boundary_loss: Optional[BoundaryLoss] = (
            BoundaryLoss(**bnd_kwargs) if w_bnd > 0 else None
        )

        # Shared zero-scalar placeholder used to fill loss-dict entries
        # for disabled heads.  Allocated lazily on the right device on
        # first ``forward`` and cached as a non-persistent buffer so the
        # module's ``state_dict`` doesn't acquire it.  Avoids a fresh
        # ``torch.tensor(0.0, device=...)`` allocation every step.
        self.register_buffer(
            "_zero_scalar", torch.tensor(0.0), persistent=False,
        )

    def _zero(self, device: torch.device) -> torch.Tensor:
        """Return the cached 0-dim float scalar on ``device``.

        The buffer is float32 (Lightning logs scalars as float anyway);
        if the criterion is moved to a different device by Lightning we
        re-materialise lazily and update the buffer in place.
        """
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
    ):
        """Precompute per-head shared targets in a single pass.

        Plural ``_build_targets`` (vs. the per-loss ``build_target``)
        because this method orchestrates **multiple** heads' targets in
        one pass and returns them as a tuple for the
        ``targets["_cached_weights"]`` cache.

        Returns a 4-tuple::

            (
                (w_edge, w_bone),     # InstanceLoss per-voxel weights
                geom_targets,         # GeometryLoss per-batch targets
                boundary_target,      # full 10-channel target or None
                aff_target,           # 6-channel aff target or None
            )

        The shared affinity target avoids rebuilding the same 6-channel
        ``_affinity_target_torch(labels)`` tensor twice when both
        :class:`BoundaryLoss` and :class:`InstanceLoss` (with
        ``weight_aff_emb > 0``) are active.  When boundary is on and
        the two heads agree on the ``background`` value, the boundary
        target's ch 4-9 slice is the aff target -- one
        ``_affinity_target_torch`` call covers both heads.  Otherwise
        each head receives a target built with its own ``background``
        setting.
        """
        ins_weights = (
            self.instance_loss.compute_weights(labels)
            if self.instance_loss is not None else (None, None)
        )

        geom_targets = None
        if self.geometry_loss is not None:
            # ``build_target`` accepts the (direction, covariance) fast
            # path directly when the datamodule has precomputed them;
            # otherwise it runs the on-the-fly transform per batch
            # element.  One entry point either way -- no separate alias.
            geom_targets = self.geometry_loss.build_target(
                labels,
                direction=targets.get("label_direction") if targets else None,
                covariance=targets.get("label_covariance") if targets else None,
            )

        boundary_target: Optional[torch.Tensor] = None
        aff_target: Optional[torch.Tensor] = None

        # Build the full 10-channel boundary target up-front when the
        # head is active and we have ``raw_image`` -- BoundaryLoss can
        # then skip its rebuild via ``cached_target=...``.
        if self.boundary_loss is not None and targets is not None:
            raw_image = targets.get("raw_image")
            if raw_image is not None:
                if raw_image.dim() == 5 and raw_image.shape[1] == 1:
                    raw_image = raw_image[:, 0]
                boundary_target = self.boundary_loss.build_target(
                    labels, raw_image,
                )

        # Resolve the aff target for InstanceLoss (when its aff_emb path
        # is active).  Reuse the boundary slice when the two heads' bg
        # settings agree; otherwise each head builds with its own bg.
        if (
            self.instance_loss is not None
            and getattr(self.instance_loss, "weight_aff_emb", 0.0) > 0
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

        return ins_weights, geom_targets, boundary_target, aff_target

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

        cached = targets.get("_cached_weights")
        if cached is not None:
            (w_edge, w_bone), geom_targets, boundary_target, aff_target = cached
        else:
            (w_edge, w_bone), geom_targets, boundary_target, aff_target = (
                self._build_targets(labels, targets)
            )

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
                cached_aff_target=aff_target,
            )
        else:
            ins = {
                "loss": zero, "pull": zero, "push": zero, "norm": zero,
                "aff_emb": zero,
            }

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

        # Boundary
        if self.boundary_loss is not None and "boundary" in predictions:
            raw_image = targets.get("raw_image")
            if raw_image is None:
                raise KeyError(
                    "BoundaryLoss requires `targets['raw_image']` "
                    "(normalised [B,D,H,W] image) to build its target."
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
            if self.weight_boundary > 0:
                total = total + torch.exp(-self.log_var_bnd) * bnd["loss"] + self.log_var_bnd
        else:
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
            # Mirror the boundary head: always emit ``aff_emb`` even when
            # ``weight_aff_emb == 0`` (InstanceLoss._maybe_add_aff_emb
            # fills it with a zero scalar).  Keeps the scalar tag set
            # stable across runs and matches the ``aff_pred`` / ``aff_avg``
            # convention below.
            out["instance/loss/aff_emb"] = ins["aff_emb"]

        if self.geometry_loss is not None:
            out["geometry/loss"] = geom["loss"]
            out["geometry/loss/dir"] = geom["dir"]
            out["geometry/loss/cov"] = geom["cov"]
            out["geometry/loss/raw"] = geom["raw"]

        if self.boundary_loss is not None:
            out["boundary/loss"] = bnd["loss"]
            out["boundary/loss/raw"] = bnd["raw"]
            out["boundary/loss/avg"] = bnd["avg"]
            out["boundary/loss/aff_avg"] = bnd["aff_avg"]
            out["boundary/loss/aff"] = bnd["aff"]
            out["boundary/loss/aff_pred"] = bnd["aff_pred"]

        if self.learned_task_weights:
            if self.weight_semantic > 0:
                out["eff_w/semantic"] = torch.exp(-self.log_var_sem).detach()
            if self.weight_instance > 0:
                out["eff_w/instance"] = torch.exp(-self.log_var_ins).detach()
            if self.weight_geometry > 0:
                out["eff_w/geometry"] = torch.exp(-self.log_var_geom).detach()
            if self.weight_boundary > 0:
                out["eff_w/boundary"] = torch.exp(-self.log_var_bnd).detach()

        return out
