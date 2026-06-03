"""
Affinity + foreground loss for the Mutex Watershed head.

The model emits a single ``[B, HEAD_CHANNELS, D, H, W]`` head whose
channels (all post-sigmoid probabilities) are the per-offset affinities
:data:`brainbow.losses._common.AFF_SLICE` and a scalar foreground
probability :data:`brainbow.losses._common.FG_SLICE`.  This loss
supervises both:

* **aff** -- per-voxel affinity ``aff[o, v] = P(label[v] == label[v+o])``
  against the binary target from
  :func:`~brainbow.losses._common.affinity_target_from_offsets`, with a
  composite (masked BCE + masked soft-Dice + optional focal).  Edges
  with a non-foreground endpoint are masked out (see
  :func:`~brainbow.losses._common.affinity_validity_mask`), and the
  short-range *attractive* offsets and long-range *repulsive* offsets
  carry independent weights.
* **sem** -- the foreground / boundary (semantic) probability against
  ``labels > 0``, via the shared
  :class:`~brainbow.losses.dice_bce_focal.DiceBCEFocalLoss` composite.
* **raw** -- the linear reconstruction channel against the (normalised)
  input EM intensity, via a plain L1 / MSE regression (an auxiliary
  self-supervised signal that stabilises the shared decoder features).

At evaluation / inference the predicted affinities are agglomerated into
instances by the Mutex Watershed (Wolf et al. 2018); see
:mod:`brainbow.inference.mutex_watershed`.  This loss is the training
supervisor only.

Configuration schema
--------------------
``weight_aff`` / ``weight_sem`` / ``weight_raw`` are each a scalar (just
the field weight) or a mapping ``{weight: ..., **sub_kwargs}``::

    weight_aff:
      weight: 1.0
      lambda_bce: 1.0
      lambda_dice: 1.0
      lambda_focal: 0.0
      gamma: 2.0
      attractive_weight: 1.0     # multiplier on the nn (attractive) offsets
      repulsive_weight: 1.0      # multiplier on the long-range offsets
      mask_to_foreground: true   # drop edges with a background endpoint
    weight_sem:
      weight: 1.0
      lambda_bce: 1.0
      lambda_dice: 1.0
      lambda_focal: 1.0
      gamma: 2.0
    weight_raw:
      weight: 1.0
      loss: l1                   # l1 / mse / smooth_l1

Output dict::

    loss        # global weighted total
    loss/aff    # affinity composite (un-weighted)
    loss/sem    # foreground (semantic) composite (un-weighted)
    loss/raw    # raw reconstruction (un-weighted)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from brainbow.losses._common import (
    AFF_SLICE,
    AFFINITY_OFFSETS,
    SEM_SLICE,
    HEAD_CHANNELS,
    N_ATTRACTIVE,
    RAW_SLICE,
    affinity_target_from_offsets,
    affinity_validity_mask,
    canonical_regression_name,
    regression_loss_fn,
    stable_bce_on_probs,
)
from brainbow.losses.dice_bce_focal import DiceBCEFocalLoss

HeadConfig = Union[float, int, Mapping[str, Any]]


def _split_field(cfg: HeadConfig) -> Tuple[float, Dict[str, Any]]:
    """Split ``weight_<field>`` into ``(weight, sub_kwargs)``.

    Scalar shorthand: ``weight_fg: 1.0`` == ``weight_fg: {weight: 1.0}``.
    A nested mapping without ``weight:`` defaults to ``weight: 1.0``.
    """
    if isinstance(cfg, Mapping):
        d = dict(cfg)
        return float(d.pop("weight", 1.0)), d
    return float(cfg), {}


class AffinityFGLoss(nn.Module):
    """Affinity + foreground loss for the Mutex Watershed head.

    Args:
        weight_aff: Field-level config for the affinity head (scalar or
            ``{weight, lambda_bce, lambda_dice, lambda_focal, gamma,
            attractive_weight, repulsive_weight, mask_to_foreground}``).
        weight_sem: Field-level config for the foreground (semantic) head
            (scalar or ``{weight, lambda_bce, lambda_dice, lambda_focal,
            gamma}``).
        weight_raw: Field-level config for the raw reconstruction head
            (scalar or ``{weight, loss}`` with ``loss`` in
            ``l1 / mse / smooth_l1``).
        offsets: Affinity offsets ``(dz, dy, dx)``.  Defaults to
            :data:`brainbow.losses._common.AFFINITY_OFFSETS`.
        n_attractive: Number of leading offsets treated as attractive
            (the rest are repulsive).  Only affects the per-offset loss
            weighting here; the actual mutex behaviour lives in the
            agglomerator.
        background: Label value treated as background when building the
            affinity target (its rows are zeroed).  ``None`` disables.
        ignore_index: Label value masked out of the foreground target
            (those voxels contribute as background, like ``CombinedLoss``).
    """

    num_channels: int = HEAD_CHANNELS

    def __init__(
        self,
        weight_aff: HeadConfig = 1.0,
        weight_sem: HeadConfig = 1.0,
        weight_raw: HeadConfig = 1.0,
        *,
        offsets: Sequence[Sequence[int]] = AFFINITY_OFFSETS,
        n_attractive: int = N_ATTRACTIVE,
        background: Optional[int] = -1,
        ignore_index: int = -100,
        eps: float = 1e-7,
        dice_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.offsets = tuple(tuple(int(c) for c in o) for o in offsets)
        self.n_attractive = int(n_attractive)
        self.background = int(background) if background is not None else None
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.dice_eps = float(dice_eps)

        # ----- aff -----
        self.weight_aff, aff_kw = _split_field(weight_aff)
        self.aff_lambda_bce = float(aff_kw.pop("lambda_bce", 1.0))
        self.aff_lambda_dice = float(aff_kw.pop("lambda_dice", 1.0))
        self.aff_lambda_focal = float(aff_kw.pop("lambda_focal", 0.0))
        self.aff_gamma = float(aff_kw.pop("gamma", 2.0))
        self.attractive_weight = float(aff_kw.pop("attractive_weight", 1.0))
        self.repulsive_weight = float(aff_kw.pop("repulsive_weight", 1.0))
        self.mask_to_foreground = bool(aff_kw.pop("mask_to_foreground", True))
        if aff_kw:
            import warnings

            warnings.warn(
                f"AffinityFGLoss: ignoring unknown weight_aff keys: "
                f"{sorted(aff_kw)}",
                stacklevel=2,
            )

        # ----- sem (composite Dice + BCE + Focal on the fg probability) -----
        self.weight_sem, sem_kw = _split_field(weight_sem)
        self._sem_loss = DiceBCEFocalLoss(
            lambda_dice=float(sem_kw.pop("lambda_dice", 1.0)),
            lambda_bce=float(sem_kw.pop("lambda_bce", 1.0)),
            lambda_focal=float(sem_kw.pop("lambda_focal", 1.0)),
            gamma=float(sem_kw.pop("gamma", 2.0)),
            smooth_nr=float(sem_kw.pop("smooth_nr", self.dice_eps)),
            smooth_dr=float(sem_kw.pop("smooth_dr", self.dice_eps)),
            eps=self.eps,
        )
        if sem_kw:
            import warnings

            warnings.warn(
                f"AffinityFGLoss: ignoring unknown weight_sem keys: "
                f"{sorted(sem_kw)}",
                stacklevel=2,
            )

        # ----- raw (linear L1 / MSE reconstruction of the input EM) -----
        self.weight_raw, raw_kw = _split_field(weight_raw)
        self.loss_raw = canonical_regression_name(raw_kw.pop("loss", "l1"))
        self._raw_fn = regression_loss_fn(self.loss_raw)
        if raw_kw:
            import warnings

            warnings.warn(
                f"AffinityFGLoss: ignoring unknown weight_raw keys: "
                f"{sorted(raw_kw)}",
                stacklevel=2,
            )

        # Per-offset channel weight vector (registered so it follows the
        # module's device / dtype under FSDP MixedPrecision).
        ch_w = torch.full((len(self.offsets),), self.repulsive_weight)
        ch_w[: self.n_attractive] = self.attractive_weight
        self.register_buffer("_offset_weights", ch_w, persistent=False)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_targets(
        self,
        labels: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the per-step shared targets used by :meth:`forward`.

        Returns ``{"aff": [B, N_AFF, ...], "aff_mask": [B, N_AFF, ...]}``
        (the binary affinity target and its foreground validity mask).
        The foreground target is derived from ``labels`` directly in
        :meth:`forward`, so it is not cached.
        """
        out: Dict[str, torch.Tensor] = {}
        if self.weight_aff > 0:
            out["aff"] = affinity_target_from_offsets(
                labels.long(), self.offsets, background=self.background,
            )
            if self.mask_to_foreground:
                out["aff_mask"] = affinity_validity_mask(
                    labels > 0, self.offsets,
                )
        return out

    # ------------------------------------------------------------------
    # Sub-losses
    # ------------------------------------------------------------------

    def _loss_aff(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Masked, offset-weighted composite on the affinity head.

        ``probs`` / ``target`` are ``[B, N_AFF, D, H, W]`` (probs already
        sigmoided).  ``mask`` is ``[B, N_AFF, D, H, W]`` (or ``None`` for
        all-valid).  Channels are weighted by :attr:`_offset_weights`
        (attractive vs repulsive).
        """
        probs = probs.float()
        target = target.float()
        ch_w = self._offset_weights.to(probs.device).view(1, -1, 1, 1, 1)
        if mask is None:
            mask = torch.ones_like(target)
        else:
            mask = mask.float()
        weighted_mask = mask * ch_w
        denom = weighted_mask.sum().clamp_min(1.0)

        total = probs.new_zeros(())

        if self.aff_lambda_bce > 0:
            bce = stable_bce_on_probs(probs, target, eps=self.eps)
            total = total + self.aff_lambda_bce * (bce * weighted_mask).sum() / denom

        if self.aff_lambda_focal > 0:
            p = probs.clamp(self.eps, 1.0 - self.eps)
            p_t = target * p + (1.0 - target) * (1.0 - p)
            focal = (1.0 - p_t).pow(self.aff_gamma) * (-p_t.log())
            total = total + self.aff_lambda_focal * (
                focal * weighted_mask
            ).sum() / denom

        if self.aff_lambda_dice > 0:
            # Masked soft Dice on probabilities, reduced over the whole
            # (channel-weighted) volume.  1 - 2|p*t| / (|p| + |t|).
            pm = probs * weighted_mask
            tm = target * weighted_mask
            inter = (pm * target).sum()
            denom_d = (pm).sum() + (tm).sum()
            dice = 1.0 - (2.0 * inter + self.dice_eps) / (denom_d + self.dice_eps)
            total = total + self.aff_lambda_dice * dice

        return total

    def _loss_sem(
        self, probs: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        """Composite Dice + BCE + Focal on the binary foreground head."""
        target = rearrange((labels > 0).float(), "b ... -> b 1 ...")
        valid = rearrange(
            (labels != self.ignore_index).float(), "b ... -> b 1 ...",
        )
        return self._sem_loss(probs, target * valid)

    def _loss_raw(
        self, pred: torch.Tensor, raw_image: torch.Tensor,
    ) -> torch.Tensor:
        """Dense L1 / MSE reconstruction of the (normalised) input image.

        ``pred`` is the linear ``raw`` channel; the target is the input
        EM intensity, taken as-is (no clamp) so a faithfully-normalised
        input drives a faithful regression.
        """
        if raw_image.dim() == pred.dim() - 1:
            raw_image = rearrange(raw_image, "b ... -> b 1 ...")
        return self._raw_fn(pred.float(), raw_image.detach().to(torch.float32))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def canonical_loss_keys(self) -> list:
        """Loss-dict keys :meth:`forward` always emits for this config.

        Gated purely by ``weight_* > 0`` (never by batch content), so the
        key set is identical on every DDP/FSDP rank -- the eval loop
        pre-seeds these into its accumulator for a deterministic
        cross-rank reduction (no ``all_gather_object``).
        """
        keys: list = ["loss"]
        if self.weight_aff > 0:
            keys.append("loss/aff")
        if self.weight_sem > 0:
            keys.append("loss/sem")
        if self.weight_raw > 0:
            keys.append("loss/raw")
        return keys

    def forward(
        self,
        head: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run the affinity + foreground sub-losses and aggregate.

        Args:
            head: ``[B, HEAD_CHANNELS, D, H, W]`` post-sigmoid head.
            targets: Dict with ``labels`` ``[B, D, H, W]`` integer ids and
                (optionally) a ``_cached_targets`` dict from
                :meth:`build_targets`.

        Returns:
            ``{"loss", "loss/aff", "loss/fg"}`` (per-field entries only for
            active fields).
        """
        if head.shape[1] != HEAD_CHANNELS:
            raise ValueError(
                f"AffinityFGLoss expects head with {HEAD_CHANNELS} channels; "
                f"got {head.shape[1]}."
            )
        labels = targets["labels"]
        cached: Dict[str, torch.Tensor] = targets.get("_cached_targets") or {}
        out: Dict[str, torch.Tensor] = {}
        total = head.new_zeros(())

        if self.weight_aff > 0:
            aff_target = cached.get("aff")
            aff_mask = cached.get("aff_mask")
            if aff_target is None:
                aff_target = affinity_target_from_offsets(
                    labels.long(), self.offsets, background=self.background,
                )
                if self.mask_to_foreground:
                    aff_mask = affinity_validity_mask(labels > 0, self.offsets)
            l_aff = self._loss_aff(head[:, AFF_SLICE], aff_target, aff_mask)
            out["loss/aff"] = l_aff
            total = total + self.weight_aff * l_aff

        if self.weight_sem > 0:
            l_sem = self._loss_sem(head[:, SEM_SLICE], labels)
            out["loss/sem"] = l_sem
            total = total + self.weight_sem * l_sem

        if self.weight_raw > 0:
            raw_image = targets.get("raw_image")
            if raw_image is None:
                raise KeyError(
                    "AffinityFGLoss requires `targets['raw_image']` when "
                    "weight_raw > 0; pass the normalised input image."
                )
            l_raw = self._loss_raw(head[:, RAW_SLICE], raw_image)
            out["loss/raw"] = l_raw
            total = total + self.weight_raw * l_raw

        out["loss"] = total
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"channels={self.num_channels}, n_offsets={len(self.offsets)}, "
            f"n_attractive={self.n_attractive}, "
            f"weight_aff={self.weight_aff}, weight_sem={self.weight_sem}, "
            f"weight_raw={self.weight_raw})"
        )


__all__ = ["AffinityFGLoss"]
