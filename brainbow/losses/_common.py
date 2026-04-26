"""
Shared helpers for the task-loss modules.

Every task loss in :mod:`brainbow.losses` follows the same skeleton::

    class XxxLoss(nn.Module):
        def __init__(...)
        @property def task_channels        # expected prediction width
        @torch.no_grad() def _build_target_<component>(...)
        @torch.no_grad() def build_target(...)
        def compute_weights(labels)        # optional per-voxel weights
        def _compute_loss_<component>(...)
        def forward(...)                   # -> {"loss": total, <component>: ...}
        def __repr__()

Helpers shared across task losses:

* :func:`canonical_regression_name` / :func:`regression_loss_fn` --
  resolve user-facing names (``mse`` / ``l1`` / ``smooth_l1`` with the
  common aliases ``l2`` / ``mae`` / ``huber``) to canonical strings or
  to the underlying ``torch.nn.functional`` callable.
* :func:`stable_bce_on_probs` -- per-voxel binary cross-entropy on
  inputs that are already sigmoided.  Used by :class:`SemanticLoss`
  and :class:`BoundaryLoss` (the model wrapper applies a single
  sigmoid before the loss; both heads then share the same bf16-safe
  BCE).  See the function docstring for the autocast caveats.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


_REGRESSION_ALIASES: dict[str, str] = {
    "mse": "mse", "l2": "mse",
    "l1": "l1", "mae": "l1",
    "smooth_l1": "smooth_l1", "huber": "smooth_l1",
}

_REGRESSION_FNS: dict[str, Callable] = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
}


def canonical_regression_name(name: str) -> str:
    """Resolve a user-facing regression-loss name to its canonical form.

    Recognised names (case-insensitive, ``-`` treated as ``_``):

        mse | l2           -> "mse"
        l1  | mae          -> "l1"
        smooth_l1 | huber  -> "smooth_l1"
    """
    key = name.lower().replace("-", "_")
    if key not in _REGRESSION_ALIASES:
        raise ValueError(
            f"Unknown regression loss '{name}'. "
            f"Choose from: {sorted(set(_REGRESSION_ALIASES))}"
        )
    return _REGRESSION_ALIASES[key]


def regression_loss_fn(name: str) -> Callable:
    """Return the ``torch.nn.functional`` callable for a regression loss.

    The callable accepts ``(input, target, reduction=..., ...)`` exactly
    like the F.* counterparts it wraps.  Use
    :func:`canonical_regression_name` instead if you need to branch on
    the name itself (e.g. to plumb through ``beta`` for smooth-L1).
    """
    return _REGRESSION_FNS[canonical_regression_name(name)]


def stable_bce_on_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Per-voxel binary cross-entropy on **probabilities** (not logits).

    The brainbow model wrappers apply a single sigmoid to the semantic
    head and the boundary head's affinity channels before the loss
    sees them, so both losses need a BCE that consumes ``[0, 1]``
    probabilities.  This helper performs the BCE in fp32 with explicit
    clamping so ``log(p)`` and ``log(1 - p)`` stay finite under
    ``bf16-mixed`` autocast (bf16 has only ~3 decimal digits of mantissa
    near ``1`` so any ``p > ~0.992`` rounds to exactly ``1`` and would
    otherwise yield ``-inf``).

    Args:
        probs:      ``[B, C, *spatial]`` already-sigmoided predictions.
        target:     ``[B, C, *spatial]`` binary target (0 / 1 floats).
        pos_weight: Optional ``[1, C, 1, ..., 1]`` per-channel positive
            weight, semantically equivalent to
            ``BCEWithLogitsLoss(pos_weight=...)`` -- it re-weights only
            the ``t == 1`` term.
        eps:        Clamp bound for numerical stability.

    Returns:
        ``[B, C, *spatial]`` per-voxel BCE.  The caller reduces (mean,
        masked sum / valid_mask, ...) as it sees fit.
    """
    p = probs.float().clamp(eps, 1.0 - eps)
    t = target.float()
    if pos_weight is not None:
        pw = pos_weight.to(p.dtype)
        return -(pw * t * p.log() + (1.0 - t) * (1.0 - p).log())
    return -(t * p.log() + (1.0 - t) * (1.0 - p).log())


__all__ = [
    "canonical_regression_name",
    "regression_loss_fn",
    "stable_bce_on_probs",
]
