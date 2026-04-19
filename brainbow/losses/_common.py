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

Only one genuinely-shared piece of code lives here: the canonical-name
resolver for regression sub-losses (``l1`` / ``mse`` / ``smooth_l1``
with their common aliases ``mae`` / ``l2`` / ``huber``).
"""

from __future__ import annotations

from typing import Callable

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


__all__ = [
    "canonical_regression_name",
    "regression_loss_fn",
]
