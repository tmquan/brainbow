"""Shared Lightning-module scaffolding for the Cosmos 2.5 family.

Houses :class:`BaseCosmosModule` -- the abstract base every
Cosmos 2.5 backbone module (Predict, Transfer, ...) inherits from.

Subclasses live next door under
:mod:`brainbow.modules.cosmos_transfer_2_5` and
:mod:`brainbow.modules.cosmos_predict_2_5`; both import this base
class to get the freeze schedule, NaN/Inf gradient handling, and
backbone / ControlNet / heads optimiser-group split for free.
"""

from brainbow.modules.cosmos_2_5_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
