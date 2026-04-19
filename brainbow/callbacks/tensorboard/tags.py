"""Hierarchical TensorBoard tag builder.

Every image and scalar tag in the :mod:`brainbow.callbacks.tensorboard`
subpackage is produced through :class:`TagContext` so that the layout
``{stage}/{mode}/[{head}/]{panel}`` is enforced in one place.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

#: Heads that produce their own sub-group inside ``{stage}/{mode}/``.
#: Ordering is purely cosmetic (controls the order of calls in
#: ``_log_predictions``; TB itself sorts tags alphabetically).
HEADS: Tuple[str, ...] = ("semantic", "instance", "geometry", "brainbow")


@dataclass(frozen=True)
class TagContext:
    """Hierarchical TB tag builder: ``{stage}/{mode}/[{head}/]{panel}``.

    Instances are immutable; use :meth:`for_head` to descend into a
    per-head namespace without mutating the parent context.
    """

    stage: str                        # "train" | "val"
    mode: str = "automatic"           # "automatic" | "prompted" | ...
    head: Optional[str] = None        # None -> mode-level panels

    @property
    def prefix(self) -> str:
        parts = [self.stage, self.mode]
        if self.head is not None:
            parts.append(self.head)
        return "/".join(parts)

    def tag(self, panel: str) -> str:
        """Return the full tag for a panel under this context."""
        return f"{self.prefix}/{panel}"

    def for_head(self, head: str) -> "TagContext":
        """Return a child context scoped to ``head``."""
        return replace(self, head=head)
