"""Hierarchical TensorBoard tag builder.

Every image tag in :mod:`brainbow.callbacks.tensorboard` is produced
through :class:`TagContext` so the layout ``{stage}/{mode}/{panel}`` is
enforced in one place.
"""

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class TagContext:
    """Hierarchical TB tag builder: ``{stage}/{mode}/[{head}/]{panel}``.

    ``head`` is retained only as a lightweight compatibility shim for
    older call sites; the unified-head logger passes ``head=None`` and
    encodes field names directly in ``panel`` (e.g. ``pred/sem``,
    ``pred/aff/01_pull_z1``, ``pred/label/pre``).
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
