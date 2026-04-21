"""Shared task-head modules.

This module holds the VISTA3D-style building blocks shared across our
3D wrappers (:class:`Vista3DWrapper`, :class:`CosmosTransfer3DWrapper`)
so every wrapper applies an *actual* VISTA3D-style task head instead of
the minimal ``Conv3-Norm-ReLU-Conv1`` adapter we used to ship.

Rationale
---------
MONAI's reference VISTA3D implementation
(``monai.networks.nets.vista3d.ClassMappingClassify``) refines its backbone
features with ``image_post_mapping`` — two residual ``UnetrBasicBlock``
stacks operating at the encoder feature width:

.. code-block:: python

    image_post_mapping = nn.Sequential(
        UnetrBasicBlock(..., res_block=True, norm_name="instance"),
        UnetrBasicBlock(..., res_block=True, norm_name="instance"),
    )

That is the "head" in VISTA3D proper.  Our previous per-task adapter
(``Conv3x3 → GroupNorm → ReLU → [Dropout] → Conv1x1``) has roughly half
the depth and no residual connection, which noticeably under-fits on
dense-prediction tasks like connectomics.  ``VistaTaskHead3D`` below is
the drop-in replacement that mirrors the real VISTA3D head and simply
swaps the class-embedding mask-attention output for a plain 1×1 conv,
since our per-voxel tasks (semantic logits, instance embeddings, geometry
regressions) don't need the learned class embedding table.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from monai.networks.blocks import UnetrBasicBlock
except ImportError as exc:  # pragma: no cover - monai is a hard dep
    raise ImportError(
        "brainbow.models.vista requires MONAI (monai.networks.blocks.UnetrBasicBlock). "
        "Install with `pip install monai>=1.3`."
    ) from exc


class VistaTaskHead3D(nn.Module):
    """VISTA3D-style dense-prediction head.

    Mirrors ``ClassMappingClassify.image_post_mapping`` from MONAI's
    reference VISTA3D: two residual 3-D UnetrBasicBlocks refine the
    feature map at a shared width, and a final 1×1 convolution projects
    to the task-specific channel count.

    Architecture::

        (optional) Conv1x1:  in_channels -> refine_channels   # width matcher
        UnetrBasicBlock(res=True, kernel=3, norm="instance")  # refine
        UnetrBasicBlock(res=True, kernel=3, norm="instance")  # refine
        Dropout3d(dropout)
        Conv1x1:             refine_channels -> out_channels   # task head

    The width matcher is inserted only when ``in_channels != refine_channels``
    so the head can be plugged on top of either the VISTA3D encoder
    (feature_size output) or the Cosmos VAE decoder (arbitrary hidden
    width ``_hidden_ch``).

    Args:
        in_channels: Channel count of the backbone/decoder feature map
            fed into the head.
        out_channels: Task-specific output width (``num_classes`` for
            semantic, ``instance_channels`` for instance, ``geometry_channels``
            for geometry).
        refine_channels: Width at which the residual refinement blocks
            operate.  Defaults to ``in_channels`` (VISTA3D's convention).
            Passing a smaller value (e.g. ``feature_size=64``) is the
            cheap-but-faithful variant used when the decoder emits many
            channels and we don't want 2 × in_channels² cost per head.
        dropout: Channel-wise dropout applied just before the final 1×1.
            VISTA3D itself uses zero dropout; we expose it to preserve
            the behaviour of the old Cosmos adapter heads.
        norm_name: Normalisation used inside the residual blocks.
            Kept as a parameter for forward-compat; VISTA3D uses
            ``"instance"`` (the default here).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        refine_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm_name: str = "instance",
    ) -> None:
        super().__init__()
        refine = int(refine_channels) if refine_channels is not None else int(in_channels)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.refine_channels = refine

        layers: list[nn.Module] = []
        if self.in_channels != refine:
            # Width matcher: 1×1 conv keeps the refinement blocks at a
            # fixed width regardless of how wide the upstream decoder is.
            layers.append(nn.Conv3d(self.in_channels, refine, kernel_size=1, bias=True))

        layers.extend([
            UnetrBasicBlock(
                spatial_dims=3,
                in_channels=refine,
                out_channels=refine,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
            UnetrBasicBlock(
                spatial_dims=3,
                in_channels=refine,
                out_channels=refine,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
        ])

        if dropout and dropout > 0.0:
            layers.append(nn.Dropout3d(float(dropout)))

        layers.append(nn.Conv3d(refine, self.out_channels, kernel_size=1, bias=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


__all__ = ["VistaTaskHead3D"]
