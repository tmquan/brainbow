"""
Geometry head regression loss: raw + direction + covariance.

Dimension-agnostic -- parameterized by ``spatial_dims``.

Channel layout (deliberately aligned with :class:`BoundaryLoss`, whose
``ch 0`` is also ``raw``)::

    ch 0                          := raw  (image intensity, dense)
    ch 1 .. 1 + S*(S+1)//2        := cov  (upper-triangle covariance, FG-only)
    ch 1 + S*(S+1)//2 .. 1 + S*(S+1)//2 + S
                                  := dir  (unit direction vectors, FG-only)

Expected geometry head output has ``1 + S*(S+1)//2 + S`` channels:
  2-D:  1 + 3 + 2 =  6
  3-D:  1 + 6 + 3 = 10
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from einops import rearrange, reduce

from brainbow.losses._common import canonical_regression_name, regression_loss_fn


def upper_tri_channels_to_matrix(
    cov_tri: torch.Tensor,
    spatial_dims: int,
) -> torch.Tensor:
    """Expand an upper-triangle covariance channel stack to full matrices.

    The channel order produced by
    :func:`brainbow.transforms.covariance.compute_covariance_field` is
    row-major upper-triangle::

        (i, j) for i in range(S) for j in range(i, S)

    i.e. 2-D: ``(Cyy, Cyx, Cxx)``  ->  channels ``(0,0), (0,1), (1,1)``;
         3-D: ``(Czz, Czy, Czx, Cyy, Cyx, Cxx)``.

    Args:
        cov_tri: ``[B, T, *spatial]`` where ``T = S*(S+1)//2``.
        spatial_dims: ``S`` (2 or 3).

    Returns:
        ``[B, *spatial, S, S]`` symmetric matrices.
    """
    S = spatial_dims
    B = cov_tri.shape[0]
    spatial_shape = cov_tri.shape[2:]
    full = cov_tri.new_zeros(B, *spatial_shape, S, S)
    ch = 0
    for i in range(S):
        for j in range(i, S):
            full[..., i, j] = cov_tri[:, ch]
            if i != j:
                full[..., j, i] = cov_tri[:, ch]
            ch += 1
    return full


class GeometryLoss(nn.Module):
    """Regression loss for the geometry head output.

    Raw is an auxiliary autoencoder-style signal supervised across all
    voxels.  Direction and covariance are foreground-only.  The three
    sub-losses each pick an independent regression function (``mse``,
    ``l1``, or ``smooth_l1``).

    Args:
        spatial_dims: 2 for images, 3 for volumes.
        weight_dir:  Weight for the direction sub-loss.
        weight_cov:  Weight for the covariance sub-loss.
        weight_raw:  Weight for the raw-reconstruction sub-loss.
        loss_dir:    Regression loss name for direction.
        loss_cov:    Regression loss name for covariance.
        loss_raw:    Regression loss name for raw.
        smooth_l1_beta: ``beta`` parameter of smooth-L1 when used.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        weight_dir: float = 1.0,
        weight_cov: float = 1.0,
        weight_raw: float = 1.0,
        loss_dir: str = "l1",
        loss_cov: str = "l1",
        loss_raw: str = "l1",
        smooth_l1_beta: float = 1.0,
        **kwargs,  # ignored; keeps ``CombinedLoss`` kwargs forwarding flexible
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.weight_dir = weight_dir
        self.weight_cov = weight_cov
        self.weight_raw = weight_raw
        self.loss_dir = canonical_regression_name(loss_dir)
        self.loss_cov = canonical_regression_name(loss_cov)
        self.loss_raw = canonical_regression_name(loss_raw)
        self.smooth_l1_beta = smooth_l1_beta

        S = spatial_dims
        self._ch_raw = 1
        self._ch_dir = S
        self._ch_cov = S * (S + 1) // 2

    @property
    def task_channels(self) -> int:
        """Expected width of the geometry head prediction tensor."""
        return self._ch_raw + self._ch_dir + self._ch_cov

    # Backwards-compat alias -- older code looked up ``geometry_channels``.
    @property
    def geometry_channels(self) -> int:
        return self.task_channels

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_target_dir(self, ins_label_b: torch.Tensor) -> torch.Tensor:
        """Unit direction field for a single batch element.

        Returns a flattened ``[S, N]`` tensor on the same device as
        ``ins_label_b``.  The underlying transform lives in NumPy, so
        we materialise back to torch before returning.
        """
        from brainbow.transforms.direction import compute_direction_field
        d = compute_direction_field(ins_label_b, normalize=True)
        d_t = torch.as_tensor(d, device=ins_label_b.device, dtype=torch.float32)
        return rearrange(d_t, "c ... -> c (...)")

    @torch.no_grad()
    def _build_target_cov(self, ins_label_b: torch.Tensor) -> torch.Tensor:
        """Upper-triangle covariance field for a single batch element.

        Returns a flattened ``[S*(S+1)//2, N]`` tensor on the same
        device as ``ins_label_b``.
        """
        from brainbow.transforms.covariance import compute_covariance_field
        c = compute_covariance_field(ins_label_b, normalized=True)
        c_t = torch.as_tensor(c, device=ins_label_b.device, dtype=torch.float32)
        return rearrange(c_t, "c ... -> c (...)")

    @torch.no_grad()
    def build_target(
        self,
        ins_label: torch.Tensor,
        direction: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Build per-batch direction + covariance targets.

        Fast path: ``direction`` / ``covariance`` are precomputed tensors
        from the datamodule (shape ``[B, C, *spatial]``) -- we just
        flatten them.  Fallback path: run the scalar transforms on the
        fly per batch element.

        Args:
            ins_label:  ``[B, *spatial]`` integer instance ids.
            direction:  optional ``[B, S, *spatial]`` unit vectors.
            covariance: optional ``[B, S*(S+1)//2, *spatial]`` upper-tri.

        Returns:
            ``{"dir_targets": [B tensors or None],
               "cov_targets": [B tensors or None]}``
        """
        if direction is not None and covariance is not None:
            return self._build_target_from_pipeline(direction, covariance)

        dir_targets: List[Optional[torch.Tensor]] = []
        cov_targets: List[Optional[torch.Tensor]] = []
        for b in range(ins_label.shape[0]):
            lbl_b = ins_label[b]
            if not (lbl_b > 0).any():
                dir_targets.append(None)
                cov_targets.append(None)
                continue
            dir_targets.append(
                self._build_target_dir(lbl_b) if self.weight_dir > 0 else None
            )
            cov_targets.append(
                self._build_target_cov(lbl_b) if self.weight_cov > 0 else None
            )
        return {"dir_targets": dir_targets, "cov_targets": cov_targets}

    def _build_target_from_pipeline(
        self,
        direction: torch.Tensor,
        covariance: torch.Tensor,
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Flatten pipeline-precomputed direction + covariance tensors.

        Foreground presence is inferred from non-zero direction vectors
        (background voxels have a zero direction by construction).

        We pay one device→host sync up-front via ``has_fg.tolist()``
        instead of one per batch element via ``if not has_fg[b]`` — the
        latter triggers a separate ``cudaStreamSynchronize`` on every
        iteration of the Python loop, which is amplified under
        ``torch.compile``-deepened launch queues.
        """
        dir_flat = rearrange(direction, "b c ... -> b c (...)")
        cov_flat = rearrange(covariance, "b c ... -> b c (...)")
        has_fg_list = (
            reduce(dir_flat.abs(), "b c n -> b", "sum") > 0
        ).tolist()

        dir_targets: List[Optional[torch.Tensor]] = []
        cov_targets: List[Optional[torch.Tensor]] = []
        for b, has_fg in enumerate(has_fg_list):
            if not has_fg:
                dir_targets.append(None)
                cov_targets.append(None)
                continue
            dir_targets.append(dir_flat[b] if self.weight_dir > 0 else None)
            cov_targets.append(cov_flat[b] if self.weight_cov > 0 else None)
        return {"dir_targets": dir_targets, "cov_targets": cov_targets}

    # Backwards-compat shims for older call sites.
    def targets_from_pipeline(
        self, direction: torch.Tensor, covariance: torch.Tensor,
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        return self._build_target_from_pipeline(direction, covariance)

    # ------------------------------------------------------------------
    # Per-voxel weights (not used by this head)
    # ------------------------------------------------------------------

    def compute_weights(self, labels: torch.Tensor) -> None:
        return None

    # ------------------------------------------------------------------
    # Sub-losses
    # ------------------------------------------------------------------

    def _regress(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Regression dispatched through :func:`regression_loss_fn`.

        ``smooth_l1`` carries an extra ``beta`` knob that is unique to
        this loss; the other variants ignore it harmlessly.
        """
        fn = regression_loss_fn(loss_type)
        if canonical_regression_name(loss_type) == "smooth_l1":
            return fn(pred, target, beta=self.smooth_l1_beta, reduction=reduction)
        return fn(pred, target, reduction=reduction)

    def _compute_loss_fg(
        self,
        pred_b: torch.Tensor,
        target_b: torch.Tensor,
        fg_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Foreground-only regression loss for a single batch element.

        Args:
            pred_b:   ``[C, N]`` flattened prediction.
            target_b: ``[C, N]`` flattened target.
            fg_b:     ``[N]`` boolean foreground mask.
            loss_type: canonical regression name ("mse" / "l1" / "smooth_l1").
        """
        n_fg = fg_b.sum().float().clamp(min=1.0)
        denom = n_fg * pred_b.shape[0]
        p, t = pred_b[:, fg_b], target_b[:, fg_b]
        return self._regress(p, t, loss_type, reduction="sum") / denom

    def _compute_loss_dir(self, pred_b, target_b, fg_b):
        return self._compute_loss_fg(pred_b, target_b, fg_b, self.loss_dir)

    def _compute_loss_cov(self, pred_b, target_b, fg_b):
        return self._compute_loss_fg(pred_b, target_b, fg_b, self.loss_cov)

    def _compute_loss_raw(
        self,
        pred: torch.Tensor,
        raw_image: torch.Tensor,
        geometry_ndim: int,
    ) -> torch.Tensor:
        """Dense raw-reconstruction loss over every voxel in the batch.

        Matches :class:`BoundaryLoss` -- the raw channel is supervised
        on all voxels (no FG mask, no batch-filter).
        """
        img = raw_image.detach().clamp(0.0, 1.0).to(torch.float32)
        if img.dim() == geometry_ndim - 1:
            img = img.unsqueeze(1)  # [B, *spatial] -> [B, 1, *spatial]
        target = rearrange(img[:, :1], "b c ... -> b c (...)")
        return self._regress(pred, target, self.loss_raw, reduction="mean")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        geometry: torch.Tensor,
        ins_label: torch.Tensor,
        raw_image: Optional[torch.Tensor] = None,
        cached_targets: Optional[Dict[str, List]] = None,
    ) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=geometry.device)

        geom_flat = rearrange(geometry, "b c ... -> b c (...)")
        # Channel layout: [raw(1) | cov(S*(S+1)/2) | dir(S)]
        c_raw = self._ch_raw
        c_cov = c_raw + self._ch_cov
        c_dir = c_cov + self._ch_dir
        pred_raw = geom_flat[:, :c_raw]
        pred_cov = geom_flat[:, c_raw:c_cov]
        pred_dir = geom_flat[:, c_cov:c_dir]

        lbl_flat = rearrange(ins_label, "b ... -> b (...)").long()
        fg = lbl_flat > 0

        if cached_targets is None:
            cached_targets = self.build_target(ins_label)
        dir_targets = cached_targets["dir_targets"]
        cov_targets = cached_targets["cov_targets"]

        L_dir, L_cov, L_raw = zero.clone(), zero.clone(), zero.clone()
        valid = 0
        for b in range(geometry.shape[0]):
            if dir_targets[b] is None and cov_targets[b] is None:
                continue
            valid += 1
            if self.weight_dir > 0 and dir_targets[b] is not None:
                L_dir = L_dir + self._compute_loss_dir(pred_dir[b], dir_targets[b], fg[b])
            if self.weight_cov > 0 and cov_targets[b] is not None:
                L_cov = L_cov + self._compute_loss_cov(pred_cov[b], cov_targets[b], fg[b])

        denom = max(valid, 1)
        L_dir = L_dir / denom
        L_cov = L_cov / denom

        if self.weight_raw > 0 and raw_image is not None:
            L_raw = self._compute_loss_raw(pred_raw, raw_image, geometry.dim())

        total = self.weight_dir * L_dir + self.weight_cov * L_cov + self.weight_raw * L_raw
        return {"loss": total, "dir": L_dir, "cov": L_cov, "raw": L_raw}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"spatial_dims={self.spatial_dims}, channels={self.task_channels} "
            f"(raw={self._ch_raw}+cov={self._ch_cov}+dir={self._ch_dir}), "
            f"loss_dir='{self.loss_dir}', loss_cov='{self.loss_cov}', loss_raw='{self.loss_raw}', "
            f"weight_dir={self.weight_dir}, weight_cov={self.weight_cov}, "
            f"weight_raw={self.weight_raw})"
        )
