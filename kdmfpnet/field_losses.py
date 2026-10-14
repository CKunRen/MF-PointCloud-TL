"""Losses for the residual dual-field method.

* ``weighted_field_loss``  -- pointwise-weighted pressure loss plus an optional
  auxiliary wall-velocity loss (paper: field loss).
* ``physics_consistency_loss`` -- matches the pressure field's surface integral
  to the solver-reported force (and optionally moment) coefficients, with a
  per-sample magnitude calibration (paper: physics consistency).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_field_loss(
    pred_cp: torch.Tensor,
    true_cp: torch.Tensor,
    weights: torch.Tensor,
    pred_velocity: torch.Tensor | None = None,
    true_velocity: torch.Tensor | None = None,
    lambda_v: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (total, components).

    pred_cp/true_cp/weights: [M, 1]; pred_velocity/true_velocity: [M, 3].
    """
    pitch = F.smooth_l1_loss(pred_cp, true_cp, reduction="none", beta=1.0)
    denom = weights.sum().clamp_min(1e-6)
    pressure = (pitch * weights).sum() / denom
    components = {"pressure": float(pressure.detach())}
    total = pressure
    if lambda_v > 0.0 and pred_velocity is not None and true_velocity is not None:
        per_component = F.smooth_l1_loss(pred_velocity, true_velocity, reduction="none", beta=1.0)
        velocity = (per_component * weights).sum() / (denom * pred_velocity.shape[1])
        total = total + float(lambda_v) * velocity
        components["velocity"] = float(velocity.detach())
    return total, components


def _integrate(
    value: torch.Tensor,
    vector: torch.Tensor,
    areas: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """-(1/scale) * sum_i value_i * vector_i * area_i, per sample -> [B, 3]."""
    return -(value * vector * areas).sum(dim=1) / scale.clamp_min(1e-12)


def physics_consistency_loss(
    pred_cp: torch.Tensor,
    normals: torch.Tensor,
    areas: torch.Tensor,
    reference_force: torch.Tensor,
    force_calibration: torch.Tensor,
    ref_area: torch.Tensor,
    points: torch.Tensor | None = None,
    reference_moment: torch.Tensor | None = None,
    moment_origin: torch.Tensor | None = None,
    ref_length: torch.Tensor | None = None,
    include_moments: bool = False,
) -> torch.Tensor:
    """Match integrated predicted loads to the solver-reported coefficients.

    Shapes (batch preserved so this is multi-sample safe): pred_cp [B, M, 1],
    normals [B, M, 3], areas [B, M, 1], reference_force [B, 3],
    force_calibration [B, 1], ref_area [B, 1]. Padded rows carry area = 0 and
    therefore contribute nothing to the integral.
    """
    force = _integrate(pred_cp, normals, areas, ref_area) * force_calibration
    loss = ((force - reference_force) ** 2).sum(dim=1).mean()

    if include_moments:
        if points is None or reference_moment is None or moment_origin is None or ref_length is None:
            raise ValueError("points/reference_moment/moment_origin/ref_length required for moments")
        lever = points - moment_origin.unsqueeze(1)
        cross = torch.cross(lever, normals, dim=-1)
        moment = _integrate(pred_cp, cross, areas, ref_area * ref_length)
        # Moment calibration is folded into the force calibration scalar (both
        # share the same estimated-geometry bias).
        moment = moment * force_calibration
        loss = loss + ((moment - reference_moment) ** 2).sum(dim=1).mean()
    return loss


def _local_gradient(field: torch.Tensor, points: torch.Tensor, knn_idx: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Least-squares local gradient of ``field`` at every point.

    field: [B, N, C] (e.g. the scalar Cp field), points: [B, N, 3] in PHYSICAL
    units, knn_idx: [B, N, k]. Returns [B, N, 3, C].

    A plain neighbour-difference magnitude is *stationary* when the prediction is
    locally flat (|x| has zero subgradient at 0), so it produces no gradient
    exactly where the field most needs sharpening. Fitting the gradient by least
    squares makes it a **linear** function of the field values, so the penalty on
    it has a non-zero gradient whenever the truth has a gradient. The ridge term
    makes the (rank-deficient, because surface neighbours span only the tangent
    plane) normal equations solvable, effectively returning the tangential part.
    """
    from .kd_model import _gather_neighbor_features

    nbr_points = _gather_neighbor_features(points, knn_idx)                 # [B, N, k, 3]
    nbr_values = _gather_neighbor_features(field, knn_idx)                  # [B, N, k, C]
    delta_p = nbr_points - points.unsqueeze(2)                              # [B, N, k, 3]
    delta_v = nbr_values - field.unsqueeze(2)                               # [B, N, k, C]

    dtd = torch.einsum("bnki,bnkj->bnij", delta_p, delta_p)                 # [B, N, 3, 3]
    dtv = torch.einsum("bnki,bnkc->bnik", delta_p, delta_v)                 # [B, N, 3, C]
    eye = torch.eye(3, device=dtd.device, dtype=dtd.dtype).view(1, 1, 3, 3)
    return torch.linalg.solve(dtd + ridge * eye, dtv)


def sharpness_matching_loss(
    pred_cp: torch.Tensor,
    true_cp: torch.Tensor,
    points: torch.Tensor,
    knn_idx: torch.Tensor,
    mask: torch.Tensor | None = None,
    one_sided: bool = True,
) -> torch.Tensor:
    """Penalise a prediction whose local surface gradient is weaker than the truth's.

    The error analysis showed the largest residual sits at the sharpest nodes and
    is correlated across independently trained models, i.e. it is a shared
    representation gap rather than model noise. This term matches the local
    gradient of the prediction to the truth's, so under-resolving a sharp feature
    is penalised directly.

    one_sided=True (default) only penalises under-sharpening: pushing a prediction
    that already resolves a feature to be even sharper would mostly amplify noise.
    The one-sided test is the projection of the shortfall onto the *true* gradient
    direction -- unlike a magnitude comparison, that keeps a non-zero gradient even
    when the prediction is locally flat.
    """
    g_pred = _local_gradient(pred_cp, points, knn_idx)                     # [B, N, 3, C]
    g_true = _local_gradient(true_cp, points, knn_idx).detach()            # [B, N, 3, C]
    diff = g_true - g_pred
    if one_sided:
        g_norm = g_true.norm(dim=2, keepdim=True).clamp_min(1e-8)          # [B, N, 1, C]
        shortfall = (diff * (g_true / g_norm)).sum(dim=2, keepdim=True)    # [B, N, 1, C]
        sq = torch.clamp(shortfall, min=0.0) ** 2
    else:
        sq = (diff ** 2).sum(dim=2, keepdim=True)                          # [B, N, 1, C]
    if mask is not None:
        m = mask.unsqueeze(-1).unsqueeze(-1).to(sq.dtype)                  # [B, N, 1, 1]
        return (sq * m).sum() / (m.sum().clamp_min(1.0) * sq.shape[-1])
    return sq.mean()


def vector_metrics(true_velocity: torch.Tensor, pred_velocity: torch.Tensor) -> dict[str, float]:
    """Per-component RMSE and a magnitude-based R2 for the wall-velocity field."""
    diff = pred_velocity - true_velocity
    rmse = torch.sqrt((diff ** 2).mean(dim=0))
    true_flat = true_velocity.reshape(-1)
    ss_res = float((diff ** 2).sum())
    ss_tot = float(((true_velocity - true_velocity.mean(dim=0, keepdim=True)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "rmse_x": float(rmse[0]),
        "rmse_y": float(rmse[1]),
        "rmse_z": float(rmse[2]),
        "rmse": float(torch.sqrt((diff ** 2).mean())),
        "r2": r2,
    }
