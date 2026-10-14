from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_weights(weights: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        mask_f = mask.unsqueeze(-1).to(weights.dtype)
        weights = weights * mask_f
        denom = mask_f.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        weights = weights / denom
    return weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)


def _geometry_prior_weights(
    raw_points: torch.Tensor,
    tail_loss_alpha: float,
    tail_loss_fraction: float,
    fin_radial_loss_alpha: float,
    fin_radial_start_fraction: float,
) -> torch.Tensor:
    weights = torch.ones((*raw_points.shape[:2], 1), device=raw_points.device, dtype=raw_points.dtype)
    if tail_loss_alpha <= 0 and fin_radial_loss_alpha <= 0:
        return weights

    x = raw_points[:, :, 0:1]
    radius = torch.linalg.norm(raw_points[:, :, 1:3], dim=2, keepdim=True)
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    axial = (x - x_min) / (x_max - x_min).clamp_min(1e-6)

    tail_fraction = float(min(max(tail_loss_fraction, 1e-6), 1.0))
    tail_score = torch.clamp((tail_fraction - axial) / tail_fraction, min=0.0, max=1.0)
    if tail_loss_alpha > 0:
        weights = weights * (1.0 + float(tail_loss_alpha) * tail_score)

    if fin_radial_loss_alpha > 0:
        radius_max = radius.amax(dim=1, keepdim=True).clamp_min(1e-6)
        radial = radius / radius_max
        radial_start = float(min(max(fin_radial_start_fraction, 0.0), 0.999999))
        radial_score = torch.clamp((radial - radial_start) / (1.0 - radial_start), min=0.0, max=1.0)
        weights = weights * (1.0 + float(fin_radial_loss_alpha) * tail_score * radial_score)

    return weights


def _topk_weights(
    residual: torch.Tensor,
    topk_fraction: float,
    topk_alpha: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = torch.ones_like(residual)
    if topk_alpha <= 0 or topk_fraction <= 0:
        return weights

    scores = residual.abs().squeeze(-1)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    count = max(1, int(round(residual.shape[1] * float(topk_fraction))))
    count = min(count, residual.shape[1])
    _, indices = torch.topk(scores, k=count, dim=1, largest=True, sorted=False)
    weights.scatter_(1, indices.unsqueeze(-1), 1.0 + float(topk_alpha))
    if mask is not None:
        weights = weights * mask.unsqueeze(-1).to(weights.dtype)
    return weights


def build_pressure_weights(
    raw_pressure: torch.Tensor,
    raw_points: torch.Tensor,
    peak_loss_alpha: float = 0.75,
    peak_loss_cap: float = 3.0,
    tail_loss_alpha: float = 1.0,
    tail_loss_fraction: float = 0.32,
    fin_radial_loss_alpha: float = 0.75,
    fin_radial_start_fraction: float = 0.55,
    topk_peak_loss_alpha: float = 2.0,
    topk_peak_loss_fraction: float = 0.05,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = torch.ones((*raw_pressure.shape[:2], 1), device=raw_pressure.device, dtype=raw_pressure.dtype)
    if peak_loss_alpha > 0:
        mean_abs = raw_pressure.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        normalized_peak = torch.clamp(raw_pressure.abs() / mean_abs, max=float(peak_loss_cap))
        weights = weights * (1.0 + float(peak_loss_alpha) * normalized_peak)

    weights = weights * _geometry_prior_weights(
        raw_points=raw_points,
        tail_loss_alpha=tail_loss_alpha,
        tail_loss_fraction=tail_loss_fraction,
        fin_radial_loss_alpha=fin_radial_loss_alpha,
        fin_radial_start_fraction=fin_radial_start_fraction,
    )
    weights = weights * _topk_weights(
        residual=raw_pressure,
        topk_fraction=topk_peak_loss_fraction,
        topk_alpha=topk_peak_loss_alpha,
        mask=mask,
    )
    return _normalize_weights(weights, mask=mask)


def build_hard_example_weights(
    prediction: torch.Tensor,
    target: torch.Tensor,
    topk_fraction: float = 0.1,
    hard_point_alpha: float = 1.0,
    target_peak_alpha: float = 0.0,
    target_peak_fraction: float = 0.0,
    target_peak_cap: float = 3.0,
    raw_points: torch.Tensor | None = None,
    tail_loss_alpha: float = 0.0,
    tail_loss_fraction: float = 0.32,
    fin_radial_loss_alpha: float = 0.0,
    fin_radial_start_fraction: float = 0.55,
    residual_cap: float = 3.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = (prediction - target).abs()
    weights = torch.ones_like(residual)
    if hard_point_alpha > 0:
        mean_abs = residual.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        normalized = torch.clamp(residual / mean_abs, max=float(residual_cap))
        weights = weights * (1.0 + float(hard_point_alpha) * normalized)

    if raw_points is not None and (tail_loss_alpha > 0 or fin_radial_loss_alpha > 0):
        weights = weights * _geometry_prior_weights(
            raw_points=raw_points,
            tail_loss_alpha=tail_loss_alpha,
            tail_loss_fraction=tail_loss_fraction,
            fin_radial_loss_alpha=fin_radial_loss_alpha,
            fin_radial_start_fraction=fin_radial_start_fraction,
        )

    if target_peak_alpha > 0:
        target_abs = target.abs()
        mean_abs = target_abs.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        normalized_peak = torch.clamp(target_abs / mean_abs, max=float(target_peak_cap))
        weights = weights * (1.0 + float(target_peak_alpha) * normalized_peak)
        if target_peak_fraction > 0:
            weights = weights * _topk_weights(
                residual=target,
                topk_fraction=target_peak_fraction,
                topk_alpha=target_peak_alpha,
                mask=mask,
            )

    weights = weights * _topk_weights(
        residual=residual,
        topk_fraction=topk_fraction,
        topk_alpha=hard_point_alpha,
        mask=mask,
    )
    return _normalize_weights(weights, mask=mask)


def weighted_pressure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_point = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    return (per_point * weights).sum() / weights.sum().clamp_min(1e-6)


def ramp_weight(epoch: int, warmup_epochs: int, ramp_epochs: int, max_value: float) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_value)
    progress = min(max(epoch - warmup_epochs, 0), ramp_epochs) / float(ramp_epochs)
    return float(max_value) * progress
