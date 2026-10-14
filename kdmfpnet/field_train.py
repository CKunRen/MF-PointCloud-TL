"""Residual multi-fidelity transfer-learning trainer (the new TL method).

Stage 1 pretrains a dual-output (Cp + wall velocity) point-cloud network on the
low-fidelity G2 data. Stage 2 freezes that prior (or a subset of it, per the
layer-freezing ablation), attaches a zero-initialised correction branch, and
fine-tunes on the high-fidelity G4 data with the pointwise-weighted field loss
plus the optional physics-consistency penalty on integrated loads.

This is intentionally a separate module from ``kd_train.py`` so the KD pipeline
and its tests are untouched; it reuses the shared data/model/utility helpers.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler

from .field_losses import (
    physics_consistency_loss,
    sharpness_matching_loss,
    vector_metrics,
    weighted_field_loss,
)
from .kd_data import build_fidelity_pressure_loader, fit_fidelity_standardizers
from .kd_model import PressureKDRegressor
from .kd_train import (
    _batch_to_numpy,
    _build_mach_sampling_weights,
    _cosine_warmup_lr_factor,
    _estimate_target_strength_reference,
    _pressure_metrics,
    _sample_priority_weights,
    _save_predictions_csv,
    _save_sample_metrics_csv,
    _to_torch,
    _validate_signal_reference,
    _validation_selection_metric,
    _write_history_row,
    build_kd_sample_lists,
)
from .losses import build_hard_example_weights, build_pressure_weights
from .residual_model import (
    FIELD_CHANNELS,
    ResidualFieldModel,
    apply_freezing_strategy,
    build_field_model,
    zero_init_correction,
)
from .utils import Standardizer, save_json, seed_everything, stable_train_val_split


@dataclass
class FieldNormalizerBundle:
    points: Standardizer
    condition: Standardizer
    cp: Standardizer
    velocity: Standardizer

    def to_state_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            "points": self.points.to_state_dict(),
            "condition": self.condition.to_state_dict(),
            "cp": self.cp.to_state_dict(),
            "velocity": self.velocity.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, dict[str, list[float]]]) -> "FieldNormalizerBundle":
        return cls(
            points=Standardizer.from_state_dict(payload["points"]),
            condition=Standardizer.from_state_dict(payload["condition"]),
            cp=Standardizer.from_state_dict(payload["cp"]),
            velocity=Standardizer.from_state_dict(payload["velocity"]),
        )


@dataclass
class FieldTrainingConfig:
    """Hyperparameters for the residual dual-field TL run."""

    pretrain_epochs: int = 120
    finetune_epochs: int = 100
    pretrain_lr: float = 8e-4
    finetune_lr: float = 4e-4
    weight_decay: float = 5e-4
    dropout: float = 0.14
    knn_k: int = 24
    seed: int = 42
    val_fraction: float = 0.1
    finetune_val_fraction: float = 0.18
    sample_limit: int | None = None  # smoke-test cap per split
    finetune_limit: int | None = None  # cap G4 fine-tuning cases only (data-budget study)
    pretrain_limit: int | None = None  # cap G2 pretraining cases; forces fresh pretraining
    from_scratch: bool = False         # skip pretraining and correction (G4-only baseline)
    use_residual: bool = True          # False: direct fine-tuning (no correction branch)
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    lambda_velocity: float = 0.3
    physics_weight: float = 0.0
    physics_in_pretrain: bool = False
    include_moments: bool = False
    freezing_strategy: str = "frozen"
    correction_point_channels: tuple[int, ...] = (32, 64, 128)
    correction_decoder_hidden: tuple[int, ...] = (128, 64)
    point_jitter_std: float = 0.0
    hard_point_alpha: float = 1.5
    hard_point_fraction: float = 0.1
    target_peak_alpha: float = 0.35
    target_peak_fraction: float = 0.08
    mach_focus_alpha: float = 0.60
    mach_focus_start: float = 3.2
    mach_focus_end: float = 4.5
    low_signal_alpha: float = 0.35
    low_signal_cap: float = 1.75
    early_stop_patience: int = 10**9
    early_stop_min_epochs: int = 60
    # --- method improvements (Tier-1 revision experiments) ---
    # Error-driven per-sample focal weighting: upweight cases by a running EMA of
    # their own training error, so capacity is redirected to the hard cases
    # (the near-sonic tail) instead of the hand-tuned Mach rules, which in this
    # dataset point at the already-easy high-Mach regime (see the dev log).
    focal_sample_alpha: float = 0.0   # 0 disables (default: original behaviour)
    focal_sample_cap: float = 4.0
    focal_ema_beta: float = 0.3
    # Project the predicted wall velocity onto the local tangent plane (exactly
    # tangential to the slip wall) before the velocity loss and metrics.
    slip_project: bool = False
    # Fourier positional-encoding bands (input feature capacity for sharp
    # features). Changing this changes the backbone input width, so a pretrained
    # prior trained at a different value is not reusable.
    fourier_bands: int = 4
    # Feed local surface descriptors (normal + shape) to the CORRECTION branch.
    # The prior is untouched, so shared_prior.pt stays reusable. Targets the
    # sharp features where the field error is concentrated.
    shape_features: bool = False
    # Weight of the local sharpness-matching term (0 disables). Penalises a
    # prediction that is locally smoother than the truth, which is the signature
    # of the error at the sharpest nodes.
    sharpness_weight: float = 0.0
    sharpness_one_sided: bool = True
    # Restrict the model's kNN to same-side neighbours (excludes the opposite
    # face of a thin fin). The physics geometry keeps the Euclidean kNN so the
    # normals/areas/calibration stay comparable with previous results.
    side_aware_knn: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.__dict__)
        payload["correction_point_channels"] = list(self.correction_point_channels)
        payload["correction_decoder_hidden"] = list(self.correction_decoder_hidden)
        return payload


def _physical_force_metrics(
    pred_cp: torch.Tensor,
    normals: torch.Tensor,
    areas: torch.Tensor,
    reference_force: torch.Tensor,
    calibration: torch.Tensor,
    ref_area: torch.Tensor,
) -> dict[str, float]:
    """Diagnostics: how close the calibrated integrated load is to the solver."""
    force = -(pred_cp * normals * areas).sum(dim=1) / ref_area.clamp_min(1e-12)
    force = force * calibration
    err = torch.linalg.norm(force - reference_force, dim=1)
    return {"force_abs_err": float(err.mean()), "force_ref_norm": float(torch.linalg.norm(reference_force, dim=1).mean())}


def _project_tangential(vector: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``vector`` along ``normals`` (slip-wall projection).

    ``vector`` and ``normals`` are [B, N, 3]; the returned tensor is the part of
    ``vector`` in the local tangent plane, i.e. exactly tangential at a slip wall.
    """
    normal_component = (vector * normals).sum(dim=-1, keepdim=True)
    return vector - normal_component * normals


def _local_shape_features(points: torch.Tensor, knn_idx: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return [B, N, 6] local surface descriptors from the kNN already available.

    Six channels: the outward-oriented local normal (3) plus three scale-free
    shape descriptors (surface variation, planarity, log elongation). They are
    fed to the correction branch so that it need not infer *where the surface is
    sharp* from coordinates alone -- the sharpest 10% of nodes carry ~69% of the
    total field error, while the current weighting already gives them only 2.2x
    weight. Computed on the GPU, a few ms per batch, and only for the correction
    branch, so the pretrained prior is reused unchanged.
    """
    from .kd_model import _gather_neighbor_features

    nbr = _gather_neighbor_features(points, knn_idx)                       # [B, N, k, 3]
    centred = nbr - nbr.mean(dim=2, keepdim=True)
    cov = torch.einsum("bnki,bnkj->bnij", centred, centred) / max(int(centred.shape[2]), 1)
    evals, evecs = torch.linalg.eigh(cov)                                  # ascending eigenvalues
    normal = evecs[..., 0]
    radial = points.clone()
    radial[..., 0] = 0.0                                                   # missile axis is x
    normal = torch.where((normal * radial).sum(dim=-1, keepdim=True) < 0.0, -normal, normal)
    l0, l1, l2 = evals[..., 0], evals[..., 1], evals[..., 2]
    total = (l0 + l1 + l2).clamp_min(eps)
    shape = torch.stack(
        [
            l0 / total,                                        # surface variation (sharpness)
            (l1 - l0) / l2.clamp_min(eps),                     # planarity
            torch.log(l2.clamp_min(eps) / total),              # scale-free elongation
        ],
        dim=-1,
    )
    return torch.cat([normal, shape], dim=-1)


def _run_field_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: AdamW | None,
    normalizers: FieldNormalizerBundle,
    grad_clip_norm: float,
    lambda_velocity: float,
    physics_weight: float,
    include_moments: bool,
    hard_point_alpha: float,
    hard_point_fraction: float,
    target_peak_alpha: float,
    target_peak_fraction: float,
    mach_focus_alpha: float,
    mach_focus_start: float,
    mach_focus_end: float,
    low_signal_alpha: float,
    low_signal_cap: float,
    low_signal_ref: float | None,
    point_jitter_std: float,
    grad_accum_steps: int,
    use_hard_example: bool,
    collect_predictions: bool = False,
    slip_project: bool = False,
    sample_error_ema: dict[str, float] | None = None,
    focal_sample_alpha: float = 0.0,
    focal_sample_cap: float = 4.0,
    focal_ema_beta: float = 0.3,
    use_shape_features: bool = False,
    sharpness_weight: float = 0.0,
    sharpness_one_sided: bool = True,
) -> dict[str, object]:
    training = optimizer is not None
    if hasattr(model, "prior") and hasattr(model, "correction"):
        # Keep the (frozen or partially frozen) prior deterministic during
        # training; only the correction branch runs in train mode.
        model.prior.eval()
        model.correction.train(training)
    else:
        model.train(training)
    total_loss = 0.0
    total_count = 0
    accum_index = 0
    physics_seen = 0.0
    physics_count = 0
    pred_cp_blocks: list[np.ndarray] = []
    true_cp_blocks: list[np.ndarray] = []
    pred_v_blocks: list[np.ndarray] = []
    true_v_blocks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    force_metrics: dict[str, list[float]] = {"force_abs_err": [], "force_ref_norm": []}
    sample_truth: defaultdict[str, list[float]] = defaultdict(list)
    sample_pred: defaultdict[str, list[float]] = defaultdict(list)

    num_batches = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index_in_epoch, batch in enumerate(loader):
        points_np = _batch_to_numpy(batch["points"]).astype(np.float32, copy=False)
        condition_np = _batch_to_numpy(batch["condition"]).astype(np.float32, copy=False)
        cp_np = _batch_to_numpy(batch["cp"]).astype(np.float32, copy=False)
        velocity_np = _batch_to_numpy(batch["velocity"]).astype(np.float32, copy=False)
        mask_np = _batch_to_numpy(batch["mask"]).astype(bool, copy=False)
        normals_np = _batch_to_numpy(batch["normal"]).astype(np.float32, copy=False)
        areas_np = _batch_to_numpy(batch["area"]).astype(np.float32, copy=False)

        points = normalizers.points.transform(points_np)
        condition = normalizers.condition.transform(condition_np)
        cp = normalizers.cp.transform(cp_np)
        velocity = normalizers.velocity.transform(velocity_np)

        raw_condition_t = _to_torch(condition_np, device)
        raw_points_t = _to_torch(points_np, device)
        raw_cp_t = _to_torch(cp_np, device)
        points_t = _to_torch(points, device)
        condition_t = _to_torch(condition, device)
        cp_t = _to_torch(cp, device)
        velocity_t = _to_torch(velocity, device)
        mask_t = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
        normals_t = _to_torch(normals_np, device)
        areas_t = _to_torch(areas_np, device)
        knn_idx_t = batch["knn_idx"].to(device) if batch.get("knn_idx") is not None else None
        # The model may run on a side-restricted neighbourhood while the physics
        # geometry keeps the Euclidean one.
        knn_model_t = batch["knn_idx_side"].to(device) if batch.get("knn_idx_side") is not None else knn_idx_t

        if training and point_jitter_std > 0:
            jitter = torch.randn_like(points_t) * point_jitter_std
            points_t = points_t + jitter * mask_t.unsqueeze(-1).to(points_t.dtype)

        sample_weights = _sample_priority_weights(
            raw_condition=raw_condition_t,
            raw_target=raw_cp_t,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            low_signal_alpha=low_signal_alpha,
            low_signal_cap=low_signal_cap,
            low_signal_ref=low_signal_ref,
        )
        sample_scale = sample_weights.reshape(sample_weights.shape[0], -1).mean(dim=1).mean().clamp_min(1e-6)
        v_mean = torch.as_tensor(normalizers.velocity.mean, device=device, dtype=torch.float32).view(1, 1, 3)
        v_std_dev = torch.as_tensor(normalizers.velocity.std, device=device, dtype=torch.float32).view(1, 1, 3)

        # Error-driven per-sample focal weight: upweight each case by a running
        # EMA of its own training error, so capacity follows the hard cases
        # rather than a hand-set Mach rule. Off by default (alpha = 0).
        if focal_sample_alpha > 0.0 and sample_error_ema:
            ref_error = float(np.mean(list(sample_error_ema.values()))) or 1.0
            focal_values = []
            for sample_id in batch["sample_id"]:
                error = sample_error_ema.get(sample_id)
                if error is None:
                    focal_values.append(1.0)
                else:
                    ratio = max(float(error), 1e-8) / max(ref_error, 1e-8)
                    focal_values.append(float(np.clip(
                        ratio ** focal_sample_alpha, 1.0 / focal_sample_cap, focal_sample_cap)))
            focal_t = torch.as_tensor(focal_values, device=device, dtype=sample_scale.dtype).mean()
            sample_scale = (sample_scale * focal_t).clamp_min(1e-6)

        with torch.set_grad_enabled(training):
            extra_t = None
            if use_shape_features and knn_idx_t is not None:
                extra_t = _local_shape_features(raw_points_t, knn_idx_t)
            features = model.forward_features(points_t, condition_t, mask=mask_t, knn_idx=knn_model_t,
                                             extra=extra_t)
            pred_cp = features["cp"]
            pred_velocity = features.get("velocity")
            if pred_velocity is None:
                raise ValueError("Field model did not return a velocity field; check out_channels")
            if slip_project:
                # Slip-wall projection: the wall velocity must be tangential, so
                # remove the normal component of both the prediction and the
                # target in physical units. The velocity loss and the reported
                # metrics are then computed on physically admissible vectors.
                pred_velocity = _project_tangential(pred_velocity * v_std_dev + v_mean, normals_t)
                pred_velocity = (pred_velocity - v_mean) / v_std_dev
                true_velocity_phys = _project_tangential(velocity_t * v_std_dev + v_mean, normals_t)
                velocity_t = (true_velocity_phys - v_mean) / v_std_dev
            # The physics term integrates physical Cp, but the network output is
            # in standardised units; convert back before integrating.
            cp_mean = torch.as_tensor(normalizers.cp.mean, device=device, dtype=pred_cp.dtype)
            cp_scale = torch.as_tensor(normalizers.cp.std, device=device, dtype=pred_cp.dtype)
            pred_cp_physical = pred_cp * cp_scale + cp_mean

            if use_hard_example:
                weights = build_hard_example_weights(
                    prediction=pred_cp.detach(),
                    target=cp_t,
                    topk_fraction=hard_point_fraction,
                    hard_point_alpha=hard_point_alpha,
                    target_peak_alpha=target_peak_alpha,
                    target_peak_fraction=target_peak_fraction,
                    mask=mask_t,
                )
            else:
                weights = build_pressure_weights(raw_pressure=raw_cp_t, raw_points=raw_points_t, mask=mask_t)

            valid_weights = weights[mask_t]
            field_loss, _ = weighted_field_loss(
                pred_cp=pred_cp[mask_t],
                true_cp=cp_t[mask_t],
                weights=valid_weights,
                pred_velocity=pred_velocity[mask_t],
                true_velocity=velocity_t[mask_t],
                lambda_v=lambda_velocity,
            )
            loss = field_loss * sample_scale

            if sharpness_weight > 0.0 and knn_idx_t is not None:
                sharp_loss = sharpness_matching_loss(
                    pred_cp=pred_cp,
                    true_cp=cp_t,
                    # standardised coordinates, i.e. the same length units the
                    # network sees: in physical units the mesh spacing (~1e-3 m)
                    # makes the gradient O(1e2-1e3) and its square would swamp
                    # the field loss.
                    points=points_t,
                    knn_idx=knn_idx_t,
                    mask=mask_t,
                    one_sided=sharpness_one_sided,
                )
                loss = loss + float(sharpness_weight) * sharp_loss

            if physics_weight > 0.0:
                phys = physics_consistency_loss(
                    pred_cp=pred_cp_physical,
                    normals=normals_t,
                    areas=areas_t,
                    reference_force=batch["ref_force"].to(device),
                    force_calibration=batch["force_calibration"].to(device),
                    ref_area=batch["ref_area"].to(device),
                    points=raw_points_t,
                    reference_moment=batch["ref_moment"].to(device),
                    moment_origin=batch["moment_origin"].to(device),
                    ref_length=batch["ref_length"].to(device),
                    include_moments=include_moments,
                )
                loss = loss + float(physics_weight) * phys
                physics_seen += float(phys.detach())
                physics_count += 1

            if training:
                accum_index += 1
                is_last_batch = batch_index_in_epoch == num_batches - 1
                remainder = num_batches % grad_accum_steps
                current_group_size = remainder if (is_last_batch and remainder != 0) else grad_accum_steps
                (loss / current_group_size).backward()
                if accum_index % grad_accum_steps == 0 or is_last_batch:
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        valid_count = int(mask_t.sum().item())
        total_loss += float(loss.item()) * valid_count
        total_count += valid_count
        pred_cp_blocks.append(pred_cp.detach().cpu().numpy()[mask_np])
        true_cp_blocks.append(cp[mask_np])
        if pred_velocity is not None:
            pred_v_blocks.append(pred_velocity.detach().cpu().numpy()[mask_np])
            true_v_blocks.append(velocity[mask_np])

        with torch.no_grad():
            if batch.get("ref_force") is not None:
                stats = _physical_force_metrics(
                    pred_cp_physical.detach(),
                    normals_t,
                    areas_t,
                    batch["ref_force"].to(device),
                    batch["force_calibration"].to(device),
                    batch["ref_area"].to(device),
                )
                for key, value in stats.items():
                    force_metrics[key].append(value)

            if training and sample_error_ema is not None:
                # Running per-case error used by the focal weighting next epoch.
                with torch.no_grad():
                    err_per_sample = (
                        (pred_cp.detach() - cp_t).abs() * mask_t.unsqueeze(-1)
                    ).sum(dim=(1, 2)) / mask_t.sum(dim=1).clamp_min(1.0)
                for sample_index, sample_id in enumerate(batch["sample_id"]):
                    value = float(err_per_sample[sample_index])
                    previous = sample_error_ema.get(sample_id)
                    sample_error_ema[sample_id] = (
                        value if previous is None else (1.0 - focal_ema_beta) * previous + focal_ema_beta * value
                    )

        if collect_predictions:
            pred_cp_raw = normalizers.cp.inverse_transform(pred_cp.detach().cpu().numpy())
            true_cp_raw = normalizers.cp.inverse_transform(cp_t.detach().cpu().numpy())
            pred_v_raw = normalizers.velocity.inverse_transform(pred_velocity.detach().cpu().numpy())
            true_v_raw = normalizers.velocity.inverse_transform(velocity_t.detach().cpu().numpy())
            point_ids_batch = _batch_to_numpy(batch["point_ids"]).astype(np.int64, copy=False)
            for batch_index, sample_id in enumerate(batch["sample_id"]):
                sample_mask = mask_np[batch_index]
                point_ids = point_ids_batch[batch_index][sample_mask]
                coords = points_np[batch_index][sample_mask]
                pred_sample = pred_cp_raw[batch_index, sample_mask, 0]
                truth_sample = true_cp_raw[batch_index, sample_mask, 0]
                abs_err = np.abs(pred_sample - truth_sample)
                sample_truth[sample_id].extend(truth_sample.tolist())
                sample_pred[sample_id].extend(pred_sample.tolist())
                vel_pred = pred_v_raw[batch_index, sample_mask]
                vel_true = true_v_raw[batch_index, sample_mask]
                for point_index, point_id in enumerate(point_ids):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "point_id": int(point_id),
                            "x": float(coords[point_index, 0]),
                            "y": float(coords[point_index, 1]),
                            "z": float(coords[point_index, 2]),
                            "true_cp": float(truth_sample[point_index]),
                            "pred_cp": float(pred_sample[point_index]),
                            "abs_error": float(abs_err[point_index]),
                            "true_vx": float(vel_true[point_index, 0]),
                            "pred_vx": float(vel_pred[point_index, 0]),
                            "true_vy": float(vel_true[point_index, 1]),
                            "pred_vy": float(vel_pred[point_index, 1]),
                            "true_vz": float(vel_true[point_index, 2]),
                            "pred_vz": float(vel_pred[point_index, 2]),
                        }
                    )

    pred_cp_std = np.concatenate(pred_cp_blocks, axis=0) if pred_cp_blocks else np.zeros((0, 1), np.float32)
    true_cp_std = np.concatenate(true_cp_blocks, axis=0) if true_cp_blocks else np.zeros((0, 1), np.float32)
    pred_v_std = np.concatenate(pred_v_blocks, axis=0) if pred_v_blocks else np.zeros((0, 3), np.float32)
    true_v_std = np.concatenate(true_v_blocks, axis=0) if true_v_blocks else np.zeros((0, 3), np.float32)
    return {
        "loss": total_loss / max(total_count, 1),
        "pred_cp_std": pred_cp_std,
        "true_cp_std": true_cp_std,
        "pred_v_std": pred_v_std,
        "true_v_std": true_v_std,
        "rows": rows,
        "force_metrics": {key: float(np.mean(value)) if value else float("nan") for key, value in force_metrics.items()},
        "physics_mean": (physics_seen / physics_count) if physics_count else float("nan"),
        "sample_metrics": {
            sample_id: _pressure_metrics(np.asarray(sample_truth[sample_id]), np.asarray(sample_pred[sample_id]))
            for sample_id in sample_truth
        },
    }


def _train_field_stage(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    normalizers: FieldNormalizerBundle,
    config: FieldTrainingConfig,
    epochs: int,
    lr: float,
    stage_name: str,
    output_dir: Path,
    low_signal_ref: float | None,
    use_hard_example: bool,
    physics_weight: float,
    select_by_rmse: bool,
    lr_schedule: str = "plateau",
    enable_focal: bool = True,
    sharpness_weight: float = 0.0,
) -> tuple[Path, Path, Path]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parameters, lr=lr, weight_decay=config.weight_decay)
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12)
        if lr_schedule == "plateau"
        else LambdaLR(optimizer, lr_lambda=lambda epoch: _cosine_warmup_lr_factor(epoch + 1, epochs, 0))
    )
    best_selection = float("inf")
    best_val_loss = float("inf")
    no_improve = 0
    sample_error_ema: dict[str, float] = {}  # per-case error EMA for focal weighting
    best_path = output_dir / f"{stage_name}_best.pt"
    last_path = output_dir / f"{stage_name}_last.pt"
    history_path = output_dir / f"{stage_name}_history.csv"
    if history_path.exists():
        history_path.unlink()
    print(f"[{stage_name}] start | epochs={epochs} | lr={lr:.3e} | params={sum(p.numel() for p in parameters)}", flush=True)

    for epoch in range(1, epochs + 1):
        train_result = _run_field_epoch(
            model=model, loader=train_loader, device=device, optimizer=optimizer, normalizers=normalizers,
            grad_clip_norm=config.grad_clip_norm, lambda_velocity=config.lambda_velocity, physics_weight=physics_weight,
            include_moments=config.include_moments, hard_point_alpha=config.hard_point_alpha,
            hard_point_fraction=config.hard_point_fraction, target_peak_alpha=config.target_peak_alpha,
            target_peak_fraction=config.target_peak_fraction, mach_focus_alpha=config.mach_focus_alpha,
            mach_focus_start=config.mach_focus_start, mach_focus_end=config.mach_focus_end,
            low_signal_alpha=config.low_signal_alpha, low_signal_cap=config.low_signal_cap,
            low_signal_ref=low_signal_ref, point_jitter_std=config.point_jitter_std,
            grad_accum_steps=config.grad_accum_steps, use_hard_example=use_hard_example,
            slip_project=config.slip_project, sample_error_ema=sample_error_ema,
            focal_sample_alpha=(config.focal_sample_alpha if enable_focal else 0.0),
            focal_sample_cap=config.focal_sample_cap,
            focal_ema_beta=config.focal_ema_beta,
            use_shape_features=config.shape_features,
            sharpness_weight=sharpness_weight,
            sharpness_one_sided=config.sharpness_one_sided,
        )
        val_result = _run_field_epoch(
            model=model, loader=val_loader, device=device, optimizer=None, normalizers=normalizers,
            grad_clip_norm=0.0, lambda_velocity=config.lambda_velocity, physics_weight=physics_weight,
            include_moments=config.include_moments, hard_point_alpha=config.hard_point_alpha,
            hard_point_fraction=config.hard_point_fraction, target_peak_alpha=config.target_peak_alpha,
            target_peak_fraction=config.target_peak_fraction, mach_focus_alpha=config.mach_focus_alpha,
            mach_focus_start=config.mach_focus_start, mach_focus_end=config.mach_focus_end,
            low_signal_alpha=config.low_signal_alpha, low_signal_cap=config.low_signal_cap,
            low_signal_ref=low_signal_ref, point_jitter_std=0.0, grad_accum_steps=1,
            use_hard_example=use_hard_example, slip_project=config.slip_project,
            use_shape_features=config.shape_features,
        )
        val_pred = normalizers.cp.inverse_transform(val_result["pred_cp_std"])
        val_truth = normalizers.cp.inverse_transform(val_result["true_cp_std"])
        val_metrics = _pressure_metrics(val_truth, val_pred)
        val_loss = float(val_result["loss"])
        name, selection = _validation_selection_metric(select_by_rmse, val_loss, val_metrics)
        if lr_schedule == "plateau":
            scheduler.step(selection)
        else:
            scheduler.step()
        best_val_loss = min(best_val_loss, val_loss)
        is_best = selection < best_selection
        if is_best:
            best_selection = selection
            no_improve = 0
            torch.save({"epoch": epoch, "config": config.to_dict(), "normalizers": normalizers.to_state_dict(),
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, best_path)
        else:
            no_improve += 1
        torch.save({"epoch": epoch, "config": config.to_dict(), "normalizers": normalizers.to_state_dict(),
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, last_path)
        _write_history_row(history_path, {
            "epoch": epoch, "train_loss_std": float(train_result["loss"]), "val_loss_std": val_loss,
            "val_mae": float(val_metrics["mae"]), "val_rmse": float(val_metrics["rmse"]),
            "selection_metric_name": name, "selection_metric": float(selection),
            "best_selection_metric": float(best_selection), "lr": float(optimizer.param_groups[0]["lr"]),
        }, is_new_file=not history_path.exists())
        print(f"[{stage_name}] epoch {epoch:03d}/{epochs} | train={train_result['loss']:.6f} | val={val_loss:.6f} | "
              f"val_rmse={val_metrics['rmse']:.6f} | {name}={selection:.6f}" + (" | best" if is_best else ""), flush=True)
        if epoch >= config.early_stop_min_epochs and no_improve >= config.early_stop_patience:
            print(f"[{stage_name}] early stop at epoch {epoch}", flush=True)
            break
    return best_path, last_path, history_path


def train_residual_tl(
    runtime_root: str | Path,
    output_dir: str | Path,
    g2_manifest: str | Path,
    g4_train_manifest: str | Path,
    g4_test_manifest: str | Path,
    config: FieldTrainingConfig | None = None,
    num_points: int | None = None,
    batch_size: int = 1,
    device: str = "",
    target_column: str = "Cp",
    pretrain_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    config = config or FieldTrainingConfig()
    seed_everything(config.seed)
    runtime_root = Path(runtime_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[residual-tl] device={resolved_device} | cuda={torch.cuda.is_available()}", flush=True)

    g2_root = runtime_root / "G2"
    g4_root = runtime_root / "G4"
    sample_lists = build_kd_sample_lists(
        runtime_root, g2_manifest=g2_manifest, g4_train_manifest=g4_train_manifest, g4_test_manifest=g4_test_manifest
    )
    g2_train_ids, g2_val_ids = stable_train_val_split(sample_lists["g2_train_ids"], config.val_fraction, config.seed)
    g4_train_ids, g4_val_ids = stable_train_val_split(sample_lists["g4_train_ids"], config.finetune_val_fraction, config.seed + 1)
    g4_test_ids = sample_lists["g4_test_ids"]
    if config.sample_limit is not None:
        limit = int(config.sample_limit)
        g2_train_ids = g2_train_ids[:limit]
        g2_val_ids = g2_val_ids[: max(1, limit // 4)]
        g4_train_ids = g4_train_ids[:limit]
        g4_val_ids = g4_val_ids[: max(1, limit // 4)]
        g4_test_ids = g4_test_ids[: max(1, limit // 2)]
    full_g4_train_ids = list(g4_train_ids)
    if config.finetune_limit is not None:
        g4_train_ids = g4_train_ids[: int(config.finetune_limit)]
    if config.pretrain_limit is not None:
        g2_train_ids = g2_train_ids[: int(config.pretrain_limit)]
    print(f"[residual-tl] split | g2_train={len(g2_train_ids)} g2_val={len(g2_val_ids)} | "
          f"g4_train={len(g4_train_ids)} g4_val={len(g4_val_ids)} g4_test={len(g4_test_ids)}", flush=True)

    g2_signal_ref = _estimate_target_strength_reference(g2_root, g2_train_ids, target_column)
    g4_signal_ref = _estimate_target_strength_reference(g4_root, g4_train_ids, target_column)
    _validate_signal_reference("G2", g2_signal_ref)
    _validate_signal_reference("G4", g4_signal_ref)

    def _normalizers(fidelity_root, ids):
        std = fit_fidelity_standardizers(
            fidelity_root=fidelity_root, sample_ids=ids, num_points=num_points, seed=config.seed,
            target_column=target_column, include_fields=True,
        )
        return FieldNormalizerBundle(points=std["points"], condition=std["condition"], cp=std["cp"], velocity=std["velocity"])

    g2_normalizers = _normalizers(g2_root, g2_train_ids)
    g4_normalizers = _normalizers(g4_root, full_g4_train_ids)

    def _sampler(fidelity_root, ids, generator_seed):
        weights = _build_mach_sampling_weights(
            fidelity_root=fidelity_root, sample_ids=ids, mach_focus_alpha=config.mach_focus_alpha,
            mach_focus_start=config.mach_focus_start, mach_focus_end=config.mach_focus_end,
        )
        return WeightedRandomSampler(weights=weights, num_samples=len(ids), replacement=True,
                                     generator=torch.Generator().manual_seed(generator_seed))

    def _loader(fidelity_root, ids, sampler):
        return build_fidelity_pressure_loader(
            fidelity_root=fidelity_root, sample_ids=ids, num_points=num_points, batch_size=batch_size,
            seed=config.seed, shuffle=False, sampler=sampler, target_column=target_column,
            sampling_mode="pressure_aware", knn_k=config.knn_k, include_fields=True,
            side_aware_knn=config.side_aware_knn,
        )

    g2_train_loader = _loader(g2_root, g2_train_ids, _sampler(g2_root, g2_train_ids, config.seed))
    g2_val_loader = _loader(g2_root, g2_val_ids, None)
    g4_train_loader = _loader(g4_root, g4_train_ids, _sampler(g4_root, g4_train_ids, config.seed + 1))
    g4_val_loader = _loader(g4_root, g4_val_ids, None)
    g4_test_loader = _loader(g4_root, g4_test_ids, None)

    config_dict = config.to_dict()
    save_json(output_dir / "field_config.json", config_dict)
    save_json(output_dir / "split_manifest.json", {
        "g2_train_ids": g2_train_ids, "g2_val_ids": g2_val_ids,
        "g4_train_ids": g4_train_ids, "g4_val_ids": g4_val_ids, "g4_test_ids": g4_test_ids,
    })
    save_json(output_dir / "normalizers.json", {
        "g2_pretrain": g2_normalizers.to_state_dict(), "g4_finetune": g4_normalizers.to_state_dict(),
    })

    # Stage 1: pretrain the dual-output prior on low-fidelity G2, unless a
    # previously trained prior is supplied/reusable (share one prior across the
    # layer-freezing / physics / data-budget ablations instead of re-pretraining).
    prior = build_field_model(knn_k=config.knn_k, dropout=config.dropout, out_channels=FIELD_CHANNELS,
                              fourier_bands=config.fourier_bands).to(resolved_device)
    if config.from_scratch:
        print("[residual-tl] from-scratch: no pretraining", flush=True)
        pretrain_best = None
    else:
        shared_prior = output_dir.parent / "shared_prior.pt"
        if config.pretrain_limit is not None:
            reuse_path = None  # force fresh pretraining at the requested G2 budget
        elif pretrain_checkpoint is not None:
            reuse_path = Path(pretrain_checkpoint)
        else:
            reuse_path = shared_prior if shared_prior.is_file() else None
        if reuse_path is not None and Path(reuse_path).is_file():
            print(f"[residual-tl] stage 1 skipped: reusing low-fidelity prior from {reuse_path}", flush=True)
            prior.load_state_dict(torch.load(reuse_path, map_location=resolved_device)["model_state_dict"])
            pretrain_best = Path(reuse_path)
        else:
            print("[residual-tl] stage 1: pretraining dual-output prior on G2", flush=True)
            pretrain_best, _, _ = _train_field_stage(
                model=prior, train_loader=g2_train_loader, val_loader=g2_val_loader, device=resolved_device,
                normalizers=g2_normalizers, config=config, epochs=config.pretrain_epochs, lr=config.pretrain_lr,
                stage_name="field_pretrain", output_dir=output_dir, low_signal_ref=g2_signal_ref,
                use_hard_example=False, physics_weight=config.physics_weight if config.physics_in_pretrain else 0.0,
                select_by_rmse=False, enable_focal=False,
            )
            prior.load_state_dict(torch.load(pretrain_best, map_location=resolved_device)["model_state_dict"])
            # Only cache a full-data prior (not a smoke-test or reduced-G2 prior).
            if config.sample_limit is None and config.pretrain_limit is None:
                shared_prior.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pretrain_best, shared_prior)
                print(f"[residual-tl] cached reusable prior at {shared_prior}", flush=True)
            print("[residual-tl] stage 1 complete", flush=True)

    if config.from_scratch or not config.use_residual:
        print(f"[residual-tl] stage 2: direct fine-tuning on G4 (no correction branch) | from_scratch={config.from_scratch}", flush=True)
        model = prior
    else:
        # Stage 2: residual fine-tuning on high-fidelity G4.
        correction = build_field_model(
            knn_k=config.knn_k, dropout=config.dropout,
            point_channels=tuple(config.correction_point_channels),
            decoder_hidden=tuple(config.correction_decoder_hidden),
            out_channels=FIELD_CHANNELS,
            fourier_bands=config.fourier_bands,
            extra_channels=(6 if config.shape_features else 0),
        ).to(resolved_device)
        zero_init_correction(correction)
        apply_freezing_strategy(prior, config.freezing_strategy)
        model = ResidualFieldModel(prior, correction).to(resolved_device)
        print(f"[residual-tl] stage 2: residual fine-tuning on G4 | freezing={config.freezing_strategy}", flush=True)
    finetune_best, _, _ = _train_field_stage(
        model=model, train_loader=g4_train_loader, val_loader=g4_val_loader, device=resolved_device,
        normalizers=g4_normalizers, config=config, epochs=config.finetune_epochs, lr=config.finetune_lr,
        stage_name="field_finetune", output_dir=output_dir, low_signal_ref=g4_signal_ref,
        use_hard_example=True, physics_weight=config.physics_weight, select_by_rmse=True,
        sharpness_weight=config.sharpness_weight,
    )
    model.load_state_dict(torch.load(finetune_best, map_location=resolved_device)["model_state_dict"])
    print("[residual-tl] stage 2 complete", flush=True)

    # Evaluation on the frozen G4 test set.
    eval_result = _run_field_epoch(
        model=model, loader=g4_test_loader, device=resolved_device, optimizer=None, normalizers=g4_normalizers,
        grad_clip_norm=0.0, lambda_velocity=config.lambda_velocity, physics_weight=config.physics_weight,
        include_moments=config.include_moments, hard_point_alpha=config.hard_point_alpha,
        hard_point_fraction=config.hard_point_fraction, target_peak_alpha=config.target_peak_alpha,
        target_peak_fraction=config.target_peak_fraction, mach_focus_alpha=config.mach_focus_alpha,
        mach_focus_start=config.mach_focus_start, mach_focus_end=config.mach_focus_end,
        low_signal_alpha=config.low_signal_alpha, low_signal_cap=config.low_signal_cap,
        low_signal_ref=g4_signal_ref, point_jitter_std=0.0, grad_accum_steps=1, use_hard_example=True,
        collect_predictions=True, slip_project=config.slip_project,
        use_shape_features=config.shape_features,
    )
    pred_cp = g4_normalizers.cp.inverse_transform(eval_result["pred_cp_std"])
    true_cp = g4_normalizers.cp.inverse_transform(eval_result["true_cp_std"])
    pred_v = g4_normalizers.velocity.inverse_transform(eval_result["pred_v_std"])
    true_v = g4_normalizers.velocity.inverse_transform(eval_result["true_v_std"])
    cp_metrics = _pressure_metrics(true_cp, pred_cp)
    vel_metrics = vector_metrics(torch.from_numpy(true_v), torch.from_numpy(pred_v))
    vm = _pressure_metrics(true_cp, pred_cp)
    vm["r2"] = 1.0 - float(((pred_cp - true_cp) ** 2).sum() / ((true_cp - true_cp.mean()) ** 2).sum())
    metrics = {
        "pressure": cp_metrics,
        "pressure_r2": vm["r2"],
        "velocity": vel_metrics,
        "force_diagnostics": eval_result["force_metrics"],
        "loss_std": float(eval_result["loss"]),
    }
    print(f"[residual-tl] test metrics: R2(p)={metrics['pressure_r2']:.4f} rmse(p)={cp_metrics['rmse']:.6f} "
          f"R2(v)={vel_metrics['r2']:.4f} rmse(v)={vel_metrics['rmse']:.6f}", flush=True)

    save_json(output_dir / "field_metrics.json", metrics)
    _save_predictions_csv(output_dir / "field_predictions.csv", eval_result["rows"])
    _save_sample_metrics_csv(output_dir / "field_sample_metrics.csv", eval_result["sample_metrics"])
    summary = {
        "pretrain_best_checkpoint": str(pretrain_best),
        "finetune_best_checkpoint": str(finetune_best),
        "field_config": config_dict,
    }
    save_json(output_dir / "run_summary.json", summary)
    return {"metrics": metrics, "summary": summary, "pretrain_best": str(pretrain_best), "finetune_best": str(finetune_best)}
