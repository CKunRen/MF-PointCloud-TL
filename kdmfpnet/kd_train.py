from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler

from .losses import build_hard_example_weights, build_pressure_weights, ramp_weight, weighted_pressure_loss
from .kd_data import (
    DEFAULT_TARGET_COLUMN,
    build_fidelity_pressure_loader,
    fit_fidelity_standardizers,
    load_fidelity_pressure_record,
    list_fidelity_sample_ids,
    load_sample_id_manifest,
)
from .kd_model import build_student_model, build_teacher_model, PressureKDRegressor
from .utils import Standardizer, save_json, seed_everything, stable_train_val_split


def _as_int_tuple(values: Sequence[int] | None, fallback: Sequence[int]) -> tuple[int, ...]:
    if values is None:
        return tuple(int(v) for v in fallback)
    return tuple(int(v) for v in values)


@dataclass
class KDTrainingConfig:
    data_root: str
    output_dir: str
    num_points: int | None = None
    batch_size: int = 1
    teacher_epochs: int = 120
    student_pretrain_epochs: int = 60
    # "g4": student_pretrain trains from random init on G4 supervision only
    # (original default). "g2": student_pretrain instead reuses the G2
    # train/val loaders (same data the teacher trains on), giving the
    # student the same low-fidelity-informed initialization as the teacher
    # before KD fine-tuning ever sees G4. Warm-starting the student this way
    # (rather than from random weights) measured R2=0.955 vs 0.949 in a
    # controlled comparison (see runs/kd_warmstart/20260916_230552 and
    # 开发日志.md's KD-vs-TL section) -- the previous "g4" default understated
    # the method's own best achievable accuracy, not a property of KD itself.
    student_pretrain_fidelity: str = "g2"
    student_epochs: int = 180
    teacher_lr: float = 8e-4
    student_lr: float = 8e-4
    weight_decay: float = 1e-4
    seed: int = 42
    val_fraction: float = 0.1
    student_val_fraction: float = 0.1
    target_column: str = DEFAULT_TARGET_COLUMN
    device: str = ""
    grad_clip_norm: float = 1.0
    teacher_point_channels: tuple[int, ...] = (96, 192, 384)
    teacher_condition_hidden: tuple[int, ...] = (64, 128)
    teacher_geometry_hidden: tuple[int, ...] = (384, 256)
    teacher_decoder_hidden: tuple[int, ...] = (384, 192, 96)
    student_point_channels: tuple[int, ...] = (64, 128, 256)
    student_condition_hidden: tuple[int, ...] = (32, 64)
    student_geometry_hidden: tuple[int, ...] = (256,)
    student_decoder_hidden: tuple[int, ...] = (256, 128, 64)
    residual_hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.08
    student_dropout: float = 0.08
    student_weight_decay: float = 1e-4
    fourier_bands: int = 4
    use_residual_decoder: bool = True
    use_edge_conv: bool = True
    knn_k: int = 16
    response_alpha: float = 1.0
    feature_alpha: float = 0.35
    hard_point_alpha: float = 1.5
    hard_point_fraction: float = 0.1
    target_peak_alpha: float = 0.35
    target_peak_fraction: float = 0.08
    mach_focus_alpha: float = 0.75
    mach_focus_start: float = 3.2
    mach_focus_end: float = 4.5
    aoa_focus_alpha: float = 0.0
    aoa_focus_start: float = 6.0
    aoa_focus_end: float = 12.0
    low_signal_alpha: float = 0.35
    low_signal_cap: float = 1.75
    kd_warmup_epochs: int = 20
    kd_ramp_epochs: int = 40
    point_alpha: float = 0.0
    point_feat_distill_channels: int | None = None
    grad_accum_steps: int = 1
    lr_schedule: str = "plateau"
    lr_warmup_epochs: int = 0
    early_stop_patience: int = 40
    early_stop_min_epochs: int = 60

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.__dict__)
        payload["teacher_point_channels"] = list(self.teacher_point_channels)
        payload["teacher_condition_hidden"] = list(self.teacher_condition_hidden)
        payload["teacher_geometry_hidden"] = list(self.teacher_geometry_hidden)
        payload["teacher_decoder_hidden"] = list(self.teacher_decoder_hidden)
        payload["student_point_channels"] = list(self.student_point_channels)
        payload["student_condition_hidden"] = list(self.student_condition_hidden)
        payload["student_geometry_hidden"] = list(self.student_geometry_hidden)
        payload["student_decoder_hidden"] = list(self.student_decoder_hidden)
        payload["residual_hidden"] = list(self.residual_hidden)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "KDTrainingConfig":
        data = dict(payload)
        data["teacher_point_channels"] = _as_int_tuple(data.get("teacher_point_channels"), cls.teacher_point_channels)
        data["teacher_condition_hidden"] = _as_int_tuple(data.get("teacher_condition_hidden"), cls.teacher_condition_hidden)
        data["teacher_geometry_hidden"] = _as_int_tuple(data.get("teacher_geometry_hidden"), cls.teacher_geometry_hidden)
        data["teacher_decoder_hidden"] = _as_int_tuple(data.get("teacher_decoder_hidden"), cls.teacher_decoder_hidden)
        data["student_point_channels"] = _as_int_tuple(data.get("student_point_channels"), cls.student_point_channels)
        data["student_condition_hidden"] = _as_int_tuple(data.get("student_condition_hidden"), cls.student_condition_hidden)
        data["student_geometry_hidden"] = _as_int_tuple(data.get("student_geometry_hidden"), cls.student_geometry_hidden)
        data["student_decoder_hidden"] = _as_int_tuple(data.get("student_decoder_hidden"), cls.student_decoder_hidden)
        data["residual_hidden"] = _as_int_tuple(data.get("residual_hidden"), cls.residual_hidden)
        return cls(**data)


@dataclass
class KDNormalizerBundle:
    points: Standardizer
    condition: Standardizer
    cp: Standardizer

    def to_state_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            "points": self.points.to_state_dict(),
            "condition": self.condition.to_state_dict(),
            "cp": self.cp.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, dict[str, list[float]]]) -> "KDNormalizerBundle":
        return cls(
            points=Standardizer.from_state_dict(payload["points"]),
            condition=Standardizer.from_state_dict(payload["condition"]),
            cp=Standardizer.from_state_dict(payload["cp"]),
        )


@dataclass
class KDTrainingResult:
    output_dir: Path
    teacher_best_checkpoint: Path
    student_best_checkpoint: Path
    teacher_metrics_json: Path
    student_metrics_json: Path
    student_predictions_csv: Path
    student_sample_metrics_csv: Path
    split_manifest_json: Path
    normalizers_json: Path


def _device_from_config(device_name: str) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cosine_warmup_lr_factor(epoch: int, total_epochs: int, warmup_epochs: int, min_lr_ratio: float = 0.05) -> float:
    """LR multiplier: linear warmup then cosine decay to min_lr_ratio*base_lr.

    epoch is 1-indexed (matches _train_one_stage's epoch loop). Unlike
    ReduceLROnPlateau (which only drops the LR after `patience` epochs of no
    val-metric improvement, and never raises it back), this follows a fixed
    schedule known in advance -- the "cosine annealing with warmup" tested as
    an alternative in 开发日志.md 38's attempt to close the remaining 0.001
    R2 gap to the 0.95 target without touching KD.
    """
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return epoch / float(warmup_epochs)
    decay_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(epoch - warmup_epochs, decay_epochs) / float(decay_epochs)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _to_torch(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)


def _batch_to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _convert_standardized_target_space(
    values: torch.Tensor,
    source: Standardizer,
    target: Standardizer,
) -> torch.Tensor:
    source_mean = torch.as_tensor(source.mean, device=values.device, dtype=values.dtype)
    source_std = torch.as_tensor(source.std, device=values.device, dtype=values.dtype)
    target_mean = torch.as_tensor(target.mean, device=values.device, dtype=values.dtype)
    target_std = torch.as_tensor(target.std, device=values.device, dtype=values.dtype)
    raw_values = values * source_std + source_mean
    return (raw_values - target_mean) / target_std.clamp_min(1e-8)


def _linear_focus_ramp(value: torch.Tensor, start: float, end: float) -> torch.Tensor:
    """Ramp from 0 at `start` to 1 at `end`, clamped to [0, 1].

    Works for both directions: start<end ramps up with increasing value
    (e.g. "focus on high Mach"), start>end ramps up with decreasing value
    (e.g. "focus on low Mach") -- unlike a plain (v - start) / (end - start)
    with a max(., eps) denominator, which breaks for start>end because the
    eps floor collapses the span instead of flipping its sign.
    """
    span = float(end) - float(start)
    if abs(span) < 1e-6:
        span = 1e-6 if span >= 0 else -1e-6
    return ((value - float(start)) / span).clamp(0.0, 1.0)


def _sample_priority_weights(
    raw_condition: torch.Tensor,
    raw_target: torch.Tensor,
    mach_focus_alpha: float,
    mach_focus_start: float,
    mach_focus_end: float,
    low_signal_alpha: float,
    low_signal_cap: float,
    low_signal_ref: float | None = None,
    aoa_focus_alpha: float = 0.0,
    aoa_focus_start: float = 6.0,
    aoa_focus_end: float = 12.0,
) -> torch.Tensor:
    weights = torch.ones((raw_condition.shape[0], 1, 1), device=raw_condition.device, dtype=raw_condition.dtype)
    mach = raw_condition[:, 0:1].unsqueeze(-1)
    mach_focus = _linear_focus_ramp(mach, mach_focus_start, mach_focus_end)
    weights = weights * (1.0 + float(mach_focus_alpha) * mach_focus)

    if aoa_focus_alpha > 0:
        aoa = raw_condition[:, 1:2].unsqueeze(-1)
        aoa_focus = _linear_focus_ramp(aoa, aoa_focus_start, aoa_focus_end)
        weights = weights * (1.0 + float(aoa_focus_alpha) * aoa_focus)

    if low_signal_alpha > 0:
        signal_strength = raw_target.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        if low_signal_ref is None:
            ref_signal = signal_strength.mean().detach().clamp_min(1e-6)
        else:
            ref_signal = torch.as_tensor(float(low_signal_ref), device=raw_target.device, dtype=raw_target.dtype).clamp_min(1e-6)
        low_signal_focus = (ref_signal / signal_strength).clamp(1.0, float(low_signal_cap))
        weights = weights * (1.0 + float(low_signal_alpha) * (low_signal_focus - 1.0))

    return weights


def _estimate_target_strength_reference(
    fidelity_root: str | Path,
    sample_ids: Sequence[str],
    target_column: str,
) -> float:
    root = Path(fidelity_root)
    values: list[float] = []
    for sample_id in sample_ids:
        record = load_fidelity_pressure_record(
            sample_id=sample_id,
            fidelity_root=root,
            num_points=None,
            seed=0,
            target_column=target_column,
            sampling_mode="random",
        )
        cp = np.asarray(record["cp"], dtype=np.float32)
        values.append(float(np.mean(np.abs(cp))))
    if not values:
        return 1.0
    return float(np.mean(values))


def _validate_signal_reference(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0 or value > 10.0:
        raise ValueError(
            f"{name} signal reference is invalid: {value!r}. "
            "Check CFD convergence, conservative variables, and Cp reconstruction."
        )


def _build_mach_sampling_weights(
    fidelity_root: str | Path,
    sample_ids: Sequence[str],
    mach_focus_alpha: float,
    mach_focus_start: float,
    mach_focus_end: float,
    aoa_focus_alpha: float = 0.0,
    aoa_focus_start: float = 6.0,
    aoa_focus_end: float = 12.0,
) -> np.ndarray:
    root = Path(fidelity_root)
    weights = []
    for sample_id in sample_ids:
        params_path = root / sample_id / "parameters.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        mach = float(params["mach"])
        mach_focus = float(np.clip((mach - float(mach_focus_start)) / _nonzero_span(mach_focus_start, mach_focus_end), 0.0, 1.0))
        weight = 1.0 + float(mach_focus_alpha) * mach_focus
        if aoa_focus_alpha > 0:
            aoa = float(params["aoa_deg"])
            aoa_focus = float(np.clip((aoa - float(aoa_focus_start)) / _nonzero_span(aoa_focus_start, aoa_focus_end), 0.0, 1.0))
            weight *= 1.0 + float(aoa_focus_alpha) * aoa_focus
        weights.append(weight)
    return np.asarray(weights, dtype=np.float64)


def _nonzero_span(start: float, end: float) -> float:
    span = float(end) - float(start)
    if abs(span) < 1e-6:
        return 1e-6 if span >= 0 else -1e-6
    return span


def build_kd_sample_lists(
    data_root: str | Path,
    g2_manifest: str | Path | None = None,
    g4_train_manifest: str | Path | None = None,
    g4_test_manifest: str | Path | None = None,
) -> dict[str, list[str]]:
    """Resolve the G2/G4 train/test id lists used for KD training.

    g2_manifest / g4_train_manifest / g4_test_manifest, when given, each point
    to a fixed sample-id list such as `usable_g2_train_800.txt` (see
    load_sample_id_manifest) and override the directory-scan `_sample_is_clean`
    filter for that split. Any split without a manifest falls back to the
    directory-scan filter, e.g. while that fidelity is still being filled in.
    """
    root = Path(data_root)
    g2_root = root / "G2"
    g4_root = root / "G4"
    g2_train_all = list_fidelity_sample_ids(g2_root, split="train")
    g4_train_all = list_fidelity_sample_ids(g4_root, split="train")
    g4_test_all = list_fidelity_sample_ids(g4_root, split="test")

    def resolve(root_dir: Path, all_ids: list[str], manifest, split, label) -> list[str]:
        if manifest is not None:
            manifest_ids = set(load_sample_id_manifest(manifest))
            missing = manifest_ids - set(all_ids)
            if missing:
                raise ValueError(
                    "%s lists %d sample id(s) not found under %s, e.g. %s"
                    % (label, len(missing), root_dir, sorted(missing)[:5])
                )
            return sorted(manifest_ids)
        return list_fidelity_sample_ids(root_dir, split=split, clean_only=True)

    g2_train_ids = resolve(g2_root, g2_train_all, g2_manifest, "train", "g2_manifest")
    g4_train_ids = resolve(g4_root, g4_train_all, g4_train_manifest, "train", "g4_train_manifest")
    g4_test_ids = resolve(g4_root, g4_test_all, g4_test_manifest, "test", "g4_test_manifest")
    return {
        "g2_train_ids": g2_train_ids,
        "g2_train_all_ids": g2_train_all,
        "g2_train_excluded_ids": sorted(set(g2_train_all) - set(g2_train_ids)),
        "g4_train_ids": g4_train_ids,
        "g4_train_all_ids": g4_train_all,
        "g4_train_excluded_ids": sorted(set(g4_train_all) - set(g4_train_ids)),
        "g4_test_ids": g4_test_ids,
        "g4_test_all_ids": g4_test_all,
        "g4_test_excluded_ids": sorted(set(g4_test_all) - set(g4_test_ids)),
    }


def kd_loss(
    student_cp: torch.Tensor,
    target_cp: torch.Tensor,
    teacher_cp: torch.Tensor,
    student_distill: torch.Tensor,
    teacher_distill: torch.Tensor,
    weights: torch.Tensor,
    response_alpha: float = 1.0,
    feature_alpha: float = 0.35,
    student_point_distill: torch.Tensor | None = None,
    teacher_point_distill: torch.Tensor | None = None,
    point_alpha: float = 0.0,
) -> torch.Tensor:
    supervised = weighted_pressure_loss(student_cp, target_cp, weights)
    response = weighted_pressure_loss(student_cp, teacher_cp, weights)
    feature = F.mse_loss(student_distill, teacher_distill)
    loss = supervised + float(response_alpha) * response + float(feature_alpha) * feature
    # Point-level feature distillation: aligns the per-point local-context
    # feature map (post-EdgeConv) rather than only the globally pooled geo/
    # hard_geo vectors used by `feature`. teacher_cp/student_cp above are
    # already computed from the identical batch of points (see _run_epoch),
    # so point_distill's point dimension lines up 1:1 between teacher and
    # student without any extra correspondence step. Off by default
    # (point_alpha=0.0) so existing callers/checkpoints are unaffected.
    if point_alpha > 0 and student_point_distill is not None and teacher_point_distill is not None:
        point_feature = F.mse_loss(student_point_distill, teacher_point_distill)
        loss = loss + float(point_alpha) * point_feature
    return loss


def _pressure_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_abs_true = float(np.mean(np.abs(y_true)))
    return {
        "mae": mae,
        "rmse": rmse,
        "mean_abs_true": mean_abs_true,
        "nmae": float(mae / mean_abs_true) if mean_abs_true > 1e-12 else float("nan"),
        "nrmse": float(rmse / mean_abs_true) if mean_abs_true > 1e-12 else float("nan"),
    }


def _validation_selection_metric(
    select_by_rmse: bool,
    val_loss: float,
    val_metrics: dict[str, float],
) -> tuple[str, float]:
    if select_by_rmse:
        return "val_rmse", float(val_metrics["rmse"])
    return "val_loss_std", float(val_loss)


def _write_history_row(history_path: Path, row: dict[str, object], is_new_file: bool) -> None:
    fieldnames = list(row.keys())
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def _save_predictions_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No prediction rows to save")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_sample_metrics_csv(path: Path, sample_metrics: dict[str, dict[str, float]]) -> None:
    rows = []
    for sample_id, metrics in sorted(sample_metrics.items()):
        row = {"sample_id": sample_id}
        row.update(metrics)
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else ["sample_id", "mae", "rmse"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _run_epoch(
    model: PressureKDRegressor,
    loader,
    device: torch.device,
    optimizer: AdamW | None,
    normalizers: KDNormalizerBundle,
    grad_clip_norm: float,
    teacher_model: PressureKDRegressor | None = None,
    teacher_normalizers: KDNormalizerBundle | None = None,
    response_alpha: float = 1.0,
    feature_alpha: float = 0.35,
    hard_point_alpha: float = 1.5,
    hard_point_fraction: float = 0.1,
    target_peak_alpha: float = 0.35,
    target_peak_fraction: float = 0.08,
    mach_focus_alpha: float = 0.75,
    mach_focus_start: float = 3.2,
    mach_focus_end: float = 4.5,
    aoa_focus_alpha: float = 0.0,
    aoa_focus_start: float = 6.0,
    aoa_focus_end: float = 12.0,
    low_signal_alpha: float = 0.35,
    low_signal_cap: float = 1.75,
    low_signal_ref: float | None = None,
    point_alpha: float = 0.0,
    point_jitter_std: float = 0.0,
    grad_accum_steps: int = 1,
    collect_predictions: bool = False,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    if teacher_model is not None:
        teacher_model.eval()
        if teacher_normalizers is None:
            raise ValueError("teacher_normalizers is required when teacher_model is provided")
    total_loss = 0.0
    total_count = 0
    # batch_size=1 (the KD training default) means every optimizer.step() is
    # a single-sample gradient -- noisy relative to the dataset's genuine
    # per-Mach-band variation. Accumulating grad_accum_steps batches before
    # stepping approximates a larger effective batch without the O(N^2)-in-
    # points memory cost of actually batching multiple 20k-point clouds
    # together. optimizer.step() is skipped on non-boundary batches; the
    # final partial group at the end of an epoch still steps once (see
    # accum_index check below) so no gradient is silently dropped.
    accum_index = 0
    pred_blocks = []
    target_blocks = []
    rows = []
    sample_truth: DefaultDict[str, list[float]] = defaultdict(list)
    sample_pred: DefaultDict[str, list[float]] = defaultdict(list)

    num_batches = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index_in_epoch, batch in enumerate(loader):
        points_np = _batch_to_numpy(batch["points"]).astype(np.float32, copy=False)
        condition_np = _batch_to_numpy(batch["condition"]).astype(np.float32, copy=False)
        cp_np = _batch_to_numpy(batch["cp"]).astype(np.float32, copy=False)
        mask_np = (
            _batch_to_numpy(batch["mask"]).astype(bool, copy=False)
            if batch.get("mask") is not None
            else np.ones(points_np.shape[:2], dtype=bool)
        )
        points = normalizers.points.transform(points_np)
        condition = normalizers.condition.transform(condition_np)
        cp = normalizers.cp.transform(cp_np)
        raw_condition_t = _to_torch(condition_np, device)
        raw_points_t = _to_torch(points_np, device)
        raw_cp_t = _to_torch(cp_np, device)
        points_t = _to_torch(points, device)
        condition_t = _to_torch(condition, device)
        cp_t = _to_torch(cp, device)
        mask_t = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
        if training and point_jitter_std > 0:
            # Jitter applied in already-standardized space (per-axis std=1
            # post-normalization) so a single scalar is scale-invariant
            # across x vs y/z, whose physical stds differ ~9x (see
            # EdgeConvBlock.forward's docstring). Train-only (gated on
            # `training`) and masked to padding-free rows so padded zero rows
            # don't get pushed off zero and treated as real geometry.
            jitter = torch.randn_like(points_t) * point_jitter_std
            points_t = points_t + jitter * mask_t.unsqueeze(-1).to(points_t.dtype)
        knn_idx_t = batch["knn_idx"].to(device) if batch.get("knn_idx") is not None else None
        sample_weights = _sample_priority_weights(
            raw_condition=raw_condition_t,
            raw_target=raw_cp_t,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            aoa_focus_alpha=aoa_focus_alpha,
            aoa_focus_start=aoa_focus_start,
            aoa_focus_end=aoa_focus_end,
            low_signal_alpha=low_signal_alpha,
            low_signal_cap=low_signal_cap,
            low_signal_ref=low_signal_ref,
        )
        sample_scale = sample_weights.reshape(sample_weights.shape[0], -1).mean(dim=1).mean().clamp_min(1e-6)

        with torch.set_grad_enabled(training):
            student_features = model.forward_features(points_t, condition_t, mask=mask_t, knn_idx=knn_idx_t)
            student_cp = student_features["cp"]
            valid_student_cp = student_cp[mask_t]
            valid_target_cp = cp_t[mask_t]
            if teacher_model is None:
                weights = build_pressure_weights(
                    raw_pressure=raw_cp_t,
                    raw_points=raw_points_t,
                    mask=mask_t,
                )
                valid_weights = weights[mask_t]
                loss = weighted_pressure_loss(valid_student_cp, valid_target_cp, valid_weights) * sample_scale
            else:
                with torch.no_grad():
                    teacher_points = teacher_normalizers.points.transform(points_np)
                    teacher_condition = teacher_normalizers.condition.transform(condition_np)
                    teacher_points_t = _to_torch(teacher_points, device)
                    teacher_condition_t = _to_torch(teacher_condition, device)
                    teacher_features = teacher_model.forward_features(
                        teacher_points_t,
                        teacher_condition_t,
                        mask=mask_t,
                        knn_idx=knn_idx_t,
                    )
                    teacher_cp_student_space = _convert_standardized_target_space(
                        teacher_features["cp"],
                        source=teacher_normalizers.cp,
                        target=normalizers.cp,
                    )
                valid_teacher_cp = teacher_cp_student_space[mask_t]
                weights = build_hard_example_weights(
                    prediction=student_cp.detach(),
                    target=cp_t,
                    topk_fraction=hard_point_fraction,
                    hard_point_alpha=hard_point_alpha,
                    target_peak_alpha=target_peak_alpha,
                    target_peak_fraction=target_peak_fraction,
                    mask=mask_t,
                )
                valid_weights = weights[mask_t]
                student_point_distill = student_features.get("point_distill")
                teacher_point_distill = teacher_features.get("point_distill")
                valid_student_point_distill = (
                    student_point_distill.transpose(1, 2)[mask_t] if student_point_distill is not None else None
                )
                valid_teacher_point_distill = (
                    teacher_point_distill.transpose(1, 2)[mask_t] if teacher_point_distill is not None else None
                )
                loss = kd_loss(
                    student_cp=valid_student_cp,
                    target_cp=valid_target_cp,
                    teacher_cp=valid_teacher_cp,
                    student_distill=student_features["distill"],
                    teacher_distill=teacher_features["distill"],
                    weights=valid_weights,
                    response_alpha=response_alpha,
                    feature_alpha=feature_alpha,
                    student_point_distill=valid_student_point_distill,
                    teacher_point_distill=valid_teacher_point_distill,
                    point_alpha=point_alpha,
                ) * sample_scale
            if training:
                accum_index += 1
                is_last_batch = batch_index_in_epoch == num_batches - 1
                # Number of batches feeding the accumulation group the
                # current batch belongs to -- grad_accum_steps for every full
                # group, but the trailing group is smaller when num_batches
                # isn't a multiple of grad_accum_steps (e.g. 300 samples,
                # grad_accum_steps=4 -> last group has 300%4=0 -> full groups
                # only, but 302 samples would leave a 2-batch last group).
                # Dividing by this rather than always grad_accum_steps keeps
                # that trailing group's effective LR the same as a full one,
                # instead of silently under-weighting it.
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
        pred_blocks.append(valid_student_cp.detach().cpu().numpy())
        target_blocks.append(valid_target_cp.detach().cpu().numpy())

        if collect_predictions:
            pred_raw = normalizers.cp.inverse_transform(student_cp.detach().cpu().numpy())
            target_raw = normalizers.cp.inverse_transform(cp_t.detach().cpu().numpy())
            point_ids_batch = _batch_to_numpy(batch["point_ids"]).astype(np.int64, copy=False)
            coords_batch = points_np
            for batch_index, sample_id in enumerate(batch["sample_id"]):
                sample_mask = mask_np[batch_index]
                point_ids = point_ids_batch[batch_index][sample_mask]
                coords = coords_batch[batch_index][sample_mask]
                pred_sample = pred_raw[batch_index, sample_mask, 0]
                truth_sample = target_raw[batch_index, sample_mask, 0]
                abs_err = np.abs(pred_sample - truth_sample)
                sample_truth[sample_id].extend(truth_sample.tolist())
                sample_pred[sample_id].extend(pred_sample.tolist())
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
                        }
                    )

    pred_std = np.concatenate(pred_blocks, axis=0) if pred_blocks else np.zeros((0, 1), dtype=np.float32)
    target_std = np.concatenate(target_blocks, axis=0) if target_blocks else np.zeros((0, 1), dtype=np.float32)
    return {
        "loss": total_loss / max(total_count, 1),
        "pred_std": pred_std,
        "target_std": target_std,
        "rows": rows,
        "sample_metrics": {
            sample_id: _pressure_metrics(np.asarray(sample_truth[sample_id]), np.asarray(sample_pred[sample_id]))
            for sample_id in sample_truth
        },
    }


def _train_one_stage(
    model: PressureKDRegressor,
    train_loader,
    val_loader,
    device: torch.device,
    normalizers: KDNormalizerBundle,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    stage_name: str,
    output_dir: Path,
    teacher_model: PressureKDRegressor | None = None,
    teacher_normalizers: KDNormalizerBundle | None = None,
    response_alpha: float = 1.0,
    feature_alpha: float = 0.35,
    hard_point_alpha: float = 1.5,
    hard_point_fraction: float = 0.1,
    target_peak_alpha: float = 0.35,
    target_peak_fraction: float = 0.08,
    mach_focus_alpha: float = 0.75,
    mach_focus_start: float = 3.2,
    mach_focus_end: float = 4.5,
    aoa_focus_alpha: float = 0.0,
    aoa_focus_start: float = 6.0,
    aoa_focus_end: float = 12.0,
    low_signal_alpha: float = 0.35,
    low_signal_cap: float = 1.75,
    low_signal_ref: float | None = None,
    kd_warmup_epochs: int = 20,
    kd_ramp_epochs: int = 40,
    point_alpha: float = 0.0,
    point_jitter_std: float = 0.0,
    grad_accum_steps: int = 1,
    lr_schedule: str = "plateau",
    lr_warmup_epochs: int = 0,
    early_stop_patience: int = 40,
    early_stop_min_epochs: int = 60,
    select_by_rmse: bool = False,
) -> tuple[Path, Path, Path]:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_schedule not in {"plateau", "cosine_warmup"}:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}")
    # "plateau" (default) preserves every existing run's behaviour exactly.
    # "cosine_warmup" is the fixed-schedule alternative tested in 开发日志
    # .md 38 to see whether a known decay curve (vs. reactive plateau
    # detection) helps close the remaining R2 gap to 0.95.
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12)
        if lr_schedule == "plateau"
        else LambdaLR(optimizer, lr_lambda=lambda epoch: _cosine_warmup_lr_factor(epoch + 1, epochs, lr_warmup_epochs))
    )
    best_val_loss = float("inf")
    best_selection_metric = float("inf")
    no_improve_epochs = 0
    best_checkpoint_path = output_dir / f"{stage_name}_best.pt"
    last_checkpoint_path = output_dir / f"{stage_name}_last.pt"
    history_path = output_dir / f"{stage_name}_history.csv"
    if history_path.exists():
        history_path.unlink()
    print(
        f"[{stage_name}] start | epochs={epochs} | lr={lr:.3e} | weight_decay={weight_decay:.3e}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        response_alpha_epoch = response_alpha
        feature_alpha_epoch = feature_alpha
        if teacher_model is not None:
            response_alpha_epoch = ramp_weight(epoch, kd_warmup_epochs, kd_ramp_epochs, response_alpha)
            feature_alpha_epoch = ramp_weight(epoch, kd_warmup_epochs, kd_ramp_epochs, feature_alpha)
        train_result = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            normalizers=normalizers,
            grad_clip_norm=grad_clip_norm,
            teacher_model=teacher_model,
            teacher_normalizers=teacher_normalizers,
            response_alpha=response_alpha_epoch,
            feature_alpha=feature_alpha_epoch,
            hard_point_alpha=hard_point_alpha,
            hard_point_fraction=hard_point_fraction,
            target_peak_alpha=target_peak_alpha,
            target_peak_fraction=target_peak_fraction,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            aoa_focus_alpha=aoa_focus_alpha,
            aoa_focus_start=aoa_focus_start,
            aoa_focus_end=aoa_focus_end,
            low_signal_alpha=low_signal_alpha,
            low_signal_cap=low_signal_cap,
            low_signal_ref=low_signal_ref,
            point_alpha=point_alpha,
            point_jitter_std=point_jitter_std,
            grad_accum_steps=grad_accum_steps,
        )
        val_result = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            normalizers=normalizers,
            grad_clip_norm=0.0,
            teacher_model=teacher_model,
            teacher_normalizers=teacher_normalizers,
            response_alpha=response_alpha_epoch,
            feature_alpha=feature_alpha_epoch,
            hard_point_alpha=hard_point_alpha,
            hard_point_fraction=hard_point_fraction,
            target_peak_alpha=target_peak_alpha,
            target_peak_fraction=target_peak_fraction,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            aoa_focus_alpha=aoa_focus_alpha,
            aoa_focus_start=aoa_focus_start,
            aoa_focus_end=aoa_focus_end,
            low_signal_alpha=low_signal_alpha,
            low_signal_cap=low_signal_cap,
            low_signal_ref=low_signal_ref,
            point_alpha=point_alpha,
        )
        val_pred = normalizers.cp.inverse_transform(val_result["pred_std"])
        val_truth = normalizers.cp.inverse_transform(val_result["target_std"])
        val_metrics = _pressure_metrics(val_truth, val_pred)
        val_loss = float(val_result["loss"])
        val_metrics["loss_std"] = val_loss

        selection_metric_name, selection_metric = _validation_selection_metric(
            select_by_rmse=select_by_rmse,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        if lr_schedule == "plateau":
            scheduler.step(selection_metric)
        else:
            scheduler.step()
        best_val_loss = min(best_val_loss, val_loss)

        is_best = selection_metric < best_selection_metric
        if is_best:
            best_selection_metric = selection_metric
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "selection_metric_name": selection_metric_name,
                    "best_selection_metric": best_selection_metric,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                best_checkpoint_path,
            )
        else:
            no_improve_epochs += 1

        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "selection_metric_name": selection_metric_name,
                "best_selection_metric": best_selection_metric,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            last_checkpoint_path,
        )

        history_row = {
            "epoch": epoch,
            "train_loss_std": float(train_result["loss"]),
            "val_loss_std": float(val_result["loss"]),
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_nmae": float(val_metrics["nmae"]),
            "val_nrmse": float(val_metrics["nrmse"]),
            "selection_metric_name": selection_metric_name,
            "selection_metric": float(selection_metric),
            "best_selection_metric": float(best_selection_metric),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "best_val_loss_std": float(best_val_loss),
        }
        _write_history_row(history_path, history_row, is_new_file=not history_path.exists())
        print(
            f"[{stage_name}] epoch {epoch:03d}/{epochs} | "
            f"train_loss_std={train_result['loss']:.6f} | "
            f"val_loss_std={val_result['loss']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f} | "
            f"val_rmse={val_metrics['rmse']:.6f} | "
            f"{selection_metric_name}={selection_metric:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"best_selection_metric={best_selection_metric:.6f} | "
            f"best_val_loss_std={best_val_loss:.6f}"
            + (" | best" if is_best else ""),
            flush=True,
        )

        if epoch >= early_stop_min_epochs and no_improve_epochs >= early_stop_patience:
            print(f"[{stage_name}] early stop at epoch {epoch}", flush=True)
            break

    print(
        f"[{stage_name}] done | best_selection_metric={best_selection_metric:.6f} | "
        f"best_val_loss_std={best_val_loss:.6f}",
        flush=True,
    )
    return best_checkpoint_path, last_checkpoint_path, history_path


def _load_student_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[PressureKDRegressor, KDTrainingConfig, dict[str, object], KDNormalizerBundle]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    config = KDTrainingConfig.from_dict(checkpoint["config"])
    model = PressureKDRegressor(
        point_channels=list(config.student_point_channels),
        condition_hidden=list(config.student_condition_hidden),
        geometry_hidden=list(config.student_geometry_hidden),
        decoder_hidden=list(config.student_decoder_hidden),
        dropout=config.student_dropout,
        fourier_bands=config.fourier_bands,
        use_residual_decoder=config.use_residual_decoder,
        residual_hidden=list(config.residual_hidden),
        distill_dim=256,
        knn_k=config.knn_k,
        point_feat_distill_channels=config.point_feat_distill_channels if config.point_alpha > 0 else None,
        use_edge_conv=config.use_edge_conv,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizers = KDNormalizerBundle.from_state_dict(checkpoint["normalizers"])
    return model, config, checkpoint["manifest"], normalizers


def train_kd_pressure(
    data_root: str | Path,
    output_dir: str | Path,
    g2_manifest: str | Path | None = None,
    g4_train_manifest: str | Path | None = None,
    g4_test_manifest: str | Path | None = None,
    num_points: int | None = None,
    batch_size: int = 1,
    teacher_epochs: int = 120,
    student_pretrain_epochs: int = 60,
    student_pretrain_fidelity: str = "g2",
    student_epochs: int = 180,
    teacher_lr: float = 8e-4,
    student_lr: float = 8e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_fraction: float = 0.1,
    student_val_fraction: float | None = None,
    device: str = "",
    grad_clip_norm: float = 1.0,
    target_column: str = DEFAULT_TARGET_COLUMN,
    dropout: float = 0.08,
    student_dropout: float | None = None,
    student_weight_decay: float | None = None,
    fourier_bands: int = 4,
    use_residual_decoder: bool = True,
    knn_k: int = 16,
    response_alpha: float = 1.0,
    feature_alpha: float = 0.35,
    hard_point_alpha: float = 1.5,
    hard_point_fraction: float = 0.1,
    target_peak_alpha: float = 0.35,
    target_peak_fraction: float = 0.08,
    mach_focus_alpha: float = 0.75,
    mach_focus_start: float = 3.2,
    mach_focus_end: float = 4.5,
    aoa_focus_alpha: float = 0.0,
    aoa_focus_start: float = 6.0,
    aoa_focus_end: float = 12.0,
    low_signal_alpha: float = 0.35,
    low_signal_cap: float = 1.75,
    kd_warmup_epochs: int = 20,
    kd_ramp_epochs: int = 40,
    point_alpha: float = 0.0,
    point_feat_distill_channels: int | None = None,
    grad_accum_steps: int = 1,
    lr_schedule: str = "plateau",
    lr_warmup_epochs: int = 0,
    use_edge_conv: bool = True,
    early_stop_patience: int = 40,
    early_stop_min_epochs: int = 60,
) -> KDTrainingResult:
    seed_everything(seed)
    # Diagnosed separately from teacher: G2's teacher train/val gap measured
    # only ~10% (no capacity problem), while the G4 student showed a 34-39%
    # train/test RMSE gap that a short student_pretrain-only sweep traced to
    # regularization strength, not sample count (raising dropout/weight_decay
    # closed the gap; adding G4 samples 200->300 did not). Defaulting these
    # to the shared dropout/weight_decay keeps existing callers' behaviour
    # unchanged when they don't pass student-specific values.
    resolved_student_dropout = dropout if student_dropout is None else student_dropout
    resolved_student_weight_decay = weight_decay if student_weight_decay is None else student_weight_decay
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    g2_root = data_root / "G2"
    g4_root = data_root / "G4"

    sample_lists = build_kd_sample_lists(
        data_root,
        g2_manifest=g2_manifest,
        g4_train_manifest=g4_train_manifest,
        g4_test_manifest=g4_test_manifest,
    )
    # G4 gets its own val_fraction: at val_fraction=0.1 a 300-sample G4 train
    # pool leaves only 30 validation samples, and checkpoint selection over
    # 300 KD epochs (select_by_rmse=True) was found to overfit to that small
    # set -- the student_pretrain checkpoint (before KD fine-tuning) measured
    # an 11.8% train/test RMSE gap, but the KD-finetuned checkpoint chosen by
    # best val_rmse over those same 30 samples widened it back to 36.0% (see
    # 开发日志.md 34-36). G2's 800-sample pool at the shared val_fraction=0.1
    # already gives 80 validation samples, which showed no equivalent
    # instability, so only G4 needs a larger fraction.
    resolved_student_val_fraction = val_fraction if student_val_fraction is None else student_val_fraction
    g2_train_ids, g2_val_ids = stable_train_val_split(sample_lists["g2_train_ids"], val_fraction=val_fraction, seed=seed)
    g4_train_ids, g4_val_ids = stable_train_val_split(sample_lists["g4_train_ids"], val_fraction=resolved_student_val_fraction, seed=seed + 1)
    g4_test_ids = sample_lists["g4_test_ids"]
    print(
        "[kd] split | "
        f"g2_train={len(g2_train_ids)} | g2_val={len(g2_val_ids)} | "
        f"g4_train={len(g4_train_ids)} | g4_val={len(g4_val_ids)} | g4_test={len(g4_test_ids)}",
        flush=True,
    )

    def _source_label(manifest: str | Path | None) -> str:
        return "manifest" if manifest is not None else "clean filter"

    print(
        "[kd] g2 source | "
        f"{_source_label(g2_manifest)} -> g2_train={len(g2_train_ids)}/{len(sample_lists['g2_train_all_ids'])}"
        + (f" | manifest={Path(g2_manifest).name}" if g2_manifest is not None else ""),
        flush=True,
    )
    print(
        "[kd] g4 source | "
        f"train: {_source_label(g4_train_manifest)} -> "
        f"g4_train={len(g4_train_ids)}/{len(sample_lists['g4_train_all_ids'])}"
        + (f" | manifest={Path(g4_train_manifest).name}" if g4_train_manifest is not None else "")
        + " | "
        f"test: {_source_label(g4_test_manifest)} -> "
        f"g4_test={len(g4_test_ids)}/{len(sample_lists['g4_test_all_ids'])}"
        + (f" | manifest={Path(g4_test_manifest).name}" if g4_test_manifest is not None else ""),
        flush=True,
    )
    g2_signal_ref = _estimate_target_strength_reference(g2_root, g2_train_ids, target_column)
    g4_signal_ref = _estimate_target_strength_reference(g4_root, g4_train_ids, target_column)
    _validate_signal_reference("G2", g2_signal_ref)
    _validate_signal_reference("G4", g4_signal_ref)
    print(
        f"[kd] signal reference | g2_mean_abs_cp={g2_signal_ref:.6f} | "
        f"g4_mean_abs_cp={g4_signal_ref:.6f}",
        flush=True,
    )

    config = KDTrainingConfig(
        data_root=str(data_root),
        output_dir=str(output_dir),
        num_points=num_points,
        batch_size=batch_size,
        teacher_epochs=teacher_epochs,
        student_pretrain_epochs=student_pretrain_epochs,
        student_pretrain_fidelity=student_pretrain_fidelity,
        student_epochs=student_epochs,
        teacher_lr=teacher_lr,
        student_lr=student_lr,
        weight_decay=weight_decay,
        seed=seed,
        val_fraction=val_fraction,
        student_val_fraction=resolved_student_val_fraction,
        target_column=target_column,
        device=device,
        grad_clip_norm=grad_clip_norm,
        dropout=dropout,
        student_dropout=resolved_student_dropout,
        student_weight_decay=resolved_student_weight_decay,
        fourier_bands=fourier_bands,
        use_residual_decoder=use_residual_decoder,
        use_edge_conv=use_edge_conv,
        knn_k=knn_k,
        response_alpha=response_alpha,
        feature_alpha=feature_alpha,
        hard_point_alpha=hard_point_alpha,
        hard_point_fraction=hard_point_fraction,
        target_peak_alpha=target_peak_alpha,
        target_peak_fraction=target_peak_fraction,
        mach_focus_alpha=mach_focus_alpha,
        mach_focus_start=mach_focus_start,
        mach_focus_end=mach_focus_end,
        aoa_focus_alpha=aoa_focus_alpha,
        aoa_focus_start=aoa_focus_start,
        aoa_focus_end=aoa_focus_end,
        low_signal_alpha=low_signal_alpha,
        low_signal_cap=low_signal_cap,
        kd_warmup_epochs=kd_warmup_epochs,
        kd_ramp_epochs=kd_ramp_epochs,
        point_alpha=point_alpha,
        point_feat_distill_channels=point_feat_distill_channels,
        grad_accum_steps=grad_accum_steps,
        lr_schedule=lr_schedule,
        lr_warmup_epochs=lr_warmup_epochs,
        early_stop_patience=early_stop_patience,
        early_stop_min_epochs=early_stop_min_epochs,
    )

    device_obj = _device_from_config(device)
    print(
        f"[kd] config | teacher_epochs={teacher_epochs} | "
        f"student_pretrain_epochs={student_pretrain_epochs} | "
        f"student_pretrain_fidelity={student_pretrain_fidelity} | student_epochs={student_epochs} | "
        f"teacher_lr={teacher_lr:.3e} | student_lr={student_lr:.3e} | "
        f"response_alpha={response_alpha} | feature_alpha={feature_alpha} | "
        f"hard_point_alpha={hard_point_alpha} | hard_point_fraction={hard_point_fraction} | "
        f"target_peak_alpha={target_peak_alpha} | target_peak_fraction={target_peak_fraction} | "
        f"mach_focus_alpha={mach_focus_alpha} | mach_focus_start={mach_focus_start} | "
        f"mach_focus_end={mach_focus_end} | aoa_focus_alpha={aoa_focus_alpha} | "
        f"aoa_focus_start={aoa_focus_start} | aoa_focus_end={aoa_focus_end} | "
        f"low_signal_alpha={low_signal_alpha} | "
        f"low_signal_cap={low_signal_cap} | "
        f"kd_warmup_epochs={kd_warmup_epochs} | kd_ramp_epochs={kd_ramp_epochs}",
        flush=True,
    )
    g2_normalizer_map = fit_fidelity_standardizers(
        fidelity_root=g2_root,
        sample_ids=g2_train_ids,
        num_points=num_points,
        seed=seed,
        target_column=target_column,
    )
    g2_normalizers = KDNormalizerBundle(
        points=g2_normalizer_map["points"],
        condition=g2_normalizer_map["condition"],
        cp=g2_normalizer_map["cp"],
    )
    g4_normalizer_map = fit_fidelity_standardizers(
        fidelity_root=g4_root,
        sample_ids=g4_train_ids,
        num_points=num_points,
        seed=seed,
        target_column=target_column,
    )
    g4_normalizers = KDNormalizerBundle(
        points=g4_normalizer_map["points"],
        condition=g4_normalizer_map["condition"],
        cp=g4_normalizer_map["cp"],
    )

    save_json(output_dir / "split_manifest.json", {
        "g2_train_ids": g2_train_ids,
        "g2_val_ids": g2_val_ids,
        "g4_train_ids": g4_train_ids,
        "g4_val_ids": g4_val_ids,
        "g4_test_ids": g4_test_ids,
        "g2_train_all_ids": sample_lists["g2_train_all_ids"],
        "g2_train_excluded_ids": sample_lists["g2_train_excluded_ids"],
        "g4_train_all_ids": sample_lists["g4_train_all_ids"],
        "g4_train_excluded_ids": sample_lists["g4_train_excluded_ids"],
        "g4_test_all_ids": sample_lists["g4_test_all_ids"],
        "g4_test_excluded_ids": sample_lists["g4_test_excluded_ids"],
        "clean_only": True,
    })
    save_json(
        output_dir / "normalizers.json",
        {
            "g2_teacher": g2_normalizers.to_state_dict(),
            "g4_student": g4_normalizers.to_state_dict(),
        },
    )
    save_json(output_dir / "config.json", config.to_dict())

    teacher_train_sampler = WeightedRandomSampler(
        weights=_build_mach_sampling_weights(
            fidelity_root=g2_root,
            sample_ids=g2_train_ids,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            aoa_focus_alpha=aoa_focus_alpha,
            aoa_focus_start=aoa_focus_start,
            aoa_focus_end=aoa_focus_end,
        ),
        num_samples=len(g2_train_ids),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    student_train_sampler = WeightedRandomSampler(
        weights=_build_mach_sampling_weights(
            fidelity_root=g4_root,
            sample_ids=g4_train_ids,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            aoa_focus_alpha=aoa_focus_alpha,
            aoa_focus_start=aoa_focus_start,
            aoa_focus_end=aoa_focus_end,
        ),
        num_samples=len(g4_train_ids),
        replacement=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    teacher_train_loader = build_fidelity_pressure_loader(
        fidelity_root=g2_root,
        sample_ids=g2_train_ids,
        num_points=num_points,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        sampler=teacher_train_sampler,
        target_column=target_column,
        sampling_mode="pressure_aware",
        knn_k=knn_k,
    )
    teacher_val_loader = build_fidelity_pressure_loader(
        fidelity_root=g2_root,
        sample_ids=g2_val_ids,
        num_points=num_points,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        target_column=target_column,
        sampling_mode="pressure_aware",
        knn_k=knn_k,
    )
    student_train_loader = build_fidelity_pressure_loader(
        fidelity_root=g4_root,
        sample_ids=g4_train_ids,
        num_points=num_points,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        sampler=student_train_sampler,
        target_column=target_column,
        sampling_mode="pressure_aware",
        knn_k=knn_k,
    )
    student_val_loader = build_fidelity_pressure_loader(
        fidelity_root=g4_root,
        sample_ids=g4_val_ids,
        num_points=num_points,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        target_column=target_column,
        sampling_mode="pressure_aware",
        knn_k=knn_k,
    )
    student_test_loader = build_fidelity_pressure_loader(
        fidelity_root=g4_root,
        sample_ids=g4_test_ids,
        num_points=num_points,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        target_column=target_column,
        sampling_mode="pressure_aware",
        knn_k=knn_k,
    )

    teacher = build_teacher_model(knn_k=knn_k, use_edge_conv=use_edge_conv).to(device_obj)
    print("[kd] training teacher", flush=True)
    teacher_best_checkpoint, teacher_last_checkpoint, teacher_history_csv = _train_one_stage(
        model=teacher,
        train_loader=teacher_train_loader,
        val_loader=teacher_val_loader,
        device=device_obj,
        normalizers=g2_normalizers,
        epochs=teacher_epochs,
        lr=teacher_lr,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        stage_name="teacher",
        output_dir=output_dir,
        teacher_model=None,
        response_alpha=response_alpha,
        feature_alpha=feature_alpha,
        mach_focus_alpha=mach_focus_alpha,
        mach_focus_start=mach_focus_start,
        mach_focus_end=mach_focus_end,
        aoa_focus_alpha=aoa_focus_alpha,
        aoa_focus_start=aoa_focus_start,
        aoa_focus_end=aoa_focus_end,
        low_signal_alpha=low_signal_alpha,
        low_signal_cap=low_signal_cap,
        low_signal_ref=g2_signal_ref,
        early_stop_patience=early_stop_patience,
        early_stop_min_epochs=early_stop_min_epochs,
        select_by_rmse=False,
    )
    teacher_checkpoint = torch.load(teacher_best_checkpoint, map_location=device_obj)
    teacher_checkpoint.update(
        {
            "config": config.to_dict(),
            "manifest": {
                "g2_train_ids": g2_train_ids,
                "g2_val_ids": g2_val_ids,
                "g4_train_ids": g4_train_ids,
                "g4_val_ids": g4_val_ids,
                "g4_test_ids": g4_test_ids,
                "g2_low_signal_ref": g2_signal_ref,
                "g4_low_signal_ref": g4_signal_ref,
            },
            "normalizers": g2_normalizers.to_state_dict(),
        }
    )
    torch.save(teacher_checkpoint, teacher_best_checkpoint)

    teacher_checkpoint = torch.load(teacher_best_checkpoint, map_location=device_obj)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # point_feat_distill_channels only added when point_alpha>0 -- otherwise
    # this is a plain build_student_model() call with no extra unused Conv1d
    # layer, matching every existing checkpoint/config exactly.
    resolved_point_feat_distill_channels = point_feat_distill_channels if point_alpha > 0 else None
    student = build_student_model(
        knn_k=knn_k,
        dropout=resolved_student_dropout,
        point_feat_distill_channels=resolved_point_feat_distill_channels,
        use_edge_conv=use_edge_conv,
    ).to(device_obj)
    pretrain_best_checkpoint = None
    pretrain_last_checkpoint = None
    pretrain_history_csv = None
    if student_pretrain_fidelity not in ("g2", "g4"):
        raise ValueError(f"student_pretrain_fidelity must be 'g2' or 'g4', got {student_pretrain_fidelity!r}")
    if student_pretrain_epochs > 0:
        # "g2": warm-start the student on the same low-fidelity data the
        # teacher trains on, using the teacher's own epoch budget/LR (this is
        # what runs/kd_warmstart/20260916_230552 validated at R2=0.955,
        # matching G2_PRETRAIN_EPOCHS=teacher_epochs and G2_PRETRAIN_LR=
        # teacher_lr in experiment_kd_warmstart.py -- student_pretrain_epochs/
        # student_lr below are the "g4" defaults and don't apply here).
        # "g4": original random-init-on-G4 behaviour (R2=0.949 at these same
        # KD settings), kept for backward-compatible reruns/comparisons.
        if student_pretrain_fidelity == "g2":
            pretrain_train_loader, pretrain_val_loader = teacher_train_loader, teacher_val_loader
            pretrain_normalizers, pretrain_signal_ref = g2_normalizers, g2_signal_ref
            pretrain_epochs, pretrain_lr = teacher_epochs, teacher_lr
        else:
            pretrain_train_loader, pretrain_val_loader = student_train_loader, student_val_loader
            pretrain_normalizers, pretrain_signal_ref = g4_normalizers, g4_signal_ref
            pretrain_epochs, pretrain_lr = student_pretrain_epochs, student_lr
        print(f"[kd] pretraining student on {student_pretrain_fidelity.upper()} supervision", flush=True)
        pretrain_best_checkpoint, pretrain_last_checkpoint, pretrain_history_csv = _train_one_stage(
            model=student,
            train_loader=pretrain_train_loader,
            val_loader=pretrain_val_loader,
            device=device_obj,
            normalizers=pretrain_normalizers,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            weight_decay=resolved_student_weight_decay,
            grad_clip_norm=grad_clip_norm,
            stage_name="student_pretrain",
            output_dir=output_dir,
            teacher_model=None,
            response_alpha=0.0,
            feature_alpha=0.0,
            hard_point_alpha=hard_point_alpha,
            hard_point_fraction=hard_point_fraction,
            target_peak_alpha=target_peak_alpha,
            target_peak_fraction=target_peak_fraction,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            aoa_focus_alpha=aoa_focus_alpha,
            aoa_focus_start=aoa_focus_start,
            aoa_focus_end=aoa_focus_end,
            low_signal_alpha=low_signal_alpha,
            low_signal_cap=low_signal_cap,
            low_signal_ref=pretrain_signal_ref,
            grad_accum_steps=grad_accum_steps,
            lr_schedule=lr_schedule,
            lr_warmup_epochs=lr_warmup_epochs,
            early_stop_patience=early_stop_patience,
            early_stop_min_epochs=early_stop_min_epochs,
            # "g2" mirrors the teacher stage (select_by_rmse=False, i.e. best
            # standardized val loss) since this reuses the teacher's own
            # epoch budget/LR; "g4" keeps the original best-val-RMSE
            # selection for backward-compatible reruns.
            select_by_rmse=(student_pretrain_fidelity == "g4"),
        )
        pretrain_checkpoint = torch.load(pretrain_best_checkpoint, map_location=device_obj)
        student.load_state_dict(pretrain_checkpoint["model_state_dict"])

    # student_epochs=0 skips KD fine-tuning entirely and ships the
    # student_pretrain checkpoint as the final model. Two full runs
    # (20260910_134152, 20260910_224134) found KD fine-tuning under the
    # STUDENT_DROPOUT/STUDENT_WEIGHT_DECAY regularization sweep was a net
    # negative: student_pretrain_best.pt measured a healthy ~17% train/test
    # RMSE gap and 0.0323 test RMSE, but KD fine-tuning -- even isolated from
    # "just more training" via a no-KD control that ran the same 100 extra
    # epochs with teacher_model=None -- pushed the gap to 47.8% and the test
    # RMSE to 0.0287, worse than the no-KD control's own 0.0272 (see 开发日志
    # .md 34-37). The teacher (dropout=0.08, unregularized) and this student
    # (dropout=0.14) ended up in different-enough regimes that distilling
    # from one to the other stopped helping. teacher_model=None throughout
    # this branch mirrors student_pretrain's own supervised-only loss.
    run_kd_finetune = student_epochs > 0
    if run_kd_finetune:
        print("[kd] fine-tuning student with weak KD", flush=True)
        student_best_checkpoint, student_last_checkpoint, student_history_csv = _train_one_stage(
            model=student,
            train_loader=student_train_loader,
            val_loader=student_val_loader,
            device=device_obj,
            normalizers=g4_normalizers,
            epochs=student_epochs,
            lr=student_lr,
            weight_decay=resolved_student_weight_decay,
            grad_clip_norm=grad_clip_norm,
            stage_name="student",
            output_dir=output_dir,
            teacher_model=teacher,
            teacher_normalizers=g2_normalizers,
            response_alpha=response_alpha,
            feature_alpha=feature_alpha,
            hard_point_alpha=hard_point_alpha,
            hard_point_fraction=hard_point_fraction,
            target_peak_alpha=target_peak_alpha,
            target_peak_fraction=target_peak_fraction,
            mach_focus_alpha=mach_focus_alpha,
            mach_focus_start=mach_focus_start,
            mach_focus_end=mach_focus_end,
            aoa_focus_alpha=aoa_focus_alpha,
            aoa_focus_start=aoa_focus_start,
            aoa_focus_end=aoa_focus_end,
            low_signal_alpha=low_signal_alpha,
            low_signal_cap=low_signal_cap,
            low_signal_ref=g4_signal_ref,
            kd_warmup_epochs=kd_warmup_epochs,
            kd_ramp_epochs=kd_ramp_epochs,
            point_alpha=point_alpha,
            grad_accum_steps=grad_accum_steps,
            lr_schedule=lr_schedule,
            lr_warmup_epochs=lr_warmup_epochs,
            early_stop_patience=early_stop_patience,
            early_stop_min_epochs=early_stop_min_epochs,
            select_by_rmse=True,
        )
        student_checkpoint = torch.load(student_best_checkpoint, map_location=device_obj)
        student.load_state_dict(student_checkpoint["model_state_dict"])
    else:
        if pretrain_best_checkpoint is None:
            raise ValueError("student_epochs=0 requires student_pretrain_epochs > 0")
        print("[kd] student_epochs=0: skipping KD fine-tuning, shipping student_pretrain as the final model", flush=True)
        student_best_checkpoint = output_dir / "student_best.pt"
        student_last_checkpoint = output_dir / "student_last.pt"
        student_history_csv = pretrain_history_csv
        student_checkpoint = torch.load(pretrain_best_checkpoint, map_location=device_obj)
    student.eval()

    student_result = _run_epoch(
        model=student,
        loader=student_test_loader,
        device=device_obj,
        optimizer=None,
        normalizers=g4_normalizers,
        grad_clip_norm=0.0,
        teacher_model=teacher if run_kd_finetune else None,
        teacher_normalizers=g2_normalizers if run_kd_finetune else None,
        response_alpha=response_alpha,
        feature_alpha=feature_alpha,
        hard_point_alpha=hard_point_alpha,
        hard_point_fraction=hard_point_fraction,
        target_peak_alpha=target_peak_alpha,
        target_peak_fraction=target_peak_fraction,
        mach_focus_alpha=mach_focus_alpha,
        mach_focus_start=mach_focus_start,
        mach_focus_end=mach_focus_end,
        aoa_focus_alpha=aoa_focus_alpha,
        aoa_focus_start=aoa_focus_start,
        aoa_focus_end=aoa_focus_end,
        low_signal_alpha=low_signal_alpha,
        low_signal_cap=low_signal_cap,
        low_signal_ref=g4_signal_ref,
        point_alpha=point_alpha,
        collect_predictions=True,
    )
    student_pred = g4_normalizers.cp.inverse_transform(student_result["pred_std"])
    student_truth = g4_normalizers.cp.inverse_transform(student_result["target_std"])
    student_metrics = _pressure_metrics(student_truth, student_pred)
    student_metrics["loss_std"] = float(student_result["loss"])
    print(
        f"[kd] student test | loss_std={student_result['loss']:.6f} | "
        f"mae={student_metrics['mae']:.6f} | rmse={student_metrics['rmse']:.6f}",
        flush=True,
    )

    teacher_metrics_json = output_dir / "teacher_metrics.json"
    student_metrics_json = output_dir / "student_metrics.json"
    student_predictions_path = output_dir / "student_predictions.csv"
    student_sample_metrics_path = output_dir / "student_sample_metrics.csv"
    save_json(teacher_metrics_json, {"train_history": str(teacher_history_csv), "best_checkpoint": str(teacher_best_checkpoint)})
    save_json(student_metrics_json, student_metrics)
    _save_predictions_csv(student_predictions_path, student_result["rows"])
    _save_sample_metrics_csv(student_sample_metrics_path, student_result["sample_metrics"])

    student_checkpoint.update(
        {
            "config": config.to_dict(),
            "manifest": {
                "g2_train_ids": g2_train_ids,
                "g2_val_ids": g2_val_ids,
                "g4_train_ids": g4_train_ids,
                "g4_val_ids": g4_val_ids,
                "g4_test_ids": g4_test_ids,
                "student_test_ids": g4_test_ids,
                "g2_low_signal_ref": g2_signal_ref,
                "g4_low_signal_ref": g4_signal_ref,
            },
            "normalizers": g4_normalizers.to_state_dict(),
        }
    )
    torch.save(student_checkpoint, student_best_checkpoint)
    torch.save(student_checkpoint, student_last_checkpoint)

    save_json(
        output_dir / "kd_manifest.json",
        {
            "teacher_best_checkpoint": str(teacher_best_checkpoint),
            "teacher_last_checkpoint": str(teacher_last_checkpoint),
            "student_best_checkpoint": str(student_best_checkpoint),
            "student_last_checkpoint": str(student_last_checkpoint),
            "teacher_history_csv": str(teacher_history_csv),
            "student_pretrain_best_checkpoint": str(pretrain_best_checkpoint) if pretrain_best_checkpoint else None,
            "student_pretrain_last_checkpoint": str(pretrain_last_checkpoint) if pretrain_last_checkpoint else None,
            "student_pretrain_history_csv": str(pretrain_history_csv) if pretrain_history_csv else None,
            "student_history_csv": str(student_history_csv),
            "teacher_train_ids": g2_train_ids,
            "teacher_val_ids": g2_val_ids,
            "student_train_ids": g4_train_ids,
            "student_val_ids": g4_val_ids,
            "student_test_ids": g4_test_ids,
            "g2_low_signal_ref": g2_signal_ref,
            "g4_low_signal_ref": g4_signal_ref,
        },
    )

    return KDTrainingResult(
        output_dir=output_dir,
        teacher_best_checkpoint=teacher_best_checkpoint,
        student_best_checkpoint=student_best_checkpoint,
        teacher_metrics_json=teacher_metrics_json,
        student_metrics_json=student_metrics_json,
        student_predictions_csv=student_predictions_path,
        student_sample_metrics_csv=student_sample_metrics_path,
        split_manifest_json=output_dir / "split_manifest.json",
        normalizers_json=output_dir / "normalizers.json",
    )


def evaluate_kd_pressure(
    data_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 1,
    num_points: int | None = None,
    seed: int = 42,
    device: str = "",
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> dict[str, object]:
    seed_everything(seed)
    data_root = Path(data_root)
    checkpoint_path = Path(checkpoint_path)
    model, config, manifest, normalizers = _load_student_checkpoint(checkpoint_path, map_location="cpu")

    g4_root = data_root / "G4"
    g4_test_ids = manifest.get("student_test_ids") or list_fidelity_sample_ids(g4_root, split="test", clean_only=True)
    test_loader = build_fidelity_pressure_loader(
        fidelity_root=g4_root,
        sample_ids=g4_test_ids,
        num_points=num_points,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        target_column=target_column,
        sampling_mode="pressure_aware",
        knn_k=config.knn_k,
    )

    device_obj = _device_from_config(device)
    model = model.to(device_obj)
    result = _run_epoch(
        model=model,
        loader=test_loader,
        device=device_obj,
        optimizer=None,
        normalizers=normalizers,
        grad_clip_norm=0.0,
        teacher_model=None,
        response_alpha=config.response_alpha,
        feature_alpha=config.feature_alpha,
        hard_point_alpha=config.hard_point_alpha,
        hard_point_fraction=config.hard_point_fraction,
        target_peak_alpha=config.target_peak_alpha,
        target_peak_fraction=config.target_peak_fraction,
        mach_focus_alpha=config.mach_focus_alpha,
        mach_focus_start=config.mach_focus_start,
        mach_focus_end=config.mach_focus_end,
        aoa_focus_alpha=config.aoa_focus_alpha,
        aoa_focus_start=config.aoa_focus_start,
        aoa_focus_end=config.aoa_focus_end,
        low_signal_alpha=config.low_signal_alpha,
        low_signal_cap=config.low_signal_cap,
        low_signal_ref=float(manifest.get("g4_low_signal_ref", 1.0)),
        collect_predictions=True,
    )
    pred = normalizers.cp.inverse_transform(result["pred_std"])
    truth = normalizers.cp.inverse_transform(result["target_std"])
    metrics = _pressure_metrics(truth, pred)
    metrics["loss_std"] = float(result["loss"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "student_metrics.json"
    predictions_path = output_dir / "student_predictions.csv"
    sample_metrics_path = output_dir / "student_sample_metrics.csv"
    save_json(metrics_path, metrics)
    _save_predictions_csv(predictions_path, result["rows"])
    _save_sample_metrics_csv(sample_metrics_path, result["sample_metrics"])
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "sample_metrics_path": sample_metrics_path,
        "num_test_samples": len(g4_test_ids),
        "manifest": manifest,
    }
