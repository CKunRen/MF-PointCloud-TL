from __future__ import annotations

import csv
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_TARGET_KEYS: tuple[str, ...] = ("CFx", "CFy", "CFz", "CMx", "CMy", "CMz")


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def locate_stage1_results_dir(root: str | Path) -> Path:
    base = Path(root)
    candidates = [base]
    candidates.extend(
        [
            base / "parametric_missile_cfd" / "datasets" / "stage1_120" / "results",
            base / "datasets" / "stage1_120" / "results",
            base / "stage1_120" / "results",
        ]
    )
    if base.parent != base:
        candidates.extend(
            [
                base.parent,
                base.parent / "parametric_missile_cfd" / "datasets" / "stage1_120" / "results",
                base.parent / "datasets" / "stage1_120" / "results",
                base.parent / "stage1_120" / "results",
            ]
        )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*_surface_flow.csv")):
            return candidate
    raise FileNotFoundError(
        f"Could not locate stage1 results directory from {base!s}. "
        "Expected a folder containing *_surface_flow.csv files."
    )


def list_sample_ids(results_dir: str | Path) -> list[str]:
    results_dir = Path(results_dir)
    suffix = "_surface_flow.csv"
    ids = sorted(
        path.name[: -len(suffix)] if path.name.endswith(suffix) else path.name
        for path in results_dir.glob("*_surface_flow.csv")
    )
    if not ids:
        raise FileNotFoundError(f"No stage1 surface files found in {results_dir!s}")
    return ids


def load_stage1_samples_table(root: str | Path) -> Path | None:
    base = Path(root)
    candidates = [
        base / "parametric_missile_cfd" / "datasets" / "stage1_120" / "samples.csv",
        base / "datasets" / "stage1_120" / "samples.csv",
        base / "stage1_120" / "samples.csv",
    ]
    if base.parent != base:
        candidates.extend(
            [
                base.parent / "parametric_missile_cfd" / "datasets" / "stage1_120" / "samples.csv",
                base.parent / "datasets" / "stage1_120" / "samples.csv",
                base.parent / "stage1_120" / "samples.csv",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def stable_sample_seed(sample_id: str, seed: int) -> int:
    return int((seed + zlib.crc32(sample_id.encode("utf-8"))) % (2**32 - 1))


def _to_float(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if value is None:
        raise ValueError("Cannot convert None to float")
    return float(str(value))


def sample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    total = int(points.shape[0])
    if total == 0:
        raise ValueError("Point cloud is empty")
    if total == num_points:
        return np.asarray(points, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    replace = total < num_points
    indices = rng.choice(total, size=num_points, replace=replace)
    return np.asarray(points, dtype=np.float32)[indices]


def load_stage1_raw_record(
    sample_id: str,
    results_dir: str | Path,
    num_points: int,
    seed: int,
    target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
) -> dict[str, object]:
    results_dir = Path(results_dir)
    surface_path = results_dir / f"{sample_id}_surface_flow.csv"
    params_path = results_dir / f"{sample_id}_parameters.json"
    summary_path = results_dir / f"{sample_id}_loads_summary.csv"
    if not surface_path.exists():
        raise FileNotFoundError(surface_path)
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    points_df = pd.read_csv(surface_path, usecols=["x", "y", "z"])
    points = points_df.to_numpy(dtype=np.float32)
    sampled_points = sample_points(points, num_points=num_points, seed=seed)

    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    condition = np.array([_to_float(params["mach"]), _to_float(params["aoa_deg"])], dtype=np.float32)

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError(f"Empty loads summary: {summary_path}")
    summary_row = summary_df.iloc[0].to_dict()
    raw_json = json.loads(summary_row["history_last_row_json"])
    target = np.array([_to_float(raw_json[key]) for key in target_keys], dtype=np.float32)
    aux = {
        "CL": _to_float(raw_json.get("CL", summary_row.get("history_last_cl"))),
        "CD": _to_float(raw_json.get("CD", summary_row.get("history_last_cd"))),
    }
    return {
        "sample_id": sample_id,
        "points": sampled_points,
        "condition": condition,
        "target": target,
        "raw_summary": raw_json,
        "aux": aux,
    }


def build_split_manifest(
    results_dir: str | Path,
    train_pool_size: int = 100,
    test_size: int = 20,
) -> dict[str, object]:
    sample_ids = list_sample_ids(results_dir)
    if len(sample_ids) < train_pool_size + test_size:
        raise ValueError(
            f"Need at least {train_pool_size + test_size} samples, found {len(sample_ids)}"
        )
    train_pool_ids = sample_ids[:train_pool_size]
    test_ids = sample_ids[-test_size:]
    return {
        "all_sample_ids": sample_ids,
        "train_pool_ids": train_pool_ids,
        "test_ids": test_ids,
        "train_pool_size": train_pool_size,
        "test_size": test_size,
    }


def build_split_manifest_from_samples(
    samples_csv: str | Path,
    train_pool_size: int = 100,
    test_size: int = 20,
) -> dict[str, object]:
    samples_csv = Path(samples_csv)
    with samples_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty samples table: {samples_csv!s}")

    all_sample_ids = [row["sample_id"] for row in rows]
    train_pool_ids = [row["sample_id"] for row in rows if row.get("split") == "train"]
    test_ids = [row["sample_id"] for row in rows if row.get("split") == "test"]

    if len(train_pool_ids) < train_pool_size or len(test_ids) < test_size:
        raise ValueError(
            f"Need at least {train_pool_size} train and {test_size} test samples in {samples_csv!s}, "
            f"found train={len(train_pool_ids)}, test={len(test_ids)}"
        )

    return {
        "all_sample_ids": all_sample_ids,
        "train_pool_ids": train_pool_ids,
        "test_ids": test_ids,
        "train_pool_size": len(train_pool_ids),
        "test_size": len(test_ids),
        "samples_csv": str(samples_csv),
    }


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return (arr - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return arr * self.std + self.mean

    def to_state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Sequence[float]]) -> "Standardizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32))


def save_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_train_val_split(sample_ids: Sequence[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    ids = list(sample_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    shuffled = [ids[i] for i in order]
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = shuffled[:val_count]
    train_ids = shuffled[val_count:]
    if not train_ids:
        raise ValueError("No training samples left after validation split")
    return train_ids, val_ids
