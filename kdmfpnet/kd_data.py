from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from .utils import Standardizer, stable_sample_seed
from .field_targets import (
    calibration_scale,
    estimate_surface_geometry,
    parse_reference_quantities,
    wall_velocity_nondimensional,
)

DEFAULT_TARGET_COLUMN = "Cp"
MAX_ABS_CONSERVATIVE_VALUE = 1.0e4
MAX_ABS_CP_VALUE = 20.0
# A handful of surface nodes sit on geometric singularities -- the sharp cone
# apex (r -> 0) and the fin-root junction -- where snappyHexMesh leaves sliver
# cells and the inviscid solution blows up locally: density reaches 170-341
# against a freestream of ~1, i.e. 30-60x the (gamma+1)/(gamma-1) ~ 6 ceiling an
# inviscid normal shock can produce. Those nodes are numerical artefacts, not
# flow features, and rejecting the whole sample because of them throws away
# otherwise physics-vetted data: of the 400 G4 samples in the frozen manifests,
# 6 carry 1-3 such nodes out of ~20k (fraction 4.6e-05 to 1.4e-04) while their
# 99.9th percentile |Cp| stays at a physical 0.8-1.5.
#
# Genuinely broken solutions are an order of magnitude away: sampled failed
# samples with more than a couple of bad nodes run 1.5e-03 to 3.3e-01 of the
# cloud, with max|Cp| 1e2-2e5. So gate on the *fraction* of bad nodes (the same
# reasoning as MAX_NON_PHYSICAL_POINTS below) and drop the survivors rather than
# clipping them -- clipping would feed a fabricated Cp to the network, and the
# CFD solution at those nodes carries no information to preserve.
MAX_UNPHYSICAL_CP_FRACTION = 5.0e-4
# The ceiling itself has to be the physical one, not a fixed number. Inviscid
# surface Cp cannot exceed the stagnation value behind a normal shock (Rayleigh
# pitot), which is Mach-dependent: 1.56 at Ma 1.5 rising to 1.81 at Ma 5. A flat
# |Cp| <= 20 gate lets a node sitting at 11x the stagnation limit through and
# still calls it physical -- of the 400 frozen G4 samples, 47 carry at least one
# node above 1.2x Cp_stag while their 99.9th percentile stays at 0.25-1.52.
#
# Allow some headroom over Cp_stag for the shock overshoot a second-order scheme
# genuinely produces (2.10 against a 1.81 ceiling, i.e. 1.16x, was measured on
# this geometry -- see 开发日志_超算CFD链路.md 2.3), then reject. Over all 400
# samples this drops 59 nodes out of 7.9e6 (7.5e-06), at most 4 in any one
# sample, and takes the G4 train max|Cp| from 19.93 to 2.62 while moving the
# normalizer std by 1%.
CP_STAGNATION_OVERSHOOT = 1.5

# SU2 prints a cumulative "non-physical points" tally in its shutdown summary,
# so the counter is non-zero whenever the startup transient briefly produced a
# bad state -- even when the converged solution is perfectly clean. Under the
# ROE+MUSCL scheme a converged Mach 4-5 case typically reports 5-10 points out
# of ~2e6, which is a transient artefact, while genuinely broken JST runs
# reported 1e2-1e6. Reject on the count, not on the presence of the string, and
# rely on the surface-field physics checks below as the real gate.
MAX_NON_PHYSICAL_POINTS = 100
_NON_PHYSICAL_RE = re.compile(r"there are (\d+) non-physical points")
_NON_PHYSICAL_RECONSTRUCTED_RE = re.compile(
    r"(\d+) reconstructed states for upwinding are non-physical"
)
# MUSCL reconstruction is limiter-clipped, so a handful of non-physical
# reconstructed states is normal and never enters the solution vector. Allow
# more headroom here than for solution points.
MAX_NON_PHYSICAL_RECONSTRUCTED = 2000


def _non_physical_counts(text: str) -> tuple[int, int]:
    """Return (max solution points, max reconstructed states) reported by SU2.

    Returns (-1, -1) when the warning string is present but no count could be
    parsed, so callers can treat an unrecognised format as unsafe.
    """
    points = [int(m) for m in _NON_PHYSICAL_RE.findall(text)]
    recon = [int(m) for m in _NON_PHYSICAL_RECONSTRUCTED_RE.findall(text)]
    if "non-physical points" in text and not points:
        return -1, max(recon, default=0)
    return max(points, default=0), max(recon, default=0)


def _sample_indices(total: int, num_points: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if total <= 0:
        raise ValueError("Point cloud is empty")
    rng = np.random.default_rng(seed)
    replace = total < num_points
    return rng.choice(total, size=num_points, replace=replace)


def _take_top_count(total: int, fraction: float, num_points: int) -> int:
    if fraction <= 0 or total <= 0:
        return 0
    return min(total, max(1, int(round(num_points * fraction))))


def _pressure_aware_sample_indices(
    points: np.ndarray,
    target: np.ndarray,
    num_points: int,
    seed: int,
    peak_fraction: float = 0.20,
    tail_fraction: float = 0.20,
    outer_fraction: float = 0.15,
) -> np.ndarray:
    total = int(points.shape[0])
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if total <= 0:
        raise ValueError("Point cloud is empty")

    rng = np.random.default_rng(seed)
    selected: list[int] = []

    def add(indices: np.ndarray, count: int) -> None:
        if count <= 0 or indices.size == 0:
            return
        chosen = indices[: min(count, indices.size)]
        selected.extend(int(i) for i in chosen)

    target_flat = np.asarray(target, dtype=np.float32).reshape(-1)
    peak_order = np.argsort(-np.abs(target_flat))
    add(peak_order, _take_top_count(total, peak_fraction, num_points))

    x = np.asarray(points[:, 0], dtype=np.float32)
    tail_order = np.argsort(x)
    add(tail_order, _take_top_count(total, tail_fraction, num_points))

    radius = np.linalg.norm(np.asarray(points[:, 1:3], dtype=np.float32), axis=1)
    outer_order = np.argsort(-radius)
    add(outer_order, _take_top_count(total, outer_fraction, num_points))

    unique_selected = list(dict.fromkeys(selected))
    if len(unique_selected) >= num_points:
        return np.asarray(unique_selected[:num_points], dtype=np.int64)

    selected_set = set(unique_selected)
    remaining = np.asarray([i for i in range(total) if i not in selected_set], dtype=np.int64)
    needed = num_points - len(unique_selected)
    if remaining.size > 0:
        replace = remaining.size < needed
        fill = rng.choice(remaining, size=needed, replace=replace)
    else:
        fill = rng.choice(np.arange(total, dtype=np.int64), size=needed, replace=True)
    return np.asarray(unique_selected + [int(i) for i in fill], dtype=np.int64)


def _sample_is_clean(sample_dir: Path) -> bool:
    status_path = sample_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if status.get("status") != "done":
        return False

    sample_id = sample_dir.name
    if not (
        (sample_dir / f"{sample_id}_history.csv").is_file()
        and (sample_dir / f"{sample_id}_surface_flow.csv").is_file()
        # parameters.json is required downstream for the Mach-focus sampling
        # weights and the Cp non-dimensionalisation; some G4 directories were
        # left with only a partial download and are missing it even though
        # status.json says "done" and the CSVs are present.
        and (sample_dir / "parameters.json").is_file()
    ):
        return False

    run_log_path = sample_dir / "run.log"
    if not run_log_path.is_file():
        return False
    text = run_log_path.read_text(encoding="utf-8", errors="ignore")
    if not (
        "All convergence criteria satisfied." in text
        and "Maximum number of iterations reached" not in text
        and "SU2 has diverged" not in text
        and "MPI_ABORT" not in text
        and "Exit Success (SU2_CFD)" in text
    ):
        return False

    non_physical_points, non_physical_recon = _non_physical_counts(text)
    if non_physical_points < 0 or non_physical_points > MAX_NON_PHYSICAL_POINTS:
        return False
    if non_physical_recon > MAX_NON_PHYSICAL_RECONSTRUCTED:
        return False

    # SU2 can emit an exit-success footer after a numerically unusable run.
    # Reject those pseudo-successes from the training pool.
    try:
        surface_path = sample_dir / f"{sample_id}_surface_flow.csv"
        values = pd.read_csv(
            surface_path,
            usecols=["Density", "Momentum_x", "Momentum_y", "Momentum_z", "Energy"],
        ).to_numpy(dtype=np.float64)
    except (OSError, ValueError, TypeError):
        return False
    if values.size == 0 or not np.isfinite(values).all():
        return False

    density = values[:, 0]
    momentum = values[:, 1:4]
    energy = values[:, 4]
    if np.any(density <= 0.0):
        return False
    if np.max(np.abs(values)) > MAX_ABS_CONSERVATIVE_VALUE:
        return False
    pressure = _pressure_from_conservative_vars(
        density=density,
        momentum_x=momentum[:, 0],
        momentum_y=momentum[:, 1],
        momentum_z=momentum[:, 2],
        energy=energy,
    )
    return bool(
        np.isfinite(pressure).all()
        and np.all(pressure > 0.0)
        and np.max(np.abs(pressure)) <= MAX_ABS_CONSERVATIVE_VALUE
    )


def list_fidelity_sample_ids(
    fidelity_root: str | Path,
    split: str | None = None,
    clean_only: bool = False,
) -> list[str]:
    fidelity_root = Path(fidelity_root)
    if not fidelity_root.is_dir():
        raise FileNotFoundError(fidelity_root)
    sample_ids = []
    for child in fidelity_root.iterdir():
        if not child.is_dir():
            continue
        parts = child.name.split("_")
        if split is not None and (len(parts) < 2 or parts[1] != split):
            continue
        surface_path = child / f"{child.name}_surface_flow.csv"
        if not surface_path.exists():
            continue
        if clean_only and not _sample_is_clean(child):
            continue
        sample_ids.append(child.name)
    return sorted(sample_ids)


def load_sample_id_manifest(manifest_path: str | Path) -> list[str]:
    """Load a fixed list of sample ids, one per line, ignoring blanks.

    Used for the G2 training set: `usable_g2_train_800.txt` is the output of
    check_validation.py's physics survey (convergence, orientation, CD
    magnitude, fin drag share) run over the combined original-800 + top-up-244
    submission, then trimmed to exactly 800 with a Mach-uniform selection. That
    is a stronger and more current filter than `_sample_is_clean` in this file,
    which only checks SU2's own exit status and conservative-variable sanity and
    knows nothing about orientation or drag magnitude. Reading the manifest
    means training uses exactly the samples that were vetted, rather than
    re-deriving a (weaker, possibly stale) clean set from the directory tree.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    ids = [sid for sid in ids if sid]
    if not ids:
        raise ValueError(f"Manifest is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Manifest contains duplicate ids: {path}")
    return sorted(ids)


def _parse_freestream_pressure(run_log_path: Path, fallback: float = 1.0) -> float:
    if not run_log_path.exists():
        return float(fallback)
    for line in run_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Static Pressure" not in line:
            continue
        values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if values:
            # SU2's table contains dimensional pressure first and the
            # nondimensional pressure in the final numeric column.
            return float(values[-1])
    return float(fallback)


def _pressure_from_conservative_vars(
    density: np.ndarray,
    momentum_x: np.ndarray,
    momentum_y: np.ndarray,
    momentum_z: np.ndarray,
    energy: np.ndarray,
    gamma: float = 1.4,
) -> np.ndarray:
    rho = np.asarray(density, dtype=np.float64)
    mom_x = np.asarray(momentum_x, dtype=np.float64)
    mom_y = np.asarray(momentum_y, dtype=np.float64)
    mom_z = np.asarray(momentum_z, dtype=np.float64)
    total_energy = np.asarray(energy, dtype=np.float64)
    kinetic = 0.5 * (mom_x * mom_x + mom_y * mom_y + mom_z * mom_z) / np.clip(rho, 1e-12, None)
    pressure = (gamma - 1.0) * (total_energy - kinetic)
    return pressure.astype(np.float32)


def stagnation_cp(mach: float, gamma: float = 1.4) -> float:
    """Cp at a stagnation point behind a normal shock (Rayleigh pitot formula).

    This is the largest Cp inviscid flow can produce on the surface, and it
    depends on Mach: 1.56 at Ma 1.5, 1.77 at Ma 3, 1.81 at Ma 5. Below Ma 1 the
    shock term does not apply, so fall back to the isentropic expression.
    """
    mach = float(mach)
    if mach <= 0.0:
        raise ValueError("Mach must be positive")
    if mach < 1.0:
        ratio = (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (gamma / (gamma - 1.0))
    else:
        ratio = ((gamma + 1.0) ** 2 * mach**2 / (4.0 * gamma * mach**2 - 2.0 * (gamma - 1.0))) ** (
            gamma / (gamma - 1.0)
        ) * ((1.0 - gamma + 2.0 * gamma * mach**2) / (gamma + 1.0))
    return 2.0 / (gamma * mach**2) * (ratio - 1.0)


def _cp_from_pressure(pressure: np.ndarray, mach: float, p_inf: float, gamma: float = 1.4) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.float32)
    q_inf = 0.5 * gamma * float(p_inf) * float(mach) ** 2
    if abs(q_inf) < 1e-12:
        raise ValueError("Freestream dynamic pressure is zero")
    return ((pressure - float(p_inf)) / q_inf).astype(np.float32)


def _resolve_sample_dir(fidelity_root: Path, sample_id: str) -> Path:
    candidate = fidelity_root / sample_id
    if candidate.is_dir():
        return candidate
    if (fidelity_root / f"{sample_id}_surface_flow.csv").exists():
        return fidelity_root
    raise FileNotFoundError(candidate)


# EdgeConvBlock's local-neighbourhood aggregation needs each point's k nearest
# neighbours. A given sample's point cloud (and hence its neighbour
# structure) is fixed across the whole run at num_points=None -- the
# production setting -- so paying an O(N^2) brute-force GPU search inside the
# model's forward pass on every single training step (measured: ~74 ms at
# N=20000, more than the rest of the forward+backward pass combined) is pure
# waste. Precomputing it once here with a CPU cKDTree (~50 ms one-time,
# amortised over hundreds of epochs via FidelityPressureDataset's existing
# per-sample record cache) and shipping the fixed indices alongside the
# points removes that cost from the training loop entirely.
DEFAULT_KNN_K = 16


def _compute_knn_indices(points: np.ndarray, k: int = DEFAULT_KNN_K) -> np.ndarray:
    n = points.shape[0]
    k_eff = min(k, max(n - 1, 1))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    # tree.query's first column is each point itself (distance 0); drop it so
    # the returned indices are strictly the k_eff *other* nearest points, and
    # pad by repeating the last neighbour if k_eff < k (tiny point clouds).
    idx = idx[:, 1:]
    if idx.shape[1] < k:
        pad = np.repeat(idx[:, -1:], k - idx.shape[1], axis=1)
        idx = np.concatenate([idx, pad], axis=1)
    return idx.astype(np.int64)


def _side_aware_knn_indices(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = DEFAULT_KNN_K,
    candidate_mult: int = 6,
) -> np.ndarray:
    """k nearest neighbours restricted to the same side of the surface.

    On a thin fin (thickness comparable to the mesh spacing) a plain Euclidean
    neighbourhood spans *both* faces of the plate, so the local geometry the
    network sees is wrong: the PCA normal mixes two surfaces that are not
    connected, and so does every edge feature.

    The split is found from the **offsets along each point's own normal**. Those
    offsets are bimodal on a thin plate -- a cluster near 0 (the same face) and a
    cluster near +/-t (the opposite face) -- so the largest gap in the sorted
    offsets locates the thickness, independent of the local grid spacing. On a
    smooth curved surface the offsets vary continuously with no such gap, so
    nothing is cut beyond what is needed to still return k neighbours.

    A fixed slab width cannot work here: the opposite face sits at r/s ~ 0.86 for
    the fin proportions in this dataset (0.6 mm thick, 0.7 mm in-plane spacing),
    i.e. closer than the second neighbour ring, while a k=24 neighbourhood reaches
    r/s ~ 2.8. Any threshold above ~0.5 t keeps the wrong face; any threshold tied
    to r_k is larger than t. The largest-gap rule avoids both traps and was
    verified on a synthetic plate with exactly those proportions.
    """
    n = int(points.shape[0])
    k_eff = min(int(k), max(n - 1, 1))
    candidates = min(n - 1, max(k_eff * int(candidate_mult), k_eff + 8))
    dist, idx = cKDTree(points).query(points, k=candidates + 1)
    dist = np.atleast_2d(dist)[:, 1:]
    idx = np.atleast_2d(idx)[:, 1:]
    m = int(idx.shape[1])

    if m > k_eff:
        delta = points[idx] - points[:, None, :]                     # [n, m, 3]
        offset = np.abs(np.einsum("nmi,ni->nm", delta, normals))     # [n, m]

        o_sorted = np.sort(offset, axis=1)                           # [n, m]
        gaps = np.diff(o_sorted, axis=1)                             # [n, m-1]
        # only allow a cut that still leaves at least k_eff survivors
        allowed = (np.arange(gaps.shape[1])[None, :] + 1) >= k_eff
        gaps = np.where(allowed, gaps, -np.inf)
        cut = np.argmax(gaps, axis=1)
        upper = np.minimum(cut + 1, m - 1)
        threshold = 0.5 * (o_sorted[np.arange(n), cut] + o_sorted[np.arange(n), upper])

        kept_dist = np.where(offset <= threshold[:, None], dist, np.inf)
        selection = np.argsort(kept_dist, axis=1)[:, :k_eff]
        out = np.take_along_axis(idx, selection, axis=1)
    else:
        out = idx[:, :k_eff]

    if out.shape[1] < k:
        out = np.concatenate([out, np.repeat(out[:, -1:], k - out.shape[1], axis=1)], axis=1)
    return out.astype(np.int64)


def load_fidelity_pressure_record(
    sample_id: str,
    fidelity_root: str | Path,
    num_points: int | None,
    seed: int,
    target_column: str = DEFAULT_TARGET_COLUMN,
    sampling_mode: str = "random",
    peak_sample_fraction: float = 0.20,
    tail_sample_fraction: float = 0.20,
    outer_sample_fraction: float = 0.15,
    knn_k: int = DEFAULT_KNN_K,
    include_fields: bool = False,
    side_aware_knn: bool = False,
) -> dict[str, object]:
    fidelity_root = Path(fidelity_root)
    sample_dir = _resolve_sample_dir(fidelity_root, sample_id)
    surface_path = sample_dir / f"{sample_id}_surface_flow.csv"
    params_path = sample_dir / "parameters.json"
    run_log_path = sample_dir / "run.log"
    if not surface_path.exists():
        raise FileNotFoundError(surface_path)
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    df = pd.read_csv(surface_path, usecols=["PointID", "x", "y", "z", "Density", "Momentum_x", "Momentum_y", "Momentum_z", "Energy"])
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    mach = float(params["mach"])
    p_inf = _parse_freestream_pressure(run_log_path)

    local_pressure = _pressure_from_conservative_vars(
        density=df["Density"].to_numpy(dtype=np.float32),
        momentum_x=df["Momentum_x"].to_numpy(dtype=np.float32),
        momentum_y=df["Momentum_y"].to_numpy(dtype=np.float32),
        momentum_z=df["Momentum_z"].to_numpy(dtype=np.float32),
        energy=df["Energy"].to_numpy(dtype=np.float32),
    )
    cp = _cp_from_pressure(local_pressure, mach=mach, p_inf=p_inf)
    cp_ceiling = min(CP_STAGNATION_OVERSHOOT * stagnation_cp(mach), MAX_ABS_CP_VALUE)
    bad = ~np.isfinite(cp) | (np.abs(cp) > cp_ceiling)
    bad_count = int(bad.sum())
    if bad_count:
        bad_fraction = bad_count / bad.size
        if bad_fraction > MAX_UNPHYSICAL_CP_FRACTION:
            raise ValueError(
                f"Unphysical Cp in {sample_id}: {bad_count}/{bad.size} nodes "
                f"({bad_fraction:.3g}) exceed |Cp|={cp_ceiling:.4g} "
                f"({CP_STAGNATION_OVERSHOOT:g}x stagnation Cp at Ma {mach:.3f}), "
                f"limit={MAX_UNPHYSICAL_CP_FRACTION:.3g}; "
                f"max_abs={float(np.nanmax(np.abs(cp))):.6g}"
            )
        # Below the fraction limit these are isolated mesh-singularity nodes
        # (see MAX_UNPHYSICAL_CP_FRACTION). Drop them from the cloud before
        # sampling so the row indices stay aligned with df.
        keep = ~bad
        df = df.loc[keep].reset_index(drop=True)
        local_pressure = local_pressure[keep]
        cp = cp[keep]
    if target_column.lower() in {"cp", "cp_coeff", "pressure_coefficient"}:
        target_values = cp.reshape(-1, 1)
    elif target_column.lower() in {"pressure", "p"}:
        target_values = local_pressure.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown target_column: {target_column!r}")

    all_points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    all_target = target_values
    if num_points is None:
        indices = np.arange(len(df), dtype=np.int64)
    elif sampling_mode == "pressure_aware":
        indices = _pressure_aware_sample_indices(
            points=all_points,
            target=all_target,
            num_points=num_points,
            seed=seed,
            peak_fraction=peak_sample_fraction,
            tail_fraction=tail_sample_fraction,
            outer_fraction=outer_sample_fraction,
        )
    elif sampling_mode == "random":
        indices = _sample_indices(len(df), num_points, seed)
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode!r}")
    sampled = df.iloc[indices].reset_index(drop=True)

    points = sampled[["x", "y", "z"]].to_numpy(dtype=np.float32)
    point_ids = sampled["PointID"].to_numpy(dtype=np.int64)
    pressure = target_values[indices].astype(np.float32)
    condition = np.array([mach, float(params["aoa_deg"])], dtype=np.float32)
    knn_idx = _compute_knn_indices(points, k=knn_k)
    record: dict[str, object] = {
        "sample_id": sample_id,
        "point_ids": point_ids,
        "points": points,
        "condition": condition,
        "cp": pressure,
        "knn_idx": knn_idx,
    }
    if include_fields:
        # Velocity target, surface-geometry estimates for the physics-consistency
        # integral, the solver's reference loads, and the per-sample calibration
        # factor (see field_targets.py). Computed on the sampled subset so the
        # point/normal/area/velocity rows all correspond.
        normals, areas = estimate_surface_geometry(points, knn_idx)
        reference = parse_reference_quantities(sample_dir, sample_id)
        ref_force = np.asarray(reference["CF"], dtype=np.float32)
        record.update(
            {
                "velocity": wall_velocity_nondimensional(sampled, mach),
                "normal": normals,
                "area": areas,
                "ref_force": ref_force,
                "ref_moment": np.asarray(reference["CM"], dtype=np.float32),
                "ref_area": np.asarray([reference["ref_area"]], dtype=np.float32),
                "ref_length": np.asarray([reference["ref_length"]], dtype=np.float32),
                "moment_origin": np.asarray(reference["moment_origin"], dtype=np.float32),
                "force_calibration": np.asarray(
                    [
                        calibration_scale(
                            true_cp=pressure,
                            normals=normals,
                            areas=areas,
                            reference_force=ref_force,
                            ref_area=float(reference["ref_area"]),
                        )
                    ],
                    dtype=np.float32,
                ),
            }
        )
        if side_aware_knn:
            # Separate neighbourhood for the *model*: same-side neighbours only,
            # which removes the thin-plate ambiguity. The physics geometry above
            # deliberately keeps the Euclidean neighbourhood so normals, areas and
            # the calibration stay comparable with every previous result.
            record["knn_idx_side"] = _side_aware_knn_indices(points, normals, knn_k)
    return record


class FidelityPressureDataset(Dataset):
    def __init__(
        self,
        fidelity_root: str | Path,
        sample_ids: Sequence[str],
        num_points: int | None,
        seed: int = 42,
        target_column: str = DEFAULT_TARGET_COLUMN,
        sampling_mode: str = "random",
        peak_sample_fraction: float = 0.20,
        tail_sample_fraction: float = 0.20,
        outer_sample_fraction: float = 0.15,
        cache_records: bool = True,
        knn_k: int = DEFAULT_KNN_K,
        include_fields: bool = False,
        side_aware_knn: bool = False,
    ) -> None:
        self.fidelity_root = Path(fidelity_root)
        self.sample_ids = list(sample_ids)
        self.num_points = None if num_points is None else int(num_points)
        self.seed = int(seed)
        self.target_column = target_column
        self.sampling_mode = sampling_mode
        self.peak_sample_fraction = float(peak_sample_fraction)
        self.tail_sample_fraction = float(tail_sample_fraction)
        self.outer_sample_fraction = float(outer_sample_fraction)
        self.cache_records = bool(cache_records)
        self.knn_k = int(knn_k)
        self.include_fields = bool(include_fields)
        self.side_aware_knn = bool(side_aware_knn)
        self._record_cache: dict[str, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        if self.cache_records:
            cached = self._record_cache.get(sample_id)
            if cached is not None:
                return cached
        record = load_fidelity_pressure_record(
            sample_id=sample_id,
            fidelity_root=self.fidelity_root,
            num_points=self.num_points,
            seed=stable_sample_seed(sample_id, self.seed),
            target_column=self.target_column,
            sampling_mode=self.sampling_mode,
            peak_sample_fraction=self.peak_sample_fraction,
            tail_sample_fraction=self.tail_sample_fraction,
            outer_sample_fraction=self.outer_sample_fraction,
            knn_k=self.knn_k,
            include_fields=self.include_fields,
            side_aware_knn=self.side_aware_knn,
        )
        if self.cache_records:
            self._record_cache[sample_id] = record
        return record


def collate_fidelity_pressure_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    lengths = np.asarray([np.asarray(item["points"]).shape[0] for item in batch], dtype=np.int64)
    max_len = int(lengths.max()) if lengths.size else 0
    batch_size = len(batch)

    points = np.zeros((batch_size, max_len, 3), dtype=np.float32)
    point_ids = np.full((batch_size, max_len), -1, dtype=np.int64)
    cp = np.zeros((batch_size, max_len, 1), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    condition = np.stack([np.asarray(item["condition"], dtype=np.float32) for item in batch], axis=0)

    knn_k = np.asarray(batch[0]["knn_idx"]).shape[1] if batch and "knn_idx" in batch[0] else 0
    # Padding rows keep knn_idx=0 (a harmless self-loop into slot 0, itself
    # zero-padded) rather than -1 -- these rows are masked out by `mask`
    # before EdgeConvBlock ever sees them, but an out-of-range index would
    # still crash torch.gather if something reads it before masking.
    knn_idx = np.zeros((batch_size, max_len, knn_k), dtype=np.int64) if knn_k else None
    has_side = bool(knn_k) and any("knn_idx_side" in item for item in batch)
    knn_idx_side = np.zeros((batch_size, max_len, knn_k), dtype=np.int64) if has_side else None

    has_fields = any("velocity" in item for item in batch)
    velocity = np.zeros((batch_size, max_len, 3), dtype=np.float32) if has_fields else None
    normal = np.zeros((batch_size, max_len, 3), dtype=np.float32) if has_fields else None
    area = np.zeros((batch_size, max_len, 1), dtype=np.float32) if has_fields else None

    for batch_index, item in enumerate(batch):
        item_points = np.asarray(item["points"], dtype=np.float32)
        item_point_ids = np.asarray(item["point_ids"], dtype=np.int64)
        item_cp = np.asarray(item["cp"], dtype=np.float32)
        count = item_points.shape[0]
        points[batch_index, :count] = item_points
        point_ids[batch_index, :count] = item_point_ids
        cp[batch_index, :count] = item_cp
        mask[batch_index, :count] = True
        if knn_idx is not None:
            knn_idx[batch_index, :count] = np.asarray(item["knn_idx"], dtype=np.int64)
        if knn_idx_side is not None:
            knn_idx_side[batch_index, :count] = np.asarray(item["knn_idx_side"], dtype=np.int64)
        if has_fields:
            velocity[batch_index, :count] = np.asarray(item["velocity"], dtype=np.float32)
            normal[batch_index, :count] = np.asarray(item["normal"], dtype=np.float32)
            area[batch_index, :count] = np.asarray(item["area"], dtype=np.float32)

    result = {
        "sample_id": [item["sample_id"] for item in batch],
        "point_ids": torch.from_numpy(point_ids),
        "points": torch.from_numpy(points),
        "condition": torch.from_numpy(condition),
        "cp": torch.from_numpy(cp),
        "mask": torch.from_numpy(mask),
        "lengths": torch.from_numpy(lengths),
    }
    if knn_idx is not None:
        result["knn_idx"] = torch.from_numpy(knn_idx)
    if knn_idx_side is not None:
        result["knn_idx_side"] = torch.from_numpy(knn_idx_side)
    if has_fields:
        result["velocity"] = torch.from_numpy(velocity)
        result["normal"] = torch.from_numpy(normal)
        result["area"] = torch.from_numpy(area)
        result["ref_force"] = torch.from_numpy(
            np.stack([np.asarray(item["ref_force"], dtype=np.float32) for item in batch], axis=0)
        )
        result["ref_moment"] = torch.from_numpy(
            np.stack([np.asarray(item["ref_moment"], dtype=np.float32) for item in batch], axis=0)
        )
        result["ref_area"] = torch.from_numpy(
            np.stack([np.asarray(item["ref_area"], dtype=np.float32) for item in batch], axis=0)
        )
        result["ref_length"] = torch.from_numpy(
            np.stack([np.asarray(item["ref_length"], dtype=np.float32) for item in batch], axis=0)
        )
        result["moment_origin"] = torch.from_numpy(
            np.stack([np.asarray(item["moment_origin"], dtype=np.float32) for item in batch], axis=0)
        )
        result["force_calibration"] = torch.from_numpy(
            np.stack([np.asarray(item["force_calibration"], dtype=np.float32) for item in batch], axis=0)
        )
    return result


def build_fidelity_pressure_loader(
    fidelity_root: str | Path,
    sample_ids: Sequence[str],
    num_points: int | None,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = False,
    drop_last: bool = False,
    sampler=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    sampling_mode: str = "random",
    peak_sample_fraction: float = 0.20,
    tail_sample_fraction: float = 0.20,
    outer_sample_fraction: float = 0.15,
    knn_k: int = DEFAULT_KNN_K,
    include_fields: bool = False,
    side_aware_knn: bool = False,
) -> DataLoader:
    dataset = FidelityPressureDataset(
        fidelity_root=fidelity_root,
        sample_ids=sample_ids,
        num_points=num_points,
        seed=seed,
        target_column=target_column,
        sampling_mode=sampling_mode,
        peak_sample_fraction=peak_sample_fraction,
        tail_sample_fraction=tail_sample_fraction,
        outer_sample_fraction=outer_sample_fraction,
        knn_k=knn_k,
        include_fields=include_fields,
        side_aware_knn=side_aware_knn,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fidelity_pressure_batch,
    )


def fit_fidelity_standardizers(
    fidelity_root: str | Path,
    sample_ids: Sequence[str],
    num_points: int | None,
    seed: int,
    target_column: str = DEFAULT_TARGET_COLUMN,
    sampling_mode: str = "random",
    peak_sample_fraction: float = 0.20,
    tail_sample_fraction: float = 0.20,
    outer_sample_fraction: float = 0.15,
    include_fields: bool = False,
) -> dict[str, Standardizer]:
    point_blocks = []
    condition_blocks = []
    pressure_blocks = []
    velocity_blocks = []
    for sample_id in sample_ids:
        record = load_fidelity_pressure_record(
            sample_id=sample_id,
            fidelity_root=fidelity_root,
            num_points=num_points,
            seed=stable_sample_seed(sample_id, seed),
            target_column=target_column,
            sampling_mode=sampling_mode,
            peak_sample_fraction=peak_sample_fraction,
            tail_sample_fraction=tail_sample_fraction,
            outer_sample_fraction=outer_sample_fraction,
            include_fields=include_fields,
        )
        point_blocks.append(np.asarray(record["points"], dtype=np.float32))
        condition_blocks.append(np.asarray(record["condition"], dtype=np.float32))
        pressure_blocks.append(np.asarray(record["cp"], dtype=np.float32))
        if include_fields:
            velocity_blocks.append(np.asarray(record["velocity"], dtype=np.float32))
    points = np.concatenate(point_blocks, axis=0)
    conditions = np.stack(condition_blocks, axis=0)
    pressures = np.concatenate(pressure_blocks, axis=0)
    standardizers = {
        "points": Standardizer.fit(points),
        "condition": Standardizer.fit(conditions),
        "cp": Standardizer.fit(pressures),
    }
    if include_fields:
        standardizers["velocity"] = Standardizer.fit(np.concatenate(velocity_blocks, axis=0))
    return standardizers
