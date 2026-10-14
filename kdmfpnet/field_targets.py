"""Field-target and physics-consistency helpers for the residual multi-fidelity
transfer-learning method (paper contributions A+B+C).

The CFD surface files store conservative variables only. Beyond the Cp target
already handled by ``kd_data``, the new method needs three extra per-node
quantities and one global quantity per case:

* the wall velocity, non-dimensionalised the way the paper describes;
* a surface normal at every node (used to project the velocity onto the tangent
  plane and, more importantly, to integrate the predicted pressure into loads);
* a nodal area element for that same integral;
* the reference force/moment coefficients that the CFD solver reported, which
  the physics-consistency loss matches against.

No surface mesh (``*.su2``/``*.vtu``) is shipped with the local runtime, so the
normals and areas are *estimated from the point cloud itself*: normals from a
local PCA of each point's k nearest neighbours, areas from the local point
density. This is an approximation -- the estimated force integral reproduces the
solver's coefficients to roughly 80-95% on the physics-vetted samples (see
``tests/test_field_targets.py`` and the development log) -- so the resulting
physics term is a soft, magnitude-calibrated penalty rather than an exact
constraint. If the surface meshes are ever made available, swap
``estimate_surface_geometry`` for an exact mesh-metric reader and the rest of
the pipeline keeps working unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GAMMA = 1.4
# SU2 is run with REF_DIMENSIONALIZATION = FREESTREAM_VEL_EQ_ONE, i.e. V_inf = 1
# in solver units, hence a_inf = V_inf / Ma = 1 / Ma. Dividing the solver
# velocity by a_inf therefore means multiplying it by Ma, which is what
# wall_velocity_nondimensional does below.
DEFAULT_REF_AREA = 1.0
DEFAULT_REF_LENGTH = 1.0
# Cap on the per-sample magnitude-calibration factor (see calibration_scale):
# a sample whose estimated ground-truth integral is nearly zero (a nearly
# force-free attitude) would otherwise produce an unbounded factor.
MIN_CALIBRATION = 0.2
MAX_CALIBRATION = 5.0

_CFG_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def wall_velocity_nondimensional(df: pd.DataFrame, mach: float, gamma: float = GAMMA) -> np.ndarray:
    """Return the wall velocity [N, 3], non-dimensionalised by a_inf.

    Velocity in solver units is momentum/density (V_inf = 1); dividing by
    a_inf = 1/Ma is a multiplication by Ma.
    """
    density = df["Density"].to_numpy(dtype=np.float64)
    momentum = df[["Momentum_x", "Momentum_y", "Momentum_z"]].to_numpy(dtype=np.float64)
    velocity = momentum / np.clip(density, 1e-12, None)[:, None]
    return (velocity * float(mach)).astype(np.float32)


def estimate_surface_geometry(points: np.ndarray, knn_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (normals [N,3], nodal_areas [N,1]) estimated from the point cloud.

    Normals are the smallest-eigenvector direction of each point's local
    neighbourhood covariance (a local plane fit), oriented outward from the
    body axis along x. Areas use the standard k-NN density estimate
    A_i ~ pi * r_k^2 / k, with r_k the distance to the k-th neighbour.
    """
    pts = np.asarray(points, dtype=np.float64)
    idx = np.asarray(knn_idx, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError(f"knn_idx must be [N, k], got {idx.shape}")
    k = max(int(idx.shape[1]), 1)
    neighbours = pts[idx]
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k
    _, basis = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = basis[:, :, 0]

    radial = pts.copy()
    radial[:, 0] = 0.0  # missile axis is x
    radial_norm = np.linalg.norm(radial, axis=1)
    flip = (radial_norm > 1e-9) & ((normals * radial).sum(axis=1) < 0.0)
    normals[flip] *= -1.0

    r_k = np.linalg.norm(pts[idx[:, -1]] - pts, axis=1)
    areas = np.pi * r_k**2 / k
    return normals.astype(np.float32), areas.astype(np.float32).reshape(-1, 1)


def _cfg_values(text: str, names: tuple[str, ...]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in names:
        match = re.search(rf"{name}\s*=\s*({_CFG_NUMBER})", text)
        values[name] = float(match.group(1)) if match else None
    return values


def parse_reference_quantities(sample_dir: str | Path, sample_id: str) -> dict[str, np.ndarray | float]:
    """Return the solver-reported force/moment coefficients and reference scales.

    ``CF``/``CM`` come from ``loads_summary.csv`` (the JSON row), while
    ``ref_area``/``ref_length``/``moment_origin`` come from the ``*_cfd.cfg``.
    ``ref_area`` falls back to ``2 * RefForce`` (since q_inf = 0.5 under
    FREESTREAM_VEL_EQ_ONE) when the cfg is missing.
    """
    sample_dir = Path(sample_dir)
    ref_area: float | None = None
    ref_length: float | None = None
    origin = np.zeros(3, dtype=np.float32)
    cfg_path = sample_dir / f"{sample_id}_cfd.cfg"
    if cfg_path.is_file():
        values = _cfg_values(
            cfg_path.read_text(encoding="utf-8", errors="ignore"),
            (
                "REF_AREA",
                "REF_LENGTH",
                "REF_ORIGIN_MOMENT_X",
                "REF_ORIGIN_MOMENT_Y",
                "REF_ORIGIN_MOMENT_Z",
            ),
        )
        ref_area = values["REF_AREA"]
        ref_length = values["REF_LENGTH"]
        origin = np.array(
            [
                values["REF_ORIGIN_MOMENT_X"] or 0.0,
                values["REF_ORIGIN_MOMENT_Y"] or 0.0,
                values["REF_ORIGIN_MOMENT_Z"] or 0.0,
            ],
            dtype=np.float32,
        )

    summary_path = sample_dir / "loads_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    row = pd.read_csv(summary_path).iloc[0]
    raw = json.loads(row["history_last_row_json"])
    force = np.array([float(raw["CFx"]), float(raw["CFy"]), float(raw["CFz"])], dtype=np.float32)
    moment = np.array([float(raw["CMx"]), float(raw["CMy"]), float(raw["CMz"])], dtype=np.float32)
    if ref_area is None:
        ref_force = float(raw.get("RefForce", 0.0))
        ref_area = 2.0 * ref_force if ref_force > 0.0 else DEFAULT_REF_AREA
    if ref_length is None:
        ref_length = DEFAULT_REF_LENGTH
    return {
        "CF": force,
        "CM": moment,
        "ref_area": float(ref_area),
        "ref_length": float(ref_length),
        "moment_origin": origin,
    }


def integrated_force_coefficient(
    cp: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    ref_area: float,
) -> np.ndarray:
    """C_F = -(1/S_ref) * sum_i Cp_i n_i A_i, shape [3]."""
    cp = np.asarray(cp, dtype=np.float64).reshape(-1)
    normals = np.asarray(normals, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    return -(cp[:, None] * normals * areas[:, None]).sum(axis=0) / max(float(ref_area), 1e-12)


def calibration_scale(
    true_cp: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    reference_force: np.ndarray,
    ref_area: float,
) -> float:
    """Per-sample scalar that maps the estimated ground-truth integral onto the
    solver's reference force magnitude.

    Because the estimated normals/areas are approximate, the raw integral carries
    a systematic magnitude bias (measured ~0.83x on the physics-vetted samples).
    Training a loss that matched the raw integral to the reference loads would
    therefore push the network to over-predict every Cp by ~1/0.83. Rescaling by
    the factor that makes the *ground-truth* field match the reference removes
    that bias to first order, leaving the loss to constrain the predicted field's
    integral consistently with the true one.
    """
    estimate = integrated_force_coefficient(true_cp, normals, areas, ref_area)
    ref_norm = float(np.linalg.norm(reference_force))
    est_norm = float(np.linalg.norm(estimate))
    if est_norm < 1e-9 or ref_norm < 1e-9:
        return 1.0
    return float(np.clip(ref_norm / est_norm, MIN_CALIBRATION, MAX_CALIBRATION))
