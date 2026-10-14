"""Entry point for the residual dual-field multi-fidelity transfer-learning run.

This implements the paper's method (discretization-error-aware residual transfer
+ pressure/velocity dual output + physics-consistency penalty). It is a new
script so the original ``experiment_transfer_learning.py`` baseline is left
intact for the ablation.

Edit the constants below and run with the project's torch environment, e.g.:

    python experiment_transfer_learning_fields.py

Before training, sanity-check the run with a quick smoke test (see
``SMOKE_TEST``) which uses a handful of samples and 1-2 epochs.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from kdmfpnet.field_train import FieldTrainingConfig, train_residual_tl

# --- run control -----------------------------------------------------------
SMOKE_TEST = False          # True: tiny run to validate plumbing before the real one
NUM_POINTS = None           # None => native surface resolution (~20k nodes for G4)
BATCH_SIZE = 1
SEED = 42
DEVICE = ""                 # "" => auto (cuda if available)
OUTPUT_SUBDIR = "tl_residual_fields"
RUN_LABEL = "main_frozen_phys0"  # appended to the run folder name (distinguishes runs)
REUSE_PRIOR = True               # reuse runs/<OUTPUT_SUBDIR>/shared_prior.pt if present (skip re-pretraining)

# --- stage budgets / optimisation -----------------------------------------
PRETRAIN_EPOCHS = 120
FINETUNE_EPOCHS = 100
PRETRAIN_LR = 8e-4
FINETUNE_LR = 4e-4
WEIGHT_DECAY = 5e-4
DROPOUT = 0.14
KNN_K = 24
GRAD_ACCUM_STEPS = 1
GRAD_CLIP_NORM = 1.0
VAL_FRACTION = 0.1
FINETUNE_VAL_FRACTION = 0.18

# --- method-specific hyperparameters --------------------------------------
LAMBDA_VELOCITY = 0.3       # weight of the auxiliary wall-velocity field loss
# Physics-consistency weight. The surface integral is estimated from the point
# cloud (no mesh file is shipped), so this term is approximate and is disabled
# by default; validate with tests/test_field_targets.py and the force
# diagnostics in field_metrics.json before enabling it.
PHYSICS_LOSS_WEIGHT = 0.0
PHYSICS_IN_PRETRAIN = False
INCLUDE_MOMENTS = False
FREEZING_STRATEGY = "frozen"  # "frozen" (prior fully frozen, the residual default) or the ablations "none"/"early"/"late"
FINETUNE_LIMIT = None         # cap G4 fine-tuning cases (None = full); for the G4 data-budget study
PRETRAIN_LIMIT = None         # cap G2 pretraining cases (None = full); forces fresh pretraining
FROM_SCRATCH = False          # True: G4-only baseline (no pretraining, no correction branch)
USE_RESIDUAL = True           # False: direct fine-tuning from the pretrained prior, no correction branch

# --- loss shaping (unchanged from the previous TL baseline) ----------------
HARD_POINT_ALPHA = 1.5
HARD_POINT_FRACTION = 0.1
TARGET_PEAK_ALPHA = 0.35
TARGET_PEAK_FRACTION = 0.08
MACH_FOCUS_ALPHA = 0.60
MACH_FOCUS_START = 3.2
MACH_FOCUS_END = 4.5
LOW_SIGNAL_ALPHA = 0.35
LOW_SIGNAL_CAP = 1.75
EARLY_STOP_PATIENCE = 10**9
EARLY_STOP_MIN_EPOCHS = 60

# --- Tier-1 method improvements (2026-10-13 revision) ----------------------
# Error-driven per-sample focal weighting. In this dataset the hand-tuned
# Mach-focus / low-signal rules upweight the *easy* high-Mach regime, while the
# error is concentrated in the near-sonic tail (worst 10% of cases = 43% of the
# total SSE). Set FOCAL_SAMPLE_ALPHA > 0 and the two rules below to 0 to redirect
# capacity to the hard cases automatically.
FOCAL_SAMPLE_ALPHA = 0.0
FOCAL_SAMPLE_CAP = 4.0
# Project the predicted wall velocity onto the local tangent plane (slip wall).
SLIP_PROJECT = False
# Fourier positional-encoding bands. >4 changes the backbone input width, so the
# cached shared_prior.pt is not reusable and a fresh pretraining is required.
FOURIER_BANDS = 4
# Feed local surface descriptors (normal + shape) to the correction branch only.
# The prior is untouched, so the cached shared_prior.pt is still reusable.
SHAPE_FEATURES = False
# Weight of the local sharpness-matching loss (0 = off). Penalises a prediction
# that is locally smoother than the truth -- the signature of the error at the
# sharpest nodes. Does not change the network, so the prior stays reusable.
SHARPNESS_WEIGHT = 0.0
SHARPNESS_ONE_SIDED = True
# Restrict the model's kNN to same-side neighbours (thin-fin disambiguation).
SIDE_AWARE_KNN = False

# --- data manifests --------------------------------------------------------
G2_MANIFEST = "usable_g2_train_800.txt"
G4_TEST_MANIFEST = "usable_g4_test_200.txt"
G4_TRAIN_MANIFEST = "usable_g4_train_300.txt"


def _config() -> FieldTrainingConfig:
    config = FieldTrainingConfig(
        pretrain_epochs=PRETRAIN_EPOCHS,
        finetune_epochs=FINETUNE_EPOCHS,
        pretrain_lr=PRETRAIN_LR,
        finetune_lr=FINETUNE_LR,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        knn_k=KNN_K,
        seed=SEED,
        val_fraction=VAL_FRACTION,
        finetune_val_fraction=FINETUNE_VAL_FRACTION,
        grad_clip_norm=GRAD_CLIP_NORM,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lambda_velocity=LAMBDA_VELOCITY,
        physics_weight=PHYSICS_LOSS_WEIGHT,
        physics_in_pretrain=PHYSICS_IN_PRETRAIN,
        include_moments=INCLUDE_MOMENTS,
        freezing_strategy=FREEZING_STRATEGY,
        finetune_limit=FINETUNE_LIMIT,
        pretrain_limit=PRETRAIN_LIMIT,
        from_scratch=FROM_SCRATCH,
        use_residual=USE_RESIDUAL,
        hard_point_alpha=HARD_POINT_ALPHA,
        hard_point_fraction=HARD_POINT_FRACTION,
        target_peak_alpha=TARGET_PEAK_ALPHA,
        target_peak_fraction=TARGET_PEAK_FRACTION,
        mach_focus_alpha=MACH_FOCUS_ALPHA,
        mach_focus_start=MACH_FOCUS_START,
        mach_focus_end=MACH_FOCUS_END,
        low_signal_alpha=LOW_SIGNAL_ALPHA,
        low_signal_cap=LOW_SIGNAL_CAP,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_min_epochs=EARLY_STOP_MIN_EPOCHS,
        focal_sample_alpha=FOCAL_SAMPLE_ALPHA,
        focal_sample_cap=FOCAL_SAMPLE_CAP,
        slip_project=SLIP_PROJECT,
        fourier_bands=FOURIER_BANDS,
        shape_features=SHAPE_FEATURES,
        sharpness_weight=SHARPNESS_WEIGHT,
        sharpness_one_sided=SHARPNESS_ONE_SIDED,
        side_aware_knn=SIDE_AWARE_KNN,
    )
    if SMOKE_TEST:
        config.pretrain_epochs = 1
        config.finetune_epochs = 1
        config.early_stop_patience = 10**9
        config.sample_limit = 8
        # Exercise the optional paths too so a smoke run validates them.
        config.physics_weight = 0.1
        config.include_moments = True
    return config


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    # Data root: override with the MF_DATA_ROOT environment variable.
    runtime_root = Path(os.environ.get(
        "MF_DATA_ROOT",
        str(script_dir.parent / "parametric_missile_cfd_hpc_runtime_kd" / "results"),
    ))
    g2_manifest = runtime_root / "G2" / G2_MANIFEST
    g4_train_manifest = runtime_root / "G4" / G4_TRAIN_MANIFEST
    g4_test_manifest = runtime_root / "G4" / G4_TEST_MANIFEST
    for path in (g2_manifest, g4_train_manifest, g4_test_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")

    started_at = datetime.now()
    run_tag = started_at.strftime("%Y%m%d_%H%M%S") + f"_{RUN_LABEL}"
    output_dir = script_dir / "runs" / OUTPUT_SUBDIR / run_tag
    latest_dir = script_dir / "runs" / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_prior = script_dir / "runs" / OUTPUT_SUBDIR / "shared_prior.pt"

    result = train_residual_tl(
        runtime_root=runtime_root,
        output_dir=output_dir,
        g2_manifest=g2_manifest,
        g4_train_manifest=g4_train_manifest,
        g4_test_manifest=g4_test_manifest,
        config=_config(),
        num_points=NUM_POINTS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        pretrain_checkpoint=shared_prior if REUSE_PRIOR else None,
    )

    finished_at = datetime.now()
    print(f"[residual-tl] finished in {(finished_at - started_at).total_seconds() / 60.0:.1f} min", flush=True)
    print(f"[residual-tl] metrics: {result['metrics']}", flush=True)

    latest_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, latest_dir / item.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
