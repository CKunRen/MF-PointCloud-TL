# MF-PointCloud-TL

Reference implementation of the point-cloud **residual multi-fidelity transfer
learning** method for high-fidelity aerodynamic **surface field** prediction
(surface pressure field as the primary output, wall-velocity field as an
auxiliary output) on finned missile configurations.

A single point-cloud network (Fourier positional encoding + a local-neighbourhood
aggregation block + condition gating + a light residual decoder) is first
pretrained on abundant coarse-grid CFD and then fine-tuned on limited fine-grid
CFD. The fine-tuning stage is formulated **residually**: instead of regressing the
fine-grid field directly, a zero-initialised correction branch is trained on top of
the pretrained coarse-grid prediction, so the low-fidelity solution acts as a
physical prior and the residual carries the fidelity gap.

## What is in this repository

Only the **network and training code**:

```
kdmfpnet/                     package
  model.py                    shared MLP / point-MLP building blocks
  pressure_model.py           Fourier positional encoder, point-wise decoder
  kd_model.py                 PressureKDRegressor: encoder, EdgeConv, pooling, decoders
  residual_model.py           ResidualFieldModel (prior + correction) and freezing strategies
  losses.py                   pointwise loss weighting
  field_losses.py             dual-field loss and the physics-consistency penalty
  field_targets.py            wall velocity, surface normals/areas, reference loads
  kd_data.py                  dataset, kNN precomputation, batch collation
  field_train.py              the two-stage residual TL trainer
  kd_train.py                 shared training utilities used by field_train
  utils.py                    standardizers, seeding, JSON/CSV helpers
experiment_transfer_learning_fields.py    entry point (edit the constants, run)
```

**No data and no trained models are included.** The CFD datasets and the trained
weights are not distributed here.

## Requirements

```
python >= 3.10
torch (with CUDA for practical runtimes)
numpy, pandas, scipy
```

Install with `pip install -r requirements.txt`. A GPU is strongly recommended: the
production runs used a single CUDA device with one full point cloud (about 2x10^4
nodes) per optimisation step.

## Expected data layout

Point the code at a data root (default: a sibling directory, override with the
`MF_DATA_ROOT` environment variable). Each case is a directory holding a surface
point cloud and the solver outputs:

```
<MF_DATA_ROOT>/
  G2/
    usable_g2_train_800.txt            one sample id per line (manifest)
    g2_train_0000/
      g2_train_0000_surface_flow.csv   surface point cloud (columns below)
      parameters.json                  {"mach": ..., "aoa_deg": ..., ...}
      loads_summary.csv                solver force/moment coefficients
      run.log                          freestream static pressure
  G4/
    usable_g4_train_300.txt
    usable_g4_test_200.txt
    g4_train_0000/ ...
    g4_test_0000/ ...
```

`*_surface_flow.csv` must contain, per surface node:

```
PointID, x, y, z, Density, Momentum_x, Momentum_y, Momentum_z, Energy
```

The pressure coefficient is reconstructed from the conservative variables in the
code. `parameters.json` needs at least `mach` and `aoa_deg`. `loads_summary.csv`
is parsed for the reference force and moment coefficients
(`history_last_row_json` -> `CFx, CFy, CFz, CMx, CMy, CMz`).

## How to run

Open `experiment_transfer_learning_fields.py` and set the constants at the top
(data manifests, epochs, learning rates, loss weights, freezing strategy), then:

```bash
python experiment_transfer_learning_fields.py
```

Set `SMOKE_TEST = True` first to validate the pipeline on a handful of samples
before launching the full run. A run writes, into a timestamped folder under
`runs/`:

```
field_config.json            the resolved configuration
split_manifest.json          the train/validation/test sample ids
normalizers.json             per-fidelity standardizers
field_pretrain_history.csv   stage-1 loss curves
field_finetune_history.csv   stage-2 loss curves
field_metrics.json           test-set metrics (pressure R^2/RMSE/MAE, velocity, loads)
field_predictions.csv        per-node predictions on the test set
field_sample_metrics.csv     per-case metrics
```

Key production settings (see the paper): pretraining 120 epochs at 8e-4,
fine-tuning 100 epochs at 4e-4, dropout 0.14, EdgeConv neighbours k = 24, Fourier
bands 4, velocity-loss weight 0.3, physics-consistency weight 0.1, full
fine-tuning (no layers frozen).

## Notes

* Each point cloud is one optimisation step (batch size 1), because cases differ
  in node count.
* Surface normals and nodal areas are **estimated from the point cloud** (local
  PCA plane fit; k-nearest-neighbour density), since no surface mesh is assumed to
  be available. The physics-consistency penalty is calibrated per sample to remove
  the first-order magnitude bias of that estimate.
* The low- and high-fidelity datasets are unpaired (different configurations and
  conditions), which is why the correction branch is trained as a function of the
  input rather than on a pointwise coarse-to-fine difference.
* If you have exact surface meshes, replacing `estimate_surface_geometry` with a
  mesh-based metric reader is a drop-in change and makes the load integral exact.

## Citation

Please cite the corresponding paper (see the repository description / release
page for the reference).

## License

MIT License (see `LICENSE`).
