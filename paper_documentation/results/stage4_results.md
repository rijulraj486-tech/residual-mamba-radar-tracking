# Stage 4 frozen T=100 extrapolation

This report reproduces the existing authoritative report text for convenient access. Original source: `artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/evaluation_report.md`. Numerical tables are also indexed in `results_provenance_index.csv`.

# Phase 3 Stage 4: Frozen 100-Step M0/M1 Evaluation on Long Test Sets

**Status:** PASS

This is frozen-checkpoint length extrapolation. No training, fine-tuning, checkpoint selection, normalization fitting, or Stage 5 decision was performed. All 20 evaluations used the Stage 2 test splits only after checkpoint/config/dataset integrity checks passed.

## Method and integrity

Loaded the five original M0 and five original M1 checkpoints strictly; all have 619014 trainable parameters. Batch size 16; 1 warmup and 3 timed full-split repetitions. Metrics match the established evaluator: joint x/y (or vx/vy, ax/ay) RMSE after scaling errors by 1,000, plus six-state MSE in scaled-state units. Temporal windows are 10 steps; six-state MSE is added per interval.

Integrity checks passed for Stage 3 checkpoint/config records, M1 train-only buffers, Stage 2 bundle/test manifests, split and tensor hashes, shapes, finiteness, trajectory IDs, and initialization alignment. Test IDs were disjoint across splits and lengths. No historical T=100 test file was loaded; its archived final metrics are used only for descriptive comparison.

## Full-horizon results

Values are mean ± sample SD across the five seeds.

| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE (scaled) |
|---:|---|---:|---:|---:|---:|
| 200 | M0 | 1000.77 ± 27.3 | 1029.53 ± 42.4 | 1725.75 ± 63.3 | 0.840821 ± 0.0582 |
| 200 | M1 | 953.493 ± 13.5 | 924.272 ± 20.8 | 1503.16 ± 56.4 | 0.67099 ± 0.0377 |
| 400 | M0 | 6900.51 ± 69.1 | 9344.06 ± 117 | 16091.6 ± 176 | 65.6514 ± 1.45 |
| 400 | M1 | 6778.8 ± 18.6 | 9094.36 ± 52 | 15589 ± 121 | 61.9483 ± 0.81 |

### Paired per-seed differences (M1 − M0)

Observed paired differences; no inferential significance is claimed.

| T | Metric | Mean difference | Sample SD |
|---:|---|---:|---:|
| 200 | position_rmse_m | -47.2801 | 22.3 |
| 200 | velocity_rmse_mps | -105.253 | 41.7 |
| 200 | acceleration_rmse_mps2 | -222.594 | 65.9 |
| 200 | six_state_mse_scaled_units | -0.169831 | 0.0536 |
| 400 | position_rmse_m | -121.706 | 55.1 |
| 400 | velocity_rmse_mps | -249.698 | 125 |
| 400 | acceleration_rmse_mps2 | -502.656 | 182 |
| 400 | six_state_mse_scaled_units | -3.7031 | 1.45 |

## Historical T=100 comparison

Archived final T=100 test results are shown descriptively; the 100-step test set was not reopened or reevaluated.

| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE (scaled) |
|---:|---|---:|---:|---:|---:|
| 100 | M0 | 332.085 ± 1.42 | 79.2557 ± 2.33 | 66.7134 ± 4.23 | 0.0201721 ± 0.000212 |
| 100 | M1 | 331.994 ± 3.09 | 72.276 ± 3.5 | 45.3557 ± 6.78 | 0.0195926 ± 0.000497 |

## Temporal behavior and resource summary

The four temporal plots compare M0/M1 at T=200/T=400 using the mean over seeds with ±1 sample SD shading. Exact per-seed 10-step metrics, including interval six-state MSE, are in `temporal_window_metrics.csv`.

| T | Variant | Mean ms/trajectory | Mean ms/observation | Peak allocated MB (max) | Peak reserved MB (max) |
|---:|---|---:|---:|---:|---:|
| 200 | M0 | 0.185 | 0.00093 | 47.47 | 70.00 |
| 200 | M1 | 0.195 | 0.00097 | 47.47 | 70.00 |
| 400 | M0 | 0.276 | 0.00069 | 82.05 | 110.00 |
| 400 | M1 | 0.286 | 0.00072 | 82.05 | 110.00 |

These Stage 4 times cover full-test-split inference with batch size 16 and GPU-resident inputs. They are separate from Stage 3’s batch-size-1 single-trajectory preflight timings.

The fixed-law T=400 test distribution has much larger range and state magnitudes than the T=100 training regime. Long-horizon errors therefore describe this frozen-model extrapolation under the Stage 2 physical law. They do not establish robustness to a new motion/noise regime and were not used to make configuration or training decisions.

## Provenance and output files

GPU: NVIDIA GeForce RTX 3060; driver 580.178.04; CUDA 12.1; PyTorch 2.4.0+cu121; mamba-ssm 2.2.0. Exact checkpoint, config, dataset, source, and output hashes are in `evaluation_report.json` and `provenance.json`.

- `per_seed_results.csv`: 20 checkpoint/horizon aggregate rows.
- `temporal_window_metrics.csv`: 600 checkpoint/interval rows (20 windows per T=200 evaluation; 40 per T=400 evaluation).
- `aggregate_results.csv`: mean and sample SD by variant/horizon/metric.
- `paired_differences.csv`: per-seed M1−M0 differences and summaries.
- `plots/`: four temporal plots.
