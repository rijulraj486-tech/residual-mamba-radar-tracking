# Stage 5B five-seed confirmation

This report reproduces the existing authoritative report text for convenient access. Original source: `artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1/stage5b_confirmation_report.md`. Numerical tables are also indexed in `results_provenance_index.csv`.

# Phase 3 Stage 5B: Five-Seed Matched-Length Confirmation

**Status:** PASS (20/20 runs completed)

All runs used the frozen Stage 5A model/training settings. Paired M0/M1 seeds shared initialization, training/dropout, and epoch-wise data ordering at each length; initial trainable-weight hashes and loader-order hashes were compared. M1 means/stds were fit separately per seed using only that seed's training split. Test tensors were not opened or evaluated.

## Validation metrics

Values are mean ± sample SD over five seeds.

| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |
|---:|---|---:|---:|---:|---:|
| 200 | M0 | 587.092 ± 15.9 | 213.427 ± 19.4 | 326.212 ± 17.5 | 0.0828987 ± 0.00609 |
| 200 | M1 | 575.857 ± 4.85 | 123.992 ± 11.3 | 134.957 ± 16.4 | 0.0609228 ± 0.000915 |
| 400 | M0 | 2774.07 ± 131 | 2014.84 ± 183 | 2162.45 ± 188 | 2.74999 ± 0.34 |
| 400 | M1 | 2217.61 ± 36.4 | 545.14 ± 59 | 827.777 ± 78.1 | 0.984821 ± 0.0243 |

## Paired M1 − M0 differences

Negative values favor M1. Differences are descriptive; no inferential significance is claimed.

| T | Metric | Mean difference | Sample SD | Mean relative change |
|---:|---|---:|---:|---:|
| 200 | position_rmse_m | -11.2344 | 13.4 | -1.87% ± 2.20% |
| 200 | velocity_rmse_mps | -89.4355 | 29 | -41.29% ± 9.17% |
| 200 | acceleration_rmse_mps2 | -191.255 | 32.5 | -58.35% ± 7.14% |
| 200 | six_state_mse_scaled_units | -0.0219759 | 0.00633 | -26.20% ± 5.49% |
| 400 | position_rmse_m | -556.46 | 119 | -19.94% ± 3.49% |
| 400 | velocity_rmse_mps | -1469.7 | 229 | -72.56% ± 5.67% |
| 400 | acceleration_rmse_mps2 | -1334.68 | 197 | -61.51% ± 5.02% |
| 400 | six_state_mse_scaled_units | -1.76517 | 0.351 | -63.66% ± 5.49% |

## Frozen validation guardrails

| T | Mean position change | Mean velocity change | Mean guardrails | Paired seeds passing | Initial weights/order identical |
|---:|---:|---:|---|---:|---|
| 200 | -1.91% | -41.90% | PASS | 5/5 | True/True |
| 400 | -20.06% | -72.94% | PASS | 5/5 | True/True |

## Artifacts and resources

GPU: NVIDIA GeForce RTX 3060; CUDA 12.1; PyTorch 2.4.0+cu121; Mamba 2.2.0. Per-run training wall time, per-epoch training/validation times, allocated/reserved GPU peaks, checkpoint/config hashes, and train-only M1 statistics are in the JSON and run summaries.

- `five_seed_results.csv`: all 20 selected-checkpoint rows.
- `validation_window_metrics.csv`: per-seed selected-checkpoint metrics by 10-step interval.
- `aggregate_validation_results.csv`, `paired_seed_differences.csv`, and `validation_window_aggregate.csv`.
- `checkpoint_selection_manifest.json`: exact validation-selected checkpoint identities and hashes, frozen before any test access.
- `plots/`: seed-aggregated training/validation curves by sequence length and five-seed validation metric comparisons.

This stage confirms validation behavior only. Stage 6 test evaluation has not started.
