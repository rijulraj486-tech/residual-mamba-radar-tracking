# Stage 6 final held-out tests

This report reproduces the existing authoritative report text for convenient access. Original source: `artifacts/results/synthetic/paper_v4_long_sequence/stage6_matched_length_test_evaluation_v1/stage6_evaluation_report.md`. Numerical tables are also indexed in `results_provenance_index.csv`.

# Phase 3 Stage 6: Final Matched-Length Test Evaluation

**Status:** PASS (20/20 evaluations passed)

The five-seed Stage 5B checkpoints were evaluated on the frozen Stage 2 T=200/T=400 test splits. No training, fine-tuning, checkpoint selection, normalization fitting, or test-based tuning was performed. Stage 4 results are included from the hash-verified report and use the same test tensor files.

## Integrity and method

All 20 Stage 5B checkpoint/config records were SHA-256 verified and state dictionaries loaded strictly before test inference. M1 train-only statistics and checkpoint buffers matched. Every model had 619,014 trainable parameters. Dataset manifests, test split manifests, tensor files, tensor hashes, dimensions, finite values, trajectory IDs, and initialization alignment passed validation.

Inference used CUDA batch size 16, 1 warmup pass, and 3 synchronized timed repetitions per checkpoint/length. Inputs were GPU-resident during timing; this matches the Stage 4 timing protocol. Hardware: NVIDIA GeForce RTX 3060, driver 580.178.04, CUDA 12.1, PyTorch 2.4.0+cu121, Mamba 2.2.0.

## Matched-length full-horizon test results

Mean ± sample SD across five seeds.

| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |
|---:|---|---:|---:|---:|---:|
| 200 | M0 | 598.58 ± 23.6 | 221.174 ± 18.4 | 346.403 ± 14.6 | 0.088016 ± 0.0074 |
| 200 | M1 | 594.035 ± 8.14 | 130.636 ± 9.36 | 153.437 ± 13.9 | 0.0656274 ± 0.00163 |
| 400 | M0 | 2754.68 ± 122 | 1960.63 ± 179 | 2022.34 ± 181 | 2.59764 ± 0.317 |
| 400 | M1 | 2232.62 ± 18.3 | 494.717 ± 61.1 | 733.479 ± 89.6 | 0.962834 ± 0.02 |

## Paired M1 − M0 test differences

Negative values favor M1. Pairing follows the Stage 5B shared initial weights and data-loader order. These differences are descriptive; no inferential significance is claimed.

| T | Metric | Mean difference | Sample SD | Relative change mean ± SD |
|---:|---|---:|---:|---:|
| 200 | position_rmse_m | -4.545 | 21.7 | -0.66% ± 3.45% |
| 200 | velocity_rmse_mps | -90.5382 | 26.8 | -40.42% ± 8.16% |
| 200 | acceleration_rmse_mps2 | -192.965 | 24.5 | -55.58% ± 5.25% |
| 200 | six_state_mse_scaled_units | -0.0223886 | 0.00774 | -25.04% ± 6.24% |
| 400 | position_rmse_m | -522.058 | 113 | -18.83% ± 3.35% |
| 400 | velocity_rmse_mps | -1465.92 | 225 | -74.40% ± 5.59% |
| 400 | acceleration_rmse_mps2 | -1288.87 | 185 | -63.55% ± 5.31% |
| 400 | six_state_mse_scaled_units | -1.6348 | 0.325 | -62.41% ± 5.46% |

## Stage 4 extrapolation versus Stage 6 matched-length

Stage 4 is the frozen T=100-trained model on the same test trajectories; Stage 6 is trained at the matching length. Changes compare arm means and are descriptive between separately trained checkpoints.

| T | Variant | Metric | Stage 4 mean ± SD | Stage 6 mean ± SD | Absolute change | Relative change |
|---:|---|---|---:|---:|---:|---:|
| 200 | M0 | position_rmse_m | 1000.77 ± 27.3 | 598.58 ± 23.6 | -402.194 | -40.19% |
| 200 | M0 | velocity_rmse_mps | 1029.53 ± 42.4 | 221.174 ± 18.4 | -808.351 | -78.52% |
| 200 | M0 | acceleration_rmse_mps2 | 1725.75 ± 63.3 | 346.403 ± 14.6 | -1379.35 | -79.93% |
| 200 | M0 | six_state_mse_scaled_units | 0.840821 ± 0.0582 | 0.088016 ± 0.0074 | -0.752805 | -89.53% |
| 200 | M1 | position_rmse_m | 953.493 ± 13.5 | 594.035 ± 8.14 | -359.459 | -37.70% |
| 200 | M1 | velocity_rmse_mps | 924.272 ± 20.8 | 130.636 ± 9.36 | -793.637 | -85.87% |
| 200 | M1 | acceleration_rmse_mps2 | 1503.16 ± 56.4 | 153.437 ± 13.9 | -1349.72 | -89.79% |
| 200 | M1 | six_state_mse_scaled_units | 0.67099 ± 0.0377 | 0.0656274 ± 0.00163 | -0.605363 | -90.22% |
| 400 | M0 | position_rmse_m | 6900.51 ± 69.1 | 2754.68 ± 122 | -4145.83 | -60.08% |
| 400 | M0 | velocity_rmse_mps | 9344.06 ± 117 | 1960.63 ± 179 | -7383.42 | -79.02% |
| 400 | M0 | acceleration_rmse_mps2 | 16091.6 ± 176 | 2022.34 ± 181 | -14069.3 | -87.43% |
| 400 | M0 | six_state_mse_scaled_units | 65.6514 ± 1.45 | 2.59764 ± 0.317 | -63.0538 | -96.04% |
| 400 | M1 | position_rmse_m | 6778.8 ± 18.6 | 2232.62 ± 18.3 | -4546.18 | -67.06% |
| 400 | M1 | velocity_rmse_mps | 9094.36 ± 52 | 494.717 ± 61.1 | -8599.64 | -94.56% |
| 400 | M1 | acceleration_rmse_mps2 | 15589 ± 121 | 733.479 ± 89.6 | -14855.5 | -95.29% |
| 400 | M1 | six_state_mse_scaled_units | 61.9483 ± 0.81 | 0.962834 ± 0.02 | -60.9855 | -98.45% |

## Temporal metrics and resources

All Stage 6 seed-wise 10-step window metrics, including six-state MSE, are in `temporal_window_metrics.csv`. Four temporal plots show Stage 4 extrapolation and Stage 6 matched-length curves separately by line style for T=200 and T=400, with ±1 sample SD shading. The Stage 4/Stage 6 interval means and SDs are also in `stage4_vs_stage6_temporal_comparison.csv`.

Per-evaluation timing, per-trajectory and per-observation latency, batch size, warmup/repetition counts, peak allocated/reserved memory, baseline memory, and checkpoint/dataset provenance are in the machine-readable JSON report.

## Interpretation and limitations

These results evaluate the fixed Stage 2 motion/noise law and do not establish robustness to other regimes. Long-horizon errors reflect the broader state/range distribution at T=400. Comparisons between Stage 4 and Stage 6 are descriptive because they are separately trained checkpoints, even though they use the same test trajectories. Stage 6 metrics were not used to change models, normalization, or training settings.

Artifacts: `per_seed_test_results.csv`, `aggregate_test_results.csv`, `temporal_window_metrics.csv`, `paired_m0_m1_differences.csv`, `stage4_vs_stage6_comparison.csv`, `plots/`, `frozen_checkpoint_provenance.json`, `frozen_test_dataset_provenance.json`, and `provenance.json`.
