# Stage 8 baseline comparison

This report reproduces the existing authoritative report text for convenient access. Original source: `artifacts/results/synthetic/paper_v4_long_sequence/stage8_baseline_comparison_v1/stage8_evaluation_report.md`. Numerical tables are also indexed in `results_provenance_index.csv`.

# Stage 8: Baseline Audit and Long-Sequence Evaluation

**Status:** PASS (42/42 evaluation rows passed)

## Comparison groups

M0/M1 rows are imported from the verified frozen Stage 6 report. KalmanNet rows are five existing T=100 checkpoints evaluated as cross-length transfer. EKF/UKF rows are deterministic model-informed filters; oracle-noise and train-estimated-noise policies are separately labeled. The classical filters use exact F/h and therefore are not information-equivalent to Mamba.

## Full-horizon results

Means ± sample SD for learned checkpoint seeds. Classical filters are deterministic single results and have no seed SD.

| T | Model / policy | Seeds | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Runtime / device |
|---:|---|---:|---:|---:|---:|---:|---|
| 200 | M0 | 5 | 598.58 ± 23.6 | 221.174 ± 18.4 | 346.403 ± 14.6 | 0.088016 ± 0.0074 | 0.1911 s (cuda:0) |
| 200 | M1 | 5 | 594.035 ± 8.14 | 130.636 ± 9.36 | 153.437 ± 13.9 | 0.0656274 ± 0.00163 | 0.2023 s (cuda:0) |
| 200 | KalmanNet T100 transfer | 5 | 39909.4 ± 977 | 10215.7 ± 1.61e+03 | 572.946 ± 145 | 283.385 ± 17 | 7.915 s (cuda:0) |
| 200 | EKF legacy_P0_oracle_QR | 1 deterministic | 884.828 | 238.025 | 123.89 | 0.142487 | 0.1946 s (cpu) |
| 200 | UKF legacy_P0_oracle_QR | 1 deterministic | 177313 | 96831.9 | 20337.1 | 6871.63 | 2.203 s (cpu) |
| 200 | EKF train_calibrated_P0_oracle_QR | 1 deterministic | 544.379 | 65.8749 | 4.9918 | 0.0501188 | 0.1878 s (cpu) |
| 200 | UKF train_calibrated_P0_oracle_QR | 1 deterministic | 543.702 | 65.845 | 4.99136 | 0.0499953 | 2.238 s (cpu) |
| 200 | EKF train_calibrated_P0_train_estimated_QR | 1 deterministic | 544.306 | 65.8602 | 4.99193 | 0.0501052 | 0.193 s (cpu) |
| 200 | UKF train_calibrated_P0_train_estimated_QR | 1 deterministic | 543.641 | 65.8307 | 4.99147 | 0.049984 | 2.284 s (cpu) |
| 400 | M0 | 5 | 2754.68 ± 122 | 1960.63 ± 179 | 2022.34 ± 181 | 2.59764 ± 0.317 | 0.2848 s (cuda:0) |
| 400 | M1 | 5 | 2232.62 ± 18.3 | 494.717 ± 61.1 | 733.479 ± 89.6 | 0.962834 ± 0.02 | 0.2959 s (cuda:0) |
| 400 | KalmanNet T100 transfer | 5 | 379679 ± 2.14e+03 | 58008.7 ± 1.84e+03 | 12213.6 ± 1.38e+03 | 24613 ± 267 | 15.99 s (cuda:0) |
| 400 | EKF legacy_P0_oracle_QR | 1 deterministic | 2219.6 | 211.203 | 87.4047 | 0.829815 | 0.3642 s (cpu) |
| 400 | UKF legacy_P0_oracle_QR | 1 deterministic | 125580 | 66192.8 | 13871.5 | 3390.69 | 4.487 s (cpu) |
| 400 | EKF train_calibrated_P0_oracle_QR | 1 deterministic | 2142.79 | 116.884 | 5.79119 | 0.767539 | 0.3969 s (cpu) |
| 400 | UKF train_calibrated_P0_oracle_QR | 1 deterministic | 2052 | 115.072 | 5.77832 | 0.703993 | 4.612 s (cpu) |
| 400 | EKF train_calibrated_P0_train_estimated_QR | 1 deterministic | 2141.5 | 116.863 | 5.79118 | 0.766621 | 0.4005 s (cpu) |
| 400 | UKF train_calibrated_P0_train_estimated_QR | 1 deterministic | 2051.65 | 115.068 | 5.77842 | 0.703754 | 4.652 s (cpu) |

## Learned per-seed records

See `learned_per_seed_metrics.csv` for all Stage 6 M0/M1 and Stage 8 KalmanNet seed values and checkpoint hashes. KalmanNet seed IDs are its training checkpoints; they are not paired inferentially with Stage 5B M0/M1 seeds.

## Numerical stability and failures

No evaluation rows failed. EKF and UKF recorded per-step finite state/covariance checks and covariance snapshots. KalmanNet outputs, gain norms, and log-innovation diagnostics were checked for finiteness; state clipping-bound hits were counted. KalmanNet's internal sanitize/clamp calls were not instrumented, so internal saturation counts are unavailable.

## Runtime

single complete CPU inference pass, no warmup; wall clock around full split loop. one warmup pass on first up-to-64 Stage 2 train trajectories, then one timed complete test pass; CUDA synchronized before/after; chunks of 64.

EKF/UKF ran on CPU and KalmanNet on CUDA. These implementation/device runtimes are not directly comparable as algorithm throughput. M0/M1 timing is taken from Stage 6, not remeasured.

## Interpretation limits

The M0/M1 results are matched-length trained; KalmanNet was trained at T=100 and is reported only as cross-length transfer. Classical filters use exact F/h; oracle_QR also uses exact manifest noise covariances, while train-estimated_QR uses only Stage 2 train labels. All classical rows are privileged model-informed references. Their single deterministic scores are not five-seed distributions. No test result changed calibration, parameters, checkpoint selection, metric definitions, or reporting rules.

The Stage 2 train/test split manifests, test-file hashes, tensor hashes, and trajectory IDs are recorded in JSON/provenance. Existing Stage 6 and Stage 7 artifacts were read-only.

Detailed outputs: `stage8_evaluation_report.json`, `evaluation_provenance.json`, `combined_metrics.csv`, `learned_per_seed_metrics.csv`, `classical_filter_metrics.csv`, and `learned_seed_aggregates.csv`.
