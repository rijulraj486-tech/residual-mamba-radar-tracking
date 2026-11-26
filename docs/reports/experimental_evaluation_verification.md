# Five-Seed Evaluation Pipeline Verification

## Changes made

- Replaced the unavailable `timeq` import with Python's standard-library `time` module. Harness import remains lazy and does not start training.
- Corrected velocity and acceleration RMSE conversion: stored state values are already in km/s and km/s², so `SCALE=1000` is the only conversion. Position RMSE and six-state MSE formulas are unchanged. The current campaign uses `dt=1`; its stored metrics therefore remain numerically unchanged.
- Expanded reference evaluator calibration to validate all available same-reference metrics, prediction shape and finiteness, the fixed test-set hash, and chunk consistency. Its receipt identifies the separate reference checkpoints and does not claim to validate five-seed checkpoints.
- Added `experiments/evaluate_five_seed.py` to load the five recorded Mamba and five KalmanNet configurations and existing checkpoints, run evaluation mode without gradients, and write checkpoint metrics, provenance, and stored-versus-recomputed comparisons exclusively to a new `independent_verification/` directory. It refuses to overwrite that directory.
- Added focused synthetic metric tests for two values of `dt` and invalid `dt`.

No model architecture, training configuration, checkpoint, or historical result file was changed. No training was started.

## Checks performed

- Python syntax parsing passed for the harness, independent evaluator, and metric tests.
- The harness imported successfully under the project virtual environment without importing PyTorch or the training pipeline; its timer resolves to the standard-library `time` module.
- All ten checkpoint SHA-256 values match their historical `run_metrics.json` records. The shared test-set SHA-256 matches the expected fixed-pool hash.
- All ten tracked source-file hashes match the fixed-pool manifest, preserving the recorded model and training configuration provenance used by the campaign.
- `pytest` passed all three focused synthetic metric tests.
- The reference evaluator calibration passed against the separate reference benchmark checkpoints. It checked position RMSE, aggregate six-state MSE, and aggregate MSE in dB where matching reference values exist; it also calculated all five metrics, checked test and prediction shapes/finiteness, verified the test hash, and passed chunk consistency (Mamba maximum delta `7.6294e-06`, KalmanNet `0.0`).
- Independent evaluation completed for all ten checkpoints on an NVIDIA GeForce RTX 3060 using PyTorch `2.4.0+cu121`, CUDA `12.1`, and Python `3.12.3`. The environment's latest Transformers release was incompatible with Mamba-SSM 2.2.0, so Transformers `4.45.2` was installed in `.venv` only; no repository dependency file was changed.
- No training command was run. The independent evaluator wrote its outputs only after all ten evaluations finished.

## Independent checkpoint evaluation results

These values were recomputed from each existing checkpoint and match the stored historical values exactly at the printed precision. Full precision is in `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/independent_verification/checkpoint_evaluations.json`.

| Seed | Model | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | MSE (dB) |
|---:|---|---:|---:|---:|---:|---:|
| 11 | Mamba | 333.260101 | 80.887711 | 69.294670 | 0.0204011444 | -16.903455 |
| 11 | KalmanNet | 387.694275 | 92.251099 | 13.476514 | 0.0264997873 | -15.767576 |
| 22 | Mamba | 330.576202 | 82.451508 | 65.860909 | 0.0200694241 | -16.974651 |
| 22 | KalmanNet | 390.398285 | 94.437836 | 14.107372 | 0.0269213933 | -15.699025 |
| 33 | Mamba | 330.671478 | 77.864372 | 61.170517 | 0.0198580567 | -17.020633 |
| 33 | KalmanNet | 429.351593 | 96.879227 | 14.867132 | 0.0323248990 | -14.904628 |
| 44 | Mamba | 333.613586 | 78.213188 | 64.991394 | 0.0202732030 | -16.930776 |
| 44 | KalmanNet | 403.787048 | 93.531776 | 13.782588 | 0.0286636893 | -15.426679 |
| 55 | Mamba | 332.302765 | 76.861740 | 72.249702 | 0.0202588160 | -16.933859 |
| 55 | KalmanNet | 395.127197 | 96.733261 | 14.629726 | 0.0276161451 | -15.588369 |

## Stored-versus-recomputed comparison

All 50 comparisons passed: five metrics for each of ten checkpoints. The maximum absolute and relative differences were both `0.0`. The complete comparison is in `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/independent_verification/stored_metric_comparison.csv`. The evaluator used `atol=1e-5` and `rtol=5e-4`; it did not rewrite historical metrics.

The reference calibration is separate. Its original purpose is checking the evaluation pipeline against `fixed_test_comparison_01/n1024` reference checkpoints; those checkpoints are not seed 11 and the calibration receipt does not verify the five-seed runs.

## Provenance

- Git commit at inspection: `90363fff84710a81d36b128b179a77762b1dd798` (working tree changes are uncommitted).
- Corrected harness SHA-256: `7f871e70bfc37d1186da734b8d2cebbf8482a60203413faf03fb05de99a78336`.
- Historical fixed-pool manifest harness SHA-256: `387d7b87bf3908a11e3f7fea59d047736c5ecd3e39450ebfe3e89189bc1ddcac`. Historical records therefore refer to a different harness hash; the original historical harness was not recovered or verified, and the manifest was not rewritten.
- Evaluation Python: project virtual environment Python `3.12.3` (the lightweight syntax and import checks also used system Python `3.13.5`). PyTorch `2.4.0+cu121`, CUDA `12.1`, GPU `NVIDIA GeForce RTX 3060`.
- Test-set SHA-256: `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`.

### Checkpoint SHA-256 values

| Seed | Mamba | KalmanNet |
|---:|---|---|
| 11 | `b3d90cc74d7eecd36023f8b7617b2bc14d7a70f871dbad646d81c13f82f10645` | `f8c7a3c1d67e8f3dc736711ea8b0392c9dfe3cbd24d1788576080665f96c4c70` |
| 22 | `60ca0fc6756be750918b386ad476d68124ed864d48d42df35f6680abacfa9b32` | `15f0a5ead9e557ce1504aea6bdf5df85cc3156672231b3af3a5fc9e3afa3eba0` |
| 33 | `73c1ca8ba10130cfef6e7f0af258af0e824063bf4aa2a702b83a1a394db0b29c` | `8d3b664d2ad71518b70ad6167ee5a0741d385616a0f2b20b080b266235b2f1b0` |
| 44 | `904a1d2da50480485c164fadd23ef4272cd9f22ec7aa4c04c321472c2ea1c1fa` | `eb476297c7fa38d8455944758ea3f7c051993ce4c694e7ee3bb53d6ee7f19048` |
| 55 | `a8b4d45ce653bda1ad6b7e156214cde65fdd1cb8a86ec73e7bcc6d052b2b9ace` | `9e8c68b3886cc56231628f1dfbbdd1852dd37bc1dfb4528760c7fa925db9fd4d` |

### Source-file SHA-256 values

| Source file | SHA-256 |
|---|---|
| `src/radar_tracking/config.py` | `ad8d3d88bc3dfce16c4a18d9e73de34400283c4f8bb3b1ba3f898fb6fd6dab38` |
| `src/radar_tracking/data/data_generation.py` | `c1566c6dbdc016d78b790735cebd475db0cfe239e7aea8bcefdd4927895b3750` |
| `src/radar_tracking/dynamics/constant_acceleration.py` | `b3179c645cb01aea5314d94d06da588991035062949470f7d0a9f9ffb4acf9d2` |
| `src/radar_tracking/models/bimamba_encoder.py` | `f32e35c8fe1dae2a588ef79d1b39f59c923af389340b4f4a55f1c5bb64580484` |
| `src/radar_tracking/models/ekf.py` | `5ae9026fa00c85d10dffcf9c2a9a634e6b3650a042242b8479449d0d4194ae8e` |
| `src/radar_tracking/models/kalman_net.py` | `f9fe65ae9370bc3e1eedde533d8eac1699737de618adecd240971ad2622b7f6c` |
| `src/radar_tracking/models/prediction_head.py` | `1613ac9c23f55f3958c2ccb14b8c5d546c6e5f5dd2908c22c78964890e578ca4` |
| `src/radar_tracking/models/tracker_model.py` | `a8ab89b063044da086a3bece898e84f5ac78af83a7ff7437dc0a656f60e3cc8f` |
| `src/radar_tracking/training/train_validate_test.py` | `d70d552794652c8e617e063d0234e0feea86b9ff09d8b277d2d801b5ef3f9b7c` |
| `src/radar_tracking/cli/benchmark_mamba.py` | `496469729319629ddf96c1cc8690e913f4b10b170ae037c97f580bc6448130d1` |

## Remaining status

The metric and import defects are repaired, all five seeds and both model types have been independently evaluated, and all 50 metric comparisons pass exactly. The five-seed campaign is independently verified against the recorded metrics on the fixed test set. Historical checkpoint hashes, results, and manifest remain intact. The corrected harness hash differs from the historical manifest; the original harness was not recovered and the historical records were not retroactively rewritten.
