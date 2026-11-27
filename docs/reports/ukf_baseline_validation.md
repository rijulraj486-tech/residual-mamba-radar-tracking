# Synthetic Radar UKF Baseline Validation

**Evaluation:** Task 3B, fixed 1,024-sequence synthetic radar test set  
**Result bundle:** `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/ukf_baseline_v1/`  
**Source revision:** `90363fff84710a81d36b128b179a77762b1dd798` (worktree was already dirty)

## Summary











































































































































































































A separate scaled unscented Kalman filter was implemented in `src/radar_tracking/models/ukf.py`; the historical EKF source and records were not changed. Both UKF variants use the exact P0 values recorded by Task 3A, the fixed test's Q/R unchanged under `oracle_QR`, and the same 100 observations, initial states, state scaling, truth, and metric function as the EKF evaluation.

The calibrated UKF is numerically stable in the tested sequences and yields metrics very close to the calibrated EKF. The preserved legacy UKF result is extremely poor. Its P0 sets velocity and acceleration variances to 10,000; sigma points therefore span very broad state distributions and the nonlinear range-bearing transform leads to poor estimates. This outcome is reported as-is; it does not justify replacing or reinterpreting the historical EKF.

## Implementation and numerical policy

The state is `[x, vx, ax, y, vy, ay]`. For `n=6`, `alpha=1`, `beta=2`, and `kappa=0`, the implementation has `lambda=0`, `n+lambda=6`, `Wm[0]=0`, `Wc[0]=2`, and all 12 off-center mean and covariance weights equal to `1/12`. Sigma offsets are `sqrt(6)` times the columns of the Cholesky factor of P. The state transition is the simulator's linear constant-acceleration F; Q is added after sigma-point propagation. The measurement is propagated through the simulator's nonlinear range-bearing function, with a circular weighted bearing mean and wrapped bearing deviations and innovations.

Covariances are computed in float64. Before Cholesky, matrices are symmetrized and checked for finite values and PSD. Eigenvalues below `-1e-8` times the largest absolute diagonal are rejected as materially non-PSD. Singular or roundoff-level PSD cases receive bounded relative diagonal jitter from `1e-12` through `1e-6`, for at most seven Cholesky attempts. A persistent failure raises `UKFNumericalError`. Gain computation uses `cholesky_solve`; posterior covariance is symmetrized and checked again. Jitter and minimum posterior eigenvalue diagnostics are retained by the filter.

For the calibrated variant's first returned observation, the measurement was already used by the two-measurement initializer. The Task 3A training-only state/measurement-noise cross covariance is reused exactly. The UKF maps it through a statistical linearization of the nonlinear measurement function and includes the resulting cross terms in innovation and state-innovation covariance. Subsequent updates use the standard independent additive measurement-noise model.

## Validation

`tests/test_ukf.py`: **10 passed**. Tests cover frozen weights, sigma point dimensions and moment reconstruction, linear prediction against both `F P Fᵀ + Q` and the legacy EKF, linear-Gaussian update agreement, circular bearing mean and wrapping, nonlinear range-bearing moments, bounded jitter and material-indefiniteness rejection, non-finite input rejection, CPU/device and batch behavior, scale extremes, and repeated covariance symmetry/PSD.

Pre-benchmark checks in the evaluator all passed:

- Sigma point mean and covariance reconstruction.
- UKF and EKF linear prediction agreement.
- Bearing boundary mean and wrapped residual.
- Exact equality of both loaded P0 variants to the Task 3A result record.
- Covariance checks over eight **training-only** trajectories, with snapshots through the sequence. Maximum symmetry error was zero; the minimum eigenvalues at recorded snapshots ranged from `1.40e-6` to `5.52e-6`.

The test file hash was checked before evaluation against the required SHA-256. The training pool was checked against its manifest. Training data was used only for the stability run; no calibration or tuning was performed here.

## Fixed-test results

All metric calculations use `experiments/five_seed_harness.py:calculate_metrics`. Position, velocity, and acceleration RMSE are in metres, m/s, and m/s². The six-state MSE and dB are in the evaluator's scaled state units.

| Method | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Six-state MSE (dB) |
|---|---:|---:|---:|---:|---:|
| `legacy_ekf_oracle` | 809.7575 | 316.3859 | 184.8840 | 0.13166490 | -8.80530 |
| `calibrated_ekf_oracle` | 325.9908 | 66.0329 | 4.7767 | 0.01844219 | -17.34187 |
| `legacy_ukf_oracle` | 255,433.4531 | 134,598.3125 | 28,276.5410 | 14,027.08691 | 41.46967 |
| `calibrated_ukf_oracle` | 325.9494 | 66.0135 | 4.7762 | 0.01843727 | -17.34303 |

Both are single deterministic filter runs over the same test tensors. The close calibrated EKF/UKF scores are not five-seed learned-model statistics. With calibrated P0, UKF changes the metrics only slightly: position RMSE improves by about 0.041 m, velocity by 0.019 m/s, acceleration by 0.0005 m/s², and six-state MSE by about 0.027% relative to calibrated EKF. These small differences do not establish a meaningful UKF advantage. With the preserved legacy P0, UKF performs dramatically worse than legacy EKF. The calibrated P0 changes the UKF outcome substantially, consistent with its much smaller and empirically calibrated velocity/acceleration uncertainty.

The legacy P0 is the exact per-test-sequence matrix recorded by Task 3A immediately before the first prediction; its velocity and acceleration diagonal entries are each `1e4`. The calibrated P0 is the exact shared 6x6 matrix stored in the result JSON. It was estimated from 819 training trajectories using the Task 3A first-returned-state residual covariance method. Both full P0 records, including the legacy batch matrix, are in `ukf_baseline_results.json`.

The test payload SHA-256 is `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`. Its tensor hashes are also recorded in the result JSON. The training pool file SHA-256 is `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733`.

## Exact noise and evaluation configuration

The policy is `oracle_QR`: Q and R are the actual tensors from the fixed shared test payload, passed to the filter unchanged. Numerically, Q has two identical constant-acceleration 3x3 blocks:

```text
[[5.00000006e-08, 1.25000000e-07, 1.66666666e-07],
 [1.25000000e-07, 3.33333332e-07, 5.00000000e-07],
 [1.66666666e-07, 5.00000000e-07, 1.00000000e-06]]
```

The measurement covariance is `diag(8.99999985e-4, 3.04617424e-4)` for range and bearing in the simulator's stored units. No Q/R estimation was conducted. Oracle Q/R gives the filter simulator noise knowledge; results should be read as an oracle-noise baseline, not a learned-noise or deployment setting.

Other settings: `dt=1.0`, sequence length 100, 1,024 test sequences, state order `[x, vx, ax, y, vy, ay]`, `RANGE_BEARING`, CPU evaluation, float64 covariance arithmetic, float32 output predictions, and `five_seed_harness.calculate_metrics`. The run used Python 3.12.3, PyTorch 2.4.0+cu121, and CUDA 12.1 was available but not used for evaluation. Runtime was 1.094 s for legacy UKF and 1.096 s for calibrated UKF on the recorded CPU evaluation environment. Task 3A's EKF run recorded 0.113 s and 0.217 s respectively; these timings are environment- and implementation-specific.

## Output and provenance

- `ukf_baseline_results.json`: variants, four-method metrics, both P0 matrices, cross covariance, Q/R, validation outcomes, runtimes, tensor hashes, source-file hashes, environment, and Git status.
- `four_method_metrics.csv`: compact four-method metric table.
- Source code: `src/radar_tracking/models/ukf.py` and `experiments/evaluate_ukf_baselines.py`.
- Tests: `tests/test_ukf.py`.

The output directory was created with exclusive `exist_ok=False` semantics and was absent before evaluation. No checkpoints, shared data, manifests, EKF implementation, historical benchmark result, or five-seed results were modified. The evaluation provenance records Python, PyTorch, CUDA availability/version, device, Git revision and dirty status, and SHA-256 values for source files.

## Limitations and unresolved issues

1. The two variants use oracle Q/R. An estimated-Q/R UKF comparison remains future work and should estimate noise only from training data.
2. The legacy UKF's poor score demonstrates sensitivity to its extremely broad legacy P0 under nonlinear sigma-point measurement propagation. This audit did not tune UT parameters or covariance floors on test data.
3. The calibrated UKF's first-observation cross-covariance adjustment uses statistical linearization for a nonlinear observation model. It reuses the Task 3A training calibration and is a reasonable matched treatment, but it is an approximation; the near-identical calibrated scores should not be overinterpreted.
4. The validation covered deterministic mathematical cases and eight training sequences plus the fixed test evaluation. It is not a proof of stability for arbitrary covariances, trajectories, or sensor geometries. The range-bearing model is singular at the origin, which is outside the tested trajectory regime.
5. UKF runtime is measured on CPU and should not be generalized to other hardware or batch sizes.
