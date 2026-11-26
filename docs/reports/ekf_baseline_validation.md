# Reproducible Synthetic Radar EKF Baseline

**Scope:** validate the existing EKF, add an explicitly separate calibrated-prior variant, and evaluate both against the fixed synthetic test set. The legacy EKF source and historical records were not edited. No Mamba or KalmanNet model was trained, and no UKF was implemented.

## Tests performed and outcomes

`tests/test_ekf_baselines.py` passed **12 tests**. Coverage includes:

- Constant-acceleration F in `[x,vx,ax,y,vy,ay]` order, with `dt=0.5`.
- White-jerk Q block construction, including off-diagonal terms.
- Range-bearing h in multiple quadrants and near the origin.
- Analytic range-bearing Jacobian against central finite differences away from the origin (`eps=1e-3`, `rtol=2e-3`, `atol=2e-4`).
- Bearing residual wrapping at the 0/2π boundary and the `[-π,π)` edge convention.
- The existing EKF Joseph covariance update against its explicit formula (`rtol=1e-6`, `atol=1e-7`), symmetry, and eigenvalue tolerance `min_eigenvalue >= -1e-6`.
- The simulator's two-observation velocity initializer and zero acceleration initializer.
- The legacy filter's fixed-batch CPU contract, including mismatch rejection.
- `oracle_QR` passing generated non-unit-scaled Q/R unchanged, and `estimated_QR` requiring explicit training-only estimates.
- Training-only Q/R and initializer covariance estimation, full-state covariance shape, finite statistics, and PSD tolerance.
- Correlated first-observation update covariance symmetry and PSD (`min_eigenvalue >= -1e-5`).

The combined metric and EKF tests pass **15/15** (`.venv/bin/python -m pytest -q tests/test_five_seed_metrics.py tests/test_ekf_baselines.py`). These checks validate specific equations and numerical properties for the tested inputs; they do not prove global EKF stability, particularly near the polar origin or under ill-conditioned innovation covariances.

## Historical `legacy_ekf_oracle`

`src/radar_tracking/models/ekf.py` remains unchanged. The legacy variant uses the existing `reset` P0, existing initialization, and existing Joseph update. For the historical unit scales (`q2_scale=r2_scale=1`), the new evaluator passes the actual generated Q/R unchanged; this is numerically identical to the former `Q/scale`, `R/scale` expressions at scale 1.

Its test score reproduces the prior `fixed_test_comparison_01/n1024/benchmark_metrics.json` EKF record:

| Metric | Recomputed | Historical record | Absolute difference |
|---|---:|---:|---:|
| Position RMSE (m) | 809.757507 | 809.757507 | 0 |
| Six-state MSE | 0.131664902 | 0.131664902 | 0 |
| Six-state MSE (dB) | -8.805299799 | -8.805299759 | 3.98e-8 |

All checks passed at `rtol=5e-4`, `atol=1e-5`. The historical result, checkpoint, and manifest were read only.

The legacy range-bearing initializer uses

```text
P0 = diag(var_x, 1e4, 1e4, var_y, 1e4, 1e4)
var_x = cos²(theta) sigma_r² + r² sin²(theta) sigma_theta²
var_y = sin²(theta) sigma_r² + r² cos²(theta) sigma_theta²
```

Its position marginals are first-order polar-to-Cartesian terms, but the matrix omits x/y and initializer-induced position/velocity cross-covariances. The velocity/acceleration diagonal value `1e4` has standard deviation 100 in stored units—100 km/s or 100 km/s²—and is a very broad heuristic relative to the simulator's velocity ±0.2 km/s and acceleration ±0.003 km/s². This legacy configuration is preserved and labeled; the test score should not be interpreted as a well-calibrated-prior result.

## `calibrated_ekf_oracle` initialization

### Training-only calibration method

The fixed-pool train split contains 819 trajectories. The simulator initializes from two range-bearing measurements, estimates velocity from their Cartesian difference, and returns the second initializer measurement again as the first sequence observation. The fixed-pool training state sequence begins at that first returned time; it does not store the preceding latent state separately.

To calibrate the prior at the actual start of the returned sequence, the evaluator computes, using **only the 819 training rows**:

```text
prior_mean_i = F @ train_initial_state_i
prior_error_i = train_state_i[:, 0] - prior_mean_i
calibrated_P0 = unbiased_sample_covariance(prior_error)
```

This covariance is located immediately before the first returned observation, after propagating the supplied initializer once. It captures the two-observation velocity estimator, all position/velocity/acceleration cross-covariances, and first-transition noise. Since the initializer also consumed the first returned observation, the evaluator estimates `Cov(prior_error, first_measurement_noise)` from the training split and uses it in a correlated first EKF update. The linearized correlated update uses `Cov(e, innovation)=P Hᵀ+C` and `S=H P Hᵀ+H C+CᵀHᵀ+R`; later updates use the legacy EKF update. This first-prior and first-update behavior is separate from the historical path.

The estimator records but does not apply the empirical prior error mean. The fixed pool has no preceding latent state stored, so this method calibrates the predictive prior at the first returned observation rather than reconstructing a separate covariance before the initializer's first transition. It is a training-calibrated empirical approximation, not a proof that prior errors are Gaussian or stationary.

### Calibrated P0

The resulting matrix is in stored-state units squared and cross-products, ordered `[x,vx,ax,y,vy,ay]`:

```text
[[ 1.13336280e-01,  7.28182914e-03, -1.22758256e-05, -1.03996284e-01, -1.90641207e-03, -3.64760708e-05],
 [ 7.28182914e-03,  6.82752505e-02, -1.24072276e-05,  1.86212914e-04, -6.03167377e-02,  3.69773920e-06],
 [-1.22758256e-05, -1.24072276e-05,  5.10469363e-06,  2.77450945e-05,  2.23818552e-05, -6.98251199e-08],
 [-1.03996284e-01,  1.86212914e-04,  2.77450945e-05,  1.14141524e-01,  9.06320382e-03,  4.79543778e-05],
 [-1.90641207e-03, -6.03167377e-02,  2.23818552e-05,  9.06320382e-03,  6.81188107e-02,  1.93360975e-05],
 [-3.64760708e-05,  3.69773920e-06, -6.98251199e-08,  4.79543778e-05,  1.93360975e-05,  4.65652738e-06]]
```

The training residual mean (recorded, not applied) is `[-0.001481 km, 0.018944 km/s, -0.000105 km/s², 0.011644 km, -0.015286 km/s, 0.000078 km/s²]`. The covariance's minimum symmetric eigenvalue was `5.14e-7`; its maximum symmetry error after evaluator serialization to float32 was `4.47e-8`. The full precision P0 and first-observation cross-covariance are in the result JSON.

Calibration provenance:

- Training pool file SHA-256: `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733`.
- Training state tensor SHA-256: `332c2764a958b36aa588ed6595d1e29cc3e5bcf038f5db540532f507e46d9f3d`.
- Training observations tensor SHA-256: `8069d8b02e6c6a06f9c420eafa0ce3a445fd9007a2b85bed66adcba73a60c91f`.
- Training initializer tensor SHA-256: `381b74fa88836817f72ba21499af29b4921d0dcbcd8c5195ab5fb4db4503edef`.
- The validation split was not passed to calibration; the fixed test truth was used only after calibration to score the final estimates.

## Q/R policy

- **`oracle_QR`:** pass the actual generated Q/R tensors unchanged. This is the policy for both evaluated variants and is explicitly an oracle/model-known setting.
- **`estimated_QR`:** estimate Q from adjacent state-transition residuals and R from state-conditioned measurement residuals, wrapping bearing residuals. The code accepts only the training split for these estimates. It is implemented, tested, and its training-only matrices are saved as reference metadata; it was not used for either reported EKF score.

The test's actual covariance values are:

```text
Q = two copies of
[[5.00000006e-08, 1.249999997e-07, 1.666666662e-07],
 [1.249999997e-07, 3.333333325e-07, 4.999999987e-07],
 [1.666666662e-07, 4.999999987e-07, 9.999999975e-07]]
R = diag(8.99999985e-04 km², 3.04617424e-04 rad²)
```

Tests with non-unit scales 2.75 and 4.5 confirm `oracle_QR` does not divide the actual generated covariances again. The old benchmark's division bug is still present in its historical path; the current fixed test has unit scales, so the historical result is unchanged. Future non-unit evaluation must call the explicit covariance policy.

## Fixed-test evaluation

The evaluator used the existing file, without regeneration:

- `artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt`
- SHA-256: `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`
- 1,024 examples × 100 steps; initial state `[1024,6,1]`; observations `[1024,2,100]`; truth `[1024,6,100]`.
- State order `[x,vx,ax,y,vy,ay]`, `dt=1`, `SCALE=1000`; metrics use `five_seed_harness.calculate_metrics`.

| Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | MSE (dB) | CPU runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| `legacy_ekf_oracle` | 809.757507 | 316.385895 | 184.884003 | 0.131664902 | -8.805300 | 0.113 |
| `calibrated_ekf_oracle` | 325.990784 | 66.032913 | 4.776696 | 0.018442193 | -17.341874 | 0.217 |

The complete P0/Q/R values, test file and tensor hashes, training hashes, source hashes, runtime, configuration, and metrics are stored in [ekf_baseline_results.json](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/ekf_baseline_v1/ekf_baseline_results.json). Outputs were written exclusively to `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/ekf_baseline_v1/`; the evaluator refuses an existing output directory.

## Comparison with existing five-seed learned results

The two EKF values are deterministic single scores on this fixed test set. Mamba and KalmanNet rows below are means and sample standard deviations across five stored training-seed checkpoints; those standard deviations describe training-seed variation, not test-set uncertainty.

| Metric | `legacy_ekf_oracle` (one score) | `calibrated_ekf_oracle` (one score) | Mamba five-seed mean ± SD | KalmanNet five-seed mean ± SD |
|---|---:|---:|---:|---:|
| Position RMSE (m) | 809.7575 | 325.9908 | 332.0848 ± 1.4177 | 401.2717 ± 16.8459 |
| Velocity RMSE (m/s) | 316.3859 | 66.0329 | 79.2557 ± 2.3254 | 94.7666 ± 2.0181 |
| Acceleration RMSE (m/s²) | 184.8840 | 4.7767 | 66.7134 ± 4.2347 | 14.1727 ± 0.5771 |
| Six-state MSE | 0.1316649 | 0.0184422 | 0.0201721 ± 0.0002118 | 0.0284052 ± 0.0023391 |
| Six-state MSE (dB) | -8.8053 | -17.3419 | -16.9527 ± 0.0457 | -15.4773 ± 0.3451 |

The calibrated EKF score is numerically lower on this test set than the Mamba/KalmanNet seed means, but it uses simulator-oracle Q/R and an initializer prior calibrated from labeled training trajectories. It should be presented as a model-informed classical baseline, not an information-matched comparison. Do not infer statistical significance from this one deterministic test score.

## Reproducibility and limitations

- Git commit: `90363fff84710a81d36b128b179a77762b1dd798`; worktree was dirty at evaluation time due to existing Task 1 changes and the current uncommitted implementation. The JSON records the complete porcelain status and source-file SHA-256 values.
- Runtime: Python 3.12.3, PyTorch 2.4.0+cu121, CUDA 12.1, NVIDIA GeForce RTX 3060. EKF variants ran on CPU, so runtime is not directly comparable to GPU Mamba/KalmanNet timings.
- The test file and five test tensors are hashed in the result JSON. Training calibration uses only the train tensors in the fixed pool; its file and tensor hashes are recorded.
- The calibrated P0 represents the prior at the first returned observation, not a covariance at the earlier latent state before the first transition. This matches the stored sequence alignment and captures the actual initializer path.
- First-observation error/noise cross-covariance is estimated from the finite 819-row training split and applied under a local EKF linearization. Prior bias is recorded but not corrected. Non-Gaussian residuals, uncertainty in covariance estimates, and transfer to another simulator regime remain untested.
- `estimated_QR` is implemented and tested but not scored. The reported variants both use oracle Q/R. No validation or held-out test values were used for P0 or Q/R calibration.
- The legacy EKF still uses explicit inversion for its innovation update and is CPU/fixed-batch only. The calibrated first update uses a linear solve, but this does not establish global numerical stability. The covariance tests and recorded eigenvalue diagnostics only support the tested conditions.
- The old benchmark Q/R division behavior and old records remain unchanged. Under current unit scales the legacy result reproduces exactly (up to 4e-8 dB rounding); non-unit scales must use `oracle_QR` or `estimated_QR` from the new policy.
- No UKF, training run, Mamba/KalmanNet checkpoint change, historical result rewrite, or commit was made.
