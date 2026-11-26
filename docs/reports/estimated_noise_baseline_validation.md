# Training-Estimated Q/R Baseline Validation

**Configuration:** `estimated_QR_train_only`  
**Evaluation:** fixed synthetic radar test set, 1,024 sequences × 100 returned observations  
**Result bundle:** `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/estimated_noise_baseline_v1/`  
**Git revision:** `90363fff84710a81d36b128b179a77762b1dd798` (worktree was dirty)

## Summary

Process and measurement noise were estimated only from the 819 fixed-pool training trajectories. The held-out test file was not opened until the estimates were computed and frozen in memory. The estimator receives only training states, training observations, and the simulator measurement/dynamics model. The shared serialized train/validation pool contains both splits; its extractor selected and verified only the `train` tensors, and no validation values were passed into estimation.

The final training-only covariance estimates are close to the simulator definitions: structured Q differs by 0.304% in relative Frobenius norm; the estimated range and bearing variances differ by -0.689% and +0.648%. The small raw range-bearing covariance corresponds to a residual correlation of -0.000669 and is not retained because the generator samples the channels independently.

With calibrated P0, estimated Q/R changes EKF and UKF scores only slightly. With legacy P0, estimated Q/R causes the legacy EKF to diverge badly, while the legacy UKF remains very poor under either noise policy. The legacy EKF's covariance also develops transient symmetry and PSD defects under both policies. The results therefore show that the learned noise estimates are close to oracle Q/R, but the historical EKF with its broad legacy P0 is highly sensitive and numerically fragile. That sensitivity should not be interpreted as evidence that these small covariance estimation errors alone explain the performance change.

## Data, units, and simulator covariance definitions

The simulator creates states in metre-based coordinates, then the fixed-pool preparation scales state and initializer values by 1,000. The filter inputs therefore use position in km, velocity in km/s, and acceleration in km/s². The transition matrix F is unchanged by this uniform state scaling. The benchmark also scales range observations from metres to km and leaves bearing in radians. Covariances for scaled quantities are divided by the square of the scale: Q by `1,000²`, and the range/range entry of R by `1,000²`; bearing variance and the time-indexed F convention are unchanged.

The simulator adds zero-mean Gaussian process noise at every state transition. Its white-jerk constant-acceleration covariance is `Q_CA_2D(dt, qx, qy)`, where `q = accn_rms_q² / dt`, with `q2_scale` applied. The state ordering is `[x,vx,ax,y,vy,ay]`, so Q consists of two independent 3×3 CA blocks. The measurement generator uses independent zero-mean Gaussian range and bearing noise. For this fixed pool, the configured standard deviations are 30 m and 1 degree; after benchmark scaling, the oracle R is diagonal in km² and rad².

The known stored-unit matrices derived from the simulator configuration and verified against the fixed test payload are:

```text
Q_oracle =
[[5.00000006e-08, 1.25000000e-07, 1.66666666e-07, 0, 0, 0],
 [1.25000000e-07, 3.33333332e-07, 5.00000000e-07, 0, 0, 0],
 [1.66666666e-07, 5.00000000e-07, 9.99999997e-07, 0, 0, 0],
 [0, 0, 0, 5.00000006e-08, 1.25000000e-07, 1.66666666e-07],
 [0, 0, 0, 1.25000000e-07, 3.33333332e-07, 5.00000000e-07],
 [0, 0, 0, 1.66666666e-07, 5.00000000e-07, 9.99999997e-07]]

R_oracle = diag(9.00000000e-04 km², 3.04617420e-04 rad²)
```

The fixed test file SHA-256 was verified as `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`; its Q/R tensors agree with these configuration-derived matrices within float serialization tolerance. No test data was regenerated.

## Training-only estimation method

Training pool file SHA-256: `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733`. It contains 819 training and 204 validation trajectories; only training state, observations, and initializer tensors were extracted. The training file and tensor hashes are recorded in the frozen config and result JSON.

| Training quantity | Count used |
|---|---:|
| Trajectories | 819 |
| Training state/observation time points | 81,900 |
| Adjacent transition residuals for Q | 81,081 |
| State-conditioned measurement residuals for R | 81,900 |

For each adjacent pair of training states, process residuals were calculated using the simulator transition:

```text
w[t] = x[t+1] - F x[t]
S_w = centered_unbiased_covariance(w)
```

The sequence begins at the first returned state, and all 99 adjacent transitions per 100-step trajectory follow the same transition plus additive process noise. The raw full six-state covariance was inspected. Cross-axis terms were around 1e-9 while within-axis terms were around 1e-6, consistent with independent x/y process noise.

Three forms were compared: raw full 6×6 empirical covariance, two shared 3×3 axis blocks (average of the x and y blocks, with cross-axis terms set to zero), and the simulator-structured common integrated-white-jerk model. The final Q uses the simulator-structured form because the generator specifies identical CA block shape and a common process scale. With `A = Q_CA_2D(dt,1,1)` in stored units before its common scale, the scale estimate is `lambda_hat = tr(A^-1 S_w)/6`; final `Q_hat=lambda_hat A`. This reduces estimation variance and does not fit unconstrained cross terms.

Measurement residuals were computed against the true training states, never from raw observation variance:

```text
v[t] = z[t] - h(x[t])
v[t,bearing] = wrap_to_pi(v[t,bearing])
S_v = centered_unbiased_covariance(v)
```

The raw covariance is retained for diagnostics. Since the generator independently draws range and bearing noise, final R is diagonal using the centered unbiased sample variances; residual means are recorded separately and are not applied as a bias correction.

## Estimated covariance and residual diagnostics

The raw full Q and R, shared-block Q, selected structured Q/R, residual means, counts, hashes, and eigenvalue checks are frozen in `estimated_QR_train_only.json`.

The selected training estimate is:

```text
Q_estimated =
[[4.98479925e-08, 1.24619981e-07, 1.66159975e-07, 0, 0, 0],
 [1.24619981e-07, 3.32319950e-07, 4.98479925e-07, 0, 0, 0],
 [1.66159975e-07, 4.98479925e-07, 9.96959850e-07, 0, 0, 0],
 [0, 0, 0, 4.98479925e-08, 1.24619981e-07, 1.66159975e-07],
 [0, 0, 0, 1.24619981e-07, 3.32319950e-07, 4.98479925e-07],
 [0, 0, 0, 1.66159975e-07, 4.98479925e-07, 9.96959850e-07]]

R_estimated = diag(8.93803476e-04 km², 3.06591779e-04 rad²)
```

The unconstrained raw full covariance estimates were:

```text
Q_raw =
[[4.96933646e-08, 1.24413270e-07, 1.66387388e-07, 1.65580109e-10, 5.67635223e-10, 1.04747029e-09],
 [1.24413270e-07, 3.32331658e-07, 5.00220300e-07, 3.95883317e-10, 1.33717840e-09, 2.35032739e-09],
 [1.66387388e-07, 5.00220300e-07, 1.00376594e-06, 3.27143675e-10, 1.36431260e-09, 1.68789557e-09],
 [1.65580109e-10, 3.95883317e-10, 3.27143675e-10, 4.97997806e-08, 1.24552760e-07, 1.66533076e-07],
 [5.67635223e-10, 1.33717840e-09, 1.36431260e-09, 1.24552760e-07, 3.32235242e-07, 4.99420050e-07],
 [1.04747029e-09, 2.35032739e-09, 1.68789557e-09, 1.66533076e-07, 4.99420050e-07, 9.99460755e-07]]

R_raw = [[8.93803476e-04, -3.50271466e-07],
         [-3.50271466e-07, 3.06591779e-04]]
```

Process residual bias in state order was `[-1.216e-6, -2.071e-6, 1.600e-7, 1.399e-6, 4.054e-6, 7.499e-6]` in corresponding state units. Measurement residual bias was `[7.418e-5 km, 4.319e-5 rad]`, or about `[0.0742 m, 0.00248 degrees]`. The raw range-bearing covariance corresponds to correlation `-0.000669`; it is negligible relative to the marginal variances and is set to zero in final R.

The final structured Q relative Frobenius error against oracle Q is `0.003040` (0.304%). Relative variance errors for final R are `-0.006885` for range and `+0.006481` for bearing. The pre-evaluation significant-discrepancy rule was 5% for Q Frobenius error or either absolute relative R variance error; it was not triggered. All selected covariances were finite and PSD.

## Fixed-test results

All eight runs use the same test file, initial states, 100 observations, ground truth, P0 records, state scaling, and `five_seed_harness.calculate_metrics`. Each oracle run reproduces its prior Task 3A/3B recorded metric values within `rtol=5e-4`, `atol=1e-5`. Estimated variants use only the frozen `estimated_QR_train_only` matrices. Metrics are position RMSE in m, velocity RMSE in m/s, acceleration RMSE in m/s², six-state MSE, and six-state MSE in dB.

| Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | MSE (dB) |
|---|---:|---:|---:|---:|---:|
| `legacy_ekf_oracle` | 809.7575 | 316.3859 | 184.8840 | 0.13166490 | -8.80530 |
| `calibrated_ekf_oracle` | 325.9908 | 66.0329 | 4.7767 | 0.01844219 | -17.34187 |
| `legacy_ukf_oracle` | 255,433.4531 | 134,598.3125 | 28,276.5410 | 14,027.08691 | 41.46967 |
| `calibrated_ukf_oracle` | 325.9494 | 66.0135 | 4.7762 | 0.01843727 | -17.34303 |
| `legacy_ekf_estimated_QR` | 63,105.8516 | 8,447.2656 | 876.5816 | 675.74536 | 28.29783 |
| `calibrated_ekf_estimated_QR` | 326.0442 | 66.0584 | 4.7769 | 0.01844856 | -17.34037 |
| `legacy_ukf_estimated_QR` | 255,398.1094 | 134,454.5000 | 28,245.9160 | 14,017.33984 | 41.46666 |
| `calibrated_ukf_estimated_QR` | 326.0004 | 66.0380 | 4.7764 | 0.01844336 | -17.34160 |

### Oracle versus estimated Q/R

| Variant pair | Δ position RMSE (m) | Δ velocity RMSE (m/s) | Δ acceleration RMSE (m/s²) | Δ six-state MSE | Δ MSE (dB) |
|---|---:|---:|---:|---:|---:|
| Legacy EKF, estimated − oracle | +62,296.09 | +8,130.88 | +691.70 | +675.61370 | +37.10313 |
| Calibrated EKF, estimated − oracle | +0.0534 | +0.0255 | +0.00017 | +0.00000637 | +0.00150 |
| Legacy UKF, estimated − oracle | -35.34 | -143.81 | -30.63 | -9.74707 | -0.00302 |
| Calibrated UKF, estimated − oracle | +0.0510 | +0.0245 | +0.00016 | +0.00000609 | +0.00143 |

The calibrated EKF and UKF are only marginally affected by replacing oracle Q/R with estimates. The legacy UKF remains dominated by the preserved broad legacy P0 and has extremely large errors under either Q/R policy. The legacy EKF reacts sharply to these small covariance changes: its estimated-Q/R score is orders of magnitude worse. Since estimates are close to oracle but the legacy EKF also shows transient covariance defects under both policies, this is evidence of legacy-path sensitivity/instability, not a clean measure of how an otherwise stable EKF depends on exact noise knowledge.

## Initialization and numerical stability

Each oracle/estimated pair reuses exactly the same P0: the legacy batch covariance from Task 3A or its training-calibrated 6×6 matrix. The calibrated initializer covariance and first-observation state/noise cross covariance were not recalibrated or changed for this task. Changing from legacy to calibrated initialization improves the calibrated-filter errors substantially under both Q/R policies; for example, six-state MSE falls from `0.1317` to `0.01844` for oracle EKF, and from `14,027` to `0.01844` for oracle UKF. This initialization effect is much larger than the calibrated filters' Q/R estimate effect.

Covariance snapshots were inspected after selected sequence steps. The legacy EKF warning flag uses relative symmetry error above `1e-6` or minimum eigenvalue below `-1e-6` of the largest absolute diagonal. Both legacy EKF policies trigger it. Oracle legacy EKF reaches relative symmetry error `1.64e-2` and relative minimum eigenvalue `-6.65e-3`; estimated legacy EKF reaches `8.69e-3` and `-5.22e-2`. Their covariance snapshots recover later, but these early violations are material and align with their sensitivity to small Q/R changes. Calibrated EKF snapshots remained symmetric to relative error below `2.5e-7` with positive minimum eigenvalues. UKF covariance validation remained finite, symmetric, and PSD at all recorded snapshots; its bounded Cholesky policy is described in the UKF validation report.

These diagnostics do not prove global numerical stability. The legacy EKF implementation was preserved; it uses float32 covariance operations and an explicit innovation inverse. No repair was made in this task.

## Reproducibility and outputs

- `estimated_QR_train_only.json`: frozen training-only estimates, raw and final covariance values, formulas, units, counts, residual diagnostics, oracle matrices, and training hashes.
- `estimated_noise_baseline_results.json`: eight metric/runtimes, covariance diagnostics, oracle reproduction checks, exact fixed-test and training provenance, configuration, Git and source hashes, and environment.
- `eight_variant_metrics.csv`: compact metrics/runtime table.
- Training file SHA-256: `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733`.
- Training state tensor SHA-256: `332c2764a958b36aa588ed6595d1e29cc3e5bcf038f5db540532f507e46d9f3d`.
- Training observations tensor SHA-256: `8069d8b02e6c6a06f9c420eafa0ce3a445fd9007a2b85bed66adcba73a60c91f`.
- Training initializer tensor SHA-256: `381b74fa88836817f72ba21499af29b4921d0dcbcd8c5195ab5fb4db4503edef`.
- Fixed test SHA-256: `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`.
- Evaluation: Python 3.12.3, PyTorch 2.4.0+cu121, CUDA 12.1 available, CPU filtering; full source hashes and Git status are in the JSON.

The evaluator refuses to create an existing output directory. It created only the new Task 4 directory. No model training, checkpoint changes, fixed-data regeneration, historical record edits, or commits were made.

At completion, `git diff --stat` for tracked unstaged files reports only the pre-existing `experiments/five_seed_harness.py` change (39 insertions, 17 deletions); the staged `docs/reports/project_explanation_detailed.pdf` and prior-task artifacts were already present. Task 4 added the estimator, evaluator, tests, report, and result bundle as untracked files; none were committed. The result JSON records the evaluation-time revision, dirty status, and run-time source hashes, plus final evaluator/test hashes after report metadata and oracle reproduction checks were added.

## Limitations and interpretation

1. The empirical residual estimates are close to oracle for this fixed simulator regime and train sample. Their uncertainty across other data seeds, noise scales, and trajectory regimes was not evaluated.
2. Q uses a structured integrated-white-jerk covariance because that is the simulator model. A full empirical Q is available in the frozen artifact; it has small sample cross-axis terms. The structured fit should not be transferred to a simulator with a different process model without rechecking assumptions.
3. R uses true-state-conditioned residuals. This is a training-supervised estimate; it is not available when only unlabeled observations are present. The centered covariance separates residual bias from variance and final R is diagonal based on the simulator's independent channel generation.
4. Small covariance estimation error caused severe degradation only for the legacy EKF with legacy P0, which also had PSD/symmetry problems. The calibrated filter results are more informative about noise-estimation impact here, but still represent one deterministic fixed test set.
5. Oracle Q/R is privileged simulator knowledge. Estimated Q/R is training-derived; neither result alone establishes deployment performance.
