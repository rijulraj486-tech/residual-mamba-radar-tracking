# Mamba Adaptive Acceleration Experiment v1

## Phase 1: preregistered design

### Question and protocol

This experiment tests whether an adaptive causal EMA can retain the acceleration benefit of fixed EMA smoothing while limiting its velocity and position costs. This is a hypothesis; no outcome is assumed. The fixed-pool synthetic regime, train/validation split, five seeds (11, 22, 33, 44, 55), Mamba architecture, M0 loss, optimizer, and checkpoint selection procedure remain those of `five_seed_n1024_fixedpool_v1`. The shared test set is reserved until a validation selection record has been written and hashed.

### Feature implementation boundary

The implementation will be isolated in `experiments/mamba_adaptive_acceleration_features.py`, with a separate training runner and evaluator. The historical A0 implementation and fixed-EMA/quadratic feature code remain untouched. Adaptive variants replace only coarse x/y acceleration channels (token indices 11 and 14). Input width remains 21, state output remains six dimensional, and the Mamba parameter count is expected to remain 619,014. Acceleration at indices 0 and 1 remains the supplied initial acceleration, as in A0 and the prior feature experiment.

### Causal reliability signal

Let measured Cartesian position be `p_t`, sample period `dt`, and smoothed velocity at the previous step be `s_(t-1)`. At `t >= 1`, form the one-step causal innovation

`e_t = p_t - (p_(t-1) + dt * s_(t-1))`.

This uses the current and historical measurements and the previous smoothed velocity only. It does not use target states, future observations, training labels, or test-set statistics. Let range and bearing observations be `r_t` and `theta_t`, with configured sensor standard deviations `sigma_r` and `sigma_theta`. The local Cartesian measurement covariance approximation is

`R_t = J_t diag(sigma_r^2, sigma_theta^2) J_t^T`,

where `J_t = [[cos(theta_t), -r_t sin(theta_t)], [sin(theta_t), r_t cos(theta_t)]]`. The range and its noise standard deviation are in the same scaled units as `p_t`; bearing is in radians. Define the normalized innovation energy

`q_t = ||e_t||_2^2 / max(trace(R_t + R_(t-1)), eps)`.

This is an online reliability proxy: low normalized innovation implies greater measurement consistency; high normalized innovation implies lower reliability. It can also respond to real maneuvers because innovation mixes measurement error and model mismatch. This confounding is explicit and will be considered when interpreting results. The denominator includes only current/past measurement covariance; it is not claimed to be a full Kalman innovation covariance.

### Adaptive coefficient

For each `t >= 1`, map the causal reliability proxy to an EMA coefficient:

`alpha_t = alpha_min + (alpha_max - alpha_min) / (1 + q_t / k)`.

Then update `s_t = alpha_t * v_t + (1 - alpha_t) * s_(t-1)`, where `v_t = (p_t-p_(t-1))/dt`. Coefficients are bounded by construction in `[alpha_min, alpha_max]`; lower reliability (larger `q_t`) gives stronger smoothing, while greater reliability gives faster adaptation. Set `s_0 = v_init`, preserve `a_0 = a_1 = a_init`, and for `t >= 2`, set `a_t = (s_t-s_(t-1))/dt`. This is dimensionally acceleration and uses no future sample.

Two adaptive configurations are predeclared, with no further sweep:

| Variant | `alpha_min` | `alpha_max` | `k` | Rationale |
|---|---:|---:|---:|---|
| A3 adaptive narrow | 0.25 | 0.50 | 2 | Stays within the prior fixed-EMA range and responds more strongly to normalized innovation. |
| A3 adaptive broad | 0.20 | 0.60 | 8 | Allows a wider response range and a gentler reliability mapping. |

Both are compared with original A0 and fixed EMA alpha 0.25 and 0.50. The existing five-seed checkpoints/results for those references will be reused; they will not be retrained or modified. Configuration selection uses only the five-seed validation summaries and a preregistered Pareto rule over position, velocity, and acceleration RMSE, with lower better. All nondominated adaptive settings will be evaluated after selection; baseline and fixed EMA references will be included in the final comparison. If no adaptive setting is on the validation Pareto frontier, that result will be recorded and no adaptive variant promoted.

### Training, validation, and held-out evaluation

Train only the two adaptive settings with seeds 11, 22, 33, 44, 55 and the original 819/204 train/validation pool. Preserve original M0: forward Mamba, 21 input channels, 619,014 parameters, unweighted six-state mean MSE, Adam (lr 0.001, weight decay 0.0001), batch size 16, 180 epoch cap, patience 25, min_delta 1e-6, and gradient clipping at 1.0. Select checkpoints by validation six-state MSE. Record the variant-selection decision and its SHA-256 before the test evaluator opens the shared test file. Test evaluation then includes A0, fixed EMA 0.25/0.50, and validation-selected adaptive settings.

Report grouped and per-axis physical RMSE, six-state MSE, paired per-seed changes, ten-step temporal RMSE, steps 1–10, adaptive coefficient summaries by time, training duration, and parameter count. Acceptance requires a consistent acceleration reduction across the five seeds without an unacceptable position or velocity regression; report the full trade-off and do not infer generalization beyond this synthetic regime.

### Verification plan

Tests will cover strict causality and future perturbation invariance, initialization and short sequences, boundaries, units and non-unit `dt`, finite/stable calculations, coefficient bounds, deterministic reproducibility, and the invariant that only the two coarse acceleration token channels change. The reliability calculation will be checked for its polar-to-Cartesian unit conversion and safe denominator. No test-set tensor will be loaded by training, tests, or validation selection.

## Phase 1 status

Phase 1 completed: the design was documented before implementation began.

## Phase 2–4: implementation, audit, selection, and held-out evaluation

### Implementation and verification

The adaptive variants are implemented separately in `experiments/mamba_adaptive_acceleration_features.py` and `experiments/mamba_adaptive_acceleration_v1.py`; the evaluator and post-selection analyzer are independent scripts. The frozen Mamba implementation and previous A0/fixed-feature source implementations were not changed. Both feature ablation test suites passed: **23 passed**. Compile checks passed for all experiment scripts.

The run audit found **10/10 adaptive seed runs complete** with valid checkpoint digests, validation histories, finite per-axis train/validation RMSE, and 619,014 parameters each. The broad seed 55 stopped after 86 epochs via configured early stopping; all other runs completed their cap or earlier configured stopping. Fixed-pool and preserved M0 checkpoint digests matched their manifests. Full audit details are in `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/artifact_audit.json`.

### Frozen validation selection

The validation-only Pareto frontier was frozen before opening the shared test set. Selection compared five-seed mean position, velocity, and acceleration RMSE using the preregistered strict-dominance rule. The frontier retained M0/A0, fixed EMA .25, and adaptive narrow; adaptive broad was dominated. Only adaptive narrow was retained for adaptive test evaluation.


| Validation candidate | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Pareto |
|---|---:|---:|---:|:---:|
| M0 / A0 | 329.79 | 79.07 | 64.25 | Yes |
| Fixed EMA .25 | 330.96 | 96.79 | 34.22 | Yes |
| Fixed EMA .50 | 329.29 | 83.59 | 58.42 | No |
| A3_adaptive_broad | 330.07 | 91.29 | 62.85 | No |
| Adaptive narrow | 328.68 | 82.57 | 52.74 | Yes |

Selection file SHA-256: `a52ee84fd63f7b992086853984603fb23470eb227d9f377b4381cca976ae6a03`. The test evaluator records this same hash and confirms it evaluated exactly M0, both fixed references, and the single selected adaptive setting. Shared test SHA-256: `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`.

### Shared test metrics

Values are mean ± sample standard deviation over the original five paired seeds. Position, velocity, and acceleration RMSE are in physical units (m, m/s, m/s²). Six-state MSE uses the benchmark’s 1000-scaled state units.


| Configuration | Position RMSE | Velocity RMSE | Acceleration RMSE | Six-state MSE |
|---|---:|---:|---:|---:|
| M0 / A0 | 332.08 ± 1.42 | 79.26 ± 2.33 | 66.71 ± 4.23 | 0.0202 ± 0.0002 |
| Fixed EMA .25 | 333.01 ± 2.05 | 94.85 ± 9.42 | 34.17 ± 2.65 | 0.0202 ± 0.0005 |
| Fixed EMA .50 | 331.54 ± 1.20 | 82.85 ± 6.00 | 58.39 ± 4.02 | 0.0200 ± 0.0002 |
| Adaptive narrow | 330.37 ± 1.14 | 81.74 ± 5.00 | 52.86 ± 1.70 | 0.0198 ± 0.0002 |

Per-axis RMSE, mean ± sample SD (physical units):


| Configuration | x (m) | vx (m/s) | ax (m/s²) | y (m) | vy (m/s) | ay (m/s²) |
|---|---:|---:|---:|---:|---:|---:|
| M0 / A0 | 239.42 ± 1.02 | 56.22 ± 1.63 | 51.29 ± 5.47 | 230.13 ± 1.20 | 55.84 ± 2.56 | 42.48 ± 2.71 |
| Fixed EMA .25 | 239.96 ± 1.78 | 68.87 ± 6.74 | 24.37 ± 1.77 | 230.90 ± 1.19 | 65.22 ± 6.61 | 23.95 ± 2.02 |
| Fixed EMA .50 | 239.04 ± 0.98 | 59.50 ± 4.45 | 41.87 ± 3.37 | 229.74 ± 0.97 | 57.65 ± 4.16 | 40.68 ± 2.42 |
| Adaptive narrow | 238.29 ± 0.75 | 58.81 ± 3.84 | 37.30 ± 1.40 | 228.83 ± 1.04 | 56.76 ± 3.42 | 37.45 ± 1.14 |

### Paired differences

Each entry is adaptive narrow minus the comparator (mean ± sample SD over seed-paired differences). Negative is better; `improved` counts seeds with a negative difference.


| Comparator | Position RMSE Δ (m) | Velocity RMSE Δ (m/s) | Acceleration RMSE Δ (m/s²) | Six-state MSE Δ |
|---|---:|---:|---:|---:|
| M0 / A0 | -1.71 ± 2.18 (4/5 improved) | 2.48 ± 5.02 (2/5 improved) | -13.86 ± 4.49 (5/5 improved) | -0.0004 ± 0.0001 (5/5 improved) |
| Fixed EMA .25 | -2.64 ± 2.45 (5/5 improved) | -13.12 ± 12.04 (5/5 improved) | 18.69 ± 3.14 (0/5 improved) | -0.0004 ± 0.0007 (4/5 improved) |
| Fixed EMA .50 | -1.17 ± 1.42 (4/5 improved) | -1.11 ± 4.49 (2/5 improved) | -5.53 ± 2.88 (5/5 improved) | -0.0003 ± 0.0003 (4/5 improved) |

For the acceleration/velocity trade-off: adaptive narrow reduced acceleration RMSE versus M0 in **5/5** seeds (mean −13.86 m/s²), while velocity RMSE improved in 2/5 and rose by 2.48 m/s on average; position improved in 4/5. Versus fixed EMA .50, acceleration improved in 5/5 (−5.53 m/s²), with position lower by 1.17 m and velocity lower by 1.11 m/s on average, though velocity improved in only 2/5. Versus fixed EMA .25, adaptive narrow improved velocity in 5/5 (−13.12 m/s) and position in 5/5, but acceleration was worse in all five (+18.69 m/s²). Thus it is a middle setting, not a replacement for the strongest acceleration result from fixed .25.

Paired per-seed group and per-axis differences, including all temporal-window paired differences, are preserved in `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/final_test_evaluation/paired_summary.json`.

Per-seed paired differences (adaptive narrow minus comparator; negative is better):


| Seed | Comparator | Δ position (m) | Δ velocity (m/s) | Δ acceleration (m/s²) | Δ six-state MSE |
|---:|---|---:|---:|---:|---:|
| 11 | M0 / A0 | -4.34 | +6.43 | -14.16 | -0.000592 |
| 11 | Fixed EMA .25 | -3.19 | -5.10 | +21.60 | -0.000186 |
| 11 | Fixed EMA .50 | -1.04 | +0.63 | -5.07 | -0.000193 |
| 22 | M0 / A0 | +1.22 | -2.74 | -13.31 | -0.000201 |
| 22 | Fixed EMA .25 | -1.13 | -5.82 | +20.55 | +0.000004 |
| 22 | Fixed EMA .50 | +1.20 | +4.03 | -7.75 | +0.000092 |
| 33 | M0 / A0 | -0.96 | -2.75 | -9.59 | -0.000356 |
| 33 | Fixed EMA .25 | -6.67 | -33.98 | +13.47 | -0.001582 |
| 33 | Fixed EMA .50 | -2.26 | -6.84 | -5.08 | -0.000520 |
| 44 | M0 / A0 | -3.33 | +7.91 | -11.02 | -0.000370 |
| 44 | Fixed EMA .25 | -0.57 | -12.78 | +18.55 | -0.000181 |
| 44 | Fixed EMA .50 | -2.19 | -4.67 | -8.55 | -0.000546 |
| 55 | M0 / A0 | -1.16 | +3.55 | -21.22 | -0.000471 |
| 55 | Fixed EMA .25 | -1.65 | -7.89 | +19.28 | -0.000138 |
| 55 | Fixed EMA .50 | -1.55 | +1.29 | -1.21 | -0.000158 |

### Temporal-window RMSE

Each cell is mean ± sample SD over paired seeds. Windows are one-based and ten steps wide. The first window includes preserved initialization acceleration at steps 1–2.


**Position RMSE (m)**


| Steps | M0 / A0 | Fixed EMA .25 | Fixed EMA .50 | Adaptive narrow |
|---|---:|---:|---:|---:|
| 1–10 | 365.50 ± 5.50 | 370.61 ± 3.69 | 367.45 ± 4.55 | 364.69 ± 4.40 |
| 11–20 | 306.16 ± 2.36 | 306.61 ± 1.89 | 304.86 ± 2.09 | 305.59 ± 2.89 |
| 21–30 | 293.30 ± 1.20 | 295.44 ± 1.05 | 293.43 ± 0.52 | 293.46 ± 1.99 |
| 31–40 | 306.76 ± 1.24 | 309.25 ± 3.84 | 307.17 ± 1.87 | 306.17 ± 1.46 |
| 41–50 | 304.94 ± 1.81 | 306.92 ± 4.09 | 304.78 ± 3.22 | 303.69 ± 2.56 |
| 51–60 | 307.86 ± 3.00 | 310.49 ± 2.24 | 308.15 ± 3.34 | 307.30 ± 2.83 |
| 61–70 | 321.08 ± 2.38 | 321.88 ± 1.89 | 321.78 ± 1.06 | 319.60 ± 2.41 |
| 71–80 | 335.05 ± 2.32 | 335.52 ± 2.43 | 335.78 ± 1.12 | 332.97 ± 0.89 |
| 81–90 | 354.25 ± 3.92 | 353.42 ± 3.23 | 353.76 ± 2.16 | 350.82 ± 2.98 |
| 91–100 | 408.22 ± 8.27 | 403.69 ± 5.36 | 401.80 ± 3.95 | 403.01 ± 5.10 |

**Velocity RMSE (m/s)**


| Steps | M0 / A0 | Fixed EMA .25 | Fixed EMA .50 | Adaptive narrow |
|---|---:|---:|---:|---:|
| 1–10 | 118.11 ± 3.30 | 122.69 ± 5.43 | 127.11 ± 2.71 | 119.08 ± 5.84 |
| 11–20 | 57.70 ± 2.15 | 71.34 ± 7.57 | 60.18 ± 5.06 | 58.52 ± 4.35 |
| 21–30 | 58.33 ± 1.42 | 72.81 ± 8.93 | 61.22 ± 5.51 | 60.96 ± 4.64 |
| 31–40 | 59.22 ± 2.20 | 73.87 ± 8.71 | 61.65 ± 5.80 | 62.63 ± 4.94 |
| 41–50 | 58.40 ± 3.32 | 72.75 ± 8.22 | 60.88 ± 5.85 | 61.92 ± 4.77 |
| 51–60 | 59.72 ± 3.21 | 76.14 ± 9.19 | 64.09 ± 5.90 | 64.91 ± 5.13 |
| 61–70 | 64.05 ± 3.08 | 80.57 ± 9.66 | 68.99 ± 6.35 | 69.95 ± 5.86 |
| 71–80 | 71.76 ± 3.78 | 90.65 ± 10.58 | 77.46 ± 6.70 | 78.20 ± 6.27 |
| 81–90 | 86.81 ± 3.97 | 105.17 ± 13.40 | 88.98 ± 8.99 | 88.34 ± 9.32 |
| 91–100 | 121.70 ± 5.88 | 148.45 ± 17.99 | 121.21 ± 13.81 | 120.70 ± 10.72 |

**Acceleration RMSE (m/s²)**


| Steps | M0 / A0 | Fixed EMA .25 | Fixed EMA .50 | Adaptive narrow |
|---|---:|---:|---:|---:|
| 1–10 | 66.89 ± 10.89 | 78.30 ± 1.67 | 125.79 ± 3.78 | 111.11 ± 2.92 |
| 11–20 | 39.06 ± 5.29 | 18.34 ± 2.99 | 29.54 ± 5.32 | 31.84 ± 2.33 |
| 21–30 | 39.03 ± 6.06 | 17.54 ± 3.06 | 29.84 ± 6.21 | 32.08 ± 2.01 |
| 31–40 | 40.31 ± 6.54 | 17.64 ± 3.06 | 31.10 ± 6.09 | 32.75 ± 1.84 |
| 41–50 | 42.53 ± 6.55 | 18.35 ± 2.95 | 32.51 ± 5.58 | 33.89 ± 1.97 |
| 51–60 | 47.44 ± 6.06 | 20.00 ± 3.31 | 35.28 ± 5.30 | 35.98 ± 1.93 |
| 61–70 | 53.49 ± 5.26 | 21.47 ± 3.33 | 38.43 ± 5.07 | 38.97 ± 1.90 |
| 71–80 | 67.75 ± 4.73 | 25.14 ± 3.56 | 46.54 ± 4.91 | 43.80 ± 1.89 |
| 81–90 | 84.96 ± 4.67 | 29.92 ± 4.49 | 56.08 ± 5.59 | 50.69 ± 2.07 |
| 91–100 | 128.32 ± 1.86 | 42.63 ± 5.63 | 79.54 ± 7.29 | 63.35 ± 2.27 |

**Six-state MSE by temporal window** (scaled-state units; mean ± sample SD):

| Steps | M0 / A0 | Fixed EMA .25 | Fixed EMA .50 | Adaptive narrow |
|---|---:|---:|---:|---:|
| 1–10 | 0.025357 ± 0.000911 | 0.026429 ± 0.000670 | 0.027838 ± 0.000583 | 0.026596 ± 0.000728 |
| 11–20 | 0.016437 ± 0.000266 | 0.016582 ± 0.000203 | 0.016246 ± 0.000120 | 0.016308 ± 0.000298 |
| 21–30 | 0.015164 ± 0.000128 | 0.015494 ± 0.000313 | 0.015132 ± 0.000153 | 0.015148 ± 0.000203 |
| 31–40 | 0.016545 ± 0.000187 | 0.016914 ± 0.000591 | 0.016530 ± 0.000297 | 0.016460 ± 0.000255 |
| 41–50 | 0.016376 ± 0.000284 | 0.016650 ± 0.000595 | 0.016286 ± 0.000347 | 0.016206 ± 0.000346 |
| 51–60 | 0.016773 ± 0.000423 | 0.017113 ± 0.000413 | 0.016728 ± 0.000222 | 0.016662 ± 0.000373 |
| 61–70 | 0.018348 ± 0.000330 | 0.018441 ± 0.000429 | 0.018306 ± 0.000237 | 0.018098 ± 0.000320 |
| 71–80 | 0.020339 ± 0.000253 | 0.020255 ± 0.000585 | 0.020162 ± 0.000282 | 0.019823 ± 0.000209 |
| 81–90 | 0.023382 ± 0.000484 | 0.022838 ± 0.000775 | 0.022717 ± 0.000557 | 0.022254 ± 0.000262 |
| 91–100 | 0.033001 ± 0.001253 | 0.031188 ± 0.001259 | 0.030445 ± 0.001286 | 0.030186 ± 0.000753 |

Early-sequence (steps 1–10) values are the first row above. Adaptive narrow has elevated early acceleration RMSE (111.11 ± 2.92 m/s²), driven in part by the mandated shared initial acceleration handling; its later windows are consistently below M0, but remain above fixed EMA .25. Ten-window six-state RMSE and MSE are also retained in the paired summary JSON.

### Adaptive coefficient and genuine acceleration-change diagnostic

Ten-step mean α across test observations over time was **0.432, 0.426, 0.426, 0.427, 0.426, 0.425, 0.426, 0.426, 0.427, 0.426** for steps 1–10 through 91–100, respectively.

Across test sequences, adaptive α had mean **0.427**, within-sequence/time pooled SD **0.061**, minimum **0.265**, and maximum **0.500** (bounds 0.25–0.50). Window-mean α stayed between **0.425** and **0.432**. The time-binned coefficient distribution is in the paired summary JSON.

To test for suppression of genuine changes, after selection I stratified test time points into tertiles by the norm of the consecutive true x/y acceleration change (ground truth used only in this post-selection diagnostic; it never enters the feature or model input). The low/middle/high cut points were approximately 0.901 and 1.483 m/s² per one-second sample interval. Adaptive α means were **0.4256 / 0.4260 / 0.4263** from low to high change. Since α rises slightly rather than falling in the high-change group, this diagnostic does **not** show stronger smoothing where true changes are larger. It does not establish that the feature never suppresses individual maneuvers; it says this test-set stratification provides no aggregate evidence of that failure mode.

Model acceleration RMSE in the same low/middle/high true-change bins was: M0 **47.46 / 46.95 / 45.63**, fixed EMA .25 **18.27 / 18.24 / 17.94**, fixed EMA .50 **32.32 / 32.05 / 31.32**, adaptive narrow **29.45 / 29.18 / 29.22 m/s²**. These nearly flat or slightly lower high-change errors offer no sign of a high-change error spike for adaptive narrow. Bins contain approximately 334.5k scalar acceleration entries each. See `acceleration_change_diagnostic.json` for bias and counts.

### Training cost and conclusion

Adaptive narrow training took **437.3 ± 2.7 seconds per seed** (sample SD), with **619,014 parameters**. Per-seed durations, epochs, and selected checkpoints are in the audit and run records.

The result supports a limited conclusion in this synthetic regime: the validation-selected adaptive setting improves acceleration over M0 and fixed EMA .50 without a large average position/velocity shift, and the effect is consistent for acceleration across all five seeds. It does not retain fixed EMA .25’s acceleration accuracy; its lower velocity error is the corresponding trade-off. There was no preregistered numeric cutoff for “unacceptable” position/velocity change, so the results support a **trade-off**, not an unqualified scientific acceptance claim. These results do not establish generalization beyond the tested synthetic data regime.


### Artifact map

- Frozen validation decision: `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/selection.json`
- Run and checkpoint audit: `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/artifact_audit.json`
- Per-run histories and metrics: `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/{variant}/seed_{seed}/run_metrics.json`
- Final test metrics and provenance: `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/final_test_evaluation/`
- Paired and temporal aggregation: `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/final_test_evaluation/paired_summary.json`
- Genuine-change diagnostic: `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/final_test_evaluation/acceleration_change_diagnostic.json`
