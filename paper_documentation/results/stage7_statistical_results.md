# Stage 7 paired seed statistics

This report reproduces the existing authoritative report text for convenient access. Original source: `artifacts/results/synthetic/paper_v4_long_sequence/stage7_statistical_analysis_v1/stage7_statistical_report.md`. Numerical tables are also indexed in `results_provenance_index.csv`.

# Stage 7: Statistical Analysis of Final Matched-Length Test Results

**Scope:** existing Stage 6 held-out results only. No inference was rerun; no model, checkpoint, normalization statistic, or dataset was changed.

## Provenance and pairing checks

Verified Stage 6 report SHA-256: `6bac932ec2ac7fc6ac8988edadd4b218ff4431b1f59b5b830df075f0f64b167c`. Verified Stage 5B selection manifest SHA-256: `9b478d6bf0727488742a56966b32762f054e50575e0b0d6bdbfeafb6e9dd9b14`.

All 20 Stage 6 evaluations passed hash checks against the Stage 5B selected checkpoint/config records, including M1 train-only feature statistics. For each of the ten same-length, same-seed M0/M1 pairs, Stage 5B records confirm matching initial trainable-weight hashes, initialization seeds, data-loader shuffle seeds, and loader-order hashes for every epoch both runs shared. Stage 6 records confirm identical test split file, manifest, and trajectory-ID-set hashes within each length; each test set contains 1,024 trajectories.

## Methods

The inferential unit is the training seed pair (n=5 per length), not the test trajectory. Differences are paired as M1 minus M0; percentage difference is 100 × (M1 − M0) / M0, so negative values favor M1. We report mean, sample SD, median, and all five paired observations.

The 95% confidence intervals are two-sided Student t intervals (df=4) over the five paired seed differences; percentage-difference intervals apply the same calculation to the five per-seed percentages. These intervals describe uncertainty in the mean paired effect across seed-level training replicates under the t model. Cohen dz is the mean paired difference divided by the sample SD of paired differences; Hedges-corrected gz applies J(df=4). Paired t-tests and exact two-sided Wilcoxon signed-rank sign-flip tests are exploratory. The exact Wilcoxon p-value enumerates all 2^5 sign assignments using average ranks for ties. Holm adjustment is reported separately across the eight length-by-metric comparisons for each test family.

With five seed pairs, distributional assumptions cannot be checked meaningfully, estimates are sensitive to individual seeds, and power is low. P-values and intervals are exploratory and should not be treated as definitive evidence. No significance claim is based on effect magnitude.

## Paired seed summary

Paired differences are M1 − M0; negative values favor M1.

| T | Metric | M0 mean ± SD | M1 mean ± SD | Mean paired difference ± SD | Median difference | Mean paired % difference ± SD | Seeds improved |
|---:|---|---:|---:|---:|---:|---:|---:|
| 200 | position_rmse_m | 598.58 ± 23.612 | 594.03 ± 8.1409 | -4.545 ± 21.67 m | 0.81079 | -0.65791% ± 3.448% | 2/5 |
| 200 | velocity_rmse_mps | 221.17 ± 18.36 | 130.64 ± 9.3574 | -90.538 ± 26.829 m/s | -74.683 | -40.417% ± 8.1581% | 5/5 |
| 200 | acceleration_rmse_mps2 | 346.4 ± 14.563 | 153.44 ± 13.929 | -192.97 ± 24.451 m/s² | -202.42 | -55.579% ± 5.249% | 5/5 |
| 200 | six_state_mse_scaled_units | 0.088016 ± 0.0074032 | 0.065627 ± 0.0016265 | -0.022389 ± 0.007742 scaled-state units | -0.021148 | -25.039% ± 6.2378% | 5/5 |
| 400 | position_rmse_m | 2754.7 ± 121.58 | 2232.6 ± 18.295 | -522.06 ± 112.67 m | -563.19 | -18.835% ± 3.3519% | 5/5 |
| 400 | velocity_rmse_mps | 1960.6 ± 178.63 | 494.72 ± 61.109 | -1465.9 ± 224.64 m/s | -1547.7 | -74.398% ± 5.5945% | 5/5 |
| 400 | acceleration_rmse_mps2 | 2022.3 ± 181.15 | 733.48 ± 89.562 | -1288.9 ± 185.48 m/s² | -1349.6 | -63.552% ± 5.309% | 5/5 |
| 400 | six_state_mse_scaled_units | 2.5976 ± 0.31737 | 0.96283 ± 0.01995 | -1.6348 ± 0.32474 scaled-state units | -1.7366 | -62.412% ± 5.4603% | 5/5 |

## Five paired seed results

All M0, M1, absolute, and percentage results are shown below. Percentage differences are relative to that seed's M0 result.

### T=200

| Metric | Seed | M0 | M1 | M1 − M0 | Difference (%) |
|---|---:|---:|---:|---:|---:|
| position_rmse_m | 11 | 584.78 | 590.28 | 5.498 | 0.9402% |
| position_rmse_m | 22 | 595.44 | 591.38 | -4.0594 | -0.6817% |
| position_rmse_m | 33 | 583.49 | 584.3 | 0.81079 | 0.139% |
| position_rmse_m | 44 | 639.98 | 599.01 | -40.969 | -6.402% |
| position_rmse_m | 55 | 589.2 | 605.19 | 15.995 | 2.715% |
| velocity_rmse_mps | 11 | 217.49 | 124.11 | -93.384 | -42.94% |
| velocity_rmse_mps | 22 | 210.12 | 136.1 | -74.015 | -35.23% |
| velocity_rmse_mps | 33 | 213.26 | 138.58 | -74.683 | -35.02% |
| velocity_rmse_mps | 44 | 253.63 | 117.42 | -136.21 | -53.7% |
| velocity_rmse_mps | 55 | 211.36 | 136.96 | -74.396 | -35.2% |
| acceleration_rmse_mps2 | 11 | 360.56 | 155.46 | -205.11 | -56.89% |
| acceleration_rmse_mps2 | 22 | 339.44 | 137.02 | -202.42 | -59.63% |
| acceleration_rmse_mps2 | 33 | 331.08 | 171.75 | -159.33 | -48.12% |
| acceleration_rmse_mps2 | 44 | 363.34 | 142.55 | -220.79 | -60.77% |
| acceleration_rmse_mps2 | 55 | 337.59 | 160.41 | -177.18 | -52.48% |
| six_state_mse_scaled_units | 11 | 0.086547 | 0.064667 | -0.02188 | -25.28% |
| six_state_mse_scaled_units | 22 | 0.085653 | 0.064505 | -0.021148 | -24.69% |
| six_state_mse_scaled_units | 33 | 0.082593 | 0.065019 | -0.017574 | -21.28% |
| six_state_mse_scaled_units | 44 | 0.10099 | 0.065488 | -0.0355 | -35.15% |
| six_state_mse_scaled_units | 55 | 0.084299 | 0.068458 | -0.015841 | -18.79% |

### T=400

| Metric | Seed | M0 | M1 | M1 − M0 | Difference (%) |
|---|---:|---:|---:|---:|---:|
| position_rmse_m | 11 | 2777.5 | 2214.3 | -563.19 | -20.28% |
| position_rmse_m | 22 | 2718.5 | 2249.9 | -468.59 | -17.24% |
| position_rmse_m | 33 | 2565.8 | 2212 | -353.8 | -13.79% |
| position_rmse_m | 44 | 2879.6 | 2238.9 | -640.71 | -22.25% |
| position_rmse_m | 55 | 2832.1 | 2248.1 | -583.99 | -20.62% |
| velocity_rmse_mps | 11 | 2084.6 | 512.29 | -1572.3 | -75.42% |
| velocity_rmse_mps | 22 | 1967.3 | 448.77 | -1518.5 | -77.19% |
| velocity_rmse_mps | 33 | 1650.6 | 580.87 | -1069.8 | -64.81% |
| velocity_rmse_mps | 44 | 2054.9 | 507.2 | -1547.7 | -75.32% |
| velocity_rmse_mps | 55 | 2045.8 | 424.45 | -1621.3 | -79.25% |
| acceleration_rmse_mps2 | 11 | 2217.8 | 793.53 | -1424.3 | -64.22% |
| acceleration_rmse_mps2 | 22 | 1895.5 | 688.8 | -1206.7 | -63.66% |
| acceleration_rmse_mps2 | 33 | 1833.4 | 827.71 | -1005.7 | -54.85% |
| acceleration_rmse_mps2 | 44 | 2212.9 | 754.82 | -1458.1 | -65.89% |
| acceleration_rmse_mps2 | 55 | 1952.1 | 602.53 | -1349.6 | -69.13% |
| six_state_mse_scaled_units | 11 | 2.8298 | 0.96587 | -1.8639 | -65.87% |
| six_state_mse_scaled_units | 22 | 2.4755 | 0.9563 | -1.5192 | -61.37% |
| six_state_mse_scaled_units | 33 | 2.1115 | 0.98588 | -1.1256 | -53.31% |
| six_state_mse_scaled_units | 44 | 2.902 | 0.97329 | -1.9287 | -66.46% |
| six_state_mse_scaled_units | 55 | 2.6694 | 0.93284 | -1.7366 | -65.05% |

## Confidence intervals and effect sizes

CIs are for the mean paired difference unless specified. Intervals crossing zero do not establish equivalence; intervals excluding zero are still exploratory at n=5 and across multiple endpoints.

| T | Metric | Mean difference 95% CI | Mean percent difference 95% CI | Cohen dz | Hedges gz |
|---:|---|---:|---:|---:|---:|
| 200 | position_rmse_m | [-31.452, 22.362] m | [-4.939%, 3.623%] | -0.20974 | -0.16779 |
| 200 | velocity_rmse_mps | [-123.85, -57.226] m/s | [-50.55%, -30.29%] | -3.3746 | -2.6997 |
| 200 | acceleration_rmse_mps2 | [-223.33, -162.61] m/s² | [-62.1%, -49.06%] | -7.8918 | -6.3134 |
| 200 | six_state_mse_scaled_units | [-0.032002, -0.012776] scaled-state units | [-32.78%, -17.29%] | -2.8918 | -2.3135 |
| 400 | position_rmse_m | [-661.95, -382.16] m | [-23%, -14.67%] | -4.6336 | -3.7069 |
| 400 | velocity_rmse_mps | [-1744.8, -1187] m/s | [-81.34%, -67.45%] | -6.5257 | -5.2206 |
| 400 | acceleration_rmse_mps2 | [-1519.2, -1058.6] m/s² | [-70.14%, -56.96%] | -6.949 | -5.5592 |
| 400 | six_state_mse_scaled_units | [-2.038, -1.2316] scaled-state units | [-69.19%, -55.63%] | -5.0342 | -4.0274 |

## Exploratory paired tests

Two-sided p-values are shown raw and Holm-adjusted within test family across the eight metric-by-length comparisons.

| T | Metric | Paired t | t p | t Holm p | Wilcoxon W | Exact Wilcoxon p | Wilcoxon Holm p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 200 | position_rmse_m | -0.469 (df=4) | 0.663502 | 0.663502 | 7 | 1 | 1 |
| 200 | velocity_rmse_mps | -7.546 (df=4) | 0.00165232 | 0.00495695 | 0 | 0.0625 | 0.5 |
| 200 | acceleration_rmse_mps2 | -17.65 (df=4) | 6.05713e-05 | 0.00048457 | 0 | 0.0625 | 0.5 |
| 200 | six_state_mse_scaled_units | -6.466 (df=4) | 0.00294616 | 0.00589231 | 0 | 0.0625 | 0.5 |
| 400 | position_rmse_m | -10.36 (df=4) | 0.000489832 | 0.00195933 | 0 | 0.0625 | 0.5 |
| 400 | velocity_rmse_mps | -14.59 (df=4) | 0.0001283 | 0.000769802 | 0 | 0.0625 | 0.5 |
| 400 | acceleration_rmse_mps2 | -15.54 (df=4) | 0.000100144 | 0.00070101 | 0 | 0.0625 | 0.5 |
| 400 | six_state_mse_scaled_units | -11.26 (df=4) | 0.000354788 | 0.00177394 | 0 | 0.0625 | 0.5 |

## Trajectory-level inference

Stage 6 retained per-checkpoint aggregate metrics and 10-step temporal-window metrics, but no per-trajectory predictions, residuals, or squared-error contributions. Identical test trajectory sets are provenance-confirmed, but paired trajectory-level outcomes cannot be calculated from the retained aggregates. A hierarchical bootstrap resampling seeds and trajectories would therefore be unsupported; no trajectory bootstrap was performed and no `trajectory_bootstrap_results.csv` is emitted.

The additional artifact needed is per-trajectory predictions or, for these metrics, per-trajectory sufficient error contributions by checkpoint: position/velocity/acceleration squared-error sums and counts, plus six-state squared-error sums and counts, keyed by trajectory ID and checkpoint. This would allow paired trajectory resampling nested within seed resampling while preserving the M0/M1 pairing.

## Interpretation

- At T=200, M1 improves by seed in: position_rmse_m 2/5, velocity_rmse_mps 5/5, acceleration_rmse_mps2 5/5, six_state_mse_scaled_units 5/5.
- At T=400, M1 improves by seed in: position_rmse_m 5/5, velocity_rmse_mps 5/5, acceleration_rmse_mps2 5/5, six_state_mse_scaled_units 5/5.
- At T=200, position RMSE is lower for M1 in 2/5 pairs; the mean paired difference is −4.54 m (95% t CI [−31.45, 22.36]) and the mean percentage difference is −0.66% (95% t CI [−4.94%, 3.62%]). This is compatible with either a modest improvement or deterioration, so position preservation/equivalence is not established. At T=400, position RMSE is lower in all five pairs (mean difference −522.06 m; 95% t CI [−661.95, −382.16]); this remains an exploratory seed-level result.
- Mean paired percentage improvements are larger at T=400 for all four endpoints: position 18.83% vs 0.66%, velocity 74.40% vs 40.42%, acceleration 63.55% vs 55.58%, and six-state MSE 62.41% vs 25.04%. This cross-length comparison is descriptive; no cross-length inferential test was specified.
- Paired t-tests with Holm adjustment are below 0.05 for velocity, acceleration, and six-state MSE at T=200 and all four metrics at T=400; T=200 position is not. However, exact Wilcoxon p-values are at least 0.0625 (the minimum attainable two-sided value with five nonzero pairs), and no Wilcoxon result is below 0.05 even before adjustment. Treat the t-test results as model-dependent exploratory evidence, not definitive significance. Shared initialization and loader order make the seed contrasts paired, but five pairs do not support strong population-level conclusions.

## Figures

- `figures/paired_position_rmse_m.png`
- `figures/paired_velocity_rmse_mps.png`
- `figures/paired_acceleration_rmse_mps2.png`
- `figures/paired_six_state_mse_scaled_units.png`
- `figures/summary_m1_improvement_by_length.png`

## Limitations

Only five paired training seeds are available. Per-trajectory Stage 6 outputs were not retained, so uncertainty over the held-out trajectory population cannot be estimated while respecting the seed/trajectory hierarchy. The seed-level confidence intervals represent seed-replicate uncertainty conditional on the fixed Stage 2 test sets; they do not represent uncertainty from drawing a new test set. All tests are exploratory, with low power and multiple endpoints. No model selection or tuning was performed from these test results.
