# Mamba Acceleration Feature Ablation v1

## Executive summary

All 20 trained variant/seed runs completed successfully. Every checkpoint loaded, matched its recorded SHA-256, had a history matching the recorded epoch count, and selected the minimum validation six-state MSE in its history. No run was restarted. Validation-only Pareto selection retained all four feature variants; the shared test set was read only after `selection.json` recorded that decision.

Causal smoothing greatly reduced acceleration RMSE for α=0.25 and the 9-sample quadratic fit, but both raised velocity RMSE. All four alternatives also had worse acceleration RMSE than M0 during steps 1–10, before their later-window gains emerged. Across five seeds, validation acceleration RMSE fell from 64.25 to 34.22 m/s² with EMA α=0.25, while velocity RMSE rose from 79.07 to 96.79 m/s. Test results show the same trade-off. EMA α=0.50 gave a smaller acceleration gain and a smaller velocity penalty. Position RMSE changed little for all configurations.

No feature is an unambiguous improvement across position, velocity, and acceleration. The strongest acceleration option is EMA α=0.25, while α=0.50 has the least adverse velocity trade-off among variants with improved acceleration. These are five-seed descriptive results, not a statistical significance claim.

## Methodology and controls

- Seeds: 11, 22, 33, 44, 55. Existing fixed train/validation pool: 819/204 sequences, length 100. Shared test: 1,024 sequences, length 100.
- A0 uses the historical M0 checkpoints and original coarse second-difference feature. A1/A2 each train new checkpoints; architecture, 21-channel input shape, original unweighted six-state mean MSE, Adam, optimizer settings, seed set, batch size, early stopping, and gradient clipping remain fixed.
- Training configuration: 180 epoch cap; batch 16; Adam lr=0.001, weight decay=0.0001; patience=25, min_delta=1e-6; clip global gradient norm at 1.0; no learning-rate scheduler; embed=128, hidden=256, 3 layers, d_state=16, d_conv=4, expand=2, dropout=0.08, forward direction. Best checkpoint is selected by validation six-state MSE.
- Model selection: nondominated Pareto frontier of five-seed mean validation position, velocity, and acceleration RMSE, with lower better. A strict dominance requires no worse in each group and strictly better in at least one. M0 served as reference, not part of choosing among feature settings. All four settings were nondominated and thus evaluated on test.
- No loss weighting, input normalization, smoothing of outputs, innovation conditioning, or architecture change was introduced. No test-set result was used to select a feature.

The physical RMSEs are calculated after scaling state units by 1,000. For group RMSE, the squared errors of the two axes are summed per sample/time then averaged and square-rooted. Six-state MSE is the elementwise mean in the model’s scaled state units. Axis RMSEs are reported separately. Temporal windows are contiguous ten-step chunks.

### Feature equations and time alignment

The existing coarse velocity uses the supplied initial velocity at index 0 and the Cartesian position difference `(p_t - p_{t-1})/dt` from index 1 onward. Existing coarse acceleration copies supplied initial acceleration at indices 0 and 1, then differences consecutive coarse velocities for `t ≥ 2`. The feature variants replace only the x/y acceleration coordinates in the coarse-state token.

- **A0:** original acceleration second difference of Cartesian measured positions.
- **A1 EMA:** let `v_0 = v_init`, and `v_t = (p_t-p_{t-1})/dt` for `t≥1`. Initialize `s_0=v_init`; for `t≥1`, `s_t=αv_t+(1-α)s_{t-1}`. Preserve `a_0=a_1=a_init`; for `t≥2`, `a_t=(s_t-s_{t-1})/dt`. Configurations: α=0.25 and 0.50.
- **A2 trailing quadratic:** for `t≥2`, fit `p(u)=c_0+c_1u+c_2u²` by least squares to the trailing `min(W,t+1)` measured Cartesian positions, with integer-step offsets `u=1-n,…,0` relative to the current sample. Output `a_t=2c_2/dt²`; preserve `a_0=a_1=a_init`. Configurations: W=5 and W=9.

The EMA derivative at index t uses the smoothed velocity ending at t and its predecessor; the quadratic fit is evaluated at its trailing-window endpoint t. Thus both are aligned to the current position index and use no future sample. Startup handling is explicit: the first two acceleration features are the supplied initial acceleration; the quadratic fit begins at t=2 with three samples and grows its window to W.

## Validation results

### Combined-state metrics

| Configuration | Position RMSE (m) | Δ vs M0 | Velocity RMSE (m/s) | Δ vs M0 | Acceleration RMSE (m/s²) | Δ vs M0 | Six-state MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 existing second difference | 329.79 ± 1.64 | 0.00 ± 0.00 | 79.07 ± 2.41 | 0.00 ± 0.00 | 64.25 ± 4.20 | 0.00 ± 0.00 | 0.0199 ± 0.0003 |
| A1 EMA α=0.25 | 330.96 ± 1.43 | 1.17 ± 3.02 | 96.79 ± 9.15 | 17.72 ± 10.07 | 34.22 ± 2.55 | -30.02 ± 6.16 | 0.0200 ± 0.0004 |
| A1 EMA α=0.50 | 329.29 ± 0.90 | -0.50 ± 2.17 | 83.59 ± 5.66 | 4.52 ± 6.39 | 58.42 ± 4.16 | -5.83 ± 7.21 | 0.0198 ± 0.0003 |
| A2 quadratic, window 5 | 329.41 ± 2.18 | -0.38 ± 2.61 | 93.36 ± 15.91 | 14.29 ± 17.02 | 49.37 ± 5.50 | -14.88 ± 6.16 | 0.0200 ± 0.0008 |
| A2 quadratic, window 9 | 329.00 ± 1.27 | -0.79 ± 2.28 | 94.94 ± 9.19 | 15.87 ± 7.86 | 35.99 ± 7.13 | -28.25 ± 9.57 | 0.0198 ± 0.0005 |

### Six state-coordinate RMSE (physical units; mean ± sample SD across seeds)

| Configuration | x (m) | vx (m/s) | ax (m/s²) | y (m) | vy (m/s) | ay (m/s²) |
|---|---:|---:|---:|---:|---:|---:|
| A0 existing second difference | 228.48 ± 0.75 | 55.44 ± 2.30 | 48.41 ± 5.61 | 237.82 ± 1.61 | 56.33 ± 2.49 | 42.08 ± 2.00 |
| A1 EMA α=0.25 | 229.42 ± 1.24 | 69.26 ± 7.18 | 24.28 ± 1.82 | 238.54 ± 1.26 | 67.60 ± 5.81 | 24.12 ± 1.81 |
| A1 EMA α=0.50 | 228.65 ± 0.55 | 58.81 ± 4.91 | 41.61 ± 3.92 | 236.97 ± 0.96 | 59.38 ± 3.27 | 40.98 ± 2.05 |
| A2 quadratic, window 5 | 228.57 ± 1.71 | 66.59 ± 13.23 | 34.89 ± 5.10 | 237.21 ± 1.52 | 65.39 ± 9.28 | 34.88 ± 3.01 |
| A2 quadratic, window 9 | 228.37 ± 0.68 | 68.01 ± 7.10 | 24.68 ± 5.56 | 236.83 ± 1.30 | 66.23 ± 6.15 | 26.18 ± 4.58 |

### Per-seed validation paired differences

Each cell is `variant RMSE (variant − M0)`; negative differences are improvements. Units are m, m/s, and m/s² for position, velocity, and acceleration respectively. Six-state MSE is shown as `variant (paired difference)`, in scaled-state units.

| Variant | Seed | Position | Velocity | Acceleration | Six-state MSE | Best epoch | Train duration (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 EMA α=0.25 | 11 | 329.17 (-2.70) | 93.48 (+11.62) | 33.63 (-34.27) | 0.01970 (-0.00054) | 180 | 176.4 |
| A1 EMA α=0.25 | 22 | 331.75 (+3.77) | 89.52 (+8.18) | 32.41 (-28.94) | 0.01985 (+0.00019) | 169 | 175.8 |
| A1 EMA α=0.25 | 33 | 332.93 (+4.64) | 111.12 (+32.77) | 38.23 (-21.94) | 0.02077 (+0.00119) | 81 | 103.5 |
| A1 EMA α=0.25 | 44 | 330.59 (+0.43) | 100.33 (+23.06) | 35.02 (-27.25) | 0.02010 (+0.00029) | 137 | 157.2 |
| A1 EMA α=0.25 | 55 | 330.38 (-0.28) | 89.48 (+12.97) | 31.83 (-37.72) | 0.01969 (-0.00031) | 156 | 166.6 |
| A1 EMA α=0.50 | 11 | 328.24 (-3.63) | 87.14 (+5.28) | 60.24 (-7.66) | 0.01983 (-0.00041) | 147 | 158.5 |
| A1 EMA α=0.50 | 22 | 329.69 (+1.71) | 77.34 (-4.00) | 60.06 (-1.29) | 0.01971 (+0.00006) | 169 | 166.2 |
| A1 EMA α=0.50 | 33 | 329.06 (+0.78) | 83.43 (+5.08) | 56.79 (-3.38) | 0.01974 (+0.00016) | 111 | 124.9 |
| A1 EMA α=0.50 | 44 | 330.62 (+0.46) | 91.04 (+13.77) | 62.92 (+0.65) | 0.02026 (+0.00045) | 151 | 162.4 |
| A1 EMA α=0.50 | 55 | 328.85 (-1.80) | 79.00 (+2.48) | 52.07 (-17.49) | 0.01952 (-0.00049) | 156 | 166.8 |
| A2 quadratic, window 5 | 11 | 328.63 (-3.24) | 90.96 (+9.10) | 44.70 (-23.20) | 0.01971 (-0.00053) | 180 | 166.1 |
| A2 quadratic, window 5 | 22 | 330.53 (+2.55) | 89.74 (+8.40) | 47.20 (-14.14) | 0.01992 (+0.00026) | 169 | 166.5 |
| A2 quadratic, window 5 | 33 | 327.56 (-0.72) | 79.45 (+1.10) | 44.53 (-15.64) | 0.01927 (-0.00032) | 180 | 168.1 |
| A2 quadratic, window 5 | 44 | 327.64 (-2.52) | 85.99 (+8.71) | 56.40 (-5.87) | 0.01965 (-0.00016) | 141 | 153.6 |
| A2 quadratic, window 5 | 55 | 332.67 (+2.02) | 120.67 (+44.15) | 54.03 (-15.52) | 0.02136 (+0.00135) | 89 | 106.4 |
| A2 quadratic, window 9 | 11 | 329.09 (-2.78) | 92.95 (+11.09) | 32.58 (-35.32) | 0.01967 (-0.00057) | 147 | 159.2 |
| A2 quadratic, window 9 | 22 | 331.00 (+3.02) | 111.10 (+29.76) | 48.44 (-12.91) | 0.02071 (+0.00105) | 96 | 113.1 |
| A2 quadratic, window 9 | 33 | 327.57 (-0.71) | 91.89 (+13.54) | 34.87 (-25.30) | 0.01949 (-0.00009) | 136 | 149.9 |
| A2 quadratic, window 9 | 44 | 328.95 (-1.21) | 88.53 (+11.25) | 30.52 (-31.75) | 0.01950 (-0.00031) | 163 | 164.9 |
| A2 quadratic, window 9 | 55 | 328.40 (-2.26) | 90.24 (+13.72) | 33.57 (-35.98) | 0.01952 (-0.00049) | 171 | 166.7 |

### Validation checkpoint losses by state

The training records retain epoch-wise per-coordinate training and validation MSE/RMSE histories. The table below reports per-coordinate MSE at each selected checkpoint, averaged over five seeds. Values are scaled-state MSE.

| Configuration | x train/val | vx train/val | ax train/val | y train/val | vy train/val | ay train/val |
|---|---:|---:|---:|---:|---:|---:|
| A0 existing second difference | 0.05478/0.05220 | 0.00284/0.00308 | 0.00239/0.00237 | 0.05114/0.05656 | 0.00303/0.00318 | 0.00173/0.00177 |
| A1 EMA α=0.25 | 0.05535/0.05263 | 0.00432/0.00484 | 0.00057/0.00059 | 0.05163/0.05690 | 0.00429/0.00460 | 0.00058/0.00058 |
| A1 EMA α=0.50 | 0.05498/0.05228 | 0.00328/0.00348 | 0.00168/0.00174 | 0.05104/0.05616 | 0.00326/0.00353 | 0.00166/0.00168 |
| A2 quadratic, window 5 | 0.05494/0.05224 | 0.00398/0.00457 | 0.00110/0.00124 | 0.05094/0.05627 | 0.00407/0.00435 | 0.00115/0.00122 |
| A2 quadratic, window 9 | 0.05478/0.05215 | 0.00413/0.00467 | 0.00069/0.00063 | 0.05085/0.05609 | 0.00420/0.00442 | 0.00065/0.00070 |

### Validation temporal windows

Mean group RMSE across seeds for each ten-step interval:

| Steps | Configuration | Position | Velocity | Acceleration | Δ acceleration vs M0 |
|---|---|---:|---:|---:|---:|
| 1–10 | A0 existing second difference | 381.15 | 119.18 | 66.04 | +0.00 |
| 1–10 | A1 EMA α=0.25 | 389.15 | 126.29 | 76.78 | +10.74 |
| 1–10 | A1 EMA α=0.50 | 383.81 | 129.40 | 123.42 | +57.39 |
| 1–10 | A2 quadratic, window 5 | 384.68 | 121.04 | 79.88 | +13.84 |
| 1–10 | A2 quadratic, window 9 | 383.87 | 120.76 | 76.09 | +10.05 |
| 11–20 | A0 existing second difference | 321.30 | 58.14 | 38.26 | +0.00 |
| 11–20 | A1 EMA α=0.25 | 322.91 | 71.71 | 18.21 | -20.05 |
| 11–20 | A1 EMA α=0.50 | 319.76 | 61.21 | 29.14 | -9.12 |
| 11–20 | A2 quadratic, window 5 | 320.79 | 70.13 | 31.61 | -6.66 |
| 11–20 | A2 quadratic, window 9 | 318.81 | 70.86 | 18.49 | -19.77 |
| 21–30 | A0 existing second difference | 303.96 | 58.73 | 39.97 | +0.00 |
| 21–30 | A1 EMA α=0.25 | 306.72 | 73.49 | 17.69 | -22.28 |
| 21–30 | A1 EMA α=0.50 | 303.81 | 62.42 | 30.26 | -9.71 |
| 21–30 | A2 quadratic, window 5 | 304.38 | 69.36 | 30.71 | -9.26 |
| 21–30 | A2 quadratic, window 9 | 304.14 | 73.76 | 18.47 | -21.49 |
| 31–40 | A0 existing second difference | 301.27 | 57.37 | 38.78 | +0.00 |
| 31–40 | A1 EMA α=0.25 | 303.61 | 71.12 | 17.39 | -21.39 |
| 31–40 | A1 EMA α=0.50 | 303.01 | 60.70 | 30.48 | -8.31 |
| 31–40 | A2 quadratic, window 5 | 300.33 | 67.90 | 29.10 | -9.68 |
| 31–40 | A2 quadratic, window 9 | 302.23 | 70.72 | 19.03 | -19.76 |
| 41–50 | A0 existing second difference | 303.76 | 56.48 | 41.61 | +0.00 |
| 41–50 | A1 EMA α=0.25 | 303.24 | 71.73 | 18.60 | -23.01 |
| 41–50 | A1 EMA α=0.50 | 304.18 | 60.13 | 32.90 | -8.72 |
| 41–50 | A2 quadratic, window 5 | 303.27 | 68.76 | 30.04 | -11.57 |
| 41–50 | A2 quadratic, window 9 | 304.31 | 71.57 | 20.24 | -21.38 |
| 51–60 | A0 existing second difference | 298.96 | 58.66 | 43.84 | +0.00 |
| 51–60 | A1 EMA α=0.25 | 299.86 | 74.85 | 19.65 | -24.19 |
| 51–60 | A1 EMA α=0.50 | 300.56 | 63.48 | 34.79 | -9.05 |
| 51–60 | A2 quadratic, window 5 | 299.72 | 72.62 | 31.22 | -12.62 |
| 51–60 | A2 quadratic, window 9 | 299.49 | 74.51 | 22.90 | -20.94 |
| 61–70 | A0 existing second difference | 315.01 | 63.70 | 52.87 | +0.00 |
| 61–70 | A1 EMA α=0.25 | 316.79 | 82.17 | 22.24 | -30.63 |
| 61–70 | A1 EMA α=0.50 | 317.48 | 69.04 | 39.17 | -13.69 |
| 61–70 | A2 quadratic, window 5 | 316.65 | 78.07 | 35.39 | -17.48 |
| 61–70 | A2 quadratic, window 9 | 317.58 | 79.91 | 25.81 | -27.05 |
| 71–80 | A0 existing second difference | 317.55 | 75.71 | 70.37 | +0.00 |
| 71–80 | A1 EMA α=0.25 | 317.47 | 98.73 | 27.41 | -42.95 |
| 71–80 | A1 EMA α=0.50 | 315.01 | 80.53 | 49.66 | -20.70 |
| 71–80 | A2 quadratic, window 5 | 316.22 | 94.29 | 42.55 | -27.81 |
| 71–80 | A2 quadratic, window 9 | 314.84 | 95.15 | 30.09 | -40.27 |
| 81–90 | A0 existing second difference | 353.59 | 91.86 | 86.23 | +0.00 |
| 81–90 | A1 EMA α=0.25 | 350.01 | 113.64 | 32.31 | -53.91 |
| 81–90 | A1 EMA α=0.50 | 350.06 | 92.35 | 59.74 | -26.49 |
| 81–90 | A2 quadratic, window 5 | 350.12 | 111.00 | 59.36 | -26.87 |
| 81–90 | A2 quadratic, window 9 | 348.48 | 112.49 | 39.14 | -47.09 |
| 91–100 | A0 existing second difference | 386.23 | 115.31 | 115.61 | +0.00 |
| 91–100 | A1 EMA α=0.25 | 384.50 | 147.68 | 42.53 | -73.09 |
| 91–100 | A1 EMA α=0.50 | 381.04 | 119.34 | 78.85 | -36.76 |
| 91–100 | A2 quadratic, window 5 | 382.97 | 143.94 | 81.01 | -34.60 |
| 91–100 | A2 quadratic, window 9 | 381.87 | 145.85 | 44.62 | -70.99 |

## Final shared-test results

These results are held out from configuration selection. Test file SHA-256 was checked against the preserved baseline manifest before loading. Group errors are computed with the same definitions as validation.

| Configuration | Position RMSE (m) | Δ vs M0 | Velocity RMSE (m/s) | Δ vs M0 | Acceleration RMSE (m/s²) | Δ vs M0 | Six-state MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 existing second difference | 332.08 ± 1.42 | 0.00 ± 0.00 | 79.26 ± 2.33 | 0.00 ± 0.00 | 66.71 ± 4.23 | 0.00 ± 0.00 | 0.0202 ± 0.0002 |
| A1 EMA α=0.25 | 333.01 ± 2.05 | 0.93 ± 3.28 | 94.85 ± 9.42 | 15.60 ± 10.73 | 34.17 ± 2.65 | -32.55 ± 6.60 | 0.0202 ± 0.0005 |
| A1 EMA α=0.50 | 331.54 ± 1.20 | -0.55 ± 1.77 | 82.85 ± 6.00 | 3.59 ± 6.99 | 58.39 ± 4.02 | -8.33 ± 6.96 | 0.0200 ± 0.0002 |
| A2 quadratic, window 5 | 331.33 ± 2.23 | -0.76 ± 2.50 | 91.28 ± 14.94 | 12.03 ± 16.02 | 48.11 ± 5.73 | -18.60 ± 6.05 | 0.0201 ± 0.0008 |
| A2 quadratic, window 9 | 331.18 ± 1.03 | -0.90 ± 2.05 | 93.83 ± 8.20 | 14.57 ± 6.67 | 36.11 ± 7.55 | -30.61 ± 9.21 | 0.0200 ± 0.0005 |

### Six state-coordinate test RMSE (physical units; mean ± sample SD across seeds)

| Configuration | x (m) | vx (m/s) | ax (m/s²) | y (m) | vy (m/s) | ay (m/s²) |
|---|---:|---:|---:|---:|---:|---:|
| A0 existing second difference | 239.42 ± 1.02 | 56.22 ± 1.63 | 51.29 ± 5.47 | 230.13 ± 1.20 | 55.84 ± 2.56 | 42.48 ± 2.71 |
| A1 EMA α=0.25 | 239.96 ± 1.78 | 68.87 ± 6.74 | 24.37 ± 1.77 | 230.90 ± 1.19 | 65.22 ± 6.61 | 23.95 ± 2.02 |
| A1 EMA α=0.50 | 239.04 ± 0.98 | 59.50 ± 4.45 | 41.87 ± 3.37 | 229.74 ± 0.97 | 57.65 ± 4.16 | 40.68 ± 2.42 |
| A2 quadratic, window 5 | 239.01 ± 2.06 | 65.60 ± 12.27 | 33.81 ± 5.48 | 229.45 ± 1.14 | 63.45 ± 8.79 | 34.18 ± 2.69 |
| A2 quadratic, window 9 | 238.71 ± 0.58 | 67.50 ± 6.47 | 25.64 ± 5.55 | 229.56 ± 0.94 | 65.15 ± 5.29 | 25.42 ± 5.15 |

### Per-seed paired test differences

| Variant | Seed | Position (m) | Velocity (m/s) | Acceleration (m/s²) | Six-state MSE (variant, Δ) |
|---|---:|---:|---:|---:|---:|
| A1 EMA α=0.25 | 11 | 332.11 (-1.15) | 92.42 (+11.53) | 33.54 (-35.75) | 0.01999 (-0.00041) |
| A1 EMA α=0.25 | 22 | 332.93 (+2.35) | 85.53 (+3.08) | 32.01 (-33.86) | 0.01986 (-0.00021) |
| A1 EMA α=0.25 | 33 | 336.38 (+5.70) | 109.09 (+31.23) | 38.11 (-23.06) | 0.02108 (+0.00123) |
| A1 EMA α=0.25 | 44 | 330.85 (-2.76) | 98.91 (+20.69) | 35.43 (-29.57) | 0.02008 (-0.00019) |
| A1 EMA α=0.25 | 55 | 332.79 (+0.49) | 88.30 (+11.44) | 31.75 (-40.50) | 0.01993 (-0.00033) |
| A1 EMA α=0.50 | 11 | 329.96 (-3.30) | 86.69 (+5.80) | 60.21 (-9.09) | 0.02000 (-0.00040) |
| A1 EMA α=0.50 | 22 | 330.60 (+0.02) | 75.68 (-6.77) | 60.30 (-5.56) | 0.01978 (-0.00029) |
| A1 EMA α=0.50 | 33 | 331.97 (+1.29) | 81.96 (+4.09) | 56.66 (-4.51) | 0.02002 (+0.00016) |
| A1 EMA α=0.50 | 44 | 332.48 (-1.14) | 90.80 (+12.58) | 62.52 (-2.47) | 0.02045 (+0.00018) |
| A1 EMA α=0.50 | 55 | 332.70 (+0.40) | 79.13 (+2.26) | 52.24 (-20.01) | 0.01995 (-0.00031) |
| A2 quadratic, window 5 | 11 | 331.07 (-2.19) | 88.50 (+7.61) | 44.01 (-25.28) | 0.01990 (-0.00050) |
| A2 quadratic, window 5 | 22 | 330.68 (+0.10) | 86.97 (+4.52) | 45.51 (-20.35) | 0.01983 (-0.00024) |
| A2 quadratic, window 5 | 33 | 329.80 (-0.87) | 78.31 (+0.44) | 42.86 (-18.31) | 0.01946 (-0.00040) |
| A2 quadratic, window 5 | 44 | 329.88 (-3.74) | 85.56 (+7.34) | 56.19 (-8.80) | 0.01988 (-0.00039) |
| A2 quadratic, window 5 | 55 | 335.20 (+2.90) | 117.08 (+40.22) | 52.00 (-20.25) | 0.02146 (+0.00120) |
| A2 quadratic, window 9 | 11 | 330.47 (-2.79) | 89.54 (+8.65) | 32.30 (-36.99) | 0.01971 (-0.00069) |
| A2 quadratic, window 9 | 22 | 332.91 (+2.34) | 108.48 (+26.03) | 49.37 (-16.49) | 0.02084 (+0.00077) |
| A2 quadratic, window 9 | 33 | 330.37 (-0.30) | 90.36 (+12.49) | 34.70 (-26.47) | 0.01975 (-0.00011) |
| A2 quadratic, window 9 | 44 | 331.29 (-2.33) | 90.35 (+12.13) | 30.86 (-34.13) | 0.01981 (-0.00046) |
| A2 quadratic, window 9 | 55 | 330.86 (-1.44) | 90.41 (+13.55) | 33.31 (-38.94) | 0.01979 (-0.00047) |

### Test temporal windows

Mean across seeds by ten-step window; `Δacc` is paired window RMSE relative to M0. This reveals whether acceleration gains persist or deteriorate late in the sequence.

| Steps | Configuration | Position | Velocity | Acceleration | Δacc vs M0 |
|---|---|---:|---:|---:|---:|
| 1–10 | A0 existing second difference | 365.50 | 118.11 | 66.89 | +0.00 |
| 1–10 | A1 EMA α=0.25 | 370.61 | 122.69 | 78.30 | +11.41 |
| 1–10 | A1 EMA α=0.50 | 367.45 | 127.11 | 125.79 | +58.90 |
| 1–10 | A2 quadratic, window 5 | 367.12 | 117.07 | 79.05 | +12.15 |
| 1–10 | A2 quadratic, window 9 | 364.61 | 117.42 | 76.46 | +9.57 |
| 11–20 | A0 existing second difference | 306.16 | 57.70 | 39.06 | +0.00 |
| 11–20 | A1 EMA α=0.25 | 306.61 | 71.34 | 18.34 | -20.72 |
| 11–20 | A1 EMA α=0.50 | 304.86 | 60.18 | 29.54 | -9.52 |
| 11–20 | A2 quadratic, window 5 | 306.40 | 69.26 | 31.68 | -7.38 |
| 11–20 | A2 quadratic, window 9 | 303.22 | 69.82 | 18.25 | -20.81 |
| 21–30 | A0 existing second difference | 293.30 | 58.33 | 39.03 | +0.00 |
| 21–30 | A1 EMA α=0.25 | 295.44 | 72.81 | 17.54 | -21.49 |
| 21–30 | A1 EMA α=0.50 | 293.43 | 61.22 | 29.84 | -9.19 |
| 21–30 | A2 quadratic, window 5 | 295.42 | 69.40 | 29.71 | -9.32 |
| 21–30 | A2 quadratic, window 9 | 294.04 | 72.34 | 18.37 | -20.67 |
| 31–40 | A0 existing second difference | 306.76 | 59.22 | 40.31 | +0.00 |
| 31–40 | A1 EMA α=0.25 | 309.25 | 73.87 | 17.64 | -22.66 |
| 31–40 | A1 EMA α=0.50 | 307.17 | 61.65 | 31.10 | -9.21 |
| 31–40 | A2 quadratic, window 5 | 306.39 | 70.19 | 29.40 | -10.91 |
| 31–40 | A2 quadratic, window 9 | 306.57 | 72.47 | 19.03 | -21.28 |
| 41–50 | A0 existing second difference | 304.94 | 58.40 | 42.53 | +0.00 |
| 41–50 | A1 EMA α=0.25 | 306.92 | 72.75 | 18.35 | -24.18 |
| 41–50 | A1 EMA α=0.50 | 304.78 | 60.88 | 32.51 | -10.03 |
| 41–50 | A2 quadratic, window 5 | 303.35 | 70.01 | 29.52 | -13.01 |
| 41–50 | A2 quadratic, window 9 | 303.17 | 71.85 | 20.58 | -21.96 |
| 51–60 | A0 existing second difference | 307.86 | 59.72 | 47.44 | +0.00 |
| 51–60 | A1 EMA α=0.25 | 310.49 | 76.14 | 20.00 | -27.44 |
| 51–60 | A1 EMA α=0.50 | 308.15 | 64.09 | 35.28 | -12.17 |
| 51–60 | A2 quadratic, window 5 | 307.76 | 73.38 | 31.36 | -16.08 |
| 51–60 | A2 quadratic, window 9 | 306.56 | 75.45 | 23.19 | -24.25 |
| 61–70 | A0 existing second difference | 321.08 | 64.05 | 53.49 | +0.00 |
| 61–70 | A1 EMA α=0.25 | 321.88 | 80.57 | 21.47 | -32.02 |
| 61–70 | A1 EMA α=0.50 | 321.78 | 68.99 | 38.43 | -15.07 |
| 61–70 | A2 quadratic, window 5 | 320.66 | 77.58 | 34.18 | -19.31 |
| 61–70 | A2 quadratic, window 9 | 321.52 | 80.85 | 26.33 | -27.16 |
| 71–80 | A0 existing second difference | 335.05 | 71.76 | 67.75 | +0.00 |
| 71–80 | A1 EMA α=0.25 | 335.52 | 90.65 | 25.14 | -42.62 |
| 71–80 | A1 EMA α=0.50 | 335.78 | 77.46 | 46.54 | -21.21 |
| 71–80 | A2 quadratic, window 5 | 334.89 | 87.19 | 41.07 | -26.68 |
| 71–80 | A2 quadratic, window 9 | 335.85 | 90.87 | 31.25 | -36.51 |
| 81–90 | A0 existing second difference | 354.25 | 86.81 | 84.96 | +0.00 |
| 81–90 | A1 EMA α=0.25 | 353.42 | 105.17 | 29.92 | -55.05 |
| 81–90 | A1 EMA α=0.50 | 353.76 | 88.98 | 56.08 | -28.88 |
| 81–90 | A2 quadratic, window 5 | 352.87 | 101.99 | 55.43 | -29.53 |
| 81–90 | A2 quadratic, window 9 | 352.94 | 105.82 | 37.73 | -47.23 |
| 91–100 | A0 existing second difference | 408.22 | 121.70 | 128.32 | +0.00 |
| 91–100 | A1 EMA α=0.25 | 403.69 | 148.45 | 42.63 | -85.69 |
| 91–100 | A1 EMA α=0.50 | 401.80 | 121.21 | 79.54 | -48.78 |
| 91–100 | A2 quadratic, window 5 | 402.13 | 143.71 | 78.76 | -49.56 |
| 91–100 | A2 quadratic, window 9 | 406.00 | 148.33 | 44.87 | -83.45 |

## Per-seed coordinate metrics

The summaries above give mean and sample SD. These tables provide all six coordinate RMSEs for each seed, separately for validation and test. Coordinate order is x, vx, ax, y, vy, ay; units are m, m/s, and m/s² by state.

### Validation

| Configuration | Seed | x | vx | ax | y | vy | ay |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 11 | 229.15 | 58.67 | 53.90 | 240.06 | 57.08 | 41.29 |
| A0 | 22 | 227.70 | 55.08 | 41.96 | 236.07 | 59.85 | 44.76 |
| A0 | 33 | 227.64 | 54.80 | 44.02 | 236.54 | 56.00 | 41.01 |
| A0 | 44 | 229.03 | 56.29 | 47.87 | 237.80 | 52.95 | 39.82 |
| A0 | 55 | 228.86 | 52.37 | 54.28 | 238.65 | 55.79 | 43.50 |
| A1 EMA α=0.25 | 11 | 228.55 | 66.96 | 23.75 | 236.89 | 65.24 | 23.80 |
| A1 EMA α=0.25 | 22 | 228.75 | 63.87 | 23.01 | 240.27 | 62.72 | 22.83 |
| A1 EMA α=0.25 | 33 | 231.60 | 80.98 | 26.99 | 239.17 | 76.10 | 27.08 |
| A1 EMA α=0.25 | 44 | 229.23 | 70.89 | 25.16 | 238.21 | 71.01 | 24.36 |
| A1 EMA α=0.25 | 55 | 228.97 | 63.60 | 22.47 | 238.16 | 62.95 | 22.55 |
| A1 EMA α=0.50 | 11 | 228.33 | 60.87 | 42.70 | 235.82 | 62.36 | 42.49 |
| A1 EMA α=0.50 | 22 | 228.40 | 54.26 | 42.82 | 237.76 | 55.11 | 42.11 |
| A1 EMA α=0.50 | 33 | 229.02 | 58.82 | 40.43 | 236.28 | 59.16 | 39.88 |
| A1 EMA α=0.50 | 44 | 229.41 | 65.90 | 46.37 | 238.08 | 62.82 | 42.53 |
| A1 EMA α=0.50 | 55 | 228.07 | 54.22 | 35.71 | 236.91 | 57.45 | 37.89 |
| A2 W=5 | 11 | 228.27 | 66.05 | 32.12 | 236.42 | 62.54 | 31.08 |
| A2 W=5 | 22 | 228.61 | 62.73 | 32.38 | 238.73 | 64.17 | 34.35 |
| A2 W=5 | 33 | 227.33 | 54.55 | 29.71 | 235.83 | 57.76 | 33.16 |
| A2 W=5 | 44 | 227.19 | 60.56 | 42.20 | 236.08 | 61.05 | 37.42 |
| A2 W=5 | 55 | 231.43 | 89.03 | 38.04 | 238.98 | 81.45 | 38.38 |
| A2 W=9 | 11 | 228.50 | 67.82 | 22.31 | 236.83 | 63.57 | 23.74 |
| A2 W=9 | 22 | 229.01 | 80.34 | 34.47 | 238.99 | 76.73 | 34.02 |
| A2 W=9 | 33 | 227.21 | 64.13 | 22.99 | 235.96 | 65.82 | 26.21 |
| A2 W=9 | 44 | 228.42 | 64.36 | 20.63 | 236.71 | 60.80 | 22.49 |
| A2 W=9 | 55 | 228.70 | 63.39 | 23.02 | 235.67 | 64.22 | 24.44 |

### Test

| Configuration | Seed | x | vx | ax | y | vy | ay |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 11 | 239.85 | 58.30 | 55.90 | 231.38 | 56.07 | 40.95 |
| A0 | 22 | 237.89 | 56.39 | 46.21 | 229.54 | 60.15 | 46.93 |
| A0 | 33 | 239.11 | 55.61 | 46.12 | 228.41 | 54.51 | 40.19 |
| A0 | 44 | 240.65 | 56.90 | 50.25 | 231.06 | 53.66 | 41.22 |
| A0 | 55 | 239.58 | 53.89 | 57.97 | 230.27 | 54.81 | 43.12 |
| A1 EMA α=0.25 | 11 | 239.40 | 67.21 | 23.93 | 230.19 | 63.44 | 23.51 |
| A1 EMA α=0.25 | 22 | 239.35 | 62.01 | 22.80 | 231.42 | 58.91 | 22.46 |
| A1 EMA α=0.25 | 33 | 242.91 | 79.32 | 26.73 | 232.69 | 74.90 | 27.17 |
| A1 EMA α=0.25 | 44 | 238.16 | 71.14 | 25.64 | 229.66 | 68.72 | 24.45 |
| A1 EMA α=0.25 | 55 | 239.99 | 64.67 | 22.75 | 230.55 | 60.13 | 22.15 |
| A1 EMA α=0.50 | 11 | 237.81 | 61.17 | 42.48 | 228.74 | 61.42 | 42.67 |
| A1 EMA α=0.50 | 22 | 238.27 | 54.50 | 43.47 | 229.18 | 52.50 | 41.79 |
| A1 EMA α=0.50 | 33 | 240.11 | 59.42 | 40.90 | 229.23 | 56.45 | 39.21 |
| A1 EMA α=0.50 | 44 | 239.19 | 65.97 | 45.77 | 230.93 | 62.38 | 42.59 |
| A1 EMA α=0.50 | 55 | 239.80 | 56.42 | 36.73 | 230.62 | 55.48 | 37.15 |
| A2 W=5 | 11 | 238.62 | 63.22 | 30.18 | 229.49 | 61.93 | 32.04 |
| A2 W=5 | 22 | 238.02 | 61.51 | 31.29 | 229.56 | 61.48 | 33.05 |
| A2 W=5 | 33 | 237.90 | 55.05 | 28.89 | 228.42 | 55.69 | 31.66 |
| A2 W=5 | 44 | 237.88 | 61.37 | 42.09 | 228.54 | 59.61 | 37.22 |
| A2 W=5 | 55 | 242.65 | 86.83 | 36.61 | 231.25 | 78.54 | 36.92 |
| A2 W=9 | 11 | 238.21 | 64.61 | 23.04 | 229.06 | 62.00 | 22.64 |
| A2 W=9 | 22 | 239.53 | 78.95 | 35.44 | 231.21 | 74.40 | 34.37 |
| A2 W=9 | 33 | 238.19 | 63.56 | 23.93 | 228.93 | 64.22 | 25.12 |
| A2 W=9 | 44 | 239.07 | 66.21 | 21.84 | 229.34 | 61.48 | 21.80 |
| A2 W=9 | 55 | 238.54 | 64.19 | 23.93 | 229.27 | 63.67 | 23.16 |

## Training behaviour and completion audit

| Configuration | Best epoch mean ± SD | Duration (s) mean ± SD | Parameter count | Completion |
|---|---:|---:|---:|---|
| A0 existing second difference | 161.6 ± 12.9 | 86.5 ± 1.8 | 619,014 | 5/5 complete |
| A1 EMA α=0.25 | 144.6 ± 39.0 | 155.9 ± 30.3 | 619,014 | 5/5 complete |
| A1 EMA α=0.50 | 146.8 ± 21.7 | 155.8 ± 17.6 | 619,014 | 5/5 complete |
| A2 quadratic, window 5 | 151.8 ± 38.6 | 152.1 ± 26.2 | 619,014 | 5/5 complete |
| A2 quadratic, window 9 | 142.6 ± 29.4 | 150.8 ± 22.0 | 619,014 | 5/5 complete |

For new runs, histories range from 106 to 180 epochs depending on validation early stopping; the cap was 180. Best epochs are the minimum-validation-MSE epochs. Every new checkpoint hash matches its run record and every checkpoint deserializes. The 15 acceleration-feature unit tests passed. No failed run or missing artifact was found.

Training histories show rapid loss reduction from epoch 1 followed by a plateau near the selected checkpoint. The new variants’ mean epoch-1 six-state MSE was 0.188–0.235 for training and 0.026–0.103 for validation; their mean final logged losses were near 0.020 for both. The small aggregate train/validation gap at selected checkpoints does not suggest strong overfitting; the high velocity RMSE on some variants is a feature/objective trade-off rather than a failed run.

| Configuration | Epoch-1 train MSE | Epoch-1 val MSE | Best val MSE | Last-epoch train MSE | Last-epoch val MSE |
|---|---:|---:|---:|---:|---:|
| A1 EMA α=0.25 | 0.20048 ± 0.01196 | 0.08480 ± 0.00721 | 0.02002 ± 0.00045 | 0.02006 ± 0.00050 | 0.02039 ± 0.00063 |
| A1 EMA α=0.50 | 0.23519 ± 0.02569 | 0.09097 ± 0.01256 | 0.01981 ± 0.00028 | 0.02016 ± 0.00021 | 0.02031 ± 0.00057 |
| A2 quadratic, window 5 | 0.20844 ± 0.02544 | 0.10290 ± 0.02047 | 0.01998 ± 0.00081 | 0.02030 ± 0.00051 | 0.02061 ± 0.00145 |
| A2 quadratic, window 9 | 0.18816 ± 0.03936 | 0.09112 ± 0.01119 | 0.01978 ± 0.00053 | 0.02020 ± 0.00056 | 0.02017 ± 0.00059 |

Per-epoch training and validation MSE and all six per-state losses are retained in each `run_metrics.json`. There is no evidence of a failed optimization run. Generalization does not improve uniformly: lower acceleration error coexists with higher velocity error for the strongest smoothing variants.

## Implementation verification

- **Causality:** EMA is a forward recurrence; quadratic uses the trailing slice ending at t. No centered window or t+1 sample is read. Future-perturbation tests check outputs before the perturbation remain unchanged.
- **Index alignment:** both formulas output acceleration at the current sequence index. EMA compares smoothed interval velocities `s_t` and `s_{t-1}`. Quadratic offsets are relative to the endpoint (last offset 0) and differentiates the fitted polynomial at that endpoint.
- **Initialization:** indices 0 and 1 copy supplied initial acceleration, matching M0. EMA starts at supplied initial velocity. Quadratic starts with three measured positions at t=2 and uses a growing trailing fit at sequence start.
- **Units / dt:** Cartesian positions and initial velocity/acceleration share the model’s scaled state units. Velocity differences divide by dt; EMA acceleration differences divide by dt; quadratic coefficient is multiplied by `2/dt²`. A dt=1 quadratic polynomial test and non-unit-dt tests verify the conversion and linear unit scaling.
- **Leakage:** calculated features use measured positions and the already supplied initial velocity/acceleration only. No sequence target or ground-truth acceleration array is passed to the feature helper. Initial acceleration retains exactly the baseline’s supplied initial-state handling.
- **Channels / model size:** override calls the original coarse builder and replaces only coarse ax/ay (token channels 11 and 14). The remaining 19 token channels are tested unchanged; the parameter count remains 619,014.
- **Numerical stability / boundaries:** quadratic fit uses float64 pseudoinverse weights with normalized sample-step coordinates, then casts to input dtype/device. Short sequences and expanding-window behavior have explicit tests.

## Interpretation and recommendation

The acceleration feature is a meaningful lever, but this simple smoothing ablation has not shown a no-cost overall gain. The first ten-step window is a consistent weakness: all variants have higher acceleration RMSE there than M0, while windows 11–100 generally improve. This startup behavior is consistent with preserving the initial acceleration for two indices and then transitioning to a smoothed/fit measurement feature; it warrants further boundary-specific study. EMA α=0.25 achieved the lowest validation acceleration RMSE and lowest test acceleration RMSE among the variants, reducing the five-seed test mean relative to M0; it also produced the largest velocity degradation. EMA α=0.50 reduced acceleration less but had substantially smaller velocity degradation. Quadratic W=9 approaches EMA α=0.25 acceleration performance, but velocity error rises more and varies more across seeds. Quadratic W=5 is intermediate on acceleration and velocity, with a notable high-error seed. Position changes are small relative to the baseline variation.

Recommendation: preserve A1 EMA α=0.50 as the balanced feature candidate and α=0.25 as the acceleration-focused candidate for follow-up, but do not replace M0 as an overall default based on this evidence. The next experiment should isolate the interaction between the smoothed acceleration feature and a state-balanced objective (or a controlled multi-objective selection criterion) on the same split, while retaining validation-only selection and paired five-seed reporting. A small α sweep between 0.25 and 0.50 is another lower-risk follow-up if the goal is to map the acceleration/velocity Pareto curve. Keep architectural changes separate.

## Limitations

- Five seeds provide a modest estimate of run variability; reported standard deviations are sample SDs and no inferential significance test is claimed.
- The synthetic fixed distribution and one shared held-out test set limit external validity.
- All variants preserve the same original loss, whose coordinatewise weighting still favors position in aggregate. The feature-only design cannot resolve that objective imbalance by itself.
- Test evaluation is final and must not be reused for tuning these or later variants.

## Provenance and artifact locations

- Experiment directory: `artifacts/results/synthetic/mamba_acceleration_feature_ablation_v1`. Original baseline checkpoints and prior experiment artifacts were not overwritten.
- Train/validation dataset SHA-256: `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733`.
- Shared test dataset SHA-256: `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`.
- Validation selection SHA-256: `1856d72a6e7e34f17a3ff6ac3497848165b768a0a579a6d085f5fcdd9daca9f1`; selection record: `selection.json`.
- Configuration manifest SHA-256: `7583bf6bd0efa35918804e3c24ed3c15e74faf29bc9060f02d6a932ce1574b31`; all four feature configurations were selected on the validation Pareto frontier.
- Evaluator SHA-256: `e03d22cc7b2200068cabd79c349a594ab178659ad1162a44c555daa7907584ef`.
- Source SHA-256 values for model, data, dynamics, runner/helper, evaluator, and tests are recorded in `manifest.json` and `completion_audit_manifest.json`; the configuration-manifest SHA-256 and checkpoint hashes are also in `completion_audit_manifest.json`.
- Validation metrics and epoch-level histories: `A*/seed_*/run_metrics.json`. Test per-seed metrics: `final_test_evaluation/A*/seed_*.json`. Test manifest: `final_test_evaluation/evaluation_manifest.json`.

## Evidence locations

- Feature math, validation and causal bounds: `experiments/mamba_acceleration_features.py:12-117`.
- Feature-only model wrapper and training controls/recording: `experiments/mamba_acceleration_feature_ablation_v1.py:91-249`.
- Fixed split, run manifest and source/checkpoint hashes: `experiments/mamba_acceleration_feature_ablation_v1.py:253-310` and experiment `manifest.json`.
- Validation-only Pareto selection: `selection.json` (created before test access).
- Test hash check, test-only evaluation and parameter count: `experiments/evaluate_mamba_acceleration_feature_ablation_v1.py:25-128`.
- Causality, initial conditions, dt/unit scaling, boundaries, numerical finiteness, token invariance and parameter-count tests: `tests/test_mamba_acceleration_features.py:19-166`; result: 15 passed.

