# Mamba Improvement Experiment 1: State-Normalized Loss Ablation

## Executive summary

This loss-only ablation used the preserved five-seed training/validation pool and seeds 11, 22, 33, 44, and 55. M0 is the existing checkpoint set; it was evaluated but not retrained. M1 and both M2 variants kept the architecture, input representation, optimizer, batch size, regularization, clipping, epoch cap, and early stopping unchanged.

**Neither loss family improved acceleration without a material position/velocity trade-off.** M1 reduced mean validation acceleration RMSE by 24.8%, but its validation position and velocity RMSE increased by 111.9% and 545.6%; M2 weight 8 was better than M1 on all three validation group metrics. M2 weight 4 reduced final test acceleration RMSE by 14.6% on average, but increased position RMSE by 3.0% and velocity RMSE by 16.9%. M2 weight 8 reduced test acceleration RMSE by 24.2% in all five seeds, while increasing position RMSE by 5.0% and velocity RMSE by 24.5%. Both weighted variants also increased six-state MSE.

The variants selected on validation were M2 weights 4 and 8, which formed the validation Pareto frontier. M1 was not evaluated on test because it was dominated by M2 weight 8 on validation position, velocity, and acceleration RMSE. No test result informed that selection.

**Recommendation:** retain M0 as the production/reference model. For the next loss experiment, sweep a narrower acceleration weight range between 1 and 4 and choose against a predeclared velocity guardrail. Do not adopt either tested M2 weight as an unconditional replacement.

## Methodology

### Data and controls

- Fixed pool: 819 training trajectories and 204 validation trajectories, each with 100 time steps. The same existing pool was used for all variants and seeds.
- Seeds: 11, 22, 33, 44, and 55.
- M0: original five-seed checkpoints and records were reused without modification or retraining.
- M1/M2: same 128-dimensional input embedding, three forward Mamba layers, 256-unit prediction head, six-state residual output, Adam `lr=1e-3`, `weight_decay=1e-4`, batch size 16, dropout 0.08, gradient norm clipping at 1.0, 180-epoch cap, patience 25, and `min_delta=1e-6`.
- Validation selected the best checkpoint according to each variant’s own validation objective.
- The shared test set was not opened during training or validation comparison. After saving the selection record, final evaluation loaded the shared test set once and evaluated M0 plus the two validation-selected M2 variants.
- No architecture, feature, input normalization, smoothing, or innovation-conditioning changes were made.

### Loss equations

Let (e_i = \hat{x}_i-x_i), and let (s_i) be the population standard deviation of state coordinate (i), computed over the training split only. State order is `[x, vx, ax, y, vy, ay]`.

- **M0:** (L_0=\frac{1}{6}\sum_{i=1}^{6}\operatorname{MSE}(e_i)), matching the original mean reduction across all six states, batches, and time steps.
- **M1:** (L_1=\frac{1}{6}\sum_{i=1}^{6}\operatorname{MSE}(e_i)/s_i^2). Training-only standard deviations in the scaled model units were `[13.2199, 0.3463, 0.007538, 12.4906, 0.3308, 0.007335]`.
- **M2:** group losses are the mean squared error over the two coordinates in each group. (L_{2,w}=(L_{pos}+L_{vel}+wL_{acc})/(2+w)), with position and velocity weights fixed at 1 and acceleration weight (w\in\{4,8\}). The denominator keeps coefficient sums normalized; the relative acceleration emphasis is (w).

The selected checkpoint minimizes that variant’s validation objective. M0 uses the original historical validation selection. Physical RMSE is computed after multiplying each state error by 1,000. The six-state MSE below remains in the original scaled model units.

## Validation results and selection

Combined validation RMSE, mean ± sample standard deviation across seeds:

| Variant | Position (m) | Velocity (m/s) | Acceleration (m/s²) | Best-checkpoint epochs (seeds 11/22/33/44/55) |
|---|---:|---:|---:|---|
| M0 original | 329.79 ± 1.64 | 79.07 ± 2.41 | 64.25 ± 4.20 | 147 / 169 / 180 / 156 / 156 |
| M1 standardized | 698.70 ± 0.41 | 510.46 ± 4.15 | 48.30 ± 5.81 | 90 / 88 / 179 / 155 / 174 |
| M2 acceleration weight 4 | 341.14 ± 2.06 | 93.21 ± 1.63 | 54.75 ± 7.12 | 142 / 102 / 136 / 133 / 136 |
| M2 acceleration weight 8 | 346.65 ± 2.21 | 97.79 ± 2.23 | 48.18 ± 5.25 | 104 / 136 / 178 / 121 / 127 |

Relative to M0, M1 improved acceleration validation RMSE by 24.8% while worsening position by 111.9% and velocity by 545.6%. M2 weight 8 had marginally lower acceleration RMSE than M1 and far lower position/velocity RMSE, so M1 is dominated and was not selected for test. M2 weights 4 and 8 are both Pareto-efficient: weight 4 has better position and velocity but worse acceleration than weight 8. Both were carried to final test comparison to measure that trade-off.

No single seed drove this selection: M2 weight 8 improved validation acceleration over M0 in all five seeds. M2 weight 4 improved it in four of five; seed 22 was worse by 5.5 m/s².

## Final test results

Test evaluation used the existing shared 1,024-trajectory set with its preserved SHA-256 `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`. Mean ± sample SD over seeds:

| Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE (scaled units) |
|---|---:|---:|---:|---:|
| M0 original | 332.08 ± 1.42 | 79.26 ± 2.33 | 66.71 ± 4.23 | 0.020172 ± 0.000212 |
| M2 acceleration weight 4 | 342.14 ± 2.56 | 92.65 ± 1.85 | 56.96 ± 8.02 | 0.021491 ± 0.000304 |
| M2 acceleration weight 8 | 348.61 ± 2.08 | 98.65 ± 1.88 | 50.57 ± 5.76 | 0.022308 ± 0.000330 |

### Per-seed physical coordinate RMSE

| Variant | Seed | x (m) | vx (m/s) | ax (m/s²) | y (m) | vy (m/s) | ay (m/s²) |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 11 | 239.85 | 58.30 | 55.90 | 231.38 | 56.07 | 40.95 |
| M0 | 22 | 237.89 | 56.39 | 46.21 | 229.54 | 60.15 | 46.93 |
| M0 | 33 | 239.11 | 55.61 | 46.12 | 228.41 | 54.51 | 40.19 |
| M0 | 44 | 240.65 | 56.90 | 50.25 | 231.06 | 53.66 | 41.22 |
| M0 | 55 | 239.58 | 53.89 | 57.97 | 230.27 | 54.81 | 43.12 |
| M2 w4 | 11 | 248.23 | 62.63 | 43.82 | 237.72 | 67.70 | 34.73 |
| M2 w4 | 22 | 244.85 | 67.78 | 54.95 | 235.84 | 67.78 | 44.78 |
| M2 w4 | 33 | 243.03 | 64.96 | 36.85 | 236.05 | 64.05 | 34.91 |
| M2 w4 | 44 | 249.12 | 66.61 | 41.26 | 237.54 | 62.95 | 35.51 |
| M2 w4 | 55 | 246.55 | 65.40 | 40.92 | 239.92 | 65.09 | 33.36 |
| M2 w8 | 11 | 249.77 | 67.11 | 42.67 | 241.55 | 72.64 | 36.11 |
| M2 w8 | 22 | 252.06 | 69.10 | 34.10 | 242.02 | 70.84 | 33.82 |
| M2 w8 | 33 | 250.45 | 72.02 | 28.98 | 238.46 | 67.51 | 30.24 |
| M2 w8 | 44 | 251.83 | 67.47 | 36.08 | 241.62 | 67.90 | 37.38 |
| M2 w8 | 55 | 252.48 | 72.61 | 42.53 | 244.29 | 70.15 | 35.06 |

### Combined physical RMSE and six-state MSE by seed

| Variant | Seed | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |
|---|---:|---:|---:|---:|---:|
| M0 | 11 | 333.26 | 80.89 | 69.29 | 0.020401 |
| M0 | 22 | 330.58 | 82.45 | 65.86 | 0.020069 |
| M0 | 33 | 330.67 | 77.86 | 61.17 | 0.019858 |
| M0 | 44 | 333.61 | 78.21 | 64.99 | 0.020273 |
| M0 | 55 | 332.30 | 76.86 | 72.25 | 0.020259 |
| M2 w4 | 11 | 343.70 | 92.22 | 55.91 | 0.021626 |
| M2 w4 | 22 | 339.96 | 95.86 | 70.89 | 0.021631 |
| M2 w4 | 33 | 338.79 | 91.23 | 50.76 | 0.020947 |
| M2 w4 | 44 | 344.22 | 91.65 | 54.44 | 0.021642 |
| M2 w4 | 55 | 344.02 | 92.27 | 52.79 | 0.021609 |
| M2 w8 | 11 | 347.47 | 98.89 | 55.89 | 0.022273 |
| M2 w8 | 22 | 349.44 | 98.96 | 48.03 | 0.022368 |
| M2 w8 | 33 | 345.82 | 98.71 | 41.88 | 0.021848 |
| M2 w8 | 44 | 348.99 | 95.72 | 51.95 | 0.022276 |
| M2 w8 | 55 | 351.32 | 100.96 | 55.12 | 0.022776 |

### Paired differences versus M0

Differences are variant minus M0 for the same seed. Negative acceleration differences mean improvement.

| Variant | Seed | Position Δ (m) | Velocity Δ (m/s) | Acceleration Δ (m/s²) |
|---|---:|---:|---:|---:|
| M2 w4 | 11 | +10.44 | +11.34 | -13.38 |
| M2 w4 | 22 | +9.38 | +13.41 | +5.03 |
| M2 w4 | 33 | +8.12 | +13.37 | -10.41 |
| M2 w4 | 44 | +10.61 | +13.43 | -10.56 |
| M2 w4 | 55 | +11.72 | +15.41 | -19.46 |
| M2 w8 | 11 | +14.21 | +18.01 | -13.40 |
| M2 w8 | 22 | +18.86 | +16.51 | -17.83 |
| M2 w8 | 33 | +15.15 | +20.85 | -19.29 |
| M2 w8 | 44 | +15.38 | +17.51 | -13.04 |
| M2 w8 | 55 | +19.01 | +24.10 | -17.13 |

Paired-difference mean ± SD:

| Variant | Position Δ | Velocity Δ | Acceleration Δ | Six-state MSE Δ |
|---|---:|---:|---:|---:|
| M2 w4 | +10.05 ± 1.36 m (+3.0%) | +13.39 ± 1.44 m/s (+16.9%) | -9.75 ± 9.04 m/s² (-14.6%) | +0.001319 ± 0.000180 (+6.5%) |
| M2 w8 | +16.52 ± 2.25 m (+5.0%) | +19.39 ± 3.08 m/s (+24.5%) | -16.14 ± 2.78 m/s² (-24.2%) | +0.002136 ± 0.000264 (+10.6%) |

M2 weight 8 improved acceleration test RMSE for all five seeds. M2 weight 4 improved it for four seeds and regressed for seed 22. Both variants increased position and velocity RMSE for every seed. This is a clear trade-off across the five-seed paired evaluation; no statistical-significance claim is made from five seeds.

## Per-state train and validation loss

The training runner records raw per-state MSE (scaled model units squared) at every epoch in each `run_metrics.json`; it also stores per-state physical RMSE for the selected checkpoint. The table below averages the selected-checkpoint MSE across seeds. It is decomposed by state coordinate and is not the M1 normalized objective.

| Variant | Split | x | vx | ax | y | vy | ay |
|---|---|---:|---:|---:|---:|---:|---:|
| M0 | Train | 0.0547763 | 0.0028396 | 0.0023904 | 0.0511389 | 0.0030281 | 0.0017346 |
| M0 | Validation | 0.0522019 | 0.0030782 | 0.0023682 | 0.0565618 | 0.0031784 | 0.0017737 |
| M1 | Train | 0.2520658 | 0.1310905 | 0.0013139 | 0.2363532 | 0.1236550 | 0.0010598 |
| M1 | Validation | 0.2287225 | 0.1223844 | 0.0011634 | 0.2594546 | 0.1382019 | 0.0011965 |
| M2 w4 | Train | 0.0588830 | 0.0038755 | 0.0016864 | 0.0550441 | 0.0041605 | 0.0012908 |
| M2 w4 | Validation | 0.0556049 | 0.0042305 | 0.0016500 | 0.0607726 | 0.0044595 | 0.0013881 |
| M2 w8 | Train | 0.0611196 | 0.0044970 | 0.0011487 | 0.0568767 | 0.0047218 | 0.0011152 |
| M2 w8 | Validation | 0.0573060 | 0.0046271 | 0.0011160 | 0.0628625 | 0.0049390 | 0.0012272 |

M1 makes acceleration component MSE smaller but leaves position/velocity errors extremely high. M2 shifts error from acceleration into velocity and, to a lesser degree, position. Each seed’s complete per-epoch train/validation component losses and temporal windows are in the records linked below.

## Temporal error profile

The table shows mean test RMSE across seeds in consecutive ten-step windows. Each value is a combined two-axis group RMSE in physical units. Per-seed windows for all three groups are saved in the final evaluation records.

| Steps | M0 pos (m) | M0 vel (m/s) | M0 acc (m/s²) | M2 w4 pos | M2 w4 vel | M2 w4 acc | M2 w8 pos | M2 w8 vel | M2 w8 acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 365.50 | 118.11 | 66.89 | 379.65 | 127.28 | 56.29 | 381.93 | 130.77 | 52.38 |
| 11–20 | 306.16 | 57.70 | 39.06 | 306.29 | 61.64 | 31.90 | 306.18 | 57.62 | 26.42 |
| 21–30 | 293.30 | 58.33 | 39.03 | 297.16 | 64.03 | 31.57 | 298.57 | 60.89 | 26.44 |
| 31–40 | 306.76 | 59.22 | 40.31 | 310.47 | 65.99 | 32.94 | 313.98 | 66.23 | 27.68 |
| 41–50 | 304.94 | 58.40 | 42.53 | 310.16 | 66.59 | 34.86 | 316.49 | 68.27 | 29.93 |
| 51–60 | 307.86 | 59.72 | 47.44 | 314.34 | 69.34 | 39.29 | 320.88 | 70.78 | 34.44 |
| 61–70 | 321.08 | 64.05 | 53.49 | 328.81 | 74.53 | 44.92 | 333.64 | 77.09 | 40.50 |
| 71–80 | 335.05 | 71.76 | 67.75 | 347.98 | 85.51 | 57.63 | 355.94 | 92.48 | 53.29 |
| 81–90 | 354.25 | 86.81 | 84.96 | 372.85 | 108.43 | 73.19 | 383.87 | 120.97 | 65.58 |
| 91–100 | 408.22 | 121.70 | 128.32 | 429.72 | 153.02 | 112.27 | 446.49 | 172.94 | 98.76 |

Acceleration weighting improves acceleration in every window for both M2 variants, most clearly in the middle and late windows. Position/velocity deterioration also grows late in the horizon, especially at weight 8. The learned state estimates are not fed back autoregressively, so these curves show horizon-dependent error, not proof of prediction-feedback accumulation.

## Training duration and best validation epoch

| Variant | Seed | Best epoch | Epochs run | Training seconds |
|---|---:|---:|---:|---:|
| M0 | 11 | 147 | 172 | 84.2 |
| M0 | 22 | 169 | 180 | 87.0 |
| M0 | 33 | 180 | 180 | 89.1 |
| M0 | 44 | 156 | 180 | 86.3 |
| M0 | 55 | 156 | 180 | 85.9 |
| M1 | 11 | 90 | 115 | 61.4 |
| M1 | 22 | 88 | 113 | 60.7 |
| M1 | 33 | 179 | 180 | 96.5 |
| M1 | 44 | 155 | 180 | 96.7 |
| M1 | 55 | 174 | 180 | 96.2 |
| M2 w4 | 11 | 142 | 167 | 93.3 |
| M2 w4 | 22 | 102 | 127 | 71.3 |
| M2 w4 | 33 | 136 | 161 | 90.3 |
| M2 w4 | 44 | 133 | 158 | 88.9 |
| M2 w4 | 55 | 136 | 161 | 90.9 |
| M2 w8 | 11 | 104 | 129 | 72.9 |
| M2 w8 | 22 | 136 | 161 | 90.8 |
| M2 w8 | 33 | 178 | 180 | 101.6 |
| M2 w8 | 44 | 121 | 146 | 82.5 |
| M2 w8 | 55 | 127 | 152 | 85.9 |

M0 duration and epoch data come from the preserved run records. M1 and M2 durations cover training and validation/checkpoint selection; final train/validation metric recomputation is excluded. All five seeds completed successfully for all experimental variants; no run was silently dropped.

## Configuration, provenance, and detailed records

The original five-seed artifacts were preserved. The new records live under [mamba_loss_ablation_v1 artifacts](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1). Configuration and source hashes are recorded in [manifest.json](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/manifest.json) and [provenance.json](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/provenance.json). The test selection rule and chosen variants are in [selection.json](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/selection.json); test-set hash and order of access are in [evaluation_manifest.json](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/final_test_evaluation/evaluation_manifest.json).

- Per-seed train/validation physical RMSEs, best epoch, training duration, and six-state MSE: [per_seed_summary.csv](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/per_seed_summary.csv).
- Per-epoch per-state train/validation losses for M1/M2 and selected-checkpoint split metrics for all variants: [M0 records](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/M0_original), [M1 records](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/M1_standardized), [M2 weight 4 records](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/M2_accel_w4), [M2 weight 8 records](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/M2_accel_w8). M0 was not retrained, so its historical epoch record contains only the original aggregate training/validation curve.
- Per-seed final test metrics and temporal windows: [M0 evaluation](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/final_test_evaluation/M0_original), [M2 weight 4 evaluation](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/final_test_evaluation/M2_accel_w4), [M2 weight 8 evaluation](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/mamba_loss_ablation_v1/final_test_evaluation/M2_accel_w8).
- Separate experiment runner and final evaluator: [mamba_loss_ablation_v1.py](/home/spcom41/BiMambaTracker/radar-tracking/experiments/mamba_loss_ablation_v1.py), [evaluate_mamba_loss_ablation_v1.py](/home/spcom41/BiMambaTracker/radar-tracking/experiments/evaluate_mamba_loss_ablation_v1.py).

## Limitations and interpretation

1. Five seeds share a fixed training/validation pool and a fixed test set. Paired seed deltas quantify initialization/training variation on this dataset; they do not establish performance across new motion/noise regimes.
2. The standardized loss uses sequence-wide training standard deviations. Since position, velocity, and acceleration occupy very different scales, this intentionally changes task priority and can create large group trade-offs, as observed.
3. M2 weights 4 and 8 were the only acceleration weights tested. The Pareto result does not identify an optimal weight.
4. Checkpoint selection uses each variant’s validation objective, which differs across M0/M1/M2. The comparison therefore asks what each specified training recipe selects under the unchanged early-stopping rule, not what would happen under a shared post-hoc metric selector.
5. No inferential significance test is claimed from five seeds. Reported standard deviations and paired differences expose consistency and variation.

## Conclusion

State normalization and acceleration group weighting can lower acceleration error, but neither tested approach meets the goal of doing so without degrading position and velocity. M1 strongly sacrifices position and velocity. M2 weight 4 is the milder trade-off; weight 8 gives a larger, consistent acceleration gain but materially worsens velocity and overall six-state MSE. Keep M0 as the reference and run a narrower, validation-guarded acceleration-weight sweep next.
