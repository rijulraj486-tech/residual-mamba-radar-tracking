# Paper v4 Mamba Improvement Plan and Results

**Data contract:** fixed synthetic pool only: 819 training, 204 validation, 1,024 test trajectories, each 100 steps. Train/validation file SHA-256 `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733`; held-out test SHA-256 `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`. Test observations and metrics were reserved through validation selection and accessed only after the five-seed validation gate and checkpoint manifest were frozen. All experiments retain the causal forward-only Mamba and the baseline physical metric definitions. The original five-seed artifacts remain unchanged. Result rows label their data split; the seed-11 screens and B remain validation-only, while A completed a frozen five-seed held-out evaluation.

## 1. Result summary

All existing full test rows below use the same held-out pool and report mean ± sample SD across the five training seeds. The table labels each metric split: M1, weight-2, and B rows are validation-only; A reports its five-seed held-out test metrics, with its validation confirmation shown separately below. Validation-only values are not comparable as held-out test results.

| Experiment | Status / seeds | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Δ vs M0 (position / velocity / acceleration / MSE) |
|---|---|---:|---:|---:|---:|---|
| M0 original Mamba | Completed, 5 | 332.08 ± 1.42 | 79.26 ± 2.33 | 66.71 ± 4.23 | 0.020172 ± 0.000212 | Reference |
| M1 standardized state loss | Completed, 5; validation-selected, not test-evaluated | 698.70 ± 0.41* | 510.46 ± 4.15* | 48.30 ± 5.81* | — | Validation: +368.91 m / +431.39 m/s / −15.95 m/s²; not promoted |
| M2 acceleration weight 2 | Pilot complete, seed 11; validation only | 337.60 | 83.92 | 63.50 | 0.020842 | Validation: +5.73 m / +2.06 m/s / −4.40 m/s² / +0.000601 |
| M2 acceleration weight 4 | Completed, 5 | 342.14 ± 2.56 | 92.65 ± 1.85 | 56.96 ± 8.02 | 0.021491 ± 0.000304 | +10.05 m / +13.39 m/s / −9.75 m/s² / +0.001319 |
| M2 acceleration weight 8 | Completed, 5 | 348.61 ± 2.08 | 98.65 ± 1.88 | 50.57 ± 5.76 | 0.022308 ± 0.000330 | +16.52 m / +19.39 m/s / −16.14 m/s² / +0.002136 |
| EMA α=.25 acceleration | Completed, 5 | 333.01 ± 2.05 | 94.85 ± 9.42 | 34.17 ± 2.65 | 0.0202 ± 0.0005 | +0.93 m / +15.60 m/s / −32.55 m/s² / +0.0000 (rounded) |
| EMA α=.50 acceleration | Completed, 5 | 331.54 ± 1.20 | 82.85 ± 6.00 | 58.39 ± 4.02 | 0.0200 ± 0.0002 | −0.55 m / +3.59 m/s / −8.33 m/s² / lower (rounded) |
| Trailing quadratic W=5 | Completed, 5 | 331.33 ± 2.23 | 91.28 ± 14.94 | 48.11 ± 5.73 | 0.0201 ± 0.0008 | −0.76 m / +12.03 m/s / −18.60 m/s² / lower (rounded) |
| Trailing quadratic W=9 | Completed, 5 | 331.18 ± 1.03 | 93.83 ± 8.20 | 36.11 ± 7.55 | 0.0200 ± 0.0005 | −0.90 m / +14.57 m/s / −30.61 m/s² / lower (rounded) |
| Adaptive acceleration, narrow | Completed, 5 | 330.37 ± 1.14 | 81.74 ± 5.00 | 52.86 ± 1.70 | 0.0198 ± 0.0002 | −1.71 m / +2.48 m/s / −13.86 m/s² / −0.0004 |
| Adaptive acceleration, broad | Completed, 5 train/val; dominated and not test-evaluated | — | — | — | — | Validation only: not selected |
| Train-only input normalization | Five-seed validation confirmed; final test evaluated | 331.99 ± 3.09 | 72.28 ± 3.50 | 45.36 ± 6.78 | 0.019593 ± 0.000497 | Test paired Δ: −0.09 m / −6.98 m/s / −21.36 m/s² / −0.000580 |
| Separate position/velocity/acceleration heads | Validation screen, seed 11 | 330.636 | 76.484 | 60.028 | 0.019796 | Validation vs paired M0: −1.234 m / −5.376 m/s / −7.870 m/s² / −0.000446 |
| Train-scaled residual output | Planned | — | — | — | — | Not run |
| Temporal kinematic consistency loss | Planned | — | — | — | — | Not run |
| Mamba embed 96 / 160 | Planned, separate one-factor runs | — | — | — | — | Not run |
| Mamba `d_state=8` | Planned | — | — | — | — | Not run |
| Mamba depth 2 / 4 | Planned, separate one-factor runs | — | — | — | — | Not run |
| Residual dropout .12 | Planned | — | — | — | — | Not run |
| Prediction head width 128 | Planned | — | — | — | — | Not run |
| Causal context window 50 | Planned | — | — | — | — | Not run |

`*` M1 figures are validation metrics, not test results. They are placed in the table for context and explicitly not compared as held-out performance. Values and SDs are from [mamba_loss_ablation_v1.md](mamba_loss_ablation_v1.md), [mamba_acceleration_feature_ablation_v1.md](mamba_acceleration_feature_ablation_v1.md), and [mamba_adaptive_acceleration_v1.md](mamba_adaptive_acceleration_v1.md). Pilot metrics and provenance are in the linked summaries under `artifacts/results/synthetic/paper_v4_mamba_improvement_v1/pilot_loss_accel_w2/` and `artifacts/results/synthetic/paper_v4_mamba_improvement_v2/`.

### Resource records

Training times are mean ± sample SD across five seeds when available. Historical ablation reports did not consistently record inference timing; `not recorded` means no comparable stored measurement was found. The pilot timing is a warmed, synchronized CUDA measurement over the 204-example validation pass, not end-to-end deployment latency.

| Experiment | Parameters | Training seconds per seed | Inference timing |
|---|---:|---:|---|
| M0 original | 619,014 | 86.5 ± 1.8 | 0.0386 s/full validation pass (seed 11 pilot comparison only) |
| M1 standardized loss | 619,014 | 82.3 ± 19.4 | Not recorded |
| M2 acceleration weight 2 pilot | 619,014 | 75.34 (seed 11) | 0.0413 s/full validation pass (seed 11) |
| Train-only input normalization (5 seeds) | 619,014 | 63.33 ± 20.60 | 0.19467 ± 0.00101 s/full 1,024-example test pass |
| Separate group heads (seed 11) | 627,014 (+8,000; +1.29%) | 101.56 | 0.04339 s/full validation pass |
| M2 acceleration weight 4 | 619,014 | 87.0 ± 8.9 | Not recorded |
| M2 acceleration weight 8 | 619,014 | 86.7 ± 10.6 | Not recorded |
| EMA α=.25 / EMA α=.50 | 619,014 | 155.9 ± 30.3 / 155.8 ± 17.6 | Not recorded |
| Quadratic W=5 / W=9 | 619,014 | 152.1 ± 26.2 / 150.8 ± 22.0 | Not recorded |
| Adaptive broad / narrow | 619,014 | 381.7 ± 98.8 / 437.3 ± 2.7 | Not recorded |
| Planned architecture/context experiments | To be counted from instantiated model | Pending | Pending |

### Pilot details

The weight-2 pilot changed only the loss and trained seed 11 from scratch on all 819 training trajectories, using the 204 validation trajectories for checkpoint selection. It ran 138 epochs and selected epoch 113. Baseline and pilot each have 619,014 parameters. The training time was 75.34 s (pilot runner, NVIDIA RTX 3060, CUDA 12.1, PyTorch 2.4.0). Three warmed-up full-validation inference passes averaged 0.0386 s for M0 and 0.0413 s for the pilot; this short measurement is descriptive only.

On validation, weight 2 lowered acceleration RMSE by 4.40 m/s² (6.5%) but raised position by 5.73 m (1.7%), velocity by 2.06 m/s (2.5%), and aggregate six-state MSE by 2.97%. This is a modest trade-off in one seed, not evidence of a multi-seed gain. The pilot did not read the held-out test file. It does **not** meet the goal of improving all three state groups, so weight 2 should not be promoted as an overall replacement on this result alone.

### Seed-11 controlled validation screens

Both screens trained on the fixed 819-trajectory training split and selected by six-state validation MSE on the same 204 trajectories. The paired baseline is the preserved M0 seed-11 checkpoint (position 331.870 m, velocity 81.860 m/s, acceleration 67.899 m/s², six-state MSE 0.020241). The runner and manifests report that the test file was not loaded (`test_accessed=false`, `test_file_not_loaded=true`). These are single-seed screening results only.

| Candidate | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Parameters | Train time | Best epoch / epochs run |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paired M0 seed 11 | 331.870 | 81.860 | 67.899 | 0.020241 | 619,014 | Not recorded for this paired timing artifact | 147 |
| A: train-only input normalization | 328.998 | 72.635 | 49.268 | 0.019324 | 619,014 | 52.19 s | 76 / 101 |
| B: separate group residual heads | 330.636 | 76.484 | 60.028 | 0.019796 | 627,014 | 101.56 s | 174 / 180 |

The A/B inference values are warmed, synchronized CUDA measurements over the 204-example validation pass (one warmup plus three timed passes): paired M0 timings were 0.03902 s for A and 0.03834 s for B; A took 0.04014 s and B 0.04339 s. These are descriptive timings, not deployment latency estimates. Pilot provenance, exact configs, checkpoint hashes, feature statistics, environment, and per-state/horizon metrics are recorded in each [A summary](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_a_feature_norm_seed11/pilot_summary.json) and [B summary](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_b_separate_heads_seed11/pilot_summary.json).

Against paired M0, A changed position/velocity/acceleration/MSE by −0.865%/−11.27%/−27.44%/−4.53%; B changed them by −0.372%/−6.57%/−11.59%/−2.20%. Both meet the plan's prospective +1% position/+3% velocity validation guardrails and lower all four metrics in this seed. A is better than B on all four metrics, has unchanged parameters, and has shorter observed training and validation-pass time. At the time of these one-seed screens, neither result alone established a multi-seed gain; the separate five-seed A confirmation and test result are reported below.

### Five-seed validation confirmation and frozen selection

The five runs used seeds 11/22/33/44/55, the same 819/204 fixed train/validation trajectories, and one independently fitted normalization transform per seed from that seed's training split only. All candidates have 619,014 parameters. The selected/stopping epochs and validation metrics are:

| Seed | Selected / stopped epoch | Train seconds | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 76 / 101 | 51.23 | 329.009 | 72.632 | 49.327 | 0.019326 |
| 22 | 129 / 154 | 78.60 | 328.895 | 70.983 | 44.800 | 0.019203 |
| 33 | 128 / 153 | 78.45 | 331.739 | 72.190 | 38.048 | 0.019452 |
| 44 | 124 / 149 | 75.65 | 329.397 | 70.292 | 42.073 | 0.019202 |
| 55 | 40 / 65 | 32.71 | 335.477 | 78.135 | 55.217 | 0.020283 |
| Mean ± sample SD | — | 63.33 ± 20.60 | 330.903 ± 2.807 | 72.847 ± 3.100 | 45.893 ± 6.634 | 0.019493 ± 0.000454 |
| Paired M0 mean ± sample SD | — | — | 329.790 ± 1.639 | 79.069 ± 2.408 | 64.249 ± 4.198 | 0.019860 ± 0.000266 |
| Paired difference mean ± sample SD | — | — | +1.113 ± 3.107 | −6.223 ± 4.695 | −18.356 ± 3.042 | −0.000367 ± 0.000457 |

The predeclared mean validation guardrails passed: position increased 0.338% (limit +1%), while velocity decreased 7.870% (maximum allowed increase +3%). The selected configuration and all five checkpoint SHA-256 values were frozen before test access in [validation_selection_manifest.json](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/validation_selection_manifest.json); its integrity hashes are in [frozen_validation_hashes.json](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/frozen_validation_hashes.json). The complete validation aggregate records the paired per-seed differences, standard deviations, parameter counts, runtimes, and baseline checkpoint identities ([validation_aggregate.json](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/validation_aggregate.json)).

### Final held-out test comparison

After the validation manifest qualified and was verified, the evaluator opened the unchanged test pool (SHA-256 `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`). It reproduced the archived M0 five-seed metrics and evaluated all five frozen A checkpoints. Test results are means ± sample SD; the paired column is candidate minus M0, with the SD of the five paired differences.

| Model | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |
|---|---:|---:|---:|---:|
| M0 original | 332.085 ± 1.418 | 79.256 ± 2.325 | 66.713 ± 4.235 | 0.020172 ± 0.000212 |
| A train-only normalization | 331.994 ± 3.088 | 72.276 ± 3.501 | 45.356 ± 6.778 | 0.019593 ± 0.000497 |
| Paired difference A − M0 | −0.091 ± 3.682 | −6.980 ± 5.073 | −21.358 ± 2.763 | −0.000580 ± 0.000524 |

Position is effectively unchanged in the paired five-seed result: its mean is lower by only 0.091 m, with mixed per-seed signs. Mean velocity, acceleration, and six-state MSE are lower on the held-out pool. Thus the measured gain is concentrated in velocity and acceleration; the evidence does not support a meaningful position-accuracy gain. Per-seed validation and test metrics/differences, epochs, runtimes, and checkpoint hashes are in [paired_seed_comparison_v1.csv](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/paired_seed_comparison_v1.csv) and [final_test_comparison.json](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/final_test_evaluation_v1/final_test_comparison.json). Candidate training averaged 63.33 ± 20.60 s per seed; warmed, synchronized inference averaged 0.19467 ± 0.00101 s for all 1,024 test trajectories on CUDA. These short evaluation timings are implementation/environment specific, not deployment latency claims. Runtime environment is recorded in [run_environment.json](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/run_environment.json).

## 2. Controlled experiment sequence

### Phase 0 — preserve and instrument

- Freeze M0 run records and checkpoint SHA-256 values; use [baseline_reference.json](../../configs/paper_v4_mamba_improvement/baseline_reference.json) as the immutable comparison contract.
- Keep one config per variant in `configs/paper_v4_mamba_improvement/`; store each run under a new, versioned result path that refuses overwrite.
- Retain componentwise train/validation MSE, physical group RMSE, ten-step windows, checkpoint epoch/digest, parameter count, training wall time, and synchronized inference timing.
- Prospective validation selection rule for future multi-objective work: first require no more than +1% position and +3% velocity relative to the paired M0 validation checkpoint, then minimize acceleration RMSE among candidates satisfying both guardrails. Also report six-state MSE; it is a metric, not a substitute for the group guardrails. If no candidate meets the guardrails, report no promotion.

The weight-2 pilot config had only a qualitative “near M0” position/velocity gate, not these numeric thresholds. The +1%/+3% rule was written after that pilot and is prospective; applying it retrospectively is a diagnostic, not a preregistered rejection. The observed single-seed acceleration gain is modest and is accompanied by worse position, velocity, and aggregate MSE, so the pilot alone does not justify a promotion claim or a five-seed commitment.

### Phase 1 — validation-only screening, one major factor per run

1. **Train-only feature normalization** (five-seed validation confirmation and final test evaluation complete; [confirmation config](../../configs/paper_v4_mamba_improvement/features_train_only_normalization_five_seed_confirmation_v2.json)): compute each seed’s per-feature statistics from training only; preserve sine/cosine and wrapped-angle semantics. The mean validation guardrails passed and all five checkpoints were frozen before the test evaluation.
2. **Separate state-group heads** (completed seed-11 screen; [config](../../configs/paper_v4_mamba_improvement/head_separate_state_groups_seed11.json)): retain shared causal encoder and split only the output MLP. The screen improved all four validation metrics but was dominated by A and added 8,000 parameters (1.29%) and runtime. Keep as a secondary five-seed candidate if isolating output specialization remains a research priority; do not combine it with A yet.
3. **Train-scaled residual coordinates** ([config](../../configs/paper_v4_mamba_improvement/head_train_scaled_residuals.json)): normalize each residual output using training coarse-reference residual RMS, then invert the scale before applying the original physical scaled-state MSE. This is separate from input normalization and from weighted loss.
4. **Kinematic consistency loss** ([config](../../configs/paper_v4_mamba_improvement/loss_temporal_consistency.json)): after the preceding loss pilot, test one small auxiliary penalty using train-only RMS scales. Do not constrain acceleration to be constant; process noise is present. Check group metrics to ensure the penalty does not suppress real state changes.
5. **Initial-state placement** ([config](../../configs/paper_v4_mamba_improvement/initial_state_first_token.json)): preserve the 21-channel input but place the initializer only in the first token instead of repeating it at every step. This tests whether repeated initializer channels crowd out time-varying evidence while relying on causal hidden state to retain the initial context.
6. **Causal context ablation** ([config](../../configs/paper_v4_mamba_improvement/context_window_50.json)): train and validate with the same rolling 50-token context and reset Mamba state at window start. Use current/past-only features; evaluate all times, not only after step 50. Validate alignment and reset implementation before training. Compare with the existing full-prefix M0; do not call the result an architecture-memory test unless context is the only changed factor.

For future screens, use one seed for debugging/training-only feasibility, then all five seeds if the validation guardrails appear promising. A single-seed pilot can identify plainly poor behavior but cannot establish improvement or be promoted. A has completed five-seed confirmation and test evaluation; B remains a one-seed validation screen. The weight-2 pilot remains a loss screen and is not a full-seed recommendation by itself.

### Phase 2 — narrow architecture sweep

Keep data, features, loss, optimizer, and training schedule fixed; evaluate these config files independently: [embed 96](../../configs/paper_v4_mamba_improvement/mamba_embed_96.json), [embed 160](../../configs/paper_v4_mamba_improvement/mamba_embed_160.json), [`d_state=8`](../../configs/paper_v4_mamba_improvement/mamba_dstate_8.json), [2 blocks](../../configs/paper_v4_mamba_improvement/mamba_blocks_2.json), [4 blocks](../../configs/paper_v4_mamba_improvement/mamba_blocks_4.json), [dropout .12](../../configs/paper_v4_mamba_improvement/mamba_dropout_012.json), and [head width 128](../../configs/paper_v4_mamba_improvement/head_width_128.json). The current model (`embed=128`, `d_state=16`, three blocks, dropout .08, head width 256) remains the reference point.

First run short training-only feasibility checks with one seed. Continue only configs that complete stably and improve validation Pareto position/velocity/acceleration or produce meaningful efficiency savings without unacceptable group regression. Then run the full five seeds with unchanged validation selection. Do not test multiple architectural changes simultaneously.

### Phase 3 — combine only selected factors

Only after a feature/architecture factor and a loss factor independently pass the validation guardrails, create a new combination config and run the same five seeds. Use a predeclared paired comparison against M0 and each selected component alone. Do not choose combinations by test performance. A combined configuration is a new model, not a continuation of either single-factor result.

## 3. Experimental safeguards and metrics

- Read train and validation from the fixed pool; verify SHA-256 and tensor shapes before any run. Never regenerate the fixed test set.
- Compute all feature means/scales, residual scales, loss scales, and auxiliary-loss scales from the 819 training trajectories only. Validation can select among predeclared configurations; it cannot estimate preprocessing statistics.
- Seed Python, NumPy, CPU Torch, and CUDA Torch using the paired seeds 11/22/33/44/55. Record all seeds and the deterministic/cuDNN flags.
- Evaluate test only after validation selection JSON and its SHA-256 have been written. Test result tables must contain exactly the frozen variants and use the original metric function.
- Report position, velocity, acceleration RMSE; six-state scaled MSE and dB; six coordinate RMSE/bias; ten-step horizon metrics; paired per-seed differences; mean/sample SD; parameter count; training time; and warmed, synchronized inference time. Mark historical inference timing unavailable where it was not recorded; do not infer it from training time or parameter count.
- Mamba inference requires CUDA in the current code; record device and report EKF/UKF timing separately unless all methods run on a shared device under a defined implementation. Do not compare CPU filter wall time directly with GPU Mamba wall time.
- Keep `five_seed_n1024_fixedpool_v1`, completed ablations, configs, and the one-seed pilot immutable. Every new run gets a new versioned output directory. A’s confirmation and final test records are stored under `paper_v4_mamba_improvement_v3/`; no original checkpoint or result was overwritten.
- Do not claim broad radar robustness from this one synthetic constant-acceleration/noise regime. A separate design is required for new noise conditions or motion models.

## 4. Recommended candidates

### Previous balanced alternative from the feature ablation

Before the train-only normalization confirmation, EMA acceleration feature `α=.50` was the most balanced existing alternative: versus M0 it improves position RMSE by 0.55 m and acceleration RMSE by 8.33 m/s², while velocity RMSE rises by 3.59 m/s (about 4.5%); aggregate MSE is slightly lower at the report's precision. It does not improve all three groups, but its trade-off is much smaller than EMA `.25`, quadratic W=9, or M2 weights 4/8. Its full five-seed record is already available; no repeat is needed merely to establish it.

### Candidates for five-seed confirmation or future screening

1. **Train-only input normalization (A)** completed the prescribed five-seed validation confirmation and frozen held-out evaluation. On test it reduced mean velocity RMSE by 6.98 m/s, acceleration RMSE by 21.36 m/s², and six-state MSE by 0.000580 versus M0. Position mean changed by only −0.09 m and had mixed paired per-seed differences; report that as parity, not a meaningful gain. The result supports this feature transform for the evaluated synthetic regime, not general radar robustness.
2. **Separate group heads (B)** are a secondary candidate. They also improved all four validation metrics and met the guardrails, but A dominates B on this seed while preserving the original parameter count and showing shorter runtime. Do not prioritize B for five-seed promotion now; reserve a five-seed confirmation for a later architecture-focused question. Do not claim B merits promotion from this pilot.
3. **Kinematic consistency loss** follows after head/feature results. Its risk is bias under process noise, so keep λ small and inspect late-horizon and maneuver-relevant validation bins.
4. **M2 weight 2** is not recommended for full-seed promotion by itself: the one-seed pilot improved acceleration modestly but also worsened position, velocity, and aggregate MSE. The numeric guardrail in this plan was set afterward, so this is not a preregistered rejection. Preserve it as a pilot, not as evidence for deployment.
5. **Architecture sweeps** are lower priority than feature scaling and output specialization. The current baseline converges across seeds, and no evidence yet shows that depth/width is the primary acceleration bottleneck.

The five-seed test comparison shows lower mean velocity, acceleration, and six-state MSE for A, with position effectively tied to M0. No formal inferential test was specified, so describe the paired means and sample standard deviations without a statistical-significance claim. The five-seed EMA `.50` setting remains a reasonable independently evaluated alternate.

## 5. Configuration registry

Each file is self-contained or inherits the frozen baseline by an explicit `baseline_config` reference. Existing result-producing configs are marked completed or planned as described in their metadata. The weight-2 pilot, two seed-11 screens, and A five-seed confirmation have separate configs and versioned result directories.

- Frozen baseline: `configs/paper_v4_mamba_improvement/baseline_reference.json`
- Loss: `loss_statewise_standardized_existing.json`, `loss_acceleration_weight_2_pilot.json`, `loss_acceleration_weight_4_existing.json`, `loss_acceleration_weight_8_existing.json`, `loss_temporal_consistency.json`
- Features: `features_train_only_normalization_seed11.json` (pilot), `features_train_only_normalization_five_seed_confirmation_v2.json` (five-seed and test complete), `initial_state_first_token.json`, `features_ema_acceleration_025_existing.json`, `features_ema_acceleration_050_existing.json`, `features_quadratic_w5_existing.json`, `features_quadratic_w9_existing.json`, `adaptive_acceleration_narrow_existing.json`, `adaptive_acceleration_broad_existing.json`
- Head/output: `head_separate_state_groups_seed11.json` (screen complete), `head_train_scaled_residuals.json`
- Mamba: `mamba_embed_96.json`, `mamba_embed_160.json`, `mamba_dstate_8.json`, `mamba_blocks_2.json`, `mamba_blocks_4.json`, `mamba_dropout_012.json`, `head_width_128.json`
- Context: `context_window_50.json`

The weight-2 runner is [pilot_loss_accel_w2.py](../../experiments/paper_v4_mamba_improvement/pilot_loss_accel_w2.py). The seed-11 screens used [run_screening_pilot.py](../../experiments/paper_v4_mamba_improvement/run_screening_pilot.py); five-seed validation used [run_feature_norm_five_seed_validation.py](../../experiments/paper_v4_mamba_improvement/run_feature_norm_five_seed_validation.py); final test evaluation was gated by [evaluate_feature_norm_five_seed_test.py](../../experiments/paper_v4_mamba_improvement/evaluate_feature_norm_five_seed_test.py). All new outputs are versioned, and the final test runner verifies the frozen selection/config/checkpoint hashes before opening test data. The historical weight-2 screen was not rerun.
