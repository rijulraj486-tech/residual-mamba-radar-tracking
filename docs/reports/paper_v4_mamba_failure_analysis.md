# Paper v4 Mamba Failure Analysis

**Scope:** audit of the preserved five-seed synthetic radar model and subsequent controlled loss, feature, and output-head experiments. This report distinguishes confirmed implementation facts from plausible mechanisms; it does not treat association as causation. The fixed test set is not used for choosing new configurations.

## Baseline and evidence

The reproducible reference is `five_seed_n1024_fixedpool_v1`: five Mamba seeds (11, 22, 33, 44, 55), 819 training and 204 validation trajectories, length 100, and one fixed 1,024-trajectory test pool. The shared test file SHA-256 is `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`; the train/validation file SHA-256 is `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733` ([fixed_pool_manifest.json](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json)). Independent checkpoint reevaluation passed ([checkpoint_evaluations.json](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/independent_verification/checkpoint_evaluations.json)).

| Metric | Mamba mean ± sample SD | KalmanNet mean ± sample SD |
|---|---:|---:|
| Position RMSE | 332.08 ± 1.42 m | 401.27 ± 16.85 m |
| Velocity RMSE | 79.26 ± 2.33 m/s | 94.77 ± 2.02 m/s |
| Acceleration RMSE | 66.71 ± 4.23 m/s² | 14.17 ± 0.58 m/s² |
| Six-state scaled MSE | 0.020172 ± 0.000212 | 0.028405 ± 0.002339 |

Source: [across_seed_summary.csv](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/summary/across_seed_summary.csv). Mamba has better position and velocity RMSE than the saved KalmanNet baseline, but its acceleration RMSE is much worse. This comparison says nothing about superiority to the calibrated EKF/UKF, which have a different information configuration and are not the focus of this failure analysis.

## Findings by proposed cause

### 1. Measurement noise is amplified by numerical differentiation — confirmed mechanism; contribution to final model error not isolated

The input range/bearing data are converted to Cartesian positions, then coarse velocity is computed by a first difference and coarse acceleration by a second difference. For independent per-sample Cartesian position error with standard deviation `σp`, first differencing has standard deviation `sqrt(2) σp / dt`; second differencing has `sqrt(6) σp / dt²`. This follows directly from the `[1,-1]` and `[1,-2,1]` coefficients.

Here the generator uses 30 m range noise and 1° bearing noise at target ranges of about 20–30 km ([data_generation.py](../../src/radar_tracking/data/data_generation.py#L120); [fixed_pool_manifest.json](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json)). A first-order polar-to-Cartesian approximation gives transverse position noise near `r σθ`, about 349–524 m over that range. At `dt=1 s`, its second-difference contribution alone is roughly 855–1,284 m/s² standard deviation. This is an approximation to the raw derived feature, not a measured error variance for the neural network output: the network has access to the full causal sequence and learns a correction. It does explain why the coarse acceleration reference is much noisier than position and why acceleration is a difficult residual target.

The six feature channels for coarse acceleration are computed from those finite differences, with the provided initial acceleration repeated at the first two indices ([tracker_model.py](../../src/radar_tracking/models/tracker_model.py#L149)). No raw measurement-level noise realization is explicitly passed as a feature; uncertainty information is not represented in the baseline token.

The train-only feature-statistics artifact from screen A quantifies the mixed input scales over 819×100 training tokens. Examples: range std 11.899, Cartesian x/y std 13.229/12.498, bearing sine/cosine std 0.252/0.287, wrapped bearing-difference std 0.0306, Cartesian differences std 0.786/0.760, coarse acceleration std 1.219/1.183, and repeated initial acceleration has zero variance. These are raw feature units (including km-based state features); the model receives the 21 token channels directly before its input projection. The values and feature names are in [A train-only feature statistics](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_a_feature_norm_seed11/train_only_feature_statistics.json). Thus the “scale mismatch” is a measured numerical spread across heterogeneous channels, not an assertion that unlike physical dimensions should share one physical unit. A normalized only dimensional channels using `(x−train_mean)/max(train_population_std, 1e−4)` and left `sin_bearing`, `cos_bearing`, and `delta_bearing_wrapped` unchanged; these train statistics were frozen for validation ([variant implementation](../../experiments/paper_v4_mamba_improvement/model_variants.py), [A config](../../configs/paper_v4_mamba_improvement/features_train_only_normalization_seed11.json)).

### 2. Unequal scales and loss contribution — confirmed design; likely contributor, not sole cause

The generator and benchmark scale all six states by 1,000. Typical scaled magnitudes are approximately 20–30 for position, 0.2 for velocity, and 0.003 for acceleration, while sine/cosine bearing features are order one. The token is linearly projected and LayerNorm is applied after that projection; this does not guarantee equal influence of input channels ([tracker_model.py](../../src/radar_tracking/models/tracker_model.py#L50); [benchmark_mamba.py](../../src/radar_tracking/cli/benchmark_mamba.py#L486)).

The loss is plain mean squared error over batch, all six states, and time ([benchmark_mamba.py](../../src/radar_tracking/cli/benchmark_mamba.py#L728)). It gives each coordinate the same coefficient, but larger position errors dominate the realized squared loss. The saved five-seed coordinate analysis attributes approximately 91% of the total coordinate squared error to position, about 5% to velocity, and about 4% to acceleration. This is a property of residual magnitudes, not an explicit position-only weighting in code ([mamba_radar_tracking_diagnostic_audit.md](mamba_radar_tracking_diagnostic_audit.md)).

The completed loss ablation is direct evidence that objective choice changes the trade-off:

| Loss setting | Test position RMSE (m) | Test velocity RMSE (m/s) | Test acceleration RMSE (m/s²) | Six-state MSE |
|---|---:|---:|---:|---:|
| Original MSE | 332.08 ± 1.42 | 79.26 ± 2.33 | 66.71 ± 4.23 | 0.020172 ± 0.000212 |
| Group loss, acceleration weight 4 | 342.14 ± 2.56 | 92.65 ± 1.85 | 56.96 ± 8.02 | 0.021491 ± 0.000304 |
| Group loss, acceleration weight 8 | 348.61 ± 2.08 | 98.65 ± 1.88 | 50.57 ± 5.76 | 0.022308 ± 0.000330 |

The weight-8 result lowers acceleration RMSE by about 24% but worsens position, velocity, and aggregate MSE. The train-standardized loss is not a good direct replacement: on validation it reached 698.70 m position, 510.46 m/s velocity, and 48.30 m/s² acceleration, versus M0 at 329.79 m, 79.07 m/s, and 64.25 m/s². Sources and selection protocol: [mamba_loss_ablation_v1.md](mamba_loss_ablation_v1.md), [loss-ablation artifacts](../../artifacts/results/synthetic/mamba_loss_ablation_v1/).

Conclusion: loss imbalance is a plausible, experimentally supported part of the failure, but simply standardizing all six coordinates or applying a large acceleration weight harms other state groups. A narrower acceleration weight and explicit group metrics are justified next steps.

A new validation-only pilot tested group acceleration weight 2 with seed 11, keeping the 619,014-parameter architecture and training pool unchanged. Relative to that seed's preserved M0 validation checkpoint, position RMSE changed from 331.87 to 337.60 m, velocity from 81.86 to 83.92 m/s, acceleration from 67.90 to 63.50 m/s², and six-state MSE from 0.020241 to 0.020842. It trained for 138 epochs and selected epoch 113. This is a modest acceleration gain with worse position, velocity, and aggregate MSE. The test set was not opened; one seed cannot support a promotion claim. Full provenance is in [pilot_summary.json](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v1/pilot_loss_accel_w2/pilot_summary.json). The result lowers the priority of a standalone weight-2 five-seed run under a no-regression objective.

### 3. Acceleration feature quality — strongest existing intervention evidence

The current shared head must correct a noisy second-difference acceleration feature. Existing five-seed feature ablations changed only those acceleration token channels while preserving architecture and objective. Fixed EMA `α=.25` achieved 34.17 ± 2.65 m/s² acceleration RMSE, compared with M0's 66.71 ± 4.23, while its velocity RMSE rose from 79.26 ± 2.33 to 94.85 ± 9.42 m/s. EMA `.50` and trailing quadratic features occupy intermediate trade-offs. The validation-selected adaptive setting achieved 52.86 ± 1.70 m/s² acceleration RMSE, 81.74 ± 5.00 m/s velocity, and 330.37 ± 1.14 m position ([mamba_acceleration_feature_ablation_v1.md](mamba_acceleration_feature_ablation_v1.md); [mamba_adaptive_acceleration_v1.md](mamba_adaptive_acceleration_v1.md)).

This shows that feature engineering can substantially improve acceleration in this regime. It does not show that smoothing is universally best, nor that the improvement generalizes to maneuvers, other noise levels, or different sampling intervals. EMA also raises velocity error; all variants require paired multi-seed selection and full reporting.

### 4. Shared residual head — one screen supports further study, but does not isolate causality

The encoder output goes through the baseline shared MLP `128→256→128→6`; its six outputs are added to the coarse state ([prediction_head.py](../../src/radar_tracking/models/prediction_head.py#L4); [tracker_model.py](../../src/radar_tracking/models/tracker_model.py#L234)). The seed-11 B screen replaced this head with three independent `128→128→64→2` SiLU MLPs for position, velocity, and acceleration. Their outputs are reordered to `[x,vx,ax,y,vy,ay]` and added to the same coarse state. The causal encoder and training objective stayed fixed. Parameter count increased from 619,014 to 627,014 (+8,000; +1.29%) ([B config](../../configs/paper_v4_mamba_improvement/head_separate_state_groups_seed11.json), [model variants](../../experiments/paper_v4_mamba_improvement/model_variants.py)).

On the fixed seed-11 validation pool, B achieved position 330.636 m, velocity 76.484 m/s, acceleration 60.028 m/s², and six-state MSE 0.019796, versus paired M0's 331.870 m, 81.860 m/s, 67.899 m/s², and 0.020241. All four improved on this seed, but normalization screen A did better in every metric (328.998 m, 72.635 m/s, 49.268 m/s², 0.019324) at baseline parameter count and with shorter observed timing. Full results, checkpoint-selection epochs, runtimes, and checkpoint provenance are in the [improvement comparison](paper_v4_mamba_improvement_plan.md) and the [B pilot summary](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_b_separate_heads_seed11/pilot_summary.json).

This one seed is consistent with useful specialization but cannot show that shared-head interference caused baseline error. B is secondary to confirming normalization across five seeds; do not combine them before separate effects are established. The test set was not loaded for either screen (`test_accessed=false`).

### 5. Convergence and validation checkpoint selection — generally appropriate for aggregate objective; incomplete for acceleration

The five Mamba checkpoints selected validation MSE at epochs 147, 169, 180, 156, and 156. Train and validation MSE at those epochs are close, and no persistent gap indicates gross aggregate overfitting. Three runs stopped before the 180-epoch cap; two reached the cap. Training uses Adam (`1e-3`, weight decay `1e-4`), batch 16, gradient-norm clipping at 1.0, patience 25, and `min_delta=1e-6`, with no learning-rate scheduler ([five_seed_harness.py](../../experiments/five_seed_harness.py#L354); seed histories in `run_metrics.json`).

Validation selects on aggregate six-state MSE. Since position contributes most of that objective, the chosen checkpoint is not necessarily the best acceleration checkpoint. Baseline epoch logs do not retain per-coordinate losses or gradient clipping frequency; later ablation runners do record per-state/group validation values. Improvement experiments should select on a prespecified validation rule that reports the full Pareto trade-off rather than selecting only acceleration or aggregate MSE after inspecting test results.

### 6. Temporal context — full prefix is available, sufficiency unproven

At each step, the forward-only Mamba processes the causal sequence prefix; there is no explicit fixed receptive-field window in the baseline. Three Mamba blocks have a finite learned recurrent state, and the output is not fed back as the next state ([bimamba_encoder.py](../../src/radar_tracking/models/bimamba_encoder.py#L37)). The saved temporal analysis shows errors rising late in the 100-step horizon, especially acceleration, but no controlled context-length or memory ablation exists ([mamba_radar_tracking_diagnostic_audit.md](mamba_radar_tracking_diagnostic_audit.md)).

It is therefore incorrect to conclude either that context is insufficient or that the Mamba has successfully exploited all 100 observations. The relevant follow-up is a causal context ablation with clearly defined window resets and identical information at each prediction, followed by validation-only selection. The model's theoretical ability to process long sequences is not evidence of sufficient learned memory.

## New controlled validation screens (seed 11 only)

Both pilots used the fixed 819/204 train/validation split (pool hash `4ac64b…a2733`) and the same preserved seed-11 M0 checkpoint (SHA-256 `b3d90c…10645`). Each used validation six-state MSE for checkpoint selection and explicitly recorded `test_accessed=false` and `test_file_not_loaded=true`. They are independent one-factor screens, not test results or multi-seed evidence. Full exact hashes and metrics are recorded in the [A summary](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_a_feature_norm_seed11/pilot_summary.json) and [B summary](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_b_separate_heads_seed11/pilot_summary.json).

| Seed-11 validation candidate | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Parameters | Train s | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paired M0 | 331.870 | 81.860 | 67.899 | 0.020241 | 619,014 | Not recorded in this paired timing artifact | 147 |
| A: train-only input normalization | 328.998 | 72.635 | 49.268 | 0.019324 | 619,014 | 52.19 | 76 (101 run) |
| B: separate group residual heads | 330.636 | 76.484 | 60.028 | 0.019796 | 627,014 | 101.56 | 174 (180 run) |

A and B each improved all four metrics over paired M0 and meet the plan's prospective validation guardrails. A dominates B in every metric, uses 8,000 fewer parameters, and had lower training and warmed validation-inference time in this run. The measured inference times (CUDA, 204 examples) were 0.03902/0.04014 s for paired M0/A and 0.03834/0.04339 s for paired M0/B; short pass timing is descriptive, not deployment latency. The prospective guardrails were established after the earlier weight-2 pilot, not preregistered for it.

## Five-seed confirmation and held-out evaluation of feature normalization

After the seed-11 screen, train-only normalization was retrained using the exact pilot model, loss, optimizer, scheduler-free training loop, and feature transform for seeds 11/22/33/44/55. Each seed fit its own population means/stds using only the 819 training trajectories; the fixed 204 validation sequences selected checkpoints. All runs retained 619,014 parameters. Mean validation results were position 330.903 ± 2.807 m, velocity 72.847 ± 3.100 m/s, acceleration 45.893 ± 6.634 m/s², and six-state MSE 0.019493 ± 0.000454. Paired M0 validation means were 329.790 ± 1.639 m, 79.069 ± 2.408 m/s, 64.249 ± 4.198 m/s², and 0.019860 ± 0.000266. The +1% mean position / +3% mean velocity guardrails passed (+0.338% position, −7.870% velocity). The [validation aggregate](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/validation_aggregate.json) and [frozen selection manifest](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/validation_selection_manifest.json) record exact per-seed values, checkpoints, hashes, and selection status.

Only after those selection and checkpoint hashes were frozen was the unchanged held-out pool opened. The saved M0 checkpoints reproduced the archived five-seed test summary. On test, M0 had position 332.085 ± 1.418 m, velocity 79.256 ± 2.325 m/s, acceleration 66.713 ± 4.235 m/s², and six-state MSE 0.020172 ± 0.000212. Normalized A had position 331.994 ± 3.088 m, velocity 72.276 ± 3.501 m/s, acceleration 45.356 ± 6.778 m/s², and six-state MSE 0.019593 ± 0.000497. Paired candidate-minus-M0 differences were −0.091 ± 3.682 m, −6.980 ± 5.073 m/s, −21.358 ± 2.763 m/s², and −0.000580 ± 0.000524 MSE. Position is effectively parity, with mixed per-seed signs; the measured gains are concentrated in velocity and acceleration. No inferential significance test was predeclared or performed, and this evidence covers only the existing synthetic pool/noise/motion regime.

Training time averaged 63.33 ± 20.60 s per seed. Warmed, synchronized CUDA inference over the 1,024-example test pool averaged 0.19467 ± 0.00101 s. The [final comparison artifact](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/final_test_evaluation_v1/final_test_comparison.json) and [paired per-seed CSV](../../artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2/paired_seed_comparison_v1.csv) include test hash, configuration/source/checkpoint provenance, metrics, and timings. Experiment B was not run across five seeds and no long-sequence experiment was started.

## Cause assessment

| Candidate cause | Evidence level | Assessment |
|---|---|---|
| Second differences amplify noisy radar-derived position | Mechanism confirmed; output contribution not isolated | Strong, high-priority feature experiment. Existing EMA results improve acceleration substantially. |
| Aggregate objective under-emphasizes acceleration in practice | Supported by loss distribution and ablation | Full coordinate standardization and group weights 2/4/8 were screened; no setting is promoted because acceleration gains traded against other metrics. |
| Feature scaling | Mixed token-channel scales confirmed; five-seed/test evidence supports velocity and acceleration gains | A lowers mean held-out velocity and acceleration RMSE and six-state MSE; position is effectively unchanged. Scope is limited to the fixed synthetic regime. |
| Shared residual-head limitation | Still unresolved; B screen is promising but dominated by A on seed 11 | B improved all four metrics on one seed at +1.29% parameters, but cannot establish causality or multi-seed benefit. Treat as secondary confirmation. |
| Poor convergence / wrong selected epoch | Weakly supported | Curves converge similarly across seeds, but aggregate checkpoint metric can miss acceleration optima. Add group-specific validation history. |
| Insufficient temporal memory | Unresolved | Full 100-step causal prefix is processed; late-horizon errors rise, but no context ablation isolates memory. |

## Practical conclusion

The evidence points to noisy second-difference acceleration features and an aggregate objective dominated by position residual magnitude. Feature-only EMA experiments show that acceleration error can fall substantially, often with a velocity cost. The weight-2 loss pilot worsened position, velocity, and aggregate MSE. Train-only feature normalization (A) passed the five-seed validation guardrails and the frozen held-out test evaluation reduced mean velocity RMSE, acceleration RMSE, and six-state MSE; position was effectively tied to M0. The seed-11 separate-head result remains only a pilot, with no five-seed confirmation. These results support A for this specific synthetic regime but do not establish robustness to new radar noise, motion, or sequence length. Do not combine A with other factors until separately evaluated.

The baseline remains unchanged and reproducible at `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/`.
