# Diagnostic Audit: Existing Mamba Radar Tracker

**Scope:** Read-only audit of the synthetic five-seed reference experiment `five_seed_n1024_fixedpool_v1` and its implementation as found in this checkout. No source, checkpoint, or historical result was changed; no training was run. Saved checkpoints were loaded for inference against the existing shared test set to calculate coordinate-wise error and temporal summaries.

## Executive findings+-

- **Confirmed:** The model predicts the complete six-value state at every time step as a residual added to a measurement-derived coarse state. It is a causal, forward-only Mamba for the five-seed experiment.
- **Confirmed:** The training objective is unweighted elementwise MSE across all six scaled state coordinates. It does not balance position, velocity, and acceleration by physical scale or task weight.
- **Confirmed:** Across five seeds, Mamba has low position and velocity RMSE relative to the KalmanNet reference, but much higher acceleration RMSE: **66.71 m/s² (SD 4.23; 95% t interval 61.46–71.97)** versus KalmanNet’s 14.17 m/s². These are reported in the preserved baseline summary.
- **Confirmed:** The Mamba training and validation losses usually converge closely, and the selected checkpoints are late in training (epochs 147–180 of a cap of 180). The stored curves do not show a persistent train/validation gap that would establish overfitting.
- **Strong hypothesis:** Acceleration is disadvantaged because raw common scaling leaves position residuals dominant in the scalar MSE, while the coarse acceleration input is formed by second differences of noisy polar measurements converted to Cartesian coordinates. Both points are directly visible in code; their causal contribution requires controlled experiments.
- **Confirmed limitation:** The fixed synthetic benchmark samples one motion/noise regime and one sequence length. It does not test robustness across acceleration, jerk, sampling interval, sensor noise, or motion profiles.

## Architecture and state path

### Encoder and head

The five-seed model has 619,014 parameters and uses `embed_dim=128`, three residual Mamba blocks, `d_state=16`, `d_conv=4`, `expand=2`, dropout 0.08, and a 256-unit two-hidden-layer prediction head. Each residual block applies LayerNorm, a Mamba sequence mixer, a residual connection, then a LayerNorm/SiLU feed-forward sub-block and another residual connection ([bimamba_encoder.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/bimamba_encoder.py:6), [tracker_model.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/tracker_model.py:38)). The input projection is linear to 128 features, LayerNorm, then SiLU ([tracker_model.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/tracker_model.py:65)).

At each sample, the 21 input features are: range; sine and cosine of bearing; Cartesian x/y derived from range-bearing; range, wrapped-bearing, Cartesian-x and Cartesian-y first differences; six coarse state values; and the six initial-state values repeated at every time step ([tracker_model.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/tracker_model.py:181)).

The coarse state uses measured Cartesian positions. Velocity is a first difference, with initial velocity supplied at the first sequence index. Acceleration is a difference of coarse velocity, with initial acceleration supplied at index zero and copied to index one ([tracker_model.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/tracker_model.py:149)). The head emits six deltas and adds them to that time step’s coarse state ([prediction_head.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/prediction_head.py:18)). Thus it estimates a **correction to a reference state**, not a free-standing direct state. The six outputs are still the complete estimated state.

The physical state order is `[x, vx, ax, y, vy, ay]`, with the constant-acceleration transition matrix in [constant_acceleration.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/dynamics/constant_acceleration.py:17).

### Causality

The reference config is `encoder_direction="forward"`; `BiMambaEncoder.forward` applies its Mamba blocks in sequence order and does not reverse or concatenate a backward stream ([five_seed_harness.py](/home/spcom41/BiMambaTracker/radar-tracking/experiments/five_seed_harness.py:354), [bimamba_encoder.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/bimamba_encoder.py:80)). Current and prior observations are used for each token; no future observation is explicitly included. This is consistent with causal tracking. Coarse differences are aligned to the current index and use only current and earlier measurements.

There is an API/configuration weakness: `encoder_direction="bidirectional"` is accepted by the encoder and CLI but the encoder unconditionally sets its effective direction to `"forward"` and constructs only forward layers ([bimamba_encoder.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/bimamba_encoder.py:61), [benchmark_mamba.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/cli/benchmark_mamba.py:405)). The five-seed baseline records effective direction as forward, so this does not invalidate that baseline.

## Training pipeline

The five-seed harness uses a common fixed pool: 819 train trajectories, 204 validation trajectories, and 1 unused trajectory, each length 100. The model uses batches of 16, Adam at 1e-3, weight decay 1e-4, a 180 epoch cap, early-stopping patience 25, and validation improvement threshold 1e-6 ([five_seed_harness.py](/home/spcom41/BiMambaTracker/radar-tracking/experiments/five_seed_harness.py:263), [five_seed_harness.py](/home/spcom41/BiMambaTracker/radar-tracking/experiments/five_seed_harness.py:354)).

Training minimizes `torch.nn.MSELoss(reduction="mean")` over the whole `[batch, 6, time]` state output. Gradients are backpropagated once per batch, clipped by global norm to 1.0, then Adam updates parameters. Validation uses the same MSE in evaluation mode. The best validation checkpoint is saved; patience stops training after 25 consecutive misses. There is **no learning-rate scheduler** in the Mamba training path. Dropout 0.08 and Adam weight decay 1e-4 are the explicit regularizers ([benchmark_mamba.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/cli/benchmark_mamba.py:728)). There is no logged per-coordinate loss or acceleration-specific term.

## Error analysis across six state dimensions

The original reports give only combined position, velocity, and acceleration metrics. For this audit, all five stored Mamba checkpoints were evaluated on the exact existing 1,024-example shared test set. The table reports the square root of the mean per-seed coordinate MSE, converted back to physical units. This enables axis comparison but does not imply every seed has exactly these errors.

| State coordinate | Mean-seed coordinate RMSE | Mean signed bias across seeds |
|---|---:|---:|
| x position | 239.42 m | -1.69 m |
| x velocity | 56.24 m/s | -0.10 m/s |
| x acceleration | 51.52 m/s² | -7.75 m/s² |
| y position | 230.13 m | -11.12 m |
| y velocity | 55.89 m/s | -11.53 m/s |
| y acceleration | 42.55 m/s² | +2.44 m/s² |

Combined state metrics from the original five-seed summary are:

| Quantity | Mamba mean (SD) | KalmanNet mean (SD) |
|---|---:|---:|
| Position RMSE | 332.08 m (1.42) | 401.27 m (16.85) |
| Velocity RMSE | 79.26 m/s (2.33) | 94.77 m/s (2.02) |
| Acceleration RMSE | 66.71 m/s² (4.23) | 14.17 m/s² (0.58) |
| Six-state scaled MSE | 0.02017 (0.00021) | 0.02841 (0.00234) |

Source: [across_seed_summary.csv](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/summary/across_seed_summary.csv) and [per_seed_metrics.csv](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/summary/per_seed_metrics.csv). The separate independent verification reports a passing checkpoint reevaluation and a consistent test-set hash ([checkpoint_evaluations.json](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/independent_verification/checkpoint_evaluations.json)).

Position contributes about 91% of aggregate coordinate squared error under the existing loss, velocity about 5%, and acceleration about 4% (computed from the six coordinate mean-seed MSEs above). This is a consequence of the observed residual scales, not a claim that the MSE implementation assigns different explicit weights.

The mean signed errors across seeds are small relative to their coordinate RMSEs, so a common constant acceleration bias is not sufficient to explain the acceleration result. Bias differs by seed and axis; x-acceleration bias ranges from about -30 to +125 m/s², while y-acceleration bias ranges from about -245 to +65 m/s². The acceleration error is predominantly variance/error magnitude rather than a stable shared offset, although a few individual seed/axis combinations have substantial bias.

## Acceleration and temporal behaviour

The derived acceleration reference is a second difference of Cartesian positions, where each Cartesian position comes from noisy range and bearing. The generator uses 30 m range noise and 1 degree bearing noise at ranges of roughly 20–30 km ([data_generation.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/data/data_generation.py:131), [data_generation.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/data/data_generation.py:219)). Bearing noise at this range creates large lateral position perturbations; differencing those positions twice amplifies high-frequency measurement error. This makes the coarse acceleration a difficult and noisy residual reference. **The amplification is a mathematical property of differencing; its exact share of model RMSE has not been isolated.**

Across all five saved models, six-state RMSE by consecutive ten-step bins (physical units, pooled over examples, states, and seeds) was approximately **159, 128, 123, 129, 128, 130, 136, 143, 153, 182**. The errors are relatively stable through the middle, then rise late in the sequence. This supports late-horizon degradation, but does not alone prove a recurrent error-feedback mechanism. The model emits each estimate from a causal encoded prefix and the coarse measurement reference; it does not feed its prior predicted state back as the next state.

Pooled ten-step acceleration RMSE across both axes is approximately **48, 28, 28, 29, 30, 34, 38, 48, 60, 91 m/s²**. It is lowest in the middle and rises sharply over the final 30 steps. This summary pools examples, axes, and seeds within each bin; a follow-up should retain per-axis and per-seed time curves. The synthetic trajectories evolve under constant-acceleration dynamics with integrated jerk process noise, so estimation difficulty can accumulate even without autoregressive model feedback.

## Training and validation behaviour

Stored epoch histories show a large expected drop in MSE during the first 10 epochs and a gradual plateau later. Best validation epochs are 147, 169, 180, 156, and 156 for seeds 11, 22, 33, 44, and 55 respectively. At each best epoch, train MSE is close to validation MSE:

| Seed | Best epoch | Train MSE at best | Validation MSE at best |
|---:|---:|---:|---:|
| 11 | 147 | 0.02094 | 0.02024 |
| 22 | 169 | 0.02043 | 0.01966 |
| 33 | 180 | 0.01994 | 0.01959 |
| 44 | 156 | 0.02069 | 0.01981 |
| 55 | 156 | 0.02038 | 0.02000 |

Two seeds use the full 180-epoch cap, three stop earlier. Source histories are embedded in each seed’s `run_metrics.json` (for example [seed 11](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/seed_11/run_metrics.json:442)).

**Assessment:** there is no clear sustained overfit. The curves do not suggest gross underfitting in aggregate because both reach a similar plateau, and the held-out aggregate score is stable across seeds. However, aggregate MSE can conceal acceleration underfitting because position accounts for most of the loss. Validation loss has occasional brief increases during descent in a few seeds, then recovers; this is modest optimization variability, not evidence of general instability. Gradient clipping at 1.0 is active, but the harness does not record unclipped gradient norms or clipping frequency.

## Data distribution and scaling audit

The fixed pool and test set share the same configuration: length 100, `dt=1`, initial x/y positions independently uniform from 20–30 km, velocities uniform from -200 to +200 m/s, and accelerations uniform from -3 to +3 m/s². State evolution uses a constant-acceleration transition plus Gaussian integrated-white-jerk process noise; observations are noisy range and bearing with fixed 30 m and 1 degree standard deviations. Train/validation trajectories are generated once and then held fixed across model seeds ([data_generation.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/data/data_generation.py:31), [data_generation.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/data/data_generation.py:155), [fixed_pool_manifest.json](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json)).

All six state coordinates are divided by 1,000, ranges are divided by 1,000, bearings remain in radians, and physical RMSE metrics multiply state error by 1,000 ([benchmark_mamba.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/cli/benchmark_mamba.py:492), [five_seed_harness.py](/home/spcom41/BiMambaTracker/radar-tracking/experiments/five_seed_harness.py:475)). The stored checkpoint metrics agree with an independent reevaluation, so **no confirmed unit-conversion mismatch was found**.

However, common scaling is not dimension normalization. After division, typical raw coordinate magnitudes still differ greatly: positions are about 20–30, velocities about 0.2, and accelerations about 0.003. Bearing trigonometric features are order one. The initial linear projection receives this mixture before LayerNorm, and loss uses unweighted MSE. LayerNorm after projection does not guarantee equal task influence. This is a confirmed design property; that it causes the acceleration gap is a hypothesis. Also, `range_std` and `bearing_std` are configured in physical measurement units before range is scaled; source flow applies range scaling consistently for train/test.

## Findings by confidence

### Confirmed implementation or methodology issues

1. **Loss scale is not balanced by state type.** One scalar MSE averages coordinates with very different physical ranges and usefulness; no per-component normalization or explicit velocity/acceleration weight exists.
2. **The noisy coarse reference uses finite differences through second order.** Acceleration’s baseline feature has greater sensitivity to measurement noise than position, and the head must learn a correction to it.
3. **Only one synthetic operating condition is evaluated.** Noise levels, dt, sequence length, initialization quality, and motion profile do not vary in the fixed baseline.
4. **Training diagnostics are aggregate only.** No component-wise train/validation loss, bias, gradient-norm, or horizon-specific metric is saved during optimization.
5. **Direction flag is misleading outside this baseline.** `bidirectional` is accepted but effective processing stays forward. The reference five-seed configuration explicitly selects forward, so baseline behavior is clear.

### Hypotheses to test

1. The unweighted objective allocates too little optimization pressure to acceleration relative to position.
2. Second-difference noise in the coarse acceleration is the dominant source of poor acceleration estimates.
3. Raw input scale mismatch makes position and Cartesian features dominate the input projection despite the LayerNorm that follows it.
4. Training only 819 fixed trajectories causes poor coverage of motion/noise combinations, especially when observations are angularly noisy.
5. Late-horizon deterioration reflects accumulated uncertainty or distribution shift over time rather than prediction feedback.

## Ranked recommendations

1. **First add per-state diagnostics and a training-only validation study.** Log six coordinate losses, position/velocity/acceleration group losses, horizon bins, signed bias, and gradient clipping statistics. Keep the current five-seed result immutable as reference.
2. **Ablate loss normalization/weighting.** Compare current unweighted MSE against target-standardized per-coordinate MSE and group-weighted MSE. Select weights from training statistics only; report all metrics in physical units. This directly tests the imbalance hypothesis.
3. **Ablate the acceleration reference.** Compare current second-difference coarse acceleration with a smoothed/regularized estimate and with a learned/direct acceleration head, retaining the same causal observations and data split. Evaluate whether acceleration improves without sacrificing position/velocity.
4. **Normalize input channels deliberately.** Use train-only statistics or physically meaningful scales per feature group, including range/cartesian, differences, trigonometric bearings, and state coordinates; apply identical saved transforms to validation/test.
5. **Expand synthetic coverage in a controlled curriculum or factorial sweep.** Vary range/bearing noise, process-noise scale, acceleration magnitude, dt, initial velocity, and sequence length. Keep a fixed held-out test matrix so gains are not tied to the current single regime.
6. **Tune optimization after diagnostics.** Compare learning-rate schedules and clipping thresholds only after measuring the existing gradient behavior. The current Adam rate and weight decay are stable enough to establish a baseline, but no scheduler is used.
7. **Clarify the encoder direction interface.** Treat forward as the online tracking baseline; if a bidirectional option is intended for offline smoothing, implement and test it as a separate explicit variant rather than relying on the current accepted-but-ignored flag.

## Proposed experiment plan

Preserve `five_seed_n1024_fixedpool_v1` and its hashes as the original reference. Do not reuse its result directory for variants.

1. **Freeze protocol:** use the original train/validation pool and test set first; store new variants under a new versioned experiment directory. Record source/config/data hashes and all six coordinate metrics. Use seeds 11, 22, 33, 44, 55 to compare directly.
2. **Baseline reproduction check:** rerun the original Mamba config under a new output name and confirm it reproduces the current curve/metric range before any ablation.
3. **One-factor ablations:**
   - A: original loss and inputs;
   - B: standardized per-coordinate loss;
   - C: explicit acceleration/group weighting;
   - D: normalized input channels only;
   - E: smoothed coarse acceleration only;
   - F: direct six-state residual head versus separate state-group heads.
4. **Use validation only for model selection.** Pick normalization statistics and loss weights from train; use validation to choose checkpoints and hyperparameters. Keep the shared test untouched until final comparison.
5. **Report:** per-axis RMSE and bias, grouped physical RMSE, per-time/horizon RMSE, train/validation component losses, seed mean/SD and paired differences, and robustness across the expanded motion/noise grid.
6. **Promotion criterion:** prefer a variant only if it materially lowers acceleration RMSE across seeds and conditions without unacceptable regression in position/velocity or an unstable seed tail. Compare paired seed differences against the preserved original five-seed baseline.

## Source and artifact locations

- Architecture and feature construction: [tracker_model.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/tracker_model.py), [bimamba_encoder.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/bimamba_encoder.py), [prediction_head.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/models/prediction_head.py).
- Training loop and scaling: [benchmark_mamba.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/cli/benchmark_mamba.py).
- Generator and dynamics: [data_generation.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/data/data_generation.py), [constant_acceleration.py](/home/spcom41/BiMambaTracker/radar-tracking/src/radar_tracking/dynamics/constant_acceleration.py).
- Five-seed settings and metric calculation: [five_seed_harness.py](/home/spcom41/BiMambaTracker/radar-tracking/experiments/five_seed_harness.py).
- Preserved baseline metrics and histories: [summary/](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/summary), [seed run records](/home/spcom41/BiMambaTracker/radar-tracking/artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/seed_11/run_metrics.json).
