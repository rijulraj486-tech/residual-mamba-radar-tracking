# Baseline Methodology Audit: EKF, UKF, KalmanNet, and Causal Mamba

**Scope:** read-only audit and experiment planning. No filter/model source, checkpoint, dataset, manifest, or result was modified. No model was trained and no new baseline was evaluated.

## 1. Existing baseline inventory

| Baseline | Implementation | Domain and current use |
|---|---|---|
| Synthetic radar EKF | `src/radar_tracking/models/ekf.py`: `extended_kf`, `batch_ekf` | Six-state 2D constant-acceleration model with either full-state, position-only, or range-bearing observations. `src/radar_tracking/cli/benchmark_mamba.py` evaluates this EKF in the synthetic/real-world benchmark path. |
| NCLT wheel/GPS EKF | `src/radar_tracking/cli/nclt_benchmark.py`: `NCLTEKF` | Separate six-element state and wheel-input/GPS-position update; not the radar range-bearing model and not comparable as the same EKF configuration. Uses fixed diagonal Q, R, and P. |
| KalmanNet | `src/radar_tracking/models/kalman_net.py` | Learned gain update with the constant-acceleration `f` and range-bearing `h`; current five-seed checkpoint campaign has five models. |
| Causal Mamba tracker | `src/radar_tracking/models/tracker_model.py`, `bimamba_encoder.py` | Sequence estimator; the five-seed records use `encoder_direction="forward"`. The older fixed-test benchmark record uses a different learned-model context. |
| UKF | No implementation found under `src/` or `experiments/` | Must be implemented and independently validated before it can be included. |
| GPS RMSE utility | `src/radar_tracking/metrics/gps_rmse.py`, called by `src/radar_tracking/cli/eval_baseline.py` | Scores filtered GPS against ground truth; it is a metric utility, not a state estimator. |

The existing `fixed_test_comparison_01/n1024/benchmark_metrics.json` contains an EKF score on the shared fixed test set (six-state MSE `0.1316649`, `-8.8053 dB`, position RMSE `809.76 m`). Its test hash matches the five-seed test. Its Mamba/KalmanNet comparison is from another benchmark context, however, so that JSON is not a complete EKF-vs-the-five-seed-checkpoints comparison. The five-seed `run_metrics.json` records themselves contain Mamba and KalmanNet results, not EKF results.

## 2. Synthetic radar EKF mathematical and implementation audit

### State, transition, and process noise

The simulator state order is `[x, vx, ax, y, vy, ay]`. The positions, velocities, and accelerations are all divided by `SCALE=1000` in the benchmark path, so their stored units are km, km/s, and km/s². The transition implemented in `ConstantAcceleration2DModel.F` is

\[
F_{1D}=\begin{bmatrix}1&dt&dt^2/2\\0&1&dt\\0&0&1\end{bmatrix}
\]

for each axis, placed in the state order above. `Q_CA_1D` uses the standard continuous white-jerk discretization

\[
q\begin{bmatrix}dt^5/20&dt^4/8&dt^3/6\\dt^4/8&dt^3/3&dt^2/2\\dt^3/6&dt^2/2&dt\end{bmatrix}, \quad q=\text{accn\_rms\_q}^2/dt,
\]

and `Q_CA_2D` block-diagonalizes the x and y blocks. This matches `[x,vx,ax,y,vy,ay]`. The data generator draws additive zero-mean Gaussian state noise with this covariance at every transition. Prediction `F P Fᵀ + Q` is mathematically consistent for that linear state model.

At the campaign's `dt=1`, unit noise scales, and `SCALE=1000`, Q has two copies of approximately

```text
[[5.0000e-08, 1.2500e-07, 1.6667e-07],
 [1.2500e-07, 3.3333e-07, 5.0000e-07],
 [1.6667e-07, 5.0000e-07, 1.0000e-06]]
```

The current benchmark obtains Q and R from the generated/shared test payload. Thus this EKF uses the simulator's actual noise covariances, not estimates learned from the Mamba/KalmanNet training pool.

### Range-bearing observation, Jacobian, and R

The simulator observes

\[
h(x)=[\sqrt{x^2+y^2},\;\operatorname{atan2}(y,x)\bmod 2\pi],
\]

where x and y are stored in km and bearing is in radians. Its Gaussian measurement noise has 30 m range standard deviation (0.03 km after scaling) and 1 degree bearing standard deviation (converted to radians). At unit measurement scale,

```text
R = diag(0.0009 km², 0.0003046174 rad²)
```

`H_jacobian` implements the correct local derivatives away from the origin: range derivatives `[x/r, y/r]` and bearing derivatives `[-y/r², x/r²]` in the x/y state columns. The `1e-12` term avoids literal division by zero but does not make bearing well-defined at the origin; current simulated positions are far from that singularity.

`h` returns bearing in `[0,2π)`, and the simulator wraps noisy bearings to the same interval. The EKF wraps only bearing innovations with `wrap_to_pi`, yielding a principal residual in `[-π,π)`. This is correct at the branch cut. A wrapped Gaussian bearing noise is only locally Gaussian, as expected for this EKF approximation, but the campaign's 1 degree noise is narrow.

### Initialization and covariance

For the current `init_type="MEAS"`, the data generator constructs `initial_state` from two range-bearing observations: position from the first observation, velocity from the Cartesian difference over `dt` (clipped at 300 m/s), and acceleration set to zero. The generator then excludes the initialization observation from the returned 100-step test sequence. Both `batch_ekf` and KalmanNet perform a state prediction before consuming the first returned observation; Mamba receives that same initializer and returned observation sequence. This alignment should be kept when reusing the data.

The `RANGE_BEARING` reset initializes

```text
P0 = diag(var_x, 1e4, 1e4, var_y, 1e4, 1e4)
var_x = cos²(theta) sigma_r² + r² sin²(theta) sigma_theta²
var_y = sin²(theta) sigma_r² + r² cos²(theta) sigma_theta²
```

The position marginal variances are first-order polar-to-Cartesian delta-method terms. The implementation omits the generally nonzero x/y covariance and the position/velocity covariance induced by reusing the first measurement in the two-measurement velocity estimate. More seriously, it assigns variance `1e4` to velocity and acceleration in scaled state units. That corresponds to standard deviation 100 km/s or 100 km/s², far larger than the configured initial ranges (velocity ±0.2 km/s and acceleration ±0.003 km/s²). This is a heuristic, not a simulator-calibrated prior, and can dominate the filter transient.

Other reset branches are materially different: `FULL` uses R as state covariance; `POS_ONLY` uses `1e4 I` then replaces observed-position diagonals with R; `GTS` sets `initial_state` from the first true state in the generator. `GTS` is privileged ground-truth initialization and is not the campaign setting. Invalid measurement types terminate via `sys.exit` rather than raising a local exception.

### Covariance update and numerical behavior

The state update and Joseph covariance formula

\[
P^+=(I-KH)P^-(I-KH)^T+KRK^T
\]

are mathematically sound for the EKF linearization and are preferable to the simpler subtractive update. The implementation explicitly inverts each innovation covariance S with `torch.linalg.inv`; a Cholesky/linear solve is normally more stable and avoids forming the inverse. It does not explicitly symmetrize P after updates, check positive semidefiniteness, or reject non-finite innovations. Float32 is used. The current range-bearing R is positive definite and positions are nonzero, so no failure was observed by this source audit; no new EKF run was performed.

The filter allocates Q, R, x, and P on CPU with leading dimension fixed to `sim_params.dataset_size`. It therefore assumes the later evaluation batch has that exact size and device. The benchmark currently sets dataset size to the test count and uses CPU tensors for EKF, so its present call satisfies this constraint. The class is not a general variable-batch/device-safe filter.

### Noise-scale issue

The data generator returns `Q * q2_scale` and `R * r2_scale`. The benchmark constructs the EKF with `q_test / q2_scale` and `r_test / r2_scale`. For the recorded campaign both scales are exactly 1, so these expressions use the true generated covariances. If either configured scale differs from 1, the caller removes that scale and gives the EKF the base covariance instead of the covariance that generated the data. Any non-unit scale result would therefore be misconfigured. The existing configuration must not be changed silently; a later implementation should either pass the generated covariances unchanged or explicitly label a base-noise setting.

## 3. UKF availability and implementation requirements

No UKF/unscented/sigma-point implementation was found. A reliable additive-noise UKF for the six-state model needs:

1. Generate `2n+1` sigma points from a Cholesky factor of `(n+lambda)P`, with a defined policy for non-positive-definite P (symmetrize, bounded jitter, and fail loudly if factorization still fails).
2. Freeze and record scaled-UT parameters. Proposed initial setting: `alpha=1`, `beta=2`, `kappa=0`; with n=6 this gives `lambda=0`, center mean weight 0, center covariance weight 2, and each off-center mean/covariance weight `1/12`.
3. Propagate every point through the simulator transition f; take the weighted state mean and covariance and add Q once for additive process noise.
4. Pass predicted sigma points through nonlinear range-bearing h. Compute the range mean normally and bearing mean with weighted sine/cosine (circular mean), not an arithmetic mean across the 0/2π cut.
5. Compute measurement deviations with bearing residuals wrapped to `[-π,π)`, innovation covariance with R added once, and state-measurement cross covariance using the same covariance weights.
6. Solve for K without explicitly inverting S; update state with the wrapped measurement residual and P with `P⁺=P⁻−KSKᵀ`. Symmetrize and validate P; use a documented PSD/jitter policy. Guard the zero-range singularity.
7. Use the same state/measurement units, process/measurement covariance policy, test sequence, initializer, P0 variant, and output metric code as the EKF.

For this problem f is linear, so a correctly implemented additive-noise UKF prediction should agree with the EKF prediction to numerical precision when initialized with the same moments. The meaningful EKF/UKF difference is nonlinear measurement moment propagation, especially for bearing.

## 4. Baseline fairness assessment

- **Known transition and observation model:** the radar EKF receives the simulator's exact F/h and Q/R. KalmanNet is built with the same deterministic f/h, but its gain is learned and its KNet prior matrices are identity placeholders rather than the generated Q/R. Mamba receives `dt`, observations, and the same initializer and learns from supervised training sequences; exact Q/R are not explicit inference inputs. UKF would also receive the exact model and, in its oracle form, Q/R.
- **Known noise:** using exact simulator Q/R is a model-known/oracle classical baseline. It is a legitimate upper-information conventional baseline, but should be labeled as such; it is not information-parity with Mamba's inference inputs. The generated noise law and parameters are represented in the training/evaluation data, but learned models do not ingest the covariance matrices at every update.
- **Initial covariance:** current range-bearing P0 is heuristic and partly inaccurate as described above. It is not ground truth, but it is a filter-only uncertainty input unavailable to Mamba. Both EKF and UKF must use an identical, recorded P0 for their algorithm comparison. Report the legacy P0 as a reproduction variant and run a separately labeled calibrated-P0 sensitivity before treating either filter as a strongest baseline.
- **Initialization:** current `MEAS` initialization is shared across all methods and is appropriate. The alternative `GTS` initialization is privileged and should be excluded from the main comparison or reported separately as an oracle-initialization ablation.
- **Recommended noise fairness:** report an `oracle-Q/R` EKF/UKF pair using the exact simulator values, explicitly labeled. As a second, less privileged track, estimate Q/R from training data only (state transition residual covariance and wrapped observation residual covariance), freeze them before test, and optionally select scalar multipliers using validation only. Do not estimate or tune any covariance from the held-out test set. Keep the exact historical EKF setting intact as its own named variant.

## 5. Evaluation compatibility assessment

The five-seed and fixed benchmark use the same shared test artifact:

- Path: `artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt`
- SHA-256: `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`
- 1,024 examples, 100 returned observations each; state `[1024,6,100]`, observations `[1024,2,100]`, initializer `[1024,6,1]`.
- Simulation: `dt=1`, range-bearing, Gaussian process and measurement noise, fixed noise, range std 30 m, bearing std 1 degree, q/r scales 1, state scale 1000, measurement-based initialization, two-observation velocity estimate clipped at 300 m/s.

The shared payload also stores the process and measurement covariance matrices. The five-seed loader currently validates and returns only true state, observations, and initializer; a future filter evaluator should read Q/R and metadata from the same file or the recorded simulation config, verify the SHA-256, and never regenerate the held-out samples.

All methods must score exactly the same returned observations and states. Use the corrected five-seed metric convention:

- Position RMSE: `sqrt(mean(dx² + dy²)) * 1000` metres.
- Velocity RMSE: `sqrt(mean(dvx² + dvy²)) * 1000` m/s.
- Acceleration RMSE: `sqrt(mean(dax² + day²)) * 1000` m/s².
- Six-state MSE: mean squared error across examples, six stored state components, and time; report in scaled state units and `10 log10(MSE)` dB.

The MSE is a repository convention over mixed physical dimensions represented after common `1/1000` scaling; do not call it an SI-unit homogeneous MSE. Keep the same formula for all four methods. Existing benchmark JSON stores only aggregate MSE/dB and position RMSE for its EKF, whereas the five-seed results include velocity and acceleration RMSE too. Those missing EKF metrics should be recomputed from predictions with the same function, not inferred from old summaries.

The previous fixed-test benchmark runs EKF alongside a Mamba/KalmanNet configuration outside the five-seed campaign and may train/load those models through its main command path. Do not rerun it to get an EKF score. The five-seed independent evaluator already evaluates frozen seed 11/22/33/44/55 checkpoints without training; a future four-method evaluator should add EKF/UKF predictions to that same read-only evaluation path. EKF/UKF scores are deterministic for a fixed test/config and are one result each, not five independent training-seed observations. Show the same classical score beside each learned seed or once as a fixed reference; do not inflate its sample size when computing across-seed uncertainty.

Timing also needs a controlled convention: existing EKF is CPU-oriented while the learned pipeline requires CUDA. Report device-specific runtime and hardware; do not interpret a single CPU-vs-GPU wall-clock number as an algorithm-only speed comparison. For any timing comparison, warm up the relevant kernels, synchronize CUDA around measured regions, and report batch size and sequence length.

## 6. Identified issues, ranked by severity

| Severity | Issue | Scope and effect |
|---|---|---|
| High | Range-bearing P0 uses `1e4` variance for velocity/acceleration and omits initializer-induced covariance/cross-covariance. | Current filter transient and score depend on a poorly calibrated prior; not faithful to the actual measurement initializer. Keep legacy score labeled, share P0 with UKF, and add a calibrated-prior sensitivity. |
| High | Benchmark divides generated Q/R by configured q/r scales. | Incorrect whenever a noise scale is non-unit. Recorded five-seed settings use unit scales, so this does not explain or change those historical results. |
| High | Exact simulator Q/R are given to EKF without a corresponding explicit inference input to learned models. | Valid oracle/model-known baseline, but an information advantage; label and pair with training-only estimated-noise variants for a deployment-fair view. |
| Medium | Explicit inverse of S; no explicit PSD/symmetry/non-finite checks. | Use solve/Cholesky and covariance validation in any new implementation; current Joseph update is a positive feature. |
| Medium | EKF arrays assume fixed batch size and CPU. | Current benchmark satisfies it; standalone or partial-batch evaluator can fail or device-mismatch. New evaluator should have explicit batch/device handling. |
| Medium | Historical EKF metrics share the test hash but were produced in a different learned-model benchmark context and record no five-seed comparison. | Do not place the old full table beside the five-seed aggregates as if the learned runs were matched. Recompute EKF/UKF with the fixed checkpoint evaluator. |
| Medium | NCLT EKF has hand-set Q/R/P and a simplified covariance update. | Separate GPS/wheel domain, not a substitute for radar EKF/UKF; document its heuristic settings if included in NCLT results. |
| Low | UKF is absent; no reusable sigma-point validation or tests exist. | Implement only after the evaluation contract and EKF prior/noise policies are frozen. |
| Low | Fixed-pool manifest has no Git revision (`git_revision` is null), although source-file hashes are present. | Future run manifests should store the commit and dirty-tree status alongside hashes. |

## 7. Recommended implementation sequence

1. Freeze the read-only data/evaluation contract: test hash, tensor shapes, time alignment, units, metrics, and output path that refuses overwrite.
2. Add unit coverage for F/Q construction, range-bearing h/Jacobian against finite differences away from the origin, wrapped innovations around 0/2π, reset branches, covariance symmetry/PSD, batch-size errors, and physical-unit metrics.
3. Reproduce and preserve the current synthetic EKF as `legacy_ekf_oracle`, including its existing P0 and Q/R policy. Do not edit historical metrics.
4. Define the paired EKF/UKF P0 policy. First evaluate legacy P0 unchanged for compatibility. Then calculate a training/calibration-only P0 from initializer error statistics and report it as a separate variant; apply the exact same P0 to EKF and UKF.
5. Fix the noise-policy contract for new runs: an `oracle_QR` setting passes the actual generated Q/R unchanged; an `estimated_QR` setting estimates both from training data only and freezes them before validation/test. Preserve all variant labels in output.
6. Implement a standalone UKF with circular bearing statistics, additive Q/R handling, stable factorization/solve, PSD checks, and the frozen sigma parameters above. Validate its linear-prediction equivalence with EKF and small-noise behavior before benchmark evaluation.
7. Extend the independent checkpoint evaluator to compute EKF and UKF on the exact same test payload, evaluate all ten existing Mamba/KalmanNet checkpoints without training, hash inputs/checkpoints before and after, and write only to a new output directory.
8. Summarize the fixed EKF/UKF scores separately from the five learned-seed distribution; report mean/sample SD and paired seed differences for learned methods, plus the one deterministic classical score. Keep hardware/runtime comparisons separated by device.

## 8. Exact proposed experiment configuration

This is a future evaluation configuration; this audit did not run it.

| Item | Frozen proposal |
|---|---|
| Test observations/states/init | `artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt`; SHA-256 `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`; metadata seed 9001; N=1024; T=100. Load `true_state`, `observations`, `initial_state`, `process_noise`, and `measurement_noise` from this exact payload. |
| Training pool provenance for learned checkpoints | Existing `fixed_train_val.pt`, SHA-256 `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733`; data seed 20261126; train/validation/unused split 819/204/1. Do not regenerate or retrain. Train tensor hashes: state `332c2764a958b36aa588ed6595d1e29cc3e5bcf038f5db540532f507e46d9f3d`, observations `8069d8b02e6c6a06f9c420eafa0ce3a445fd9007a2b85bed66adcba73a60c91f`, initializer `381b74fa88836817f72ba21499af29b4921d0dcbcd8c5195ab5fb4db4503edef`. Validation tensor hashes: state `d685f7315dbcf64f5e8116351d6f00e8c50bae0aca649a25b62e5fa60c64b9b8`, observations `d6c11852d0e1d5635dc6e19b0f78b0e8d784b511aee3188d5bb18fd10523e489`, initializer `31e6c8a0defbaa849b778b4794c209b2a6953af34a9cf7da998d7d7031aefec`. |
| Simulator | `dt=1`; sequence length 100; state order `[x,vx,ax,y,vy,ay]`; scale 1000; `RANGE_BEARING`; `MEAS` initialization; fixed Gaussian noise; process and measurement scales 1; range std 30 m; bearing std 1 degree; motion bounds x/y position 20–30 km, velocity ±0.2 km/s, acceleration ±0.003 km/s²; acceleration RMS used for Q 0.001 km/s². |
| F/Q/R | Use the simulator's exact constant-acceleration F and shared payload Q/R. At the recorded setting, Q is two 3×3 white-jerk blocks shown above and `R=diag(0.0009,0.0003046174)`. Label this variant `oracle_QR`; do not divide Q/R by their scale factors. A distinct `estimated_QR` sensitivity, if later approved, estimates Q/R only from the 819 training trajectories. |
| EKF/UKF state init | Identical shared `initial_state`; both predict before each of the 100 returned observations. Main compatibility result uses the legacy EKF P0 formula unchanged and passes the same per-example P0 to UKF. Label it `legacy_P0`. Add a separately named `calibrated_P0` sensitivity derived without test data before drawing quality conclusions. |
| UKF | n=6, scaled UT `alpha=1`, `beta=2`, `kappa=0`; circular bearing mean/residual; additive Q and R; Cholesky sigma points and innovation solve; documented bounded jitter and PSD checks. |
| Mamba/KalmanNet | Load the existing five seeds `11,22,33,44,55`. Mamba: forward direction, 180-epoch cap, batch/eval batch 16, embed 128, hidden 256, 3 layers, `d_state=16`, `d_conv=4`, expand 2, dropout 0.08, lr `1e-3`, weight decay `1e-4`, patience 25, min delta `1e-6`. KalmanNet: batch 64, epoch cap 1000, lr `1e-3`, weight decay `1e-4`, patience 20, min delta 0.01 dB, `tbptt_k=2`, `tbptt_w=4`, wrap and arc-length handling true. Preserve recorded checkpoints. Mamba training seeds are the five listed; KalmanNet seeds are 100011, 100022, 100033, 100044, 100055. |
| Metrics | Shared corrected five-seed formulas for position, velocity, acceleration RMSE, six-state MSE and dB. Use `atol=1e-5`, `rtol=5e-4` for comparisons to existing recorded metrics. Keep EKF/UKF deterministic point scores separate from learned five-seed summaries. |
| Runtime environment | Python 3.12.3, PyTorch 2.4.0+cu121, CUDA 12.1, NVIDIA GeForce RTX 3060 (the environment used for the preceding independent checkpoint evaluation). Keep current EKF/UKF CPU evaluation separate from Mamba/KalmanNet CUDA evaluation for timing; report exact device and do not treat those times as same-hardware algorithm throughput. |
| Outputs/provenance | Proposed new directory `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/baseline_comparison_v1/`; refuse an existing path. Record test/train tensor and file SHA-256, all checkpoint hashes, config JSON, P0 per variant, Q/R values/source, sigma parameters, random seeds, commit and dirty status, source-file hashes, Python/PyTorch/CUDA/GPU, per-method batch/device, evaluation timestamp, metrics, and runtime. Never rewrite existing benchmark records. |

For future oracle comparisons, exact simulator Q/R is intentional privileged information and must appear in the variant name and interpretation. The `estimated_QR` sensitivity addresses deployment fairness; it must use no held-out-test information.

## 9. Reproducibility recommendations and unresolved choices

Every new run should record: fixed test file SHA-256 and tensor shapes/hashes; training pool file and per-split tensor hashes; simulation parameters and complete Q/R values plus whether each is oracle or estimated; P0 formula and per-sample or shared matrix; seed and split; checkpoint SHA-256; source commit, dirty-tree state, and source-file hashes; Python, PyTorch, CUDA, driver, GPU and dependency versions; evaluation device/batch size/time; all metric definitions and values; and per-method runtime protocol. The current fixed-pool manifest does hash the training tensors and sources but has no Git revision.

Unresolved design choices for the implementation/evaluation task are: (1) whether the headline should be the historical heuristic P0 or a calibrated initializer covariance (recommend report both); (2) whether the headline classical track is simulator-oracle Q/R or train-estimated Q/R (recommend report both with explicit labels); and (3) whether timing aims to compare end-to-end deployment on current CPU/GPU paths or algorithm runtime on a common device. None requires changing historical records or the five-seed checkpoints.
