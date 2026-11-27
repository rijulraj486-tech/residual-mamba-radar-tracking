# Research Paper v4: Experimental Audit and Manuscript Alignment

**Audit date:** 2026-11-27  
**Scope:** read-only audit of the supplied v3 PDF, repository implementation, reports, saved checkpoints, manifests, and evaluation outputs. No training was started and no historical artifact was overwritten.  
**Manuscript audited:** `/home/spcom41/Downloads/Radar_Tracking_Paper_v3(2).pdf` (supplied PDF; text extraction was used for its tables and claims).

## Executive findings

1. The supplied paper reports a Mamba model of approximately **1.42M parameters**, with 160-dimensional embedding and 320-dimensional head width. No matching model configuration, checkpoint, training log, data manifest, or source revision is present in the audited repository. The exact historical implementation/checkpoint behind that number is therefore **unverified**. The repository pins `mamba-ssm==2.2.0` and imports `mamba_ssm.Mamba`, but that only establishes the current implementation, not the provenance of the paper's original run ([requirements.txt](../../requirements.txt#L7); [bimamba_encoder.py](../../src/radar_tracking/models/bimamba_encoder.py#L1)).
2. The strongest reproducible learned result currently available is the later **619,014-parameter**, forward-only Mamba, evaluated across five training seeds on one hash-verified 1,024-sequence test pool. It is a distinct experiment from the PDF results. The manuscript should use this architecture and result set as its reproducible baseline unless the missing original run artifacts can be recovered.
3. The manuscript's Table II values are recoverable as printed, but not independently reproducible from the supplied files. Its EKF scores differ by 2.265 dB between nominal training-set sizes even though the text says the test trajectories are identical. The current archived 512 and 1,024 reference runs actually share the same test SHA-256 and produce the same EKF score, **−8.8053 dB**. Thus, the Table II EKF discrepancy is unresolved and is not explained by the currently archived reference runs.
4. On the verified test pool, a training-calibrated EKF/UKF is much stronger than the legacy-P0 filter and slightly lower in aggregate MSE than the five-seed Mamba mean. It uses simulator-oracle Q/R and a training-calibrated initial covariance, so it must be labeled as a model-informed baseline; it is not an information-matched learned-model comparison.
5. Subsequent loss and acceleration-feature studies show trade-offs, not a universally better Mamba. Fixed EMA acceleration features lower acceleration RMSE while increasing velocity error; the selected adaptive feature is intermediate. These are all one synthetic noise/motion regime and do not support broad robustness or long-sequence claims.

## 1. Evidence inventory and audit limits

### Manuscript claims that can be read, but not traced to run artifacts

The PDF reports a synthetic constant-acceleration task, `T=100`, `dt=1`, range noise 30 m, bearing noise 1 degree, Mamba parameters near 1.42M, and Table II aggregate MSE values. It prints figures for per-step MSE and a sample trajectory. The supplied PDF itself is evidence for what was reported, but no source plotting arrays, per-step metric file, trajectory identity, data generator seed, checkpoint, or original training/evaluation logs were found in the available result bundles. Consequently, the printed values and curves are **transcribed, not reproduced**.

The PDF does not identify a checkpoint filename or digest, code commit, Mamba package version, random seeds, train/validation/test counts, dataset hashes, early-stopping delta, full initialization/covariance protocol, or exact KalmanNet checkpoint. Its Table I gives some Mamba dimensions and optimizer settings, but cannot establish that those settings generated the reported table and figures.

### Reproducible later evidence

The later fixed-pool experiment has explicit train/test files, hashes, source hashes, configuration, five pairs of saved Mamba/KalmanNet checkpoints, per-seed metrics, and an independent checkpoint reevaluation. The shared test has 1,024 trajectories × 100 observations, metadata seed 9001, and SHA-256 `d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2`. The fixed training/validation pool has 819/204 trajectories (one unused) and file SHA-256 `4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733` ([fixed_pool_manifest.json](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json)). Independent checkpoint evaluation confirms all five model parameter counts and compares stored/recomputed metrics ([checkpoint_evaluations.json](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/independent_verification/checkpoint_evaluations.json); [experimental_evaluation_verification.md](experimental_evaluation_verification.md)).

No new evaluation was run for this audit. All numbers below are read from their named artifacts/reports; calculated conversions are marked as such.

## 2. Audit of the manuscript's original experiments

### 2.1 Mamba implementation, architecture, and training settings

**What the paper says:** Table I gives a 160-dimensional embedding, a 320-dimensional state head, three Mamba blocks, `d_state=16`, `d_conv=4`, expansion 2, dropout 0.08, Adam, learning rate `1e-3`, weight decay `1e-4`, batch size 64, 180 maximum epochs, patience 25, and approximately 1.42M parameters. It describes 21-dimensional tokens and a six-state residual output.

**What can be established:** The extant implementation imports `Mamba` from `mamba_ssm`, with pre-normalized Mamba residuals and a feed-forward residual sublayer; the package requirement is `mamba-ssm==2.2.0` ([bimamba_encoder.py](../../src/radar_tracking/models/bimamba_encoder.py#L1); [requirements.txt](../../requirements.txt#L7)). The actual saved five-seed model uses a 128-dimensional embedding, head width 256, three forward layers, `d_state=16`, `d_conv=4`, expansion 2, dropout 0.08, and **619,014** parameters ([five_seed_harness.py](../../experiments/five_seed_harness.py#L354); seed records under `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/seed_*/run_metrics.json`).

The historical 1.42M count is consistent with the PDF's larger reported widths only as a paper claim; it cannot be tied to an exact implementation or saved checkpoint from available evidence. Do not attribute it to the later 619,014-parameter checkpoints or cite the current 2.2.0 implementation as proof of the original run's Mamba version.

### 2.2 Dataset identity and Table II / Figures 2–3

The PDF states `T=100`, `dt=1`, range-bearing observations, 30 m range noise, and 1-degree bearing noise. It labels two configurations by training-set size 512 and 1,024 and claims common held-out trajectories. It prints:

| PDF Table II training-size label | EKF, dB | KalmanNet, dB | Mamba, dB | Evidence status |
|---:|---:|---:|---:|---|
| 512 | −5.871 | −15.954 | −16.232 | Exact values exist only in the supplied PDF; no matching logs/checkpoints/data manifest located. |
| 1,024 | −8.136 | −15.260 | −20.893 | Exact values exist only in the supplied PDF; no matching logs/checkpoints/data manifest located. |

Figure 2 is a visual per-step test MSE curve for the paper's 1,024-size condition; Figure 3 is a visual sample trajectory overlay. The underlying numeric per-step data and selected trajectory identifier are **unavailable**. Figure 2's descriptive text says EKF repeatedly rises toward 0 to +3 dB, KalmanNet stays near −15 to −17 dB, and Mamba near −20 to −23 dB for roughly the first two-thirds, with late-horizon rise. This is only a reading of the PDF figure/caption, not a recovered numeric result file.

The available later reference runs do not match Table II. They use the **same** fixed test SHA across train-size conditions, but their 619,014-parameter Mamba and current KNet yield:

| Archived reference run | Train/validation/test counts | EKF dB | KalmanNet dB | Mamba dB | Test identity |
|---|---:|---:|---:|---:|---|
| `fixed_test_comparison_01/n512` | 409/102/1,024 | −8.8053 | −14.6835 | −16.5117 | SHA `d5476695…705c2` |
| `fixed_test_comparison_01/n1024` | 819/204/1,024 | −8.8053 | −15.5499 | −17.0428 | SHA `d5476695…705c2` |

Sources: each run's `benchmark_metrics.json`. The two archived EKF estimates and scores are identical because the EKF evaluation is independent of learned-model training-set size when the test payload and filter configuration are held fixed. These archives **do not** establish the original PDF's run identity.

### 2.3 Why the PDF's EKF scores differ

For a fixed deterministic EKF, identical ordered test trajectories, observations, initial state, `F/Q/R`, covariance initialization, and numerical implementation imply the same result regardless of how many sequences were used to train Mamba/KalmanNet. The two Table II EKF values therefore establish that at least one condition was not actually held fixed, or that a value/configuration was transcribed incorrectly. Plausible causes include different test draws, different EKF initial covariance/reset settings, different test subsets or ordering, or an error in the table; available evidence does not distinguish them.

The present reference records have common test-file SHA and identical EKF results across 512/1,024 training-pool sizes, so they do not explain the PDF difference. The manuscript's claim of identical test trajectories cannot be accepted alongside both EKF values without the missing evaluation artifacts. Resolve by locating the original 512/1,024 data, EKF config/P0, predictions or logs, and table-generation source. If those cannot be recovered, remove the two-size comparison and replace it with a fully specified, reproducible comparison.

### 2.4 EKF and KalmanNet configurations

The PDF describes EKF as using exact transition/process/measurement models and KalmanNet as a three-GRU gain network with multipliers 5 and 40, totaling 107,656 parameters. It does not document its EKF P0 or provide KNet checkpoint/config provenance.

For the available five-seed study, KalmanNet uses `n_batch=64`, up to 1,000 epochs, Adam `1e-3`, weight decay `1e-4`, patience 20, update threshold 0.01 dB, `in_mult_KNet=5`, `out_mult_KNet=40`, wrapped bearing and arc-length handling. It has 107,656 parameters. KNet training seeds are 100011, 100022, 100033, 100044, and 100055, paired with Mamba seeds 11, 22, 33, 44, and 55 ([five_seed_harness.py](../../experiments/five_seed_harness.py#L376); per-seed `run_metrics.json`).

The historical EKF uses the legacy range-bearing P0: position diagonal terms from a polar-to-Cartesian variance approximation and `1e4` on velocity/acceleration diagonals, with no cross-covariances. The later calibrated EKF instead uses a full training-calibrated covariance and a correlated first-observation update. These are materially different methods and are reported separately below. Details and exact matrices are in [ekf_baseline_validation.md](ekf_baseline_validation.md) and its JSON result.

## 3. Later experiments and verified baseline comparison

### 3.1 Compatibility key

The five-seed Mamba/KNet, EKF, UKF, estimated-Q/R, loss-ablation, and acceleration-feature results below use the same 100-step fixed test file (SHA `d5476695…705c2`) and train pool (SHA `4ac64b0b…2733`) unless noted. They share the state order `[x,vx,ax,y,vy,ay]`, stored-state scale 1,000, `dt=1`, measurement initialization, and the repository metric function. Group RMSE is reported in physical units; aggregate six-state MSE remains in scaled state units and mixes coordinate dimensions by the stated implementation convention. It is not a homogeneous SI-unit MSE ([five_seed_harness.py](../../experiments/five_seed_harness.py#L481)).

The table separates deterministic classical-filter point results from learned five-seed mean ± sample SD. It does not pool either across configurations.

| Method / configuration | Parameters | Data and seeds | Initial state / covariance; Q/R | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE (scaled) | Provenance |
|---|---:|---|---|---:|---:|---:|---:|---|
| Mamba five-seed reference | 619,014 | Train 819 / val 204; test 1,024, 100 steps; seeds 11/22/33/44/55 | Same measurement-derived initial state as KNet; no covariance input | 332.08 ± 1.42 | 79.26 ± 2.33 | 66.71 ± 4.23 | 0.020172 ± 0.000212 | Five saved checkpoints; independent reevaluation passed. |
| KalmanNet five-seed reference | 107,656 | Same pool/test; seeds 100011/100022/100033/100044/100055 | Same initializer; learned gain, model-based recursion | 401.27 ± 16.85 | 94.77 ± 2.02 | 14.17 ± 0.58 | 0.028405 ± 0.002339 | Five saved checkpoints; independent reevaluation passed. |
| Legacy EKF, oracle Q/R | n/a | Same fixed test; deterministic | Measurement initializer; legacy diagonal P0 (`1e4` velocity/acceleration variances); actual generated Q/R | 809.76 | 316.39 | 184.88 | 0.1316649 | Exact reproduction of archived EKF reference. |
| Calibrated EKF, oracle Q/R | n/a | Same test; P0 estimated on train split only | Measurement initializer; full 6×6 training-calibrated predictive P0 plus first-measurement cross-covariance; oracle Q/R | 325.99 | 66.03 | 4.777 | 0.0184422 | Training-calibrated P0; deterministic point score. |
| Legacy UKF, oracle Q/R | n/a | Same fixed test; deterministic | Same legacy P0 as legacy EKF; alpha=1, beta=2, kappa=0; oracle Q/R | 255,433.45 | 134,598.31 | 28,276.54 | 14,027.0869 | Preserved as-is; extremely poor with broad legacy P0. |
| Calibrated UKF, oracle Q/R | n/a | Same fixed test; deterministic | Same calibrated P0/cross-covariance policy as calibrated EKF; alpha=1, beta=2, kappa=0; oracle Q/R | 325.95 | 66.01 | 4.776 | 0.0184373 | Numerically close to calibrated EKF. |
| Calibrated EKF, train-estimated Q/R | n/a | Q/R estimated on 819 training sequences; same fixed test | Same calibrated P0; training-only structured Q/R estimate | 326.04 | 66.06 | 4.777 | 0.0184486 | Estimate frozen before test evaluation. |
| Calibrated UKF, train-estimated Q/R | n/a | Q/R estimated on 819 training sequences; same fixed test | Same calibrated P0; training-only structured Q/R estimate; same UT parameters | 326.00 | 66.04 | 4.776 | 0.0184434 | Estimate frozen before test evaluation. |

Sources: [five-seed summary](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/summary/across_seed_summary.csv), [EKF results](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/ekf_baseline_v1/ekf_baseline_results.json), [UKF results](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/ukf_baseline_v1/four_method_metrics.csv), [estimated-noise results](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/estimated_noise_baseline_v1/eight_variant_metrics.csv), plus the three linked validation reports.

**Interpretation:** calibrated classical filters have better aggregate state MSE than the Mamba/KNet five-seed means in this setting, while the learned methods have different state-group trade-offs. Since calibrated filters use training labels to estimate P0 (and oracle Q/R in the oracle rows), the table must explain the information boundary. Estimated Q/R are close to oracle on this regime, but these filters still use the calibrated P0 and first-update cross-covariance; this is not evidence of robustness to other regimes. The legacy UKF/estimated-Q/R variants are also retained as configuration diagnostics, not competitive baselines.

### 3.2 Model configuration, optimization, and checkpoint provenance

| Experiment | Model and training | Seeds | Test pool / selection | Artifact status |
|---|---|---|---|---|
| Five-seed Mamba/KNet | Mamba: 21 features, 128 embed, 3 forward Mamba blocks, `d_state=16`, `d_conv=4`, expansion 2, 256-wide head, dropout .08; Adam `1e-3`, wd `1e-4`, batch 16, 180 cap, patience 25, `min_delta=1e-6`, clip 1.0. KNet as §2.4. | Mamba 11/22/33/44/55; KNet 100011/100022/100033/100044/100055. | Fixed train/validation and hashed shared test. Best validation checkpoint. | Run configs, source hashes, checkpoint SHA-256, train logs and metrics are present. |
| Adaptive acceleration | Same 619,014-param Mamba, loss, optimizer, pool, seeds and selection protocol. Replaces only coarse x/y acceleration features. Adaptive narrow/broad variants trained; narrow selected using validation Pareto rule. | 11/22/33/44/55 for each trained feature setting. | Same test hash; selection record written before final test evaluation. | Checkpoint/run audit: `mamba_adaptive_acceleration_v1/artifact_audit.json`; report includes selection and test metrics. |
| EKF / UKF | No trainable model. EKF legacy and calibrated; UKF scaled UT (`alpha=1,beta=2,kappa=0`, 13 sigma points) with wrapped/circular bearing handling. | No model training seed. Training split used to estimate calibrated P0 and/or Q/R. | Same fixed test, same initial-state sequence; exact P0 and covariance policies recorded. | Full matrices, numerical settings, hashes and result metrics in JSON artifacts. |
| Loss ablation | Same 619,014-param model; M0 original loss, M1 train-standardized coordinate loss, M2 position/velocity/acceleration group loss with acceleration weights 4 and 8. | Five paired seeds. | Same pool/test; validation frontier selected before test. M0 checkpoints reused. | M1 not test-evaluated because dominated; M2 w4/w8 test artifacts exist. |
| Acceleration-feature ablation | Same 619,014-param model and objective; original second difference, EMA alpha .25/.50, trailing quadratic W=5/9. | Five paired seeds. | Same pool/test; all nondominated settings test-evaluated. | Training, selection, final test metrics and checkpoint files are present. |

### 3.3 Adaptive acceleration and adjacent ablations

The adaptive study's final test results (mean ± sample SD, five paired seeds) are:

| Feature setting | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |
|---|---:|---:|---:|---:|
| Original second difference (M0/A0) | 332.08 ± 1.42 | 79.26 ± 2.33 | 66.71 ± 4.23 | 0.0202 ± 0.0002 |
| Fixed EMA α=.25 | 333.01 ± 2.05 | 94.85 ± 9.42 | 34.17 ± 2.65 | 0.0202 ± 0.0005 |
| Fixed EMA α=.50 | 331.54 ± 1.20 | 82.85 ± 6.00 | 58.39 ± 4.02 | 0.0200 ± 0.0002 |
| Adaptive narrow | 330.37 ± 1.14 | 81.74 ± 5.00 | 52.86 ± 1.70 | 0.0198 ± 0.0002 |

Adaptive narrow reduced acceleration RMSE versus M0 in 5/5 paired seeds (mean −13.86 m/s²), with mean velocity RMSE +2.48 m/s. Fixed EMA .25 gives the strongest acceleration value here but at higher velocity error. These outcomes establish a trade-off in this one setting, not a global improvement. Sources: [adaptive acceleration report](mamba_adaptive_acceleration_v1.md), its [final test records](../../artifacts/results/synthetic/mamba_adaptive_acceleration_v1/final_test_evaluation/), and [run audit](../../artifacts/results/synthetic/mamba_adaptive_acceleration_v1/artifact_audit.json).

The separate acceleration-feature report gives test results for the four EMA/quadratic options and confirms the same qualitative trade-off. For example, EMA .25 reports 34.17 m/s² acceleration and 94.85 m/s velocity RMSE; quadratic window 9 reports 35.99 m/s² and 94.94 m/s. Source: [mamba_acceleration_feature_ablation_v1.md](mamba_acceleration_feature_ablation_v1.md).

The loss study reports M2 weight 4 at 342.14 m position, 92.65 m/s velocity, 56.96 m/s² acceleration, and 0.021491 six-state MSE; M2 weight 8 at 348.61 m, 98.65 m/s, 50.57 m/s², and 0.022308. Both reduce acceleration error relative to M0 but raise position/velocity RMSE and aggregate MSE. M1 was not test-evaluated because validation results were dominated by M2 weight 8. Source: [mamba_loss_ablation_v1.md](mamba_loss_ablation_v1.md).

## 4. Architecture reconciliation for v4

### Reproducible architecture to report

Use the **619,014-parameter** model for the v4 reproducible baseline because those are the parameters present in the five saved checkpoints, independent checkpoint evaluations, and all subsequent five-seed ablations. Present the PDF's 1.42M model as an unverified historical configuration and remove any claim that its result is represented by those later checkpoints.

Exact current architecture and output definition:

- **Token width: 21.** Per time step: range; `sin(bearing)` and `cos(bearing)`; Cartesian x/y measurement; range, wrapped-bearing, Cartesian-x and Cartesian-y first differences; six-dimensional coarse state; and the six-dimensional initial state repeated across time. Feature assembly is in [tracker_model.py](../../src/radar_tracking/models/tracker_model.py#L181).
- **Coarse state:** `[x,vx,ax,y,vy,ay]`; measured Cartesian positions; velocity from first differences (initial velocity at first sample); acceleration from second differences (initial acceleration for the first two samples). See [tracker_model.py](../../src/radar_tracking/models/tracker_model.py#L149).
- **Input projection:** Linear 21→128, LayerNorm(128), SiLU.
- **Encoder:** three forward-only residual blocks, each with LayerNorm, `mamba_ssm.Mamba(d_model=128,d_state=16,d_conv=4,expand=2)`, dropout residual, then LayerNorm + 128→256→128 SiLU feed-forward residual; final LayerNorm. See [bimamba_encoder.py](../../src/radar_tracking/models/bimamba_encoder.py#L6).
- **Prediction head:** 128→256→128→6, with SiLU after the first two linear layers. The six outputs are residual corrections added directly to the six-dimensional coarse state, `x_hat_t = x_coarse_t + delta_x_t` ([prediction_head.py](../../src/radar_tracking/models/prediction_head.py#L4); [tracker_model.py](../../src/radar_tracking/models/tracker_model.py#L234)).
- **Direction:** actual saved baseline is causal/forward. The class name `BiMambaEncoder` and some configs' `bidirectional` label must not be mistaken for a bidirectional implementation: the encoder sets effective direction to forward and builds only forward layers ([bimamba_encoder.py](../../src/radar_tracking/models/bimamba_encoder.py#L45)).

### Original 1.42M architecture status

The PDF states 160 embed and 320 head dimensions, 3 blocks, SSM 16, convolution 4, expand 2, dropout .08, and ≈1.42M parameters, but the exact Mamba block implementation and checkpoint are not established. Its token and residual formulas broadly resemble the current design, but architecture similarity is not evidence that the later model produced the published scores. A v4 architecture table should show verified 619,014 architecture only, unless original checkpoint/config provenance is recovered.

## 5. Missing experiments: feasibility and design prerequisites

No experiments below were run as part of this audit.

| Study | Feasibility | Design/evaluation protocol required before training |
|---|---|---|
| Sequence lengths 200 and 400 | Feasible: Mamba has no learned position table and model code accepts arbitrary sequence length. Current model has only 100-step trained weights; length extrapolation is not established. Generator currently rescales velocity, acceleration, and process noise as length rises, which would confound a horizon-only comparison ([data_generation.py](../../src/radar_tracking/data/data_generation.py#L31)). | Decide whether to hold the 100-step motion/noise law fixed or deliberately study rescaled trajectories; preferably separate both questions. Predeclare train/eval lengths: train separate models at each T for in-distribution performance and test 100-trained weights at 200/400 for length extrapolation. Use fixed split/test manifests per T, paired trajectory distributions, matched filter initialization/noise, seeds, physical RMSE + six-state MSE, per-step/time-bin curves, and checkpoint hashes. Ensure initial-state alignment and scoring horizons match. |
| Multiple radar noise conditions | Feasible with the existing simulator's range/bearing and process-noise controls. | Predeclare a small factorial grid for range standard deviation, bearing standard deviation, and process-noise scale; preserve a common test seed/pool per cell and a distinct unseen test seed for generalization. Train-only calibration/normalization; no test-tuned Q/R or feature selection. Compare oracle-Q/R and train-estimated-Q/R filters under explicit labels. Report every cell plus across-condition summaries, not a pooled score alone. |
| Different target motion | Feasible but requires explicit dynamics and baseline policies. Existing generator is constant acceleration with integrated white jerk. | Include at minimum constant velocity and piecewise/switching acceleration or coordinated-turn trajectories. Freeze state labels, observation model and split generation; distinguish matched-model filters from mismatched-model filters. Specify whether filter F changes with motion or whether maneuvering is an intentional model mismatch. Train/validation/test must be split by trajectory, with held-out motion profiles for out-of-distribution claims. |
| Feature / architecture ablations | Partly available: loss, EMA/quadratic acceleration features, and adaptive acceleration have already been studied on the same synthetic pool. | Do not repeat those as if absent. If extending, predeclare one-factor controls: remove feature groups, direct-state vs coarse residual, block depth/width, and causal vs offline bidirectional (only if truly implemented). Keep parameter-count controls where possible; select on validation only; report parameter count and paired seed differences. Existing ablation reports are exploratory support for the trade-off, not evidence across conditions. |
| Computational efficiency | Feasible; current runtimes are not an apples-to-apples algorithm comparison. | Measure inference latency/throughput and peak memory for fixed T and batch sizes after warm-up, with CUDA synchronization where applicable. Report hardware, software, precision, batch size, sequence length, repetitions, and preprocessing included/excluded. Since current Mamba path requires CUDA and EKF/UKF evaluation is CPU-oriented, report device-specific results separately or implement all methods on a common device before direct comparison. Include parameter counts and training time as separate quantities; do not equate asymptotic linear complexity with measured speed. |

**Priority order:** P0 recover original run provenance and freeze the v4 baseline; P1 reproduce/read-only re-evaluate all retained checkpoints and define one primary table; P2 sequence-length study at 200/400 with a motion-scaling control; P3 noise and motion robustness matrix; P4 predeclared feature/architecture ablations; P5 controlled efficiency measurement. The existing 100-step M0 baseline and artifacts remain immutable references.

## 6. Discrepancy list for the current manuscript

| Priority | Current manuscript issue | Evidence / required correction |
|---|---|---|
| Critical | Headline table and figures use results without recoverable checkpoints, splits, hashes, seeds, or logs. | Table II is only in the PDF; later archived metrics do not match it. Recover original artifacts or replace with verified results and label the experiment precisely. |
| Critical | The PDF claims identical test trajectories but reports EKF −5.871 and −8.136 dB across training-size rows. | EKF should be independent of training size under fixed test/config; current fixed-test archive yields identical −8.8053 dB in both reference contexts. Mark the historical discrepancy unresolved and remove the identical-test assertion unless proven. |
| Critical | The paper compares 1.42M Mamba to 107,656-parameter KalmanNet and draws data-scaling conclusions, while available five-seed model is 619,014 parameters. | Replace parameter comparison with the verified architecture and perform matched repeated-seed comparison; do not infer scaling from two unverified table rows. |
| High | Reported 512/1,024 table values differ from the extant similarly named reference runs. | The current refs use shared hash `d547…705c2`; their results are `−8.8053/−14.6835/−16.5117 dB` and `−8.8053/−15.5499/−17.0428 dB` for EKF/KNet/Mamba. Keep separate from the PDF results. |
| High | Mamba batch size is 64 in PDF Table I, but verified later baseline batch is 16. | Cite exact training run. Do not mix paper hyperparameters with later checkpoints. |
| High | The abstract and discussion say Mamba improves aggregate MSE by 12.76 dB vs EKF and 5.63 dB vs KNet on largest training config. | Those differences are arithmetic on the unverified PDF Table II. They must be recomputed from the replacement primary result and its declared baseline policy. |
| High | The paper characterizes calibrated EKF/oracle information as a generic baseline comparison without explaining information parity. | Distinguish legacy-P0 reproduction, calibrated-P0, oracle Q/R, and train-estimated Q/R. Explain learned models do not receive Q/R at inference. |
| High | Claims about long-sequence superiority and fixed-window boundary effects are inferred from a single T=100 plot. | No numeric Figure 2 series is available and no T=200/400 experiment exists. Phrase as a hypothesis and run the length study before asserting boundary cause or expected widening. |
| Medium | EKF sensitivity to linearization error is asserted as explanation, though legacy-P0 initialization is poorly calibrated and causes very large errors. | Show P0 sensitivity and matched EKF/UKF results. Attribute legacy performance to the complete configuration unless an ablation isolates linearization. |
| Medium | Manuscript refers to all Mamba tokens as range-bearing plus coarse estimates but does not give exact feature ordering / initialization timing. | Add exact 21-feature layout, finite-difference startup rule, physical scaling, and first-observation initializer overlap from the implementation and data generator. |
| Medium | Figures 2 and 3 lack traceable test trajectory IDs and numeric plot inputs in available evidence. | Recover plotting source/data; otherwise regenerate from retained exact test and checkpoint artifacts with IDs and hashes. |
| Medium | Only aggregate six-state MSE is prominent, despite mixed position/velocity/acceleration scales. | State metric convention explicitly and report physical group RMSE and, where helpful, per-axis/horizon metrics. Do not call scaled aggregate MSE SI-unit MSE. |
| Medium | Conclusion says future multi-target extension is preliminary and mentions unvalidated real data; unrelated claims should not obscure the synthetic single-target evidence. | Narrow the contribution and conclusion to what was validated; keep NCLT/multi-target work separate unless a fully specified experiment is included. |

## 7. Section-by-section manuscript revision plan

1. **Title, abstract, index terms:** Retain the single-target radar scope. Replace all headline values with the selected verified experiment; include five seeds, 619,014 parameters, test-pool identity/protocol, physical-unit metrics, and the calibrated/oracle filter caveat. Avoid claims of 1.42M architecture or long-horizon benefit unless recovered/validated.
2. **Introduction and contributions:** State a narrow question: how the causal residual Mamba compares with KalmanNet and explicitly configured classical filters on the fixed synthetic range-bearing regime. Remove the claim that larger training size explains an observed wider gap unless matched repeated-size runs are recoverable.
3. **Related work:** Keep technical citations, but ensure no paragraph implies novelty or established benefits beyond these experiments. Separate sequence-model motivation from demonstrated results.
4. **System model:** Verify equations, state order, units, `dt`, integrated-white-jerk Q and range-bearing noise. Include exact initialization construction and the fact that the returned sequence starts with a measurement already involved in the measurement-derived initializer.
5. **Proposed architecture:** Replace claimed 160/320 model with 21→128 projection, three forward blocks, SSM settings, 128→256→128→6 head, and residual addition to coarse state. Name the actual `mamba-ssm` dependency and checkpoint/source hashes. Clarify this is causal; avoid calling current code bidirectional.
6. **Experimental setup:** Provide fixed training/validation counts and seed IDs, test count and SHA, trajectory distribution/noise, scaling, selection rule, training optimizer/batch/early stop/clip, exact KNet settings, initial state and covariance handling, and hardware/software. State which result is one deterministic filter score and which are five model-seed summaries.
7. **Baselines:** Report separate legacy and calibrated EKF/UKF configurations. Label oracle Q/R and training-estimated Q/R. Explain calibrated P0 uses training labels and the calibrated first update uses an estimated state/measurement-noise cross covariance. Present oracle and estimated-Q/R tracks as distinct configurations.
8. **Results:** Replace Table II with the verified comparison table in this audit, preserving mean ± sample SD only for learned seeds. Add paired differences and state-group RMSE. Recreate Figures 2–3 from retained test and checkpoint artifacts; save the underlying per-step metrics and trajectory ID.
9. **Ablations:** Include adaptive acceleration and feature/loss studies only if they serve the main research question; report trade-offs and selection policy, avoid promoting one setting as universally best. Keep each ablation tied to same pool/hash and exact config.
10. **Discussion/limitations:** Explain calibrated filter information advantage, mixed-unit aggregate MSE, single synthetic condition, one trajectory profile, and absent T=200/400 evidence. Treat boundary effect and generalization as open hypotheses.
11. **Conclusion/future work:** Summarize only supported evidence. List length, noise, motion, and efficiency experiments as planned work with protocol controls; do not predict that a method's advantage widens without measurements.
12. **References/reproducibility appendix:** Audit bibliography metadata, add code/artifact manifest, checkpoint and dataset SHA-256 values, software versions, metric equations, and a table linking each manuscript number/figure to its producing source file.

## 8. Artifact index and unavailable evidence

### Primary reproducible artifacts

- Main five-seed pool, checkpoints, logs, metrics: `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/`.
- Fixed shared test and historical single-seed-size references: `artifacts/results/synthetic/fixed_test_comparison_01/`.
- EKF configuration/results: [ekf_baseline_validation.md](ekf_baseline_validation.md) and `.../ekf_baseline_v1/ekf_baseline_results.json`.
- UKF configuration/results: [ukf_baseline_validation.md](ukf_baseline_validation.md) and `.../ukf_baseline_v1/ukf_baseline_results.json`.
- Training-estimated Q/R: [estimated_noise_baseline_validation.md](estimated_noise_baseline_validation.md), `.../estimated_noise_baseline_v1/estimated_QR_train_only.json`, and `estimated_noise_baseline_results.json`.
- Adaptive acceleration: [mamba_adaptive_acceleration_v1.md](mamba_adaptive_acceleration_v1.md), `artifacts/results/synthetic/mamba_adaptive_acceleration_v1/`.
- Other existing model ablations: [mamba_acceleration_feature_ablation_v1.md](mamba_acceleration_feature_ablation_v1.md), [mamba_loss_ablation_v1.md](mamba_loss_ablation_v1.md).
- Main model and data source: [tracker_model.py](../../src/radar_tracking/models/tracker_model.py), [bimamba_encoder.py](../../src/radar_tracking/models/bimamba_encoder.py), [prediction_head.py](../../src/radar_tracking/models/prediction_head.py), [data_generation.py](../../src/radar_tracking/data/data_generation.py), [five_seed_harness.py](../../experiments/five_seed_harness.py).

### Unavailable or unresolved

- Original 1.42M model checkpoint and exact config/source revision: **not found**.
- Exact package/build version and CUDA environment used for the PDF's original Mamba run: **not established**; current repository requirement is not historical proof.
- Original training/validation/test dataset files, split indices, seeds, and file/tensor hashes: **not found**.
- Original PDF Table II producing logs/checkpoints/predictions: **not found**; values transcribed only.
- Figure 2 numeric per-step values and Figure 3 trajectory sample identity/source data: **not found**; PDF graphics remain visible but numeric source files are unavailable.
- Original EKF covariance/reset settings and precise KNet checkpoint/training provenance for the PDF: **not established**.
- Reason for the PDF's two EKF values under its identical-test claim: **unresolved**.
- Noise-condition, held-out motion-profile, 200/400-step, and controlled efficiency studies: **not performed**.

## Audit conclusion

The evidence supports a v4 rewrite around the **619,014-parameter five-seed experiment**, with the fixed test hash and explicit covariance variants. The 1.42M architecture and Table II/Figure 2–3 results should remain labeled historical and unverified unless their originating artifacts can be recovered. The original Table II EKF discrepancy must be resolved or removed. Calibrated EKF/UKF rows should be interpreted with their training/oracle information boundary in view. Long-sequence, noise robustness, motion robustness, and speed claims require the protocol and experiments listed above before they enter the final manuscript.
