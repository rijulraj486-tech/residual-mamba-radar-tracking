# Phase 3 Long-Sequence Radar Tracking: Experiment Plan

## Questions and comparisons

The study has two distinct questions and they must be reported separately:

1. **Length extrapolation:** frozen M0 and M1 checkpoints trained on the existing 100-step training data, tested on independent 200- and 400-step test trajectories. Use each model's frozen 100-step training configuration and (for M1) frozen feature statistics.
2. **Matched-length training:** train separate M0 and M1 models at 200 steps and at 400 steps, then evaluate each on the corresponding independent test pool. Use separate configs, validation selection, checkpoints, and output directories for every length and model. Five-seed confirmation follows only after a configuration passes validation screening.

The 100-step historical benchmark is the reference. Do not modify or regenerate its files. Frozen 100-step checkpoint extrapolation on the independent 200/400-step test pools is complete and reported separately below. Matching-length training remains to be done.

## Dataset design

For each new length, T=200 and T=400, create a new fixed-law dataset set with:

- 819 training trajectories;
- 204 validation trajectories;
- 1,024 test trajectories.

Use disjoint trajectory IDs across train, validation, and test, and independent IDs between the 200- and 400-step pools. Use separate generation seeds per split and record them. Generate the same constant-acceleration model and fixed Gaussian radar/process noise as the 100-step benchmark, with `dt=1 s`, unchanged initial-state ranges, and the explicit 100-step motion profile. Do not reduce velocity, acceleration, or process-noise scales as T increases.

The 200/400 training and validation tensors may also supply the test split for frozen 100-step M0/M1 extrapolation evaluation, so matched-length and extrapolation comparisons see the same horizon test trajectories. Test tensors remain inaccessible to training, validation selection, normalization fitting, or hyperparameter decisions. Store train, validation, and test as distinct files where practical; store exact trajectory IDs and tensor hashes in a manifest.

Do not use the existing 100-step test pool for long-sequence model selection. Keep the original pool unchanged and its historic role intact.

## Generation and split invariants

1. **Done:** `generate_synthetic_data(..., motion_profile_mode="fixed_reference")` supplies the original 100-step profile at any requested sequence length. The default `"sequence_scaled"` path is retained. `generate_independent_dataset_bundle()` provides per-split generators, identity namespaces, validation, hashes, and provenance. The independent T=200/T=400 bundles were generated and validated in Stage 2.
2. Set Python and PyTorch generation seeds independently for each split. Prefer explicit per-split generators where supported; record library and code versions. Generate no overlapping trajectory identity across splits or lengths.
3. Save states, observations, and initialization values in the same shape and scaled representation used by current M0/M1 training: `[N,6,T]`, `[N,2,T]`, and `[N,6,1]`.
4. Hash each split file, the manifest, and the generator/config/source files with SHA-256. Freeze the manifest before training.
5. Check actual tensor dimensions, finite values, split-ID disjointness, resolved profile/noise/dt values, and first-step initialization alignment before any model work.

## Evaluation and reporting

Use the current metric definitions without changing them:

- Position RMSE in meters: joint x/y RMSE.
- Velocity RMSE in m/s: joint vx/vy RMSE.
- Acceleration RMSE in m/s²: joint ax/ay RMSE.
- Six-state MSE in existing scaled-state units, explicitly labelled as such.

Report all metrics by 10-step intervals (steps 1–10, 11–20, etc.; short final interval where required), as well as full-horizon aggregates. Add six-state MSE to each interval. Report seed-wise values and mean/sample standard deviation for five-seed matching-length runs; do not treat the 1,024 test trajectories as model-selection data.

For every run, record parameter count, training wall time, inference wall time, peak allocated and reserved GPU memory, device/GPU/software versions, batch size, warmup count, timing protocol, dataset manifest/hash, config hash, checkpoint hash, source hashes, seed, selected epoch, and checkpoint provenance. Report inference latency both per trajectory and per observation so T=100/200/400 costs can be compared fairly. Record peak GPU memory using a documented reset/measurement window around training and inference separately.

The matched-length validation checkpoint is selected only by validation six-state MSE under the frozen initial optimizer/loss. Validation metrics can screen configurations. Freeze the passing configuration and checkpoint identities before any final test evaluation. Never choose epochs, models, thresholds, or feature statistics from test metrics.

## Training plan

- Keep the forward causal Residual Mamba encoder; six-dimensional state output; current feature representation and causal differences; physical metrics; Adam optimizer, `1e-3` learning rate, `1e-4` weight decay, batch size 16, MSE loss, gradient clip 1.0, max 180 epochs, patience 25, and `1e-6` minimum improvement for the first screening runs.
- Train M0 and M1 as separate variants. M0 has no input normalization. M1 fits its 18-channel means and population standard deviations on that run's training split only; freeze the resulting buffers for validation and test. M1's 100-step statistics stay frozen only in the length-extrapolation arm.
- Use one seed per configuration for validation screening. If a configuration passes the predeclared validation guardrails, run five seeds (11, 22, 33, 44, 55) with per-seed train-only M1 statistics. Preserve one config and run directory per length/model/seed; do not overwrite results.
- Keep the optimizer and loss unchanged initially. If long-sequence optimization fails, stop and report the failure. Any schedule, loss, truncation, or batch-size adaptation is a new, separately specified experiment and must not be mixed into the first comparison.

## Order of execution

1. Implement the opt-in fixed-profile generation path and provenance/ID recording without touching the original 100-step artifacts.
2. **Done:** Generated and validated independent T=200 and T=400 bundles at 819/204/1,024. Used unique deterministic seeds across all six splits, froze full and split manifests/checksums, validated all six split manifests together, and wrote distribution summaries. See [Stage 2 dataset report](paper_v4_long_sequence_stage2_dataset_report.md) and [consolidated validation](../../artifacts/results/synthetic/paper_v4_long_sequence/stage2_consolidated_validation.json).
3. **Done:** Ran the CUDA preflight on all five frozen M0 and five frozen M1 checkpoints at T=100/200/400. Strict loads, shapes, finite values, feature alignment, future-suffix causality, and sequence-extension invariance passed. See the [Stage 3 Markdown report](../../artifacts/results/synthetic/paper_v4_long_sequence/stage3_frozen_preflight_v1/preflight_report.md) and [machine-readable report](../../artifacts/results/synthetic/paper_v4_long_sequence/stage3_frozen_preflight_v1/preflight_report.json). No test split, training, or performance evaluation was used.
4. **Done:** Evaluated all five frozen 100-trained M0/M1 seeds on the independent T=200/400 test splits, with full-horizon and 10-step metrics, paired descriptive differences, resource records, and temporal plots. No tuning or Stage 5 decisions used these results. See the [Stage 4 report](../../artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/evaluation_report.md), [JSON report](../../artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/evaluation_report.json), and [provenance](../../artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/provenance.json).
5. **Stage 5A done:** trained one-seed M0/M1 at T=200 and T=400 using only train/validation splits. All four runs completed; M1 passed the prospective position/velocity guardrails at both lengths. Details, per-epoch logs, hashes, and checkpoints are in the [Stage 5A report](../../artifacts/results/synthetic/paper_v4_long_sequence/stage5A_matched_length_screen_v1/stage5a_screening_report.md), [machine-readable summary](../../artifacts/results/synthetic/paper_v4_long_sequence/stage5A_matched_length_screen_v1/stage5a_screening_summary.json), and [run directory](../../artifacts/results/synthetic/paper_v4_long_sequence/stage5A_matched_length_screen_v1/).
6. **Stage 5B done:** completed all 20 five-seed matched-length runs. Both M1 configurations met the frozen +1% position / +3% velocity validation guardrails at the five-seed mean; all five paired seeds passed at both lengths. Initial trainable weights and epoch-wise loader order matched for each M0/M1 pair. The [Stage 5B report](../../artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1/stage5b_confirmation_report.md), [machine-readable summary](../../artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1/stage5b_confirmation_summary.json), and [frozen checkpoint-selection manifest](../../artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1/checkpoint_selection_manifest.json) contain the results and exact hashes. No test split was accessed.
7. Evaluate matching-length checkpoints on their untouched test splits only after confirmation selection is frozen. This has not started; report it separately from Stage 4.

## Expected limitations and decision gates

- Long-sequence datasets used the explicit fixed-reference profile, retaining the original 100-step physical parameters. Do not infer robustness to a new motion/noise regime from the length extrapolation results.
- Fixed CA acceleration and process noise can produce large accumulated position/range changes by 200/400 seconds. This is faithful to the chosen motion model, but creates a meaningful distribution shift relative to the 100-step model's training inputs. Record the realized state/range distributions; do not silently constrain trajectories.
- CUDA variable-length and causal preflight passed for the frozen five-seed M0/M1 checkpoints using the installed Mamba kernels. Mamba cannot run on CPU in this repository.
- Training/inference memory and wall time are hardware dependent. Full-sequence training at T=400 can be more expensive than inference and may fail under the available GPU budget. Measure before committing to five seeds; do not silently switch to chunked or truncated training.
- Stage 4 added six-state MSE to each temporal interval while preserving the established physical metric definitions.
- Stage 2 generation uses split-local Torch generators and records trajectory identities and hashes in frozen manifests.
- Stage 5A used the prospective +1% position / +3% velocity guardrails documented in the Mamba improvement plan. Both M1 screens passed; the evidence is one-seed validation screening and does not establish five-seed confirmation.

## Templates

- [Experiment configuration template](../../configs/paper_v4_long_sequence/experiment_config.template.json)
- [Dataset manifest template](../../configs/paper_v4_long_sequence/dataset_manifest.template.json)

Stage 1 implementation tests: `PYTHONPATH=src .venv/bin/pytest -q tests/test_long_sequence_generation.py` — **5 passed**. This exercised only small temporary fixtures and in-memory trajectories; no 200/400 dataset bundle, training run, or model evaluation was performed.
