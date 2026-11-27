# Figure captions

Figure numbers below follow the recommended paper sequence. Each produced figure group has one caption shared by its PNG/PDF/SVG exports. Paths are relative to the repository root; `generate_supported_figures.py` is the plotting source for generated plots.

## Figure 1 — Research and evaluation pipeline

Research pipeline from synthetic range-bearing data generation through feature construction, M0/M1 estimation, validation, frozen evaluation, baseline comparison, and statistical audit. Stage labels refer to completed project records; the diagram does not depict an additional experiment.

**Source:** `paper_documentation/figures/source_scripts/generate_supported_figures.py`; stage inventory in `paper_documentation/results/complete_experiment_history.md`.

## Figure 2 — Radar geometry schematic

Schematic of the implemented range-bearing observation geometry, with radar origin, target position, range (r), and bearing (	heta). Coordinates are illustrative; this is not measured radar data and axes are not presented as a physical-scale trajectory.

**Source:** `src/radar_tracking/dynamics/constant_acceleration.py` (`h(x)`); `paper_documentation/figures/source_scripts/generate_supported_figures.py`.

## Figure 3 — Causal Residual Mamba architecture

M0/M1 share 21-channel feature construction, a 21-to-128 input projection, three forward residual Mamba blocks, and a six-state correction head. M1 applies training-only normalization to 18 token channels before projection; M0 uses the raw constructed features. The output is the coarse state plus the predicted correction.

**Source:** `src/radar_tracking/models/tracker_model.py`, `src/radar_tracking/models/bimamba_encoder.py`, `src/radar_tracking/models/prediction_head.py`, and Stage 5B configs under `configs/paper_v4_long_sequence/`.

## Figure 4 — Example test trajectory

Not produced. No retained per-trajectory model prediction overlay is available; no inference was run for this cleanup.

## Figure 5 — Long-sequence dataset distributions

Stage 2 test-set pooled-observation p50, p95, and p99 quantiles of measured radar range (m), target speed (m/s), and acceleration magnitude (m/s²) for T=200 and T=400. Axes are logarithmic. Points summarize quantiles and do not show fitted densities.

**Source data:** `paper_documentation/datasets/dataset_statistics.csv`, originally `artifacts/results/synthetic/paper_v4_long_sequence/distribution_comparison.json` and `.md`. T=100 is not plotted in this existing figure; its newly recovered statistics are in `paper_documentation/final_manuscript_readiness/T100_distribution_statistics.csv`.

## Figure 6 — Representative training and validation curves

Stage 5B seed 11 train MSE (dashed) and validation six-state MSE (solid) versus epoch for M0 and M1 at T=200 and T=400. MSE is in stored scaled-state squared units; the logarithmic y-axis is shown. This is one representative seed, not a five-seed aggregate.

**Source data:** `artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1/T200_M{0,1}_seed11/epoch_log.jsonl` and corresponding `T400_M{0,1}_seed11/epoch_log.jsonl`.

## Figure 7 — Frozen T=100 checkpoint extrapolation

Stage 4 temporal-window results for frozen T=100-trained M0/M1 checkpoints evaluated at T=200 and T=400: position RMSE (m), velocity RMSE (m/s), acceleration RMSE (m/s²), and six-state MSE (stored scaled-state squared units). This is cross-length extrapolation, not matched-length training. The packaged PNGs reproduce the original Stage 4 v2 plots; PDFs embed the same raster images.

**Source data:** `artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/temporal_window_metrics.csv`; source plot files are in its `plots/` directory.

## Figure 8 — Matched-length Stage 6 test results

Full-horizon Stage 6 test position RMSE (m), velocity RMSE (m/s), acceleration RMSE (m/s²), and six-state MSE (stored scaled-state squared units) for M0 and M1 trained at the evaluated length. Points/error bars show the mean and sample SD across five seeds on each fixed test set.

**Source data:** `artifacts/results/synthetic/paper_v4_long_sequence/stage6_matched_length_test_evaluation_v1/per_seed_test_results.csv` and `aggregate_test_results.csv`.

## Figure 9 — Paired M0/M1 seed changes

Per-seed percentage changes in Stage 6 error, computed as (100(M1-M0)/M0), for position RMSE, velocity RMSE, acceleration RMSE, and six-state MSE. Points show the five matched seed pairs; diamonds and whiskers show mean ± one sample SD across seed pairs. These are not confidence intervals over test trajectories.

**Source data:** `artifacts/results/synthetic/paper_v4_long_sequence/stage6_matched_length_test_evaluation_v1/per_seed_test_results.csv`; paired summaries are in `stage7_statistical_analysis_v1/paired_seed_statistics.csv` and `confidence_intervals.csv`.

## Figure 10 — Stage 6 temporal-window performance

Stage 6 ten-step-window position RMSE (m), velocity RMSE (m/s), acceleration RMSE (m/s²), and six-state MSE (stored scaled-state squared units) across the full sequence. Solid lines denote T=200; dashed lines denote T=400. Shading is ± one sample SD across five seeds.

**Source data:** `artifacts/results/synthetic/paper_v4_long_sequence/stage6_matched_length_test_evaluation_v1/temporal_window_metrics.csv`.

## Figure 11 — Stage 8 baseline context

Stage 8 test acceleration RMSE (m/s²; logarithmic axis) for matched-length M0/M1, T=100-trained KalmanNet transfer, and model-informed EKF/UKF policies. Classical policies are shown as deterministic single outcomes; KalmanNet bars summarize its five T=100-trained checkpoint results. EKF/UKF use known dynamics, with oracle-Q/R policies receiving simulator noise covariances. This is an information-stratified accuracy comparison, not a runtime comparison.

**Source data:** `artifacts/results/synthetic/paper_v4_long_sequence/stage8_baseline_comparison_v1/combined_metrics.csv`; fairness distinctions are in `baseline_fairness_protocol.md`.

## Figure 12 — Runtime and model complexity

Not produced as a comparative figure. Runtime and parameter-count records are tabulated, but CPU filter timings, CUDA KalmanNet timings, and imported Stage 6 M0/M1 timings use different protocols and are not directly comparable. See `paper_documentation/final_manuscript_readiness/runtime_protocol_resolution.md` and `paper_documentation/tables/runtime_complexity.csv`.
