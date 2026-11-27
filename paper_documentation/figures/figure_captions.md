# Figure captions
## architecture

Implemented M0/M1 architecture. M1 applies training-only normalization to 18 of 21 token features before projection; M0 bypasses that operation.
## architecture

Implemented M0/M1 architecture. M1 applies training-only normalization to 18 of 21 token features before projection; M0 bypasses that operation.
## dataset_test_quantiles

Stage 2 test pooled-observation p50/p95/p99 quantiles for range, speed, and acceleration magnitude. Log axes; not density plots.
## radar_geometry_schematic

Schematic of implemented range-bearing geometry. This is not measured radar data; target coordinates are illustrative.
## research_pipeline

Research pipeline schematic. Stages refer to completed experiment records; no new inference is represented.
## stage5b_learning_curves_seed11

Representative Stage 5B seed 11 training MSE (dashed) and validation six-state MSE (solid), shown by sequence length and variant. No aggregation across seeds.
## stage6_matched_length_results

Matched-length Stage 6 test performance. Points and error bars show five-seed mean and sample SD on fixed T=200/T=400 test sets.
## stage6_temporal_windows

Stage 6 ten-step temporal-window metrics by window start. Solid lines indicate T=200 and dashed lines T=400; shaded bands show ±1 sample SD across seeds.
## stage7_paired_seed_effects

Per-seed paired percent change (M1 relative to M0); points show the five seed pairs and diamonds ± one across-seed sample SD. This spread is not test-trajectory uncertainty.
## stage8_baseline_acceleration

Stage 8 acceleration RMSE by method class. M0/M1 are matched-length, KalmanNet is T=100-trained transfer, and EKF/UKF are model-informed variants; log scale. Runtime devices and fairness distinctions are in Stage 8 report.
## architecture

Implemented M0/M1 architecture. M1 applies training-only normalization to 18 of 21 token features before projection; M0 bypasses that operation.
## dataset_test_quantiles

Stage 2 test pooled-observation p50/p95/p99 quantiles for range, speed, and acceleration magnitude. Log axes; not density plots.
## radar_geometry_schematic

Schematic of implemented range-bearing geometry. This is not measured radar data; target coordinates are illustrative.
## research_pipeline

Research pipeline schematic. Stages refer to completed experiment records; no new inference is represented.
## stage5b_learning_curves_seed11

Representative Stage 5B seed 11 training MSE (dashed) and validation six-state MSE (solid), shown by sequence length and variant. No aggregation across seeds.
## stage6_matched_length_results

Matched-length Stage 6 test performance. Points and error bars show five-seed mean and sample SD on fixed T=200/T=400 test sets.
## stage6_temporal_windows

Stage 6 ten-step temporal-window metrics by window start. Solid lines indicate T=200 and dashed lines T=400; shaded bands show ±1 sample SD across seeds.
## stage7_paired_seed_effects

Per-seed paired percent change (M1 relative to M0); points show the five seed pairs and diamonds ± one across-seed sample SD. This spread is not test-trajectory uncertainty.
## stage8_baseline_acceleration

Stage 8 acceleration RMSE by method class. M0/M1 are matched-length, KalmanNet is T=100-trained transfer, and EKF/UKF are model-informed variants; log scale. Runtime devices and fairness distinctions are in Stage 8 report.
## radar_geometry_schematic

Schematic of implemented range-bearing geometry. This is not measured radar data; target coordinates are illustrative.
## research_pipeline

Research pipeline schematic. Stages refer to completed experiment records; no new inference is represented.

## Unavailable or omitted figures

- Example test trajectory: PyTorch absent, so Stage 2 `.pt` tensor was not loaded.
- Runtime/computational complexity: per-method runtime and device data are retained in tables, but not plotted together because CPU and CUDA timings are not directly comparable.
- Figure 7 frozen-checkpoint extrapolation and Figure 4 example trajectory: original stage CSVs/plots are available in source artifact paths; this package prioritizes generated matched-length figures. Stage 4 plots remain in the original result directory.


## Stage 4 frozen extrapolation figures

The four `stage4_*` figures are the supplied Stage 4 v2 temporal RMSE/MSE plots, copied byte-for-byte as PNG and embedded in PDF wrappers. Sources: `artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/plots/`. They show T=100-trained frozen-checkpoint extrapolation at T=200/T=400; see Stage 4 report for legend and protocol.
