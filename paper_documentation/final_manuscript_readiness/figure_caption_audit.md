# Figure caption audit

The former `paper_documentation/figures/figure_captions.md` repeated several captions in multiple blocks and did not consistently identify source data. It has been rewritten with one caption for each produced figure group, explicit units where numerical quantities appear, and repository-relative source paths. Source-code schematics identify their implementation source. A requested figure is not given a fabricated caption when no image exists.

## Produced figure groups

- Figure 1: research pipeline schematic.
- Figure 2: radar range-bearing geometry schematic.
- Figure 3: Causal Residual Mamba architecture.
- Figure 5: T=200/T=400 test-set distribution quantiles (m, m/s, m/s²; logarithmic axes).
- Figure 6: representative Stage 5B seed 11 training/validation curves (MSE in scaled-state squared units).
- Figure 7: Stage 4 frozen T=100 extrapolation temporal plots: position RMSE (m), velocity RMSE (m/s), acceleration RMSE (m/s²), and six-state MSE (scaled-state squared units).
- Figure 8: Stage 6 matched-length test means and sample SDs across five seeds for position, velocity, acceleration RMSE and six-state MSE; axes state their units.
- Figure 9: five paired-seed M1-versus-M0 percentage changes. Points are seed pairs; diamonds and whiskers show mean ± one sample SD across seeds, not a confidence interval over trajectories.
- Figure 10: Stage 6 ten-step temporal-window metrics with position (m), velocity (m/s), acceleration (m/s²), and six-state MSE (scaled-state squared units).
- Figure 11: Stage 8 acceleration RMSE (m/s², log scale), with matched-length M0/M1, T=100-trained KalmanNet transfer, and model-informed EKF/UKF policies identified separately.

## Requested figures not produced

- **Figure 4, example test trajectory:** no image or saved per-trajectory model prediction overlay is present. This cleanup used the saved tensors only for descriptive T=100 statistics and did not generate a new trajectory plot.
- **Figure 12, runtime and complexity:** runtime evidence is in tables, but a cross-method runtime plot would invite an invalid CPU-versus-CUDA comparison. Stage 8 runtime values should not be used for comparative performance claims (see `runtime_protocol_resolution.md`).

The clean captions and per-figure source data references are in `../figures/figure_captions.md`; the existing `../figures/figure_source_registry.csv` records generated files and scripts. Stage 4 files are copies/wrappers of the original Stage 4 v2 plots and retain the original source path in that registry.
