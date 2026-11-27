# Table index

- `dataset_specifications.csv/.tex`: long-sequence split sizes and sampling notation.
- `stage5b_validation.csv`: exact Stage 5B per-seed validation results.
- `stage6_full_results.csv`: exact per-seed Stage 6 test rows, including runtime and provenance.
- `stage7_inference.csv`: exact statistical-test rows; see companion confidence interval and paired tables under results.
- `stage8_all_baselines.csv`: all Stage 8 method rows, including group, variant, runtime device, failures and metrics.
- `limitations_and_information_access.csv/.tex`: baseline information classes.
- `model_architecture.tex`: compact architecture parameter table; full CSV at `methodology/model_parameter_table.csv`.

Full-width tables use `table*` and standard IEEE-compatible tabular syntax. Wide result tables are kept as CSV for supplement; paper should use compact selected metrics and cite the supplement for per-seed details. These TeX fragments were generated directly from CSV; standalone compilation was not run because no TeX project/compiler was found.

- `../training/complete_hyperparameter_table.csv`: Original T=100 plus Stage 5A/5B hyperparameters.
- `runtime_complexity.csv`: runtime, device, and parameter-count evidence; CPU/GPU methods remain separated.
- `../results/complete_results_master.csv`: standardized long form across original T=100, Stages 4–6 and Stage 8.
