# Repository and experiment inventory

Inventory generated 2026-11-27 from the working tree. Git HEAD: `e34bf682166c43bc371f33e68f2a4ced8ec64206`. Working tree was clean before documentation generation. `source_file_inventory.csv`, `configuration_inventory.csv`, `checkpoint_inventory.csv`, `figure_inventory.csv`, and `bibliography_inventory.csv` give file-level records; hashes are SHA-256 of current bytes. The experiment directory listing is recursive and includes historical and current results.

## Main project areas

- `src/radar_tracking/`: dynamics, synthetic data generation, neural and classical models, training/evaluation utilities, NCLT tools, multi-target utilities, and metrics.
- `experiments/`: runnable experiment/evaluation entry points, including original five-seed harness and paper v4 long-sequence stages.
- `configs/`: JSON experiment settings.
- `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/`: original T=100 five-seed records and filter baseline outputs.
- `artifacts/results/synthetic/paper_v4_long_sequence/`: Stage 2 fixed-reference datasets and Stages 3–9 evidence.
- `docs/reports/`: design, validation, and audit reports.
- `notebooks/`: implementation/results notebooks.

## Important inventory counts

- Python source files: 9561.
- Configuration-like files: 86.
- Checkpoints: 174.
- Existing raster/vector figures: 189.
- Bibliography/LaTeX source files: 0. No `.bib` or `.tex` file was found in the working tree; citation audit is therefore limited to citations/references found in Markdown/notebooks and reports `NOT VERIFIED FROM AVAILABLE PROJECT ARTIFACTS` where full metadata is missing.

## Authority and scope

For current long-sequence matched test metrics use Stage 6 CSV/JSON and the report; Stage 7 is the source for paired inferential summaries. Stage 8 `combined_metrics.csv` is the source for baseline rows, with separate qualification groups in its fairness protocol. Stage 9 is the source for audit findings. The older `stage4...v1` folder is explicitly incomplete; use v2. Detailed authority rules are in `integrity/source_of_truth_hierarchy.md`.
