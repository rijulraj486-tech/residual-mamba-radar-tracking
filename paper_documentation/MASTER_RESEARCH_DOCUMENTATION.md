# Master research documentation

**Research:** Causal Residual Mamba for Radar Target State Estimation. **Evidence scope:** synthetic single-target range-bearing experiments, original T=100 and paper-v4 long-sequence stages. This package is documentation and evidence extraction, not a new paper or new experimentation.

## Guide

- [Repository and experiment inventory](inventory/repository_inventory.md)
- [Mathematical problem formulation](methodology/problem_formulation.md)
- [Architecture](methodology/model_architecture.md) and [M0/M1 comparison](methodology/m0_m1_comparison.md)
- [Dataset methodology](datasets/dataset_methodology.md), [statistics](datasets/dataset_statistics.csv), [splits](datasets/dataset_split_summary.csv), [manifest index](datasets/dataset_manifest_index.csv), [hashes](datasets/dataset_hashes.csv)
- [Training methodology](training/training_methodology.md), [run registry](training/seed_and_run_registry.csv), [checkpoint selection](training/checkpoint_selection_summary.csv), [environment](training/training_environment.md)
- Validation: [methodology](validation/validation_methodology.md), Stage 5B per-seed and window CSVs under results.
- Results: [experiment history](results/complete_experiment_history.md), Stage [3](results/stage3_results.md), [4](results/stage4_results.md), [5A](results/stage5A_results.md), [5B](results/stage5B_results.md), [6](results/stage6_final_results.md), [7](results/stage7_statistical_results.md), [8](results/stage8_baseline_results.md), and [9](results/stage9_readiness_findings.md).
- Master tables: `results/complete_results_master.csv`, `per_seed_results_master.csv`, `paired_comparison_master.csv`, `baseline_comparison_master.csv`, `temporal_window_results_master.csv`.
- Figures, tables, reproducibility, literature, scientific claims, six-page plan, and integrity records are linked from their subdirectory indexes/README files.

## Authoritative interpretation

Stage 6 is authoritative for matched-length held-out M0/M1 evaluation. Stage 7 is authoritative for seed-paired inference and clearly reports five-seed limitations. Stage 8 results are separated into learned transfer and model-informed classical filtering; oracle Q/R is privileged simulator information. Stage 9 records the Stage 8 to Stage 6 hash-pointer discrepancy. T=400 distributions reflect accumulated simulator motion and include very high speeds/ranges. No real-world radar, multi-target, or broad noise-regime claim is supported.

## Reproducibility boundary

Source scripts, exact configs, manifests, sidecars, checkpoint records, logs, and reports remain in the repository at paths listed in inventories. Missing details are marked explicitly. No training, dataset generation, test tuning, or original artifact modification was performed during documentation generation.
