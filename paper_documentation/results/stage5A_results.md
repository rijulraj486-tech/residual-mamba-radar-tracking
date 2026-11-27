# Stage 5A one-seed screen

This report reproduces the existing authoritative report text for convenient access. Original source: `artifacts/results/synthetic/paper_v4_long_sequence/stage5A_matched_length_screen_v1/stage5a_screening_report.md`. Numerical tables are also indexed in `results_provenance_index.csv`.

# Phase 3 Stage 5A: Matched-Length M0/M1 Screening

**Status:** PASS

One screening seed (101) was used. Only train and validation tensors were opened. No T=200/T=400 test tensor was opened or evaluated. No five-seed confirmation or hyperparameter adaptation occurred.

## Frozen screening rule

Checkpoints were selected by minimum validation six-state MSE. M1 eligibility uses the previously documented prospective guardrails against the same-length M0 validation reference: position RMSE may increase by at most 1%, and velocity RMSE by at most 3%. The source rule is recorded in each frozen config. Stability also requires finite training/validation values and a selected validation checkpoint. No additional threshold was introduced.

## Results

Metrics below are for each run's validation-selected checkpoint.

| T | Variant | Status | Selected epoch | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Training (s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 200 | M0 | COMPLETED | 95 | 580.444 | 207.356 | 319.686 | 0.0803518 | 62.48 |
| 200 | M1 | COMPLETED | 75 | 581.609 | 122.11 | 120.622 | 0.0612882 | 52.62 |
| 400 | M0 | COMPLETED | 149 | 2376.71 | 1204.57 | 1941.94 | 1.81182 | 137.97 |
| 400 | M1 | COMPLETED | 56 | 2204.61 | 546.932 | 832.998 | 0.975557 | 65.65 |

### Validation guardrails

| T | M1 position change | M1 velocity change | M1 acceleration change | M1 confirmation eligible |
|---:|---:|---:|---:|---|
| 200 | +0.20% | -41.11% | -62.27% | Yes |
| 400 | -7.24% | -54.60% | -57.10% | Yes |

## Resources and provenance

GPU: NVIDIA GeForce RTX 3060; driver 580.178.04; CUDA 12.1; PyTorch 2.4.0+cu121; Mamba 2.2.0; Python 3.12.3.

Training and validation peak allocated/reserved memory are recorded separately per epoch in each run summary. Exact checkpoint, config, dataset, and source hashes are in the JSON summaries and run directories.

- `screening_summary.csv`: one row per configuration.
- `plots/training_and_validation_loss.png`: train and validation six-state MSE curves.
- `plots/validation_metrics_comparison.png`: four validation metrics across runs.
- Per-run `epoch_log.jsonl`, `run_summary.json`, and successful selected checkpoint files are stored in their isolated run folders.

No test-based model selection or Stage 5B decision was performed. These single-seed screens are feasibility evidence only; they do not establish five-seed performance.
