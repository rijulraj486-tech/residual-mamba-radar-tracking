# Final manuscript readiness report

**Status:** Technically ready for manuscript drafting after author-side bibliography recovery and normal IEEE assembly. This cleanup resolved the Stage 8 timing wording, deduplicated figure captions, and recovered the T=100 distribution summaries from hash-verified saved tensors. It did not alter historical experiment artifacts or results.

## Resolved

1. **Stage 8 runtime description:** the evaluator, report JSON, and raw timing records agree on distinct protocols. EKF/UKF each used one CPU pass with zero warmups; KalmanNet used a training-split warmup followed by one synchronized CUDA test pass; M0/M1 timings were imported from Stage 6. The Markdown paragraph is poorly joined and has been clarified in the new report without editing historical files. Stage 8 runtime values should be omitted from comparative claims because devices/protocols differ.
2. **Figure captions:** duplicate blocks in `paper_documentation/figures/figure_captions.md` have been replaced by one source-linked caption per produced figure group. Units and unavailable figures are called out.
3. **T=100 distributions:** recovered saved train/validation and shared test tensors; verified their file hashes against `fixed_pool_manifest.json`; used the existing `.venv` PyTorch 2.4.0+cu121 environment for read-only descriptive calculations. No data were regenerated and no model was run. Results and an analysis script are in `T100_distribution_statistics.csv` and `derive_T100_distribution_statistics.py`.

T=100 test pooled-observation medians are 36.75 km position radius, 277.5 m/s speed, 7.71 m/s² acceleration magnitude, and 36.75 km measured range. The 99th-percentile speed is 1,351.7 m/s and maximum range is 130.7 km. These are accumulated-motion simulator distributions, not evidence of physical realism. Values for all splits and quantiles are in the CSV.

## Still unavailable

- No original manuscript or bibliography source was found. The exact missing source material is enumerated in `bibliography_recovery_status.md`; no references or metadata were invented.
- Figure 4 (a genuine test-trajectory example with prediction overlays) remains absent because per-trajectory model predictions were not retained and this cleanup did not run inference.
- Figure 12 runtime/complexity is retained as a table only; the Stage 8 timings are not suitable for comparative runtime claims.
- Author identities/affiliations and final IEEE numbering are author-side manuscript details, pending the checklist.

## Provenance preservation

The Stage 8 provenance discrepancy remains unchanged: the `stage6_report_sha256` field in `evaluation_provenance.json` identifies the Stage 6 Markdown hash (`25120873…`), while Stage 6 JSON report hash is `6bac932e…`; separate Stage 8 `provenance.json` correctly points to the JSON hash. Stage 9 found no score discrepancy. No historical provenance file was edited.

## Files in this readiness package

- `runtime_protocol_resolution.md`
- `bibliography_recovery_status.md`
- `figure_caption_audit.md`
- `final_submission_checklist.md`
- `T100_distribution_statistics.csv` (supporting derived evidence)
- `derive_T100_distribution_statistics.py` (read-only analysis source)
