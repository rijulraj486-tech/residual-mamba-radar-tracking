# Source-of-truth hierarchy

1. **Numerical outcomes:** raw machine-readable stage output CSV/JSON with valid sidecars and the stage report describing metric/protocol. Use Stage 6 `per_seed_test_results.csv`/aggregate JSON for final M0/M1 tests; Stage 7 CSVs for paired statistics; Stage 8 `combined_metrics.csv` for baseline rows.
2. **Configuration and implementation:** selected immutable run `config.json`, run summary/checkpoint selection manifest, then source code hash recorded by run provenance. Current source can clarify behavior, while run provenance identifies historical code.
3. **Dataset provenance:** Stage 2 dataset and split manifests, generation config, file hashes, and Stage 2 consolidated validation report.
4. **Interpretation/audit:** Stage 9 readiness and Stage 8 fairness/stability audit qualify any numbers; these do not overwrite machine outputs.
5. **Historical work:** original T=100 `fixed_pool_manifest.json`, per-seed run records and summaries. Stage 4 v1 is explicitly incomplete; use Stage 4 v2. Never silently combine historical, screening, confirmation, validation, and test evidence.

If report prose and machine rows disagree, report discrepancy and prefer verified raw rows for individual numerical values. Do not modify historical evidence to reconcile it.
