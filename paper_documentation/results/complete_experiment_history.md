# Complete experiment history

## Original T=100

Five-seed M0/M1 shared fixed-pool experiment: seeds 11/22/33/44/55, manifest `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json`; per-seed run metrics, paired differences, across-seed summaries, independent verification, and T=100 EKF/UKF/estimated-noise baselines retained under that directory. Use source records directly; T=100 numerical summary is not copied into a new recalculated table in this package. The manifest notes some Git revision metadata unavailable at generation time.

## Paper v4 stages

- Stage 2: deterministic generation/provenance audit for fixed-reference T=200/T=400 datasets; no model training or performance evaluation.
- Stage 3: frozen checkpoint preflight, causal/future-suffix and sequence-extension checks, numerical and resource checks.
- Stage 4: T=100 frozen checkpoints evaluated at T=200/T=400; extrapolation, not matched-length training.
- Stage 5A: one-seed matched-length screen, four runs, validation-only.
- Stage 5B: five-seed matched-length M0/M1 training and validation, paired seed controls.
- Stage 6: final held-out Stage 2 tests on selected Stage 5B checkpoints.
- Stage 7: post hoc paired seed statistics on Stage 6 outputs; no trajectory bootstrap.
- Stage 8: M0/M1 imports plus KalmanNet T=100 transfer and model-informed EKF/UKF variants at T=200/T=400.
- Stage 9: metric, fairness, stability and provenance audit.

Detailed stage records are linked in this directory's companion files. Do not merge stages as a single training cohort.
