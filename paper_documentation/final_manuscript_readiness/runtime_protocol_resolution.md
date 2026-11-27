# Stage 8 runtime protocol resolution

## Resolution

The original evaluator and machine-readable per-evaluation records distinguish two protocols. The apparent conflict in the Stage 8 Markdown report is a presentation ambiguity: the sentence joins the classical-filter protocol and the KalmanNet protocol without naming the subject of its second clause. It does not describe one run as both warmed and unwarmed.

- **EKF and UKF:** one complete CPU inference pass per filter policy and sequence length, no warmup. `run_ekf` and `run_ukf` in `experiments/paper_v4_long_sequence/stage8_long_sequence_evaluation.py` record `warmup_runs: 0`. The Stage 8 JSON records six EKF and six UKF rows with CPU and zero warmup (three policies × two lengths, for each filter family). The wall timer surrounds the full time-step loop. Filter construction and input conversion happen before the timer; per-step estimates, state/covariance checks and selected covariance diagnostics are inside it.
- **KalmanNet:** one warmup pass using the first `min(64, train trajectory count)` Stage 2 training trajectories, with chunk size 64. This pass occurs before the timed test pass. The timed pass evaluates the full test set in chunks of 64 on CUDA, with `torch.cuda.synchronize` immediately before and after timing. The ten KNet records (five checkpoint seeds × two lengths) report `warmup_runs: 1`. The timed call includes inference helper work such as batch transfer, hidden-state initialization, prediction, output collection, and transfer of predictions back to CPU; metric scoring is after the timer.
- **M0/M1:** Stage 8 imports Stage 6 timings and does not time these models again. Stage 6 used its own CUDA protocol: batch size 16, one warmup, and three synchronized timed repetitions. These timings are not Stage 8 measurements.

## Source comparison

The protocol is supported by three independent records:

1. Evaluator functions `run_ekf`, `run_ukf`, `infer_knet`, and the Stage 8 evaluation loop in `experiments/paper_v4_long_sequence/stage8_long_sequence_evaluation.py`.
2. `stage8_evaluation_report.json`: its `timing_protocol` object separates `classical_filters`, `kalman_net`, and `M0_M1`; its `per_evaluation_results` record each device and warmup count.
3. The raw Stage 8 result JSON and CSVs, which show CPU / 0 warmups for EKF and UKF, CUDA / 1 warmup for KalmanNet, and imported Stage 6 timings for M0/M1. The report Markdown's runtime paragraph concatenates the first two protocol descriptions in a way that can be read as a contradiction.

The Stage 8 Markdown report and JSON are historical experimental outputs. They have not been edited. This cleanup resolves their wording in documentation only.

## Manuscript recommendation

Retain the Stage 8 runtime records for provenance, but omit Stage 8 runtime values from comparative speed claims, method rankings, and cross-method runtime figures. The filters and KalmanNet used different devices and timing protocols; M0/M1 timings were imported from a separate benchmark. These values are implementation-specific wall times, not a controlled throughput comparison. If runtime is mentioned, describe the protocol and device for each method separately and explicitly state that the numbers are not directly comparable. A common-device, harmonized benchmark would be needed for a comparative runtime claim; no such experiment was run for this cleanup.

The Stage 8 to Stage 6 provenance discrepancy remains exactly as previously documented: `evaluation_provenance.json` has `stage6_report_sha256` set to `25120873…`, the Stage 6 Markdown report hash, while the Stage 6 JSON report hash is `6bac932e…`; separate Stage 8 `provenance.json` correctly points to the JSON hash. Stage 9 found no score discrepancy. Historical provenance was not changed.
