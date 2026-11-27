# Phase 3 Stage 2: Independent Long-Sequence Datasets

## Result

Generated and validated the T=200 and T=400 fixed-reference dataset bundles. Each length has 819 training, 204 validation, and 1,024 test trajectories. All six split manifests were loaded together with `load_and_validate_manifest_set()`; all 4,094 trajectory IDs were unique across splits and lengths.

The generator used CPU-local `torch.Generator` instances with distinct seeds for every split and length:

| Length | Train | Validation | Test |
|---:|---:|---:|---:|
| 200 | 202611271 | 202611272 | 202611273 |
| 400 | 202611274 | 202611275 | 202611276 |

The 200 and 400 bundles were written to separate directories. Existing historical 100-step files were not accessed for writing or regenerated. No models or checkpoints were loaded; no training, inference, or test-performance evaluation was run. No trajectory was clipped, rescaled, or discarded. The existing initialization velocity clip of 300 m/s remains part of the established initialization rule.

## Validation

Each split passed checks for sample count, state shape `[N,6,T]`, observation shape `[N,2,T]`, initial-state shape `[N,6,1]`, finite values, and initialization alignment with the saved two-measurement context. Split file hashes and individual tensor hashes matched their manifests. The full-bundle and split-manifest detached checksums passed; generation-config file and canonical hashes matched; source hashes for the generator, dynamics, simulation config, and bundle writer all matched.

Both manifests record `dt=1 s`, the fixed 100-step initial profile (position 20,000–30,000 m, velocity -200–200 m/s, acceleration -3–3 m/s², jerk-driving `accn_rms_q=1`), Gaussian process and measurement noise, range standard deviation 30 m, bearing standard deviation 1°, and `q2_scale=r2_scale=1`. The resolved process covariance is the constant-acceleration covariance for `dt=1`, jerk variance 1, independently applied to x and y. The measurement covariance is `diag(30², (π/180)²)` in physical units.

The test split was validated for integrity and summarized descriptively only. No model predictions or test scores were produced.

## Distribution comparison

Summary statistics pool all observations in each split and convert state values back to physical units. Near-origin counts use a true position radius or measured radar range below 1,000 m. Extreme-range counts use measured radar range above 100,000 m.

| Length | Split | Position radius p50 / p95 / p99 (m) | Speed p50 / p95 / p99 (m/s) | Acceleration magnitude p50 / p95 / p99 (m/s²) | Radar range p50 / p95 / p99 / max (m) | Near-origin true samples | Near-range samples | Extreme-range samples |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | Train | 43,815.5 / 185,972.0 / 276,109.6 | 595.44 / 2,583.72 / 3,732.89 | 10.323 / 27.818 / 36.428 | 43,816.5 / 185,968.7 / 276,112.7 / 503,705.3 | 27 | 25 | 30,098 |
| 200 | Validation | 44,721.2 / 178,777.1 / 255,262.9 | 626.31 / 2,535.20 / 3,438.83 | 10.609 / 26.395 / 35.085 | 44,716.7 / 178,800.2 / 255,267.2 / 410,970.3 | 0 | 0 | 7,925 |
| 200 | Test | 44,294.2 / 184,368.6 / 276,638.5 | 595.52 / 2,544.93 / 3,679.07 | 10.370 / 26.955 / 35.184 | 44,294.9 / 184,371.1 / 276,682.4 / 532,729.4 | 29 | 27 | 37,982 |
| 400 | Train | 136,653.5 / 964,284.1 / 1,425,847.7 | 1,640.80 / 7,061.26 / 9,825.72 | 14.808 / 37.487 / 49.929 | 136,652.4 / 964,286.0 / 1,425,822.1 / 2,502,082.5 | 18 | 18 | 185,510 |
| 400 | Validation | 140,775.9 / 960,713.2 / 1,418,390.5 | 1,677.59 / 6,924.88 / 10,063.34 | 14.534 / 37.727 / 49.702 | 140,778.8 / 960,705.2 / 1,418,340.7 / 3,056,388.5 | 2 | 3 | 46,705 |
| 400 | Test | 137,718.1 / 971,559.3 / 1,443,160.1 | 1,659.00 / 7,146.13 / 9,889.16 | 14.702 / 37.346 / 47.635 | 137,730.2 / 971,567.2 / 1,443,188.6 / 2,231,836.5 | 6 | 6 | 231,562 |

The fixed physical law produces a substantial horizon-dependent state and range expansion: at T=400, the median radar range is about 137–141 km and the largest observed range across splits is about 3,056 km. The high extreme-range counts are retained valid trajectories, not filtering failures. This is the intended consequence of keeping the 100-step motion and noise parameters fixed; it is not a claim of robustness to another motion or noise regime.

## Artifacts

- [T=200 bundle directory](../../artifacts/results/synthetic/paper_v4_long_sequence/T200_fixed_profile_v1/)
- [T=200 bundle manifest](../../artifacts/results/synthetic/paper_v4_long_sequence/T200_fixed_profile_v1/dataset_manifest.json)
- [T=200 split-manifest index](../../artifacts/results/synthetic/paper_v4_long_sequence/T200_fixed_profile_v1/manifests/index.json)
- [T=400 bundle directory](../../artifacts/results/synthetic/paper_v4_long_sequence/T400_fixed_profile_v1/)
- [T=400 bundle manifest](../../artifacts/results/synthetic/paper_v4_long_sequence/T400_fixed_profile_v1/dataset_manifest.json)
- [T=400 split-manifest index](../../artifacts/results/synthetic/paper_v4_long_sequence/T400_fixed_profile_v1/manifests/index.json)
- [Consolidated validation JSON](../../artifacts/results/synthetic/paper_v4_long_sequence/stage2_consolidated_validation.json)
- [Distribution comparison table](../../artifacts/results/synthetic/paper_v4_long_sequence/distribution_comparison.md)
- [Full distribution statistics JSON](../../artifacts/results/synthetic/paper_v4_long_sequence/distribution_comparison.json)
- [Stage 2 generation and validation runner](../../experiments/paper_v4_long_sequence/generate_stage2.py)

The consolidated validation report SHA-256 is `93f5a5f3e2cd1f77f614c5164c89746afcd95c897c7d07b13f39fef376e81e0c`. The distribution Markdown SHA-256 is `5482cfeb859ee9efc4394a4c7dc39f0337dbb367fdc025ffe4fc2df88f759885`; the full JSON SHA-256 is `21bf2b8040620b04dccba28321880ab8c358d7ac62e7027e23d0cf20d12fa6d0`.
