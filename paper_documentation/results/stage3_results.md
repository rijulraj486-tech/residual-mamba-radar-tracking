# Stage 3 frozen preflight

This report reproduces the existing authoritative report text for convenient access. Original source: `artifacts/results/synthetic/paper_v4_long_sequence/stage3_frozen_preflight_v1/preflight_report.md`. Numerical tables are also indexed in `results_provenance_index.csv`.

# Phase 3 Stage 3: Frozen M0/M1 Variable-Length Preflight

**Overall status:** PASS

No training, fine-tuning, checkpoint updates, or test-set performance evaluation was performed. Inputs came from training splits only. Batch size was 1, with 2 warmups and 5 timed repetitions per checkpoint and length.

## Checkpoint integrity and architecture

- **M0:** `radar_tracking.models.tracker_model.BiMambaTrackerModel`, built by `radar_tracking.cli.benchmark_mamba.build_mamba_model`; paired original runs are under `five_seed_n1024_fixedpool_v1/seed_{11,22,33,44,55}`.
- **M1:** `experiments.paper_v4_mamba_improvement.model_variants.TrainNormalizedFeatureTracker` over `BiMambaTrackerModel`, built by `run_screening_pilot.construct_candidate`; original runs are under `experiment_a_feature_norm_five_seed_v2/seed_{11,22,33,44,55}`.

All ten checkpoint hashes matched their run records and frozen selection records. Per-checkpoint integrity conditions are explicitly marked PASS in the JSON report. Config hashes matched the baseline config and M1 frozen validation/config records. All state dictionaries loaded strictly; each model has 619,014 trainable parameters. M1 means/stds, channel indices, train-only source, and `1e-4` floor matched each seed’s saved training record and checkpoint buffers. Exact per-seed checkpoint/config/source hashes are in the JSON report.

| Variant | Seed | Strict load | Trainable parameters | Checkpoint SHA-256 | Config SHA-256 |
|---|---:|---|---:|---|---|
| M0 | 11 | True | 619014 | `b3d90cc74d7eecd36023f8b7617b2bc14d7a70f871dbad646d81c13f82f10645` | `2c31208a41be9593ae8f2995fb92d906e5a34833b7fece3df0955b7782c7a52b` |
| M1 | 11 | True | 619014 | `358bec253b6bd3f3df66f0720332059fb29e32b62f1a43280dc728f4c42603c8` | `db36a127d661ccefcc3b28104900bdfcb21b25858142172c732ce413d03a2b60` |
| M0 | 22 | True | 619014 | `60ca0fc6756be750918b386ad476d68124ed864d48d42df35f6680abacfa9b32` | `2c31208a41be9593ae8f2995fb92d906e5a34833b7fece3df0955b7782c7a52b` |
| M1 | 22 | True | 619014 | `63d7850cc59a2896e804b94a090dc445667227d82910b51c1f1010339cb44d46` | `db36a127d661ccefcc3b28104900bdfcb21b25858142172c732ce413d03a2b60` |
| M0 | 33 | True | 619014 | `73c1ca8ba10130cfef6e7f0af258af0e824063bf4aa2a702b83a1a394db0b29c` | `2c31208a41be9593ae8f2995fb92d906e5a34833b7fece3df0955b7782c7a52b` |
| M1 | 33 | True | 619014 | `7731dbe5d173085471532a2fd94029097a2fe5dea66da58bf8e916d3dc06161c` | `db36a127d661ccefcc3b28104900bdfcb21b25858142172c732ce413d03a2b60` |
| M0 | 44 | True | 619014 | `904a1d2da50480485c164fadd23ef4272cd9f22ec7aa4c04c321472c2ea1c1fa` | `2c31208a41be9593ae8f2995fb92d906e5a34833b7fece3df0955b7782c7a52b` |
| M1 | 44 | True | 619014 | `5cb6270ad4872a071437ea06b16d61d3dde948f56ae2689a861cd3ca77c09202` | `db36a127d661ccefcc3b28104900bdfcb21b25858142172c732ce413d03a2b60` |
| M0 | 55 | True | 619014 | `a8b4d45ce653bda1ad6b7e156214cde65fdd1cb8a86ec73e7bcc6d052b2b9ace` | `2c31208a41be9593ae8f2995fb92d906e5a34833b7fece3df0955b7782c7a52b` |
| M1 | 55 | True | 619014 | `85df0862fc58ba7b3ad3d32ee3d010265f5881e198d59038e4dcdd87401fb7f3` | `db36a127d661ccefcc3b28104900bdfcb21b25858142172c732ce413d03a2b60` |

## CUDA inference summary

| Length | Variant | Seed checkpoints passing shape/finite/CUDA | Mean ms/trajectory | Mean ms/observation | Peak allocated MB (max) | Peak reserved MB (max) |
|---:|---|---:|---:|---:|---:|---:|
| 100 | M0 | 5/5 | 2.521 | 0.02521 | 11.85 | 24.00 |
| 100 | M1 | 5/5 | 2.578 | 0.02578 | 11.85 | 24.00 |
| 200 | M0 | 5/5 | 2.429 | 0.01214 | 12.21 | 26.00 |
| 200 | M1 | 5/5 | 2.618 | 0.01309 | 12.22 | 26.00 |
| 400 | M0 | 5/5 | 2.517 | 0.00629 | 13.66 | 26.00 |
| 400 | M1 | 5/5 | 2.611 | 0.00653 | 13.66 | 26.00 |

All tested calls used CUDA, returned one finite state estimate per observation, and produced finite `[B,21,T]` features and `[B,6,T]` predictions. Per-checkpoint resource values and raw repetitions are in JSON. Memory figures include resident model/input allocations; the report also records baseline and peak deltas.

## Causality and sequence extension

Future-suffix tolerance: atol=1e-5, rtol=1e-5. Extension tolerance: atol=1e-4, rtol=1e-4. One T=400 training trajectory per checkpoint was used; the initial state and stored two-measurement context were unchanged in each comparison.

| Variant | Seed | Suffix test max abs / rel | First 100 vs 400 max abs / rel | First 200 vs 400 max abs / rel |
|---|---:|---:|---:|---:|
| M0 | 11 | 0 / 0 | 1.91e-06 / 0.000345 | 3.81e-06 / 0.000482 |
| M1 | 11 | 0 / 0 | 1.91e-06 / 0.000191 | 7.63e-06 / 0.000219 |
| M0 | 22 | 0 / 0 | 1.91e-06 / 0.00134 | 7.63e-06 / 0.00667 |
| M1 | 22 | 0 / 0 | 1.91e-06 / 0.000247 | 7.63e-06 / 0.000335 |
| M0 | 33 | 0 / 0 | 1.91e-06 / 0.00028 | 3.81e-06 / 0.000114 |
| M1 | 33 | 0 / 0 | 1.91e-06 / 0.0388 | 7.63e-06 / 0.000798 |
| M0 | 44 | 0 / 0 | 1.91e-06 / 0.00162 | 7.63e-06 / 0.000406 |
| M1 | 44 | 0 / 0 | 1.91e-06 / 0.00108 | 3.81e-06 / 0.00158 |
| M0 | 55 | 0 / 0 | 1.91e-06 / 0.00234 | 7.63e-06 / 0.000132 |
| M1 | 55 | 0 / 0 | 1.91e-06 / 2.97e-05 | 7.63e-06 / 0.000372 |

Suffix perturbation left all predictions through step 200 exactly unchanged in these runs. Prefix-versus-extended outputs passed `torch.allclose` at the declared tolerances. Some maximum relative differences exceed `rtol` because they occur where reference outputs are near zero; absolute differences remain below `atol`, so the combined `allclose` criterion passes.

## Hardware and source provenance

GPU: NVIDIA GeForce RTX 3060 (11.6 GiB); driver 580.178.04; CUDA 12.1; PyTorch 2.4.0+cu121; cuDNN 90100; mamba-ssm 2.2.0; Python 3.12.3.

- The original training-source hash for `data_generation.py` is verified against the Git HEAD blob. The current workspace file has a different hash because Stage 1 added an opt-in generation path; this file is not invoked during inference. Tracker, encoder, prediction-head, and M1 normalization implementation hashes are recorded and verified.
- Only the T=100 historical training split and T=200/T=400 training split files were opened. The 200/400 test files were not opened.
