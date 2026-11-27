# Phase 3 Long-Sequence Radar Tracking: Code and Data Audit

**Scope:** read-only audit of the current synthetic radar pipeline and the feasibility of 200- and 400-observation experiments. No data was generated, no training was launched, and no historical benchmark or checkpoint was changed.

## Findings

| Area | Current implementation | Phase 3 consequence |
|---|---|---|
| State and dynamics | `ConstantAcceleration2DModel` uses `[x, vx, ax, y, vy, ay]`. Each step applies the same discrete constant-acceleration transition at `dt=1 s`; process noise is sampled from a constant-acceleration covariance driven by jerk variance. | The recurrence can produce arbitrarily long sequences. At fixed initial-state and process-noise scales, the distribution of position and velocity grows with elapsed time; that is a consequence of the current physical model, not an implementation defect. |
| Initial state | Position components are sampled independently from 20,000–30,000 m; velocity from -200–200 m/s and acceleration from -3–3 m/s² in the 100-step regime. | Preserve those ranges for the primary horizon study. Do not rescale them to keep long trajectories near the radar. |
| Radar observations | Range and bearing are derived from Cartesian position; range noise is Gaussian with 30 m standard deviation and bearing noise Gaussian with 1 degree standard deviation in the fixed-noise benchmark. Bearing is wrapped to `[0, 2π)`. | The measurement law supports longer sequences without a sequence-length restriction. Keep the same noise distribution and scales. Near-origin trajectories and changing range distributions should be documented as outcomes of the same model. |
| Time step | `dt=1` second is used in the dynamics, coarse differences, and initial velocity calculation. | Keep `dt=1`; changing it would change both the motion law and feature units. |
| Generator length/profile | The generator allocates `seq_len + 1` states, evolves the state continuously, then drops the extra initialisation step. However, `get_motion_profile()` scales velocity by `100/T` and acceleration/process-noise scale by `(100/T)^2` when `T>100`. | Implemented as an opt-in `motion_profile_mode="fixed_reference"` keyword on `generate_synthetic_data()`; it resolves the profile at 100 steps while leaving requested `seq_len` unchanged. The default remains `"sequence_scaled"` and retains the former profile rule. No historical benchmark files were changed. |
| Feature generation | Tokens contain range, bearing sine/cosine, Cartesian position, first differences, coarse velocity/acceleration, and the initial state repeated over time. First differences are current-minus-previous; bearing differences are wrapped to `[-π,π)`. | Initialization uses an extra two-measurement context pair. Its second sample is the first observation returned to the model; the earlier context sample is dropped from the returned sequence. The opt-in `return_initialization_observations=True` exposes that pair for validation. Default five-item return remains unchanged. This is causal: the initial state uses one prior/context measurement and the first returned measurement. |
| Train-only normalization (M1) | Eighteen dimensional token channels are normalized by training-split means and population standard deviations (floor `1e-4`); bearing sine/cosine and bearing difference channels remain unscaled. State outputs remain in the existing scaled state representation. | Load each M1 checkpoint with its own frozen normalization buffers/statistics and verify hashes. Never refit normalization on validation or test, including for the 200/400 evaluation. These 100-step statistics are intentionally frozen for the extrapolation evaluation. |
| Sequence model and causality | `BiMambaTrackerModel` accepts `[B,2,T]` for any `T>=1`, with no learned positional table or configured maximum length. The encoder uses Mamba's forward causal mixer. Feature calculations use only current/past observations, and the initial state is based on the first two measurements. The encoder rejects CPU execution and requires CUDA. | Code inspection supports variable-length inference and causal operation. It does not itself prove the installed Mamba kernel is causal or that loaded M0/M1 checkpoints run at T=200/400. Before metrics, run a no-training preflight: load frozen weights, compare prefixes when future suffixes are changed, check output/token lengths and finite values, and verify normalization buffers. Do not use test performance to tune or select models. |
| Checkpoints and parameter count | M0 uses the forward-only 619,014-parameter model. M1 uses the same architecture plus train-only normalization buffers (buffers do not add trainable parameters); its feature statistics are part of its checkpoint/state and run provenance. | Keep historical files read-only. Reconstruct the right model class from each run record/config, check strict state-dict loading and checkpoint SHA-256, and use separate output directories for all Phase 3 artifacts. |
| Metrics | Existing evaluator reports physical position, velocity, and acceleration RMSE, and six-state MSE in scaled-state units. It also computes 10-step temporal windows for the first three metrics. Physical RMSE is obtained by multiplying scaled state errors by 1,000. | Retain this convention and report every 10-step interval (with a shorter final interval if needed), plus aggregates. Add temporal-window six-state MSE so the required six-state metric is also visible over time. Label the aggregate MSE as scaled units, not SI-unit MSE. |
| Dataset/split provenance | The historical 100-step pool has 819 training, 204 validation, and a separate 1,024-trajectory test pool. The legacy API uses global random state. | Added `generate_independent_dataset_bundle()` in `data/independent_dataset.py`. It uses a local `torch.Generator` per split, emits IDs scoped by dataset ID/length/split, validates shapes, finite values and initialization alignment, and writes split-file/tensor/source/config hashes plus resolved settings and library versions. It refuses to overwrite an existing bundle directory. |

## Feasibility decision

The underlying discrete motion and radar measurement equations can generate continuous T=200 and T=400 trajectories while retaining the 100-step physical assumptions. The opt-in fixed-reference path is now implemented; the default profile policy remains unchanged. Actual 200/400 datasets have not been generated. The original benchmark artifacts remain read-only.

The longer horizon will naturally expose broader state/range values and accumulated process noise. That is the intended fixed-law horizon experiment. It is not evidence of robustness to a different motion or noise regime. A separately labelled rescaled-profile study could be considered later, but it must not replace the primary comparison.

## Variable-length evaluation preflight

Before evaluating any long test data, record a preflight result for each M0/M1 checkpoint:

1. Verify config and checkpoint hashes and instantiate the matching M0 or M1 implementation.
2. Load weights strictly; for M1 verify the stored means, standard deviations, normalized channel indices, and standard-deviation floor against the frozen 100-step training record.
3. Run inference at T=100, 200, and 400 without gradients. Verify output and token shape `[B,6,T]` / `[B,21,T]`, finite values, and successful CUDA execution.
4. Verify causality by perturbing observations strictly after a chosen prefix and checking that all estimates through that prefix remain unchanged within numerical tolerance. Also inspect the first-step and difference-feature alignment.
5. Do not alter weights, statistics, inputs, or settings in response to long-test metrics.

The generator-level shape/finiteness/alignment checks are implemented and tested. The model/checkpoint CUDA preflight remains required and has not been run; no model was loaded or evaluated in Stage 1.

## Stage 1 implementation and tests

- Enable fixed physical scales with `generate_synthetic_data(..., motion_profile_mode="fixed_reference", generator=<split-local torch.Generator>, device=<generator device>)`. The default mode remains `"sequence_scaled"`.
- For the hashed split workflow, use `generate_independent_dataset_bundle(output_dir, sequence_length=T, split_seeds={"train": ..., "validation": ..., "test": ...}, device="cpu")`. It defaults to 819/204/1,024 samples, stores tensors in benchmark-scaled units, and writes `dataset_manifest.json` with a detached `.sha256` checksum file.
- The optional `return_initialization_observations=True` generator result is a sixth tuple item containing the two-sample initialization context. It is used by the new bundle validator; existing callers still receive five items.
- Test result: `PYTHONPATH=src .venv/bin/pytest -q tests/test_long_sequence_generation.py` — **5 passed**. Tests cover legacy default equivalence to the explicit legacy mode, fixed profile and Q/R invariance at T=100/200/400, tensor validation and initialization alignment, ID collision detection, and bundle hashes/metadata. Only tiny temporary fixtures were generated by tests; no Phase 3 dataset was created.
- The host Python without project dependencies could not collect the tests (`torch` missing); the repository `.venv` with `PYTHONPATH=src` ran them successfully.

## Audit sources

- [Synthetic trajectory generation](../../src/radar_tracking/data/data_generation.py)
- [Constant-acceleration model](../../src/radar_tracking/dynamics/constant_acceleration.py)
- [Simulation parameters](../../src/radar_tracking/config.py)
- [Tracker token construction and forward pass](../../src/radar_tracking/models/tracker_model.py)
- [Forward Mamba encoder](../../src/radar_tracking/models/bimamba_encoder.py)
- [Train-only feature normalization](../../experiments/paper_v4_mamba_improvement/model_variants.py)
- [Existing metric and temporal-window implementation](../../experiments/paper_v4_mamba_improvement/run_screening_pilot.py)
- [Historical fixed-pool manifest](../../artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json)
