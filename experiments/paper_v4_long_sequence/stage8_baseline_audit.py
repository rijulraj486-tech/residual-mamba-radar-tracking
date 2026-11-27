#!/usr/bin/env python3
"""Read-only Phase A audit and fairness contract for Stage 8."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
LONG = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence"
STAGE6 = LONG / "stage6_matched_length_test_evaluation_v1"
STAGE7 = LONG / "stage7_statistical_analysis_v1"
STAGE2 = LONG / "stage2_consolidated_validation.json"
BASE = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
OUTPUT = LONG / "stage8_baseline_comparison_v1"
SEEDS = (11, 22, 33, 44, 55)
LENGTHS = (200, 400)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Checksum sidecar mismatch: {path}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_and_write() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 8 output: {OUTPUT}")

    stage6_path = STAGE6 / "stage6_evaluation_report.json"
    stage6_hash = verify_sidecar(stage6_path)
    stage6 = read_json(stage6_path)
    if stage6["status"] != "PASS" or stage6["evaluation_matrix"]["completed_evaluations"] != 20:
        raise ValueError("Stage 6 report is incomplete")
    for filename, expected in stage6["output_artifact_hashes"].items():
        if filename == "plots":
            for relative, digest in expected.items():
                if sha(PROJECT / relative) != digest:
                    raise ValueError(f"Stage 6 plot changed: {relative}")
        elif sha(STAGE6 / filename) != expected:
            raise ValueError(f"Stage 6 artifact changed: {filename}")
    for relative, expected in stage6["source_hashes"].items():
        if sha(PROJECT / relative) != expected:
            raise ValueError(f"Stage 6 source changed: {relative}")

    stage7_report = STAGE7 / "stage7_statistical_report.md"
    stage7_summary = STAGE7 / "stage7_statistical_summary.json"
    stage7_report_hash = verify_sidecar(stage7_report)
    stage7_summary_hash = verify_sidecar(stage7_summary)
    stage7_provenance_path = STAGE7 / "provenance.json"
    stage7_provenance_hash = verify_sidecar(stage7_provenance_path)
    stage7_provenance = read_json(stage7_provenance_path)
    if stage7_provenance["source_stage6_and_stage5b"]["stage6_report_sha256"] != stage6_hash:
        raise ValueError("Stage 7 provenance does not point to the verified Stage 6 report")

    stage2_hash = verify_sidecar(STAGE2)
    stage2 = read_json(STAGE2)
    if stage2["status"] != "passed" or stage2["inference_run"] is not False:
        raise ValueError("Stage 2 data validation record failed or performed inference")

    dataset_provenance = {}
    for length in LENGTHS:
        bundle = LONG / f"T{length}_fixed_profile_v1"
        manifest_path = bundle / "dataset_manifest.json"
        manifest_hash = verify_sidecar(manifest_path)
        manifest = read_json(manifest_path)
        split = manifest["splits"]["test"]
        test_path = bundle / split["file"]
        test_hash = sha(test_path)
        stage6_data = stage6["dataset_integrity"]["lengths"][str(length)]
        if test_hash != split["file_sha256"] or test_hash != stage6_data["test_file_sha256"]:
            raise ValueError(f"Stage 2 test tensor hash mismatch at T={length}")
        if split["trajectory_count"] != 1024 or len(split["trajectory_ids"]) != 1024:
            raise ValueError(f"T={length} test split is not 1,024 trajectories")
        trajectory_ids_sha = hashlib.sha256("\n".join(split["trajectory_ids"]).encode()).hexdigest()
        if trajectory_ids_sha != stage6_data["trajectory_ids_sha256"]:
            raise ValueError(f"Stage 6 test trajectory IDs do not match T={length} manifest")
        simulation = manifest["simulation"]
        if simulation["measurement_type"] != "RANGE_BEARING" or simulation["initialization_type"] != "MEAS":
            raise ValueError(f"Unexpected Stage 2 sensor or initializer at T={length}")
        dataset_provenance[str(length)] = {
            "bundle_manifest_path": str(manifest_path.relative_to(PROJECT)),
            "bundle_manifest_sha256": manifest_hash,
            "test_file_path": str(test_path.relative_to(PROJECT)),
            "test_file_sha256": test_hash,
            "test_trajectory_count": split["trajectory_count"],
            "trajectory_ids_sha256": trajectory_ids_sha,
            "measurement_type": simulation["measurement_type"],
            "observation_order": manifest["observation_order"],
            "initialization_type": simulation["initialization_type"],
            "process_noise_covariance_physical": simulation["process_noise_covariance_physical"],
            "measurement_noise_covariance_physical": simulation["measurement_noise_covariance_physical"],
            "motion_profile": simulation["motion_profile"],
            "state_scale": simulation["state_scale"],
        }

    ekf_path = BASE / "ekf_baseline_v1/ekf_baseline_results.json"
    ukf_path = BASE / "ukf_baseline_v1/ukf_baseline_results.json"
    ekf, ukf = read_json(ekf_path), read_json(ukf_path)
    legacy_pool_manifest_path = BASE / "fixed_pool_manifest.json"
    legacy_pool_manifest = read_json(legacy_pool_manifest_path)
    if legacy_pool_manifest["sequence_length"] != 100:
        raise ValueError("Expected historical fixed pool at T=100")

    # The historic T=100 sequence_scaled profile equals the Stage 2 fixed-reference profile.
    old_profile = legacy_pool_manifest["shared_test_metadata"]["simulation_config"]
    fixed_profile = dataset_provenance["200"]["motion_profile"]
    profile_match = (
        old_profile["motion_profile"]["pos_min"] == fixed_profile["pos_min"]
        and old_profile["motion_profile"]["pos_max"] == fixed_profile["pos_max"]
        and old_profile["motion_profile"]["vel_min"] == fixed_profile["vel_min"]
        and old_profile["motion_profile"]["vel_max"] == fixed_profile["vel_max"]
        and old_profile["motion_profile"]["accn_min"] == fixed_profile["accn_min"]
        and old_profile["motion_profile"]["accn_max"] == fixed_profile["accn_max"]
        and old_profile["motion_profile"]["accn_rms_q"] == fixed_profile["accn_rms_q"]
    )
    if not profile_match:
        raise ValueError("Historical T=100 learned baseline training profile differs from Stage 2")

    source_hashes: dict[str, str] = {}
    historical_source_mismatches: dict[str, dict[str, str]] = {}
    for result in (ekf, ukf):
        for relative, expected in result["provenance"]["source_file_sha256"].items():
            actual = sha(PROJECT / relative)
            if actual != expected:
                # The EKF result predates the opt-in fixed-reference generator
                # addition. The frozen test file and its hashes remain intact;
                # data generation is not called during inference.
                if relative != "src/radar_tracking/data/data_generation.py":
                    raise ValueError(f"Historical filter implementation source hash mismatch: {relative}")
                historical_source_mismatches[relative] = {"recorded_sha256": expected, "current_sha256": actual}
            source_hashes[relative] = actual
    for relative in (
        "src/radar_tracking/models/kalman_net.py",
        "src/radar_tracking/training/train_validate_test.py",
        "src/radar_tracking/dynamics/constant_acceleration.py",
        "src/radar_tracking/config.py",
        "experiments/five_seed_harness.py",
    ):
        source_hashes[relative] = sha(PROJECT / relative)

    # Check all five existing T=100 KNet checkpoints, metadata and exact architecture
    # compatibility at T=200 and T=400 without reading any Stage 2 test tensors.
    sys.path.insert(0, str(PROJECT / "src"))
    import torch
    from radar_tracking.config import simparams_class, validate_simparams
    from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
    from radar_tracking.models.kalman_net import KalmanNetNN

    checkpoint_records = []
    fixed_root = BASE
    for length in LENGTHS:
        args = simparams_class()
        args.seq_len, args.dataset_size, args.dt = length, 1024, 1.0
        args.q2_scale = args.r2_scale = 1.0
        args.meas_type, args.init_type, args.noise_ref = "RANGE_BEARING", "MEAS", "FIXED"
        args.range_std, args.bearing_std = 30.0, 1.0
        args.n_batch, args.epochs = 64, 1000
        args.learning_rate, args.weight_decay = 1e-3, 1e-4
        args.stopping_criteria, args.update_thr_db = 20, 0.01
        args.in_mult_KNet, args.out_mult_KNet = 5, 40
        args.knet_rb_wrap_theta, args.knet_rb_arc_length = True, True
        args.use_cuda = False
        validate_simparams(args)
        dynamics = ConstantAcceleration2DModel(args)
        for seed in SEEDS:
            run_path = fixed_root / f"seed_{seed}/run_metrics.json"
            run = read_json(run_path)
            checkpoint_path = fixed_root / f"seed_{seed}/{run['knet']['checkpoint']}"
            checkpoint_hash = sha(checkpoint_path)
            if checkpoint_hash != run["knet"]["checkpoint_sha256"]:
                raise ValueError(f"KNet checkpoint hash mismatch: seed {seed}")
            if run["knet"]["parameter_count"] != 107656:
                raise ValueError(f"Unexpected KNet parameter count: seed {seed}")
            expected_source = run["source_file_sha256"]["src/radar_tracking/models/kalman_net.py"]
            if expected_source != source_hashes["src/radar_tracking/models/kalman_net.py"]:
                raise ValueError("KNet implementation changed since these checkpoints were trained")
            model = KalmanNetNN()
            model.NNBuild(dynamics, args)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            del model
            checkpoint_records.append({
                "length_runtime_compatibility_checked": length,
                "training_sequence_length": legacy_pool_manifest["sequence_length"],
                "seed": seed,
                "checkpoint_path": str(checkpoint_path.relative_to(PROJECT)),
                "checkpoint_sha256": checkpoint_hash,
                "run_record_path": str(run_path.relative_to(PROJECT)),
                "run_record_sha256": sha(run_path),
                "training_seed": run["knet_seed"],
                "parameter_count": run["knet"]["parameter_count"],
                "strict_state_dict_load_at_target_length": True,
            })

    baseline_rows = [
        {
            "baseline": "EKF legacy_P0 + oracle_QR", "supports_T200": True, "supports_T400": True,
            "same_radar_measurements": True, "same_initial_state_mean": True,
            "extra_model_information": "Exact simulator F, h, Q, R; filter covariance P0 supplied.",
            "calibration_or_parameters": "No fitted parameters; historical heuristic P0. Oracle Q/R available in manifest.",
            "train_or_validation_calibration_without_test_leakage": "Optional; use separate explicitly named calibrated variant.",
            "publication_reliability": "Qualified reproduction only: float32 covariance, explicit innovation inverse, no general PSD/non-finite guards; legacy P0 is poorly calibrated.",
            "comparison_class": "model-informed oracle filter; not information-equivalent to Mamba",
            "decision": "evaluate with explicit oracle label; legacy sensitivity only",
            "source_or_artifact": "src/radar_tracking/models/ekf.py; experiments/evaluate_ekf_baselines.py",
        },
        {
            "baseline": "EKF calibrated_P0 + oracle_QR", "supports_T200": True, "supports_T400": True,
            "same_radar_measurements": True, "same_initial_state_mean": True,
            "extra_model_information": "Exact simulator F, h, Q, R; train-calibrated P0 and initializer/first-measurement cross covariance.",
            "calibration_or_parameters": "P0 and first-observation cross covariance fit from labeled Stage 2 train split; Q/R are oracle.",
            "train_or_validation_calibration_without_test_leakage": "Yes; existing calibration routines accept training tensors only; freeze before test.",
            "publication_reliability": "Moderate with qualification: linearized correlated first update; legacy update still uses explicit inverse and lacks broad covariance guards.",
            "comparison_class": "model-informed oracle-noise filter; not information-equivalent to Mamba",
            "decision": "evaluate; label oracle Q/R and train-only prior",
            "source_or_artifact": "src/radar_tracking/models/ekf_baselines.py; experiments/evaluate_ekf_baselines.py",
        },
        {
            "baseline": "EKF calibrated_P0 + train_estimated_QR", "supports_T200": True, "supports_T400": True,
            "same_radar_measurements": True, "same_initial_state_mean": True,
            "extra_model_information": "Exact simulator F and h; Q/R and initializer covariance estimated from labeled train data.",
            "calibration_or_parameters": "Q from state-transition residuals; R from wrapped observation residuals; P0/cross covariance from first returned-state errors.",
            "train_or_validation_calibration_without_test_leakage": "Yes; Stage 2 train split only, no validation tuning required.",
            "publication_reliability": "Moderate with same EKF numerical caveats; train-only calibration path exists and is audited.",
            "comparison_class": "model-informed, train-calibrated filter; still uses known F/h",
            "decision": "evaluate as separate train-calibrated variant",
            "source_or_artifact": "src/radar_tracking/models/ekf_baselines.py; experiments/evaluate_ekf_baselines.py",
        },
        {
            "baseline": "UKF legacy_P0 + oracle_QR", "supports_T200": True, "supports_T400": True,
            "same_radar_measurements": True, "same_initial_state_mean": True,
            "extra_model_information": "Exact simulator F, h, Q, R; filter covariance P0 and sigma-point parameters supplied.",
            "calibration_or_parameters": "Fixed scaled-UT alpha=1,beta=2,kappa=0; legacy heuristic P0; oracle Q/R.",
            "train_or_validation_calibration_without_test_leakage": "Optional; use separately named calibrated variant.",
            "publication_reliability": "Implementation has float64 covariance, bounded Cholesky jitter and PSD checks plus unit validation; legacy P0 causes severe failure on prior T=100 test.",
            "comparison_class": "model-informed oracle filter; not information-equivalent to Mamba",
            "decision": "evaluate with explicit oracle label; legacy sensitivity only",
            "source_or_artifact": "src/radar_tracking/models/ukf.py; experiments/evaluate_ukf_baselines.py",
        },
        {
            "baseline": "UKF calibrated_P0 + oracle_QR", "supports_T200": True, "supports_T400": True,
            "same_radar_measurements": True, "same_initial_state_mean": True,
            "extra_model_information": "Exact simulator F, h, Q, R; train-calibrated P0 and first-observation cross covariance.",
            "calibration_or_parameters": "Fixed scaled-UT alpha=1,beta=2,kappa=0; Stage 2 train-calibrated P0/cross covariance; oracle Q/R.",
            "train_or_validation_calibration_without_test_leakage": "Yes; existing train-only prior calibration.",
            "publication_reliability": "Qualified: robust covariance handling and T=100 validation, but nonlinear first-update cross covariance is a statistical-linearization approximation.",
            "comparison_class": "model-informed oracle-noise filter; not information-equivalent to Mamba",
            "decision": "evaluate; label oracle Q/R and train-only prior",
            "source_or_artifact": "src/radar_tracking/models/ukf.py; experiments/evaluate_ukf_baselines.py",
        },
        {
            "baseline": "UKF calibrated_P0 + train_estimated_QR", "supports_T200": True, "supports_T400": True,
            "same_radar_measurements": True, "same_initial_state_mean": True,
            "extra_model_information": "Exact simulator F and h; train-estimated Q/R and train-calibrated prior.",
            "calibration_or_parameters": "Fixed scaled-UT alpha=1,beta=2,kappa=0; all covariances estimated/calibrated from Stage 2 train split.",
            "train_or_validation_calibration_without_test_leakage": "Yes; no test or validation results used.",
            "publication_reliability": "Qualified: robust covariance handling; training-estimated matrices and nonlinear first-update approximation must be disclosed.",
            "comparison_class": "model-informed, train-calibrated filter; still uses known F/h",
            "decision": "evaluate as separate train-calibrated variant",
            "source_or_artifact": "src/radar_tracking/models/ukf.py; experiments/evaluate_ukf_baselines.py",
        },
        {
            "baseline": "KalmanNet T100-trained five-seed checkpoints", "supports_T200": True, "supports_T400": True,
            "same_radar_measurements": True, "same_initial_state_mean": True,
            "extra_model_information": "Uses known simulator f and h in recurrent update features; prior Q/Sigma/S are identity initializers, not exact oracle covariances.",
            "calibration_or_parameters": "Five frozen checkpoints (seeds 11,22,33,44,55); inference supports arbitrary sequence length, batch in configured chunks.",
            "train_or_validation_calibration_without_test_leakage": "No calibration needed; do not select/tune on Stage 2 test. Existing T100 training data has same fixed motion/noise profile.",
            "publication_reliability": "Checkpoint/source hashes and strict loads pass; T200/T400 behavior is unvalidated until Stage 8 inference; sanitization/clamping may conceal instability.",
            "comparison_class": "learned, model-assisted T100-to-long cross-length transfer; not matched-length training",
            "decision": "evaluate as separate cross-length learned reference; do not call matched-length",
            "source_or_artifact": "src/radar_tracking/models/kalman_net.py; experiments/five_seed_harness.py; five seed checkpoints",
        },
        {
            "baseline": "NCLT wheel/GPS EKF", "supports_T200": False, "supports_T400": False,
            "same_radar_measurements": False, "same_initial_state_mean": False,
            "extra_model_information": "Different wheel-input/GPS observation model and state semantics.",
            "calibration_or_parameters": "Separate hand-set Q/R/P; not applicable to synthetic radar test.",
            "train_or_validation_calibration_without_test_leakage": "Not compatible with this dataset; would require a different evaluation task.",
            "publication_reliability": "Not applicable to the synthetic radar comparison.",
            "comparison_class": "excluded domain-specific baseline",
            "decision": "exclude",
            "source_or_artifact": "src/radar_tracking/cli/nclt_benchmark.py",
        },
    ]

    checks = {
        "stage6_report_sidecar_and_output_hashes_verified": True,
        "stage6_source_hashes_verified": True,
        "stage7_report_summary_and_provenance_sidecars_verified": True,
        "stage7_points_to_verified_stage6_report": True,
        "stage2_validation_report_verified": True,
        "T200_T400_bundle_and_test_tensor_hashes_match_stage6": True,
        "T200_T400_trajectory_id_hashes_match_stage6": True,
        "all_10_KalmanNet_checkpoints_hash_verified": True,
        "all_10_KalmanNet_state_dicts_strictly_loaded_at_T200_and_T400": True,
        "KalmanNet_source_matches_training_records": True,
        "historical_filter_implementation_sources_match_result_provenance": True,
        "historical_EKF_data_generator_hash_difference_documented": True,
        "historical_T100_learned_training_profile_matches_Stage2_fixed_profile": True,
        "no_stage2_test_tensor_deserialized_or_used_for_inference_during_phase_A": True,
    }

    audit_md = f"""# Stage 8 Baseline Audit

**Audit status:** PASS WITH QUALIFICATIONS — a separated, explicitly labeled evaluation protocol is defensible. This audit used metadata, manifests, existing reports/checkpoints, and strict KalmanNet state-dict loading only. Stage 2 test file bytes were hash-verified but no test tensors were deserialized or passed to a model; no inference, training, calibration, or tuning was performed.

## Scope and inventory

- **Synthetic radar EKF:** `src/radar_tracking/models/ekf.py` (`extended_kf`, `batch_ekf`); calibration helpers in `src/radar_tracking/models/ekf_baselines.py`; evaluator `experiments/evaluate_ekf_baselines.py`.
- **Synthetic radar UKF:** `src/radar_tracking/models/ukf.py` (`RadarUKF`); evaluator `experiments/evaluate_ukf_baselines.py`; prior standalone validation is in `docs/reports/ukf_baseline_validation.md`.
- **KalmanNet:** `src/radar_tracking/models/kalman_net.py`; inference helper `evaluate_loss_knet` in `src/radar_tracking/training/train_validate_test.py`; original five-seed harness is `experiments/five_seed_harness.py`. Five T=100 checkpoints live under `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/seed_{{11,22,33,44,55}}/best_knet_seq100.pt`.
- **Other EKF:** the NCLT wheel/GPS EKF in `src/radar_tracking/cli/nclt_benchmark.py` has a different sensor/state task and is excluded.
- **Existing results:** historical EKF/UKF outputs in `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/{{ekf_baseline_v1,ukf_baseline_v1}}/`; training-only estimated Q/R reference in `estimated_noise_baseline_v1/`; five-seed KalmanNet records/checkpoints in `seed_*/run_metrics.json`; current long-sequence M0/M1 results in the verified Stage 6 and Stage 7 bundles. Historical EKF/UKF result scores are T=100 only and are not reused as T=200/T=400 scores.

## Stage 2 compatibility

The verified Stage 2 T=200 and T=400 bundles each have 1,024 unique frozen test trajectory IDs. Both use the same synthetic radar observation order (`range_scaled_by_1000`, `bearing_rad`), Gaussian range-bearing measurements, `MEAS` initialization, state order `[x,vx,ax,y,vy,ay]`, and `dt=1`. The first returned observation is the second measurement used in the two-observation initializer. All evaluated methods must consume the exact same observations and supplied `initial_state`; test truth is used only for scoring.

The T=100 fixed-pool training profile is identical to the Stage 2 fixed-reference profile at the parameter level. Thus KalmanNet's existing training domain matches the marginal simulator profile, while its training horizon is T=100 and its T=200/T=400 use is cross-length transfer. M0/M1 were trained at the evaluated length. This makes the comparison useful but not a matched-training-length learned-model comparison.

## Baseline findings

### EKF

The six-state constant-acceleration EKF accepts arbitrary sequence length through its time loop and can run the fixed 1,024-example batch on CPU. It consumes the same range-bearing observations and `initial_state`. It uses the exact simulator transition `F` and nonlinear range-bearing function `h`. Existing `legacy_ekf_oracle` and `calibrated_ekf_oracle` scores are single T=100 deterministic results. The legacy reset has heuristic, very broad velocity/acceleration variances and omits initializer-induced cross covariance. The implementation uses float32 and explicitly inverts innovation covariance without broad finite/PSD guards. It is suitable as a clearly labeled legacy reproduction, not as the only strongest EKF.

Training-only calibration is feasible: estimate `Q` from labeled adjacent state-transition residuals, `R` from state-conditioned range/bearing residuals with wrapped angle errors, and `P0` plus first-measurement cross covariance from the first returned-state residual against `F @ initial_state`. The existing helpers accept train tensors; these parameters can be frozen before test evaluation. The calibrated first update is a local EKF linearization and remains an approximation.

### UKF

`RadarUKF` supports arbitrary sequence lengths and the same input schema. It uses the exact simulator `F` and `h`; the oracle variant can use exact Q/R. The fixed scaled-UT setting is `alpha=1, beta=2, kappa=0`; covariance operations use float64, Cholesky solves, bounded jitter, and PSD checks. Existing tests/validation cover sigma reconstruction, linear prediction, circular bearing behavior, covariance stability, and a T=100 test. Its legacy P0 was catastrophically poor in prior T=100 results, so that variant is a compatibility diagnostic rather than a strong baseline. Train-only estimated Q/R and calibrated P0 can be computed without test leakage. The calibrated nonlinear first-update cross-covariance treatment is an approximation.

### KalmanNet

The five existing KNet checkpoints have verified checkpoint hashes and strict state-dict loads at both target model configurations (T=200 and T=400); each has 107,656 parameters. Its inference loop uses the requested input sequence length, and evaluation can be chunked at the configured batch size 64. It receives the same radar observation tensor and initial state, but also uses the known deterministic `f` and `h` inside its recurrent update features. It does not receive the true Q/R matrices as per-step inputs. The current implementation sanitizes and clamps hidden states, gains, and innovations; therefore the long-sequence evaluation must record clipping/saturation and non-finite/failure diagnostics. Existing validation is at T=100, so long-context reliability remains to be demonstrated by Stage 8.

## Information-access and publication assessment

The classical EKF/UKF implementations know the exact motion and measurement equations. Their `oracle_QR` variants additionally use covariance values exposed in the Stage 2 dataset manifest. These are privileged model-informed references and must be displayed separately from an information-matched learned-model comparison. A `train_estimated_QR` plus train-calibrated P0 variant reduces access to the exact noise matrices, but still knows `F` and `h`; it also remains model-informed. KalmanNet is learned but model-assisted and trained at T=100. Stage 6 M0/M1 infer from observations and initializer-derived tokens without explicit oracle Q/R or exact `F/h` evaluation at test time.

**Suitability decision:**

- **EKF:** suitable for the paper as a labeled model-informed baseline, with legacy and training-calibrated/noise policies kept distinct and numerical caveats reported.
- **UKF:** suitable for the paper as a labeled model-informed baseline; use calibrated P0 and report both oracle-noise and train-estimated-noise policies. Keep legacy-P0 output as a diagnostic only.
- **KalmanNet:** suitable as a learned T=100-to-long cross-length transfer baseline, not as a matched-length-trained comparator. Report its five seed checkpoint values, without pairing it inferentially to M0/M1 seeds.
- **NCLT wheel/GPS EKF:** unsuitable for this radar experiment; excluded.

## Phase B/C gate

**Gate: PASS WITH QUALIFICATIONS.** Proceed with the frozen Stage 2 test sets and identical trajectory IDs using separate result groups:

1. Matched-length M0/M1: import verified Stage 6 results, do not rerun or alter checkpoints.
2. KalmanNet: evaluate all five existing T=100 checkpoints on T=200/T=400 as a cross-length transfer arm. Do not select checkpoints or tune from test outcomes.
3. Model-informed filters: evaluate `legacy_P0 + oracle_QR`, `train_calibrated_P0 + oracle_QR`, and `train_calibrated_P0 + train_estimated_QR` for EKF and UKF. The oracle and train-calibrated variants must be explicitly separated and must never be described as information-equivalent to Mamba.

Use only Stage 2 train splits for Q/R/P0 calibration, freeze everything before opening the test tensors, and use Stage 2 test files solely for final inference and scoring. Preserve original Stage 6/Stage 7 outputs. Deterministic filters produce one score per length and policy; do not assign them five-seed variation. KNet retains its five trained seeds. Timing must record device and batch size and must not imply a CPU-vs-GPU algorithm speed comparison.

## Provenance

- Stage 6 report SHA-256: `{stage6_hash}`.
- Stage 7 report SHA-256: `{stage7_report_hash}`; Stage 7 JSON SHA-256: `{stage7_summary_hash}`; Stage 7 provenance SHA-256: `{stage7_provenance_hash}`.
- Stage 2 validation report SHA-256: `{stage2_hash}`.
- Stage 2 test manifests, files, IDs, noise covariances, checkpoint hashes, run-record hashes, and source hashes are recorded in `provenance.json`.
"""

    compatibility_path = OUTPUT / "baseline_compatibility.csv"
    audit_path = OUTPUT / "stage8_baseline_audit.md"
    fairness_path = OUTPUT / "baseline_fairness_protocol.md"
    provenance_path = OUTPUT / "provenance.json"
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with compatibility_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    audit_path.write_text(audit_md, encoding="utf-8")
    fairness_path.write_text(
        """# Stage 8 Fairness Protocol

## Frozen test inputs

Use only the existing Stage 2 test splits for T=200 and T=400. Verify each bundle/test manifest and test file SHA-256 immediately before evaluation. Use all 1,024 trajectories and manifest trajectory IDs in stored order. M0/M1 and every baseline must receive the exact same per-length `observations` and `initial_state`; no dataset regeneration, filtering, sample exclusion, or reordering is allowed. Test `state` is used only to calculate the frozen Stage 6 metric definitions.

## Explicit information-access groups

| Group | Methods | Information available | Reporting rule |
|---|---|---|---|
| Matched-length learned | Stage 6 M0/M1 | 200- or 400-step training; observed range/bearing sequence and its initializer-derived features | Import frozen Stage 6 metrics/checkpoint hashes; no retraining or test-based changes. |
| Learned cross-length transfer | Five frozen T=100 KalmanNet checkpoints | T=100 training data; same fixed motion/noise profile; recurrent `f/h` update structure; same test observations and initializer | Label T=100-trained transfer. This is not matched-length training and not an inference-information-equivalent Mamba model. Preserve all five seeds. |
| Model-informed classical | EKF and UKF variants | Exact constant-acceleration F and nonlinear radar h; filter-specific covariance state; either manifest-oracle Q/R or training-estimated Q/R | Show in a separate model-informed table/section. State explicitly that known dynamics and oracle noise provide privileged access absent as explicit Mamba inference inputs. |

Never combine these groups into a single unqualified “fair” ranking. The filters and KalmanNet use model structure unavailable explicitly to Mamba, while KNet also has a different training horizon.

## Frozen calibration rules

- `oracle_QR`: read Q/R values from each verified Stage 2 dataset manifest. Convert physical units to stored units once: divide state covariance Q by 1,000,000; divide range variance/covariance in R by 1,000,000; keep bearing in radians. Label this oracle/model-known.
- `train_estimated_QR`: use only the same length's 819-row Stage 2 train split. Estimate Q from adjacent ground-truth state-transition residuals and R from ground-truth-conditioned measurement residuals, wrapping bearing residuals. Do not use validation or test data for covariance fitting or multiplier selection.
- `train_calibrated_P0`: use only the same Stage 2 train split to estimate covariance of `state[:, :, 0] - F @ initial_state`, plus the covariance between that error and the first measurement noise. Use the documented correlated first update. Freeze before test inference.
- `legacy_P0`: retain the existing heuristic reset exactly; do not silently replace it. Label it legacy and diagnostic.
- UKF scaled-UT parameters stay fixed at alpha=1, beta=2, kappa=0. No test-based tuning of UT, covariance floors, clipping, or reset behavior.
- Do not modify any M0/M1 checkpoint or feature statistics. Do not train or fine-tune KalmanNet; use only the five verified existing checkpoints.

## Scoring, failures, and timing

Use Stage 6 definitions: joint x/y position RMSE in metres, joint vx/vy velocity RMSE in m/s, joint ax/ay acceleration RMSE in m/s², and mean squared error across all six stored state channels in scaled-state units. Check prediction shape and finiteness, covariance symmetry/PSD where applicable, UKF jitter/failures, EKF singular solves/non-finite updates, and KalmanNet hidden/gain/innovation clipping and saturation counts. Report failed trajectories and whole-run failures; do not silently drop them.

Record runtime per model, sequence length, policy, device, batch size, warmup and timing repetitions. Existing classical evaluators run on CPU; Stage 6 Mamba ran on CUDA. Report those as implementation/hardware timings, not a direct algorithm throughput comparison. EKF/UKF produce one deterministic result per policy and length; do not fabricate seed standard deviations. KNet's across-seed SD describes its five trained checkpoints only. Do not inferentially pair KNet seeds with M0/M1 seeds.

## Test-leakage boundary

Before each evaluation, hash and verify the frozen checkpoint, config/record, source, manifests, and dataset files. Calibration may read only the Stage 2 training split. Validation may be used only if a predeclared calibration choice requires it and must be documented before test access. Test trajectories may be read only after all policy parameters are frozen; their results cannot change any filter, checkpoint, covariance, metric, exclusion rule, or reporting choice.
""",
        encoding="utf-8",
    )
    provenance = {
        "stage": "Stage 8 Phase A baseline audit and Phase B fairness protocol",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "PASS_WITH_QUALIFICATIONS",
        "phase_c_gate": "PASS_WITH_QUALIFICATIONS",
        "project": str(PROJECT),
        "audit_runner": str(Path(__file__).resolve().relative_to(PROJECT)),
        "audit_runner_sha256": sha(Path(__file__).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "stage6_report_path": str(stage6_path.relative_to(PROJECT)),
        "stage6_report_sha256": stage6_hash,
        "stage7_report_sha256": stage7_report_hash,
        "stage7_summary_sha256": stage7_summary_hash,
        "stage7_provenance_sha256": stage7_provenance_hash,
        "stage2_validation_report_sha256": stage2_hash,
        "dataset_test_provenance": dataset_provenance,
        "baseline_results": {
            "ekf_json_sha256": sha(ekf_path),
            "ukf_json_sha256": sha(ukf_path),
            "historical_source_mismatches": historical_source_mismatches,
            "ekf_sources_verified": {k: v for k, v in source_hashes.items() if k in ekf["provenance"]["source_file_sha256"]},
            "ukf_sources_verified": {k: v for k, v in source_hashes.items() if k in ukf["provenance"]["source_file_sha256"]},
        },
        "historical_kalman_net_training_pool_manifest_sha256": sha(legacy_pool_manifest_path),
        "historical_kalman_net_motion_profile_matches_stage2": profile_match,
        "kalman_net_checkpoints": checkpoint_records,
        "source_hashes": source_hashes,
        "validation_checks": checks,
        "decisions": {
            "ekf": "evaluate as separated model-informed baseline; report legacy and train-calibrated variants",
            "ukf": "evaluate as separated model-informed baseline; report legacy diagnostic and calibrated variants",
            "kalman_net": "evaluate existing five T100 checkpoints as cross-length learned transfer only",
            "nclt_ekf": "exclude because sensor and state task differ",
            "test_based_tuning": False,
            "training_or_finetuning_authorized_or_performed": False,
            "datasets_or_stage6_stage7_artifacts_modified": False,
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_QUALIFICATIONS", "phase_c_gate": "PASS_WITH_QUALIFICATIONS", "output": str(OUTPUT), "kalman_net_checkpoint_target_length_strict_loads": len(checkpoint_records), "baseline_rows": len(baseline_rows)}, indent=2))


if __name__ == "__main__":
    validate_and_write()
