#!/usr/bin/env python3
"""Freeze training-only Q/R estimates and evaluate eight EKF/UKF variants."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "experiments")]
import five_seed_harness as five_seed
from radar_tracking.config import simparams_class, validate_simparams
from radar_tracking.data.data_generation import Q_CA_2D, get_motion_profile
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.models.ekf import extended_kf
from radar_tracking.models.ekf_baselines import (
    correlated_first_observation_update, covariance_for_policy, tensor_sha256,
    training_split_from_pool,
)
from radar_tracking.models.noise_estimation import estimate_training_noise
from radar_tracking.models.ukf import RadarUKF

ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
OUTPUT = ROOT / "estimated_noise_baseline_v1"
POOL_PATH = ROOT / "fixed_train_val.pt"
MANIFEST_PATH = ROOT / "fixed_pool_manifest.json"
TEST_PATH = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"
TEST_SHA256 = "d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2"
EKF_RESULT_PATH = ROOT / "ekf_baseline_v1/ekf_baseline_results.json"
UKF_RESULT_PATH = ROOT / "ukf_baseline_v1/ukf_baseline_results.json"
METRIC_KEYS = ("position_rmse_m", "velocity_rmse_mps", "acceleration_rmse_mps2", "six_state_mse_scaled_units", "six_state_mse_db")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json_exclusive(path: Path, value: Any):
    with path.open("x", encoding="utf-8") as f:
        json.dump(jsonable(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def parameters(n: int):
    args = simparams_class()
    args.dt = 1.0; args.seq_len = 100; args.dataset_size = n
    args.meas_type = "RANGE_BEARING"; args.init_type = "MEAS"; args.noise_ref = "FIXED"
    args.range_std = 30.0; args.bearing_std = 1.0; args.q2_scale = 1.0; args.r2_scale = 1.0
    validate_simparams(args)
    return args


def load_training_only():
    """Load and hash only train split tensors; does not open the test dataset."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pool_hash = sha256_file(POOL_PATH)
    if pool_hash != manifest.get("fixed_pool_file_sha256"):
        raise RuntimeError("Training pool file SHA-256 does not match its manifest")
    pool = torch.load(POOL_PATH, map_location="cpu", weights_only=True)
    train_state, train_obs, train_init = training_split_from_pool(pool, manifest)
    # Drop the container so validation tensors in the serialized pool cannot be
    # accidentally threaded into the estimation or evaluation call paths.
    del pool
    return manifest, pool_hash, train_state, train_obs, train_init


def simulator_oracle_covariances(args):
    """Derive generator Q/R and transform them to stored benchmark units."""
    profile = get_motion_profile(args)
    jerk_variance = (profile["accn_rms_q"] ** 2) / args.dt
    q_m2 = Q_CA_2D(args.dt, jerk_variance, jerk_variance) * args.q2_scale
    q_stored = q_m2 / (1000.0 ** 2)
    r_range_bearing = torch.diag(torch.tensor([
        (args.range_std ** 2) * args.r2_scale / (1000.0 ** 2),
        (args.bearing_std * math.pi / 180.0) ** 2 * args.r2_scale,
    ], dtype=torch.float64))
    return q_stored.to(torch.float64), r_range_bearing


def compare_estimates(estimated, q_oracle, r_oracle):
    q = estimated["process_covariance_final"]
    r = estimated["measurement_covariance_final"]
    q_relative = torch.linalg.norm(q - q_oracle) / torch.linalg.norm(q_oracle)
    r_diag_relative = torch.diag(r) / torch.diag(r_oracle) - 1.0
    r_full = estimated["measurement_residual_covariance_full"]
    r_cross_correlation = r_full[0, 1] / torch.sqrt(r_full[0, 0] * r_full[1, 1])
    significant = bool(float(q_relative) > 0.05 or torch.any(r_diag_relative.abs() > 0.05))
    return {
        "Q_relative_frobenius_error": q_relative,
        "R_diagonal_relative_error": r_diag_relative,
        "R_raw_range_bearing_correlation": r_cross_correlation,
        "significant_discrepancy_threshold": "5% relative Frobenius error for Q or absolute relative diagonal variance error for R",
        "significant_discrepancy": significant,
    }


def covariance_diagnostics(p: torch.Tensor):
    sym = 0.5 * (p + p.transpose(-1, -2))
    scale = sym.diagonal(dim1=-2, dim2=-1).abs().amax().clamp_min(1e-12)
    symmetry_error = (p - p.transpose(-1, -2)).abs().max()
    minimum_eigenvalue = torch.linalg.eigvalsh(sym).min()
    return {"max_abs_symmetry_error": float(symmetry_error),
            "minimum_eigenvalue": float(minimum_eigenvalue),
            "covariance_diagonal_scale": float(scale),
            "relative_symmetry_error": float(symmetry_error / scale),
            "relative_minimum_eigenvalue": float(minimum_eigenvalue / scale)}


def run_ekf(model, args, q, r, test, *, p0_legacy, p0_calibrated, cross, calibrated: bool):
    filt = extended_kf(model, q.to(torch.float32), r.to(torch.float32), args)
    filt.reset(test["initial_state"])
    estimates = torch.empty_like(test["true_state"])
    diagnostics = []
    start = time.perf_counter()
    with torch.inference_mode():
        if calibrated:
            filt.x = model.f(test["initial_state"])
            filt.P = p0_calibrated.to(torch.float32).unsqueeze(0).expand(args.dataset_size, -1, -1).clone()
            diagnostics.append(covariance_diagnostics(filt.P))
            for t in range(args.seq_len):
                if t == 0:
                    correlated_first_observation_update(filt, test["observations"][:, :, t], cross)
                else:
                    filt.predict(); filt.update(test["observations"][:, :, t])
                estimates[:, :, t] = filt.x.squeeze(-1)
                if t in (0, 1, 9, 49, args.seq_len - 1): diagnostics.append(covariance_diagnostics(filt.P))
        else:
            filt.P = p0_legacy.to(torch.float32).clone()
            # Same historical batch EKF path and prediction-before-observation convention.
            # This matches batch_ekf.evaluate_batch: predict then update for each
            # returned observation, beginning from the exact recorded P0.
            for t in range(args.seq_len):
                filt.predict(); filt.update(test["observations"][:, :, t])
                estimates[:, :, t] = filt.x.squeeze(-1)
                if t in (0, 1, 9, 49, args.seq_len - 1): diagnostics.append(covariance_diagnostics(filt.P))
    return estimates, time.perf_counter() - start, diagnostics


def run_ukf(model, q, r, test, *, p0_legacy, p0_calibrated, cross, calibrated: bool):
    filt = RadarUKF(model, q, r)
    observations = test["observations"]
    n, _, length = observations.shape
    estimates = torch.empty((n, 6, length), dtype=torch.float32)
    diagnostics = []
    if calibrated:
        mean = (model.F.to(torch.float64) @ test["initial_state"].to(torch.float64)).squeeze(-1)
        cov = p0_calibrated.to(torch.float64).unsqueeze(0).expand(n, -1, -1).clone()
    else:
        mean = test["initial_state"].to(torch.float64).squeeze(-1)
        cov = p0_legacy.to(torch.float64).clone()
    start = time.perf_counter()
    with torch.inference_mode():
        if calibrated:
            mean, cov = filt.update(mean, cov, observations[:, :, 0], state_measurement_noise_cross_covariance=cross)
            estimates[:, :, 0] = mean.squeeze(-1).to(torch.float32)
            diagnostics.append(covariance_diagnostics(cov))
            begin = 1
        else:
            begin = 0
        for t in range(begin, length):
            mean, cov = filt.predict(mean, cov)
            mean, cov = filt.update(mean, cov, observations[:, :, t])
            estimates[:, :, t] = mean.squeeze(-1).to(torch.float32)
            if t in (0, 1, 9, 49, length - 1): diagnostics.append(covariance_diagnostics(cov))
    return estimates, time.perf_counter() - start, diagnostics, filt.last_diagnostics


def git_info():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_porcelain": status}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT}")
    manifest, pool_hash, train_state, train_obs, train_init = load_training_only()
    args = parameters(train_state.shape[0])
    model = ConstantAcceleration2DModel(args)
    # Estimate before the held-out test file is opened.
    estimate = estimate_training_noise(train_state, train_obs, model)
    oracle_q, oracle_r = simulator_oracle_covariances(args)
    discrepancy = compare_estimates(estimate, oracle_q, oracle_r)
    training_hashes = {"state": tensor_sha256(train_state), "observations": tensor_sha256(train_obs),
                       "initial_state": tensor_sha256(train_init)}
    frozen = {
        "name": "estimated_QR_train_only",
        "training_pool_path": str(POOL_PATH), "training_pool_sha256": pool_hash,
        "training_manifest_sha256": sha256_file(MANIFEST_PATH),
        "training_tensor_sha256": training_hashes,
        "training_trajectory_count": estimate["training_trajectory_count"],
        "state_observation_count": estimate["state_observation_count"],
        "process_transition_count": estimate["process_transition_count"],
        "process_residual_sample_count": estimate["process_residual_sample_count"],
        "measurement_residual_sample_count": estimate["measurement_residual_sample_count"],
        "estimation_formula": estimate["method"],
        "units": {"state": estimate["state_units"], "observation": estimate["observation_units"]},
        "training_estimates": estimate,
        "simulator_oracle_Q": oracle_q, "simulator_oracle_R": oracle_r,
        "oracle_comparison": discrepancy,
        "information_boundary": "estimator received only train state and train observations; the fixed test file was opened only after the frozen estimate was constructed. The serialized fixed-pool container also contains validation tensors, but the training-split extractor and estimator do not read or receive those values",
    }
    # Capture an immutable in-memory configuration before opening held-out data.
    frozen_snapshot = jsonable(frozen)
    print("Pre-evaluation Q/R comparison:", json.dumps(jsonable(discrepancy), indent=2))

    if sha256_file(TEST_PATH) != TEST_SHA256:
        raise RuntimeError("Fixed test file SHA-256 mismatch")
    test = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    expected_shapes = {"true_state": (1024, 6, 100), "observations": (1024, 2, 100),
                       "initial_state": (1024, 6, 1), "process_noise": (6, 6), "measurement_noise": (2, 2)}
    for key, shape in expected_shapes.items():
        if tuple(test[key].shape) != shape or not torch.isfinite(test[key]).all():
            raise ValueError(f"Malformed fixed test tensor {key}")
    # The EKF allocates a fixed batch at construction; this evaluation setting
    # comes from the held-out test shape only after the Q/R estimate is frozen.
    args.dataset_size = test["true_state"].shape[0]
    test_q = test["process_noise"].to(torch.float64)
    test_r = test["measurement_noise"].to(torch.float64)
    if not torch.allclose(test_q, oracle_q, rtol=2e-6, atol=1e-12) or not torch.allclose(test_r, oracle_r, rtol=2e-6, atol=1e-10):
        raise RuntimeError("Fixed test Q/R disagree with simulator configuration-derived oracle matrices")

    p0_record = json.loads(EKF_RESULT_PATH.read_text(encoding="utf-8"))["initialization"]
    p0_legacy = torch.as_tensor(p0_record["legacy_P0_before_first_predict"], dtype=torch.float64)
    p0_cal = torch.as_tensor(p0_record["calibrated_P0"], dtype=torch.float64)
    cross = torch.as_tensor(p0_record["first_observation_state_noise_cross_covariance"], dtype=torch.float64)
    if not torch.equal(p0_legacy, torch.as_tensor(p0_record["legacy_P0_before_first_predict"], dtype=torch.float64)):
        raise RuntimeError("Legacy P0 mismatch")
    if not torch.equal(p0_cal, torch.as_tensor(p0_record["calibrated_P0"], dtype=torch.float64)):
        raise RuntimeError("Calibrated P0 mismatch")

    q_est = estimate["process_covariance_final"]
    r_est = estimate["measurement_covariance_final"]
    # Same P0 treatment and first-observation cross covariance across each pair.
    outputs: dict[str, Any] = {}
    for family in ("ekf", "ukf"):
        for init_variant in ("legacy", "calibrated"):
            for policy, q, r in (("oracle", oracle_q, oracle_r), ("estimated_QR", q_est, r_est)):
                calibrated = init_variant == "calibrated"
                name = f"{init_variant}_{family}_{policy}" if policy == "estimated_QR" else f"{init_variant}_{family}_oracle"
                if family == "ekf":
                    pred, runtime, covdiag = run_ekf(model, args, q, r, test, p0_legacy=p0_legacy,
                                                     p0_calibrated=p0_cal, cross=cross, calibrated=calibrated)
                    extra = {"covariance_snapshots": covdiag}
                else:
                    pred, runtime, covdiag, diag = run_ukf(model, q, r, test, p0_legacy=p0_legacy,
                                                           p0_calibrated=p0_cal, cross=cross, calibrated=calibrated)
                    extra = {"covariance_snapshots": covdiag, "filter_diagnostics": diag}
                outputs[name] = {"metrics": five_seed.calculate_metrics(torch, test["true_state"], pred, args.dt),
                                 "runtime_seconds": runtime, "evaluation_batch_size": 1024,
                                 "finite_predictions": bool(torch.isfinite(pred).all()),
                                 "initialization": f"{init_variant}_P0 from Task 3A", "Q_R_policy": policy,
                                 "Q_R_configuration": "estimated_QR_train_only" if policy == "estimated_QR" else "simulator oracle matrices",
                                 "covariance_stability_warning": any(
                                     snapshot["relative_symmetry_error"] > 1e-6
                                     or snapshot["relative_minimum_eigenvalue"] < -1e-6
                                     for snapshot in extra["covariance_snapshots"]
                                 ), **extra}
                if not outputs[name]["finite_predictions"]:
                    raise RuntimeError(f"Non-finite predictions from {name}")

    oracle_reproduction = {}
    for family, result_path in (("ekf", EKF_RESULT_PATH), ("ukf", UKF_RESULT_PATH)):
        historical = json.loads(result_path.read_text(encoding="utf-8"))["variants"]
        for init_variant in ("legacy", "calibrated"):
            name = f"{init_variant}_{family}_oracle"
            historical_name = f"{init_variant}_{family}_oracle"
            differences = {}
            for metric in METRIC_KEYS:
                actual = outputs[name]["metrics"][metric]
                expected = historical[historical_name]["metrics"][metric]
                differences[metric] = {"current": actual, "recorded": expected, "absolute_difference": abs(actual - expected),
                                       "pass": math.isclose(actual, expected, rel_tol=5e-4, abs_tol=1e-5)}
                if not differences[metric]["pass"]:
                    raise RuntimeError(f"Oracle reproduction check failed for {name}/{metric}")
            oracle_reproduction[name] = differences

    # Confirm no source dataset changed while running.
    if sha256_file(TEST_PATH) != TEST_SHA256 or sha256_file(POOL_PATH) != pool_hash:
        raise RuntimeError("An input dataset changed during evaluation")
    learned_meta = {
        "fixed_test_path": str(TEST_PATH), "fixed_test_sha256": TEST_SHA256,
        "fixed_test_tensor_sha256": {key: tensor_sha256(test[key]) for key in expected_shapes},
        "training_path": str(POOL_PATH), "training_file_sha256": pool_hash,
        "training_tensor_sha256": training_hashes,
        "training_manifest_sha256": sha256_file(MANIFEST_PATH),
    }
    compare_rows = {}
    for family in ("ekf", "ukf"):
        for init_variant in ("legacy", "calibrated"):
            oracle_name = f"{init_variant}_{family}_oracle"
            est_name = f"{init_variant}_{family}_estimated_QR"
            compare_rows[f"{init_variant}_{family}"] = {
                metric: {"oracle": outputs[oracle_name]["metrics"][metric],
                         "estimated_QR": outputs[est_name]["metrics"][metric],
                         "estimated_minus_oracle": outputs[est_name]["metrics"][metric] - outputs[oracle_name]["metrics"][metric]}
                for metric in METRIC_KEYS
            }

    sources = ["src/radar_tracking/models/noise_estimation.py", "src/radar_tracking/models/ekf_baselines.py",
               "src/radar_tracking/models/ekf.py", "src/radar_tracking/models/ukf.py",
               "src/radar_tracking/data/data_generation.py", "src/radar_tracking/dynamics/constant_acceleration.py",
               "experiments/five_seed_harness.py", "experiments/evaluate_ekf_baselines.py",
               "experiments/evaluate_ukf_baselines.py", "experiments/evaluate_estimated_noise_baselines.py",
               "tests/test_noise_estimation.py"]
    result = {
        "format_version": 1, "configuration_name": "estimated_QR_train_only",
        "evaluation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_estimator_configuration": frozen_snapshot,
        "oracle_matrices_verified_against_fixed_test": {"Q": oracle_q, "R": oracle_r},
        "fixed_test": learned_meta,
        "evaluation": {"dt": args.dt, "sequence_length": args.seq_len, "test_count": 1024,
                       "state_order": ["x", "vx", "ax", "y", "vy", "ay"], "state_scale": five_seed.SCALE,
                       "metric_function": "experiments/five_seed_harness.py:calculate_metrics", "device": "CPU",
                       "P0_source": "same exact legacy/calibrated P0 and first-observation cross covariance from Task 3A"},
        "variants": outputs, "oracle_vs_estimated_QR": compare_rows,
        "oracle_reproduction_checks": oracle_reproduction,
        "provenance": {"git": git_info(), "source_file_sha256": {s: sha256_file(PROJECT / s) for s in sources},
                       "python_version": platform.python_version(), "pytorch_version": torch.__version__,
                       "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available()},
        "preservation_note": "No model training, checkpoint, historical result, fixed dataset, or existing oracle baseline output was modified."}

    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(OUTPUT / "estimated_QR_train_only.json", frozen_snapshot)
    write_json_exclusive(OUTPUT / "estimated_noise_baseline_results.json", result)
    with (OUTPUT / "eight_variant_metrics.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", *METRIC_KEYS, "runtime_seconds"])
        for name, record in outputs.items():
            writer.writerow([name, *(record["metrics"][key] for key in METRIC_KEYS), record["runtime_seconds"]])
    print(f"Results written to {OUTPUT}")
    print(json.dumps({name: value["metrics"] for name, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
