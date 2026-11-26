#!/usr/bin/env python3
"""Evaluate legacy and calibrated UKF variants on the immutable fixed test set."""
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
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))

import five_seed_harness as five_seed
from radar_tracking.config import simparams_class, validate_simparams
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.models.ekf_baselines import covariance_for_policy, tensor_sha256, training_split_from_pool
from radar_tracking.models.ukf import RadarUKF

OUTPUT_DIR = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/ukf_baseline_v1"
TEST_PATH = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"
EXPECTED_TEST_SHA256 = "d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2"
POOL_ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
EKF_RESULTS = POOL_ROOT / "ekf_baseline_v1/ekf_baseline_results.json"
METRICS = (
    "position_rmse_m", "velocity_rmse_mps", "acceleration_rmse_mps2",
    "six_state_mse_scaled_units", "six_state_mse_db",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(type(value).__name__)


def make_params(n: int):
    args = simparams_class()
    args.dt = 1.0
    args.seq_len = 100
    args.dataset_size = n
    args.meas_type = "RANGE_BEARING"
    args.init_type = "MEAS"
    args.noise_ref = "FIXED"
    args.range_std = 30.0
    args.bearing_std = 1.0
    args.q2_scale = 1.0
    args.r2_scale = 1.0
    validate_simparams(args)
    return args


def load_data():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT_DIR}")
    if sha256_file(TEST_PATH) != EXPECTED_TEST_SHA256:
        raise RuntimeError("Fixed test file SHA-256 mismatch")
    test = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    for name, shape in {
        "true_state": (1024, 6, 100), "observations": (1024, 2, 100),
        "initial_state": (1024, 6, 1), "process_noise": (6, 6), "measurement_noise": (2, 2),
    }.items():
        if name not in test or tuple(test[name].shape) != shape or not torch.isfinite(test[name]).all():
            raise ValueError(f"Fixed test tensor {name} missing, malformed, or non-finite")
    ekf_results = json.loads(EKF_RESULTS.read_text(encoding="utf-8"))
    init = ekf_results["initialization"]
    p0_legacy = torch.as_tensor(init["legacy_P0_before_first_predict"], dtype=torch.float64)
    p0_calibrated = torch.as_tensor(init["calibrated_P0"], dtype=torch.float64)
    cross = torch.as_tensor(init["first_observation_state_noise_cross_covariance"], dtype=torch.float64)
    if tuple(p0_legacy.shape) != (1024, 6, 6) or tuple(p0_calibrated.shape) != (6, 6) or tuple(cross.shape) != (6, 2):
        raise ValueError("Recorded EKF P0/cross-covariance shapes are not the expected variants")
    pool_path = POOL_ROOT / "fixed_train_val.pt"
    manifest_path = POOL_ROOT / "fixed_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_hash = sha256_file(pool_path)
    if pool_hash != manifest["fixed_pool_file_sha256"]:
        raise RuntimeError("Fixed training pool hash mismatch")
    pool = torch.load(pool_path, map_location="cpu", weights_only=True)
    train_state, train_obs, train_init = training_split_from_pool(pool, manifest)
    return test, ekf_results, p0_legacy, p0_calibrated, cross, manifest, pool_path, pool_hash, train_state, train_obs, train_init


def covariance_check(p: torch.Tensor) -> dict[str, float]:
    sym_err = (p - p.transpose(-1, -2)).abs().max()
    min_eig = torch.linalg.eigvalsh(0.5 * (p + p.transpose(-1, -2))).min()
    return {"max_abs_symmetry_error": float(sym_err), "minimum_eigenvalue": float(min_eig)}


def validate_before_benchmark(model, q, r, p0_legacy, p0_calibrated, cross, train_state, train_obs, train_init):
    checks: dict[str, Any] = {}
    ukf = RadarUKF(model, q, r)
    # Sigma reconstruction with non-diagonal covariance.
    mean = torch.tensor([[1.0, 0.2, -0.01, 3.0, -0.1, 0.02]], dtype=torch.float64)
    a = torch.tensor([[1., .1, 0, .2, 0, 0], [0, .5, .05, 0, .1, 0], [0, 0, .2, 0, 0, .03],
                      [0, 0, 0, 1.1, .1, 0], [0, 0, 0, 0, .6, .1], [0, 0, 0, 0, 0, .25]], dtype=torch.float64)
    p = (a @ a.T).unsqueeze(0)
    sigmas, stabilized = ukf.sigma_points(mean, p)
    rec_mean = ukf.weighted_mean(sigmas)
    dx = sigmas - rec_mean[:, None, :]
    rec_p = torch.einsum("s,bsi,bsj->bij", ukf.weights_covariance, dx, dx)
    if not torch.allclose(rec_mean, mean, rtol=1e-12, atol=1e-12) or not torch.allclose(rec_p, stabilized, rtol=1e-11, atol=1e-12):
        raise RuntimeError("Sigma-point reconstruction validation failed")
    checks["sigma_reconstruction"] = "pass"

    # Linear transition exactly agrees with F P F^T + Q.
    pred_mean, pred_p = ukf.predict(mean, p)
    F = model.F.to(torch.float64)
    expected_mean = F @ mean.unsqueeze(-1)
    expected_p = F @ p @ F.T + q
    if not torch.allclose(pred_mean, expected_mean, rtol=1e-12, atol=1e-12) or not torch.allclose(pred_p, expected_p, rtol=1e-10, atol=1e-12):
        raise RuntimeError("UKF/EKF linear state prediction validation failed")
    checks["linear_prediction"] = "pass"

    # Symmetric sigma bearing values around zero have a circular mean at zero.
    angles = torch.zeros((1, 13, 2), dtype=torch.float64)
    angles[:, 1:7, 1] = 2 * math.pi - 0.01
    angles[:, 7:13, 1] = 0.01
    circular_mean = ukf.weighted_mean(angles, angle_indices=(1,))[0, 1]
    residual = torch.remainder(torch.tensor(0.005 - (2 * math.pi - 0.005), dtype=torch.float64) + math.pi, 2 * math.pi) - math.pi
    circular_error = math.atan2(math.sin(float(circular_mean)), math.cos(float(circular_mean)))
    if abs(circular_error) > 1e-10 or abs(float(residual) - 0.01) > 1e-12:
        raise RuntimeError("Bearing boundary validation failed")
    checks["bearing_boundary"] = "pass"

    # Explicitly assert loaded P0 arrays equal the recorded JSON representation.
    recorded = json.loads(EKF_RESULTS.read_text(encoding="utf-8"))["initialization"]
    if not torch.equal(p0_legacy, torch.as_tensor(recorded["legacy_P0_before_first_predict"], dtype=torch.float64)):
        raise RuntimeError("Legacy P0 does not match EKF baseline record")
    if not torch.equal(p0_calibrated, torch.as_tensor(recorded["calibrated_P0"], dtype=torch.float64)):
        raise RuntimeError("Calibrated P0 does not match EKF baseline record")
    checks["P0_exact_recorded_values"] = "pass"

    # Run a handful of training-only trajectories to inspect covariance behavior.
    batch = min(8, train_state.shape[0])
    local_model = model
    local_ukf = RadarUKF(local_model, q, r)
    m = (F @ train_init[:batch].to(torch.float64)).squeeze(-1)
    cov = p0_calibrated.unsqueeze(0).expand(batch, -1, -1).clone()
    diagnostics = [covariance_check(cov)]
    with torch.inference_mode():
        m, cov = local_ukf.update(m, cov, train_obs[:batch, :, 0], state_measurement_noise_cross_covariance=cross)
        diagnostics.append(covariance_check(cov))
        for t in range(1, train_obs.shape[2]):
            m, cov = local_ukf.predict(m, cov)
            m, cov = local_ukf.update(m, cov, train_obs[:batch, :, t])
            if t in (1, 9, 49, train_obs.shape[2] - 1):
                diagnostics.append(covariance_check(cov))
    if not torch.isfinite(m).all() or any(d["max_abs_symmetry_error"] > 1e-9 or d["minimum_eigenvalue"] < -1e-8 for d in diagnostics):
        raise RuntimeError(f"Training sequence covariance stability validation failed: {diagnostics}")
    checks["training_sequence_covariance"] = {"trajectories": batch, "snapshots": diagnostics, "pass": True}
    return checks


def run_filter(model, q, r, mean, covariance, observations, *, cross=None):
    filt = RadarUKF(model, q, r)
    n, _, length = observations.shape
    estimates = torch.empty((n, 6, length), dtype=torch.float32)
    cov_checks = []
    started = time.perf_counter()
    with torch.inference_mode():
        if cross is not None:
            mean, covariance = filt.update(mean, covariance, observations[:, :, 0], state_measurement_noise_cross_covariance=cross)
            estimates[:, :, 0] = mean.squeeze(-1).to(torch.float32)
            cov_checks.append(covariance_check(covariance))
            begin = 1
        else:
            begin = 0
        for t in range(begin, length):
            mean, covariance = filt.predict(mean, covariance)
            mean, covariance = filt.update(mean, covariance, observations[:, :, t])
            estimates[:, :, t] = mean.squeeze(-1).to(torch.float32)
            if t in (0, 1, 9, 49, length - 1):
                cov_checks.append(covariance_check(covariance))
    return estimates, time.perf_counter() - started, cov_checks, filt.last_diagnostics


def git_provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_porcelain": status}


def main():
    (test, ekf_results, p0_legacy, p0_calibrated, cross, manifest, pool_path, pool_hash,
     train_state, train_obs, train_init) = load_data()
    args = make_params(test["true_state"].shape[0])
    model = ConstantAcceleration2DModel(args)
    q, r = covariance_for_policy("oracle_QR", test["process_noise"], test["measurement_noise"])
    checks = validate_before_benchmark(model, q, r, p0_legacy, p0_calibrated, cross, train_state, train_obs, train_init)

    variants = {}
    mean_legacy = test["initial_state"].to(torch.float64).squeeze(-1)
    legacy_est, legacy_runtime, legacy_cov, legacy_diag = run_filter(
        model, q, r, mean_legacy, p0_legacy, test["observations"]
    )
    mean_calibrated = (model.F.to(torch.float64) @ test["initial_state"].to(torch.float64)).squeeze(-1)
    cal_est, cal_runtime, cal_cov, cal_diag = run_filter(
        model, q, r, mean_calibrated, p0_calibrated.unsqueeze(0).expand(1024, -1, -1).clone(),
        test["observations"], cross=cross,
    )
    for name, estimate, runtime, cov_diagnostics, filter_diagnostics in (
        ("legacy_ukf_oracle", legacy_est, legacy_runtime, legacy_cov, legacy_diag),
        ("calibrated_ukf_oracle", cal_est, cal_runtime, cal_cov, cal_diag),
    ):
        variants[name] = {
            "metrics": five_seed.calculate_metrics(torch, test["true_state"], estimate, args.dt),
            "runtime_seconds_cpu_float64_covariance": runtime,
            "evaluation_batch_size": 1024,
            "finite_predictions": bool(torch.isfinite(estimate).all()),
            "covariance_snapshots": cov_diagnostics,
            "filter_final_diagnostics": filter_diagnostics,
        }
    all_methods = {name: {"metrics": value["metrics"], "source": "UKF fixed-test evaluation"} for name, value in variants.items()}
    for name in ("legacy_ekf_oracle", "calibrated_ekf_oracle"):
        all_methods[name] = {"metrics": ekf_results["variants"][name]["metrics"], "source": "Task 3A EKF fixed-test results"}

    source_files = [
        "src/radar_tracking/models/ukf.py", "src/radar_tracking/models/ekf.py",
        "src/radar_tracking/models/ekf_baselines.py", "src/radar_tracking/dynamics/constant_acceleration.py",
        "experiments/five_seed_harness.py", "experiments/evaluate_ekf_baselines.py", "experiments/evaluate_ukf_baselines.py",
    ]
    source_hashes = {name: sha256_file(PROJECT / name) for name in source_files}
    payload = {
        "format_version": 1,
        "evaluation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "validation": checks,
        "variants": variants,
        "four_method_comparison": all_methods,
        "covariance_policy": {"name": "oracle_QR", "source": "actual Q and R tensors in verified shared test payload, passed unchanged", "Q": q, "R": r},
        "initialization": {
            "legacy_P0_source": "Task 3A ekf_baseline_results.json; exact batch P0 immediately before first predict",
            "legacy_P0": p0_legacy,
            "calibrated_P0_source": "Task 3A training-only calibration result; exact same 6x6 matrix used by calibrated EKF",
            "calibrated_P0": p0_calibrated,
            "first_observation_state_measurement_noise_cross_covariance": cross,
            "calibration_method": ekf_results["initialization"]["calibrated_P0_method"],
            "calibration_training_pool_sha256": sha256_file(pool_path),
        },
        "unscented_transform": {"n": 6, "alpha": 1.0, "beta": 2.0, "kappa": 0.0, "lambda": 0.0,
            "scale_n_plus_lambda": 6.0, "Wm": RadarUKF(model, q, r).weights_mean, "Wc": RadarUKF(model, q, r).weights_covariance,
            "covariance_policy": "symmetrize; reject eigenvalues below -1e-8 relative diagonal scale; Cholesky with relative jitter 1e-12 through 1e-6; fail after seven attempts"},
        "test_dataset": {"path": str(TEST_PATH), "file_sha256": sha256_file(TEST_PATH),
            "tensor_sha256": {name: tensor_sha256(test[name]) for name in ("true_state", "observations", "initial_state", "process_noise", "measurement_noise")}},
        "training_dataset": {"path": str(pool_path), "file_sha256": pool_hash, "manifest_sha256": sha256_file(POOL_ROOT / "fixed_pool_manifest.json"),
            "train_state_sha256": tensor_sha256(train_state), "train_observation_sha256": tensor_sha256(train_obs),
            "train_initial_state_sha256": tensor_sha256(train_init), "purpose": "short covariance stability validation only; P0 was reused from Task 3A"},
        "evaluation_settings": {"dt": args.dt, "sequence_length": 100, "test_count": 1024, "state_order": ["x", "vx", "ax", "y", "vy", "ay"],
            "metric_source": "experiments/five_seed_harness.py:calculate_metrics", "state_scale": five_seed.SCALE, "device": "CPU", "covariance_dtype": "float64", "prediction_dtype": "float32"},
        "provenance": {"git": git_provenance(), "source_file_sha256": source_hashes, "python_version": platform.python_version(),
            "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "device": "CPU"},
        "historical_note": "No existing EKF, learned checkpoint, shared test, manifest, or historical result was modified."}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    with (OUTPUT_DIR / "ukf_baseline_results.json").open("x", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    with (OUTPUT_DIR / "four_method_metrics.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *METRICS])
        for name, row in all_methods.items():
            writer.writerow([name, *(row["metrics"][metric] for metric in METRICS)])
    print(f"UKF fixed-test results written to {OUTPUT_DIR}")
    print(json.dumps({name: row["metrics"] for name, row in all_methods.items()}, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
