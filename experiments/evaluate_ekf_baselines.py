#!/usr/bin/env python3
"""Evaluate legacy and training-calibrated EKF variants on the fixed test set."""

from __future__ import annotations

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
from radar_tracking.models.ekf import batch_ekf, extended_kf
from radar_tracking.models.ekf_baselines import (
    assert_legacy_ekf_inputs,
    calibrate_first_observation_prior,
    correlated_first_observation_update,
    covariance_for_policy,
    estimate_qr_from_training,
    tensor_sha256,
    training_split_from_pool,
)

OUTPUT_DIR = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/ekf_baseline_v1"
TEST_PATH = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"
EXPECTED_TEST_SHA256 = "d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2"
POOL_ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
REFERENCE_METRICS = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/n1024/benchmark_metrics.json"
SEEDS = (11, 22, 33, 44, 55)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def to_json(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def make_simparams(test_size: int):
    args = simparams_class()
    args.dt = 1.0
    args.seq_len = 100
    args.dataset_size = test_size
    args.meas_type = "RANGE_BEARING"
    args.init_type = "MEAS"
    args.noise_ref = "FIXED"
    args.range_std = 30.0
    args.bearing_std = 1.0
    args.q2_scale = 1.0
    args.r2_scale = 1.0
    validate_simparams(args)
    return args


def load_inputs() -> tuple[dict, dict, dict, str, str]:
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Fixed test dataset is missing: {TEST_PATH}")
    test_file_hash = sha256_file(TEST_PATH)
    if test_file_hash != EXPECTED_TEST_SHA256:
        raise RuntimeError(f"Test SHA-256 mismatch: {test_file_hash}")
    test = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    expected_shapes = {
        "true_state": (1024, 6, 100),
        "observations": (1024, 2, 100),
        "initial_state": (1024, 6, 1),
        "process_noise": (6, 6),
        "measurement_noise": (2, 2),
    }
    for name, shape in expected_shapes.items():
        if name not in test or not torch.is_tensor(test[name]) or tuple(test[name].shape) != shape:
            raise ValueError(f"Test tensor {name} must have shape {shape}")
        if not torch.isfinite(test[name]).all():
            raise ValueError(f"Test tensor {name} contains non-finite values")
    manifest_path = POOL_ROOT / "fixed_pool_manifest.json"
    pool_path = POOL_ROOT / "fixed_train_val.pt"
    if not manifest_path.is_file() or not pool_path.is_file():
        raise FileNotFoundError("Fixed training pool or its manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_hash = sha256_file(pool_path)
    if pool_hash != manifest.get("fixed_pool_file_sha256"):
        raise RuntimeError("Fixed training pool file hash mismatch")
    pool = torch.load(pool_path, map_location="cpu", weights_only=True)
    train_state, train_observations, train_initial_state = training_split_from_pool(pool, manifest)
    train_hashes = {
        "state": tensor_sha256(train_state),
        "observations": tensor_sha256(train_observations),
        "initial_state": tensor_sha256(train_initial_state),
    }
    return test, manifest, {
        "path": pool_path,
        "sha256": pool_hash,
        "tensor_sha256": train_hashes,
        "state": train_state,
        "observations": train_observations,
        "initial_state": train_initial_state,
    }, test_file_hash, sha256_file(manifest_path)


def run_legacy(model, args, q, r, test) -> tuple[torch.Tensor, torch.Tensor, float]:
    filt = extended_kf(model, q, r, args)
    filt.reset(test["initial_state"])
    legacy_p0 = filt.P.clone()
    assert_legacy_ekf_inputs(
        test["observations"].shape[0], args.dataset_size, test["observations"].device
    )
    started = time.perf_counter()
    with torch.inference_mode():
        estimate, *_ = batch_ekf(filt).evaluate_batch(
            test["true_state"], test["observations"], test["initial_state"]
        )
    return estimate, legacy_p0, time.perf_counter() - started


def covariance_diagnostics(covariances: torch.Tensor) -> dict[str, float]:
    values = covariances.detach().to(torch.float64)
    symmetric = 0.5 * (values + values.transpose(-1, -2))
    symmetry_error = (values - values.transpose(-1, -2)).abs().max()
    min_eigenvalue = torch.linalg.eigvalsh(symmetric).min()
    return {
        "max_abs_symmetry_error": float(symmetry_error),
        "minimum_symmetric_eigenvalue": float(min_eigenvalue),
    }


def run_calibrated(model, args, q, r, test, calibration) -> tuple[torch.Tensor, dict, float]:
    filt = extended_kf(model, q, r, args)
    filt.reset(test["initial_state"])
    # Calibration P0 is the one-step prior immediately before observation 0.
    filt.x = model.f(test["initial_state"])
    p0 = calibration["P0"].to(torch.float32)
    filt.P = p0.unsqueeze(0).expand(args.dataset_size, -1, -1).clone()
    diagnostics = [covariance_diagnostics(filt.P)]
    estimates = torch.zeros_like(test["true_state"])
    assert_legacy_ekf_inputs(
        test["observations"].shape[0], args.dataset_size, test["observations"].device
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for time_index in range(test["observations"].shape[2]):
            if time_index == 0:
                correlated_first_observation_update(
                    filt,
                    test["observations"][:, :, time_index],
                    calibration["first_observation_state_noise_cross_covariance"],
                )
            else:
                filt.predict()
                filt.update(test["observations"][:, :, time_index])
            estimates[:, :, time_index] = filt.x.squeeze(-1)
            diagnostics.append(covariance_diagnostics(filt.P))
    runtime = time.perf_counter() - started
    summary = {
        "P0": p0,
        "first_observation_state_noise_cross_covariance": calibration[
            "first_observation_state_noise_cross_covariance"
        ],
        "covariance_diagnostics": {
            "max_abs_symmetry_error": max(item["max_abs_symmetry_error"] for item in diagnostics),
            "minimum_symmetric_eigenvalue": min(item["minimum_symmetric_eigenvalue"] for item in diagnostics),
        },
    }
    return estimates, summary, runtime


def stored_learned_summaries() -> dict[str, Any]:
    metrics = (
        "position_rmse_m",
        "velocity_rmse_mps",
        "acceleration_rmse_mps2",
        "six_state_mse_scaled_units",
        "six_state_mse_db",
    )
    rows = {method: {metric: [] for metric in metrics} for method in ("mamba", "knet")}
    checkpoint_hashes = []
    for seed in SEEDS:
        run_dir = POOL_ROOT / f"seed_{seed}"
        record = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
        for method, filename in (("mamba", "best_mamba_tracker.pt"), ("knet", "best_knet_seq100.pt")):
            checkpoint = run_dir / filename
            actual_hash = sha256_file(checkpoint)
            if actual_hash != record[method]["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
            checkpoint_hashes.append({"seed": seed, "model": method, "sha256": actual_hash})
            for metric in metrics:
                rows[method][metric].append(float(record[method]["metrics"][metric]))
    summaries = {}
    for method, method_rows in rows.items():
        summaries[method] = {}
        for metric, values in method_rows.items():
            mean = sum(values) / len(values)
            sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            summaries[method][metric] = {
                "seed_values": values,
                "mean": mean,
                "sample_sd": sample_sd,
            }
    return {"summaries": summaries, "checkpoint_sha256": checkpoint_hashes}


def git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(PROJECT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_porcelain": status}


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT_DIR}")
    test, pool_manifest, training, test_file_hash, pool_manifest_hash = load_inputs()
    args = make_simparams(test["true_state"].shape[0])
    model = ConstantAcceleration2DModel(args)

    oracle_q, oracle_r = covariance_for_policy(
        "oracle_QR", test["process_noise"], test["measurement_noise"]
    )
    if args.q2_scale != 1.0 or args.r2_scale != 1.0:
        raise RuntimeError("This fixed test record expects unit scales; oracle_QR must use its actual Q/R unchanged")

    calibration = calibrate_first_observation_prior(
        training["state"], training["observations"], training["initial_state"], model
    )
    estimated_q, estimated_r = estimate_qr_from_training(
        training["state"], training["observations"], model
    )
    estimate_qr = covariance_for_policy(
        "estimated_QR", oracle_q, oracle_r, estimated=(estimated_q, estimated_r)
    )

    legacy_estimate, legacy_p0, legacy_seconds = run_legacy(
        model, args, oracle_q, oracle_r, test
    )
    calibrated_estimate, calibrated_meta, calibrated_seconds = run_calibrated(
        model, args, oracle_q, oracle_r, test, calibration
    )
    if sha256_file(TEST_PATH) != test_file_hash:
        raise RuntimeError("Fixed test file changed during evaluation")
    if sha256_file(training["path"]) != training["sha256"]:
        raise RuntimeError("Fixed training pool changed during calibration")

    metric_results = {}
    for name, prediction in (
        ("legacy_ekf_oracle", legacy_estimate),
        ("calibrated_ekf_oracle", calibrated_estimate),
    ):
        metric_results[name] = {
            "metrics": five_seed.calculate_metrics(torch, test["true_state"], prediction, args.dt),
            "runtime_seconds_cpu": legacy_seconds if name == "legacy_ekf_oracle" else calibrated_seconds,
            "evaluation_batch_size": int(args.dataset_size),
            "prediction_shape": list(prediction.shape),
            "finite_predictions": bool(torch.isfinite(prediction).all()),
        }

    stored = json.loads(REFERENCE_METRICS.read_text(encoding="utf-8"))["ekf"]
    legacy_metrics = metric_results["legacy_ekf_oracle"]["metrics"]
    historical_comparison = {}
    for observed_key, stored_key in (
        ("position_rmse_m", "position_rmse_m"),
        ("six_state_mse_scaled_units", "aggregate_six_state_mse"),
        ("six_state_mse_db", "total_mse_db"),
    ):
        actual = float(legacy_metrics[observed_key])
        expected = float(stored[stored_key])
        historical_comparison[observed_key] = {
            "recomputed": actual,
            "recorded": expected,
            "absolute_difference": abs(actual - expected),
            "pass": math.isclose(actual, expected, rel_tol=five_seed.EVAL_RTOL, abs_tol=five_seed.EVAL_ATOL),
        }

    source_files = (
        "src/radar_tracking/config.py",
        "src/radar_tracking/data/data_generation.py",
        "src/radar_tracking/dynamics/constant_acceleration.py",
        "src/radar_tracking/models/ekf.py",
        "src/radar_tracking/models/ekf_baselines.py",
        "experiments/five_seed_harness.py",
        "experiments/evaluate_ekf_baselines.py",
    )
    source_hashes = {name: sha256_file(PROJECT / name) for name in source_files}
    payload = {
        "format_version": 1,
        "evaluation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "variants": metric_results,
        "historical_legacy_ekf_comparison": historical_comparison,
        "covariance_policies": {
            "legacy_ekf_oracle": {
                "policy": "oracle_QR",
                "source": "actual process_noise and measurement_noise tensors in the verified fixed test payload; passed unchanged",
                "Q": oracle_q,
                "R": oracle_r,
            },
            "calibrated_ekf_oracle": {
                "policy": "oracle_QR",
                "source": "actual process_noise and measurement_noise tensors in the verified fixed test payload; passed unchanged",
                "Q": oracle_q,
                "R": oracle_r,
            },
            "estimated_QR_reference_from_training_only_not_evaluated": {
                "Q": estimate_qr[0],
                "R": estimate_qr[1],
                "training_only": True,
            },
        },
        "initialization": {
            "legacy_P0_before_first_predict": legacy_p0,
            "calibrated_P0": calibrated_meta["P0"],
            "calibrated_P0_method": calibration["method"],
            "calibrated_P0_location": calibration["covariance_location"],
            "calibrated_P0_units": calibration["units"],
            "calibration_sample_count": calibration["sample_count"],
            "calibration_training_error_mean": calibration["prior_error_mean"],
            "first_observation_state_noise_cross_covariance": calibrated_meta[
                "first_observation_state_noise_cross_covariance"
            ],
            "calibrated_covariance_diagnostics": calibrated_meta["covariance_diagnostics"],
        },
        "test_dataset": {
            "path": str(TEST_PATH),
            "file_sha256": test_file_hash,
            "metadata": test.get("metadata", {}),
            "tensor_sha256": {
                name: tensor_sha256(test[name])
                for name in ("true_state", "observations", "initial_state", "process_noise", "measurement_noise")
            },
            "tensor_shapes": {name: list(test[name].shape) for name in expected_test_names()},
        },
        "training_calibration_dataset": {
            "path": str(training["path"]),
            "pool_file_sha256": training["sha256"],
            "pool_manifest_sha256": pool_manifest_hash,
            "split": "train only; 819 trajectories; validation/test not passed to calibration functions",
            "tensor_sha256": training["tensor_sha256"],
            "data_seed": pool_manifest["data_seed"],
        },
        "evaluation_settings": {
            "dt": float(args.dt),
            "sequence_length": int(args.seq_len),
            "test_count": int(args.dataset_size),
            "state_order": ["x", "vx", "ax", "y", "vy", "ay"],
            "state_scale": 1000.0,
            "metric_source": "experiments/five_seed_harness.py:calculate_metrics",
            "legacy_filter_device": "cpu",
            "calibrated_filter_device": "cpu",
        },
        "five_seed_learned_checkpoint_summaries": stored_learned_summaries(),
        "provenance": {
            "git": git_provenance(),
            "source_file_sha256": source_hashes,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "historical_note": "Existing benchmark results and manifests were read only; no historical file was rewritten.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(OUTPUT_DIR / "ekf_baseline_results.json", to_json(payload))
    print(f"EKF evaluation written to {OUTPUT_DIR}")
    print(json.dumps({name: value["metrics"] for name, value in metric_results.items()}, indent=2))
    print("Legacy reproduction checks:", json.dumps(historical_comparison, sort_keys=True))


def expected_test_names() -> tuple[str, ...]:
    return ("true_state", "observations", "initial_state", "process_noise", "measurement_noise")


if __name__ == "__main__":
    main()
