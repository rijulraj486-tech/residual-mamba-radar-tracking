#!/usr/bin/env python3
"""Phase C: evaluate audited baselines on frozen Phase 3 T=200/T=400 data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[2]
LONG = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence"
OUTPUT = LONG / "stage8_baseline_comparison_v1"
STAGE6_DIR = LONG / "stage6_matched_length_test_evaluation_v1"
STAGE6_REPORT_PATH = STAGE6_DIR / "stage6_evaluation_report.json"
BASE = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
LENGTHS = (200, 400)
SEEDS = (11, 22, 33, 44, 55)
KNET_BATCH = 64
KNET_WARMUP_BATCH = 64
STATE_SCALE = 1000.0
METRICS = (
    "position_rmse_m", "velocity_rmse_mps", "acceleration_rmse_mps2", "six_state_mse_scaled_units"
)

sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))

import five_seed_harness as five_seed  # noqa: E402
from radar_tracking.config import simparams_class, validate_simparams  # noqa: E402
from radar_tracking.data.independent_dataset import tensor_sha256  # noqa: E402
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel  # noqa: E402
from radar_tracking.models.ekf import extended_kf  # noqa: E402
from radar_tracking.models.ekf_baselines import (  # noqa: E402
    calibrate_first_observation_prior,
    correlated_first_observation_update,
    estimate_qr_from_training,
)
from radar_tracking.models.kalman_net import KalmanNetNN  # noqa: E402
from radar_tracking.models.ukf import RadarUKF  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sidecar(path: Path) -> str:
    expected = path.with_name(path.name + ".sha256").read_text(encoding="ascii").split()[0]
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Checksum sidecar mismatch: {path}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sim_params(length: int, dataset_size: int = 1024):
    args = simparams_class()
    args.seq_len = length
    args.dataset_size = dataset_size
    args.dt = 1.0
    args.meas_type = "RANGE_BEARING"
    args.init_type = "MEAS"
    args.noise_ref = "FIXED"
    args.range_std = 30.0
    args.bearing_std = 1.0
    args.q2_scale = 1.0
    args.r2_scale = 1.0
    args.n_batch = KNET_BATCH
    args.epochs = 1000
    args.learning_rate = 1e-3
    args.weight_decay = 1e-4
    args.stopping_criteria = 20
    args.update_thr_db = 0.01
    args.in_mult_KNet = 5
    args.out_mult_KNet = 40
    args.knet_rb_wrap_theta = True
    args.knet_rb_arc_length = True
    args.use_cuda = torch.cuda.is_available()
    validate_simparams(args)
    return args


def load_split(bundle: Path, manifest: dict[str, Any], split_name: str) -> tuple[dict[str, Any], str]:
    record = manifest["splits"][split_name]
    path = bundle / record["file"]
    file_hash = sha(path)
    if file_hash != record["file_sha256"]:
        raise ValueError(f"{split_name} file hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mapping = {"state": "state", "observations": "observations", "initial_state": "initial_state", "initialization_observations": "initialization_observations"}
    for key, tensor_key in mapping.items():
        tensor = payload.get(tensor_key)
        expected = record["tensor_sha256"][key]
        if not torch.is_tensor(tensor) or tensor_sha256(tensor) != expected:
            raise ValueError(f"{split_name}/{key} tensor hash mismatch at T={manifest['sequence_length_observations']}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor {split_name}/{key}")
    if payload.get("trajectory_ids") != record["trajectory_ids"]:
        raise ValueError(f"{split_name} trajectory IDs differ from manifest")
    return payload, file_hash


def physical_covariances(manifest: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    sim = manifest["simulation"]
    q_physical = torch.tensor(sim["process_noise_covariance_physical"], dtype=torch.float64)
    r_physical = torch.tensor(sim["measurement_noise_covariance_physical"], dtype=torch.float64)
    # Stored state and range are physical values divided by 1,000; bearing remains radians.
    q_stored = q_physical / (STATE_SCALE**2)
    measurement_scale = torch.diag(torch.tensor([1.0 / STATE_SCALE, 1.0], dtype=torch.float64))
    r_stored = measurement_scale @ r_physical @ measurement_scale.T
    return q_stored, r_stored


def covariance_summary(p: torch.Tensor) -> dict[str, float]:
    value = p.detach().to(torch.float64)
    sym = 0.5 * (value + value.transpose(-1, -2))
    return {
        "max_abs_symmetry_error": float((value - value.transpose(-1, -2)).abs().max()),
        "minimum_symmetric_eigenvalue": float(torch.linalg.eigvalsh(sym).min()),
        "finite": bool(torch.isfinite(value).all()),
    }


def fit_filter_parameters(length: int, manifest: dict[str, Any], train: dict[str, Any], model):
    prior = calibrate_first_observation_prior(train["state"], train["observations"], train["initial_state"], model)
    estimate_q, estimate_r = estimate_qr_from_training(train["state"], train["observations"], model)
    oracle_q, oracle_r = physical_covariances(manifest)
    return {
        "prior": prior,
        "oracle_q": oracle_q,
        "oracle_r": oracle_r,
        "estimated_q": estimate_q.to(torch.float64),
        "estimated_r": estimate_r.to(torch.float64),
        "calibration_data_count": int(train["state"].shape[0]),
        "calibration_data_scope": "Stage 2 train split only",
    }


def run_ekf(
    model,
    args,
    observations: torch.Tensor,
    initial_state: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    *,
    prior: dict[str, Any] | None,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    q = q.to(dtype=torch.float32, device="cpu")
    r = r.to(dtype=torch.float32, device="cpu")
    filt = extended_kf(model, q, r, args)
    filt.reset(initial_state)
    if prior is None:
        covariance_policy = "legacy_P0 heuristic from current initializer and supplied R"
        cov = filt.P
    else:
        filt.x = model.f(initial_state)
        cov = prior["P0"].to(dtype=torch.float32).unsqueeze(0).expand(args.dataset_size, -1, -1).clone()
        filt.P = cov
        covariance_policy = "train-calibrated P0 at prior before first returned observation"
    assert filt.x.device.type == "cpu" and observations.device.type == "cpu"
    estimates = torch.empty((args.dataset_size, 6, observations.shape[2]), dtype=torch.float32)
    covariance_checks = []
    selected = {0, 1, 9, 49, observations.shape[2] - 1}
    started = time.perf_counter()
    with torch.inference_mode():
        for t in range(observations.shape[2]):
            if prior is not None and t == 0:
                correlated_first_observation_update(
                    filt, observations[:, :, t], prior["first_observation_state_noise_cross_covariance"]
                )
            else:
                filt.predict()
                filt.update(observations[:, :, t])
            estimates[:, :, t] = filt.x.squeeze(-1)
            if t in selected:
                covariance_checks.append({"step_1based": t + 1, **covariance_summary(filt.P)})
            if not torch.isfinite(filt.x).all() or not torch.isfinite(filt.P).all():
                raise FloatingPointError(f"EKF non-finite state/covariance at step {t + 1}")
    elapsed = time.perf_counter() - started
    return estimates, elapsed, {
        "device": "cpu", "batch_size": args.dataset_size, "warmup_runs": 0,
        "covariance_policy": covariance_policy,
        "covariance_snapshots": covariance_checks,
        "all_state_and_covariance_steps_finite": True,
        "numerical_warning": "Legacy EKF uses explicit innovation inverse and float32 covariance arithmetic.",
    }


def run_ukf(
    model,
    observations: torch.Tensor,
    initial_state: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    *,
    legacy_p0: torch.Tensor | None,
    prior: dict[str, Any] | None,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    obs = observations.to(dtype=torch.float64, device="cpu")
    ukf = RadarUKF(model, q.to(dtype=torch.float64, device="cpu"), r.to(dtype=torch.float64, device="cpu"))
    if prior is None:
        if legacy_p0 is None:
            raise ValueError("legacy P0 required for legacy UKF")
        mean = initial_state.to(torch.float64).squeeze(-1)
        covariance = legacy_p0.to(torch.float64)
        covariance_policy = "legacy_P0 heuristic from current initializer and supplied R"
        first_cross = None
    else:
        mean = (model.F.to(torch.float64) @ initial_state.to(torch.float64)).squeeze(-1)
        covariance = prior["P0"].to(torch.float64).unsqueeze(0).expand(initial_state.shape[0], -1, -1).clone()
        covariance_policy = "train-calibrated P0 at prior before first returned observation"
        first_cross = prior["first_observation_state_noise_cross_covariance"]
    estimates = torch.empty((obs.shape[0], 6, obs.shape[2]), dtype=torch.float32)
    covariance_checks = []
    selected = {0, 1, 9, 49, obs.shape[2] - 1}
    started = time.perf_counter()
    with torch.inference_mode():
        for t in range(obs.shape[2]):
            if t == 0 and first_cross is not None:
                mean, covariance = ukf.update(
                    mean, covariance, obs[:, :, t],
                    state_measurement_noise_cross_covariance=first_cross,
                )
            else:
                mean, covariance = ukf.predict(mean, covariance)
                mean, covariance = ukf.update(mean, covariance, obs[:, :, t])
            estimates[:, :, t] = mean.squeeze(-1).to(torch.float32)
            if t in selected:
                covariance_checks.append({"step_1based": t + 1, **covariance_summary(covariance)})
            if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
                raise FloatingPointError(f"UKF non-finite state/covariance at step {t + 1}")
    elapsed = time.perf_counter() - started
    return estimates, elapsed, {
        "device": "cpu", "batch_size": int(obs.shape[0]), "warmup_runs": 0,
        "covariance_dtype": "float64", "prediction_dtype": "float32",
        "alpha": ukf.alpha, "beta": ukf.beta, "kappa": ukf.kappa,
        "covariance_policy": covariance_policy,
        "covariance_snapshots": covariance_checks,
        "filter_final_diagnostics": ukf.last_diagnostics,
        "all_state_and_covariance_steps_finite": True,
    }


def build_knet_models(length: int, device: torch.device):
    args = sim_params(length)
    args.use_cuda = device.type == "cuda"
    dynamics = ConstantAcceleration2DModel(args)
    models, checkpoint_records = [], []
    root = BASE
    for seed in SEEDS:
        run_path = root / f"seed_{seed}/run_metrics.json"
        run = read_json(run_path)
        checkpoint = root / f"seed_{seed}/{run['knet']['checkpoint']}"
        digest = sha(checkpoint)
        if digest != run["knet"]["checkpoint_sha256"]:
            raise ValueError(f"KNet checkpoint hash mismatch: {checkpoint}")
        if run["source_file_sha256"]["src/radar_tracking/models/kalman_net.py"] != sha(PROJECT / "src/radar_tracking/models/kalman_net.py"):
            raise ValueError("Current KalmanNet source differs from checkpoint training source")
        model = KalmanNetNN()
        model.NNBuild(dynamics, args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        if sum(p.numel() for p in model.parameters()) != 107656:
            raise ValueError(f"Unexpected KNet parameter count for seed={seed}")
        models.append(model)
        checkpoint_records.append({
            "seed": seed,
            "training_seed": run["knet_seed"],
            "training_sequence_length": 100,
            "checkpoint_path": str(checkpoint.relative_to(PROJECT)),
            "checkpoint_sha256": digest,
            "run_record_path": str(run_path.relative_to(PROJECT)),
            "run_record_sha256": sha(run_path),
            "parameter_count": 107656,
            "strict_state_dict_load": True,
            "config": run["knet"]["config"],
        })
    return args, models, checkpoint_records


def infer_knet(model, observations: torch.Tensor, initial_state: torch.Tensor, args, device: torch.device, chunk: int):
    model.eval()
    predictions = []
    all_gain_norms, all_log_innov = [], []
    with torch.inference_mode():
        for start in range(0, observations.shape[0], chunk):
            end = min(start + chunk, observations.shape[0])
            obs = observations[start:end].to(device)
            init = initial_state[start:end].to(device)
            model.batch_size = end - start
            model.init_hidden_KNet()
            model.InitSequence(init, observations.shape[2])
            output = torch.empty((end - start, 6, observations.shape[2]), dtype=torch.float32, device=device)
            for t in range(observations.shape[2]):
                estimate, log_innov, _, gain = model(obs[:, :, t].unsqueeze(-1))
                output[:, :, t] = estimate.squeeze(-1)
                all_gain_norms.append(torch.linalg.vector_norm(gain.reshape(gain.shape[0], -1), dim=1).mean())
                all_log_innov.append(log_innov.mean())
            predictions.append(output.cpu())
    prediction = torch.cat(predictions, dim=0)
    gain_norms = torch.stack(all_gain_norms).to(torch.float64)
    log_innov = torch.stack(all_log_innov).to(torch.float64)
    stability = {
        "device": str(device), "chunk_batch_size": chunk, "warmup_runs": 1,
        "output_shape": list(prediction.shape),
        "finite_predictions": bool(torch.isfinite(prediction).all()),
        "prediction_abs_max_scaled_state": float(prediction.abs().max()),
        "predictions_at_or_above_state_clip_100_count": int((prediction.abs() >= 100.0 - 1e-5).sum()),
        "finite_gain_norms": bool(torch.isfinite(gain_norms).all()),
        "gain_norm_max": float(gain_norms.max()),
        "finite_log_innovation_diagnostics": bool(torch.isfinite(log_innov).all()),
        "log_innovation_abs_max": float(log_innov.abs().max()),
        "internal_sanitize_clamp_count": None,
        "internal_sanitize_note": "The existing model sanitizes/clamps internal values; this read-only evaluator records output bounds and returned gain/innovation diagnostics, not internal clamp-call counts.",
    }
    return prediction, stability


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return sha(path)


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def main() -> None:
    if not OUTPUT.is_dir():
        raise FileNotFoundError("Run the Stage 8 Phase A audit first")
    audit_provenance_path = OUTPUT / "provenance.json"
    audit_provenance = read_json(audit_provenance_path)
    if audit_provenance["phase_c_gate"] != "PASS_WITH_QUALIFICATIONS":
        raise RuntimeError("Phase C gate did not pass")
    if (OUTPUT / "stage8_evaluation_report.json").exists():
        raise FileExistsError("Refusing to overwrite Stage 8 evaluation results")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the archived KNet evaluation protocol")

    stage6_hash = verify_sidecar(STAGE6_REPORT_PATH)
    stage6 = read_json(STAGE6_REPORT_PATH)
    if stage6_hash != audit_provenance["stage6_report_sha256"]:
        raise ValueError("Stage 6 report changed after Phase A audit")
    for filename, expected in stage6["output_artifact_hashes"].items():
        if filename == "plots":
            for relative, digest in expected.items():
                if sha(PROJECT / relative) != digest:
                    raise ValueError(f"Stage 6 plot changed after Phase A audit: {relative}")
        elif sha(STAGE6_DIR / filename) != expected:
            raise ValueError(f"Stage 6 output changed after Phase A audit: {filename}")
    for relative, expected in stage6["source_hashes"].items():
        if sha(PROJECT / relative) != expected:
            raise ValueError(f"Stage 6 recorded source changed after Phase A audit: {relative}")
    for relative, expected in audit_provenance["source_hashes"].items():
        if sha(PROJECT / relative) != expected:
            raise ValueError(f"Baseline source changed after Phase A audit: {relative}")

    # First load only train splits and metadata. Fit/freeze all filter covariance
    # parameters before opening either test tensor.
    training_data, manifests, data_paths = {}, {}, {}
    filter_parameters, models = {}, {}
    filter_models = {}
    knet_models_by_length, knet_args_by_length, knet_checkpoint_provenance = {}, {}, {}
    stage2_path = LONG / "stage2_consolidated_validation.json"
    stage2_hash = verify_sidecar(stage2_path)
    for length in LENGTHS:
        bundle = LONG / f"T{length}_fixed_profile_v1"
        manifest_path = bundle / "dataset_manifest.json"
        manifest_hash = verify_sidecar(manifest_path)
        manifest = read_json(manifest_path)
        train, train_hash = load_split(bundle, manifest, "train")
        manifests[length] = {"manifest": manifest, "path": manifest_path, "sha256": manifest_hash}
        training_data[length] = train
        data_paths[length] = {"train_file_path": str((bundle / manifest["splits"]["train"]["file"]).relative_to(PROJECT)), "train_file_sha256": train_hash}
        args = sim_params(length)
        dynamics = ConstantAcceleration2DModel(args)
        filter_parameters[length] = fit_filter_parameters(length, manifest, train, dynamics)
        models[length] = dynamics
        knet_args, knet_models, knet_records = build_knet_models(length, torch.device("cuda:0"))
        knet_models_by_length[length] = knet_models
        knet_args_by_length[length] = knet_args
        knet_checkpoint_provenance[str(length)] = knet_records

    # Load and validate test files only after all checkpoint/source checks and
    # training-only calibration are complete. No policy choices depend on test.
    test_data = {}
    test_provenance = {}
    for length in LENGTHS:
        bundle = LONG / f"T{length}_fixed_profile_v1"
        manifest = manifests[length]["manifest"]
        test, test_hash = load_split(bundle, manifest, "test")
        stage6_data = stage6["dataset_integrity"]["lengths"][str(length)]
        test_ids_hash = hashlib.sha256("\n".join(test["trajectory_ids"]).encode()).hexdigest()
        if test_hash != stage6_data["test_file_sha256"] or test_ids_hash != stage6_data["trajectory_ids_sha256"]:
            raise ValueError(f"Test data differs from Stage 6 provenance at T={length}")
        test_data[length] = test
        test_provenance[str(length)] = {
            "test_file_path": str((bundle / manifest["splits"]["test"]["file"]).relative_to(PROJECT)),
            "test_file_sha256": test_hash,
            "bundle_manifest_sha256": manifests[length]["sha256"],
            "test_trajectory_count": len(test["trajectory_ids"]),
            "trajectory_ids_sha256": test_ids_hash,
            "test_tensor_sha256": {key: tensor_sha256(test[key]) for key in ("state", "observations", "initial_state", "initialization_observations")},
        }

    all_rows: list[dict[str, Any]] = []
    learned_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    timing_protocol = {
        "classical_filters": "single complete CPU inference pass, no warmup; wall clock around full split loop",
        "kalman_net": "one warmup pass on first up-to-64 Stage 2 train trajectories, then one timed complete test pass; CUDA synchronized before/after; chunks of 64",
        "M0_M1": "import previously measured Stage 6 full-horizon results; not rerun",
    }

    for length in LENGTHS:
        test = test_data[length]
        args = sim_params(length)
        model = models[length]
        params = filter_parameters[length]
        obs, init, truth = test["observations"], test["initial_state"], test["state"]

        # Frozen Stage 6 learned model results: copy references and the original
        # checkpoint/test hashes; never reload or modify M0/M1 weights here.
        for variant in ("M0", "M1"):
            for seed in SEEDS:
                row = next(r for r in stage6["per_seed_results"] if r["sequence_length"] == length and r["variant"] == variant and r["seed"] == seed)
                result = {
                    "model": variant, "method_group": "matched_length_learned", "variant": variant,
                    "sequence_length": length, "seed": seed, "training_sequence_length": length,
                    "status": row["status"], "metrics": row["metrics"], "runtime_seconds": row["timing"]["mean_full_split_inference_seconds"],
                    "runtime_device": row["timing"]["device"], "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_sha256": row["checkpoint_sha256"], "config_sha256": row["config_sha256"],
                    "test_file_sha256": row["test_file_sha256"], "numerical_stability": {"finite_predictions": True, "source": "verified Stage 6 result"},
                    "interpretation": "matched-length learned model; frozen Stage 6 evaluation",
                }
                evaluation_rows.append(result)
                learned_rows.append({"model": variant, "method_group": "matched_length_learned", "sequence_length": length, "seed": seed, "training_sequence_length": length, "status": row["status"], **row["metrics"], "runtime_seconds": result["runtime_seconds"], "runtime_device": result["runtime_device"], "checkpoint_sha256": row["checkpoint_sha256"], "test_file_sha256": row["test_file_sha256"]})
                all_rows.append(flatten_result(result))

        # Existing KNet five seeds. Warm up only on the train split; reset
        # recurrent hidden state on every batch and on the timed test pass.
        warm_count = min(KNET_WARMUP_BATCH, training_data[length]["observations"].shape[0])
        device = torch.device("cuda:0")
        for seed, knet_model, checkpoint_record in zip(SEEDS, knet_models_by_length[length], knet_checkpoint_provenance[str(length)]):
            _ = infer_knet(knet_model, training_data[length]["observations"][:warm_count], training_data[length]["initial_state"][:warm_count], knet_args_by_length[length], device, KNET_BATCH)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction, stability = infer_knet(knet_model, obs, init, knet_args_by_length[length], device, KNET_BATCH)
            torch.cuda.synchronize(device)
            runtime = time.perf_counter() - started
            if prediction.shape != truth.shape or not torch.isfinite(prediction).all():
                raise FloatingPointError(f"KalmanNet failed output checks at T={length}, seed={seed}")
            metrics = five_seed.calculate_metrics(torch, truth, prediction, args.dt)
            result = {
                "model": "KalmanNet", "method_group": "learned_cross_length_transfer", "variant": "T100_checkpoint_transfer",
                "sequence_length": length, "seed": seed, "training_sequence_length": 100,
                "status": "PASS", "metrics": metrics, "runtime_seconds": runtime,
                "runtime_device": str(device), "checkpoint_path": checkpoint_record["checkpoint_path"],
                "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                "config_sha256": None, "test_file_sha256": test_provenance[str(length)]["test_file_sha256"],
                "numerical_stability": stability,
                "interpretation": "T=100-trained learned, model-assisted cross-length transfer; not matched-length training",
            }
            evaluation_rows.append(result)
            learned_rows.append({"model": "KalmanNet", "method_group": result["method_group"], "sequence_length": length, "seed": seed, "training_sequence_length": 100, "status": "PASS", **metrics, "runtime_seconds": runtime, "runtime_device": str(device), "checkpoint_sha256": checkpoint_record["checkpoint_sha256"], "test_file_sha256": result["test_file_sha256"]})
            all_rows.append(flatten_result(result))

        # Get the legacy heuristic P0 for this exact initializer/noise setup.
        legacy_filter = extended_kf(model, params["oracle_q"].float(), params["oracle_r"].float(), args)
        legacy_filter.reset(init)
        legacy_p0 = legacy_filter.P.clone()
        policies = (
            ("legacy_P0_oracle_QR", params["oracle_q"], params["oracle_r"], None),
            ("train_calibrated_P0_oracle_QR", params["oracle_q"], params["oracle_r"], params["prior"]),
            ("train_calibrated_P0_train_estimated_QR", params["estimated_q"], params["estimated_r"], params["prior"]),
        )
        for name, q, r, prior in policies:
            for family, runner in (("EKF", run_ekf), ("UKF", run_ukf)):
                try:
                    if family == "EKF":
                        prediction, runtime, stability = runner(model, args, obs, init, q, r, prior=prior)
                    else:
                        prediction, runtime, stability = runner(model, obs, init, q, r, legacy_p0=legacy_p0, prior=prior)
                    if prediction.shape != truth.shape or not torch.isfinite(prediction).all():
                        raise FloatingPointError("prediction shape/finiteness check failed")
                    metrics = five_seed.calculate_metrics(torch, truth, prediction, args.dt)
                    result = {
                        "model": family, "method_group": "model_informed_classical", "variant": name,
                        "sequence_length": length, "seed": None, "training_sequence_length": None,
                        "status": "PASS", "metrics": metrics, "runtime_seconds": runtime,
                        "runtime_device": stability["device"], "checkpoint_path": None, "checkpoint_sha256": None,
                        "config_sha256": None, "test_file_sha256": test_provenance[str(length)]["test_file_sha256"],
                        "numerical_stability": stability,
                        "calibration_trajectory_count": params["calibration_data_count"],
                        "interpretation": "privileged model-informed filter; policy fully disclosed",
                    }
                except Exception as exc:
                    result = {
                        "model": family, "method_group": "model_informed_classical", "variant": name,
                        "sequence_length": length, "seed": None, "training_sequence_length": None,
                        "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                        "test_file_sha256": test_provenance[str(length)]["test_file_sha256"],
                    }
                evaluation_rows.append(result)
                if result["status"] == "PASS":
                    filter_rows.append({"model": family, "variant": name, "method_group": result["method_group"], "sequence_length": length, "status": "PASS", **result["metrics"], "runtime_seconds": runtime, "runtime_device": stability["device"], "calibration_trajectory_count": params["calibration_data_count"], "test_file_sha256": result["test_file_sha256"]})
                    all_rows.append(flatten_result(result))
                else:
                    filter_rows.append({"model": family, "variant": name, "method_group": result["method_group"], "sequence_length": length, "status": "FAIL", "error": result["error"], "test_file_sha256": result["test_file_sha256"]})
                    all_rows.append(flatten_result(result))

    # Assert frozen test files did not change during the evaluation.
    for length in LENGTHS:
        record = test_provenance[str(length)]
        if sha(PROJECT / record["test_file_path"]) != record["test_file_sha256"]:
            raise RuntimeError(f"Frozen test file changed during evaluation at T={length}")

    summary_rows = []
    for length in LENGTHS:
        for model_name in ("M0", "M1", "KalmanNet"):
            rows = [r for r in evaluation_rows if r["sequence_length"] == length and r["model"] == model_name and r["status"] == "PASS"]
            for metric in METRICS:
                mean, sd = mean_sd([r["metrics"][metric] for r in rows])
                summary_rows.append({"model": model_name, "sequence_length": length, "metric": metric, "mean": mean, "sample_sd": sd, "seed_count": len(rows), "interpretation": "checkpoint seed variability; KNet remains T100-trained" if model_name == "KalmanNet" else "matched-length checkpoint seed variability"})

    csv_hashes = {
        "combined_metrics.csv": write_csv(OUTPUT / "combined_metrics.csv", all_rows),
        "learned_per_seed_metrics.csv": write_csv(OUTPUT / "learned_per_seed_metrics.csv", learned_rows),
        "classical_filter_metrics.csv": write_csv(OUTPUT / "classical_filter_metrics.csv", filter_rows),
        "learned_seed_aggregates.csv": write_csv(OUTPUT / "learned_seed_aggregates.csv", summary_rows),
    }
    failed = [r for r in evaluation_rows if r["status"] != "PASS"]
    report = {
        "stage": "Stage 8 baseline audit and long-sequence evaluation",
        "status": "PASS" if not failed else "PARTIAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase_a_gate": audit_provenance["phase_c_gate"],
        "scope": {"training": False, "fine_tuning": False, "test_tuning": False, "dataset_regeneration": False, "m0_m1_checkpoint_changes": False},
        "evaluation_matrix": {"lengths": list(LENGTHS), "test_trajectories_per_length": 1024, "classical_filter_policies_per_family_per_length": 3, "knet_seed_count_per_length": 5, "m0_m1_seed_count_per_length": 5},
        "metric_definitions": {
            "position_rmse_m": "joint x/y RMSE across trajectories/time after multiplying scaled state errors by 1,000",
            "velocity_rmse_mps": "joint vx/vy RMSE across trajectories/time after multiplying scaled state errors by 1,000",
            "acceleration_rmse_mps2": "joint ax/ay RMSE across trajectories/time after multiplying scaled state errors by 1,000",
            "six_state_mse_scaled_units": "mean squared error across batch, six stored state channels, and time",
        },
        "fairness_groups": {"matched_length_learned": ["M0", "M1"], "learned_cross_length_transfer": ["KalmanNet T100 checkpoints"], "model_informed_classical": ["EKF", "UKF"]},
        "timing_protocol": timing_protocol,
        "dataset_provenance": test_provenance,
        "training_calibration_provenance": {str(length): {
            **data_paths[length],
            "train_trajectory_count": params["calibration_data_count"],
            "calibration_scope": params["calibration_data_scope"],
            "oracle_Q_sha256": tensor_sha256(params["oracle_q"]),
            "oracle_R_sha256": tensor_sha256(params["oracle_r"]),
            "estimated_Q_sha256": tensor_sha256(params["estimated_q"]),
            "estimated_R_sha256": tensor_sha256(params["estimated_r"]),
            "calibrated_P0_sha256": tensor_sha256(params["prior"]["P0"]),
            "first_measurement_cross_covariance_sha256": tensor_sha256(params["prior"]["first_observation_state_noise_cross_covariance"]),
        } for length, params in filter_parameters.items()},
        "kalman_net_checkpoints": knet_checkpoint_provenance,
        "per_evaluation_results": evaluation_rows,
        "learned_seed_aggregates": summary_rows,
        "failures": failed,
        "no_seed_variability_for_deterministic_filters": True,
        "classical_filters_are_information_privileged": True,
        "kalman_net_is_t100_trained_transfer_not_matched_length": True,
        "output_csv_sha256": csv_hashes,
    }
    json_path = OUTPUT / "stage8_evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    json_hash = sha(json_path)
    json_path.with_name(json_path.name + ".sha256").write_text(f"{json_hash}  {json_path.name}\n", encoding="ascii")
    markdown = render_report(report)
    md_path = OUTPUT / "stage8_evaluation_report.md"
    md_path.write_text(markdown, encoding="utf-8")
    md_hash = sha(md_path)
    md_path.with_name(md_path.name + ".sha256").write_text(f"{md_hash}  {md_path.name}\n", encoding="ascii")
    provenance = {
        "stage": "Stage 8 Phase C evaluation provenance",
        "created_at_utc": report["created_at_utc"],
        "status": report["status"],
        "audit_provenance_sha256": sha(audit_provenance_path),
        "stage6_report_sha256": stage6_hash,
        "stage2_validation_report_sha256": stage2_hash,
        "evaluation_script_path": str(Path(__file__).resolve().relative_to(PROJECT)),
        "evaluation_script_sha256": sha(Path(__file__).resolve()),
        "source_hashes": {relative: sha(PROJECT / relative) for relative in (
            "src/radar_tracking/config.py", "src/radar_tracking/data/independent_dataset.py",
            "src/radar_tracking/dynamics/constant_acceleration.py", "src/radar_tracking/models/ekf.py",
            "src/radar_tracking/models/ekf_baselines.py", "src/radar_tracking/models/ukf.py",
            "src/radar_tracking/models/kalman_net.py", "src/radar_tracking/training/train_validate_test.py",
            "experiments/five_seed_harness.py", "experiments/evaluate_ekf_baselines.py",
            "experiments/evaluate_ukf_baselines.py", "experiments/paper_v4_long_sequence/stage6_matched_length_test_evaluation.py",
        )},
        "hardware": {"python_version": sys.version, "pytorch_version": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "platform": platform.platform()},
        "test_provenance": test_provenance,
        "training_calibration_provenance": report["training_calibration_provenance"],
        "output_hashes": {**csv_hashes, "stage8_evaluation_report.json": json_hash, "stage8_evaluation_report.md": md_hash},
        "failure_count": len(failed),
    }
    provenance_path = OUTPUT / "evaluation_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provenance_hash = sha(provenance_path)
    provenance_path.with_name(provenance_path.name + ".sha256").write_text(f"{provenance_hash}  {provenance_path.name}\n", encoding="ascii")
    print(json.dumps({"status": report["status"], "evaluation_count": len(evaluation_rows), "failures": len(failed), "evaluation_report_sha256": json_hash, "markdown_sha256": md_hash, "provenance_sha256": provenance_hash, "output_directory": str(OUTPUT)}, indent=2), flush=True)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row = {k: result.get(k) for k in ("model", "method_group", "variant", "sequence_length", "seed", "training_sequence_length", "status", "runtime_seconds", "runtime_device", "checkpoint_path", "checkpoint_sha256", "config_sha256", "test_file_sha256", "calibration_trajectory_count")}
    if result["status"] == "PASS":
        row.update(result["metrics"])
        stability = result.get("numerical_stability", {})
        row["finite_predictions"] = stability.get("finite_predictions", stability.get("all_state_and_covariance_steps_finite"))
        row["prediction_abs_max_scaled_state"] = stability.get("prediction_abs_max_scaled_state")
        row["predictions_at_or_above_state_clip_100_count"] = stability.get("predictions_at_or_above_state_clip_100_count")
        row["error"] = None
    else:
        row["error"] = result.get("error")
    return row


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 8: Baseline Audit and Long-Sequence Evaluation", "",
        f"**Status:** {report['status']} ({len(report['per_evaluation_results']) - len(report['failures'])}/{len(report['per_evaluation_results'])} evaluation rows passed)", "",
        "## Comparison groups", "",
        "M0/M1 rows are imported from the verified frozen Stage 6 report. KalmanNet rows are five existing T=100 checkpoints evaluated as cross-length transfer. EKF/UKF rows are deterministic model-informed filters; oracle-noise and train-estimated-noise policies are separately labeled. The classical filters use exact F/h and therefore are not information-equivalent to Mamba.", "",
        "## Full-horizon results", "",
        "Means ± sample SD for learned checkpoint seeds. Classical filters are deterministic single results and have no seed SD.", "",
        "| T | Model / policy | Seeds | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Runtime / device |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    lookup = {(r["sequence_length"], r["model"], r["metric"]): r for r in report["learned_seed_aggregates"]}
    for length in LENGTHS:
        for model_name in ("M0", "M1", "KalmanNet"):
            agg = [lookup[(length, model_name, metric)] for metric in METRICS]
            runtime_rows = [r for r in report["per_evaluation_results"] if r["sequence_length"] == length and r["model"] == model_name and r["status"] == "PASS"]
            runtime_mean = sum(r["runtime_seconds"] for r in runtime_rows) / len(runtime_rows)
            device = runtime_rows[0]["runtime_device"]
            label = "KalmanNet T100 transfer" if model_name == "KalmanNet" else model_name
            values = [f"{row['mean']:.6g} ± {row['sample_sd']:.3g}" for row in agg]
            lines.append(f"| {length} | {label} | {agg[0]['seed_count']} | " + " | ".join(values) + f" | {runtime_mean:.4g} s ({device}) |")
        for result in [r for r in report["per_evaluation_results"] if r["sequence_length"] == length and r["method_group"] == "model_informed_classical"]:
            if result["status"] != "PASS":
                lines.append(f"| {length} | {result['model']} {result['variant']} | 1 deterministic | FAILED: {result.get('error','unknown')} | — | — | — | — |")
                continue
            metrics = result["metrics"]
            vals = [f"{metrics[m]:.6g}" for m in METRICS]
            lines.append(f"| {length} | {result['model']} {result['variant']} | 1 deterministic | " + " | ".join(vals) + f" | {result['runtime_seconds']:.4g} s ({result['runtime_device']}) |")
    lines.extend(["", "## Learned per-seed records", "", "See `learned_per_seed_metrics.csv` for all Stage 6 M0/M1 and Stage 8 KalmanNet seed values and checkpoint hashes. KalmanNet seed IDs are its training checkpoints; they are not paired inferentially with Stage 5B M0/M1 seeds.", "", "## Numerical stability and failures", ""])
    if report["failures"]:
        lines.append(f"{len(report['failures'])} evaluation row(s) failed; see the JSON `failures` list and `classical_filter_metrics.csv` for details.")
    else:
        lines.append("No evaluation rows failed. EKF and UKF recorded per-step finite state/covariance checks and covariance snapshots. KalmanNet outputs, gain norms, and log-innovation diagnostics were checked for finiteness; state clipping-bound hits were counted. KalmanNet's internal sanitize/clamp calls were not instrumented, so internal saturation counts are unavailable.")
    lines.extend(["", "## Runtime", "", report["timing_protocol"]["classical_filters"] + ". " + report["timing_protocol"]["kalman_net"] + ".", "", "EKF/UKF ran on CPU and KalmanNet on CUDA. These implementation/device runtimes are not directly comparable as algorithm throughput. M0/M1 timing is taken from Stage 6, not remeasured.", "", "## Interpretation limits", "", "The M0/M1 results are matched-length trained; KalmanNet was trained at T=100 and is reported only as cross-length transfer. Classical filters use exact F/h; oracle_QR also uses exact manifest noise covariances, while train-estimated_QR uses only Stage 2 train labels. All classical rows are privileged model-informed references. Their single deterministic scores are not five-seed distributions. No test result changed calibration, parameters, checkpoint selection, metric definitions, or reporting rules.", "", "The Stage 2 train/test split manifests, test-file hashes, tensor hashes, and trajectory IDs are recorded in JSON/provenance. Existing Stage 6 and Stage 7 artifacts were read-only.", "", "Detailed outputs: `stage8_evaluation_report.json`, `evaluation_provenance.json`, `combined_metrics.csv`, `learned_per_seed_metrics.csv`, `classical_filter_metrics.csv`, and `learned_seed_aggregates.csv`.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
