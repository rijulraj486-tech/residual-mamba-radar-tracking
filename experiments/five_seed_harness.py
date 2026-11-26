#!/usr/bin/env python3
"""External, guarded harness for the fixed-pool n1024 radar experiment.

This file intentionally lives outside ``src/``. Importing it does not import
PyTorch, construct models, read datasets, or create output directories.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import timeq
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

OUTPUT_ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
TEST_DATA = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"
EXPECTED_TEST_SHA256 = "d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2"
REFERENCE_DIR = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/n1024"
TRAIN_SEEDS = (11, 22, 33, 44, 55)
DATA_SEED_DEFAULT = 20261126
SCALE = 1000.0
EXPECTED_MAMBA_PARAMS = 619014
EXPECTED_KNET_PARAMS = 107656
EVAL_ATOL = 1e-5
EVAL_RTOL = 5e-4
T_CRIT_DF4 = 2.7764451051977987

SOURCE_FILES = (
    "src/radar_tracking/config.py",
    "src/radar_tracking/data/data_generation.py",
    "src/radar_tracking/dynamics/constant_acceleration.py",
    "src/radar_tracking/models/bimamba_encoder.py",
    "src/radar_tracking/models/ekf.py",
    "src/radar_tracking/models/kalman_net.py",
    "src/radar_tracking/models/prediction_head.py",
    "src/radar_tracking/models/tracker_model.py",
    "src/radar_tracking/training/train_validate_test.py",
    "src/radar_tracking/cli/benchmark_mamba.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    hashes = {name: sha256_file(PROJECT / name) for name in SOURCE_FILES}
    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = "available"
    except (OSError, subprocess.CalledProcessError):
        revision = None
        status = "unavailable: project path has no accessible Git revision"
    return {"git_revision": revision, "git_revision_status": status, "source_file_sha256": hashes}


def json_write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def torch_save_new(path: Path, payload: Any) -> None:
    import torch

    # Exclusive creation prevents accidental replacement of any existing file.
    with path.open("xb") as stream:
        torch.save(payload, stream)


def verify_test_hash() -> str:
    if not TEST_DATA.is_file():
        raise FileNotFoundError(f"Shared test dataset is missing: {TEST_DATA}")
    actual = sha256_file(TEST_DATA)
    if actual != EXPECTED_TEST_SHA256:
        raise RuntimeError(
            f"Shared test SHA-256 mismatch: expected {EXPECTED_TEST_SHA256}, got {actual}"
        )
    return actual


def project_imports():
    """Lazy imports: called only by an explicit command, never at module import."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from radar_tracking.cli import benchmark_mamba as benchmark
    from radar_tracking.config import simparams_class, validate_simparams
    from radar_tracking.data.data_generation import generate_synthetic_data
    from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
    from radar_tracking.models.kalman_net import KalmanNetNN
    from radar_tracking.models.tracker_model import BiMambaTrackerModel
    from radar_tracking.training.train_validate_test import evaluate_loss_knet, train_validate_knet

    return {
        "np": np,
        "torch": torch,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
        "benchmark": benchmark,
        "simparams_class": simparams_class,
        "validate_simparams": validate_simparams,
        "generate_synthetic_data": generate_synthetic_data,
        "ConstantAcceleration2DModel": ConstantAcceleration2DModel,
        "KalmanNetNN": KalmanNetNN,
        "BiMambaTrackerModel": BiMambaTrackerModel,
        "evaluate_loss_knet": evaluate_loss_knet,
        "train_validate_knet": train_validate_knet,
    }


def make_simparams(api: dict[str, Any]):
    args = api["simparams_class"]()
    args.seq_len = 100
    args.dataset_size = 1024
    args.dt = 1
    args.q2_scale = 1
    args.r2_scale = 1
    args.meas_type = "RANGE_BEARING"
    args.init_type = "MEAS"
    args.noise_ref = "FIXED"
    args.range_std = 30
    args.bearing_std = 1
    args.n_batch = 64
    args.epochs = 1000
    args.learning_rate = 1e-3
    args.weight_decay = 1e-4
    args.stopping_criteria = 20
    args.update_thr_db = 0.01
    args.in_mult_KNet = 5
    args.out_mult_KNet = 40
    args.knet_rb_wrap_theta = True
    args.knet_rb_arc_length = True
    api["validate_simparams"](args)
    return args


def load_test_payload(api: dict[str, Any]) -> tuple[dict[str, Any], str]:
    digest = verify_test_hash()
    payload = api["torch"].load(TEST_DATA, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Shared test payload must be a mapping")
    expected = {
        "true_state": (1024, 6, 100),
        "observations": (1024, 2, 100),
        "initial_state": (1024, 6, 1),
    }
    for key, shape in expected.items():
        value = payload.get(key)
        if not api["torch"].is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"Shared test {key} must have shape {shape}")
        if not api["torch"].isfinite(value).all().item():
            raise ValueError(f"Shared test {key} has non-finite values")
    return payload, digest


def assert_fresh_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")


def check_source_unchanged(manifest: dict[str, Any]) -> None:
    expected = manifest.get("source_file_sha256", {})
    for relative, digest in expected.items():
        path = PROJECT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Source file changed since pool preparation: {relative}")


def pool_manifest_path() -> Path:
    return OUTPUT_ROOT / "fixed_pool_manifest.json"


def load_pool(api: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = pool_manifest_path()
    pool_path = OUTPUT_ROOT / "fixed_train_val.pt"
    if not manifest_path.is_file() or not pool_path.is_file():
        raise FileNotFoundError("Run prepare-pool first; fixed pool artifacts are absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_source_unchanged(manifest)
    if sha256_file(Path(__file__).resolve()) != manifest.get("harness_sha256"):
        raise RuntimeError("Harness changed since fixed-pool preparation")
    if verify_test_hash() != manifest["shared_test_sha256"]:
        raise RuntimeError("Shared test file changed since pool preparation")
    if sha256_file(pool_path) != manifest["fixed_pool_file_sha256"]:
        raise RuntimeError("Fixed pool file hash mismatch")
    payload = api["torch"].load(pool_path, map_location="cpu", weights_only=True)
    expected_counts = {"train": 819, "validation": 204}
    for split in ("train", "validation"):
        for key in ("state", "observations", "initial_state"):
            tensor = payload[split][key]
            expected_shapes = {
                "state": (expected_counts[split], 6, 100),
                "observations": (expected_counts[split], 2, 100),
                "initial_state": (expected_counts[split], 6, 1),
            }
            if tuple(tensor.shape) != expected_shapes[key]:
                raise ValueError(f"Unexpected fixed pool shape: {split}/{key}")
            expected_hash = manifest["tensor_sha256"][split][key]
            if tensor_sha256(tensor) != expected_hash:
                raise RuntimeError(f"Fixed pool tensor hash mismatch: {split}/{key}")
            if not api["torch"].isfinite(tensor).all().item():
                raise ValueError(f"Non-finite values in fixed pool tensor {split}/{key}")
    index_groups = payload["indices"]
    expected_index_counts = {"train": 819, "validation": 204, "unused": 1}
    manifest_index_keys = {
        "train": "train_indices",
        "validation": "validation_indices",
        "unused": "unused_indices",
    }
    for split, expected_count in expected_index_counts.items():
        indices = index_groups.get(split)
        if not isinstance(indices, list) or len(indices) != expected_count:
            raise RuntimeError(
                f"Saved {split} index count must be {expected_count}"
            )
        if indices != manifest["split"].get(manifest_index_keys[split]):
            raise RuntimeError(f"Saved {split} indices differ from the manifest")
    all_indices = (
        index_groups["train"] + index_groups["validation"] + index_groups["unused"]
    )
    if sorted(all_indices) != list(range(1024)):
        raise RuntimeError("Saved train/validation/unused indices are not a partition")
    return payload, manifest


def prepare_pool(data_seed: int) -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output root already exists; refusing resume or overwrite: {OUTPUT_ROOT}"
        )
    api = project_imports()
    test_digest = verify_test_hash()
    test_payload = api["torch"].load(TEST_DATA, map_location="cpu", weights_only=True)
    if test_payload.get("metadata", {}).get("sample_count") != 1024:
        raise ValueError("Shared test metadata does not report 1,024 samples")

    args = make_simparams(api)
    ca_model = api["ConstantAcceleration2DModel"](args)
    benchmark = api["benchmark"]

    # The generator uses Torch RNGs; split_data uses Python random. The existing
    # generator has no Python random draws, so resetting once reproduces its
    # prepare_train_val_data path while preserving indices locally.
    benchmark.set_seed(data_seed)
    x, y, init, _, _ = api["generate_synthetic_data"](ca_model, args, snr_dB=20)
    x, y, init, _, _ = benchmark.scale_dataset(
        x,
        y,
        init,
        api["torch"].eye(ca_model.m),
        api["torch"].eye(ca_model.n),
    )
    x, y, init = x.cpu(), y.cpu(), init.cpu()
    n = x.shape[0]
    indices = list(range(n))
    random.shuffle(indices)
    n_train, n_val = int(n * 0.8), int(n * 0.2)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    unused_idx = indices[n_train + n_val:]

    def select(rows: list[int]) -> dict[str, Any]:
        idx = api["torch"].as_tensor(rows, dtype=api["torch"].long)
        return {"state": x[idx], "observations": y[idx], "initial_state": init[idx]}

    payload = {
        "format_version": 1,
        "train": select(train_idx),
        "validation": select(val_idx),
        "indices": {"train": train_idx, "validation": val_idx, "unused": unused_idx},
    }
    tensor_hashes = {
        split: {key: tensor_sha256(value) for key, value in payload[split].items()}
        for split in ("train", "validation")
    }
    provenance = source_provenance()
    metadata = {
        "format_version": 1,
        "data_seed": int(data_seed),
        "pool_size": int(n),
        "sequence_length": int(args.seq_len),
        "split": {
            "method": "source-equivalent Python random.shuffle; floor(0.8N), floor(0.2N)",
            "train_count": len(train_idx),
            "validation_count": len(val_idx),
            "unused_count": len(unused_idx),
            "train_indices": train_idx,
            "validation_indices": val_idx,
            "unused_indices": unused_idx,
        },
        "tensor_shapes": {
            split: {key: list(value.shape) for key, value in payload[split].items()}
            for split in ("train", "validation")
        },
        "tensor_sha256": tensor_hashes,
        "shared_test_path": str(TEST_DATA),
        "shared_test_sha256": test_digest,
        "shared_test_metadata": test_payload.get("metadata", {}),
        "simulation_config": benchmark.simulation_config(args),
        "state_order": ["x", "vx", "ax", "y", "vy", "ay"],
        "scale": SCALE,
        "source_file_sha256": provenance["source_file_sha256"],
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "git_revision": provenance["git_revision"],
        "git_revision_status": provenance["git_revision_status"],
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    torch_save_new(OUTPUT_ROOT / "fixed_train_val.pt", payload)
    metadata["fixed_pool_file_sha256"] = sha256_file(OUTPUT_ROOT / "fixed_train_val.pt")
    json_write_new(pool_manifest_path(), metadata)
    print(f"Fixed pool created at {OUTPUT_ROOT}")
    print(f"Train/validation/unused: {len(train_idx)}/{len(val_idx)}/{len(unused_idx)}")
    print(f"Shared test SHA-256: {test_digest}")


def fixed_mamba_config(benchmark: Any, seed: int):
    return benchmark.MambaTrainConfig(
        seed=seed,
        epochs=180,
        batch_size=16,
        eval_batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=25,
        min_delta=1e-6,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.08,
        encoder_direction="forward",
        checkpoint_name="best_mamba_tracker.pt",
    )


def set_knet_config(args: Any) -> None:
    args.n_batch = 64
    args.epochs = 1000
    args.learning_rate = 1e-3
    args.weight_decay = 1e-4
    args.stopping_criteria = 20
    args.update_thr_db = 0.01
    args.in_mult_KNet = 5
    args.out_mult_KNet = 40
    args.knet_rb_wrap_theta = True
    args.knet_rb_arc_length = True


def capture_call(function, log_path: Path):
    class Tee:
        def __init__(self, stream, path):
            self.stream = stream
            self.path = path.open("x", encoding="utf-8", buffering=1)
            self.buffer = io.StringIO()

        def write(self, value):
            self.stream.write(value)
            self.path.write(value)
            self.buffer.write(value)
            return len(value)

        def flush(self):
            self.stream.flush()
            self.path.flush()

        def isatty(self):
            return self.stream.isatty()

        def close(self):
            self.path.close()

    import sys

    tee = Tee(sys.stdout, log_path)
    try:
        with contextlib.redirect_stdout(tee):
            result = function()
    finally:
        tee.flush()
        tee.close()
    return result, tee.buffer.getvalue()


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def parse_knet_log(log: str) -> dict[str, Any]:
    epoch_rows = []
    pattern = re.compile(r"Epoch\s+(\d+), Train Loss\(dB\):\s*([^,]+), Val Loss\(dB\):\s*([^\s]+)")
    for match in pattern.finditer(log):
        epoch_rows.append({
            "epoch": int(match.group(1)),
            "train_loss_db": finite_or_none(float(match.group(2))),
            "validation_loss_db": finite_or_none(float(match.group(3))),
        })
    best = re.search(r"Best Validation Loss\(dB\):\s*([^\s]+)\s+at Epoch\s+(\d+)", log)
    if best is None:
        raise RuntimeError("KalmanNet log did not report a selected validation checkpoint")
    return {
        "history": epoch_rows,
        "actual_epochs": len(epoch_rows),
        "best_validation_loss_db": finite_or_none(float(best.group(1))),
        "best_epoch": int(best.group(2)),
    }


def load_checkpoint_weights(torch: Any, model: Any, checkpoint: Path, device: Any) -> None:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)


def predict_mamba(api: dict[str, Any], model: Any, truth: Any, obs: Any, init: Any,
                  device: Any, chunk_size: int):
    _, prediction = api["benchmark"].evaluate_mamba(
        model,
        x_pool=truth.to(device),
        y_pool=obs.to(device),
        init_state=init.to(device),
        batch_size=chunk_size,
    )
    return prediction.detach().cpu()


def predict_knet(api: dict[str, Any], model: Any, args: Any, truth: Any, obs: Any,
                 init: Any, device: Any, chunk_size: int):
    predictions = []
    for start in range(0, truth.shape[0], chunk_size):
        end = min(start + chunk_size, truth.shape[0])
        _, estimate, _, _ = api["evaluate_loss_knet"](
            model,
            truth[start:end].to(device),
            obs[start:end].to(device),
            init[start:end].to(device),
            args,
        )
        predictions.append(estimate.detach().cpu())
    return api["torch"].cat(predictions, dim=0)


def calculate_metrics(torch: Any, truth: Any, estimate: Any, dt: float) -> dict[str, float]:
    if tuple(truth.shape) != tuple(estimate.shape):
        raise ValueError(f"Prediction shape {tuple(estimate.shape)} != truth {tuple(truth.shape)}")
    if not torch.isfinite(estimate).all().item():
        raise ValueError("Predictions contain non-finite values")
    error = estimate - truth
    mse = error.square().mean()

    def vector_rmse(i: int, j: int, time_scale: float) -> float:
        pair = error[:, [i, j], :]
        return float(torch.sqrt(pair.square().sum(dim=1).mean()) * SCALE / time_scale)

    mse_value = float(mse)
    return {
        "position_rmse_m": vector_rmse(0, 3, 1.0),
        "velocity_rmse_mps": vector_rmse(1, 4, dt),
        "acceleration_rmse_mps2": vector_rmse(2, 5, dt * dt),
        "six_state_mse_scaled_units": mse_value,
        "six_state_mse_db": 10.0 * math.log10(mse_value) if mse_value > 0 else None,
    }


def evaluate_chunk_consistency(api: dict[str, Any], mamba: Any, knet: Any, args: Any,
                               test: dict[str, Any], device: Any) -> dict[str, float]:
    torch = api["torch"]
    truth = test["true_state"][:8]
    obs = test["observations"][:8]
    init = test["initial_state"][:8]
    m_full = predict_mamba(api, mamba, truth, obs, init, device, 8)
    m_chunk = predict_mamba(api, mamba, truth, obs, init, device, 3)
    k_full = predict_knet(api, knet, args, truth, obs, init, device, 8)
    k_chunk = predict_knet(api, knet, args, truth, obs, init, device, 3)
    if not torch.allclose(m_full, m_chunk, rtol=EVAL_RTOL, atol=EVAL_ATOL):
        raise RuntimeError("Mamba chunked and unchunked predictions disagree")
    if not torch.allclose(k_full, k_chunk, rtol=EVAL_RTOL, atol=EVAL_ATOL):
        raise RuntimeError("KalmanNet chunked and unchunked predictions disagree")
    return {
        "mamba_max_abs_delta": float((m_full - m_chunk).abs().max()),
        "knet_max_abs_delta": float((k_full - k_chunk).abs().max()),
        "subset_examples": 8,
        "chunk_size": 3,
    }


def make_models_from_reference(api: dict[str, Any], reference_metrics: dict[str, Any], device: Any):
    benchmark = api["benchmark"]
    simparams = make_simparams(api)
    ca_model = api["ConstantAcceleration2DModel"](simparams)
    config_data = dict(reference_metrics["mamba_train_config"])
    config_data["encoder_direction"] = "forward"
    config = benchmark.MambaTrainConfig(**config_data)
    benchmark.set_seed(0)
    mamba = benchmark.build_mamba_model(simparams, config, device)

    knet_cfg = reference_metrics["knet"]["train_config"]
    simparams.n_batch = int(knet_cfg["batch_size"])
    simparams.epochs = int(knet_cfg["epochs"])
    simparams.learning_rate = float(knet_cfg["learning_rate"])
    simparams.weight_decay = float(knet_cfg["weight_decay"])
    simparams.stopping_criteria = int(knet_cfg["stopping_criteria"])
    simparams.update_thr_db = float(knet_cfg["update_threshold_db"])
    simparams.knet_rb_wrap_theta = bool(knet_cfg["range_bearing_wrap_theta"])
    simparams.knet_rb_arc_length = bool(knet_cfg["range_bearing_arc_length"])
    benchmark.set_seed(1)
    knet = api["KalmanNetNN"]()
    knet.NNBuild(ca_model, simparams)
    return mamba, knet, simparams, ca_model, config


def verify_reference_evaluator(api: dict[str, Any], device: Any) -> dict[str, Any]:
    test, test_digest = load_test_payload(api)
    reference = json.loads((REFERENCE_DIR / "benchmark_metrics.json").read_text(encoding="utf-8"))
    if reference.get("fixed_test_dataset", {}).get("sha256") != test_digest:
        raise RuntimeError("Reference n1024 metrics do not point to the verified test set")
    mamba, knet, args, _, _ = make_models_from_reference(api, reference, device)
    load_checkpoint_weights(api["torch"], mamba, REFERENCE_DIR / "best_mamba_tracker.pt", device)
    load_checkpoint_weights(api["torch"], knet, REFERENCE_DIR / "best_knet_seq100.pt", device)

    chunk_check = evaluate_chunk_consistency(api, mamba, knet, args, test, device)
    m_pred = predict_mamba(api, mamba, test["true_state"], test["observations"],
                           test["initial_state"], device, 16)
    k_pred = predict_knet(api, knet, args, test["true_state"], test["observations"],
                          test["initial_state"], device, 64)
    expected = {
        "mamba": reference["mamba"],
        "knet": reference["knet"],
    }
    observed = {
        "mamba": calculate_metrics(api["torch"], test["true_state"], m_pred, args.dt),
        "knet": calculate_metrics(api["torch"], test["true_state"], k_pred, args.dt),
    }
    for method in ("mamba", "knet"):
        for metric, reference_key in (
            ("position_rmse_m", "position_rmse_m"),
            ("six_state_mse_scaled_units", "aggregate_six_state_mse"),
        ):
            actual = observed[method][metric]
            expected_value = float(expected[method][reference_key])
            if not math.isclose(actual, expected_value, rel_tol=EVAL_RTOL, abs_tol=EVAL_ATOL):
                raise RuntimeError(
                    f"Reference {method} {metric} mismatch: {actual} vs {expected_value}"
                )
    return {
        "shared_test_sha256": test_digest,
        "chunk_consistency": chunk_check,
        "reference_metrics": {
            method: {
                "position_rmse_m": expected[method]["position_rmse_m"],
                "six_state_mse_scaled_units": expected[method]["aggregate_six_state_mse"],
            }
            for method in expected
        },
        "recomputed_metrics": observed,
        "tolerances": {"rtol": EVAL_RTOL, "atol": EVAL_ATOL},
        "status": "passed",
    }


def count_adam_steps(api: dict[str, Any], function):
    """Count actual Adam.step calls with a transparent temporary wrapper."""
    torch = api["torch"]
    original = torch.optim.Adam.step
    counter = {"count": 0}

    def wrapped(optimizer, *args, **kwargs):
        counter["count"] += 1
        return original(optimizer, *args, **kwargs)

    torch.optim.Adam.step = wrapped
    try:
        result = function()
    finally:
        torch.optim.Adam.step = original
    return result, counter["count"]


def synchronize_cuda(api: dict[str, Any], device: Any) -> None:
    """Finish queued CUDA work before measuring elapsed training time."""
    if getattr(device, "type", None) == "cuda":
        api["torch"].cuda.synchronize(device)


def validate_seed11_pilot() -> None:
    """Require a passed evaluator receipt and intact seed-11 checkpoints."""
    pilot_dir = OUTPUT_ROOT / f"seed_{TRAIN_SEEDS[0]}"
    receipt_path = pilot_dir / "baseline_evaluator_validation.json"
    run_path = pilot_dir / "run_metrics.json"
    if not receipt_path.is_file() or not run_path.is_file():
        raise RuntimeError("Complete and validate the seed-11 pilot before other seeds")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "passed"
        or receipt.get("shared_test_sha256") != EXPECTED_TEST_SHA256
        or run.get("seed") != TRAIN_SEEDS[0]
        or run.get("shared_test_sha256") != EXPECTED_TEST_SHA256
    ):
        raise RuntimeError("Seed-11 pilot validation receipt or run record is invalid")

    expected_names = {
        "mamba": "best_mamba_tracker.pt",
        "knet": "best_knet_seq100.pt",
    }
    for method, expected_name in expected_names.items():
        record = run.get(method, {})
        checkpoint_name = record.get("checkpoint")
        checkpoint_hash = record.get("checkpoint_sha256")
        if checkpoint_name != expected_name or not checkpoint_hash:
            raise RuntimeError(f"Seed-11 {method} checkpoint record is invalid")
        checkpoint = pilot_dir / expected_name
        if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError(f"Seed-11 {method} checkpoint hash mismatch: {checkpoint}")


def run_seed(seed: int) -> None:
    """Train one seed, then compute and store its held-out test metrics."""
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"Seed must be one of {TRAIN_SEEDS}")
    api = project_imports()
    pool, manifest = load_pool(api)
    run_dir = OUTPUT_ROOT / f"seed_{seed}"
    assert_fresh_path(run_dir)
    test_payload, test_digest = load_test_payload(api)
    device = api["benchmark"].require_cuda()
    benchmark = api["benchmark"]
    args = make_simparams(api)
    ca_model = api["ConstantAcceleration2DModel"](args)

    run_dir.mkdir(parents=False, exist_ok=False)
    baseline_validation = None
    if seed == TRAIN_SEEDS[0]:
        # Required calibration happens before any new-model training. Its scores
        # are only a metric-pipeline check and are never used for selection.
        baseline_validation = verify_reference_evaluator(api, device)
        json_write_new(run_dir / "baseline_evaluator_validation.json", baseline_validation)
    else:
        # This gate runs before constructing either new seed model or training it.
        validate_seed11_pilot()

    train, val = pool["train"], pool["validation"]
    x_train, y_train, init_train = train["state"], train["observations"], train["initial_state"]
    x_val, y_val, init_val = val["state"], val["observations"], val["initial_state"]
    config = fixed_mamba_config(benchmark, seed)
    train_dataset = api["TensorDataset"](y_train, init_train, x_train)
    train_loader = api["DataLoader"](
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_tensors = {"obs": y_val, "init": init_val, "state": x_val}
    mamba_seed, knet_seed = seed, 100000 + seed

    benchmark.set_seed(mamba_seed)
    mamba = benchmark.build_mamba_model(args, config, device)
    mamba_path = run_dir / "best_mamba_tracker.pt"
    synchronize_cuda(api, device)
    started = time.perf_counter()
    (mamba_training_result, mamba_steps) = count_adam_steps(
        api,
        lambda: capture_call(
            lambda: benchmark.train_mamba(
                mamba, train_loader, val_tensors, config, device, str(mamba_path)
            ),
            run_dir / "mamba_training.log",
        ),
    )
    synchronize_cuda(api, device)
    mamba_seconds = time.perf_counter() - started
    (mamba_result, mamba_log) = mamba_training_result
    mamba, mamba_history, mamba_best_epoch, mamba_best_val = mamba_result
    if not mamba_path.is_file():
        raise RuntimeError("Mamba trainer did not create its best checkpoint")

    set_knet_config(args)
    benchmark.set_seed(knet_seed)
    knet = api["KalmanNetNN"]()
    knet.NNBuild(ca_model, args)
    knet_path = run_dir / "best_knet_seq100.pt"
    synchronize_cuda(api, device)
    started = time.perf_counter()
    (knet_result, knet_steps) = count_adam_steps(
        api,
        lambda: capture_call(
            lambda: api["train_validate_knet"](
                ca_model,
                knet,
                args,
                x_train.to(device),
                y_train.to(device),
                x_val.to(device),
                y_val.to(device),
                init_train.to(device),
                init_val.to(device),
                str(knet_path),
            ),
            run_dir / "knet_training.log",
        ),
    )
    synchronize_cuda(api, device)
    knet_seconds = time.perf_counter() - started
    (knet_return, knet_log) = knet_result
    knet, knet_val_history = knet_return
    knet_selection = parse_knet_log(knet_log)
    if not knet_path.is_file():
        raise RuntimeError("KalmanNet trainer did not create its best checkpoint")

    # Reconstruct and load best checkpoints before the held-out test pass.
    benchmark.set_seed(mamba_seed)
    mamba_eval = benchmark.build_mamba_model(args, config, device)
    load_checkpoint_weights(api["torch"], mamba_eval, mamba_path, device)
    benchmark.set_seed(knet_seed)
    knet_eval = api["KalmanNetNN"]()
    knet_eval.NNBuild(ca_model, args)
    load_checkpoint_weights(api["torch"], knet_eval, knet_path, device)

    mamba_prediction = predict_mamba(
        api, mamba_eval, test_payload["true_state"], test_payload["observations"],
        test_payload["initial_state"], device, 16,
    )
    knet_prediction = predict_knet(
        api, knet_eval, args, test_payload["true_state"], test_payload["observations"],
        test_payload["initial_state"], device, 64,
    )
    if tuple(mamba_prediction.shape) != (1024, 6, 100):
        raise RuntimeError(f"Unexpected Mamba output shape: {tuple(mamba_prediction.shape)}")
    if tuple(knet_prediction.shape) != (1024, 6, 100):
        raise RuntimeError(f"Unexpected KalmanNet output shape: {tuple(knet_prediction.shape)}")
    metrics = {
        "mamba": calculate_metrics(api["torch"], test_payload["true_state"], mamba_prediction, args.dt),
        "knet": calculate_metrics(api["torch"], test_payload["true_state"], knet_prediction, args.dt),
    }
    if verify_test_hash() != test_digest:
        raise RuntimeError("Shared test dataset changed during the run")

    mamba_param_count = benchmark.count_parameters(mamba_eval)
    knet_param_count = sum(parameter.numel() for parameter in knet_eval.parameters())
    if mamba_param_count != EXPECTED_MAMBA_PARAMS or knet_param_count != EXPECTED_KNET_PARAMS:
        raise RuntimeError(
            f"Parameter-count mismatch: Mamba={mamba_param_count}, KalmanNet={knet_param_count}"
        )
    payload = {
        "seed": seed,
        "mamba_seed": mamba_seed,
        "knet_seed": knet_seed,
        "fixed_pool_file_sha256": manifest["fixed_pool_file_sha256"],
        "train_tensor_sha256": manifest["tensor_sha256"]["train"],
        "validation_tensor_sha256": manifest["tensor_sha256"]["validation"],
        "shared_test_sha256": test_digest,
        "mamba": {
            "effective_direction": mamba_eval.encoder.encoder_direction,
            "parameter_count": mamba_param_count,
            "config": vars(config),
            "actual_epochs": len(mamba_history["train_loss"]),
            "best_epoch": int(mamba_best_epoch),
            "best_validation_mse": float(mamba_best_val),
            "optimizer_steps": mamba_steps,
            "elapsed_seconds": mamba_seconds,
            "checkpoint": mamba_path.name,
            "checkpoint_sha256": sha256_file(mamba_path),
            "training_history": {
                key: [finite_or_none(float(value)) for value in values]
                for key, values in mamba_history.items()
            },
            "metrics": metrics["mamba"],
        },
        "knet": {
            "parameter_count": knet_param_count,
            "config": {
                "batch_size": args.n_batch,
                "epoch_cap": args.epochs,
                "learning_rate_initial": args.learning_rate,
                "weight_decay": args.weight_decay,
                "patience": args.stopping_criteria,
                "min_delta_db": args.update_thr_db,
                "tbptt_k": 2,
                "tbptt_w": 4,
            },
            "actual_epochs": knet_selection["actual_epochs"],
            "best_epoch": knet_selection["best_epoch"],
            "best_validation_loss_db": knet_selection["best_validation_loss_db"],
            "optimizer_steps": knet_steps,
            "elapsed_seconds": knet_seconds,
            "checkpoint": knet_path.name,
            "checkpoint_sha256": sha256_file(knet_path),
            "validation_history_db": [finite_or_none(float(v)) for v in knet_val_history],
            "parsed_history": knet_selection["history"],
            "metrics": metrics["knet"],
        },
        "baseline_evaluator_validation": baseline_validation,
        "source_file_sha256": manifest["source_file_sha256"],
        "harness_sha256": manifest["harness_sha256"],
    }
    json_write_new(run_dir / "run_metrics.json", payload)
    print(f"Seed {seed} complete: {run_dir}")


def summarize() -> None:
    """Aggregate the held-out metrics already stored by each run-seed command."""
    api = project_imports()
    # This verifies the actual pool file/tensors/indices, source hashes,
    # harness fingerprint and shared test hash before any tables are written.
    _, pool_manifest = load_pool(api)
    pilot_receipt = OUTPUT_ROOT / "seed_11" / "baseline_evaluator_validation.json"
    if not pilot_receipt.is_file():
        raise RuntimeError("Seed-11 baseline evaluator validation has not passed")
    receipt = json.loads(pilot_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("shared_test_sha256") != EXPECTED_TEST_SHA256:
        raise RuntimeError("Seed-11 evaluator validation receipt is invalid")

    rows = []
    for seed in TRAIN_SEEDS:
        run_path = OUTPUT_ROOT / f"seed_{seed}" / "run_metrics.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"Missing completed run: {run_path}")
        row = json.loads(run_path.read_text(encoding="utf-8"))
        if row["seed"] != seed or row["shared_test_sha256"] != EXPECTED_TEST_SHA256:
            raise RuntimeError(f"Run metadata mismatch for seed {seed}")
        if row["fixed_pool_file_sha256"] != pool_manifest["fixed_pool_file_sha256"]:
            raise RuntimeError(f"Fixed pool mismatch for seed {seed}")
        for method in ("mamba", "knet"):
            checkpoint = OUTPUT_ROOT / f"seed_{seed}" / row[method]["checkpoint"]
            if sha256_file(checkpoint) != row[method]["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
            if row[method]["parameter_count"] != (
                EXPECTED_MAMBA_PARAMS if method == "mamba" else EXPECTED_KNET_PARAMS
            ):
                raise RuntimeError(f"Parameter count mismatch for seed {seed}, {method}")
            if row[method]["metrics"].get("six_state_mse_db") is None:
                raise RuntimeError(f"Invalid six-state MSE dB for seed {seed}, {method}")
        if row["train_tensor_sha256"] != pool_manifest["tensor_sha256"]["train"]:
            raise RuntimeError(f"Train tensor hash mismatch for seed {seed}")
        if row["validation_tensor_sha256"] != pool_manifest["tensor_sha256"]["validation"]:
            raise RuntimeError(f"Validation tensor hash mismatch for seed {seed}")
        if row["mamba"]["effective_direction"] != "forward":
            raise RuntimeError(f"Mamba direction mismatch for seed {seed}")
        rows.append(row)

    summary_dir = OUTPUT_ROOT / "summary"
    assert_fresh_path(summary_dir)
    summary_dir.mkdir(parents=False, exist_ok=False)
    metric_names = (
        "position_rmse_m",
        "velocity_rmse_mps",
        "acceleration_rmse_mps2",
        "six_state_mse_scaled_units",
        "six_state_mse_db",
    )
    summaries = []
    paired = []
    per_seed_path = summary_dir / "per_seed_metrics.csv"
    with per_seed_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "method", *metric_names])
        for row in rows:
            for method in ("mamba", "knet"):
                values = row[method]["metrics"]
                writer.writerow([row["seed"], method, *[values[name] for name in metric_names]])

    for method in ("mamba", "knet"):
        for metric in metric_names:
            values = [float(row[method]["metrics"][metric]) for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            sd = math.sqrt(variance)
            half_width = T_CRIT_DF4 * sd / math.sqrt(len(values))
            summaries.append({
                "method": method,
                "metric": metric,
                "mean": mean,
                "sample_sd": sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            })
    for metric in metric_names:
        diffs = [
            float(row["mamba"]["metrics"][metric]) - float(row["knet"]["metrics"][metric])
            for row in rows
        ]
        mean = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((value - mean) ** 2 for value in diffs) / (len(diffs) - 1))
        half_width = T_CRIT_DF4 * sd / math.sqrt(len(diffs))
        paired.append({
            "metric": metric,
            "difference": "mamba_minus_knet",
            "mean": mean,
            "sample_sd": sd,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        })

    with (summary_dir / "across_seed_summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (summary_dir / "paired_differences.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    json_write_new(summary_dir / "summary.json", {
        "seeds": list(TRAIN_SEEDS),
        "shared_test_sha256": EXPECTED_TEST_SHA256,
        "fixed_pool_file_sha256": pool_manifest["fixed_pool_file_sha256"],
        "confidence_interval": "two-sided Student t, df=4, critical value 2.7764451051977987",
        "method_summaries": summaries,
        "paired_mamba_minus_knet": paired,
    })
    print(f"Summary tables written to {summary_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-pool")
    prepare.add_argument("--data-seed", type=int, default=DATA_SEED_DEFAULT)
    run = commands.add_parser("run-seed")
    run.add_argument("--seed", type=int, required=True, choices=TRAIN_SEEDS)
    commands.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare-pool":
        prepare_pool(args.data_seed)
    elif args.command == "run-seed":
        run_seed(args.seed)
    elif args.command == "summarize":
        summarize()


if __name__ == "__main__":
    main()
