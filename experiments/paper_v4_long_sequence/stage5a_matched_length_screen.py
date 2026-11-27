#!/usr/bin/env python3
"""One-seed matched-length M0/M1 training using train/validation only."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.paper_v4_mamba_improvement.model_variants import (  # noqa: E402
    NORMALIZED_FEATURE_INDICES,
    TrainNormalizedFeatureTracker,
)
from radar_tracking.data.independent_dataset import (  # noqa: E402
    sha256_file,
    tensor_sha256,
    validate_split_tensors,
)
from radar_tracking.models.tracker_model import BiMambaTrackerModel  # noqa: E402

OUTPUT_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage5A_matched_length_screen_v1"
LENGTHS = (200, 400)
VARIANTS = ("M0", "M1")
METRICS = (
    "position_rmse_m",
    "velocity_rmse_mps",
    "acceleration_rmse_mps2",
    "six_state_mse_scaled_units",
)
STATE_SCALE = 1000.0
EXPECTED_BUNDLE_HASHES = {
    200: "e1e774aea741a97bde552c2ecd4e9878bd07eaff847ed38bf9aef6d6d3ccebfe",
    400: "1ebede2aa4e4a210b45af5ccd3e6b9b49952d117b5e7a7968f54ac4d81a904e7",
}
SOURCE_FILES = (
    "experiments/paper_v4_long_sequence/stage5a_matched_length_screen.py",
    "experiments/paper_v4_mamba_improvement/model_variants.py",
    "src/radar_tracking/cli/benchmark_mamba.py",
    "src/radar_tracking/models/tracker_model.py",
    "src/radar_tracking/models/bimamba_encoder.py",
    "src/radar_tracking/models/prediction_head.py",
    "src/radar_tracking/data/independent_dataset.py",
    "src/radar_tracking/data/data_generation.py",
    "src/radar_tracking/dynamics/constant_acceleration.py",
    "src/radar_tracking/config.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_sidecar(path: Path) -> str:
    expected = path.with_name(path.name + ".sha256").read_text(encoding="ascii").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")
    return actual


def write_json(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_values(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    error = prediction - truth
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("non-finite validation predictions")

    def group_rmse(first: int, second: int, time_slice: slice | None = None) -> float:
        selected = error[:, [first, second], :] if time_slice is None else error[:, [first, second], time_slice]
        return float((selected.square().sum(dim=1).mean().sqrt() * STATE_SCALE).item())

    mse = float(error.square().mean().item())
    if not math.isfinite(mse):
        raise FloatingPointError("non-finite validation MSE")
    windows = []
    for start in range(0, truth.shape[-1], 10):
        end = min(start + 10, truth.shape[-1])
        sl = slice(start, end)
        windows.append({
            "start_step_1based": start + 1,
            "end_step_1based": end,
            "position_rmse_m": group_rmse(0, 3, sl),
            "velocity_rmse_mps": group_rmse(1, 4, sl),
            "acceleration_rmse_mps2": group_rmse(2, 5, sl),
            "six_state_mse_scaled_units": float(error[:, :, sl].square().mean().item()),
        })
    return {
        "position_rmse_m": group_rmse(0, 3),
        "velocity_rmse_mps": group_rmse(1, 4),
        "acceleration_rmse_mps2": group_rmse(2, 5),
        "six_state_mse_scaled_units": mse,
        "six_state_mse_db": 10.0 * math.log10(mse) if mse > 0 else -math.inf,
        "temporal_windows": windows,
    }


def load_training_splits(length: int) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = PROJECT / f"artifacts/results/synthetic/paper_v4_long_sequence/T{length}_fixed_profile_v1"
    manifest_path = bundle / "dataset_manifest.json"
    bundle_hash = verify_sidecar(manifest_path)
    if bundle_hash != EXPECTED_BUNDLE_HASHES[length]:
        raise ValueError(f"T={length} frozen bundle-manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "generated_validated" or manifest["sequence_length_observations"] != length:
        raise ValueError(f"T={length} bundle manifest status/length invalid")

    payloads: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {}
    for split, count in (("train", 819), ("validation", 204)):
        split_manifest_path = bundle / "manifests" / f"{split}_manifest.json"
        split_manifest_hash = verify_sidecar(split_manifest_path)
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if split_manifest["parent_manifest_sha256"] != bundle_hash:
            raise ValueError(f"T={length} {split} manifest parent hash mismatch")
        if split_manifest["manifest_scope"] != "single_split":
            raise ValueError(f"T={length} {split} manifest scope mismatch")
        split_path = bundle / "splits" / f"{split}.pt"
        split_hash = sha256_file(split_path)
        record = manifest["splits"][split]
        if split_hash != record["file_sha256"]:
            raise ValueError(f"T={length} {split} split file hash mismatch")
        payload = torch.load(split_path, map_location="cpu", weights_only=True)
        if payload["split"] != split or payload["sequence_length"] != length:
            raise ValueError(f"T={length} {split} payload identity mismatch")
        if payload["generation_seed"] != record["generation_seed"]:
            raise ValueError(f"T={length} {split} generation seed mismatch")
        if payload["trajectory_ids"] != record["trajectory_ids"]:
            raise ValueError(f"T={length} {split} IDs differ from frozen manifest")
        if payload["trajectory_ids"] != split_manifest["splits"][split]["trajectory_ids"]:
            raise ValueError(f"T={length} {split} IDs differ from split manifest")
        if len(payload["trajectory_ids"]) != count:
            raise ValueError(f"T={length} {split} trajectory count mismatch")
        expected_shapes = {
            "state": [count, 6, length],
            "observations": [count, 2, length],
            "initial_state": [count, 6, 1],
            "initialization_observations": [count, 2, 2],
        }
        for name, shape in expected_shapes.items():
            if list(payload[name].shape) != shape or list(payload[name].shape) != record["tensor_shapes"][name]:
                raise ValueError(f"T={length} {split} {name} dimensions mismatch")
            if tensor_sha256(payload[name]) != record["tensor_sha256"][name]:
                raise ValueError(f"T={length} {split} {name} tensor hash mismatch")
        validate_split_tensors(
            state=payload["state"],
            observations=payload["observations"],
            initial_state=payload["initial_state"],
            initialization_observations=payload["initialization_observations"],
            trajectory_ids=payload["trajectory_ids"],
            sequence_length=length,
        )
        payloads[split] = payload
        records[split] = {
            "split_file": str(split_path.relative_to(PROJECT)),
            "split_file_sha256": split_hash,
            "split_manifest": str(split_manifest_path.relative_to(PROJECT)),
            "split_manifest_sha256": split_manifest_hash,
            "generation_seed": record["generation_seed"],
            "trajectory_count": count,
            "trajectory_ids_sha256": sha256_bytes("\n".join(payload["trajectory_ids"]).encode()),
            "tensor_sha256": record["tensor_sha256"],
            "tensor_shapes": expected_shapes,
            "finite_and_initialization_alignment": "PASS",
        }
    if set(payloads["train"]["trajectory_ids"]) & set(payloads["validation"]["trajectory_ids"]):
        raise ValueError(f"T={length} train and validation IDs overlap")
    return {
        "bundle_directory": str(bundle.relative_to(PROJECT)),
        "bundle_manifest": str(manifest_path.relative_to(PROJECT)),
        "bundle_manifest_sha256": bundle_hash,
        "train": payloads["train"],
        "validation": payloads["validation"],
        "split_records": records,
        "test_file_opened": False,
    }, {"length": length, "bundle_manifest_sha256": bundle_hash, "splits": records, "test_file_opened": False}


def make_kwargs(config: dict[str, Any], length: int) -> dict[str, Any]:
    model = config["model"]
    return {
        "state_dim": 6,
        "obs_dim": 2,
        "dt": 1.0,
        "embed_dim": model["embed_dim"],
        "hidden_dim": model["hidden_dim"],
        "num_layers": model["num_layers"],
        "d_state": model["d_state"],
        "d_conv": model["d_conv"],
        "expand": model["expand"],
        "dropout": model["dropout"],
        "encoder_direction": "forward",
        "task": "radar_benchmark",
    }


def fit_feature_statistics(train: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit M1 means/population stds from training split only, in bounded batches."""
    cpu_model = BiMambaTrackerModel(**make_kwargs(config, config["sequence_length"])).cpu().eval()
    selected = torch.tensor(NORMALIZED_FEATURE_INDICES, dtype=torch.long)
    count = 0
    sum_values = torch.zeros(len(NORMALIZED_FEATURE_INDICES), dtype=torch.float64)
    sum_squares = torch.zeros_like(sum_values)
    with torch.no_grad():
        for start in range(0, len(train["state"]), 16):
            end = min(start + 16, len(train["state"]))
            tokens, _ = cpu_model.build_tokens(train["observations"][start:end], train["initial_state"][start:end])
            values = tokens.index_select(1, selected).to(torch.float64)
            sum_values += values.sum(dim=(0, 2))
            sum_squares += values.square().sum(dim=(0, 2))
            count += values.shape[0] * values.shape[2]
    mean = sum_values / count
    variance = (sum_squares / count - mean.square()).clamp_min(0.0)
    raw_std = variance.sqrt()
    std = raw_std.clamp_min(1e-4)
    mean_full = torch.zeros(21, dtype=torch.float32)
    std_full = torch.ones(21, dtype=torch.float32)
    mean_full[selected] = mean.float()
    std_full[selected] = std.float()
    details = {
        "statistics_source": "train split only",
        "train_trajectory_count": int(len(train["state"])),
        "time_points_per_trajectory": int(train["state"].shape[-1]),
        "population_standard_deviation": True,
        "normalization_formula": "(x - train_mean) / max(train_population_std, 1e-4)",
        "normalized_token_indices_zero_based": list(NORMALIZED_FEATURE_INDICES),
        "preserved_unscaled_token_indices_zero_based": [1, 2, 6],
        "std_floor": 1e-4,
        "feature_mean_21": mean_full.tolist(),
        "feature_std_21": std_full.tolist(),
        "raw_std_18_before_floor": raw_std.tolist(),
    }
    del cpu_model
    return mean_full, std_full, details


def build_model(config: dict[str, Any], feature_details: dict[str, Any] | None, device: torch.device) -> torch.nn.Module:
    kwargs = make_kwargs(config, config["sequence_length"])
    if config["model"]["variant"] == "M1":
        if feature_details is None:
            raise ValueError("M1 requires train-only normalization statistics")
        model = TrainNormalizedFeatureTracker(
            feature_mean=torch.tensor(feature_details["feature_mean_21"]),
            feature_std=torch.tensor(feature_details["feature_std_21"]),
            **kwargs,
        )
    else:
        model = BiMambaTrackerModel(**kwargs)
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if params != 619014:
        raise ValueError(f"expected 619014 trainable parameters, found {params}")
    return model.to(device)


def evaluate_validation(model: torch.nn.Module, validation: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], float]:
    model.eval()
    predictions = []
    sync_cuda(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(validation["state"]), 16):
            end = min(start + 16, len(validation["state"]))
            out = model(
                observations=validation["observations"][start:end].to(device, non_blocking=True),
                init_state=validation["initial_state"][start:end].to(device, non_blocking=True),
            )
            estimate = out["state_estimates"]
            expected_shape = (end - start, 6, validation["state"].shape[-1])
            if tuple(estimate.shape) != expected_shape:
                raise ValueError(f"validation output shape {tuple(estimate.shape)} != {expected_shape}")
            if not torch.isfinite(estimate).all():
                raise FloatingPointError("non-finite validation output")
            predictions.append(estimate.detach().cpu())
            del out, estimate
    sync_cuda(device)
    elapsed = time.perf_counter() - started
    return metric_values(torch.cat(predictions, dim=0), validation["state"]), elapsed


def device_memory() -> dict[str, int]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def save_best_checkpoint(path: Path, model: torch.nn.Module, epoch: int, val_mse: float, config: dict[str, Any], config_hash: str, stats_hash: str | None) -> str:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({
        "state_dict": state,
        "checkpoint_selection": "minimum validation six-state MSE",
        "selected_epoch": epoch,
        "validation_six_state_mse_scaled_units": val_mse,
        "config_sha256": config_hash,
        "feature_statistics_sha256": stats_hash,
    }, path)
    return sha256_file(path)


def train_one(config_path: Path, data: dict[str, Any], device: torch.device, environment: dict[str, Any], common_sources: dict[str, str]) -> dict[str, Any]:
    config_hash = sha256_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    length = config["sequence_length"]
    variant = config["model"]["variant"]
    run_dir = PROJECT / config["outputs"]["run_directory"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite screening run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_bytes(config_path.read_bytes())
    (run_dir / "config.json.sha256").write_text(f"{config_hash}  config.json\n", encoding="ascii")
    data_record = next(item for item in data["dataset_records"] if item["length"] == length)
    run_summary: dict[str, Any] = {
        "status": "RUNNING",
        "experiment_id": config["experiment_id"],
        "variant": variant,
        "sequence_length": length,
        "screening_seed": config["training"]["screening_seed"],
        "random_seeds": {
            key: config["training"][key]
            for key in ("initialization_seed", "training_random_seed", "dataloader_shuffle_seed")
        },
        "config_path": str(config_path.relative_to(PROJECT)),
        "config_sha256": config_hash,
        "dataset": data_record,
        "parameter_count": 619014,
        "training_config": config["training"],
        "validation_screening_criteria": config["validation_screening_criteria"],
        "environment": environment,
        "source_hashes": common_sources,
        "test_accessed": False,
        "test_files_opened": False,
        "epoch_history": [],
    }
    write_json(run_dir / "run_summary.json", run_summary)
    checkpoint_path = run_dir / "best_validation_checkpoint.pt"
    log_path = run_dir / "epoch_log.jsonl"
    try:
        feature_details = None
        feature_hash = None
        if variant == "M1":
            _, _, feature_details = fit_feature_statistics(data["train"], config)
            feature_hash = write_json(run_dir / "train_only_feature_statistics.json", feature_details)
        set_seed(config["training"]["initialization_seed"])
        model = build_model(config, feature_details, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(config["training"]["dataloader_shuffle_seed"])
        loader = DataLoader(
            TensorDataset(data["train"]["observations"], data["train"]["initial_state"], data["train"]["state"]),
            batch_size=16,
            shuffle=True,
            generator=loader_generator,
            pin_memory=True,
            num_workers=0,
        )
        set_seed(config["training"]["training_random_seed"])
        best_val = float("inf")
        best_epoch = 0
        no_improve = 0
        whole_train_started = time.perf_counter()
        total_validation_seconds = 0.0

        for epoch in range(config["training"]["max_epochs"]):
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            sync_cuda(device)
            epoch_start = time.perf_counter()
            loss_total = 0.0
            example_count = 0
            for obs, initial, target in loader:
                obs = obs.to(device, non_blocking=True)
                initial = initial.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(observations=obs, init_state=initial)["state_estimates"]
                loss = F.mse_loss(prediction, target, reduction="mean")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_total += float(loss.detach().item()) * obs.shape[0]
                example_count += obs.shape[0]
                del obs, initial, target, prediction, loss
            sync_cuda(device)
            train_seconds = time.perf_counter() - epoch_start
            train_memory = device_memory()

            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            sync_cuda(device)
            validation_started = time.perf_counter()
            with torch.inference_mode():
                predictions = []
                for start in range(0, len(data["validation"]["state"]), 16):
                    end = min(start + 16, len(data["validation"]["state"]))
                    out = model(
                        observations=data["validation"]["observations"][start:end].to(device, non_blocking=True),
                        init_state=data["validation"]["initial_state"][start:end].to(device, non_blocking=True),
                    )
                    estimate = out["state_estimates"]
                    if tuple(estimate.shape) != (end - start, 6, length):
                        raise ValueError(f"validation output shape mismatch at epoch {epoch + 1}")
                    if not torch.isfinite(estimate).all():
                        raise FloatingPointError(f"non-finite validation output at epoch {epoch + 1}")
                    predictions.append(estimate.detach().cpu())
                    del out, estimate
            sync_cuda(device)
            validation_seconds = time.perf_counter() - validation_started
            total_validation_seconds += validation_seconds
            validation_memory = device_memory()
            val_metrics = metric_values(torch.cat(predictions, dim=0), data["validation"]["state"])
            train_mse = loss_total / example_count
            if not math.isfinite(train_mse):
                raise FloatingPointError(f"non-finite epoch training loss at epoch {epoch + 1}")
            val_mse = val_metrics["six_state_mse_scaled_units"]
            row = {
                "epoch": epoch + 1,
                "train_mse": train_mse,
                "validation_metrics": val_metrics,
                "training_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "training_peak_memory": train_memory,
                "validation_peak_memory": validation_memory,
            }
            run_summary["epoch_history"].append(row)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
            if val_mse < best_val - config["training"]["min_delta"]:
                best_val = val_mse
                best_epoch = epoch + 1
                save_best_checkpoint(checkpoint_path, model, best_epoch, best_val, config, config_hash, feature_hash)
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
                print(
                    f"{variant} T={length} epoch={epoch + 1:03d} train_mse={train_mse:.6g} "
                    f"val_mse={val_mse:.6g} pos={val_metrics['position_rmse_m']:.2f} "
                    f"vel={val_metrics['velocity_rmse_mps']:.2f} acc={val_metrics['acceleration_rmse_mps2']:.2f}",
                    flush=True,
                )
            if no_improve >= config["training"]["early_stopping_patience"]:
                break

        sync_cuda(device)
        training_seconds = time.perf_counter() - whole_train_started
        if not checkpoint_path.is_file():
            raise RuntimeError("no validation-selected checkpoint was written")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        sync_cuda(device)
        timed_started = time.perf_counter()
        selected_metrics, validation_timing_seconds = evaluate_validation(model, data["validation"], device)
        sync_cuda(device)
        selected_validation_memory = device_memory()
        checkpoint_hash = sha256_file(checkpoint_path)
        if abs(selected_metrics["six_state_mse_scaled_units"] - best_val) > 1e-7:
            raise RuntimeError("reloaded selected checkpoint does not reproduce the selected validation MSE")
        run_summary.update({
            "status": "COMPLETED",
            "epochs_run": len(run_summary["epoch_history"]),
            "selected_epoch": best_epoch,
            "best_validation_six_state_mse_scaled_units": best_val,
            "selected_validation_metrics": selected_metrics,
            "training_wall_seconds": training_seconds,
            "sum_epoch_training_seconds": sum(row["training_seconds"] for row in run_summary["epoch_history"]),
            "sum_epoch_validation_seconds": total_validation_seconds,
            "selected_checkpoint_validation_seconds": validation_timing_seconds,
            "selected_checkpoint_validation_peak_memory": selected_validation_memory,
            "training_peak_allocated_bytes_max_over_epochs": max(row["training_peak_memory"]["peak_allocated_bytes"] for row in run_summary["epoch_history"]),
            "training_peak_reserved_bytes_max_over_epochs": max(row["training_peak_memory"]["peak_reserved_bytes"] for row in run_summary["epoch_history"]),
            "validation_peak_allocated_bytes_max_over_epochs": max(row["validation_peak_memory"]["peak_allocated_bytes"] for row in run_summary["epoch_history"]),
            "validation_peak_reserved_bytes_max_over_epochs": max(row["validation_peak_memory"]["peak_reserved_bytes"] for row in run_summary["epoch_history"]),
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT)),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "feature_statistics_path": str((run_dir / "train_only_feature_statistics.json").relative_to(PROJECT)) if feature_hash else None,
            "feature_statistics_sha256": feature_hash,
            "strict_state_dict_reload": True,
            "test_accessed": False,
            "test_files_opened": False,
        })
        write_json(run_dir / "run_summary.json", run_summary)
        print(f"Completed {variant} T={length}; selected epoch {best_epoch}", flush=True)
        del model, optimizer, loader
        torch.cuda.empty_cache()
        return run_summary
    except Exception as exc:
        sync_cuda(device)
        run_summary.update({
            "status": "FAILED",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "epochs_run": len(run_summary["epoch_history"]),
            "test_accessed": False,
            "test_files_opened": False,
        })
        write_json(run_dir / "run_summary.json", run_summary)
        print(f"FAILED {variant} T={length}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
        return run_summary


def hardware_record() -> dict[str, Any]:
    import mamba_ssm
    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception as exc:
        driver = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_free_memory_bytes_at_start": int(torch.cuda.mem_get_info(0)[0]),
        "gpu_driver_version": driver,
        "cuda_runtime_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "mamba_ssm_version": importlib.metadata.version("mamba-ssm"),
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(results: list[dict[str, Any]], output: Path) -> dict[str, str]:
    plot_dir = output / "plots"
    plot_dir.mkdir()
    hashes = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for result in results:
        history = result["epoch_history"]
        label = f"{result['variant']} T={result['sequence_length']}"
        epochs = [row["epoch"] for row in history]
        axes[0, 0].plot(epochs, [row["train_mse"] for row in history], label=label)
        axes[0, 1].plot(epochs, [row["validation_metrics"]["six_state_mse_scaled_units"] for row in history], label=label)
    axes[0, 0].set_title("Training MSE")
    axes[0, 1].set_title("Validation six-state MSE")
    for ax in axes[0]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Scaled-state MSE")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.delaxes(axes[1, 0])
    fig.delaxes(axes[1, 1])
    fig.tight_layout()
    path = plot_dir / "training_and_validation_loss.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    hashes[str(path.relative_to(PROJECT))] = sha256_file(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, METRICS):
        positions = np.arange(2)
        width = 0.18
        successes = [result for result in results if result.get("selected_validation_metrics")]
        for index, result in enumerate(successes):
            pos = (0 if result["sequence_length"] == 200 else 1) + (-1.5 + index) * width
            val = result["selected_validation_metrics"][metric]
            ax.bar(pos, val, width=width, label=f"{result['variant']} T={result['sequence_length']}")
        ax.set_xticks(positions, ["T=200", "T=400"])
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = plot_dir / "validation_metrics_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    hashes[str(path.relative_to(PROJECT))] = sha256_file(path)
    return hashes


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 5A output root: {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the installed Mamba implementation")
    config_paths = [
        PROJECT / f"configs/paper_v4_long_sequence/stage5A_T{length}_{variant}_seed101.json"
        for length in LENGTHS for variant in VARIANTS
    ]
    for path in config_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen screening config: {path}")
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in config_paths]
    config_hashes = {str(path.relative_to(PROJECT)): sha256_file(path) for path in config_paths}
    criteria_doc_hash = sha256_file(PROJECT / "docs/reports/paper_v4_mamba_improvement_plan.md")
    if any(config["validation_screening_criteria"]["source_document_sha256"] != criteria_doc_hash for config in configs):
        raise ValueError("The documented prospective validation guardrail source changed after config freeze")
    if any(config["training"]["batch_size"] != 16 or config["training"]["max_epochs"] != 180 for config in configs):
        raise ValueError("A frozen screening config differs from the prescribed training schedule")
    datasets = {}
    dataset_records = []
    for length in LENGTHS:
        datasets[length], record = load_training_splits(length)
        dataset_records.append(record)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: sha256_file(PROJECT / name) for name in SOURCE_FILES}
    hardware = hardware_record()
    environment = {**hardware, "cuda_device_index": 0, "device": "cuda:0", "peak_memory_measurement": "reset at start of each epoch's training and validation windows; cuda synchronized at window boundaries"}
    data_per_length = {
        length: {
            **datasets[length],
            "dataset_records": dataset_records,
        }
        for length in LENGTHS
    }

    results = []
    for config_path, config in zip(config_paths, configs):
        result = train_one(config_path, data_per_length[config["sequence_length"]], torch.device("cuda:0"), environment, source_hashes)
        results.append(result)

    by_length = {}
    for length in LENGTHS:
        m0 = next(r for r in results if r["sequence_length"] == length and r["variant"] == "M0")
        m1 = next(r for r in results if r["sequence_length"] == length and r["variant"] == "M1")
        m0m = m0.get("selected_validation_metrics")
        m1m = m1.get("selected_validation_metrics")
        if m0["status"] != "COMPLETED" or m1["status"] != "COMPLETED":
            by_length[str(length)] = {"status": "INCOMPLETE", "M0_status": m0["status"], "M1_status": m1["status"], "test_accessed": False}
            continue
        position_relative = m1m["position_rmse_m"] / m0m["position_rmse_m"] - 1.0
        velocity_relative = m1m["velocity_rmse_mps"] / m0m["velocity_rmse_mps"] - 1.0
        eligible = position_relative <= 0.01 and velocity_relative <= 0.03
        by_length[str(length)] = {
            "status": "SCREENED",
            "M0_status": m0["status"],
            "M1_status": m1["status"],
            "M1_vs_M0_position_relative_change": position_relative,
            "M1_vs_M0_velocity_relative_change": velocity_relative,
            "M1_vs_M0_acceleration_relative_change": m1m["acceleration_rmse_mps2"] / m0m["acceleration_rmse_mps2"] - 1.0,
            "M1_vs_M0_six_state_mse_relative_change": m1m["six_state_mse_scaled_units"] / m0m["six_state_mse_scaled_units"] - 1.0,
            "position_guardrail_pass": position_relative <= 0.01,
            "velocity_guardrail_pass": velocity_relative <= 0.03,
            "M1_eligible_for_five_seed_confirmation": eligible,
            "M0_reference_control_eligible": True,
            "selection_is_validation_only": True,
            "test_accessed": False,
        }

    summary = {
        "stage": "Phase 3 Stage 5A one-seed matched-length screening",
        "status": "PASS" if all(result["status"] == "COMPLETED" for result in results) else "PARTIAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"test_accessed": False, "test_files_opened": False, "training": True, "confirmation_seeds_used": False},
        "screening_seed": 101,
        "config_hashes": config_hashes,
        "dataset_integrity": dataset_records,
        "validation_screening_results": by_length,
        "run_summaries": results,
        "hardware": hardware,
        "source_hashes": source_hashes,
        "output_artifact_hashes": {},
    }
    flat = []
    for result in results:
        row = {
            "variant": result["variant"],
            "sequence_length": result["sequence_length"],
            "status": result["status"],
            "selected_epoch": result.get("selected_epoch"),
            "epochs_run": result.get("epochs_run"),
            "training_wall_seconds": result.get("training_wall_seconds"),
            "sum_epoch_validation_seconds": result.get("sum_epoch_validation_seconds"),
            "parameter_count": result["parameter_count"],
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "config_sha256": result["config_sha256"],
            "train_split_sha256": result["dataset"]["splits"]["train"]["split_file_sha256"],
            "validation_split_sha256": result["dataset"]["splits"]["validation"]["split_file_sha256"],
            "test_accessed": result["test_accessed"],
        }
        if result.get("selected_validation_metrics"):
            row.update({f"validation_{key}": result["selected_validation_metrics"][key] for key in METRICS})
        flat.append(row)
    write_csv(OUTPUT_ROOT / "screening_summary.csv", flat)
    plot_hashes = make_plots(results, OUTPUT_ROOT)
    summary["output_artifact_hashes"] = {
        "screening_summary.csv": sha256_file(OUTPUT_ROOT / "screening_summary.csv"),
        "plots": plot_hashes,
    }
    json_hash = write_json(OUTPUT_ROOT / "stage5a_screening_summary.json", summary)
    md_lines = [
        "# Phase 3 Stage 5A: Matched-Length M0/M1 Screening",
        "",
        f"**Status:** {summary['status']}",
        "",
        "One screening seed (101) was used. Only train and validation tensors were opened. No T=200/T=400 test tensor was opened or evaluated. No five-seed confirmation or hyperparameter adaptation occurred.",
        "",
        "## Frozen screening rule",
        "",
        "Checkpoints were selected by minimum validation six-state MSE. M1 eligibility uses the previously documented prospective guardrails against the same-length M0 validation reference: position RMSE may increase by at most 1%, and velocity RMSE by at most 3%. The source rule is recorded in each frozen config. Stability also requires finite training/validation values and a selected validation checkpoint. No additional threshold was introduced.",
        "",
        "## Results",
        "",
        "Metrics below are for each run's validation-selected checkpoint.",
        "",
        "| T | Variant | Status | Selected epoch | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE | Training (s) |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result.get("selected_validation_metrics", {})
        metric_text = " | ".join(f"{metrics[key]:.6g}" if key in metrics else "—" for key in METRICS)
        runtime_text = f"{result['training_wall_seconds']:.2f}" if result.get("training_wall_seconds") is not None else "—"
        selected_epoch = result.get("selected_epoch", "—")
        md_lines.append(
            f"| {result['sequence_length']} | {result['variant']} | {result['status']} | {selected_epoch} | {metric_text} | {runtime_text} |"
        )
    md_lines.extend(["", "### Validation guardrails", "", "| T | M1 position change | M1 velocity change | M1 acceleration change | M1 confirmation eligible |", "|---:|---:|---:|---:|---|"])
    for length in LENGTHS:
        row = by_length[str(length)]
        if row["status"] != "SCREENED":
            md_lines.append(f"| {length} | — | — | — | No: one or more runs did not complete |")
        else:
            md_lines.append(
                f"| {length} | {row['M1_vs_M0_position_relative_change']:+.2%} | {row['M1_vs_M0_velocity_relative_change']:+.2%} | "
                f"{row['M1_vs_M0_acceleration_relative_change']:+.2%} | {'Yes' if row['M1_eligible_for_five_seed_confirmation'] else 'No'} |"
            )
    md_lines.extend([
        "",
        "## Resources and provenance",
        "",
        f"GPU: {hardware['gpu_name']}; driver {hardware['gpu_driver_version']}; CUDA {hardware['cuda_runtime_version']}; PyTorch {hardware['pytorch_version']}; Mamba {hardware['mamba_ssm_version']}; Python {platform.python_version()}.",
        "",
        "Training and validation peak allocated/reserved memory are recorded separately per epoch in each run summary. Exact checkpoint, config, dataset, and source hashes are in the JSON summaries and run directories.",
        "",
        "- `screening_summary.csv`: one row per configuration.",
        "- `plots/training_and_validation_loss.png`: train and validation six-state MSE curves.",
        "- `plots/validation_metrics_comparison.png`: four validation metrics across runs.",
        "- Per-run `epoch_log.jsonl`, `run_summary.json`, and successful selected checkpoint files are stored in their isolated run folders.",
        "",
        "No test-based model selection or Stage 5B decision was performed. These single-seed screens are feasibility evidence only; they do not establish five-seed performance.",
    ])
    md_path = OUTPUT_ROOT / "stage5a_screening_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    md_hash = sha256_file(md_path)
    md_path.with_name(md_path.name + ".sha256").write_text(f"{md_hash}  {md_path.name}\n", encoding="ascii")
    write_json(OUTPUT_ROOT / "provenance.json", {
        "stage": "Phase 3 Stage 5A provenance",
        "status": summary["status"],
        "config_hashes": config_hashes,
        "dataset_integrity": dataset_records,
        "source_hashes": source_hashes,
        "hardware": hardware,
        "outputs": {
            "summary_json_sha256": json_hash,
            "summary_markdown_sha256": md_hash,
            **summary["output_artifact_hashes"],
        },
    })
    print(json.dumps({"status": summary["status"], "output_directory": str(OUTPUT_ROOT), "run_statuses": [{"T": r["sequence_length"], "variant": r["variant"], "status": r["status"], "selected_epoch": r.get("selected_epoch")} for r in results], "json_sha256": json_hash}, indent=2), flush=True)


if __name__ == "__main__":
    main()
