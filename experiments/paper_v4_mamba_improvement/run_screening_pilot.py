#!/usr/bin/env python3
"""Run exactly one seed-11, train/validation-only screening pilot at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.paper_v4_mamba_improvement.model_variants import (  # noqa: E402
    NORMALIZED_FEATURE_INDICES,
    SeparateStateGroupResidualHead,
    TrainNormalizedFeatureTracker,
    parameter_counts,
)
from radar_tracking.cli import benchmark_mamba as benchmark  # noqa: E402
from radar_tracking.config import simparams_class  # noqa: E402


POOL = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_train_val.pt"
BASELINE_DIR = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
EXPECTED_POOL_SHA256 = "4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733"
SEED = 11
SCALE = 1000.0
TOKEN_NAMES = [
    "range", "sin_bearing", "cos_bearing", "cartesian_x", "cartesian_y",
    "delta_range", "delta_bearing_wrapped", "delta_cartesian_x", "delta_cartesian_y",
    "coarse_x", "coarse_vx", "coarse_ax", "coarse_y", "coarse_vy", "coarse_ay",
    "initial_x_repeated", "initial_vx_repeated", "initial_ax_repeated",
    "initial_y_repeated", "initial_vy_repeated", "initial_ay_repeated",
]
VARIANTS = {
    "feature_norm": {
        "config": "configs/paper_v4_mamba_improvement/features_train_only_normalization_seed11.json",
        "output": "artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_a_feature_norm_seed11",
    },
    "separate_heads": {
        "config": "configs/paper_v4_mamba_improvement/head_separate_state_groups_seed11.json",
        "output": "artifacts/results/synthetic/paper_v4_mamba_improvement_v2/experiment_b_separate_heads_seed11",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metrics(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    error = prediction - truth
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Non-finite model prediction")
    state_mse = error.square().mean(dim=(0, 2))

    def group_rmse(i: int, j: int, sl: slice | None = None) -> float:
        selected = error[:, [i, j], :] if sl is None else error[:, [i, j], sl]
        return float((selected.square().sum(dim=1).mean().sqrt() * SCALE).item())

    output = {
        "position_rmse_m": group_rmse(0, 3),
        "velocity_rmse_mps": group_rmse(1, 4),
        "acceleration_rmse_mps2": group_rmse(2, 5),
        "six_state_mse_scaled_units": float(error.square().mean().item()),
        "six_state_mse_db": 10.0 * math.log10(float(error.square().mean().item())),
        "per_state_rmse_physical": [float(x) for x in (state_mse.sqrt() * SCALE).cpu().tolist()],
        "per_state_bias_physical": [float(x) for x in (error.mean(dim=(0, 2)) * SCALE).cpu().tolist()],
    }
    windows = []
    for start in range(0, truth.shape[-1], 10):
        end = min(start + 10, truth.shape[-1])
        row = {"start_step_1based": start + 1, "end_step_1based": end}
        row["position_rmse_m"] = group_rmse(0, 3, slice(start, end))
        row["velocity_rmse_mps"] = group_rmse(1, 4, slice(start, end))
        row["acceleration_rmse_mps2"] = group_rmse(2, 5, slice(start, end))
        windows.append(row)
    output["temporal_windows"] = windows
    return output


@torch.no_grad()
def evaluate(model, tensors: dict[str, torch.Tensor], device: torch.device, batch_size: int = 16):
    model.eval()
    predictions = []
    for start in range(0, len(tensors["state"]), batch_size):
        end = min(start + batch_size, len(tensors["state"]))
        output = model(
            observations=tensors["obs"][start:end].to(device, non_blocking=True),
            init_state=tensors["init"][start:end].to(device, non_blocking=True),
        )["state_estimates"]
        predictions.append(output.detach().cpu())
    prediction = torch.cat(predictions, dim=0)
    return metrics(prediction, tensors["state"]), prediction


def timed_validation(model, validation, device, repeats: int = 3) -> dict[str, Any]:
    evaluate(model, validation, device)
    durations = []
    for _ in range(repeats):
        sync_cuda(device)
        started = time.perf_counter()
        evaluate(model, validation, device)
        sync_cuda(device)
        durations.append(time.perf_counter() - started)
    mean_seconds = sum(durations) / len(durations)
    return {
        "device": str(device),
        "warmup_passes": 1,
        "timed_passes": repeats,
        "mean_full_validation_seconds": mean_seconds,
        "seconds_per_trajectory": mean_seconds / len(validation["state"]),
        "raw_seconds": durations,
    }


def build_kwargs(simparams, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dim": 6,
        "obs_dim": 2,
        "dt": simparams.dt,
        "embed_dim": int(config["embed_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "d_state": int(config["d_state"]),
        "d_conv": int(config["d_conv"]),
        "expand": int(config["expand"]),
        "dropout": float(config["dropout"]),
        "encoder_direction": "forward",
        "task": "radar_benchmark",
    }


def compute_train_feature_stats(benchmark_config, simparams, train, device):
    """Fit feature mean/population std using only training observations/states."""
    benchmark.set_seed(SEED)
    token_builder = benchmark.build_mamba_model(simparams, benchmark_config, device)
    token_builder.eval()
    with torch.no_grad():
        tokens, _ = token_builder.build_tokens(train["obs"].to(device), train["init"].to(device))
        selected = torch.as_tensor(NORMALIZED_FEATURE_INDICES, device=device)
        values = tokens.index_select(1, selected).to(torch.float64)
        mean = values.mean(dim=(0, 2))
        std = values.var(dim=(0, 2), unbiased=False).sqrt().clamp_min(1e-4)
        feature_mean = torch.zeros(21, dtype=torch.float64, device=device)
        feature_std = torch.ones(21, dtype=torch.float64, device=device)
        feature_mean[selected] = mean
        feature_std[selected] = std
    details = {
        "statistics_source": "train split only",
        "train_trajectory_count": int(train["state"].shape[0]),
        "time_points_per_trajectory": int(train["state"].shape[-1]),
        "normalized_token_indices_zero_based": list(NORMALIZED_FEATURE_INDICES),
        "token_names_in_order": TOKEN_NAMES,
        "normalized_feature_names": [TOKEN_NAMES[index] for index in NORMALIZED_FEATURE_INDICES],
        "preserved_unscaled_token_indices_zero_based": [1, 2, 6],
        "normalization_formula": "(x - train_mean) / max(train_population_std, 1e-4)",
        "feature_mean_21": [float(x) for x in feature_mean.cpu().tolist()],
        "feature_std_21": [float(x) for x in feature_std.cpu().tolist()],
        "train_only_raw_channel_mean_21": [float(x) for x in tokens.to(torch.float64).mean(dim=(0, 2)).cpu().tolist()],
        "train_only_raw_channel_std_21": [float(x) for x in tokens.to(torch.float64).var(dim=(0, 2), unbiased=False).sqrt().cpu().tolist()],
    }
    return feature_mean.cpu().float(), feature_std.cpu().float(), details


def construct_candidate(experiment: str, simparams, cfg, device, feature_mean=None, feature_std=None):
    benchmark.set_seed(SEED)
    if experiment == "feature_norm":
        model = TrainNormalizedFeatureTracker(
            feature_mean=feature_mean,
            feature_std=feature_std,
            **build_kwargs(simparams, cfg),
        )
    else:
        model = benchmark.build_mamba_model(simparams, benchmark.MambaTrainConfig(**cfg), device)
        model.state_head = SeparateStateGroupResidualHead(input_dim=128, hidden_dim=128)
    return model.to(device)


def train_candidate(model, train, validation, device, output: Path, train_config: dict[str, Any]):
    dataset = TensorDataset(train["obs"], train["init"], train["state"])
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    checkpoint = output / "best_validation_checkpoint.pt"
    started = time.perf_counter()

    for epoch in range(180):
        model.train()
        loss_sum = 0.0
        batches = 0
        for obs_batch, init_batch, state_batch in loader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            init_batch = init_batch.to(device, non_blocking=True)
            state_batch = state_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observations=obs_batch, init_state=init_batch)["state_estimates"]
            loss = torch.nn.functional.mse_loss(prediction, state_batch, reduction="mean")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.detach().item())
            batches += 1

        val_metrics, _ = evaluate(model, validation, device)
        val_mse = val_metrics["six_state_mse_scaled_units"]
        row = {
            "epoch": epoch + 1,
            "train_mse_batch_mean": loss_sum / max(batches, 1),
            "validation_metrics": val_metrics,
        }
        history.append(row)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_epoch = epoch + 1
            with checkpoint.open("wb") as stream:
                torch.save(model.state_dict(), stream)
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"{train_config['experiment']} seed=11 epoch={epoch + 1:03d} "
                f"val_mse={val_mse:.6f} val_pos={val_metrics['position_rmse_m']:.2f} "
                f"val_vel={val_metrics['velocity_rmse_mps']:.2f} "
                f"val_acc={val_metrics['acceleration_rmse_mps2']:.2f}",
                flush=True,
            )
        if no_improve >= 25:
            break

    sync_cuda(device)
    training_seconds = time.perf_counter() - started
    if not checkpoint.is_file():
        raise RuntimeError("No validation-selected checkpoint was written")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_metrics, _ = evaluate(model, train, device)
    validation_metrics, _ = evaluate(model, validation, device)
    run_record = {
        "experiment": train_config["experiment"],
        "seed": SEED,
        "training_config": train_config,
        "epochs_run": len(history),
        "checkpoint_selection_epoch": best_epoch,
        "best_validation_six_state_mse": best_val,
        "training_seconds": training_seconds,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": parameter_counts(model),
        "train_metrics_at_selected_checkpoint": train_metrics,
        "validation_metrics_at_selected_checkpoint": validation_metrics,
        "history": history,
        "test_accessed": False,
    }
    write_new(output / "run_metrics.json", run_record)
    return run_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    spec = VARIANTS[args.experiment]
    config_path = PROJECT / spec["config"]
    output = PROJECT / spec["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output {output}")
    if not POOL.is_file() or sha256(POOL) != EXPECTED_POOL_SHA256:
        raise RuntimeError("Fixed train/validation pool missing or hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("Current Mamba implementation requires CUDA")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["data"]["test_access"] is not False or config["training"]["seed"] != SEED:
        raise RuntimeError("Pilot config violates the seed/test access contract")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    raw_pool = torch.load(POOL, map_location="cpu", weights_only=True)
    train = {
        "state": raw_pool["train"]["state"],
        "obs": raw_pool["train"]["observations"],
        "init": raw_pool["train"]["initial_state"],
    }
    validation = {
        "state": raw_pool["validation"]["state"],
        "obs": raw_pool["validation"]["observations"],
        "init": raw_pool["validation"]["initial_state"],
    }
    if train["state"].shape != (819, 6, 100) or validation["state"].shape != (204, 6, 100):
        raise RuntimeError("Unexpected fixed train/validation tensor shapes")

    baseline_record = json.loads((BASELINE_DIR / "seed_11" / "run_metrics.json").read_text())
    baseline_checkpoint = BASELINE_DIR / "seed_11" / "best_mamba_tracker.pt"
    if sha256(baseline_checkpoint) != baseline_record["mamba"]["checkpoint_sha256"]:
        raise RuntimeError("Seed-11 baseline checkpoint hash does not match its run record")

    device = torch.device("cuda")
    simparams = simparams_class()
    simparams.seq_len = 100
    baseline_cfg = benchmark.MambaTrainConfig(**baseline_record["mamba"]["config"])
    benchmark.set_seed(SEED)
    baseline_model = benchmark.build_mamba_model(simparams, baseline_cfg, device)
    baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
    baseline_val_metrics, _ = evaluate(baseline_model, validation, device)
    baseline_timing = timed_validation(baseline_model, validation, device)
    baseline_parameter_count = parameter_counts(baseline_model)

    train_config = {
        "experiment": config["experiment"],
        "seed": SEED,
        "epochs_cap": 180,
        "batch_size": 16,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 25,
        "min_delta": 1e-6,
        "gradient_clip_norm": 1.0,
        "loss": "mean squared error across batch, six states and time",
        "checkpoint_selection": "minimum validation six-state MSE",
        "model": {
            "embed_dim": 128,
            "hidden_dim": 256,
            "num_layers": 3,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.08,
            "direction": "forward_causal",
        },
        "factor": config["factor_changed"],
    }

    feature_details = None
    feature_mean = feature_std = None
    if args.experiment == "feature_norm":
        feature_mean, feature_std, feature_details = compute_train_feature_stats(
            baseline_cfg, simparams, train, device
        )
        write_new(output / "train_only_feature_statistics.json", feature_details)
        candidate = construct_candidate(
            args.experiment,
            simparams,
            {
                **train_config["model"],
            },
            device,
            feature_mean,
            feature_std,
        )
        train_config["feature_normalization"] = {
            "normalized_indices": list(NORMALIZED_FEATURE_INDICES),
            "preserved_indices": [1, 2, 6],
            "std_floor": 1e-4,
            "statistics_source": "training split only",
        }
    else:
        candidate = benchmark.build_mamba_model(simparams, baseline_cfg, device)
        candidate.state_head = SeparateStateGroupResidualHead(input_dim=128, hidden_dim=128).to(device)
        train_config["model"]["prediction_head"] = "3 independent 128->128->64->2 branches; concatenate to [dx,dvx,dax,dy,dvy,day]"
        train_config["model"]["branch_parameter_count"] = sum(p.numel() for p in candidate.state_head.parameters())

    manifest = {
        "experiment": config["experiment"],
        "test_accessed": False,
        "train_validation_file": str(POOL.relative_to(PROJECT)),
        "train_validation_sha256": sha256(POOL),
        "train_count": len(train["state"]),
        "validation_count": len(validation["state"]),
        "sequence_length": 100,
        "seed": SEED,
        "baseline_checkpoint": str(baseline_checkpoint.relative_to(PROJECT)),
        "baseline_checkpoint_sha256": sha256(baseline_checkpoint),
        "baseline_validation_metrics": baseline_val_metrics,
        "baseline_parameter_count": baseline_parameter_count,
        "test_file_not_loaded": True,
    }
    write_new(output / "manifest.json", manifest)

    run_record = train_candidate(candidate, train, validation, device, output, train_config)
    candidate_timing = timed_validation(candidate, validation, device)
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("model_variants.py").resolve(),
        PROJECT / "src/radar_tracking/cli/benchmark_mamba.py",
        PROJECT / "src/radar_tracking/models/tracker_model.py",
        PROJECT / "src/radar_tracking/models/bimamba_encoder.py",
        PROJECT / "src/radar_tracking/models/prediction_head.py",
        PROJECT / "src/radar_tracking/data/data_generation.py",
        config_path,
    ]
    source_hashes = {str(path.relative_to(PROJECT)): sha256(path) for path in source_paths}
    summary = {
        "experiment": config["experiment"],
        "test_accessed": False,
        "train_validation_pool_sha256": sha256(POOL),
        "baseline_checkpoint_sha256": sha256(baseline_checkpoint),
        "pilot_checkpoint_sha256": run_record["checkpoint_sha256"],
        "seed": SEED,
        "epochs_run": run_record["epochs_run"],
        "checkpoint_selection_epoch": run_record["checkpoint_selection_epoch"],
        "training_seconds": run_record["training_seconds"],
        "baseline_parameter_count": baseline_parameter_count,
        "pilot_parameter_count": run_record["parameter_count"],
        "baseline_validation_metrics": baseline_val_metrics,
        "pilot_validation_metrics": run_record["validation_metrics_at_selected_checkpoint"],
        "validation_inference_timing": {"baseline": baseline_timing, "pilot": candidate_timing},
        "feature_statistics": feature_details,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "source_sha256": source_hashes,
    }
    write_new(output / "pilot_summary.json", summary)
    print(json.dumps({
        "output": str(output),
        "test_accessed": False,
        "selection_epoch": run_record["checkpoint_selection_epoch"],
        "training_seconds": run_record["training_seconds"],
        "parameters": run_record["parameter_count"],
        "baseline_validation": {
            key: baseline_val_metrics[key]
            for key in ["position_rmse_m", "velocity_rmse_mps", "acceleration_rmse_mps2", "six_state_mse_scaled_units"]
        },
        "pilot_validation": {
            key: run_record["validation_metrics_at_selected_checkpoint"][key]
            for key in ["position_rmse_m", "velocity_rmse_mps", "acceleration_rmse_mps2", "six_state_mse_scaled_units"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
