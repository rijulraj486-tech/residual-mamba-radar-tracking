#!/usr/bin/env python3
"""Train causal acceleration-feature variants without touching M0 or test data."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))

from mamba_acceleration_features import causal_acceleration_features

OUTPUT = PROJECT / "artifacts/results/synthetic/mamba_acceleration_feature_ablation_v1"
BASELINE = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
POOL_PATH = BASELINE / "fixed_train_val.pt"
SEEDS = (11, 22, 33, 44, 55)
SCALE = 1000.0
WINDOWS = [(i, min(i + 10, 100)) for i in range(0, 100, 10)]
VARIANTS = {
    "A1_ema_alpha025": {"method": "ema_velocity", "alpha": 0.25},
    "A1_ema_alpha050": {"method": "ema_velocity", "alpha": 0.50},
    "A2_quadratic_w5": {"method": "trailing_quadratic", "window": 5},
    "A2_quadratic_w9": {"method": "trailing_quadratic", "window": 9},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().tolist()]


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    err = pred - target
    state_mse = err.square().mean(dim=(0, 2))
    groups = {"position": [0, 3], "velocity": [1, 4], "acceleration": [2, 5]}
    result: dict[str, Any] = {
        "state_order": ["x", "vx", "ax", "y", "vy", "ay"],
        "per_state_mse_scaled_units": finite_list(state_mse),
        "per_state_rmse_physical": finite_list(state_mse.sqrt() * SCALE),
        "per_state_bias_physical": finite_list(err.mean(dim=(0, 2)) * SCALE),
        "six_state_mse_scaled_units": float(err.square().mean()),
    }
    for name, indices in groups.items():
        group_error = err[:, indices, :]
        result[f"{name}_rmse_physical"] = float(
            (group_error.square().sum(dim=1).mean().sqrt() * SCALE).item()
        )
    windows = []
    for start, end in WINDOWS:
        row: dict[str, Any] = {"start_step_1based": start + 1, "end_step_1based": end}
        for name, indices in groups.items():
            group_error = err[:, indices, start:end]
            row[f"{name}_rmse_physical"] = float(
                (group_error.square().sum(dim=1).mean().sqrt() * SCALE).item()
            )
        row["six_state_rmse_physical"] = float(
            (err[:, :, start:end].square().mean().sqrt() * SCALE).item()
        )
        windows.append(row)
    result["temporal_windows"] = windows
    return result


class CausalAccelerationTracker:
    """Factory-like wrapper placeholder; subclass is built after imports."""


def make_model_class(base_model: type[torch.nn.Module]) -> type[torch.nn.Module]:
    class CausalAccelerationTrackerModel(base_model):
        def __init__(self, *args: Any, acceleration_feature: dict[str, Any], **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.acceleration_feature = dict(acceleration_feature)

        def coarse_state_sequence(self, observations: torch.Tensor,
                                  init_state: torch.Tensor) -> torch.Tensor:
            positions = self.measurement_to_cartesian(observations)
            coarse = super().coarse_state_sequence(observations, init_state)
            init_velocity = torch.stack((init_state[:, 1, 0], init_state[:, 4, 0]), dim=1)
            init_acceleration = torch.stack((init_state[:, 2, 0], init_state[:, 5, 0]), dim=1)
            acceleration = causal_acceleration_features(
                positions,
                init_velocity,
                init_acceleration,
                self.dt,
                **self.acceleration_feature,
            )
            coarse[:, 2, :] = acceleration[:, 0, :]
            coarse[:, 5, :] = acceleration[:, 1, :]
            return coarse

    return CausalAccelerationTrackerModel


def evaluate(model: torch.nn.Module, data: dict[str, torch.Tensor], device: torch.device,
             batch_size: int = 16) -> tuple[dict[str, Any], torch.Tensor]:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(data["state"]), batch_size):
            end = min(start + batch_size, len(data["state"]))
            parts.append(model(
                observations=data["obs"][start:end].to(device),
                init_state=data["init"][start:end].to(device),
            )["state_estimates"].cpu())
    prediction = torch.cat(parts)
    return metrics(prediction, data["state"]), prediction


def train_variant(api: dict[str, Any], seed: int, variant: str,
                  feature: dict[str, Any], train: dict[str, torch.Tensor],
                  val: dict[str, torch.Tensor], run_dir: Path,
                  device: torch.device) -> dict[str, Any]:
    torch = api["torch"]
    benchmark = api["benchmark"]
    config = benchmark.MambaTrainConfig(
        seed=seed, epochs=180, batch_size=16, eval_batch_size=16,
        learning_rate=1e-3, weight_decay=1e-4, patience=25, min_delta=1e-6,
        embed_dim=128, hidden_dim=256, num_layers=3, d_state=16, d_conv=4,
        expand=2, dropout=0.08, encoder_direction="forward",
        checkpoint_name="best_mamba_tracker.pt",
    )
    loader = DataLoader(TensorDataset(train["obs"], train["init"], train["state"]),
                        batch_size=config.batch_size, shuffle=True,
                        pin_memory=True, num_workers=0)
    benchmark.set_seed(seed)
    model = api["tracker_class"](
        task="radar_benchmark", state_dim=6, obs_dim=2, dt=1.0,
        embed_dim=config.embed_dim, hidden_dim=config.hidden_dim,
        num_layers=config.num_layers, d_state=config.d_state,
        d_conv=config.d_conv, expand=config.expand, dropout=config.dropout,
        encoder_direction="forward", acceleration_feature=feature,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    checkpoint = run_dir / "best_mamba_tracker.pt"
    started = time.perf_counter()
    train_batches = math.ceil(len(train["state"]) / config.batch_size)

    for epoch in range(config.epochs):
        model.train()
        squared_sum = torch.zeros(6, device=device)
        element_count = 0
        for obs, init, target in loader:
            obs, init, target = obs.to(device), init.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            estimate = model(observations=obs, init_state=init)["state_estimates"]
            loss = torch.nn.functional.mse_loss(estimate, target, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            squared_sum += (estimate.detach() - target).square().sum(dim=(0, 2))
            element_count += target.shape[0] * target.shape[2]
        train_state_mse = squared_sum / max(element_count, 1)
        val_metrics, val_prediction = evaluate(model, val, device, config.eval_batch_size)
        val_loss = val_metrics["six_state_mse_scaled_units"]
        history.append({
            "epoch": epoch + 1,
            "train_mse": float(train_state_mse.mean().item()),
            "validation_mse": val_loss,
            "train_per_state_mse_scaled_units": finite_list(train_state_mse),
            "validation_per_state_mse_scaled_units": val_metrics["per_state_mse_scaled_units"],
            "train_per_state_rmse_physical": finite_list(train_state_mse.sqrt() * SCALE),
            "validation_per_state_rmse_physical": val_metrics["per_state_rmse_physical"],
            "validation_group_rmse_physical": {
                k: val_metrics[f"{k}_rmse_physical"]
                for k in ("position", "velocity", "acceleration")
            },
        })
        if val_loss < best_val - config.min_delta:
            best_val, best_epoch, wait = val_loss, epoch + 1, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            wait += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{variant} seed={seed} epoch={epoch + 1} "
                  f"train={history[-1]['train_mse']:.6f} val={val_loss:.6f} "
                  f"val_acc={val_metrics['acceleration_rmse_physical']:.2f}", flush=True)
        if wait >= config.patience:
            break

    elapsed = time.perf_counter() - started
    if not checkpoint.is_file():
        raise RuntimeError(f"No validation-selected checkpoint for {variant}, seed {seed}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_metrics, _ = evaluate(model, train, device, config.eval_batch_size)
    val_metrics, _ = evaluate(model, val, device, config.eval_batch_size)
    record = {
        "variant": variant,
        "seed": seed,
        "feature": feature,
        "feature_description": (
            "EMA-smoothed Cartesian first-difference velocities; acceleration is "
            "the difference of smoothed velocities."
            if feature["method"] == "ema_velocity" else
            "Trailing local quadratic least-squares fit of measured Cartesian positions; "
            "acceleration is 2*c2/dt^2 from the trailing window."
        ),
        "initialization_handling": "coarse acceleration at indices 0 and 1 is copied from init_state",
        "config": {
            "epochs_cap": config.epochs, "actual_epochs": len(history),
            "batch_size": config.batch_size, "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay, "patience": config.patience,
            "min_delta": config.min_delta, "embed_dim": config.embed_dim,
            "hidden_dim": config.hidden_dim, "num_layers": config.num_layers,
            "d_state": config.d_state, "d_conv": config.d_conv,
            "expand": config.expand, "dropout": config.dropout,
            "encoder_direction": config.encoder_direction,
            "loss": "torch.nn.MSELoss(reduction='mean')",
        },
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "training_duration_seconds": elapsed,
        "optimizer_steps": train_batches * len(history),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "training_metrics_at_selected_checkpoint": train_metrics,
        "validation_metrics_at_selected_checkpoint": val_metrics,
        "history": history,
    }
    json_new(run_dir / "run_metrics.json", record)
    return record


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {OUTPUT}")
    if not POOL_PATH.is_file():
        raise FileNotFoundError(f"Fixed pool not found: {POOL_PATH}")
    from radar_tracking.cli import benchmark_mamba as benchmark
    from radar_tracking.models.tracker_model import BiMambaTrackerModel

    if not torch.cuda.is_available():
        raise RuntimeError("The existing Mamba implementation requires CUDA")
    device = torch.device("cuda")
    pool = torch.load(POOL_PATH, map_location="cpu", weights_only=True)
    train = {"state": pool["train"]["state"], "obs": pool["train"]["observations"],
             "init": pool["train"]["initial_state"]}
    val = {"state": pool["validation"]["state"], "obs": pool["validation"]["observations"],
           "init": pool["validation"]["initial_state"]}
    tracker_class = make_model_class(BiMambaTrackerModel)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source_files = [
        "src/radar_tracking/models/tracker_model.py",
        "src/radar_tracking/models/bimamba_encoder.py",
        "src/radar_tracking/models/prediction_head.py",
        "src/radar_tracking/cli/benchmark_mamba.py",
        "src/radar_tracking/data/data_generation.py",
        "src/radar_tracking/dynamics/constant_acceleration.py",
        "experiments/mamba_acceleration_features.py",
        "experiments/mamba_acceleration_feature_ablation_v1.py",
    ]
    json_new(OUTPUT / "manifest.json", {
        "experiment": "mamba_acceleration_feature_ablation_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": list(SEEDS), "train_count": len(train["state"]),
        "validation_count": len(val["state"]), "sequence_length": train["state"].shape[2],
        "dataset_path": str(POOL_PATH.relative_to(PROJECT)),
        "dataset_sha256": sha256(POOL_PATH),
        "original_test_accessed": False,
        "state_order": ["x", "vx", "ax", "y", "vy", "ay"],
        "input_token_dim": 21,
        "model_parameter_count_expected": 619014,
        "variants": VARIANTS,
        "config_common": {
            "loss": "torch.nn.MSELoss(reduction='mean')", "epochs": 180,
            "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 1e-4,
            "patience": 25, "min_delta": 1e-6, "gradient_clip_norm": 1.0,
            "embed_dim": 128, "hidden_dim": 256, "num_layers": 3,
            "d_state": 16, "d_conv": 4, "expand": 2, "dropout": 0.08,
            "encoder_direction": "forward",
        },
        "source_sha256": {name: sha256(PROJECT / name) for name in source_files},
        "m0_checkpoint_sha256": {
            str((BASELINE / f"seed_{seed}" / "best_mamba_tracker.pt").relative_to(PROJECT)):
                sha256(BASELINE / f"seed_{seed}" / "best_mamba_tracker.pt")
            for seed in SEEDS
        },
    })

    # M0 is the historical checkpoint. Evaluate it only on train/validation.
    m0_dir = OUTPUT / "A0_existing_reference"
    m0_dir.mkdir()
    simparams = benchmark.simparams_class()
    for seed in SEEDS:
        run_dir = m0_dir / f"seed_{seed}"
        run_dir.mkdir()
        baseline_record = json.loads((BASELINE / f"seed_{seed}" / "run_metrics.json").read_text())["mamba"]
        config = benchmark.MambaTrainConfig(**baseline_record["config"])
        benchmark.set_seed(seed)
        model = benchmark.build_mamba_model(simparams, config, device)
        checkpoint = BASELINE / f"seed_{seed}" / "best_mamba_tracker.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        train_metrics, _ = evaluate(model, train, device)
        val_metrics, _ = evaluate(model, val, device)
        json_new(run_dir / "run_metrics.json", {
            "variant": "A0_existing_reference", "seed": seed,
            "feature": {"method": "existing_second_difference"},
            "source": "preserved M0 checkpoint; not retrained",
            "best_epoch": baseline_record["best_epoch"],
            "actual_epochs": baseline_record["actual_epochs"],
            "training_duration_seconds": baseline_record["elapsed_seconds"],
            "parameter_count": baseline_record["parameter_count"],
            "checkpoint_sha256": sha256(checkpoint),
            "training_metrics_at_selected_checkpoint": train_metrics,
            "validation_metrics_at_selected_checkpoint": val_metrics,
        })

    for variant, feature in VARIANTS.items():
        for seed in SEEDS:
            run_dir = OUTPUT / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            train_variant({"torch": torch, "benchmark": benchmark,
                           "tracker_class": tracker_class},
                          seed, variant, feature, train, val, run_dir, device)
    print(f"Training/validation phase complete: {OUTPUT}")
    print("Shared test set was not opened.")


if __name__ == "__main__":
    main()
