#!/usr/bin/env python3
"""Train loss-only Mamba ablations against the preserved five-seed baseline.

This runner reads the fixed train/validation pool and historical M0 checkpoints.
It deliberately does not open the shared test set. Test evaluation is a separate
post-selection step in evaluate_mamba_loss_ablation_v1.py.
"""

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

OUTPUT = PROJECT / "artifacts/results/synthetic/mamba_loss_ablation_v1"
BASELINE = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
POOL_PATH = BASELINE / "fixed_train_val.pt"
SEEDS = (11, 22, 33, 44, 55)
VARIANTS = {"M1_standardized": None, "M2_accel_w4": 4.0, "M2_accel_w8": 8.0}
PHYSICAL_SCALE = 1000.0
TIME_WINDOWS = [(i, min(i + 10, 100)) for i in range(0, 100, 10)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().tolist()]


def state_mse_from_sums(sums: torch.Tensor, count: int) -> torch.Tensor:
    return sums / max(count, 1)


def loss_from_state_mse(state_mse: torch.Tensor, variant: str,
                        state_std: torch.Tensor | None, accel_weight: float | None) -> torch.Tensor:
    if variant == "M0_original":
        return state_mse.mean()
    if variant == "M1_standardized":
        assert state_std is not None
        return (state_mse / state_std.square()).mean()
    pos = state_mse[[0, 3]].mean()
    vel = state_mse[[1, 4]].mean()
    acc = state_mse[[2, 5]].mean()
    assert accel_weight is not None
    return (pos + vel + accel_weight * acc) / (2.0 + accel_weight)


def prediction_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    # pred/target are [N, 6, T] in the training representation (all state
    # channels divided by 1,000); reported coordinate RMSE is physical units.
    err = pred - target
    state_mse = err.square().mean(dim=(0, 2))
    group = {
        "position": [0, 3],
        "velocity": [1, 4],
        "acceleration": [2, 5],
    }
    metrics: dict[str, Any] = {
        "state_order": ["x", "vx", "ax", "y", "vy", "ay"],
        "per_state_mse_scaled_units": finite_list(state_mse),
        "per_state_rmse_physical": finite_list(state_mse.sqrt() * PHYSICAL_SCALE),
        "per_state_bias_physical": finite_list(err.mean(dim=(0, 2)) * PHYSICAL_SCALE),
        "six_state_mse_scaled_units": float(err.square().mean().item()),
    }
    for name, indices in group.items():
        pair = err[:, indices, :]
        metrics[f"{name}_rmse_physical"] = float(
            (pair.square().sum(dim=1).mean().sqrt() * PHYSICAL_SCALE).item()
        )
    windows = []
    for start, end in TIME_WINDOWS:
        row: dict[str, Any] = {"start_step_1based": start + 1, "end_step_1based": end}
        for name, indices in group.items():
            pair = err[:, indices, start:end]
            row[f"{name}_rmse_physical"] = float(
                (pair.square().sum(dim=1).mean().sqrt() * PHYSICAL_SCALE).item()
            )
        row["six_state_rmse_physical"] = float(
            (err[:, :, start:end].square().mean().sqrt() * PHYSICAL_SCALE).item()
        )
        windows.append(row)
    metrics["temporal_windows"] = windows
    return metrics


def evaluate(model: torch.nn.Module, tensors: dict[str, torch.Tensor], device: torch.device,
             batch_size: int = 16) -> tuple[dict[str, Any], torch.Tensor]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, tensors["state"].shape[0], batch_size):
            end = min(start + batch_size, tensors["state"].shape[0])
            out = model(
                observations=tensors["obs"][start:end].to(device),
                init_state=tensors["init"][start:end].to(device),
            )["state_estimates"]
            predictions.append(out.cpu())
    pred = torch.cat(predictions, dim=0)
    return prediction_metrics(pred, tensors["state"]), pred


def train_one(api: dict[str, Any], seed: int, variant: str, accel_weight: float | None,
              train: dict[str, torch.Tensor], val: dict[str, torch.Tensor],
              state_std: torch.Tensor | None, run_dir: Path, device: torch.device) -> dict[str, Any]:
    torch = api["torch"]
    benchmark = api["benchmark"]
    config = benchmark.MambaTrainConfig(
        seed=seed, epochs=180, batch_size=16, eval_batch_size=16,
        learning_rate=1e-3, weight_decay=1e-4, patience=25, min_delta=1e-6,
        embed_dim=128, hidden_dim=256, num_layers=3, d_state=16, d_conv=4,
        expand=2, dropout=0.08, encoder_direction="forward",
        checkpoint_name="best_mamba_tracker.pt",
    )
    dataset = TensorDataset(train["obs"], train["init"], train["state"])
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        pin_memory=True, num_workers=0)
    benchmark.set_seed(seed)
    model = benchmark.build_mamba_model(api["simparams"], config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    state_std_dev = state_std.to(device) if state_std is not None else None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    checkpoint = run_dir / "best_mamba_tracker.pt"
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        train_sq_sum = torch.zeros(6, device=device)
        train_count = 0
        train_obj_sum = 0.0
        batch_count = 0
        for obs_batch, init_batch, state_batch in loader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            init_batch = init_batch.to(device, non_blocking=True)
            state_batch = state_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(observations=obs_batch, init_state=init_batch)["state_estimates"]
            err = pred - state_batch
            state_mse_batch = err.square().mean(dim=(0, 2))
            loss = loss_from_state_mse(state_mse_batch, variant, state_std_dev, accel_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sq_sum += err.detach().square().sum(dim=(0, 2))
            train_count += state_batch.shape[0] * state_batch.shape[2]
            train_obj_sum += float(loss.detach().item())
            batch_count += 1

        train_state_mse = state_mse_from_sums(train_sq_sum, train_count)
        val_metrics, val_pred = evaluate(model, val, device, config.eval_batch_size)
        val_err = val_pred - val["state"]
        val_state_mse = val_err.square().mean(dim=(0, 2))
        val_objective = float(loss_from_state_mse(
            val_state_mse.to(device), variant, state_std_dev, accel_weight).item())
        row = {
            "epoch": epoch + 1,
            "train_objective": train_obj_sum / max(batch_count, 1),
            "validation_objective": val_objective,
            "train_per_state_mse_scaled_units": finite_list(train_state_mse),
            "validation_per_state_mse_scaled_units": val_metrics["per_state_mse_scaled_units"],
            "train_per_state_rmse_physical": finite_list(train_state_mse.sqrt() * PHYSICAL_SCALE),
            "validation_per_state_rmse_physical": val_metrics["per_state_rmse_physical"],
            "validation_group_rmse_physical": {
                key: val_metrics[f"{key}_rmse_physical"]
                for key in ("position", "velocity", "acceleration")
            },
        }
        history.append(row)
        if val_objective < best_val - config.min_delta:
            best_val = val_objective
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint)
            no_improve = 0
        else:
            no_improve += 1
        print(f"{variant} seed={seed} epoch={epoch + 1:03d} "
              f"train={row['train_objective']:.6f} val={val_objective:.6f} "
              f"val_acc={val_metrics['acceleration_rmse_physical']:.2f}", flush=True)
        if no_improve >= config.patience:
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No selected checkpoint was produced: {checkpoint}")
    elapsed = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_metrics, _ = evaluate(model, train, device, config.eval_batch_size)
    val_metrics, _ = evaluate(model, val, device, config.eval_batch_size)
    payload = {
        "seed": seed,
        "variant": variant,
        "loss": {
            "kind": "standardized_state_mse" if variant == "M1_standardized" else "group_weighted_mse",
            "coordinate_state_std_scaled_units": finite_list(state_std) if state_std is not None else None,
            "position_weight": 1.0 if variant.startswith("M2") else None,
            "velocity_weight": 1.0 if variant.startswith("M2") else None,
            "acceleration_weight": accel_weight,
            "formula": (
                "mean_i(mean(error_i^2) / train_std_i^2)" if variant == "M1_standardized"
                else "(MSE_position + MSE_velocity + w_acc * MSE_acceleration) / (2 + w_acc)"
            ),
        },
        "config": {
            "epochs_cap": config.epochs,
            "actual_epochs": len(history),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "min_delta": config.min_delta,
            "embed_dim": config.embed_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "d_state": config.d_state,
            "d_conv": config.d_conv,
            "expand": config.expand,
            "dropout": config.dropout,
            "encoder_direction": config.encoder_direction,
        },
        "best_epoch": best_epoch,
        "best_validation_objective": best_val,
        "training_duration_seconds": elapsed,
        "optimizer_steps": sum(math.ceil(train["state"].shape[0] / config.batch_size)
                                for _ in range(len(history))),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "training_metrics_at_selected_checkpoint": train_metrics,
        "validation_metrics_at_selected_checkpoint": val_metrics,
        "history": history,
    }
    write_json_new(run_dir / "run_metrics.json", payload)
    return payload


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {OUTPUT}")
    if not POOL_PATH.is_file():
        raise FileNotFoundError(f"Fixed train/validation pool not found: {POOL_PATH}")

    from radar_tracking.cli import benchmark_mamba as benchmark
    from radar_tracking.config import simparams_class
    from radar_tracking.models.tracker_model import BiMambaTrackerModel

    api = {"torch": torch, "benchmark": benchmark, "simparams": simparams_class()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The existing Mamba implementation requires CUDA.")
    payload = torch.load(POOL_PATH, map_location="cpu", weights_only=True)
    train = {"state": payload["train"]["state"], "obs": payload["train"]["observations"],
             "init": payload["train"]["initial_state"]}
    val = {"state": payload["validation"]["state"], "obs": payload["validation"]["observations"],
           "init": payload["validation"]["initial_state"]}
    state_std = train["state"].std(dim=(0, 2), unbiased=False)
    if not torch.isfinite(state_std).all() or (state_std <= 0).any():
        raise ValueError("Training-only state standard deviations must be positive and finite")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    hashes = {
        "runner_sha256": sha256(Path(__file__).resolve()),
        "fixed_pool_sha256": sha256(POOL_PATH),
        "baseline_source_sha256": {},
        "baseline_mamba_checkpoint_sha256": {},
    }
    for relative in (
        "src/radar_tracking/cli/benchmark_mamba.py",
        "src/radar_tracking/models/tracker_model.py",
        "src/radar_tracking/models/bimamba_encoder.py",
        "src/radar_tracking/models/prediction_head.py",
    ):
        hashes["baseline_source_sha256"][relative] = sha256(PROJECT / relative)
    for seed in SEEDS:
        cp = BASELINE / f"seed_{seed}" / "best_mamba_tracker.pt"
        hashes["baseline_mamba_checkpoint_sha256"][str(cp.relative_to(PROJECT))] = sha256(cp)

    manifest = {
        "experiment": "mamba_loss_ablation_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": list(SEEDS),
        "training_pool": str(POOL_PATH.relative_to(PROJECT)),
        "train_count": len(train["state"]),
        "validation_count": len(val["state"]),
        "sequence_length": int(train["state"].shape[2]),
        "training_only_population_std_scaled_units": finite_list(state_std),
        "variants": {
            "M0_original": {"loss": "mean squared error over all six states", "source": "preserved historical checkpoints"},
            "M1_standardized": {"loss": "mean_i(mean(error_i^2) / training_std_i^2)"},
            "M2_accel_w4": {"group_weights": {"position": 1.0, "velocity": 1.0, "acceleration": 4.0}},
            "M2_accel_w8": {"group_weights": {"position": 1.0, "velocity": 1.0, "acceleration": 8.0}},
        },
        "test_accessed": False,
        "hashes": hashes,
    }
    write_json_new(OUTPUT / "manifest.json", manifest)

    # M0 validation/train metrics come from the preserved checkpoints only.
    m0_dir = OUTPUT / "M0_original"
    m0_dir.mkdir()
    for seed in SEEDS:
        run_dir = m0_dir / f"seed_{seed}"
        run_dir.mkdir()
        record = json.loads((BASELINE / f"seed_{seed}" / "run_metrics.json").read_text())
        mamba_record = record["mamba"]
        config_data = mamba_record["config"]
        config = benchmark.MambaTrainConfig(**config_data)
        benchmark.set_seed(seed)
        model = benchmark.build_mamba_model(api["simparams"], config, device)
        checkpoint = BASELINE / f"seed_{seed}" / "best_mamba_tracker.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        train_metrics, _ = evaluate(model, train, device)
        val_metrics, _ = evaluate(model, val, device)
        m0 = {
            "seed": seed,
            "variant": "M0_original",
            "source": "preserved historical checkpoint; not retrained",
            "best_epoch": mamba_record["best_epoch"],
            "training_duration_seconds": mamba_record["elapsed_seconds"],
            "checkpoint_sha256": sha256(checkpoint),
            "training_metrics_at_selected_checkpoint": train_metrics,
            "validation_metrics_at_selected_checkpoint": val_metrics,
        }
        write_json_new(run_dir / "run_metrics.json", m0)

    for variant, accel_weight in VARIANTS.items():
        for seed in SEEDS:
            run_dir = OUTPUT / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            train_one(api, seed, variant, accel_weight, train, val,
                      state_std if variant == "M1_standardized" else None,
                      run_dir, device)
    print(f"Training-only ablation complete: {OUTPUT}")
    print("The shared test set was not opened. Select variants from validation records before running the evaluator.")


if __name__ == "__main__":
    main()
