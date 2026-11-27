#!/usr/bin/env python3
"""Train-only feature-normalization confirmation; validation pool only."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.paper_v4_mamba_improvement import run_screening_pilot as pilot
from radar_tracking.cli import benchmark_mamba as benchmark
from radar_tracking.config import simparams_class
from torch.utils.data import DataLoader, TensorDataset

CONFIG_PATH = PROJECT / "configs/paper_v4_mamba_improvement/features_train_only_normalization_five_seed_confirmation_v2.json"
PILOT_CONFIG_PATH = PROJECT / "configs/paper_v4_mamba_improvement/features_train_only_normalization_seed11.json"
BASELINE_CONFIG_PATH = PROJECT / "configs/paper_v4_mamba_improvement/baseline_reference.json"
OUTPUT = PROJECT / "artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2"
POOL = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_train_val.pt"
BASELINE_ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
EXPECTED_POOL_HASH = "4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733"
METRIC_KEYS = (
    "position_rmse_m", "velocity_rmse_mps", "acceleration_rmse_mps2",
    "six_state_mse_scaled_units",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def mean_sd(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "sample_sd": math.sqrt(variance)}


def train_candidate_for_seed(model, train, validation, device, seed_dir, training_config):
    """Pilot-identical training loop with the active seed recorded in logs."""
    dataset = TensorDataset(train["obs"], train["init"], train["state"])
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    checkpoint = seed_dir / "best_validation_checkpoint.pt"
    start_time = time.perf_counter()

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
        val_metrics, _ = pilot.evaluate(model, validation, device)
        val_mse = val_metrics["six_state_mse_scaled_units"]
        history.append({
            "epoch": epoch + 1,
            "train_mse_batch_mean": loss_sum / max(batches, 1),
            "validation_metrics": val_metrics,
        })
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
                f"{training_config['experiment']} seed={training_config['seed']} "
                f"epoch={epoch + 1:03d} val_mse={val_mse:.6f} "
                f"val_pos={val_metrics['position_rmse_m']:.2f} "
                f"val_vel={val_metrics['velocity_rmse_mps']:.2f} "
                f"val_acc={val_metrics['acceleration_rmse_mps2']:.2f}",
                flush=True,
            )
        if no_improve >= 25:
            break

    pilot.sync_cuda(device)
    training_seconds = time.perf_counter() - start_time
    if not checkpoint.is_file():
        raise RuntimeError("No validation-selected checkpoint was written")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_metrics, _ = pilot.evaluate(model, train, device)
    validation_metrics, _ = pilot.evaluate(model, validation, device)
    run_record = {
        "experiment": training_config["experiment"],
        "seed": training_config["seed"],
        "training_config": training_config,
        "epochs_run": len(history),
        "checkpoint_selection_epoch": best_epoch,
        "best_validation_six_state_mse": best_val,
        "training_seconds": training_seconds,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": pilot.sha256(checkpoint),
        "parameter_count": pilot.parameter_counts(model),
        "train_metrics_at_selected_checkpoint": train_metrics,
        "validation_metrics_at_selected_checkpoint": validation_metrics,
        "history": history,
        "test_accessed": False,
    }
    write_new(seed_dir / "run_metrics.json", run_record)
    return run_record


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    pilot_config = json.loads(PILOT_CONFIG_PATH.read_text())
    baseline_contract = json.loads(BASELINE_CONFIG_PATH.read_text())
    if sha256(PILOT_CONFIG_PATH) != config["pilot_config_sha256"]:
        raise RuntimeError("Seed-11 pilot config hash changed; refusing non-identical confirmation")
    if sha256(POOL) != EXPECTED_POOL_HASH or EXPECTED_POOL_HASH != config["data"]["train_validation_pool_sha256"]:
        raise RuntimeError("Fixed train/validation pool hash mismatch")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite results: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("This Mamba implementation requires CUDA")
    if config["seeds"] != baseline_contract["training"]["seeds"]:
        raise RuntimeError("Confirmation seeds do not match frozen paired baseline seeds")
    if any(config["model"].get(key) != value for key, value in pilot_config["model"].items()):
        raise RuntimeError("Model config differs from the completed pilot")
    pilot_features = pilot_config["feature_normalization"]
    confirmation_features = config["feature_normalization"]
    if confirmation_features["normalized_token_indices_zero_based"] != pilot_features["normalized_token_indices_zero_based"]:
        raise RuntimeError("Normalized feature indices differ from the completed pilot")
    if confirmation_features["preserved_unscaled_token_indices_zero_based"] != pilot_features["preserved_unscaled_token_indices_zero_based"]:
        raise RuntimeError("Preserved feature indices differ from the completed pilot")
    if confirmation_features["formula"] != pilot_features["formula"]:
        raise RuntimeError("Normalization formula differs from the completed pilot")
    training = config["training"]
    pilot_training = pilot_config["training"]
    training_pairs = {
        "batch_size": "batch_size", "optimizer": "optimizer",
        "learning_rate": "learning_rate", "weight_decay": "weight_decay",
        "max_epochs": "max_epochs", "early_stopping_patience": "early_stopping_patience",
        "min_delta": "min_delta", "gradient_clip_norm": "gradient_clip_norm",
        "loss": "loss", "checkpoint_selection": "checkpoint_selection",
    }
    for actual_key, pilot_key in training_pairs.items():
        if training[actual_key] != pilot_training[pilot_key]:
            raise RuntimeError(f"Training setting {actual_key} differs from the completed pilot")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "config.json").write_text(CONFIG_PATH.read_text(), encoding="utf-8")
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
        raise RuntimeError("Unexpected fixed train/validation tensor shape")

    device = torch.device("cuda")
    simparams = simparams_class()
    simparams.seq_len = 100
    seed_rows = []
    source_paths = [
        CONFIG_PATH, PILOT_CONFIG_PATH, BASELINE_CONFIG_PATH,
        Path(__file__).resolve(), Path(pilot.__file__).resolve(),
        Path(pilot.__file__).with_name("model_variants.py").resolve(),
        PROJECT / "src/radar_tracking/cli/benchmark_mamba.py",
        PROJECT / "src/radar_tracking/models/tracker_model.py",
        PROJECT / "src/radar_tracking/models/bimamba_encoder.py",
        PROJECT / "src/radar_tracking/models/prediction_head.py",
        PROJECT / "src/radar_tracking/data/data_generation.py",
    ]
    source_hashes = {str(path.relative_to(PROJECT)): sha256(path) for path in source_paths}

    for seed in config["seeds"]:
        seed = int(seed)
        pilot.SEED = seed
        seed_dir = OUTPUT / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=False)
        baseline_dir = BASELINE_ROOT / f"seed_{seed}"
        baseline_record = json.loads((baseline_dir / "run_metrics.json").read_text())
        baseline_checkpoint = baseline_dir / baseline_record["mamba"]["checkpoint"]
        baseline_hash = sha256(baseline_checkpoint)
        if baseline_hash != baseline_record["mamba"]["checkpoint_sha256"]:
            raise RuntimeError(f"Seed {seed}: baseline checkpoint hash mismatch")

        baseline_cfg = benchmark.MambaTrainConfig(**baseline_record["mamba"]["config"])
        benchmark.set_seed(seed)
        baseline_model = benchmark.build_mamba_model(simparams, baseline_cfg, device)
        baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
        baseline_metrics, _ = pilot.evaluate(baseline_model, validation, device)
        baseline_timing = pilot.timed_validation(baseline_model, validation, device)
        baseline_parameters = pilot.parameter_counts(baseline_model)

        feature_mean, feature_std, feature_stats = pilot.compute_train_feature_stats(
            baseline_cfg, simparams, train, device
        )
        pilot.write_new(seed_dir / "train_only_feature_statistics.json", feature_stats)
        model_config = pilot_config["model"]
        candidate = pilot.construct_candidate(
            "feature_norm", simparams, model_config, device, feature_mean, feature_std
        )
        training_config = {
            "experiment": config["experiment"],
            "seed": seed,
            "epochs_cap": pilot_config["training"]["max_epochs"],
            "batch_size": pilot_config["training"]["batch_size"],
            "optimizer": pilot_config["training"]["optimizer"],
            "learning_rate": pilot_config["training"]["learning_rate"],
            "weight_decay": pilot_config["training"]["weight_decay"],
            "patience": pilot_config["training"]["early_stopping_patience"],
            "min_delta": pilot_config["training"]["min_delta"],
            "gradient_clip_norm": pilot_config["training"]["gradient_clip_norm"],
            "loss": pilot_config["training"]["loss"],
            "checkpoint_selection": pilot_config["training"]["checkpoint_selection"],
            "model": model_config,
            "factor": "input feature normalization only; copied from completed seed-11 pilot",
            "feature_normalization": {
                "normalized_indices": config["feature_normalization"]["normalized_token_indices_zero_based"],
                "preserved_indices": config["feature_normalization"]["preserved_unscaled_token_indices_zero_based"],
                "std_floor": 1e-4,
                "statistics_source": "training split only, independently fit for this seed",
            },
        }
        run_record = train_candidate_for_seed(
            candidate, train, validation, device, seed_dir, training_config
        )
        candidate_timing = pilot.timed_validation(candidate, validation, device)
        candidate_metrics = run_record["validation_metrics_at_selected_checkpoint"]
        if run_record["parameter_count"]["total"] != config["model"]["parameter_count_expected"]:
            raise RuntimeError(f"Seed {seed}: unexpected parameter count")

        row = {
            "seed": seed,
            "baseline_checkpoint": str(baseline_checkpoint.relative_to(PROJECT)),
            "baseline_checkpoint_sha256": baseline_hash,
            "baseline_validation_metrics": baseline_metrics,
            "baseline_parameter_count": baseline_parameters,
            "candidate_validation_metrics": candidate_metrics,
            "candidate_parameter_count": run_record["parameter_count"],
            "checkpoint_selection_epoch": run_record["checkpoint_selection_epoch"],
            "stopping_epoch": run_record["epochs_run"],
            "best_validation_six_state_mse": run_record["best_validation_six_state_mse"],
            "training_seconds": run_record["training_seconds"],
            "baseline_validation_inference_timing": baseline_timing,
            "candidate_validation_inference_timing": candidate_timing,
            "candidate_checkpoint": "best_validation_checkpoint.pt",
            "candidate_checkpoint_sha256": run_record["checkpoint_sha256"],
            "feature_statistics_file": "train_only_feature_statistics.json",
            "feature_statistics_sha256": sha256(seed_dir / "train_only_feature_statistics.json"),
            "test_accessed": False,
        }
        pilot.write_new(seed_dir / "seed_summary.json", row)
        seed_rows.append(row)
        print(json.dumps({
            "seed": seed,
            "selection_epoch": row["checkpoint_selection_epoch"],
            "stopping_epoch": row["stopping_epoch"],
            "training_seconds": row["training_seconds"],
            "validation": {key: candidate_metrics[key] for key in METRIC_KEYS},
            "test_accessed": False,
        }), flush=True)

    aggregate = {"experiment": config["experiment"], "seeds": seed_rows, "test_accessed": False}
    aggregate["candidate_mean_sample_sd"] = {
        key: mean_sd([row["candidate_validation_metrics"][key] for row in seed_rows])
        for key in METRIC_KEYS
    }
    aggregate["baseline_mean_sample_sd"] = {
        key: mean_sd([row["baseline_validation_metrics"][key] for row in seed_rows])
        for key in METRIC_KEYS
    }
    aggregate["paired_difference_candidate_minus_baseline"] = {
        key: {
            "per_seed": [
                row["candidate_validation_metrics"][key] - row["baseline_validation_metrics"][key]
                for row in seed_rows
            ],
            "mean": sum(
                row["candidate_validation_metrics"][key] - row["baseline_validation_metrics"][key]
                for row in seed_rows
            ) / len(seed_rows),
            "sample_sd": mean_sd([
                row["candidate_validation_metrics"][key] - row["baseline_validation_metrics"][key]
                for row in seed_rows
            ])["sample_sd"],
        }
        for key in METRIC_KEYS
    }
    write_new(OUTPUT / "validation_aggregate.json", aggregate)

    position_ratio = aggregate["candidate_mean_sample_sd"]["position_rmse_m"]["mean"] / aggregate["baseline_mean_sample_sd"]["position_rmse_m"]["mean"] - 1.0
    velocity_ratio = aggregate["candidate_mean_sample_sd"]["velocity_rmse_mps"]["mean"] / aggregate["baseline_mean_sample_sd"]["velocity_rmse_mps"]["mean"] - 1.0
    qualifies = (
        position_ratio <= config["selection_rule"]["validation_guardrails"]["mean_position_rmse_max_relative_increase"]
        and velocity_ratio <= config["selection_rule"]["validation_guardrails"]["mean_velocity_rmse_max_relative_increase"]
    )
    frozen = {
        "experiment": config["experiment"],
        "status": "validation_confirmed_qualifies_for_final_test" if qualifies else "validation_not_confirmed_no_test_evaluation",
        "qualifies_for_final_test_evaluation": qualifies,
        "selection_rule": config["selection_rule"],
        "mean_position_relative_difference": position_ratio,
        "mean_velocity_relative_difference": velocity_ratio,
        "selected_configuration": str(CONFIG_PATH.relative_to(PROJECT)),
        "selected_configuration_sha256": sha256(CONFIG_PATH),
        "pilot_configuration_sha256": sha256(PILOT_CONFIG_PATH),
        "train_validation_pool_sha256": sha256(POOL),
        "source_file_sha256": source_hashes,
        "candidate_checkpoints": {
            str(row["seed"]): row["candidate_checkpoint_sha256"] for row in seed_rows
        },
        "baseline_checkpoints": {
            str(row["seed"]): row["baseline_checkpoint_sha256"] for row in seed_rows
        },
        "validation_aggregate_sha256": sha256(OUTPUT / "validation_aggregate.json"),
        "test_accessed": False,
    }
    write_new(OUTPUT / "validation_selection_manifest.json", frozen)
    hash_record = {
        "validation_selection_manifest_sha256": sha256(OUTPUT / "validation_selection_manifest.json"),
        "validation_aggregate_sha256": sha256(OUTPUT / "validation_aggregate.json"),
        "config_sha256": sha256(CONFIG_PATH),
    }
    write_new(OUTPUT / "frozen_validation_hashes.json", hash_record)
    print(json.dumps({
        "output": str(OUTPUT),
        "status": frozen["status"],
        "position_relative_difference": position_ratio,
        "velocity_relative_difference": velocity_ratio,
        "mean_sample_sd": aggregate["candidate_mean_sample_sd"],
        "paired_differences": aggregate["paired_difference_candidate_minus_baseline"],
        "test_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
