#!/usr/bin/env python3
"""Five-seed matched-length M0/M1 confirmation; validation only, never test."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
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

from experiments.paper_v4_long_sequence import stage5a_matched_length_screen as stage5a  # noqa: E402

OUTPUT_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1"
LENGTHS = (200, 400)
VARIANTS = ("M0", "M1")
SEEDS = (11, 22, 33, 44, 55)
METRICS = stage5a.METRICS


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return stage5a.sha256_file(path)


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_config_paths() -> list[Path]:
    return [
        PROJECT / f"configs/paper_v4_long_sequence/stage5B_T{length}_{variant}_seed{seed}.json"
        for length in LENGTHS for variant in VARIANTS for seed in SEEDS
    ]


def validate_frozen_prerequisites(config_paths: list[Path]) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, str]]:
    stage5a_root = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage5A_matched_length_screen_v1"
    stage5a_report = stage5a_root / "stage5a_screening_summary.json"
    stage5a_hash = stage5a.verify_sidecar(stage5a_report)
    stage5a_summary = json.loads(stage5a_report.read_text(encoding="utf-8"))
    if stage5a_summary["status"] != "PASS":
        raise ValueError("Stage 5A screening did not pass")
    if any(not row["M1_eligible_for_five_seed_confirmation"] for row in stage5a_summary["validation_screening_results"].values()):
        raise ValueError("A Stage 5A length did not qualify for confirmation")

    configs = {}
    config_hashes = {}
    for path in config_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing confirmation config: {path}")
        digest = stage5a.sha256_file(path)
        config = json.loads(path.read_text(encoding="utf-8"))
        length, variant, seed = config["sequence_length"], config["model"]["variant"], config["training"]["confirmation_seed"]
        base_path = PROJECT / f"configs/paper_v4_long_sequence/stage5A_T{length}_{variant}_seed101.json"
        base_hash = stage5a.sha256_file(base_path)
        if config["configuration_provenance"]["stage5A_config_sha256"] != base_hash:
            raise ValueError(f"T={length} {variant} seed {seed} base Stage 5A config changed")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for key in ("embed_dim", "hidden_dim", "num_layers", "d_state", "d_conv", "expand", "dropout", "encoder_direction", "parameter_count_expected", "feature_normalization"):
            if config["model"].get(key) != base["model"].get(key):
                raise ValueError(f"Stage 5B model setting {key} differs from frozen Stage 5A config")
        for key in ("batch_size", "optimizer", "learning_rate", "weight_decay", "loss", "gradient_clip_norm", "max_epochs", "early_stopping_patience", "min_delta", "checkpoint_selection", "sequence_truncation", "chunked_training", "batch_size_fallback"):
            if config["training"].get(key) != base["training"].get(key):
                raise ValueError(f"Stage 5B training setting {key} differs from frozen Stage 5A config")
        if config["data"] != base["data"]:
            raise ValueError(f"Stage 5B dataset contract differs from Stage 5A at {path}")
        expected_seed = seed + (2000 if length == 200 else 4000)
        if config["training"]["initialization_seed"] != expected_seed:
            raise ValueError(f"Unexpected initialization seed in {path}")
        if config["training"]["training_random_seed"] != expected_seed + 100000:
            raise ValueError(f"Unexpected training seed in {path}")
        if config["training"]["dataloader_shuffle_seed"] != expected_seed + 200000:
            raise ValueError(f"Unexpected shuffle seed in {path}")
        key = (length, variant, seed)
        if key in configs:
            raise ValueError(f"Duplicate confirmation config for {key}")
        configs[key] = config
        config_hashes[str(path.relative_to(PROJECT))] = digest
    expected = {(length, variant, seed) for length in LENGTHS for variant in VARIANTS for seed in SEEDS}
    if set(configs) != expected:
        raise ValueError("Confirmation config matrix is incomplete")

    source_hashes = {name: stage5a.sha256_file(PROJECT / name) for name in stage5a.SOURCE_FILES}
    source_hashes["experiments/paper_v4_long_sequence/stage5b_matched_length_confirmation.py"] = stage5a.sha256_file(Path(__file__).resolve())
    source_hashes["experiments/paper_v4_long_sequence/stage5a_matched_length_screen.py"] = stage5a.sha256_file(Path(stage5a.__file__).resolve())
    source_hashes["docs/reports/paper_v4_mamba_improvement_plan.md"] = stage5a.sha256_file(PROJECT / "docs/reports/paper_v4_mamba_improvement_plan.md")

    datasets = {}
    for length in LENGTHS:
        datasets[length], _ = stage5a.load_training_splits(length)
        datasets[length]["length"] = length
        datasets[length]["splits"] = datasets[length].pop("split_records")
    return {"summary": stage5a_summary, "report_sha256": stage5a_hash, "configs": configs}, datasets, {**source_hashes, **{f"config::{k}": v for k, v in config_hashes.items()}}


def verify_run_inputs(config_path: Path, config_hash: str, source_hashes: dict[str, str], data: dict[str, Any]) -> None:
    if stage5a.sha256_file(config_path) != config_hash:
        raise ValueError(f"confirmation config hash changed before run: {config_path}")
    for name, expected in source_hashes.items():
        if name.startswith("config::"):
            continue
        if stage5a.sha256_file(PROJECT / name) != expected:
            raise ValueError(f"source hash changed before run: {name}")
    length = data["length"]
    bundle = PROJECT / data["bundle_directory"]
    if stage5a.verify_sidecar(bundle / "dataset_manifest.json") != data["bundle_manifest_sha256"]:
        raise ValueError(f"T={length} dataset manifest changed before run")
    for split in ("train", "validation"):
        row = data["splits"][split]
        if stage5a.verify_sidecar(PROJECT / row["split_manifest"]) != row["split_manifest_sha256"]:
            raise ValueError(f"T={length} {split} split manifest changed before run")
        if stage5a.sha256_file(PROJECT / row["split_file"]) != row["split_file_sha256"]:
            raise ValueError(f"T={length} {split} tensor file changed before run")


def train_one(config_path: Path, config: dict[str, Any], config_hash: str, data: dict[str, Any], device: torch.device, environment: dict[str, Any], source_hashes: dict[str, str]) -> dict[str, Any]:
    length, variant, seed = config["sequence_length"], config["model"]["variant"], config["training"]["confirmation_seed"]
    verify_run_inputs(config_path, config_hash, source_hashes, data)
    run_dir = PROJECT / config["outputs"]["run_directory"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite confirmation run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_copy = run_dir / "config.json"
    config_copy.write_bytes(config_path.read_bytes())
    config_copy.with_name("config.json.sha256").write_text(f"{config_hash}  config.json\n", encoding="ascii")

    dataset_record = {
        "bundle_directory": data["bundle_directory"],
        "bundle_manifest": data["bundle_manifest"],
        "bundle_manifest_sha256": data["bundle_manifest_sha256"],
        "train": data["splits"]["train"],
        "validation": data["splits"]["validation"],
        "test_file_opened": False,
    }
    record: dict[str, Any] = {
        "status": "RUNNING",
        "variant": variant,
        "sequence_length": length,
        "confirmation_seed": seed,
        "screening_seed_101_config_sha256": config["configuration_provenance"]["stage5A_config_sha256"],
        "config_path": str(config_path.relative_to(PROJECT)),
        "config_sha256": config_hash,
        "dataset": dataset_record,
        "training_config": config["training"],
        "model_config": config["model"],
        "parameter_count_expected": 619014,
        "environment": {**environment, "gpu_free_memory_bytes_before_run": int(torch.cuda.mem_get_info(device)[0])},
        "source_hashes": {k: v for k, v in source_hashes.items() if not k.startswith("config::")},
        "test_accessed": False,
        "test_files_opened": False,
        "epoch_history": [],
    }
    stage5a.write_json(run_dir / "run_summary.json", record)
    checkpoint_path = run_dir / "best_validation_checkpoint.pt"
    try:
        stats_detail = None
        stats_hash = None
        stats_started = time.perf_counter()
        if variant == "M1":
            mean, std, stats_detail = stage5a.fit_feature_statistics(data["train"], config)
            stats_detail["confirmation_seed"] = seed
            stats_detail["train_split_sha256"] = data["splits"]["train"]["split_file_sha256"]
            stats_hash = stage5a.write_json(run_dir / "train_only_feature_statistics.json", stats_detail)
        stats_seconds = time.perf_counter() - stats_started

        stage5a.set_seed(config["training"]["initialization_seed"])
        model = stage5a.build_model(config, stats_detail, device)
        initial_parameter_hash = parameter_hash(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        train_set = TensorDataset(data["train"]["observations"], data["train"]["initial_state"], data["train"]["state"])
        stage5a.set_seed(config["training"]["training_random_seed"])
        best_val = float("inf")
        best_epoch = 0
        no_improve = 0
        epoch_log_path = run_dir / "epoch_log.jsonl"
        total_train_start = time.perf_counter()
        total_validation_seconds = 0.0

        for epoch in range(config["training"]["max_epochs"]):
            model.train()
            order_generator = torch.Generator(device="cpu")
            order_generator.manual_seed(config["training"]["dataloader_shuffle_seed"] + epoch)
            order = torch.randperm(len(train_set), generator=order_generator).tolist()
            order_hash = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
            loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], sampler=order, pin_memory=True, num_workers=0)
            torch.cuda.reset_peak_memory_stats(device)
            stage5a.sync_cuda(device)
            train_epoch_start = time.perf_counter()
            loss_sum, sample_count = 0.0, 0
            for observations, init_state, truth in loader:
                observations = observations.to(device, non_blocking=True)
                init_state = init_state.to(device, non_blocking=True)
                truth = truth.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = model(observations=observations, init_state=init_state)["state_estimates"]
                loss = F.mse_loss(output, truth, reduction="mean")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"]["gradient_clip_norm"])
                optimizer.step()
                loss_sum += float(loss.detach().item()) * observations.shape[0]
                sample_count += observations.shape[0]
                del observations, init_state, truth, output, loss
            stage5a.sync_cuda(device)
            train_seconds = time.perf_counter() - train_epoch_start
            train_memory = stage5a.device_memory()

            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            stage5a.sync_cuda(device)
            validation_started = time.perf_counter()
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(data["validation"]["state"]), 16):
                    end = min(start + 16, len(data["validation"]["state"]))
                    output = model(
                        observations=data["validation"]["observations"][start:end].to(device, non_blocking=True),
                        init_state=data["validation"]["initial_state"][start:end].to(device, non_blocking=True),
                    )
                    estimate = output["state_estimates"]
                    if tuple(estimate.shape) != (end - start, 6, length):
                        raise ValueError(f"validation output shape mismatch at epoch {epoch + 1}")
                    if not torch.isfinite(estimate).all():
                        raise FloatingPointError(f"non-finite validation output at epoch {epoch + 1}")
                    predictions.append(estimate.detach().cpu())
                    del output, estimate
            stage5a.sync_cuda(device)
            validation_seconds = time.perf_counter() - validation_started
            total_validation_seconds += validation_seconds
            validation_memory = stage5a.device_memory()
            validation_metrics = stage5a.metric_values(torch.cat(predictions), data["validation"]["state"])
            train_mse = loss_sum / sample_count
            if not math.isfinite(train_mse):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
            val_mse = validation_metrics["six_state_mse_scaled_units"]
            row = {
                "epoch": epoch + 1,
                "train_mse": train_mse,
                "validation_metrics": validation_metrics,
                "training_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "training_peak_memory": train_memory,
                "validation_peak_memory": validation_memory,
                "dataloader_order_sha256": order_hash,
            }
            record["epoch_history"].append(row)
            with epoch_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if val_mse < best_val - config["training"]["min_delta"]:
                best_val = val_mse
                best_epoch = epoch + 1
                checkpoint_hash = stage5a.save_best_checkpoint(
                    checkpoint_path, model, best_epoch, best_val, config, config_hash, stats_hash
                )
                no_improve = 0
            else:
                no_improve += 1
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"{variant} T={length} seed={seed} epoch={epoch + 1:03d} "
                    f"train_mse={train_mse:.6g} val_mse={val_mse:.6g} "
                    f"pos={validation_metrics['position_rmse_m']:.2f} vel={validation_metrics['velocity_rmse_mps']:.2f} "
                    f"acc={validation_metrics['acceleration_rmse_mps2']:.2f}", flush=True
                )
            if no_improve >= config["training"]["early_stopping_patience"]:
                break

        stage5a.sync_cuda(device)
        training_wall_seconds = time.perf_counter() - total_train_start
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        stage5a.sync_cuda(device)
        selected_metrics, selected_validation_seconds = stage5a.evaluate_validation(model, data["validation"], device)
        selected_validation_memory = stage5a.device_memory()
        checkpoint_hash = stage5a.sha256_file(checkpoint_path)
        if abs(selected_metrics["six_state_mse_scaled_units"] - best_val) > 1e-7:
            raise RuntimeError("selected checkpoint validation metric does not reproduce")
        record.update({
            "status": "COMPLETED",
            "epochs_run": len(record["epoch_history"]),
            "selected_epoch": best_epoch,
            "selected_validation_metrics": selected_metrics,
            "best_validation_six_state_mse_scaled_units": best_val,
            "training_wall_seconds": training_wall_seconds,
            "feature_statistics_fit_seconds": stats_seconds,
            "sum_epoch_training_seconds": sum(item["training_seconds"] for item in record["epoch_history"]),
            "sum_epoch_validation_seconds": total_validation_seconds,
            "selected_checkpoint_validation_seconds": selected_validation_seconds,
            "selected_checkpoint_validation_peak_memory": selected_validation_memory,
            "training_peak_allocated_bytes_max_over_epochs": max(item["training_peak_memory"]["peak_allocated_bytes"] for item in record["epoch_history"]),
            "training_peak_reserved_bytes_max_over_epochs": max(item["training_peak_memory"]["peak_reserved_bytes"] for item in record["epoch_history"]),
            "validation_peak_allocated_bytes_max_over_epochs": max(item["validation_peak_memory"]["peak_allocated_bytes"] for item in record["epoch_history"]),
            "validation_peak_reserved_bytes_max_over_epochs": max(item["validation_peak_memory"]["peak_reserved_bytes"] for item in record["epoch_history"]),
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "initial_trainable_weights_sha256": initial_parameter_hash,
            "initial_weights_seed_policy": "paired M0/M1 initialization seed at this length and confirmation seed",
            "data_loader_order_policy": "same randperm seed per epoch for paired M0/M1; order hash recorded per epoch",
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT)),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "feature_statistics_path": str((run_dir / "train_only_feature_statistics.json").relative_to(PROJECT)) if stats_hash else None,
            "feature_statistics_sha256": stats_hash,
            "strict_state_dict_reload": True,
            "test_accessed": False,
            "test_files_opened": False,
        })
        stage5a.write_json(run_dir / "run_summary.json", record)
        print(f"Completed {variant} T={length} seed={seed} selected_epoch={best_epoch}", flush=True)
        del model, optimizer, loader
        torch.cuda.empty_cache()
        return record
    except Exception as exc:
        stage5a.sync_cuda(device)
        record.update({
            "status": "FAILED",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "epochs_run": len(record["epoch_history"]),
            "test_accessed": False,
            "test_files_opened": False,
        })
        stage5a.write_json(run_dir / "run_summary.json", record)
        print(f"FAILED {variant} T={length} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
        return record


def sample_summary(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {"mean": mean, "sample_sd": sd}


def aggregate(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    overall_rows = []
    paired_rows = []
    window_rows = []
    guardrails = {}
    for length in LENGTHS:
        selected = [row for row in results if row["sequence_length"] == length and row["status"] == "COMPLETED"]
        by_variant = {variant: [row for row in selected if row["variant"] == variant] for variant in VARIANTS}
        for variant in VARIANTS:
            for metric in METRICS:
                values = [row["selected_validation_metrics"][metric] for row in by_variant[variant]]
                overall_rows.append({"sequence_length": length, "variant": variant, "metric": metric, **sample_summary(values), "seed_count": len(values)})
        if len(by_variant["M0"]) == len(by_variant["M1"]) == 5:
            paired_guard = []
            for seed in SEEDS:
                m0 = next(row for row in by_variant["M0"] if row["confirmation_seed"] == seed)
                m1 = next(row for row in by_variant["M1"] if row["confirmation_seed"] == seed)
                differences = {}
                changes = {}
                for metric in METRICS:
                    v0 = m0["selected_validation_metrics"][metric]
                    v1 = m1["selected_validation_metrics"][metric]
                    differences[metric] = v1 - v0
                    changes[metric] = (v1 / v0 - 1.0) if v0 else None
                paired_rows.append({"sequence_length": length, "seed": seed, **{f"{metric}_difference_M1_minus_M0": differences[metric] for metric in METRICS}, **{f"{metric}_relative_change": changes[metric] for metric in METRICS}})
                paired_guard.append({"seed": seed, "position_relative_change": changes["position_rmse_m"], "velocity_relative_change": changes["velocity_rmse_mps"], "passes_position": changes["position_rmse_m"] <= 0.01, "passes_velocity": changes["velocity_rmse_mps"] <= 0.03})
            paired_means = {}
            for metric in METRICS:
                values = [row[f"{metric}_difference_M1_minus_M0"] for row in paired_rows if row["sequence_length"] == length]
                paired_means[metric] = sample_summary(values)
            m0means = {metric: sample_summary([row["selected_validation_metrics"][metric] for row in by_variant["M0"]])["mean"] for metric in METRICS}
            m1means = {metric: sample_summary([row["selected_validation_metrics"][metric] for row in by_variant["M1"]])["mean"] for metric in METRICS}
            mean_position_change = m1means["position_rmse_m"] / m0means["position_rmse_m"] - 1.0
            mean_velocity_change = m1means["velocity_rmse_mps"] / m0means["velocity_rmse_mps"] - 1.0
            guardrails[str(length)] = {
                "frozen_limits": {"position_max_relative_increase": 0.01, "velocity_max_relative_increase": 0.03},
                "mean_position_relative_change": mean_position_change,
                "mean_velocity_relative_change": mean_velocity_change,
                "mean_position_guardrail_pass": mean_position_change <= 0.01,
                "mean_velocity_guardrail_pass": mean_velocity_change <= 0.03,
                "mean_guardrails_pass": mean_position_change <= 0.01 and mean_velocity_change <= 0.03,
                "per_seed_outcomes": paired_guard,
                "per_seed_guardrail_pass_count": sum(x["passes_position"] and x["passes_velocity"] for x in paired_guard),
                "M0_and_M1_initial_weights_equal_per_paired_seed": all(next(r for r in by_variant["M0"] if r["confirmation_seed"] == seed)["initial_trainable_weights_sha256"] == next(r for r in by_variant["M1"] if r["confirmation_seed"] == seed)["initial_trainable_weights_sha256"] for seed in SEEDS),
                "M0_and_M1_epoch_orders_equal_for_common_epochs_per_paired_seed": all(
                    [x["dataloader_order_sha256"] for x in next(r for r in by_variant["M0"] if r["confirmation_seed"] == seed)["epoch_history"][:min(len(next(r for r in by_variant["M0"] if r["confirmation_seed"] == seed)["epoch_history"]), len(next(r for r in by_variant["M1"] if r["confirmation_seed"] == seed)["epoch_history"]))]]
                    == [x["dataloader_order_sha256"] for x in next(r for r in by_variant["M1"] if r["confirmation_seed"] == seed)["epoch_history"][:min(len(next(r for r in by_variant["M0"] if r["confirmation_seed"] == seed)["epoch_history"]), len(next(r for r in by_variant["M1"] if r["confirmation_seed"] == seed)["epoch_history"]))]]
                    for seed in SEEDS
                ),
                "paired_epoch_order_lengths": {
                    str(seed): {
                        "M0_epochs": len(next(r for r in by_variant["M0"] if r["confirmation_seed"] == seed)["epoch_history"]),
                        "M1_epochs": len(next(r for r in by_variant["M1"] if r["confirmation_seed"] == seed)["epoch_history"]),
                    } for seed in SEEDS
                },
                "paired_differences_are_descriptive_only": True,
            }
            for variant in VARIANTS:
                for start_index in range(length // 10):
                    start = start_index * 10 + 1
                    end = start_index * 10 + 10
                    for metric in METRICS:
                        values = []
                        for run in by_variant[variant]:
                            window = run["selected_validation_metrics"]["temporal_windows"][start_index]
                            values.append(window[metric])
                        row = {"sequence_length": length, "variant": variant, "start_step_1based": start, "end_step_1based": end, "metric": metric, **sample_summary(values), "seed_count": len(values)}
                        window_rows.append(row)
    return overall_rows, paired_rows, window_rows, guardrails


def make_plots(results: list[dict[str, Any]], output: Path) -> dict[str, str]:
    plot_dir = output / "plots"
    plot_dir.mkdir()
    hashes = {}
    for length in LENGTHS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for variant, color in (("M0", "#2458A6"), ("M1", "#D14A32")):
            runs = [r for r in results if r["sequence_length"] == length and r["variant"] == variant and r["status"] == "COMPLETED"]
            max_epochs = max(len(r["epoch_history"]) for r in runs)
            epochs = range(1, max_epochs + 1)
            train_means, train_sds, val_means, val_sds, xs = [], [], [], [], []
            for epoch in epochs:
                train = [r["epoch_history"][epoch - 1]["train_mse"] for r in runs if len(r["epoch_history"]) >= epoch]
                val = [r["epoch_history"][epoch - 1]["validation_metrics"]["six_state_mse_scaled_units"] for r in runs if len(r["epoch_history"]) >= epoch]
                xs.append(epoch)
                train_means.append(float(np.mean(train)))
                train_sds.append(float(np.std(train, ddof=1)) if len(train) > 1 else 0.0)
                val_means.append(float(np.mean(val)))
                val_sds.append(float(np.std(val, ddof=1)) if len(val) > 1 else 0.0)
            for ax, means, sds, title in ((axes[0], train_means, train_sds, "Training MSE"), (axes[1], val_means, val_sds, "Validation six-state MSE")):
                ax.plot(xs, means, color=color, label=variant)
                ax.fill_between(xs, np.asarray(means) - sds, np.asarray(means) + sds, color=color, alpha=0.15)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Scaled-state MSE")
                ax.set_title(title)
                ax.grid(alpha=0.25)
                ax.legend()
        fig.suptitle(f"Five-seed matched-length training: T={length}")
        fig.tight_layout()
        path = plot_dir / f"training_validation_curves_T{length}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        hashes[str(path.relative_to(PROJECT))] = stage5a.sha256_file(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics_labels = (("position_rmse_m", "Position RMSE (m)"), ("velocity_rmse_mps", "Velocity RMSE (m/s)"), ("acceleration_rmse_mps2", "Acceleration RMSE (m/s²)"), ("six_state_mse_scaled_units", "Six-state MSE"))
    for ax, (metric, label) in zip(axes.flat, metrics_labels):
        x = np.arange(2)
        width = 0.34
        for shift, variant, color in ((-width/2, "M0", "#2458A6"), (width/2, "M1", "#D14A32")):
            means, sds = [], []
            for length in LENGTHS:
                vals = [r["selected_validation_metrics"][metric] for r in results if r["sequence_length"] == length and r["variant"] == variant and r["status"] == "COMPLETED"]
                agg = sample_summary(vals)
                means.append(agg["mean"])
                sds.append(agg["sample_sd"])
            ax.bar(x + shift, means, width=width, yerr=sds, capsize=3, color=color, label=variant)
        ax.set_xticks(x, ["T=200", "T=400"])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    path = plot_dir / "five_seed_validation_metrics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    hashes[str(path.relative_to(PROJECT))] = stage5a.sha256_file(path)
    return hashes


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 5B output root: {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by installed Mamba implementation")
    config_paths = run_config_paths()
    prerequisites, datasets, source_and_config_hashes = validate_frozen_prerequisites(config_paths)
    config_hashes = {str(path.relative_to(PROJECT)): stage5a.sha256_file(path) for path in config_paths}
    source_hashes = {key: value for key, value in source_and_config_hashes.items() if not key.startswith("config::")}
    for path in config_paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        data = datasets[config["sequence_length"]]
        verify_run_inputs(path, config_hashes[str(path.relative_to(PROJECT))], source_hashes, data)
        run_dir = PROJECT / config["outputs"]["run_directory"]
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite confirmation run: {run_dir}")
    hardware = stage5a.hardware_record()
    environment = {**hardware, "device": "cuda:0", "cuda_device_index": 0, "seed_determinism": "Python/NumPy/CPU Torch/CUDA Torch seeded; cuDNN deterministic=True, benchmark=False"}
    data_records = []
    for length in LENGTHS:
        d = datasets[length]
        data_records.append({"length": length, "bundle_directory": d["bundle_directory"], "bundle_manifest_sha256": d["bundle_manifest_sha256"], "splits": d["splits"], "test_file_opened": False})
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    results = []
    for path in config_paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        length = config["sequence_length"]
        result = train_one(path, config, config_hashes[str(path.relative_to(PROJECT))], datasets[length], torch.device("cuda:0"), environment, source_hashes)
        results.append(result)

    overall_rows, paired_rows, window_aggregate_rows, guardrails = aggregate(results)
    per_seed_rows = []
    validation_window_rows = []
    for run in results:
        basic = {
            "sequence_length": run["sequence_length"],
            "variant": run["variant"],
            "seed": run["confirmation_seed"],
            "status": run["status"],
            "selected_epoch": run.get("selected_epoch"),
            "epochs_run": run.get("epochs_run"),
            "training_wall_seconds": run.get("training_wall_seconds"),
            "selected_checkpoint_validation_seconds": run.get("selected_checkpoint_validation_seconds"),
            "parameter_count": run["parameter_count_expected"],
            "config_sha256": run["config_sha256"],
            "checkpoint_sha256": run.get("checkpoint_sha256"),
            "train_split_sha256": run["dataset"]["train"]["split_file_sha256"],
            "validation_split_sha256": run["dataset"]["validation"]["split_file_sha256"],
            "test_accessed": False,
        }
        basic.update({f"validation_{metric}": run.get("selected_validation_metrics", {}).get(metric) for metric in METRICS})
        per_seed_rows.append(basic)
        for window in run.get("selected_validation_metrics", {}).get("temporal_windows", []):
            validation_window_rows.append({"sequence_length": run["sequence_length"], "variant": run["variant"], "seed": run["confirmation_seed"], **window})
    csv_hashes = {
        "five_seed_results.csv": write_csv(OUTPUT_ROOT / "five_seed_results.csv", per_seed_rows),
        "validation_window_metrics.csv": write_csv(OUTPUT_ROOT / "validation_window_metrics.csv", validation_window_rows),
        "aggregate_validation_results.csv": write_csv(OUTPUT_ROOT / "aggregate_validation_results.csv", overall_rows),
        "paired_seed_differences.csv": write_csv(OUTPUT_ROOT / "paired_seed_differences.csv", paired_rows),
        "validation_window_aggregate.csv": write_csv(OUTPUT_ROOT / "validation_window_aggregate.csv", window_aggregate_rows),
    }
    plot_hashes = make_plots(results, OUTPUT_ROOT)

    selection = {
        "stage": "Phase 3 Stage 5B frozen validation checkpoint-selection manifest",
        "status": "FROZEN_BEFORE_TEST_ACCESS" if all(r["status"] == "COMPLETED" for r in results) else "INCOMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "selection_metric": "minimum validation six-state MSE, min_delta=1e-6, early stopping patience=25",
        "screening_guardrails": "Stage 5A criteria: M1 relative validation position RMSE increase <=1%; velocity RMSE increase <=3% versus same-length M0; applied to paired means and per-seed outcomes descriptively.",
        "dataset_records": data_records,
        "selected_checkpoints": [
            {"sequence_length": r["sequence_length"], "variant": r["variant"], "seed": r["confirmation_seed"], "status": r["status"], "selected_epoch": r.get("selected_epoch"), "validation_metrics": r.get("selected_validation_metrics"), "checkpoint_path": r.get("checkpoint_path"), "checkpoint_sha256": r.get("checkpoint_sha256"), "checkpoint_size_bytes": r.get("checkpoint_size_bytes"), "config_path": r["config_path"], "config_sha256": r["config_sha256"], "feature_statistics_sha256": r.get("feature_statistics_sha256"), "initial_trainable_weights_sha256": r.get("initial_trainable_weights_sha256")} for r in results
        ],
        "paired_initialization_and_loader_order": {str(length): {"identical_initial_trainable_weights": guardrails.get(str(length), {}).get("M0_and_M1_initial_weights_equal_per_paired_seed"), "identical_epoch_loader_orders_for_common_epochs": guardrails.get(str(length), {}).get("M0_and_M1_epoch_orders_equal_for_common_epochs_per_paired_seed")} for length in LENGTHS},
        "selection_manifest_sha256": "detached sidecar",
    }
    selection_hash = stage5a.write_json(OUTPUT_ROOT / "checkpoint_selection_manifest.json", selection)

    summary = {
        "stage": "Phase 3 Stage 5B five-seed matched-length confirmation",
        "status": "PASS" if all(r["status"] == "COMPLETED" for r in results) else "PARTIAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"training": True, "fine_tuning": False, "test_accessed": False, "test_files_opened": False, "test_performance_evaluated": False},
        "seeds": list(SEEDS),
        "lengths": list(LENGTHS),
        "expected_run_count": 20,
        "completed_run_count": sum(r["status"] == "COMPLETED" for r in results),
        "config_hashes": config_hashes,
        "stage5A_summary_sha256": prerequisites["report_sha256"],
        "dataset_integrity": data_records,
        "hardware": hardware,
        "source_hashes": source_hashes,
        "run_summaries": results,
        "aggregate_validation_results": overall_rows,
        "paired_seed_differences_M1_minus_M0": paired_rows,
        "validation_window_aggregate": window_aggregate_rows,
        "validation_guardrails": guardrails,
        "checkpoint_selection_manifest": {"path": str((OUTPUT_ROOT / "checkpoint_selection_manifest.json").relative_to(PROJECT)), "sha256": selection_hash},
        "output_artifact_hashes": {**csv_hashes, "plots": plot_hashes},
        "paired_differences_are_descriptive_only": True,
    }
    summary_hash = stage5a.write_json(OUTPUT_ROOT / "stage5b_confirmation_summary.json", summary)

    lines = [
        "# Phase 3 Stage 5B: Five-Seed Matched-Length Confirmation",
        "",
        f"**Status:** {summary['status']} ({summary['completed_run_count']}/20 runs completed)",
        "",
        "All runs used the frozen Stage 5A model/training settings. Paired M0/M1 seeds shared initialization, training/dropout, and epoch-wise data ordering at each length; initial trainable-weight hashes and loader-order hashes were compared. M1 means/stds were fit separately per seed using only that seed's training split. Test tensors were not opened or evaluated.",
        "",
        "## Validation metrics",
        "",
        "Values are mean ± sample SD over five seeds.",
        "",
        "| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    lookup = {(r["sequence_length"], r["variant"], r["metric"]): r for r in overall_rows}
    for length in LENGTHS:
        for variant in VARIANTS:
            values = []
            for metric in METRICS:
                row = lookup.get((length, variant, metric))
                values.append(f"{row['mean']:.6g} ± {row['sample_sd']:.3g}" if row else "incomplete")
            lines.append(f"| {length} | {variant} | " + " | ".join(values) + " |")
    lines.extend(["", "## Paired M1 − M0 differences", "", "Negative values favor M1. Differences are descriptive; no inferential significance is claimed.", "", "| T | Metric | Mean difference | Sample SD | Mean relative change |", "|---:|---|---:|---:|---:|"])
    for length in LENGTHS:
        for metric in METRICS:
            row = lookup.get((length, "M1", metric))
            paired = [x for x in paired_rows if x["sequence_length"] == length]
            if row and len(paired) == 5:
                diffs = [x[f"{metric}_difference_M1_minus_M0"] for x in paired]
                rel = [x[f"{metric}_relative_change"] for x in paired]
                diff_sum, rel_sum = sample_summary(diffs), sample_summary(rel)
                lines.append(f"| {length} | {metric} | {diff_sum['mean']:.6g} | {diff_sum['sample_sd']:.3g} | {rel_sum['mean']:+.2%} ± {rel_sum['sample_sd']:.2%} |")
    lines.extend(["", "## Frozen validation guardrails", "", "| T | Mean position change | Mean velocity change | Mean guardrails | Paired seeds passing | Initial weights/order identical |", "|---:|---:|---:|---|---:|---|"])
    for length in LENGTHS:
        g = guardrails.get(str(length), {})
        lines.append(
            f"| {length} | {g.get('mean_position_relative_change', float('nan')):+.2%} | {g.get('mean_velocity_relative_change', float('nan')):+.2%} | "
            f"{'PASS' if g.get('mean_guardrails_pass') else 'FAIL/INCOMPLETE'} | {g.get('per_seed_guardrail_pass_count', 0)}/5 | "
            f"{g.get('M0_and_M1_initial_weights_equal_per_paired_seed', False)}/{g.get('M0_and_M1_epoch_orders_equal_for_common_epochs_per_paired_seed', False)} |"
        )
    lines.extend([
        "",
        "## Artifacts and resources",
        "",
        f"GPU: {hardware['gpu_name']}; CUDA {hardware['cuda_runtime_version']}; PyTorch {hardware['pytorch_version']}; Mamba {hardware['mamba_ssm_version']}. Per-run training wall time, per-epoch training/validation times, allocated/reserved GPU peaks, checkpoint/config hashes, and train-only M1 statistics are in the JSON and run summaries.",
        "",
        "- `five_seed_results.csv`: all 20 selected-checkpoint rows.",
        "- `validation_window_metrics.csv`: per-seed selected-checkpoint metrics by 10-step interval.",
        "- `aggregate_validation_results.csv`, `paired_seed_differences.csv`, and `validation_window_aggregate.csv`.",
        "- `checkpoint_selection_manifest.json`: exact validation-selected checkpoint identities and hashes, frozen before any test access.",
        "- `plots/`: seed-aggregated training/validation curves by sequence length and five-seed validation metric comparisons.",
        "",
        "This stage confirms validation behavior only. Stage 6 test evaluation has not started.",
    ])
    md_path = OUTPUT_ROOT / "stage5b_confirmation_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_hash = stage5a.sha256_file(md_path)
    md_path.with_name(md_path.name + ".sha256").write_text(f"{md_hash}  {md_path.name}\n", encoding="ascii")
    stage5a.write_json(OUTPUT_ROOT / "provenance.json", {
        "stage": "Phase 3 Stage 5B provenance",
        "status": summary["status"],
        "config_hashes": config_hashes,
        "source_hashes": source_hashes,
        "dataset_integrity": data_records,
        "hardware": hardware,
        "checkpoint_selection_manifest_sha256": selection_hash,
        "summary_json_sha256": summary_hash,
        "summary_markdown_sha256": md_hash,
        "output_artifact_hashes": summary["output_artifact_hashes"],
    })
    print(json.dumps({"status": summary["status"], "completed_runs": summary["completed_run_count"], "selection_manifest_sha256": selection_hash, "summary_sha256": summary_hash, "output_directory": str(OUTPUT_ROOT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
