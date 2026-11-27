#!/usr/bin/env python3
"""Final matched-length test evaluation for the frozen Stage 5B checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.paper_v4_long_sequence import stage4_frozen_test_evaluation as stage4  # noqa: E402
from experiments.paper_v4_long_sequence import stage5a_matched_length_screen as stage5a  # noqa: E402

OUTPUT_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage6_matched_length_test_evaluation_v1"
STAGE2_REPORT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage2_consolidated_validation.json"
STAGE4_REPORT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/evaluation_report.json"
STAGE5B_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1"
SELECTION_MANIFEST = STAGE5B_ROOT / "checkpoint_selection_manifest.json"
STAGE5B_SUMMARY = STAGE5B_ROOT / "stage5b_confirmation_summary.json"
SEEDS = (11, 22, 33, 44, 55)
LENGTHS = (200, 400)
VARIANTS = ("M0", "M1")
METRICS = stage4.METRICS
BATCH_SIZE = 16
WARMUP_COUNT = 1
TIMED_REPETITIONS = 3
EXPECTED_BUNDLE_HASHES = {
    200: "e1e774aea741a97bde552c2ecd4e9878bd07eaff847ed38bf9aef6d6d3ccebfe",
    400: "1ebede2aa4e4a210b45af5ccd3e6b9b49952d117b5e7a7968f54ac4d81a904e7",
}
stage4_report: dict[str, Any] = {}


def fail_unless(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes_from_stage5b(stage5b_summary: dict[str, Any]) -> dict[str, str]:
    hashes = stage5b_summary["source_hashes"]
    for relative, expected in hashes.items():
        actual = stage5a.sha256_file(PROJECT / relative)
        fail_unless(actual == expected, f"Stage 5B frozen source hash mismatch: {relative}")
    hashes["experiments/paper_v4_long_sequence/stage4_frozen_test_evaluation.py"] = stage5a.sha256_file(Path(stage4.__file__).resolve())
    hashes["experiments/paper_v4_long_sequence/stage5a_matched_length_screen.py"] = stage5a.sha256_file(Path(stage5a.__file__).resolve())
    hashes["experiments/paper_v4_long_sequence/stage6_matched_length_test_evaluation.py"] = stage5a.sha256_file(Path(__file__).resolve())
    return dict(hashes)


def load_and_strictly_verify_checkpoints() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    selection_hash = stage5a.verify_sidecar(SELECTION_MANIFEST)
    stage5b_hash = stage5a.verify_sidecar(STAGE5B_SUMMARY)
    stage5b_summary = load_json(STAGE5B_SUMMARY)
    selection = load_json(SELECTION_MANIFEST)
    fail_unless(stage5b_summary["status"] == "PASS" and stage5b_summary["completed_run_count"] == 20, "Stage 5B did not complete all 20 runs")
    fail_unless(selection["status"] == "FROZEN_BEFORE_TEST_ACCESS" and selection["test_accessed"] is False, "Stage 5B selection manifest is not frozen before test access")
    fail_unless(len(selection["selected_checkpoints"]) == 20, "Stage 5B selection manifest must contain exactly 20 checkpoints")
    source_hashes = source_hashes_from_stage5b(stage5b_summary)

    run_records = {
        (int(row["sequence_length"]), row["variant"], int(row["confirmation_seed"])): row
        for row in stage5b_summary["run_summaries"]
    }
    selected_records = {
        (int(row["sequence_length"]), row["variant"], int(row["seed"])): row
        for row in selection["selected_checkpoints"]
    }
    expected_keys = {(length, variant, seed) for length in LENGTHS for variant in VARIANTS for seed in SEEDS}
    fail_unless(set(run_records) == expected_keys and set(selected_records) == expected_keys, "Stage 5B checkpoint matrix does not match requested 20 evaluations")

    verified = []
    for length in LENGTHS:
        for variant in VARIANTS:
            for seed in SEEDS:
                key = (length, variant, seed)
                run = run_records[key]
                frozen = selected_records[key]
                fail_unless(run["status"] == "COMPLETED" and frozen["status"] == "COMPLETED", f"Stage 5B run did not complete: {key}")
                config_path = PROJECT / run["config_path"]
                config_hash = stage5a.sha256_file(config_path)
                fail_unless(config_hash == run["config_sha256"] == frozen["config_sha256"], f"Stage 5B config hash mismatch: {key}")
                config = load_json(config_path)
                fail_unless(config["sequence_length"] == length and config["model"]["variant"] == variant and config["training"]["confirmation_seed"] == seed, f"Stage 5B config identity mismatch: {key}")
                run_dir = PROJECT / run["checkpoint_path"]
                checkpoint_hash = stage5a.sha256_file(run_dir)
                fail_unless(checkpoint_hash == run["checkpoint_sha256"] == frozen["checkpoint_sha256"], f"Selected checkpoint hash mismatch: {key}")
                fail_unless(run_dir.stat().st_size == frozen["checkpoint_size_bytes"], f"Selected checkpoint size mismatch: {key}")
                run_record_path = run_dir.parent / "run_summary.json"
                stage5a.verify_sidecar(run_record_path)
                run_record_json = load_json(run_record_path)
                fail_unless(run_record_json["checkpoint_sha256"] == checkpoint_hash and run_record_json["config_sha256"] == config_hash, f"Stage 5B run record provenance mismatch: {key}")
                stats = None
                stats_hash = None
                if variant == "M1":
                    stats_path = run_dir.parent / "train_only_feature_statistics.json"
                    stats_hash = stage5a.sha256_file(stats_path)
                    fail_unless(stats_hash == run["feature_statistics_sha256"] == frozen["feature_statistics_sha256"], f"M1 statistics hash mismatch: {key}")
                    stats = load_json(stats_path)
                    fail_unless(stats["statistics_source"] == "train split only", f"M1 statistics not train-only: {key}")
                    fail_unless(stats["confirmation_seed"] == seed, f"M1 statistics seed mismatch: {key}")
                    fail_unless(stats["time_points_per_trajectory"] == length and stats["train_trajectory_count"] == 819, f"M1 statistics data scope mismatch: {key}")
                    fail_unless(stats["normalized_token_indices_zero_based"] == list(stage5a.NORMALIZED_FEATURE_INDICES), f"M1 normalized feature indices mismatch: {key}")
                    fail_unless(stats["preserved_unscaled_token_indices_zero_based"] == [1, 2, 6] and stats["std_floor"] == 1e-4, f"M1 feature normalization policy mismatch: {key}")

                model = stage5a.build_model(config, stats, torch.device("cpu"))
                saved = torch.load(run_dir, map_location="cpu", weights_only=True)
                fail_unless(saved["selected_epoch"] == frozen["selected_epoch"] == run["selected_epoch"], f"Checkpoint selected epoch mismatch: {key}")
                fail_unless(saved["config_sha256"] == config_hash and saved["feature_statistics_sha256"] == stats_hash, f"Checkpoint metadata mismatch: {key}")
                model.load_state_dict(saved["state_dict"], strict=True)
                trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                fail_unless(trainable == 619014 and trainable == run["parameter_count"], f"Parameter count mismatch: {key}")
                if variant == "M1":
                    mean = torch.tensor(stats["feature_mean_21"], dtype=torch.float32)
                    std = torch.tensor(stats["feature_std_21"], dtype=torch.float32)
                    fail_unless(torch.equal(model.feature_mean, mean) and torch.equal(model.feature_std, std), f"M1 normalization buffers differ from training stats: {key}")
                    fail_unless(torch.equal(model.feature_mean, saved["state_dict"]["feature_mean"]) and torch.equal(model.feature_std, saved["state_dict"]["feature_std"]), f"M1 normalization buffers differ from checkpoint: {key}")
                verified.append({
                    "length": length,
                    "variant": variant,
                    "seed": seed,
                    "model": model,
                    "checkpoint_path": str(run_dir.relative_to(PROJECT)),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_size_bytes": run_dir.stat().st_size,
                    "config_path": str(config_path.relative_to(PROJECT)),
                    "config_sha256": config_hash,
                    "training_record_path": str(run_record_path.relative_to(PROJECT)),
                    "training_record_sha256": stage5a.sha256_file(run_record_path),
                    "feature_statistics_path": str((run_dir.parent / "train_only_feature_statistics.json").relative_to(PROJECT)) if stats is not None else None,
                    "feature_statistics_sha256": stats_hash,
                    "parameter_count": trainable,
                    "selected_epoch": frozen["selected_epoch"],
                    "validation_metrics": frozen["validation_metrics"],
                    "strict_state_dict_load": True,
                })
    integrity = {
        "stage5B_summary_path": str(STAGE5B_SUMMARY.relative_to(PROJECT)),
        "stage5B_summary_sha256": stage5b_hash,
        "selection_manifest_path": str(SELECTION_MANIFEST.relative_to(PROJECT)),
        "selection_manifest_sha256": selection_hash,
        "all_20_configs_verified": True,
        "all_20_checkpoints_verified_and_strict_loaded": True,
        "all_parameter_counts_619014": True,
        "all_M1_statistics_and_buffers_verified": True,
        "source_hashes_match_frozen_stage5B_record": True,
    }
    return verified, integrity, source_hashes


def load_test_datasets(stage2: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    # Reuse the audited Stage 4 manifest/tensor validator. This reads only the
    # T=200/T=400 test tensors after every Stage 5B checkpoint has strict-loaded.
    payloads, integrity = stage4.verify_long_test_datasets(stage2)
    records = {}
    for length in LENGTHS:
        bundle = PROJECT / f"artifacts/results/synthetic/paper_v4_long_sequence/T{length}_fixed_profile_v1"
        manifest_path = bundle / "dataset_manifest.json"
        manifest_hash = stage5a.verify_sidecar(manifest_path)
        fail_unless(manifest_hash == EXPECTED_BUNDLE_HASHES[length], f"T={length} frozen dataset manifest mismatch")
        manifest = load_json(manifest_path)
        test_record = manifest["splits"]["test"]
        test_path = bundle / test_record["file"]
        test_manifest_path = bundle / "manifests/test_manifest.json"
        test_manifest_hash = stage5a.verify_sidecar(test_manifest_path)
        test_manifest = load_json(test_manifest_path)
        test_file_hash = stage5a.sha256_file(test_path)
        stage2_row = stage2["length_results"][str(length)]
        fail_unless(test_file_hash == stage2_row["split_validation"]["test"]["split_file_sha256"], f"T={length} test tensor differs from Stage 2 frozen hash")
        stage4_manifest_hashes = {
            row["test_split_manifest_sha256"]
            for row in stage4_report["per_seed_results"]
            if row["sequence_length"] == length
        }
        fail_unless(stage4_manifest_hashes == {test_manifest_hash}, f"T={length} test manifest hash differs from frozen Stage 4 provenance")
        fail_unless(test_manifest["parent_manifest_sha256"] == manifest_hash, f"T={length} test manifest parent mismatch")
        fail_unless(test_record["trajectory_count"] == 1024, f"T={length} test trajectory count mismatch")
        payload = payloads[length]
        records[length] = {
            "length": length,
            "bundle_directory": str(bundle.relative_to(PROJECT)),
            "bundle_manifest_path": str(manifest_path.relative_to(PROJECT)),
            "bundle_manifest_sha256": manifest_hash,
            "test_manifest_path": str(test_manifest_path.relative_to(PROJECT)),
            "test_manifest_sha256": test_manifest_hash,
            "test_file_path": str(test_path.relative_to(PROJECT)),
            "test_file_sha256": test_file_hash,
            "generation_seed": test_record["generation_seed"],
            "trajectory_count": len(payload["trajectory_ids"]),
            "trajectory_ids_sha256": hashlib.sha256("\n".join(payload["trajectory_ids"]).encode()).hexdigest(),
            "tensor_shapes": {name: list(payload[name].shape) for name in ("state", "observations", "initial_state", "initialization_observations")},
            "tensor_sha256": {name: stage4.tensor_sha256(payload[name]) for name in ("state", "observations", "initial_state", "initialization_observations")},
            "all_finite_and_initialization_aligned": True,
        }
    integrity["stage2_validation_report_sha256"] = stage5a.verify_sidecar(STAGE2_REPORT)
    integrity["test_manifests_and_tensor_hashes_match_stage2"] = True
    return payloads, {"lengths": records, "stage4_manifest_validation": integrity}


def evaluate_one(record: dict[str, Any], payload: dict[str, Any], device: torch.device, hardware: dict[str, Any]) -> dict[str, Any]:
    model = record["model"].to(device)
    model.eval()
    observations_gpu = payload["observations"].to(device)
    init_gpu = payload["initial_state"].to(device)
    truth_cpu = payload["state"].contiguous()
    fail_unless(torch.isfinite(observations_gpu).all() and torch.isfinite(init_gpu).all(), "test input contains non-finite values")

    precheck = stage4.infer_full_test(model, observations_gpu, init_gpu, BATCH_SIZE)
    fail_unless(precheck.shape == (1024, 6, record["length"]), "prediction output shape mismatch")
    del precheck
    torch.cuda.synchronize(device)
    for _ in range(WARMUP_COUNT):
        warmup = stage4.infer_full_test(model, observations_gpu, init_gpu, BATCH_SIZE)
        del warmup
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    durations = []
    final_prediction = None
    for _ in range(TIMED_REPETITIONS):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = stage4.infer_full_test(model, observations_gpu, init_gpu, BATCH_SIZE)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        if final_prediction is not None:
            del final_prediction
        final_prediction = prediction
    mean_seconds = sum(durations) / len(durations)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if final_prediction is None:
        raise RuntimeError("timed inference produced no prediction")
    prediction_cpu = final_prediction.cpu().contiguous()
    del final_prediction
    fail_unless(prediction_cpu.shape == truth_cpu.shape and torch.isfinite(prediction_cpu).all(), "final prediction shape/finiteness check failed")
    aggregate_metrics = stage4.metrics_for_tensor(prediction_cpu, truth_cpu)
    windows = stage4.temporal_metrics(prediction_cpu, truth_cpu, interval=10)
    result = {
        "status": "PASS",
        "variant": record["variant"],
        "seed": record["seed"],
        "sequence_length": record["length"],
        "trajectory_count": 1024,
        "checkpoint_path": record["checkpoint_path"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "config_path": record["config_path"],
        "config_sha256": record["config_sha256"],
        "training_record_path": record["training_record_path"],
        "training_record_sha256": record["training_record_sha256"],
        "feature_statistics_path": record["feature_statistics_path"],
        "feature_statistics_sha256": record["feature_statistics_sha256"],
        "selected_epoch": record["selected_epoch"],
        "parameter_count": record["parameter_count"],
        "dataset_manifest_path": payload["_provenance"]["bundle_manifest_path"],
        "dataset_manifest_sha256": payload["_provenance"]["bundle_manifest_sha256"],
        "test_manifest_path": payload["_provenance"]["test_manifest_path"],
        "test_manifest_sha256": payload["_provenance"]["test_manifest_sha256"],
        "test_file_path": payload["_provenance"]["test_file_path"],
        "test_file_sha256": payload["_provenance"]["test_file_sha256"],
        "metrics": aggregate_metrics,
        "temporal_windows": windows,
        "timing": {
            "gpu": hardware["gpu_name"],
            "device": "cuda:0",
            "batch_size": BATCH_SIZE,
            "warmup_count": WARMUP_COUNT,
            "timed_repetitions": TIMED_REPETITIONS,
            "raw_seconds": durations,
            "mean_full_split_inference_seconds": mean_seconds,
            "seconds_per_trajectory": mean_seconds / 1024,
            "seconds_per_observation": mean_seconds / (1024 * record["length"]),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "baseline_allocated_bytes": int(baseline_allocated),
            "baseline_reserved_bytes": int(baseline_reserved),
            "peak_allocated_delta_bytes": max(0, peak_allocated - int(baseline_allocated)),
            "peak_reserved_delta_bytes": max(0, peak_reserved - int(baseline_reserved)),
            "cuda_synchronized_around_timing": True,
            "peak_memory_reset_after_warmup": True,
            "inputs_gpu_resident_during_timing": True,
            "stage4_timing_protocol_matched": True,
        },
        "state_estimate_count_equals_observation_count": True,
        "test_accessed_for_evaluation_only": True,
    }
    model.cpu()
    del model, observations_gpu, init_gpu
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


def mean_sd(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {"mean": mean, "sample_sd": sd}


def aggregate_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates, paired_summary, paired_seed_rows, stage4_comparisons = [], [], [], []
    for length in LENGTHS:
        grouped = {variant: [r for r in results if r["status"] == "PASS" and r["sequence_length"] == length and r["variant"] == variant] for variant in VARIANTS}
        for variant in VARIANTS:
            for metric in METRICS:
                values = [row["metrics"][metric] for row in grouped[variant]]
                aggregates.append({"sequence_length": length, "variant": variant, "metric": metric, **mean_sd(values), "seed_count": len(values)})
        for metric in METRICS:
            differences, relative_changes = [], []
            for seed in SEEDS:
                m0 = next(r for r in grouped["M0"] if r["seed"] == seed)
                m1 = next(r for r in grouped["M1"] if r["seed"] == seed)
                delta = m1["metrics"][metric] - m0["metrics"][metric]
                relative = delta / m0["metrics"][metric] if m0["metrics"][metric] else 0.0
                differences.append(delta)
                relative_changes.append(relative)
                paired_seed_rows.append({"sequence_length": length, "seed": seed, "metric": metric, "M0": m0["metrics"][metric], "M1": m1["metrics"][metric], "difference_M1_minus_M0": delta, "relative_change": relative, "relative_change_percent": 100.0 * relative})
            paired_summary.append({"sequence_length": length, "metric": metric, **{f"difference_{k}": v for k, v in mean_sd(differences).items()}, **{f"relative_change_{k}": v for k, v in mean_sd(relative_changes).items()}, "paired_seed_count": 5, "inferential_significance_claimed": False})

        for variant in VARIANTS:
            extrap = [r for r in stage4_report["per_seed_results"] if r["sequence_length"] == length and r["variant"] == variant]
            matched = grouped[variant]
            for metric in METRICS:
                extrap_summary = mean_sd([r["metrics"][metric] for r in extrap])
                matched_summary = mean_sd([r["metrics"][metric] for r in matched])
                change = matched_summary["mean"] - extrap_summary["mean"]
                stage4_comparisons.append({
                    "sequence_length": length,
                    "variant": variant,
                    "metric": metric,
                    "stage4_T100_trained_mean": extrap_summary["mean"],
                    "stage4_T100_trained_sample_sd": extrap_summary["sample_sd"],
                    "stage6_matched_length_mean": matched_summary["mean"],
                    "stage6_matched_length_sample_sd": matched_summary["sample_sd"],
                    "absolute_change_matched_minus_extrapolation": change,
                    "relative_change_percent_matched_vs_extrapolation": 100.0 * change / extrap_summary["mean"] if extrap_summary["mean"] else 0.0,
                    "comparison_interpretation": "descriptive between separately trained checkpoints on the same fixed test trajectories; no inferential significance claimed",
                })
    return aggregates, paired_summary, paired_seed_rows, stage4_comparisons


def temporal_comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for length in LENGTHS:
        for variant in VARIANTS:
            stage4_rows = [r for r in stage4_report["per_seed_results"] if r["sequence_length"] == length and r["variant"] == variant]
            stage6_rows = [r for r in results if r["status"] == "PASS" and r["sequence_length"] == length and r["variant"] == variant]
            if len(stage6_rows) != 5:
                continue
            for window_index, window in enumerate(stage6_rows[0]["temporal_windows"]):
                for metric in METRICS:
                    s4 = mean_sd([r["temporal_windows"][window_index][metric] for r in stage4_rows])
                    s6 = mean_sd([r["temporal_windows"][window_index][metric] for r in stage6_rows])
                    rows.append({
                        "sequence_length": length,
                        "variant": variant,
                        "start_step_1based": window["start_step_1based"],
                        "end_step_1based": window["end_step_1based"],
                        "metric": metric,
                        "stage4_extrapolation_mean": s4["mean"],
                        "stage4_extrapolation_sample_sd": s4["sample_sd"],
                        "stage6_matched_mean": s6["mean"],
                        "stage6_matched_sample_sd": s6["sample_sd"],
                    })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return stage5a.sha256_file(path)


def make_plots(results: list[dict[str, Any]], output: Path) -> dict[str, str]:
    plot_dir = output / "plots"
    plot_dir.mkdir()
    labels = {
        "position_rmse_m": ("Position RMSE (m)", "position_rmse_over_time.png"),
        "velocity_rmse_mps": ("Velocity RMSE (m/s)", "velocity_rmse_over_time.png"),
        "acceleration_rmse_mps2": ("Acceleration RMSE (m/s²)", "acceleration_rmse_over_time.png"),
        "six_state_mse_scaled_units": ("Six-state MSE (scaled units)", "six_state_mse_over_time.png"),
    }
    hashes = {}
    for metric, (ylabel, filename) in labels.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
        for ax, length in zip(axes, LENGTHS):
            for variant, arm, linestyle, color in (
                ("M0", "Stage 4 extrapolation", "-", "#2458A6"),
                ("M1", "Stage 4 extrapolation", "--", "#D14A32"),
                ("M0", "Stage 6 matched-length", "-.", "#2458A6"),
                ("M1", "Stage 6 matched-length", ":", "#D14A32"),
            ):
                if arm == "Stage 4 extrapolation":
                    subset = [r for r in stage4_report["per_seed_results"] if r["sequence_length"] == length and r["variant"] == variant]
                    windows = [r["temporal_windows"] for r in subset]
                else:
                    subset = [r for r in results if r["status"] == "PASS" and r["sequence_length"] == length and r["variant"] == variant]
                    windows = [r["temporal_windows"] for r in subset]
                if not windows:
                    continue
                centers = [(w["start_step_1based"] + w["end_step_1based"]) / 2 for w in windows[0]]
                means, sds = [], []
                for i in range(len(windows[0])):
                    summary = mean_sd([window[i][metric] for window in windows])
                    means.append(summary["mean"])
                    sds.append(summary["sample_sd"])
                label = f"{arm}: {variant}"
                ax.plot(centers, means, color=color, linestyle=linestyle, linewidth=1.8, label=label)
                ax.fill_between(centers, np.asarray(means) - sds, np.asarray(means) + sds, color=color, alpha=0.08)
            ax.set_title(f"T={length}")
            ax.set_xlabel("Observation step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Frozen T=100 extrapolation vs matched-length checkpoints: {ylabel}")
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        hashes[str(path.relative_to(PROJECT))] = stage5a.sha256_file(path)
    return hashes


def main() -> None:
    global stage4_report
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 6 output directory: {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the installed Mamba implementation")

    # Integrity gate 1: checkpoint-selection record, all configs, source files,
    # statistics, strict state-dict loading, and parameter counts.
    checkpoints, checkpoint_integrity, source_hashes = load_and_strictly_verify_checkpoints()
    stage2_hash = stage5a.verify_sidecar(STAGE2_REPORT)
    stage2 = load_json(STAGE2_REPORT)
    fail_unless(stage2["status"] == "passed" and stage2["inference_run"] is False, "Stage 2 dataset validation report status invalid")

    stage4_hash = stage5a.verify_sidecar(STAGE4_REPORT)
    stage4_report = load_json(STAGE4_REPORT)
    fail_unless(stage4_report["status"] == "PASS" and len(stage4_report["per_seed_results"]) == 20, "Stage 4 comparison report is not complete")
    fail_unless(
        stage4_report["software_and_sources"]["stage4_runner_sha256"]
        == stage5a.sha256_file(Path(stage4.__file__).resolve()),
        "Stage 4 runner source hash differs from its report",
    )
    for relative, expected in stage4_report["software_and_sources"]["model_source_hashes"].items():
        fail_unless(stage5a.sha256_file(PROJECT / relative) == expected, f"Stage 4 model source hash mismatch: {relative}")
    seen_stage4 = set()
    for old_row in stage4_report["per_seed_results"]:
        key = (old_row["sequence_length"], old_row["variant"], old_row["seed"])
        if key in seen_stage4:
            continue
        seen_stage4.add(key)
        cp = PROJECT / old_row["checkpoint_path"]
        cfg = PROJECT / old_row["config_path"]
        fail_unless(stage5a.sha256_file(cp) == old_row["checkpoint_sha256"], f"Stage 4 checkpoint hash mismatch: {key}")
        fail_unless(stage5a.sha256_file(cfg) == old_row["config_sha256"], f"Stage 4 config hash mismatch: {key}")

    # Integrity gate 2: load and validate test tensors only after all frozen
    # checkpoint/config/source checks passed, and before any model inference.
    test_payloads, dataset_integrity = load_test_datasets(stage2)
    for old_row in stage4_report["per_seed_results"]:
        length = old_row["sequence_length"]
        new_row = dataset_integrity["lengths"][length]
        fail_unless(old_row["dataset_manifest_sha256"] == new_row["bundle_manifest_sha256"], f"Stage 4 bundle differs at T={length}")
        fail_unless(old_row["test_split_manifest_sha256"] == new_row["test_manifest_sha256"], f"Stage 4 test manifest differs at T={length}")
        fail_unless(old_row["test_file_sha256"] == new_row["test_file_sha256"], f"Stage 4 used a different T={length} test tensor")

    hardware = stage4._hardware()
    device = torch.device("cuda:0")
    # Preserve exact frozen selection-manifest bytes in the new output bundle.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    frozen_copy = OUTPUT_ROOT / "frozen_checkpoint_selection_manifest.json"
    shutil.copyfile(SELECTION_MANIFEST, frozen_copy)
    shutil.copyfile(SELECTION_MANIFEST.with_name(SELECTION_MANIFEST.name + ".sha256"), frozen_copy.with_name(frozen_copy.name + ".sha256"))
    fail_unless(stage5a.sha256_file(frozen_copy) == checkpoint_integrity["selection_manifest_sha256"], "copied frozen selection manifest hash mismatch")

    results = []
    for checkpoint in checkpoints:
        length = checkpoint["length"]
        payload = dict(test_payloads[length])
        payload["_provenance"] = dataset_integrity["lengths"][length]
        label = f"{checkpoint['variant']} seed={checkpoint['seed']} T={length}"
        print(f"Evaluating {label}", flush=True)
        try:
            results.append(evaluate_one(checkpoint, payload, device, hardware))
        except Exception as exc:
            checkpoint["model"].cpu()
            del checkpoint["model"]
            torch.cuda.empty_cache()
            results.append({
                "status": "FAIL",
                "variant": checkpoint["variant"],
                "seed": checkpoint["seed"],
                "sequence_length": length,
                "checkpoint_path": checkpoint["checkpoint_path"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "config_path": checkpoint["config_path"],
                "config_sha256": checkpoint["config_sha256"],
                "error": f"{type(exc).__name__}: {exc}",
                "test_accessed_for_evaluation_only": True,
            })
            print(f"FAILED {label}: {type(exc).__name__}: {exc}", flush=True)

    aggregates, paired_summary, paired_seed_rows, stage4_comparisons = aggregate_results(results)
    temporal_rows = [
        {"variant": r["variant"], "seed": r["seed"], "sequence_length": r["sequence_length"], **window}
        for r in results if r["status"] == "PASS" for window in r["temporal_windows"]
    ]
    temporal_comparison = temporal_comparison_rows(results)
    result_csv_rows = []
    for r in results:
        row = {
            "variant": r["variant"], "seed": r["seed"], "sequence_length": r["sequence_length"], "status": r["status"],
            "trajectory_count": r.get("trajectory_count"), "selected_epoch": r.get("selected_epoch"), "parameter_count": r.get("parameter_count"),
            "checkpoint_sha256": r["checkpoint_sha256"], "config_sha256": r["config_sha256"],
            "dataset_manifest_sha256": r.get("dataset_manifest_sha256"), "test_manifest_sha256": r.get("test_manifest_sha256"), "test_file_sha256": r.get("test_file_sha256"),
            "test_accessed_for_evaluation_only": r.get("test_accessed_for_evaluation_only", False),
        }
        if r["status"] == "PASS":
            row.update(r["metrics"])
            row.update({
                "mean_inference_seconds": r["timing"]["mean_full_split_inference_seconds"],
                "seconds_per_trajectory": r["timing"]["seconds_per_trajectory"],
                "seconds_per_observation": r["timing"]["seconds_per_observation"],
                "peak_allocated_bytes": r["timing"]["peak_allocated_bytes"],
                "peak_reserved_bytes": r["timing"]["peak_reserved_bytes"],
                "batch_size": r["timing"]["batch_size"],
                "warmup_count": r["timing"]["warmup_count"],
                "timed_repetitions": r["timing"]["timed_repetitions"],
            })
        else:
            row["error"] = r["error"]
        result_csv_rows.append(row)

    csv_hashes = {
        "per_seed_test_results.csv": write_csv(OUTPUT_ROOT / "per_seed_test_results.csv", result_csv_rows),
        "temporal_window_metrics.csv": write_csv(OUTPUT_ROOT / "temporal_window_metrics.csv", temporal_rows),
        "paired_m0_m1_differences.csv": write_csv(OUTPUT_ROOT / "paired_m0_m1_differences.csv", paired_seed_rows),
        "stage4_vs_stage6_comparison.csv": write_csv(OUTPUT_ROOT / "stage4_vs_stage6_comparison.csv", stage4_comparisons),
        "aggregate_test_results.csv": write_csv(OUTPUT_ROOT / "aggregate_test_results.csv", aggregates),
        "stage4_vs_stage6_temporal_comparison.csv": write_csv(OUTPUT_ROOT / "stage4_vs_stage6_temporal_comparison.csv", temporal_comparison),
    }
    plot_hashes = make_plots(results, OUTPUT_ROOT)

    status = "PASS" if len(results) == 20 and all(r["status"] == "PASS" for r in results) else "PARTIAL"
    report = {
        "stage": "Phase 3 Stage 6 final matched-length independent test evaluation",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"training": False, "fine_tuning": False, "test_based_tuning": False, "test_based_checkpoint_selection": False, "test_accessed_for_final_evaluation_only": True},
        "evaluation_matrix": {"lengths": list(LENGTHS), "variants": list(VARIANTS), "seeds": list(SEEDS), "expected_evaluations": 20, "completed_evaluations": sum(r["status"] == "PASS" for r in results), "test_trajectories_per_length": 1024},
        "batch_size": BATCH_SIZE,
        "warmup_count": WARMUP_COUNT,
        "timed_repetitions": TIMED_REPETITIONS,
        "metric_definitions": {
            "position_rmse_m": "joint x/y RMSE across trajectories/time after multiplying scaled state errors by 1,000",
            "velocity_rmse_mps": "joint vx/vy RMSE across trajectories/time after multiplying scaled state errors by 1,000",
            "acceleration_rmse_mps2": "joint ax/ay RMSE across trajectories/time after multiplying scaled state errors by 1,000",
            "six_state_mse_scaled_units": "mean squared error across batch, six scaled state channels, and time",
            "temporal_windows": "non-overlapping 10-step windows, including six-state MSE",
        },
        "checkpoint_integrity": checkpoint_integrity,
        "dataset_integrity": {"stage2_report_path": str(STAGE2_REPORT.relative_to(PROJECT)), "stage2_report_sha256": stage2_hash, **dataset_integrity},
        "stage4_comparison_source": {"path": str(STAGE4_REPORT.relative_to(PROJECT)), "sha256": stage4_hash, "same_test_files_confirmed": True},
        "hardware": hardware,
        "source_hashes": source_hashes,
        "checkpoints": [{k: v for k, v in row.items() if k != "model"} for row in checkpoints],
        "per_seed_results": results,
        "aggregate_test_results": aggregates,
        "paired_seed_differences_M1_minus_M0": paired_summary,
        "paired_seed_differences_detail": paired_seed_rows,
        "stage4_vs_stage6_comparison": stage4_comparisons,
        "stage4_vs_stage6_temporal_comparison": temporal_comparison,
        "paired_differences_descriptive_only": True,
        "between_stage_comparisons_descriptive_only": True,
        "output_artifact_hashes": {**csv_hashes, "plots": plot_hashes},
    }
    report_hash = stage5a.write_json(OUTPUT_ROOT / "stage6_evaluation_report.json", report)
    md_path = OUTPUT_ROOT / "stage6_evaluation_report.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    md_hash = stage5a.sha256_file(md_path)
    md_path.with_name(md_path.name + ".sha256").write_text(f"{md_hash}  {md_path.name}\n", encoding="ascii")
    dataset_provenance = {
        "stage": "Phase 3 Stage 6 frozen dataset provenance",
        "stage2_report_sha256": stage2_hash,
        "stage4_comparison_report_sha256": stage4_hash,
        "datasets": dataset_integrity["lengths"],
    }
    dataset_provenance_hash = stage5a.write_json(OUTPUT_ROOT / "frozen_test_dataset_provenance.json", dataset_provenance)
    checkpoint_provenance = {
        "stage": "Phase 3 Stage 6 frozen checkpoint provenance",
        "selection_manifest_sha256": checkpoint_integrity["selection_manifest_sha256"],
        "stage5B_summary_sha256": checkpoint_integrity["stage5B_summary_sha256"],
        "checkpoints": [{k: v for k, v in row.items() if k != "model"} for row in checkpoints],
    }
    checkpoint_provenance_hash = stage5a.write_json(OUTPUT_ROOT / "frozen_checkpoint_provenance.json", checkpoint_provenance)
    provenance = {
        "stage": "Phase 3 Stage 6 provenance",
        "status": status,
        "checkpoint_selection_manifest_sha256": checkpoint_integrity["selection_manifest_sha256"],
        "dataset_provenance_sha256": dataset_provenance_hash,
        "checkpoint_provenance_sha256": checkpoint_provenance_hash,
        "stage2_report_sha256": stage2_hash,
        "stage4_comparison_report_sha256": stage4_hash,
        "source_hashes": source_hashes,
        "hardware": hardware,
        "summary_json_sha256": report_hash,
        "summary_markdown_sha256": md_hash,
        "output_artifact_hashes": report["output_artifact_hashes"],
    }
    provenance_hash = stage5a.write_json(OUTPUT_ROOT / "provenance.json", provenance)
    print(json.dumps({"status": status, "evaluations": len(results), "passed": sum(r["status"] == "PASS" for r in results), "report_sha256": report_hash, "markdown_sha256": md_hash, "provenance_sha256": provenance_hash, "output_directory": str(OUTPUT_ROOT)}, indent=2), flush=True)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Stage 6: Final Matched-Length Test Evaluation",
        "",
        f"**Status:** {report['status']} ({report['evaluation_matrix']['completed_evaluations']}/20 evaluations passed)",
        "",
        "The five-seed Stage 5B checkpoints were evaluated on the frozen Stage 2 T=200/T=400 test splits. No training, fine-tuning, checkpoint selection, normalization fitting, or test-based tuning was performed. Stage 4 results are included from the hash-verified report and use the same test tensor files.",
        "",
        "## Integrity and method",
        "",
        "All 20 Stage 5B checkpoint/config records were SHA-256 verified and state dictionaries loaded strictly before test inference. M1 train-only statistics and checkpoint buffers matched. Every model had 619,014 trainable parameters. Dataset manifests, test split manifests, tensor files, tensor hashes, dimensions, finite values, trajectory IDs, and initialization alignment passed validation.",
        "",
        f"Inference used CUDA batch size {report['batch_size']}, {report['warmup_count']} warmup pass, and {report['timed_repetitions']} synchronized timed repetitions per checkpoint/length. Inputs were GPU-resident during timing; this matches the Stage 4 timing protocol. Hardware: {report['hardware']['gpu_name']}, driver {report['hardware']['gpu_driver_version']}, CUDA {report['hardware']['cuda_runtime_version']}, PyTorch {report['hardware']['pytorch_version']}, Mamba {report['hardware']['mamba_ssm_version']}.",
        "",
        "## Matched-length full-horizon test results",
        "",
        "Mean ± sample SD across five seeds.",
        "",
        "| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    lookup = {(row["sequence_length"], row["variant"], row["metric"]): row for row in report["aggregate_test_results"]}
    for length in LENGTHS:
        for variant in VARIANTS:
            values = []
            for metric in METRICS:
                row = lookup.get((length, variant, metric))
                values.append(f"{row['mean']:.6g} ± {row['sample_sd']:.3g}" if row else "incomplete")
            lines.append(f"| {length} | {variant} | " + " | ".join(values) + " |")
    lines.extend(["", "## Paired M1 − M0 test differences", "", "Negative values favor M1. Pairing follows the Stage 5B shared initial weights and data-loader order. These differences are descriptive; no inferential significance is claimed.", "", "| T | Metric | Mean difference | Sample SD | Relative change mean ± SD |", "|---:|---|---:|---:|---:|"])
    paired = {(row["sequence_length"], row["metric"]): row for row in report["paired_seed_differences_M1_minus_M0"]}
    for length in LENGTHS:
        for metric in METRICS:
            row = paired.get((length, metric))
            if row:
                lines.append(f"| {length} | {metric} | {row['difference_mean']:.6g} | {row['difference_sample_sd']:.3g} | {row['relative_change_mean']:+.2%} ± {row['relative_change_sample_sd']:.2%} |")
    lines.extend(["", "## Stage 4 extrapolation versus Stage 6 matched-length", "", "Stage 4 is the frozen T=100-trained model on the same test trajectories; Stage 6 is trained at the matching length. Changes compare arm means and are descriptive between separately trained checkpoints.", "", "| T | Variant | Metric | Stage 4 mean ± SD | Stage 6 mean ± SD | Absolute change | Relative change |", "|---:|---|---|---:|---:|---:|---:|"])
    for row in report["stage4_vs_stage6_comparison"]:
        lines.append(f"| {row['sequence_length']} | {row['variant']} | {row['metric']} | {row['stage4_T100_trained_mean']:.6g} ± {row['stage4_T100_trained_sample_sd']:.3g} | {row['stage6_matched_length_mean']:.6g} ± {row['stage6_matched_length_sample_sd']:.3g} | {row['absolute_change_matched_minus_extrapolation']:+.6g} | {row['relative_change_percent_matched_vs_extrapolation']:+.2f}% |")
    lines.extend([
        "",
        "## Temporal metrics and resources",
        "",
        "All Stage 6 seed-wise 10-step window metrics, including six-state MSE, are in `temporal_window_metrics.csv`. Four temporal plots show Stage 4 extrapolation and Stage 6 matched-length curves separately by line style for T=200 and T=400, with ±1 sample SD shading. The Stage 4/Stage 6 interval means and SDs are also in `stage4_vs_stage6_temporal_comparison.csv`.",
        "",
        "Per-evaluation timing, per-trajectory and per-observation latency, batch size, warmup/repetition counts, peak allocated/reserved memory, baseline memory, and checkpoint/dataset provenance are in the machine-readable JSON report.",
        "",
        "## Interpretation and limitations",
        "",
        "These results evaluate the fixed Stage 2 motion/noise law and do not establish robustness to other regimes. Long-horizon errors reflect the broader state/range distribution at T=400. Comparisons between Stage 4 and Stage 6 are descriptive because they are separately trained checkpoints, even though they use the same test trajectories. Stage 6 metrics were not used to change models, normalization, or training settings.",
        "",
        "Artifacts: `per_seed_test_results.csv`, `aggregate_test_results.csv`, `temporal_window_metrics.csv`, `paired_m0_m1_differences.csv`, `stage4_vs_stage6_comparison.csv`, `plots/`, `frozen_checkpoint_provenance.json`, `frozen_test_dataset_provenance.json`, and `provenance.json`.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
