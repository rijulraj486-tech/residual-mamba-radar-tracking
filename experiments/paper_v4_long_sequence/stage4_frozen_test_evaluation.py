#!/usr/bin/env python3
"""Frozen T=100 M0/M1 evaluation on independent T=200/T=400 test pools."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.paper_v4_mamba_improvement.model_variants import (
    NORMALIZED_FEATURE_INDICES,
    TrainNormalizedFeatureTracker,
)
from radar_tracking.data.independent_dataset import (
    load_and_validate_manifest_set,
    sha256_file,
    tensor_sha256,
    validate_split_tensors,
)
from radar_tracking.models.tracker_model import BiMambaTrackerModel


OUTPUT_DIR = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2"
M0_ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
M1_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2"
LONG_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence"
STAGE2_REPORT = LONG_ROOT / "stage2_consolidated_validation.json"
STAGE3_DIR = LONG_ROOT / "stage3_frozen_preflight_v1"
M0_CONFIG = PROJECT / "configs/paper_v4_mamba_improvement/baseline_reference.json"
M1_CONFIG = PROJECT / "configs/paper_v4_mamba_improvement/features_train_only_normalization_five_seed_confirmation_v2.json"
M1_PILOT_CONFIG = PROJECT / "configs/paper_v4_mamba_improvement/features_train_only_normalization_seed11.json"
M1_FINAL_TEST = M1_ROOT / "final_test_evaluation_v1/final_test_comparison.json"
SEEDS = (11, 22, 33, 44, 55)
LENGTHS = (200, 400)
EXPECTED_SPLIT_SIZE = 1024
STATE_SCALE = 1000.0
BATCH_SIZE = 16
WARMUP_COUNT = 1
TIMED_REPETITIONS = 3
METRICS = (
    "position_rmse_m",
    "velocity_rmse_mps",
    "acceleration_rmse_mps2",
    "six_state_mse_scaled_units",
)
PHYSICAL_METRICS = METRICS[:3]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 sidecar mismatch for {path}")
    return actual


def verify_recorded_source_hashes(stage3: dict[str, Any]) -> dict[str, Any]:
    sources = stage3["software_and_sources"]["project_sources"]
    critical = (
        "tracker_model.py",
        "bimamba_encoder.py",
        "prediction_head.py",
        "model_variants.py",
        "benchmark_mamba.py",
        "run_screening_pilot.py",
    )
    result = {}
    for name in critical:
        entry = sources[name]
        path = PROJECT / entry["path"]
        actual = sha256_file(path)
        if actual != entry["current_sha256"]:
            raise ValueError(f"Stage 3 source hash changed for {entry['path']}")
        result[entry["path"]] = actual
    return result


def m0_kwargs(run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dim": 6,
        "obs_dim": 2,
        "dt": 1.0,
        "embed_dim": int(run_config["embed_dim"]),
        "hidden_dim": int(run_config["hidden_dim"]),
        "num_layers": int(run_config["num_layers"]),
        "d_state": int(run_config["d_state"]),
        "d_conv": int(run_config["d_conv"]),
        "expand": int(run_config["expand"]),
        "dropout": float(run_config["dropout"]),
        "encoder_direction": "forward",
        "task": "radar_benchmark",
    }


def m1_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        "state_dim": 6,
        "obs_dim": 2,
        "dt": 1.0,
        "embed_dim": int(model["embed_dim"]),
        "hidden_dim": int(model["hidden_dim"]),
        "num_layers": int(model["num_layers"]),
        "d_state": int(model["d_state"]),
        "d_conv": int(model["d_conv"]),
        "expand": int(model["expand"]),
        "dropout": float(model["dropout"]),
        "encoder_direction": "forward",
        "task": "radar_benchmark",
    }


def _verify_stage3_integrity(
    stage3: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    m0_config = json.loads(M0_CONFIG.read_text(encoding="utf-8"))
    m1_config = json.loads(M1_CONFIG.read_text(encoding="utf-8"))
    m1_pilot_config = json.loads(M1_PILOT_CONFIG.read_text(encoding="utf-8"))
    m1_config_copy = M1_ROOT / "config.json"
    frozen_hashes_path = M1_ROOT / "frozen_validation_hashes.json"
    selection_path = M1_ROOT / "validation_selection_manifest.json"
    aggregate_path = M1_ROOT / "validation_aggregate.json"
    frozen_hashes = json.loads(frozen_hashes_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    m0_config_hash = sha256_file(M0_CONFIG)
    m1_config_hash = sha256_file(M1_CONFIG)
    pilot_hash = sha256_file(M1_PILOT_CONFIG)
    if sha256_file(m1_config_copy) != m1_config_hash:
        raise ValueError("M1 copied config differs from frozen config file")
    if frozen_hashes["config_sha256"] != m1_config_hash:
        raise ValueError("M1 config hash mismatch against frozen validation record")
    if selection["selected_configuration_sha256"] != m1_config_hash:
        raise ValueError("M1 config hash mismatch against frozen selection record")
    if sha256_file(selection_path) != frozen_hashes["validation_selection_manifest_sha256"]:
        raise ValueError("M1 validation selection manifest checksum mismatch")
    if sha256_file(aggregate_path) != frozen_hashes["validation_aggregate_sha256"]:
        raise ValueError("M1 validation aggregate checksum mismatch")
    if pilot_hash != m1_config["pilot_config_sha256"]:
        raise ValueError("M1 pilot configuration checksum mismatch")
    if selection["source_file_sha256"].get(
        "configs/paper_v4_mamba_improvement/baseline_reference.json"
    ) != m0_config_hash:
        raise ValueError("M0 baseline config hash mismatch against frozen source record")
    if m0_config["model"]["parameter_count"] != 619014:
        raise ValueError("M0 config parameter count mismatch")
    if m1_config["model"]["parameter_count_expected"] != 619014:
        raise ValueError("M1 config parameter count mismatch")

    checkpoints = []
    stage3_checkpoints = {
        (item["variant"], int(item["seed"])): item
        for item in stage3["checkpoints"]
    }
    if len(stage3_checkpoints) != 10:
        raise ValueError("Stage 3 report does not contain the ten expected checkpoints")

    for seed in SEEDS:
        m0_run_path = M0_ROOT / f"seed_{seed}/run_metrics.json"
        m0_run_hash = sha256_file(m0_run_path)
        m0_run = json.loads(m0_run_path.read_text(encoding="utf-8"))
        m0_record = m0_run["mamba"]
        m0_checkpoint = M0_ROOT / f"seed_{seed}/{m0_record['checkpoint']}"
        m0_expected_hash = m0_record["checkpoint_sha256"]
        m0_actual_hash = sha256_file(m0_checkpoint)
        m0_stage3 = stage3_checkpoints[("M0", seed)]
        if m0_actual_hash != m0_expected_hash or m0_actual_hash != m0_stage3["checkpoint_sha256"]:
            raise ValueError(f"M0 seed {seed} checkpoint hash mismatch")
        if m0_run_hash != m0_stage3["run_record_sha256"]:
            raise ValueError(f"M0 seed {seed} run record hash changed since Stage 3")
        if m0_config_hash != m0_stage3["config_sha256"]:
            raise ValueError(f"M0 seed {seed} config hash changed since Stage 3")
        if canonical_sha256(m0_record["config"]) != m0_stage3[
            "training_config_canonical_sha256"
        ]:
            raise ValueError(f"M0 seed {seed} run config changed since Stage 3")
        if m0_record["parameter_count"] != 619014:
            raise ValueError(f"M0 seed {seed} parameter count mismatch")

        m1_run_path = M1_ROOT / f"seed_{seed}/run_metrics.json"
        m1_run_hash = sha256_file(m1_run_path)
        m1_run = json.loads(m1_run_path.read_text(encoding="utf-8"))
        m1_record = m1_run
        m1_checkpoint = M1_ROOT / f"seed_{seed}/{m1_record['checkpoint']}"
        m1_expected_hash = m1_record["checkpoint_sha256"]
        m1_actual_hash = sha256_file(m1_checkpoint)
        m1_stage3 = stage3_checkpoints[("M1", seed)]
        summary_path = M1_ROOT / f"seed_{seed}/seed_summary.json"
        stats_path = M1_ROOT / f"seed_{seed}/train_only_feature_statistics.json"
        stats_hash = sha256_file(stats_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if m1_actual_hash != m1_expected_hash or m1_actual_hash != m1_stage3["checkpoint_sha256"]:
            raise ValueError(f"M1 seed {seed} checkpoint hash mismatch")
        if m1_run_hash != m1_stage3["run_record_sha256"]:
            raise ValueError(f"M1 seed {seed} run record hash changed since Stage 3")
        if m1_config_hash != m1_stage3["config_sha256"]:
            raise ValueError(f"M1 seed {seed} config hash changed since Stage 3")
        if summary["candidate_checkpoint_sha256"] != m1_actual_hash:
            raise ValueError(f"M1 seed {seed} selection checkpoint hash mismatch")
        if m1_stage3["feature_statistics_sha256"] != stats_hash:
            raise ValueError(f"M1 seed {seed} feature-statistics hash changed")
        if summary["feature_statistics_sha256"] != stats_hash:
            raise ValueError(f"M1 seed {seed} feature-statistics run hash mismatch")
        if m1_record["parameter_count"]["total"] != 619014:
            raise ValueError(f"M1 seed {seed} parameter count mismatch")

        expected_indices = list(NORMALIZED_FEATURE_INDICES)
        if stats["normalized_token_indices_zero_based"] != expected_indices:
            raise ValueError(f"M1 seed {seed} normalized-channel list changed")
        if stats["preserved_unscaled_token_indices_zero_based"] != [1, 2, 6]:
            raise ValueError(f"M1 seed {seed} preserved-channel list changed")
        if stats["statistics_source"] != "train split only":
            raise ValueError(f"M1 seed {seed} normalization is not train-only")
        if stats["train_trajectory_count"] != 819 or stats[
            "time_points_per_trajectory"
        ] != 100:
            raise ValueError(f"M1 seed {seed} normalization training record changed")
        training_config = m1_record["training_config"]
        if training_config["feature_normalization"]["std_floor"] != 1e-4:
            raise ValueError(f"M1 seed {seed} normalization std floor changed")
        if training_config["feature_normalization"]["normalized_indices"] != expected_indices:
            raise ValueError(f"M1 seed {seed} normalization config indices changed")
        if training_config["feature_normalization"]["preserved_indices"] != [1, 2, 6]:
            raise ValueError(f"M1 seed {seed} preserved-channel config changed")
        if stats["normalization_formula"] != "(x - train_mean) / max(train_population_std, 1e-4)":
            raise ValueError(f"M1 seed {seed} normalization formula changed")
        if m1_config["feature_normalization"]["normalized_token_indices_zero_based"] != expected_indices:
            raise ValueError("M1 frozen config normalized-channel list mismatch")
        if m1_config["feature_normalization"]["preserved_unscaled_token_indices_zero_based"] != [1, 2, 6]:
            raise ValueError("M1 frozen config preserved-channel list mismatch")

        m0_state = torch.load(m0_checkpoint, map_location="cpu", weights_only=True)
        m0_model = BiMambaTrackerModel(**m0_kwargs(m0_record["config"]))
        m0_model.load_state_dict(m0_state, strict=True)
        m0_trainable = sum(p.numel() for p in m0_model.parameters() if p.requires_grad)
        if m0_trainable != 619014:
            raise ValueError(f"M0 seed {seed} strict-loaded parameter count mismatch")

        feature_mean = torch.tensor(stats["feature_mean_21"], dtype=torch.float32)
        feature_std = torch.tensor(stats["feature_std_21"], dtype=torch.float32)
        m1_state = torch.load(m1_checkpoint, map_location="cpu", weights_only=True)
        m1_model = TrainNormalizedFeatureTracker(
            feature_mean=feature_mean,
            feature_std=feature_std,
            **m1_kwargs(m1_config),
        )
        m1_model.load_state_dict(m1_state, strict=True)
        if not torch.equal(m1_model.feature_mean, m1_state["feature_mean"]):
            raise ValueError(f"M1 seed {seed} mean buffer differs from checkpoint")
        if not torch.equal(m1_model.feature_std, m1_state["feature_std"]):
            raise ValueError(f"M1 seed {seed} std buffer differs from checkpoint")
        if not torch.allclose(m1_model.feature_mean, feature_mean, atol=1e-6, rtol=1e-6):
            raise ValueError(f"M1 seed {seed} means differ from original training record")
        if not torch.allclose(m1_model.feature_std, feature_std, atol=1e-6, rtol=1e-6):
            raise ValueError(f"M1 seed {seed} stds differ from original training record")
        m1_trainable = sum(p.numel() for p in m1_model.parameters() if p.requires_grad)
        if m1_trainable != 619014:
            raise ValueError(f"M1 seed {seed} strict-loaded parameter count mismatch")

        checkpoints.extend(
            [
                {
                    "variant": "M0",
                    "seed": seed,
                    "model": m0_model,
                    "checkpoint_path": str(m0_checkpoint.relative_to(PROJECT)),
                    "checkpoint_sha256": m0_actual_hash,
                    "training_record_path": str(m0_run_path.relative_to(PROJECT)),
                    "training_record_sha256": m0_run_hash,
                    "training_config": m0_record["config"],
                    "training_config_canonical_sha256": canonical_sha256(m0_record["config"]),
                    "config_path": str(M0_CONFIG.relative_to(PROJECT)),
                    "config_sha256": m0_config_hash,
                    "parameter_count": m0_trainable,
                    "state_dict_strict_load": True,
                    "feature_statistics": None,
                },
                {
                    "variant": "M1",
                    "seed": seed,
                    "model": m1_model,
                    "checkpoint_path": str(m1_checkpoint.relative_to(PROJECT)),
                    "checkpoint_sha256": m1_actual_hash,
                    "training_record_path": str(m1_run_path.relative_to(PROJECT)),
                    "training_record_sha256": m1_run_hash,
                    "training_config": training_config,
                    "training_config_canonical_sha256": canonical_sha256(training_config),
                    "config_path": str(M1_CONFIG.relative_to(PROJECT)),
                    "config_sha256": m1_config_hash,
                    "parameter_count": m1_trainable,
                    "state_dict_strict_load": True,
                    "feature_statistics": {
                        "path": str(stats_path.relative_to(PROJECT)),
                        "sha256": stats_hash,
                        "mean_21": m1_model.feature_mean.cpu().tolist(),
                        "std_21": m1_model.feature_std.cpu().tolist(),
                        "normalized_indices": expected_indices,
                        "preserved_indices": [1, 2, 6],
                        "std_floor": 1e-4,
                    },
                },
            ]
        )

    integrity = {
        "M0_config_sha256": m0_config_hash,
        "M1_config_sha256": m1_config_hash,
        "M1_pilot_config_sha256": pilot_hash,
        "M1_frozen_validation_hashes_sha256": sha256_file(frozen_hashes_path),
        "M1_validation_selection_manifest_sha256": sha256_file(selection_path),
        "M1_validation_aggregate_sha256": sha256_file(aggregate_path),
        "Stage3_preflight_report_sha256": sha256_file(STAGE3_DIR / "preflight_report.json"),
        "Stage3_preflight_report_sidecar_verified": True,
        "all_ten_checkpoints_sha256_verified": True,
        "all_ten_state_dicts_strict_loaded": True,
        "all_parameter_counts_619014": True,
        "all_M1_statistics_and_buffers_verified": True,
        "original_source_hashes": verify_recorded_source_hashes(stage3),
    }
    return checkpoints, integrity


def verify_long_test_datasets(stage2: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    manifest_paths = []
    bundle_records = {}
    loaded_payloads = {}
    verified = {}

    for length in LENGTHS:
        expected_row = stage2["length_results"][str(length)]
        bundle_dir = LONG_ROOT / f"T{length}_fixed_profile_v1"
        bundle_manifest_path = bundle_dir / "dataset_manifest.json"
        bundle_manifest_hash = verify_sidecar(bundle_manifest_path)
        if bundle_manifest_hash != expected_row["bundle_manifest_sha256"]:
            raise ValueError(f"T={length} bundle manifest hash differs from Stage 2")
        manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        if manifest["sequence_length_observations"] != length:
            raise ValueError(f"T={length} manifest sequence length mismatch")
        if manifest["status"] != "generated_validated":
            raise ValueError(f"T={length} bundle manifest not marked validated")

        index_path = bundle_dir / "manifests/index.json"
        index_sha = verify_sidecar(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        split_item = next(row for row in index["split_manifests"] if row["split"] == "test")
        split_manifest_path = bundle_dir / split_item["file"]
        split_manifest_sha = verify_sidecar(split_manifest_path)
        if split_manifest_sha != split_item["sha256"]:
            raise ValueError(f"T={length} test split manifest index hash mismatch")
        if split_manifest_sha != index["split_manifests"][2]["sha256"]:
            # The order is train, validation, test in the frozen Stage 2 writer.
            raise ValueError(f"T={length} test split manifest identity mismatch")
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if split_manifest["parent_manifest_sha256"] != bundle_manifest_hash:
            raise ValueError(f"T={length} test manifest parent hash mismatch")
        manifest_paths.extend(
            bundle_dir / row["file"] for row in index["split_manifests"]
        )

        test_record = manifest["splits"]["test"]
        split_path = bundle_dir / test_record["file"]
        test_file_sha = sha256_file(split_path)
        if test_file_sha != test_record["file_sha256"]:
            raise ValueError(f"T={length} test tensor-file hash mismatch")
        if test_file_sha != expected_row["split_validation"]["test"]["split_file_sha256"]:
            raise ValueError(f"T={length} test tensor hash differs from Stage 2 validation")
        payload = torch.load(split_path, map_location="cpu", weights_only=True)
        if payload["split"] != "test" or payload["sequence_length"] != length:
            raise ValueError(f"T={length} test payload identity mismatch")
        if payload["generation_seed"] != test_record["generation_seed"]:
            raise ValueError(f"T={length} test split seed mismatch")
        if payload["trajectory_ids"] != test_record["trajectory_ids"]:
            raise ValueError(f"T={length} test trajectory IDs differ from frozen manifest")
        if len(payload["trajectory_ids"]) != EXPECTED_SPLIT_SIZE:
            raise ValueError(f"T={length} test trajectory count mismatch")
        for name in ("state", "observations", "initial_state", "initialization_observations"):
            tensor = payload[name]
            if list(tensor.shape) != test_record["tensor_shapes"][name]:
                raise ValueError(f"T={length} test {name} shape differs from manifest")
            if tensor_sha256(tensor) != test_record["tensor_sha256"][name]:
                raise ValueError(f"T={length} test {name} tensor SHA-256 mismatch")
        if payload["state"].shape != (EXPECTED_SPLIT_SIZE, 6, length):
            raise ValueError(f"T={length} test state shape mismatch")
        if payload["observations"].shape != (EXPECTED_SPLIT_SIZE, 2, length):
            raise ValueError(f"T={length} test observations shape mismatch")
        if payload["initial_state"].shape != (EXPECTED_SPLIT_SIZE, 6, 1):
            raise ValueError(f"T={length} initial_state shape mismatch")
        validate_split_tensors(
            state=payload["state"],
            observations=payload["observations"],
            initial_state=payload["initial_state"],
            initialization_observations=payload["initialization_observations"],
            trajectory_ids=payload["trajectory_ids"],
            sequence_length=length,
        )
        if split_manifest["splits"]["test"]["trajectory_ids"] != payload["trajectory_ids"]:
            raise ValueError(f"T={length} split-specific manifest IDs mismatch")
        loaded_payloads[length] = payload
        bundle_records[length] = {
            "bundle_manifest_path": str(bundle_manifest_path.relative_to(PROJECT)),
            "bundle_manifest_sha256": bundle_manifest_hash,
            "split_manifest_path": str(split_manifest_path.relative_to(PROJECT)),
            "split_manifest_sha256": split_manifest_sha,
            "manifest_index_sha256": index_sha,
            "test_file_path": str(split_path.relative_to(PROJECT)),
            "test_file_sha256": test_file_sha,
            "generation_seed": test_record["generation_seed"],
            "trajectory_count": len(payload["trajectory_ids"]),
            "trajectory_ids_sha256": hashlib.sha256(
                "\n".join(payload["trajectory_ids"]).encode("utf-8")
            ).hexdigest(),
            "tensor_shapes": test_record["tensor_shapes"],
            "tensor_sha256": test_record["tensor_sha256"],
            "all_tensors_finite": True,
            "initialization_alignment": "PASS",
        }
        verified[length] = True

    # Verify the six frozen split manifests together before using test tensors.
    all_splits = load_and_validate_manifest_set(manifest_paths)
    if len(all_splits) != 6:
        raise ValueError("Expected six Stage 2 split manifests")
    for length in LENGTHS:
        train_ids = set(
            json.loads(
                (LONG_ROOT / f"T{length}_fixed_profile_v1/manifests/train_manifest.json").read_text()
            )["splits"]["train"]["trajectory_ids"]
        )
        val_ids = set(
            json.loads(
                (LONG_ROOT / f"T{length}_fixed_profile_v1/manifests/validation_manifest.json").read_text()
            )["splits"]["validation"]["trajectory_ids"]
        )
        test_ids = set(loaded_payloads[length]["trajectory_ids"])
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError(f"T={length} trajectory IDs overlap across splits")
    all_test_ids = [
        trajectory_id
        for length in LENGTHS
        for trajectory_id in loaded_payloads[length]["trajectory_ids"]
    ]
    if len(all_test_ids) != len(set(all_test_ids)):
        raise ValueError("T=200 and T=400 test trajectory IDs overlap")
    return loaded_payloads, {
        "all_test_manifests_verified": True,
        "all_six_stage2_manifests_validated_together": True,
        "test_ids_disjoint_across_splits_and_lengths": True,
        "unique_test_trajectory_ids": len(all_test_ids),
        "lengths": bundle_records,
    }


def metrics_for_tensor(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """Match the established Mamba metric definitions and add interval MSE."""
    error = prediction - truth

    def group_rmse(first: int, second: int) -> float:
        return float(
            (error[:, [first, second], :].square().sum(dim=1).mean().sqrt() * STATE_SCALE).item()
        )

    return {
        "position_rmse_m": group_rmse(0, 3),
        "velocity_rmse_mps": group_rmse(1, 4),
        "acceleration_rmse_mps2": group_rmse(2, 5),
        "six_state_mse_scaled_units": float(error.square().mean().item()),
    }


def temporal_metrics(
    prediction: torch.Tensor, truth: torch.Tensor, interval: int = 10
) -> list[dict[str, Any]]:
    error = prediction - truth
    windows = []
    for start in range(0, truth.shape[-1], interval):
        end = min(start + interval, truth.shape[-1])
        pred_slice, truth_slice = prediction[:, :, start:end], truth[:, :, start:end]
        e = pred_slice - truth_slice

        def group_rmse(first: int, second: int) -> float:
            return float(
                (e[:, [first, second], :].square().sum(dim=1).mean().sqrt() * STATE_SCALE).item()
            )

        windows.append(
            {
                "start_step_1based": start + 1,
                "end_step_1based": end,
                "position_rmse_m": group_rmse(0, 3),
                "velocity_rmse_mps": group_rmse(1, 4),
                "acceleration_rmse_mps2": group_rmse(2, 5),
                "six_state_mse_scaled_units": float(e.square().mean().item()),
            }
        )
    return windows


def infer_full_test(
    model: BiMambaTrackerModel,
    observations_gpu: torch.Tensor,
    init_gpu: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, observations_gpu.shape[0], batch_size):
            end = min(start + batch_size, observations_gpu.shape[0])
            output = model(
                observations=observations_gpu[start:end],
                init_state=init_gpu[start:end],
            )
            prediction = output["state_estimates"]
            if prediction.shape != (end - start, 6, observations_gpu.shape[-1]):
                raise ValueError(
                    f"inference shape mismatch: {tuple(prediction.shape)}"
                )
            if not torch.isfinite(prediction).all():
                raise ValueError("non-finite prediction during test evaluation")
            outputs.append(prediction.detach())
            del output, prediction
    return torch.cat(outputs, dim=0)


def evaluate_one(
    checkpoint: dict[str, Any],
    length: int,
    payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = checkpoint["model"].to(device)
    model.eval()
    observations_gpu = payload["observations"].to(device)
    init_gpu = payload["initial_state"].to(device)
    truth_cpu = payload["state"].contiguous()
    if not torch.isfinite(observations_gpu).all() or not torch.isfinite(init_gpu).all():
        raise ValueError("non-finite CUDA test inputs")

    # Shape/finite precheck is separate from the timed windows.
    precheck_prediction = infer_full_test(model, observations_gpu, init_gpu, BATCH_SIZE)
    if precheck_prediction.shape != (EXPECTED_SPLIT_SIZE, 6, length):
        raise ValueError("full-horizon prediction shape mismatch")
    torch.cuda.synchronize()
    del precheck_prediction

    for _ in range(WARMUP_COUNT):
        warm_prediction = infer_full_test(model, observations_gpu, init_gpu, BATCH_SIZE)
        del warm_prediction
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    durations = []
    final_prediction_gpu = None
    for _ in range(TIMED_REPETITIONS):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        candidate = infer_full_test(model, observations_gpu, init_gpu, BATCH_SIZE)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        if final_prediction_gpu is not None:
            del final_prediction_gpu
        final_prediction_gpu = candidate

    mean_seconds = sum(durations) / len(durations)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if final_prediction_gpu is None:
        raise RuntimeError("no timed inference prediction produced")
    prediction_cpu = final_prediction_gpu.cpu().contiguous()
    del final_prediction_gpu
    if not torch.isfinite(prediction_cpu).all():
        raise ValueError("non-finite final prediction")
    if prediction_cpu.shape != truth_cpu.shape:
        raise ValueError("prediction/truth shape mismatch")

    aggregate = metrics_for_tensor(prediction_cpu, truth_cpu)
    windows = temporal_metrics(prediction_cpu, truth_cpu, interval=10)
    timing = {
        "device": torch.cuda.get_device_name(device),
        "batch_size": BATCH_SIZE,
        "warmup_count": WARMUP_COUNT,
        "timed_repetitions": TIMED_REPETITIONS,
        "raw_seconds": durations,
        "mean_full_split_inference_seconds": mean_seconds,
        "seconds_per_trajectory": mean_seconds / EXPECTED_SPLIT_SIZE,
        "seconds_per_observation": mean_seconds / (EXPECTED_SPLIT_SIZE * length),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
        "baseline_allocated_bytes": int(baseline_allocated),
        "baseline_reserved_bytes": int(baseline_reserved),
        "peak_allocated_delta_bytes": int(max(0, peak_allocated - baseline_allocated)),
        "peak_reserved_delta_bytes": int(max(0, peak_reserved - baseline_reserved)),
        "cuda_synchronized_around_timing": True,
        "peak_memory_reset_before_timing": True,
        "stage3_preflight_timing_comparison_note": "Separate full-test-split timing window; not comparable to Stage 3 batch-size-1 per-trajectory preflight timings.",
    }
    model.cpu()
    del model, observations_gpu, init_gpu
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return {"metrics": aggregate, "temporal_windows": windows, "timing": timing}


def read_historical_t100_comparison(stage3: dict[str, Any]) -> dict[str, Any]:
    final = json.loads(M1_FINAL_TEST.read_text(encoding="utf-8"))
    if final["status"] != "final_test_evaluation_complete" or not final["test_accessed"]:
        raise ValueError("historical T=100 final test record is incomplete")
    if final["configuration_sha256"] != sha256_file(M1_CONFIG):
        raise ValueError("historical T=100 final test config hash mismatch")
    if final["test_pool_sha256"] != "d547669524487190d505f08551b6a30e9c6881465892a447c6ee2dfbdaa705c2":
        raise ValueError("historical T=100 test-pool hash differs from frozen benchmark")
    seeds = {int(row["seed"]): row for row in final["seeds"]}
    baseline = {}
    candidate = {}
    for seed in SEEDS:
        row = seeds[seed]
        cp_m0 = next(c for c in stage3["checkpoints"] if c["variant"] == "M0" and c["seed"] == seed)
        cp_m1 = next(c for c in stage3["checkpoints"] if c["variant"] == "M1" and c["seed"] == seed)
        if row["baseline_checkpoint_sha256"] != cp_m0["checkpoint_sha256"]:
            raise ValueError(f"historical M0 T=100 metrics checkpoint mismatch for seed {seed}")
        if row["candidate_checkpoint_sha256"] != cp_m1["checkpoint_sha256"]:
            raise ValueError(f"historical M1 T=100 metrics checkpoint mismatch for seed {seed}")
        baseline[seed] = row["baseline_test_metrics"]
        candidate[seed] = row["candidate_test_metrics"]
    return {
        "source_path": str(M1_FINAL_TEST.relative_to(PROJECT)),
        "source_sha256": sha256_file(M1_FINAL_TEST),
        "test_pool_sha256": final["test_pool_sha256"],
        "metrics_by_variant_seed": {"M0": baseline, "M1": candidate},
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return {"mean": mean, "sample_sd": math.sqrt(variance)}


def aggregate_results(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows = []
    paired_rows = []
    for length in LENGTHS:
        by_variant = {
            variant: [
                row for row in rows if row["sequence_length"] == length and row["variant"] == variant
            ]
            for variant in ("M0", "M1")
        }
        for variant, model_rows in by_variant.items():
            for metric in METRICS:
                summary = mean_sd([row["metrics"][metric] for row in model_rows])
                aggregate_rows.append(
                    {
                        "sequence_length": length,
                        "variant": variant,
                        "metric": metric,
                        **summary,
                        "seed_count": len(model_rows),
                    }
                )
        for metric in METRICS:
            differences = []
            for seed in SEEDS:
                m0 = next(r for r in by_variant["M0"] if r["seed"] == seed)["metrics"][metric]
                m1 = next(r for r in by_variant["M1"] if r["seed"] == seed)["metrics"][metric]
                differences.append(m1 - m0)
            paired_rows.append(
                {
                    "sequence_length": length,
                    "metric": metric,
                    "difference_definition": "M1 minus M0 paired by seed",
                    **mean_sd(differences),
                    "per_seed_differences": {
                        str(seed): differences[index] for index, seed in enumerate(SEEDS)
                    },
                    "inferential_significance_claimed": False,
                }
            )
    return aggregate_rows, paired_rows


def _csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> str:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir()
    hashes = {}
    labels = {
        "position_rmse_m": ("Position RMSE (m)", "position_rmse_over_time.png"),
        "velocity_rmse_mps": ("Velocity RMSE (m/s)", "velocity_rmse_over_time.png"),
        "acceleration_rmse_mps2": ("Acceleration RMSE (m/s²)", "acceleration_rmse_over_time.png"),
        "six_state_mse_scaled_units": ("Six-state MSE (scaled units)", "six_state_mse_over_time.png"),
    }
    for metric, (ylabel, filename) in labels.items():
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for length in LENGTHS:
            for variant, color, style in (
                ("M0", "#2458A6", "-"),
                ("M1", "#D14A32", "--"),
            ):
                subset = [
                    row
                    for row in rows
                    if row["sequence_length"] == length and row["variant"] == variant
                ]
                windows_by_seed = {row["seed"]: row["temporal_windows"] for row in subset}
                n_windows = len(subset[0]["temporal_windows"])
                centers = [
                    (w["start_step_1based"] + w["end_step_1based"]) / 2
                    for w in subset[0]["temporal_windows"]
                ]
                means, standard_deviations = [], []
                for index in range(n_windows):
                    values = [windows_by_seed[seed][index][metric] for seed in SEEDS]
                    summary = mean_sd(values)
                    means.append(summary["mean"])
                    standard_deviations.append(summary["sample_sd"])
                label = f"{variant}, T={length}"
                ax.plot(centers, means, color=color, linestyle=style, linewidth=2, label=label)
                ax.fill_between(
                    centers,
                    [m - sd for m, sd in zip(means, standard_deviations)],
                    [m + sd for m, sd in zip(means, standard_deviations)],
                    color=color,
                    alpha=0.12,
                )
        ax.set_xlabel("Observation step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Frozen 100-step checkpoints on long test sequences: {ylabel}")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        hashes[str(path.relative_to(PROJECT))] = sha256_file(path)
    return hashes


def _hardware() -> dict[str, Any]:
    import mamba_ssm
    import mamba_ssm.ops.selective_scan_interface as ssi

    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception as exc:
        driver = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_driver_version": driver,
        "cuda_runtime_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "mamba_ssm_version": importlib.metadata.version("mamba-ssm"),
        "mamba_package_init_sha256": sha256_file(mamba_ssm.__file__),
        "mamba_selective_scan_source_sha256": sha256_file(ssi.__file__),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }


def _write_markdown(report: dict[str, Any], aggregate_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], temporal_csv_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 3 Stage 4: Frozen 100-Step M0/M1 Evaluation on Long Test Sets",
        "",
        f"**Status:** {report['status']}",
        "",
        "This is frozen-checkpoint length extrapolation. No training, fine-tuning, checkpoint selection, normalization fitting, or Stage 5 decision was performed. All 20 evaluations used the Stage 2 test splits only after checkpoint/config/dataset integrity checks passed.",
        "",
        "## Method and integrity",
        "",
        f"Loaded the five original M0 and five original M1 checkpoints strictly; all have {report['checkpoints'][0]['parameter_count']} trainable parameters. Batch size {BATCH_SIZE}; {WARMUP_COUNT} warmup and {TIMED_REPETITIONS} timed full-split repetitions. Metrics match the established evaluator: joint x/y (or vx/vy, ax/ay) RMSE after scaling errors by 1,000, plus six-state MSE in scaled-state units. Temporal windows are 10 steps; six-state MSE is added per interval.",
        "",
        "Integrity checks passed for Stage 3 checkpoint/config records, M1 train-only buffers, Stage 2 bundle/test manifests, split and tensor hashes, shapes, finiteness, trajectory IDs, and initialization alignment. Test IDs were disjoint across splits and lengths. No historical T=100 test file was loaded; its archived final metrics are used only for descriptive comparison.",
        "",
        "## Full-horizon results",
        "",
        "Values are mean ± sample SD across the five seeds.",
        "",
        "| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE (scaled) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    table_lookup = {(r['sequence_length'], r['variant'], r['metric']): r for r in aggregate_rows}
    for length in LENGTHS:
        for variant in ("M0", "M1"):
            vals = []
            for metric in METRICS:
                row = table_lookup[(length, variant, metric)]
                vals.append(f"{row['mean']:.6g} ± {row['sample_sd']:.3g}")
            lines.append(f"| {length} | {variant} | " + " | ".join(vals) + " |")
    lines.extend(["", "### Paired per-seed differences (M1 − M0)", "", "Observed paired differences; no inferential significance is claimed.", "", "| T | Metric | Mean difference | Sample SD |", "|---:|---|---:|---:|"])
    for row in paired_rows:
        lines.append(
            f"| {row['sequence_length']} | {row['metric']} | {row['mean']:.6g} | {row['sample_sd']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Historical T=100 comparison",
            "",
            "Archived final T=100 test results are shown descriptively; the 100-step test set was not reopened or reevaluated.",
            "",
            "| T | Variant | Position RMSE (m) | Velocity RMSE (m/s) | Acceleration RMSE (m/s²) | Six-state MSE (scaled) |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for variant in ("M0", "M1"):
        metrics_by_seed = report["historical_T100_comparison"]["metrics_by_variant_seed"][variant]
        vals = []
        for metric in METRICS:
            summary = mean_sd([metrics_by_seed[seed][metric] for seed in SEEDS])
            vals.append(f"{summary['mean']:.6g} ± {summary['sample_sd']:.3g}")
        lines.append("| 100 | " + variant + " | " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Temporal behavior and resource summary",
            "",
            "The four temporal plots compare M0/M1 at T=200/T=400 using the mean over seeds with ±1 sample SD shading. Exact per-seed 10-step metrics, including interval six-state MSE, are in `temporal_window_metrics.csv`.",
            "",
            "| T | Variant | Mean ms/trajectory | Mean ms/observation | Peak allocated MB (max) | Peak reserved MB (max) |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for length in LENGTHS:
        for variant in ("M0", "M1"):
            timing_rows = [
                row["timing"]
                for row in report["per_seed_results"]
                if row["sequence_length"] == length and row["variant"] == variant
            ]
            mean_per_traj = sum(row["seconds_per_trajectory"] for row in timing_rows) / len(timing_rows)
            mean_per_obs = sum(row["seconds_per_observation"] for row in timing_rows) / len(timing_rows)
            max_alloc = max(row["peak_allocated_bytes"] for row in timing_rows) / (1024**2)
            max_reserved = max(row["peak_reserved_bytes"] for row in timing_rows) / (1024**2)
            lines.append(
                f"| {length} | {variant} | {mean_per_traj*1000:.3f} | {mean_per_obs*1000:.5f} | {max_alloc:.2f} | {max_reserved:.2f} |"
            )
    lines.extend(
        [
            "",
            "These Stage 4 times cover full-test-split inference with batch size 16 and GPU-resident inputs. They are separate from Stage 3’s batch-size-1 single-trajectory preflight timings.",
            "",
            "The fixed-law T=400 test distribution has much larger range and state magnitudes than the T=100 training regime. Long-horizon errors therefore describe this frozen-model extrapolation under the Stage 2 physical law. They do not establish robustness to a new motion/noise regime and were not used to make configuration or training decisions.",
            "",
            "## Provenance and output files",
            "",
            f"GPU: {report['hardware']['gpu_name']}; driver {report['hardware']['gpu_driver_version']}; CUDA {report['hardware']['cuda_runtime_version']}; PyTorch {report['hardware']['pytorch_version']}; mamba-ssm {report['hardware']['mamba_ssm_version']}. Exact checkpoint, config, dataset, source, and output hashes are in `evaluation_report.json` and `provenance.json`.",
            "",
            "- `per_seed_results.csv`: 20 checkpoint/horizon aggregate rows.",
            "- `temporal_window_metrics.csv`: 200 checkpoint/interval rows.",
            "- `aggregate_results.csv`: mean and sample SD by variant/horizon/metric.",
            "- `paired_differences.csv`: per-seed M1−M0 differences and summaries.",
            "- `plots/`: four temporal plots.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 4 output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "stage": "Phase 3 Stage 4 frozen T=100 checkpoint test evaluation",
        "status": "RUNNING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "training": False,
            "fine_tuning": False,
            "test_based_checkpoint_selection": False,
            "normalization_refit": False,
            "historical_T100_test_reopened": False,
            "stage5_decisions_made": False,
        },
        "seeds": list(SEEDS),
        "lengths": list(LENGTHS),
        "test_trajectory_count_per_length": EXPECTED_SPLIT_SIZE,
        "batch_size": BATCH_SIZE,
        "warmup_count": WARMUP_COUNT,
        "timed_repetitions": TIMED_REPETITIONS,
        "metric_definitions": {
            "position_rmse_m": "sqrt(mean over trajectory/time of squared x-error + squared y-error)) * 1000",
            "velocity_rmse_mps": "sqrt(mean over trajectory/time of squared vx-error + squared vy-error)) * 1000",
            "acceleration_rmse_mps2": "sqrt(mean over trajectory/time of squared ax-error + squared ay-error)) * 1000",
            "six_state_mse_scaled_units": "mean squared error over batch, six scaled state channels, and time",
            "temporal_interval_steps": 10,
        },
        "conditions": {},
        "checkpoint_integrity": {},
        "dataset_integrity": {},
        "per_seed_results": [],
        "temporal_window_results": [],
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; frozen Mamba evaluation cannot proceed")
        hardware = _hardware()
        if not hardware["cuda_available"]:
            raise RuntimeError("CUDA is unavailable")
        report["hardware"] = hardware

        stage3_path = STAGE3_DIR / "preflight_report.json"
        stage3_hash = verify_sidecar(stage3_path)
        stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
        if stage3["status"] != "PASS":
            raise ValueError("Stage 3 preflight report is not PASS")
        if stage3_hash != sha256_file(stage3_path):
            raise ValueError("Stage 3 report hash changed")
        checkpoints, checkpoint_integrity = _verify_stage3_integrity(stage3)
        report["checkpoint_integrity"] = checkpoint_integrity
        report["conditions"]["checkpoint_config_hashes"] = "PASS"
        report["conditions"]["strict_loading_all_ten_checkpoints"] = "PASS"
        report["conditions"]["M1_normalization_buffers_and_training_records"] = "PASS"

        stage2_path = STAGE2_REPORT
        stage2_hash = verify_sidecar(stage2_path)
        stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
        if stage2["status"] != "passed" or stage2["test_performance_evaluated"]:
            raise ValueError("Stage 2 frozen dataset validation status is invalid")
        test_payloads, dataset_integrity = verify_long_test_datasets(stage2)
        report["dataset_integrity"] = {
            "stage2_report_path": str(stage2_path.relative_to(PROJECT)),
            "stage2_report_sha256": stage2_hash,
            **dataset_integrity,
        }
        report["conditions"]["test_manifests_and_tensor_hashes"] = "PASS"
        report["conditions"]["test_shapes_finiteness_counts_alignment"] = "PASS"
        report["conditions"]["test_IDs_match_frozen_manifests"] = "PASS"

        historical = read_historical_t100_comparison(stage3)
        report["historical_T100_comparison"] = historical
        report["conditions"]["historical_T100_comparison_uses_archived_results_only"] = "PASS"

        checkpoint_metadata = []
        for checkpoint in checkpoints:
            checkpoint_metadata.append(
                {key: value for key, value in checkpoint.items() if key != "model"}
            )
        report["checkpoints"] = checkpoint_metadata

        # No test inference begins until every checkpoint, config, test file,
        # tensor, initialization context, and trajectory ID has passed above.
        device = torch.device("cuda:0")
        for checkpoint in checkpoints:
            for length in LENGTHS:
                print(
                    f"Evaluating {checkpoint['variant']} seed {checkpoint['seed']} on T={length}",
                    flush=True,
                )
                result = evaluate_one(
                    checkpoint,
                    length,
                    test_payloads[length],
                    device,
                )
                manifest_record = dataset_integrity["lengths"][length]
                row = {
                    "variant": checkpoint["variant"],
                    "seed": checkpoint["seed"],
                    "sequence_length": length,
                    "trajectory_count": EXPECTED_SPLIT_SIZE,
                    "checkpoint_path": checkpoint["checkpoint_path"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "config_path": checkpoint["config_path"],
                    "config_sha256": checkpoint["config_sha256"],
                    "training_record_path": checkpoint["training_record_path"],
                    "training_record_sha256": checkpoint["training_record_sha256"],
                    "dataset_manifest_path": manifest_record["bundle_manifest_path"],
                    "dataset_manifest_sha256": manifest_record["bundle_manifest_sha256"],
                    "test_split_manifest_path": manifest_record["split_manifest_path"],
                    "test_split_manifest_sha256": manifest_record["split_manifest_sha256"],
                    "test_file_path": manifest_record["test_file_path"],
                    "test_file_sha256": manifest_record["test_file_sha256"],
                    "parameter_count": checkpoint["parameter_count"],
                    "metrics": result["metrics"],
                    "temporal_windows": result["temporal_windows"],
                    "timing": result["timing"],
                }
                report["per_seed_results"].append(row)
                for window in result["temporal_windows"]:
                    report["temporal_window_results"].append(
                        {
                            "variant": checkpoint["variant"],
                            "seed": checkpoint["seed"],
                            "sequence_length": length,
                            **window,
                        }
                    )

        aggregate_rows, paired_rows = aggregate_results(report["per_seed_results"])
        report["aggregate_results"] = aggregate_rows
        report["paired_differences_M1_minus_M0"] = paired_rows
        report["conditions"].update(
            {
                "all_twenty_checkpoint_length_evaluations": "PASS",
                "all_full_horizon_metrics": "PASS",
                "all_ten_step_metrics_including_six_state_MSE": "PASS",
                "paired_differences_reported_descriptively": "PASS",
                "no_training_or_checkpoint_updates": "PASS",
                "no_T100_historical_artifact_modified": "PASS",
                "no_stage5_decisions_made": "PASS",
            }
        )
        report["status"] = "PASS"

    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["conditions"]["stopped_before_completion"] = report["error"]
        write_json(OUTPUT_DIR / "evaluation_report.json", report)
        (OUTPUT_DIR / "evaluation_report.md").write_text(
            "# Phase 3 Stage 4 Evaluation\n\n**Status: FAIL**\n\nStopped before completing evaluation.\n\n"
            f"Error: `{report['error']}`\n",
            encoding="utf-8",
        )
        raise

    per_seed_csv_rows = []
    for row in report["per_seed_results"]:
        per_seed_csv_rows.append(
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "sequence_length": row["sequence_length"],
                "trajectory_count": row["trajectory_count"],
                **row["metrics"],
                "parameter_count": row["parameter_count"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "config_sha256": row["config_sha256"],
                "dataset_manifest_sha256": row["dataset_manifest_sha256"],
                "test_file_sha256": row["test_file_sha256"],
                "mean_inference_seconds": row["timing"]["mean_full_split_inference_seconds"],
                "seconds_per_trajectory": row["timing"]["seconds_per_trajectory"],
                "seconds_per_observation": row["timing"]["seconds_per_observation"],
                "peak_allocated_bytes": row["timing"]["peak_allocated_bytes"],
                "peak_reserved_bytes": row["timing"]["peak_reserved_bytes"],
                "batch_size": row["timing"]["batch_size"],
                "warmup_count": row["timing"]["warmup_count"],
                "timed_repetitions": row["timing"]["timed_repetitions"],
            }
        )
    temporal_csv_rows = [
        {
            "variant": row["variant"],
            "seed": row["seed"],
            "sequence_length": row["sequence_length"],
            "start_step_1based": row["start_step_1based"],
            "end_step_1based": row["end_step_1based"],
            **{metric: row[metric] for metric in METRICS},
        }
        for row in report["temporal_window_results"]
    ]

    per_seed_csv_sha = _csv_write(OUTPUT_DIR / "per_seed_results.csv", per_seed_csv_rows)
    temporal_csv_sha = _csv_write(
        OUTPUT_DIR / "temporal_window_metrics.csv", temporal_csv_rows
    )
    aggregate_csv_sha = _csv_write(OUTPUT_DIR / "aggregate_results.csv", aggregate_rows)
    paired_csv_sha = _csv_write(OUTPUT_DIR / "paired_differences.csv", paired_rows)
    plot_hashes = write_plots(report["per_seed_results"], OUTPUT_DIR)
    report["output_artifact_hashes"] = {
        "per_seed_results.csv": per_seed_csv_sha,
        "temporal_window_metrics.csv": temporal_csv_sha,
        "aggregate_results.csv": aggregate_csv_sha,
        "paired_differences.csv": paired_csv_sha,
        "plots": plot_hashes,
    }
    report["software_and_sources"] = {
        "hardware": hardware,
        "stage4_runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_source_hashes": verify_recorded_source_hashes(stage3),
    }

    markdown = _write_markdown(report, aggregate_rows, paired_rows, temporal_csv_rows)
    md_path = OUTPUT_DIR / "evaluation_report.md"
    md_path.write_text(markdown, encoding="utf-8")
    md_sha = sha256_file(md_path)
    md_path.with_name(md_path.name + ".sha256").write_text(
        f"{md_sha}  {md_path.name}\n", encoding="ascii"
    )
    json_sha = write_json(OUTPUT_DIR / "evaluation_report.json", report)
    provenance = {
        "stage": "Phase 3 Stage 4 frozen evaluation provenance",
        "status": "PASS",
        "checkpoint_records": [
            {
                "variant": c["variant"],
                "seed": c["seed"],
                "checkpoint_path": c["checkpoint_path"],
                "checkpoint_sha256": c["checkpoint_sha256"],
                "config_path": c["config_path"],
                "config_sha256": c["config_sha256"],
                "training_record_path": c["training_record_path"],
                "training_record_sha256": c["training_record_sha256"],
                "feature_statistics": c["feature_statistics"],
            }
            for c in report["checkpoints"]
        ],
        "dataset_integrity": report["dataset_integrity"],
        "model_source_hashes": report["software_and_sources"]["model_source_hashes"],
        "hardware": report["hardware"],
        "output_artifact_hashes": {
            **report["output_artifact_hashes"],
            "evaluation_report.json": json_sha,
            "evaluation_report.md": md_sha,
        },
    }
    provenance_sha = write_json(OUTPUT_DIR / "provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_directory": str(OUTPUT_DIR),
                "evaluations": len(report["per_seed_results"]),
                "temporal_rows": len(temporal_csv_rows),
                "evaluation_report_sha256": json_sha,
                "markdown_report_sha256": md_sha,
                "provenance_sha256": provenance_sha,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
