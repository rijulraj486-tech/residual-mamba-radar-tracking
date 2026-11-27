#!/usr/bin/env python3
"""No-training variable-length and causality preflight for frozen M0/M1."""

from __future__ import annotations

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

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.paper_v4_mamba_improvement.model_variants import (
    NORMALIZED_FEATURE_INDICES,
    TrainNormalizedFeatureTracker,
)
from radar_tracking.data.independent_dataset import (
    tensor_sha256,
    validate_split_tensors,
)
from radar_tracking.models.tracker_model import BiMambaTrackerModel, wrap_to_pi


OUTPUT_DIR = (
    PROJECT
    / "artifacts/results/synthetic/paper_v4_long_sequence/stage3_frozen_preflight_v1"
)
M0_ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
M1_ROOT = (
    PROJECT
    / "artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2"
)
LONG_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence"
M0_CONFIG = PROJECT / "configs/paper_v4_mamba_improvement/baseline_reference.json"
M1_CONFIG = PROJECT / "configs/paper_v4_mamba_improvement/features_train_only_normalization_five_seed_confirmation_v2.json"
M1_PILOT_CONFIG = PROJECT / "configs/paper_v4_mamba_improvement/features_train_only_normalization_seed11.json"
SEEDS = (11, 22, 33, 44, 55)
LENGTHS = (100, 200, 400)
BATCH_SIZE = 1
WARMUP_COUNT = 2
TIMED_REPETITIONS = 5
CAUSAL_ATOL = 1e-5
CAUSAL_RTOL = 1e-5
EXTENSION_ATOL = 1e-4
EXTENSION_RTOL = 1e-4
FEATURE_ATOL = 2e-5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def write_markdown(path: Path, value: str) -> str:
    path.write_text(value, encoding="utf-8")
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def git_blob_sha256(relative_path: str) -> str | None:
    try:
        content = subprocess.check_output(
            ["git", "show", f"HEAD:{relative_path}"], cwd=PROJECT
        )
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(content).hexdigest()


def source_hash_inventory() -> dict[str, dict[str, str | bool | None]]:
    current_paths = {
        "tracker_model.py": PROJECT / "src/radar_tracking/models/tracker_model.py",
        "bimamba_encoder.py": PROJECT / "src/radar_tracking/models/bimamba_encoder.py",
        "prediction_head.py": PROJECT / "src/radar_tracking/models/prediction_head.py",
        "model_variants.py": PROJECT / "experiments/paper_v4_mamba_improvement/model_variants.py",
        "benchmark_mamba.py": PROJECT / "src/radar_tracking/cli/benchmark_mamba.py",
        "run_screening_pilot.py": PROJECT / "experiments/paper_v4_mamba_improvement/run_screening_pilot.py",
        "run_feature_norm_five_seed_validation.py": PROJECT / "experiments/paper_v4_mamba_improvement/run_feature_norm_five_seed_validation.py",
        "data_generation.py": PROJECT / "src/radar_tracking/data/data_generation.py",
        "constant_acceleration.py": PROJECT / "src/radar_tracking/dynamics/constant_acceleration.py",
        "config.py": PROJECT / "src/radar_tracking/config.py",
        "stage3_frozen_preflight.py": Path(__file__).resolve(),
    }
    sources: dict[str, dict[str, str | bool | None]] = {}
    for name, path in current_paths.items():
        relative = str(path.relative_to(PROJECT))
        sources[name] = {
            "path": relative,
            "current_sha256": sha256_file(path),
            "git_head_sha256": git_blob_sha256(relative),
        }
    return sources


def _verify_recorded_sources(
    expected_hashes: dict[str, str], current_sources: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, expected in expected_hashes.items():
        path = PROJECT / relative
        current_hash = sha256_file(path) if path.is_file() else None
        head_hash = git_blob_sha256(relative)
        result[relative] = {
            "training_record_sha256": expected,
            "current_workspace_sha256": current_hash,
            "git_head_sha256": head_hash,
            "historical_source_verified": expected == current_hash or expected == head_hash,
            "current_matches_training_source": expected == current_hash,
        }
        if not result[relative]["historical_source_verified"]:
            raise ValueError(
                f"historical source hash mismatch for {relative}: expected {expected}, "
                f"current {current_hash}, git HEAD {head_hash}"
            )
    return result


def load_training_inputs() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}

    historic = torch.load(
        M0_ROOT / "fixed_train_val.pt", map_location="cpu", weights_only=True
    )
    train100 = historic["train"]
    result[100] = {
        "state": train100["state"][:BATCH_SIZE].contiguous(),
        "observations": train100["observations"][:BATCH_SIZE].contiguous(),
        "initial_state": train100["initial_state"][:BATCH_SIZE].contiguous(),
        "initialization_observations": None,
        "trajectory_id": "historical_100_train_index_0",
        "source_file": str((M0_ROOT / "fixed_train_val.pt").relative_to(PROJECT)),
        "source_split": "train",
    }

    for length in (200, 400):
        path = LONG_ROOT / f"T{length}_fixed_profile_v1/splits/train.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        selection = {
            "state": payload["state"][:BATCH_SIZE].contiguous(),
            "observations": payload["observations"][:BATCH_SIZE].contiguous(),
            "initial_state": payload["initial_state"][:BATCH_SIZE].contiguous(),
            "initialization_observations": payload[
                "initialization_observations"
            ][:BATCH_SIZE].contiguous(),
            "trajectory_id": payload["trajectory_ids"][0],
            "source_file": str(path.relative_to(PROJECT)),
            "source_split": "train",
        }
        validate_split_tensors(
            state=selection["state"],
            observations=selection["observations"],
            initial_state=selection["initial_state"],
            initialization_observations=selection["initialization_observations"],
            trajectory_ids=[selection["trajectory_id"]],
            sequence_length=length,
        )
        result[length] = selection
    return result


def _m0_constructor_config(run_config: dict[str, Any]) -> dict[str, Any]:
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


def _m1_constructor_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dim": 6,
        "obs_dim": 2,
        "dt": 1.0,
        "embed_dim": int(model_config["embed_dim"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "num_layers": int(model_config["num_layers"]),
        "d_state": int(model_config["d_state"]),
        "d_conv": int(model_config["d_conv"]),
        "expand": int(model_config["expand"]),
        "dropout": float(model_config["dropout"]),
        "encoder_direction": "forward",
        "task": "radar_benchmark",
    }


def verify_and_load_all_checkpoints(
    report: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    m0_config = json.loads(M0_CONFIG.read_text(encoding="utf-8"))
    m1_config = json.loads(M1_CONFIG.read_text(encoding="utf-8"))
    m1_pilot_config = json.loads(M1_PILOT_CONFIG.read_text(encoding="utf-8"))
    m1_copy_config_path = M1_ROOT / "config.json"
    frozen_path = M1_ROOT / "frozen_validation_hashes.json"
    selection_path = M1_ROOT / "validation_selection_manifest.json"
    aggregate_path = M1_ROOT / "validation_aggregate.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    m1_config_hash = sha256_file(M1_CONFIG)
    if sha256_file(m1_copy_config_path) != m1_config_hash:
        raise ValueError("M1 run config copy does not match the selected config file")
    if frozen["config_sha256"] != m1_config_hash:
        raise ValueError("M1 config hash does not match frozen validation hashes")
    if selection["selected_configuration_sha256"] != m1_config_hash:
        raise ValueError("M1 config hash does not match frozen selection manifest")
    if sha256_file(selection_path) != frozen["validation_selection_manifest_sha256"]:
        raise ValueError("M1 validation-selection manifest hash mismatch")
    if sha256_file(aggregate_path) != frozen["validation_aggregate_sha256"]:
        raise ValueError("M1 validation aggregate hash mismatch")
    if sha256_file(M1_PILOT_CONFIG) != m1_config["pilot_config_sha256"]:
        raise ValueError("M1 seed-11 pilot config hash mismatch")
    if m1_config["experiment"] != selection["experiment"]:
        raise ValueError("M1 config and selection experiment names differ")
    if m1_config["data"]["train_validation_pool_sha256"] != sha256_file(
        M0_ROOT / "fixed_train_val.pt"
    ):
        raise ValueError("M1 training pool hash mismatch")

    m0_recorded_hash = selection["source_file_sha256"].get(
        "configs/paper_v4_mamba_improvement/baseline_reference.json"
    )
    m0_config_hash = sha256_file(M0_CONFIG)
    if m0_recorded_hash != m0_config_hash:
        raise ValueError("M0 baseline config hash does not match frozen M1 provenance")
    m0_config_payload = m0_config["model"]
    if m0_config_payload["parameter_count"] != 619014:
        raise ValueError("M0 reference config parameter count is not 619,014")
    if m1_config["model"]["parameter_count_expected"] != 619014:
        raise ValueError("M1 confirmation config parameter count is not 619,014")

    recorded_m1_sources = selection["source_file_sha256"]
    current_sources = report["software_and_sources"]["project_sources"]
    m1_source_verification = _verify_recorded_sources(
        recorded_m1_sources, current_sources
    )
    report["provenance_checks"]["m1_original_source_hashes"] = m1_source_verification

    # M0's original run records source hashes independently for each seed.
    loaded: list[dict[str, Any]] = []
    for seed in SEEDS:
        m0_run_path = M0_ROOT / f"seed_{seed}/run_metrics.json"
        m0_run = json.loads(m0_run_path.read_text(encoding="utf-8"))
        m0_record = m0_run["mamba"]
        m0_ckpt = M0_ROOT / f"seed_{seed}/{m0_record['checkpoint']}"
        expected_m0_hash = m0_record["checkpoint_sha256"]
        actual_m0_hash = sha256_file(m0_ckpt)
        if actual_m0_hash != expected_m0_hash:
            raise ValueError(f"M0 seed {seed} checkpoint SHA-256 mismatch")
        if selection["baseline_checkpoints"][str(seed)] != expected_m0_hash:
            raise ValueError(f"M0 seed {seed} checkpoint hash differs from frozen selection")
        if m0_record["parameter_count"] != 619014:
            raise ValueError(f"M0 seed {seed} recorded parameter count mismatch")
        if int(m0_record["config"]["seed"]) != seed:
            raise ValueError(f"M0 seed {seed} run config seed mismatch")
        if not m0_run.get("source_file_sha256"):
            raise ValueError(f"M0 seed {seed} lacks source hashes")
        m0_source_verification = _verify_recorded_sources(
            m0_run["source_file_sha256"], current_sources
        )

        m1_run_path = M1_ROOT / f"seed_{seed}/run_metrics.json"
        m1_summary_path = M1_ROOT / f"seed_{seed}/seed_summary.json"
        m1_stats_path = M1_ROOT / f"seed_{seed}/train_only_feature_statistics.json"
        m1_run = json.loads(m1_run_path.read_text(encoding="utf-8"))
        m1_summary = json.loads(m1_summary_path.read_text(encoding="utf-8"))
        m1_stats = json.loads(m1_stats_path.read_text(encoding="utf-8"))
        m1_ckpt = M1_ROOT / f"seed_{seed}/{m1_run['checkpoint']}"
        expected_m1_hash = m1_run["checkpoint_sha256"]
        actual_m1_hash = sha256_file(m1_ckpt)
        if actual_m1_hash != expected_m1_hash:
            raise ValueError(f"M1 seed {seed} checkpoint SHA-256 mismatch")
        if m1_summary["candidate_checkpoint_sha256"] != expected_m1_hash:
            raise ValueError(f"M1 seed {seed} summary checkpoint hash mismatch")
        if selection["candidate_checkpoints"][str(seed)] != expected_m1_hash:
            raise ValueError(f"M1 seed {seed} checkpoint differs from frozen selection")
        stats_hash = sha256_file(m1_stats_path)
        if m1_summary["feature_statistics_sha256"] != stats_hash:
            raise ValueError(f"M1 seed {seed} feature-statistics hash mismatch")
        if m1_run["parameter_count"]["total"] != 619014:
            raise ValueError(f"M1 seed {seed} recorded parameter count mismatch")
        training_config = m1_run["training_config"]
        if training_config["seed"] != seed:
            raise ValueError(f"M1 seed {seed} run config seed mismatch")

        expected_indices = list(NORMALIZED_FEATURE_INDICES)
        preserved_indices = [1, 2, 6]
        if m1_stats["normalized_token_indices_zero_based"] != expected_indices:
            raise ValueError(f"M1 seed {seed} normalized feature indices mismatch")
        if m1_stats["preserved_unscaled_token_indices_zero_based"] != preserved_indices:
            raise ValueError(f"M1 seed {seed} preserved feature indices mismatch")
        if training_config["feature_normalization"]["normalized_indices"] != expected_indices:
            raise ValueError(f"M1 seed {seed} run config normalization indices mismatch")
        if training_config["feature_normalization"]["preserved_indices"] != preserved_indices:
            raise ValueError(f"M1 seed {seed} run config preserved indices mismatch")
        if training_config["feature_normalization"]["std_floor"] != 1e-4:
            raise ValueError(f"M1 seed {seed} standard-deviation floor mismatch")
        if m1_stats["normalization_formula"] != "(x - train_mean) / max(train_population_std, 1e-4)":
            raise ValueError(f"M1 seed {seed} normalization formula mismatch")
        if m1_stats["statistics_source"] != "train split only":
            raise ValueError(f"M1 seed {seed} stats are not recorded as train-only")
        if m1_stats["train_trajectory_count"] != 819 or m1_stats["time_points_per_trajectory"] != 100:
            raise ValueError(f"M1 seed {seed} statistics training dimensions mismatch")

        mean = torch.tensor(m1_stats["feature_mean_21"], dtype=torch.float32)
        std = torch.tensor(m1_stats["feature_std_21"], dtype=torch.float32)
        if mean.shape != (21,) or std.shape != (21,):
            raise ValueError(f"M1 seed {seed} feature statistics shape mismatch")
        if torch.any(std < 1e-4):
            raise ValueError(f"M1 seed {seed} stored feature std is below the floor")

        # Verify architecture state dictionaries strictly on CPU before any CUDA run.
        m0_kwargs = _m0_constructor_config(m0_record["config"])
        m0_model = BiMambaTrackerModel(**m0_kwargs)
        m0_state = torch.load(m0_ckpt, map_location="cpu", weights_only=True)
        m0_model.load_state_dict(m0_state, strict=True)
        m0_parameters = sum(p.numel() for p in m0_model.parameters() if p.requires_grad)
        if m0_parameters != 619014:
            raise ValueError(f"M0 seed {seed} strict-loaded parameter count mismatch")

        m1_kwargs = _m1_constructor_config(m1_config["model"])
        m1_model = TrainNormalizedFeatureTracker(
            feature_mean=mean, feature_std=std, **m1_kwargs
        )
        m1_state = torch.load(m1_ckpt, map_location="cpu", weights_only=True)
        m1_model.load_state_dict(m1_state, strict=True)
        if not torch.equal(m1_model.feature_mean.cpu(), m1_state["feature_mean"].cpu()):
            raise ValueError(f"M1 seed {seed} loaded feature means differ from checkpoint")
        if not torch.equal(m1_model.feature_std.cpu(), m1_state["feature_std"].cpu()):
            raise ValueError(f"M1 seed {seed} loaded feature stds differ from checkpoint")
        if not torch.allclose(m1_model.feature_mean.cpu(), mean, atol=1e-6, rtol=1e-6):
            raise ValueError(f"M1 seed {seed} checkpoint feature means differ from training record")
        if not torch.allclose(m1_model.feature_std.cpu(), std, atol=1e-6, rtol=1e-6):
            raise ValueError(f"M1 seed {seed} checkpoint feature stds differ from training record")
        m1_parameters = sum(p.numel() for p in m1_model.parameters() if p.requires_grad)
        if m1_parameters != 619014:
            raise ValueError(f"M1 seed {seed} strict-loaded parameter count mismatch")

        loaded.extend(
            [
                {
                    "variant": "M0",
                    "seed": seed,
                    "model": m0_model,
                    "checkpoint_path": str(m0_ckpt.relative_to(PROJECT)),
                    "checkpoint_sha256": actual_m0_hash,
                    "run_record_path": str(m0_run_path.relative_to(PROJECT)),
                    "run_record_sha256": sha256_file(m0_run_path),
                    "training_config": m0_record["config"],
                    "training_config_canonical_sha256": canonical_sha256(m0_record["config"]),
                    "config_path": str(M0_CONFIG.relative_to(PROJECT)),
                    "config_sha256": m0_config_hash,
                    "source_verification": m0_source_verification,
                    "parameter_count_trainable": m0_parameters,
                    "strict_state_dict_load": True,
                },
                {
                    "variant": "M1",
                    "seed": seed,
                    "model": m1_model,
                    "checkpoint_path": str(m1_ckpt.relative_to(PROJECT)),
                    "checkpoint_sha256": actual_m1_hash,
                    "run_record_path": str(m1_run_path.relative_to(PROJECT)),
                    "run_record_sha256": sha256_file(m1_run_path),
                    "training_config": training_config,
                    "training_config_canonical_sha256": canonical_sha256(training_config),
                    "config_path": str(M1_CONFIG.relative_to(PROJECT)),
                    "config_sha256": m1_config_hash,
                    "pilot_config_path": str(M1_PILOT_CONFIG.relative_to(PROJECT)),
                    "pilot_config_sha256": sha256_file(M1_PILOT_CONFIG),
                    "feature_statistics_path": str(m1_stats_path.relative_to(PROJECT)),
                    "feature_statistics_sha256": stats_hash,
                    "feature_mean_21": m1_model.feature_mean.cpu().tolist(),
                    "feature_std_21": m1_model.feature_std.cpu().tolist(),
                    "normalized_indices": expected_indices,
                    "preserved_indices": preserved_indices,
                    "std_floor": 1e-4,
                    "source_verification": m1_source_verification,
                    "parameter_count_trainable": m1_parameters,
                    "strict_state_dict_load": True,
                },
            ]
        )

    report["provenance_checks"]["m0_baseline_config_sha256"] = m0_config_hash
    report["provenance_checks"]["m1_confirmation_config_sha256"] = m1_config_hash
    report["provenance_checks"]["m1_pilot_config_sha256"] = sha256_file(M1_PILOT_CONFIG)
    report["provenance_checks"]["m1_frozen_validation_hashes"] = {
        "file_sha256": sha256_file(frozen_path),
        **frozen,
    }
    report["provenance_checks"]["m1_validation_selection_manifest_sha256"] = sha256_file(
        selection_path
    )
    report["provenance_checks"]["m1_validation_aggregate_sha256"] = sha256_file(
        aggregate_path
    )
    return loaded


def _expected_base_tokens(
    model: BiMambaTrackerModel,
    observations: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    ranges = observations[:, 0, :]
    bearings = observations[:, 1, :]
    cartesian = model.measurement_to_cartesian(observations)
    batch, _, length = cartesian.shape
    coarse = torch.zeros(
        batch, 6, length, device=observations.device, dtype=observations.dtype
    )
    coarse[:, 0, :] = cartesian[:, 0, :]
    coarse[:, 3, :] = cartesian[:, 1, :]
    coarse[:, 1, 0] = init_state[:, 1, 0]
    coarse[:, 4, 0] = init_state[:, 4, 0]
    if length > 1:
        coarse[:, 1, 1:] = (cartesian[:, 0, 1:] - cartesian[:, 0, :-1]) / model.dt
        coarse[:, 4, 1:] = (cartesian[:, 1, 1:] - cartesian[:, 1, :-1]) / model.dt
    coarse[:, 2, 0] = init_state[:, 2, 0]
    coarse[:, 5, 0] = init_state[:, 5, 0]
    if length > 1:
        coarse[:, 2, 1] = coarse[:, 2, 0]
        coarse[:, 5, 1] = coarse[:, 5, 0]
    if length > 2:
        coarse[:, 2, 2:] = (coarse[:, 1, 2:] - coarse[:, 1, 1:-1]) / model.dt
        coarse[:, 5, 2:] = (coarse[:, 4, 2:] - coarse[:, 4, 1:-1]) / model.dt

    delta_range = torch.zeros_like(ranges)
    delta_bearing = torch.zeros_like(bearings)
    delta_x = torch.zeros_like(cartesian[:, 0, :])
    delta_y = torch.zeros_like(cartesian[:, 1, :])
    delta_range[:, 1:] = ranges[:, 1:] - ranges[:, :-1]
    delta_bearing[:, 1:] = wrap_to_pi(bearings[:, 1:] - bearings[:, :-1])
    delta_x[:, 1:] = cartesian[:, 0, 1:] - cartesian[:, 0, :-1]
    delta_y[:, 1:] = cartesian[:, 1, 1:] - cartesian[:, 1, :-1]
    init_repeat = init_state.squeeze(-1).unsqueeze(-1).expand(-1, -1, length)
    return torch.cat(
        (
            ranges.unsqueeze(1),
            torch.sin(bearings).unsqueeze(1),
            torch.cos(bearings).unsqueeze(1),
            cartesian,
            delta_range.unsqueeze(1),
            delta_bearing.unsqueeze(1),
            delta_x.unsqueeze(1),
            delta_y.unsqueeze(1),
            coarse,
            init_repeat,
        ),
        dim=1,
    )


def verify_feature_alignment(
    model: BiMambaTrackerModel,
    observations: torch.Tensor,
    init_state: torch.Tensor,
) -> dict[str, Any]:
    base = _expected_base_tokens(model, observations, init_state)
    if isinstance(model, TrainNormalizedFeatureTracker):
        indices = torch.as_tensor(NORMALIZED_FEATURE_INDICES, device=observations.device)
        expected = base.clone()
        expected[:, indices, :] = (
            base[:, indices, :] - model.feature_mean[indices][None, :, None]
        ) / model.feature_std[indices].clamp_min(1e-4)[None, :, None]
        actual, _ = model.build_tokens(observations, init_state)
    else:
        expected = base
        actual, _ = model.build_tokens(observations, init_state)
    max_abs = float((actual - expected).abs().max().item())
    passed = bool(torch.allclose(actual, expected, atol=FEATURE_ATOL, rtol=1e-6))
    if actual.shape != (observations.shape[0], 21, observations.shape[-1]):
        raise ValueError(f"token feature shape mismatch: {tuple(actual.shape)}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("non-finite token feature values")
    if not passed:
        raise ValueError(f"token feature alignment mismatch; max abs error={max_abs}")
    return {"pass": True, "shape": list(actual.shape), "max_abs_error": max_abs}


def _max_differences(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    difference = (reference - candidate).abs()
    relative = difference / reference.abs().clamp_min(1e-8)
    return {
        "max_absolute_difference": float(difference.max().item()),
        "max_relative_difference": float(relative.max().item()),
        "allclose_at_declared_tolerance": bool(
            torch.allclose(reference, candidate, atol=CAUSAL_ATOL, rtol=CAUSAL_RTOL)
        ),
    }


def predict(model: BiMambaTrackerModel, observations: torch.Tensor, init_state: torch.Tensor):
    output = model(observations=observations, init_state=init_state)
    return output["state_estimates"], output["tokens"]


def measure_inference(
    model: BiMambaTrackerModel,
    observations: torch.Tensor,
    init_state: torch.Tensor,
    sequence_length: int,
) -> dict[str, Any]:
    for _ in range(WARMUP_COUNT):
        with torch.inference_mode():
            prediction, tokens = predict(model, observations, init_state)
        del prediction, tokens
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    durations = []
    for _ in range(TIMED_REPETITIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            prediction, tokens = predict(model, observations, init_state)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        del prediction, tokens
    mean_seconds = sum(durations) / len(durations)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "batch_size": BATCH_SIZE,
        "warmup_count": WARMUP_COUNT,
        "timed_repetitions": TIMED_REPETITIONS,
        "raw_seconds": durations,
        "mean_batch_seconds": mean_seconds,
        "seconds_per_trajectory": mean_seconds / BATCH_SIZE,
        "seconds_per_observation": mean_seconds / (BATCH_SIZE * sequence_length),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_delta_bytes": max(0, peak_reserved - baseline_reserved),
        "timing_window_cuda_synchronized": True,
        "peak_stats_reset_before_window": True,
    }


def preflight_checkpoint(
    checkpoint: dict[str, Any],
    training_inputs: dict[int, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model = checkpoint["model"].to(device)
    model.eval()
    seed = checkpoint["seed"]
    result: dict[str, Any] = {
        "variant": checkpoint["variant"],
        "seed": seed,
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "parameter_count_trainable": checkpoint["parameter_count_trainable"],
        "strict_state_dict_load": checkpoint["strict_state_dict_load"],
        "evaluation_mode": not model.training,
        "gradient_disabled": True,
        "length_results": {},
        "causality": None,
        "sequence_extension": None,
    }
    context_hashes = {}
    cuda_model_name = torch.cuda.get_device_name(device)

    for length in LENGTHS:
        entry = training_inputs[length]
        observations_cpu = entry["observations"]
        init_cpu = entry["initial_state"]
        if observations_cpu.shape != (BATCH_SIZE, 2, length):
            raise ValueError(
                f"T={length}: input shape {tuple(observations_cpu.shape)} is not [1,2,T]"
            )
        if not torch.isfinite(observations_cpu).all() or not torch.isfinite(init_cpu).all():
            raise ValueError(f"T={length}: training input contains non-finite values")
        context = entry["initialization_observations"]
        if context is not None:
            context_hashes[str(length)] = tensor_sha256(context)
            if not torch.equal(observations_cpu[:, :, 0], context[:, :, 1]):
                raise ValueError(f"T={length}: saved initialization context misaligned")
        observations = observations_cpu.to(device)
        init_state = init_cpu.to(device)
        feature_alignment = verify_feature_alignment(model, observations, init_state)
        with torch.inference_mode():
            prediction, tokens = predict(model, observations, init_state)
        if tokens.shape != (BATCH_SIZE, 21, length):
            raise ValueError(f"T={length}: token output shape {tuple(tokens.shape)}")
        if prediction.shape != (BATCH_SIZE, 6, length):
            raise ValueError(f"T={length}: state output shape {tuple(prediction.shape)}")
        if not torch.isfinite(tokens).all() or not torch.isfinite(prediction).all():
            raise ValueError(f"T={length}: non-finite features or predictions")
        timing = measure_inference(model, observations, init_state, length)
        result["length_results"][str(length)] = {
            "status": "PASS",
            "input_shape": list(observations.shape),
            "input_finite": True,
            "token_shape": list(tokens.shape),
            "tokens_finite": True,
            "prediction_shape": list(prediction.shape),
            "prediction_finite": True,
            "one_prediction_per_observation": prediction.shape[-1] == observations.shape[-1],
            "feature_alignment": feature_alignment,
            "source_split": entry["source_split"],
            "source_file": entry["source_file"],
            "trajectory_id": entry["trajectory_id"],
            "initialization_context_sha256": context_hashes.get(str(length)),
            "cuda_device": cuda_model_name,
            "timing_and_memory": timing,
        }
        del prediction, tokens

    # A single T=400 training trajectory anchors both causality tests. Its
    # init_state and two-measurement context stay bit-identical throughout.
    long_input = training_inputs[400]
    obs400_cpu = long_input["observations"]
    init_cpu = long_input["initial_state"]
    context = long_input["initialization_observations"]
    if context is None:
        raise ValueError("T=400 training split has no stored initialization context")
    original_context_hash = tensor_sha256(context)
    obs400 = obs400_cpu.to(device)
    init_state = init_cpu.to(device)
    prefix = 200
    perturbed = obs400.clone()
    perturbed[:, 0, prefix:] += 0.2
    perturbed[:, 1, prefix:] = torch.remainder(
        perturbed[:, 1, prefix:] + 0.2, 2.0 * math.pi
    )
    with torch.inference_mode():
        original_prediction, _ = predict(model, obs400, init_state)
        perturbed_prediction, _ = predict(model, perturbed, init_state)
    suffix_diff = _max_differences(
        original_prediction[:, :, :prefix], perturbed_prediction[:, :, :prefix]
    )
    suffix_diff["allclose_at_declared_tolerance"] = bool(
        torch.allclose(
            original_prediction[:, :, :prefix],
            perturbed_prediction[:, :, :prefix],
            atol=CAUSAL_ATOL,
            rtol=CAUSAL_RTOL,
        )
    )
    result["causality"] = {
        "status": "PASS" if suffix_diff["allclose_at_declared_tolerance"] else "FAIL",
        "input_length": 400,
        "unchanged_prefix_length": prefix,
        "perturbed_suffix_start_index_zero_based": prefix,
        "perturbation": "range +0.2 scaled units and bearing +0.2 radians modulo 2pi",
        "absolute_tolerance": CAUSAL_ATOL,
        "relative_tolerance": CAUSAL_RTOL,
        "difference": suffix_diff,
        "initial_state_unchanged": True,
        "initialization_context_unchanged": tensor_sha256(context) == original_context_hash,
        "initialization_context_sha256": original_context_hash,
    }

    extension_results = {}
    with torch.inference_mode():
        full_prediction, _ = predict(model, obs400, init_state)
        for prefix_length in (100, 200):
            prefix_prediction, _ = predict(
                model, obs400[:, :, :prefix_length], init_state
            )
            extension_difference = _max_differences(
                prefix_prediction, full_prediction[:, :, :prefix_length]
            )
            extension_difference["allclose_at_declared_tolerance"] = bool(
                torch.allclose(
                    prefix_prediction,
                    full_prediction[:, :, :prefix_length],
                    atol=EXTENSION_ATOL,
                    rtol=EXTENSION_RTOL,
                )
            )
            extension_results[str(prefix_length)] = {
                **extension_difference,
                "absolute_tolerance": EXTENSION_ATOL,
                "relative_tolerance": EXTENSION_RTOL,
            }
    extension_passed = all(
        item["allclose_at_declared_tolerance"]
        for item in extension_results.values()
    )
    result["sequence_extension"] = {
        "status": "PASS" if extension_passed else "FAIL",
        "source_length": 400,
        "compared_prefix_lengths": extension_results,
        "same_initial_state_for_all_calls": True,
        "same_initialization_context_for_all_calls": tensor_sha256(context)
        == original_context_hash,
        "initialization_context_sha256": original_context_hash,
    }
    del original_prediction, perturbed_prediction, full_prediction, perturbed
    model.cpu()
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


def _hardware_and_versions() -> dict[str, Any]:
    try:
        import mamba_ssm
        import mamba_ssm.ops.selective_scan_interface as selective_scan_interface

        mamba_version = importlib.metadata.version("mamba-ssm")
        mamba_module = str(Path(mamba_ssm.__file__).resolve())
        selective_scan_path = Path(selective_scan_interface.__file__).resolve()
        mamba_hashes = {
            "package_init_sha256": sha256_file(mamba_module),
            "selective_scan_interface_path": str(selective_scan_path),
            "selective_scan_interface_sha256": sha256_file(selective_scan_path),
        }
    except Exception as exc:
        mamba_version = None
        mamba_hashes = {"import_error": f"{type(exc).__name__}: {exc}"}
    hardware: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "mamba_ssm_version": mamba_version,
        "mamba_source_hashes": mamba_hashes,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hardware["gpu"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "device_index": 0,
        }
        try:
            driver = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip().splitlines()[0]
        except Exception as exc:
            driver = f"unavailable: {type(exc).__name__}: {exc}"
        hardware["gpu"]["driver_version"] = driver
    return hardware


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Stage 3: Frozen M0/M1 Variable-Length Preflight",
        "",
        f"**Overall status:** {report.get('status', 'unknown')}",
        "",
        "No training, fine-tuning, checkpoint updates, or test-set performance evaluation was performed. Inputs came from training splits only.",
        "",
    ]
    if report.get("error"):
        lines.extend([f"**Stopped on:** `{report['error']}`", ""])
    lines.extend(
        [
            "## Checkpoint integrity and architecture",
            "",
            "| Variant | Seed | Strict load | Trainable parameters | Checkpoint SHA-256 | Config SHA-256 |",
            "|---|---:|---|---:|---|---|",
        ]
    )
    for item in report.get("checkpoints", []):
        lines.append(
            f"| {item['variant']} | {item['seed']} | {item.get('strict_state_dict_load', False)} | "
            f"{item.get('parameter_count_trainable', '—')} | `{item.get('checkpoint_sha256', '—')}` | "
            f"`{item.get('config_sha256', '—')}` |"
        )
    lines.extend(
        [
            "",
            "## CUDA inference summary",
            "",
            "| Length | Variant | Seed checkpoints passing shape/finite/CUDA | Mean ms/trajectory | Mean ms/observation | Peak allocated MB (max) | Peak reserved MB (max) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for length in LENGTHS:
        for variant in ("M0", "M1"):
            rows = [
                c["length_results"].get(str(length), {})
                for c in report.get("inference_results", [])
                if c["variant"] == variant
            ]
            rows = [row for row in rows if row.get("status") == "PASS"]
            times = [row["timing_and_memory"]["seconds_per_trajectory"] * 1000 for row in rows]
            per_obs = [row["timing_and_memory"]["seconds_per_observation"] * 1000 for row in rows]
            alloc = [row["timing_and_memory"]["peak_allocated_bytes"] / (1024**2) for row in rows]
            reserved = [row["timing_and_memory"]["peak_reserved_bytes"] / (1024**2) for row in rows]
            mean_time = f"{sum(times) / len(times):.3f}" if times else "—"
            mean_per_observation = f"{sum(per_obs) / len(per_obs):.5f}" if per_obs else "—"
            max_allocated = f"{max(alloc):.2f}" if alloc else "—"
            max_reserved = f"{max(reserved):.2f}" if reserved else "—"
            lines.append(
                f"| {length} | {variant} | {len(rows)}/{len(SEEDS)} | "
                f"{mean_time} | {mean_per_observation} | {max_allocated} | {max_reserved} |"
            )
    lines.extend(["", "## Causality and sequence extension", ""])
    lines.append(
        f"Future-suffix tolerance: atol={CAUSAL_ATOL:g}, rtol={CAUSAL_RTOL:g}. "
        f"Sequence-extension tolerance: atol={EXTENSION_ATOL:g}, rtol={EXTENSION_RTOL:g}."
    )
    lines.append("")
    lines.append("| Variant | Seed | Suffix perturbation | First 100 vs 400 | First 200 vs 400 |")
    lines.append("|---|---:|---|---:|---:|")
    for row in report.get("inference_results", []):
        extension = row.get("sequence_extension") or {}
        compared = extension.get("compared_prefix_lengths", {})
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row.get('causality', {}).get('status', 'NOT RUN')} | "
            f"{compared.get('100', {}).get('max_absolute_difference', float('nan')):.3g} | "
            f"{compared.get('200', {}).get('max_absolute_difference', float('nan')):.3g} |"
        )
    lines.extend(["", "## Hardware and notes", ""])
    versions = report.get("software_and_sources", {}).get("hardware", {})
    if versions:
        lines.append(
            f"GPU: {versions.get('gpu', {}).get('name', 'unavailable')}; "
            f"CUDA {versions.get('cuda_runtime_version')}; PyTorch {versions.get('pytorch_version')}; "
            f"mamba-ssm {versions.get('mamba_ssm_version')}."
        )
    if report.get("notes"):
        lines.extend(["", *[f"- {note}" for note in report["notes"]]])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 3 output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "stage": "Phase 3 Stage 3 frozen checkpoint preflight",
        "status": "RUNNING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(OUTPUT_DIR.relative_to(PROJECT)),
        "seeds": list(SEEDS),
        "lengths": list(LENGTHS),
        "batch_size": BATCH_SIZE,
        "warmup_count": WARMUP_COUNT,
        "timed_repetitions": TIMED_REPETITIONS,
        "tolerances": {
            "feature_alignment_atol": FEATURE_ATOL,
            "suffix_causality_atol": CAUSAL_ATOL,
            "suffix_causality_rtol": CAUSAL_RTOL,
            "sequence_extension_atol": EXTENSION_ATOL,
            "sequence_extension_rtol": EXTENSION_RTOL,
        },
        "models_loaded": False,
        "training_run": False,
        "test_performance_evaluation": False,
        "test_split_opened": False,
        "provenance_checks": {},
        "checkpoints": [],
        "inference_results": [],
        "conditions": {},
        "notes": [],
    }
    try:
        hardware = _hardware_and_versions()
        report["software_and_sources"] = {
            "hardware": hardware,
            "project_sources": source_hash_inventory(),
        }
        if not hardware["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; Mamba preflight requires CUDA")
        if hardware["mamba_ssm_version"] is None:
            raise RuntimeError(
                "Could not import installed mamba_ssm: "
                + str(hardware["mamba_source_hashes"].get("import_error"))
            )
        report["provenance_checks"]["cuda_available"] = True
        report["provenance_checks"]["mamba_import"] = True

        # Every training-record source hash must match either the current source
        # or its immutable Git HEAD blob. Stage 1 intentionally extended only
        # data_generation.py; the original recorded hash is checked against HEAD.
        historic_generator = git_blob_sha256("src/radar_tracking/data/data_generation.py")
        current_generator = report["software_and_sources"]["project_sources"][
            "data_generation.py"
        ]["current_sha256"]
        report["notes"].append(
            "The current data_generation.py includes the Stage 1 opt-in path; the original training-source hash "
            f"{historic_generator} is verified against Git HEAD. Current workspace hash is {current_generator}. "
            "The preflight uses unchanged tracker/model implementation files and does not invoke the generator."
        )

        device = torch.device("cuda:0")
        all_models = verify_and_load_all_checkpoints(report, device)
        report["models_loaded"] = True
        report["conditions"]["checkpoint_hashes"] = "PASS"
        report["conditions"]["configuration_hashes"] = "PASS"
        report["conditions"]["strict_state_dict_load"] = "PASS"
        report["conditions"]["parameter_count_619014"] = "PASS"
        report["conditions"]["m1_frozen_normalization_statistics"] = "PASS"

        training_inputs = load_training_inputs()
        report["training_input_sources"] = {
            str(length): {
                "source_file": value["source_file"],
                "source_split": value["source_split"],
                "trajectory_id": value["trajectory_id"],
                "input_shape": list(value["observations"].shape),
                "initial_state_shape": list(value["initial_state"].shape),
                "initialization_context_shape": None
                if value["initialization_observations"] is None
                else list(value["initialization_observations"].shape),
                "initialization_context_sha256": None
                if value["initialization_observations"] is None
                else tensor_sha256(value["initialization_observations"]),
            }
            for length, value in training_inputs.items()
        }
        report["conditions"]["training_split_only_inputs"] = "PASS"
        report["conditions"]["no_test_split_opened"] = "PASS"

        for checkpoint in all_models:
            print(
                f"Preflighting {checkpoint['variant']} seed {checkpoint['seed']}",
                flush=True,
            )
            result = preflight_checkpoint(checkpoint, training_inputs, device)
            report["checkpoints"].append(
                {
                    key: value
                    for key, value in checkpoint.items()
                    if key != "model"
                }
            )
            report["inference_results"].append(result)
            if result["causality"]["status"] != "PASS":
                raise RuntimeError(
                    f"future-suffix causality failed for {result['variant']} seed {result['seed']}"
                )
            if result["sequence_extension"]["status"] != "PASS":
                raise RuntimeError(
                    f"sequence-extension invariance failed for {result['variant']} seed {result['seed']}"
                )
        report["conditions"].update(
            {
                "cuda_inference_T100_T200_T400": "PASS",
                "input_token_output_shapes": "PASS",
                "input_feature_prediction_finiteness": "PASS",
                "future_suffix_causality": "PASS",
                "sequence_extension_invariance": "PASS",
            }
        )
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["conditions"]["stopped_on_error"] = report["error"]
        write_json(OUTPUT_DIR / "preflight_report.json", report)
        write_markdown(OUTPUT_DIR / "preflight_report.md", _markdown_report(report))
        raise

    json_hash = write_json(OUTPUT_DIR / "preflight_report.json", report)
    markdown_hash = write_markdown(
        OUTPUT_DIR / "preflight_report.md", _markdown_report(report)
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "preflight_report": str(OUTPUT_DIR / "preflight_report.json"),
                "preflight_report_sha256": json_hash,
                "markdown_report": str(OUTPUT_DIR / "preflight_report.md"),
                "markdown_report_sha256": markdown_hash,
                "tested_checkpoints": len(report["checkpoints"]),
                "timed_model_length_rows": sum(
                    len(row["length_results"]) for row in report["inference_results"]
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
