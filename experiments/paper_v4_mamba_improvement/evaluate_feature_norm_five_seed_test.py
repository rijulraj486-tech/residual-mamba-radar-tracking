#!/usr/bin/env python3
"""Final test evaluation gated by a frozen, qualifying validation selection."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.paper_v4_mamba_improvement import run_screening_pilot as pilot
from radar_tracking.cli import benchmark_mamba as benchmark
from radar_tracking.config import simparams_class

CONFIG_PATH = PROJECT / "configs/paper_v4_mamba_improvement/features_train_only_normalization_five_seed_confirmation_v2.json"
BASELINE_CONFIG_PATH = PROJECT / "configs/paper_v4_mamba_improvement/baseline_reference.json"
RUN_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_mamba_improvement_v3/experiment_a_feature_norm_five_seed_v2"
SELECTION_PATH = RUN_ROOT / "validation_selection_manifest.json"
FROZEN_HASH_PATH = RUN_ROOT / "frozen_validation_hashes.json"
AGGREGATE_PATH = RUN_ROOT / "validation_aggregate.json"
BASELINE_ROOT = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
TEST_OUTPUT = RUN_ROOT / "final_test_evaluation_v1"
METRICS = (
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
    return {
        "mean": mean,
        "sample_sd": math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)),
    }


def main() -> None:
    # Verify all frozen validation records and checkpoint identities before any test access.
    if TEST_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite final evaluation: {TEST_OUTPUT}")
    selection = json.loads(SELECTION_PATH.read_text())
    frozen_hashes = json.loads(FROZEN_HASH_PATH.read_text())
    config = json.loads(CONFIG_PATH.read_text())
    baseline_contract = json.loads(BASELINE_CONFIG_PATH.read_text())
    if selection["status"] != "validation_confirmed_qualifies_for_final_test":
        raise RuntimeError("Validation selection did not qualify; refusing to access test data")
    if selection["qualifies_for_final_test_evaluation"] is not True:
        raise RuntimeError("Validation selection gate is false")
    if sha256(SELECTION_PATH) != frozen_hashes["validation_selection_manifest_sha256"]:
        raise RuntimeError("Frozen selection manifest hash mismatch")
    if sha256(AGGREGATE_PATH) != frozen_hashes["validation_aggregate_sha256"]:
        raise RuntimeError("Frozen validation aggregate hash mismatch")
    if sha256(CONFIG_PATH) != frozen_hashes["config_sha256"] or sha256(CONFIG_PATH) != selection["selected_configuration_sha256"]:
        raise RuntimeError("Selected configuration hash mismatch")
    seeds = config["seeds"]
    if set(selection["candidate_checkpoints"]) != {str(seed) for seed in seeds}:
        raise RuntimeError("Frozen selection does not contain the exact five candidate seeds")

    # Verify every checkpoint digest before reading the held-out test file.
    for seed in seeds:
        seed_dir = RUN_ROOT / f"seed_{seed}"
        checkpoint = seed_dir / "best_validation_checkpoint.pt"
        if sha256(checkpoint) != selection["candidate_checkpoints"][str(seed)]:
            raise RuntimeError(f"Candidate seed {seed} checkpoint hash mismatch")
        baseline_dir = BASELINE_ROOT / f"seed_{seed}"
        baseline_record = json.loads((baseline_dir / "run_metrics.json").read_text())
        baseline_checkpoint = baseline_dir / baseline_record["mamba"]["checkpoint"]
        if sha256(baseline_checkpoint) != selection["baseline_checkpoints"][str(seed)]:
            raise RuntimeError(f"Baseline seed {seed} checkpoint hash mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("The recorded Mamba implementation requires CUDA")
    test_path = PROJECT / baseline_contract["data"]["test_file"]
    expected_test_hash = baseline_contract["data"]["test_sha256"]
    actual_test_hash = sha256(test_path)
    if actual_test_hash != expected_test_hash:
        raise RuntimeError("Held-out test pool hash mismatch")
    # Test observations are opened only after the validation selection is frozen and verified.
    test_payload = torch.load(test_path, map_location="cpu", weights_only=True)
    expected_shapes = {
        "true_state": (1024, 6, 100),
        "observations": (1024, 2, 100),
        "initial_state": (1024, 6, 1),
    }
    for key, shape in expected_shapes.items():
        if tuple(test_payload[key].shape) != shape:
            raise RuntimeError(f"Unexpected held-out test tensor shape for {key}")
    test = {
        "state": test_payload["true_state"],
        "obs": test_payload["observations"],
        "init": test_payload["initial_state"],
    }
    del test_payload

    device = torch.device("cuda")
    simparams = simparams_class()
    simparams.seq_len = 100
    pilot_config = json.loads((PROJECT / config["pilot_config"]).read_text())
    rows = []
    for seed in seeds:
        seed = int(seed)
        baseline_dir = BASELINE_ROOT / f"seed_{seed}"
        baseline_record = json.loads((baseline_dir / "run_metrics.json").read_text())["mamba"]
        baseline_checkpoint = baseline_dir / baseline_record["checkpoint"]
        baseline_cfg = benchmark.MambaTrainConfig(**baseline_record["config"])
        benchmark.set_seed(seed)
        baseline_model = benchmark.build_mamba_model(simparams, baseline_cfg, device)
        baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
        baseline_metrics, _ = pilot.evaluate(baseline_model, test, device)
        baseline_timing = pilot.timed_validation(baseline_model, test, device)

        stats_path = RUN_ROOT / f"seed_{seed}" / "train_only_feature_statistics.json"
        stats = json.loads(stats_path.read_text())
        feature_mean = torch.tensor(stats["feature_mean_21"], dtype=torch.float32)
        feature_std = torch.tensor(stats["feature_std_21"], dtype=torch.float32)
        candidate = pilot.construct_candidate(
            "feature_norm", simparams, pilot_config["model"], device, feature_mean, feature_std
        )
        candidate_checkpoint = RUN_ROOT / f"seed_{seed}" / "best_validation_checkpoint.pt"
        candidate.load_state_dict(torch.load(candidate_checkpoint, map_location=device, weights_only=True))
        candidate_metrics, _ = pilot.evaluate(candidate, test, device)
        candidate_timing = pilot.timed_validation(candidate, test, device)
        rows.append({
            "seed": seed,
            "test_pool_sha256": actual_test_hash,
            "baseline_checkpoint_sha256": sha256(baseline_checkpoint),
            "candidate_checkpoint_sha256": sha256(candidate_checkpoint),
            "baseline_parameter_count": pilot.parameter_counts(baseline_model),
            "candidate_parameter_count": pilot.parameter_counts(candidate),
            "baseline_test_metrics": baseline_metrics,
            "candidate_test_metrics": candidate_metrics,
            "paired_difference_candidate_minus_baseline": {
                key: candidate_metrics[key] - baseline_metrics[key] for key in METRICS
            },
            "baseline_inference_timing": baseline_timing,
            "candidate_inference_timing": candidate_timing,
        })
        print(json.dumps({
            "seed": seed,
            "baseline": {key: baseline_metrics[key] for key in METRICS},
            "candidate": {key: candidate_metrics[key] for key in METRICS},
        }), flush=True)
        del baseline_model, candidate
        torch.cuda.empty_cache()

    if sha256(test_path) != actual_test_hash:
        raise RuntimeError("Held-out test pool changed during final evaluation")
    result = {
        "experiment": config["experiment"],
        "status": "final_test_evaluation_complete",
        "selection_manifest_sha256": sha256(SELECTION_PATH),
        "validation_aggregate_sha256": sha256(AGGREGATE_PATH),
        "configuration_sha256": sha256(CONFIG_PATH),
        "test_pool_path": str(test_path.relative_to(PROJECT)),
        "test_pool_sha256": actual_test_hash,
        "test_accessed": True,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": rows,
    }
    for group, metric_getter in (
        ("baseline", lambda row: row["baseline_test_metrics"]),
        ("candidate", lambda row: row["candidate_test_metrics"]),
        ("paired_difference_candidate_minus_baseline", lambda row: row["paired_difference_candidate_minus_baseline"]),
    ):
        result[group + "_mean_sample_sd"] = {
            key: mean_sd([float(metric_getter(row)[key]) for row in rows]) for key in METRICS
        }

    source_paths = [
        Path(__file__).resolve(),
        Path(pilot.__file__).resolve(),
        Path(pilot.__file__).with_name("model_variants.py").resolve(),
        PROJECT / "src/radar_tracking/cli/benchmark_mamba.py",
        PROJECT / "src/radar_tracking/models/tracker_model.py",
        PROJECT / "src/radar_tracking/models/bimamba_encoder.py",
        PROJECT / "src/radar_tracking/models/prediction_head.py",
        CONFIG_PATH,
        BASELINE_CONFIG_PATH,
    ]
    result["source_sha256"] = {str(path.relative_to(PROJECT)): sha256(path) for path in source_paths}
    TEST_OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new(TEST_OUTPUT / "final_test_comparison.json", result)
    print(json.dumps({
        "output": str(TEST_OUTPUT),
        "baseline_mean_sample_sd": result["baseline_mean_sample_sd"],
        "candidate_mean_sample_sd": result["candidate_mean_sample_sd"],
        "paired_difference_mean_sample_sd": result["paired_difference_candidate_minus_baseline_mean_sample_sd"],
        "test_pool_sha256": actual_test_hash,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
