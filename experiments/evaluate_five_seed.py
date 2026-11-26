#!/usr/bin/env python3
"""Independently evaluate the ten existing fixed-pool campaign checkpoints.

This script never trains or writes into historical campaign directories. It
uses the per-run recorded model configs and writes only to a new verification
directory, refusing to replace any existing output.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

import five_seed_harness as harness

OUTPUT_DIR = (
    PROJECT
    / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/independent_verification"
)
METRICS = (
    "position_rmse_m",
    "velocity_rmse_mps",
    "acceleration_rmse_mps2",
    "six_state_mse_scaled_units",
    "six_state_mse_db",
)


def _recorded_inputs() -> list[dict[str, Any]]:
    """Preflight all records and checkpoint evidence before model construction."""
    runs = []
    missing = []
    for seed in harness.TRAIN_SEEDS:
        run_dir = harness.OUTPUT_ROOT / f"seed_{seed}"
        run_path = run_dir / "run_metrics.json"
        if not run_path.is_file():
            missing.append(str(run_path))
            continue
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if record.get("seed") != seed:
            raise RuntimeError(f"Seed mismatch in {run_path}")
        if record.get("shared_test_sha256") != harness.EXPECTED_TEST_SHA256:
            raise RuntimeError(f"Test-set hash mismatch in {run_path}")
        for model_type, checkpoint_name in (
            ("mamba", "best_mamba_tracker.pt"),
            ("knet", "best_knet_seq100.pt"),
        ):
            model_record = record.get(model_type)
            if not isinstance(model_record, dict):
                missing.append(f"{run_path}: {model_type} record")
                continue
            config = model_record.get("config")
            checkpoint = run_dir / checkpoint_name
            expected_hash = model_record.get("checkpoint_sha256")
            if not isinstance(config, dict):
                missing.append(f"{run_path}: {model_type} config")
            if not checkpoint.is_file():
                missing.append(str(checkpoint))
            elif not expected_hash:
                missing.append(f"{run_path}: {model_type} checkpoint SHA-256")
            elif harness.sha256_file(checkpoint) != expected_hash:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
            else:
                runs.append({
                    "seed": seed,
                    "model_type": model_type,
                    "record": record,
                    "model_record": model_record,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": expected_hash,
                })
    if missing:
        raise FileNotFoundError("Missing required evaluation evidence:\n- " + "\n- ".join(missing))
    if len(runs) != 10:
        raise RuntimeError(f"Expected ten checkpoint records, found {len(runs)}")
    return runs


def _build_knet(api: dict[str, Any], record: dict[str, Any], device: Any):
    args = harness.make_simparams(api)
    cfg = record["config"]
    args.n_batch = int(cfg["batch_size"])
    args.epochs = int(cfg["epoch_cap"])
    args.learning_rate = float(cfg["learning_rate_initial"])
    args.weight_decay = float(cfg["weight_decay"])
    args.stopping_criteria = int(cfg["patience"])
    args.update_thr_db = float(cfg["min_delta_db"])
    args.in_mult_KNet = 5
    args.out_mult_KNet = 40
    args.knet_rb_wrap_theta = True
    args.knet_rb_arc_length = True
    ca_model = api["ConstantAcceleration2DModel"](args)
    model = api["KalmanNetNN"]()
    model.NNBuild(ca_model, args)
    return model.to(device), args


def _evaluate_one(api: dict[str, Any], item: dict[str, Any], test: dict[str, Any], device: Any):
    torch = api["torch"]
    benchmark = api["benchmark"]
    model_record = item["model_record"]
    if item["model_type"] == "mamba":
        config_data = dict(model_record["config"])
        config = benchmark.MambaTrainConfig(**config_data)
        args = harness.make_simparams(api)
        benchmark.set_seed(int(config.seed))
        model = benchmark.build_mamba_model(args, config, device)
        batch_size = int(config.eval_batch_size)
    else:
        model, args = _build_knet(api, model_record, device)
        batch_size = int(model_record["config"]["batch_size"])

    harness.load_checkpoint_weights(torch, model, item["checkpoint"], device)
    model.eval()
    with torch.inference_mode():
        if item["model_type"] == "mamba":
            prediction = harness.predict_mamba(
                api, model, test["true_state"], test["observations"],
                test["initial_state"], device, batch_size,
            )
        else:
            prediction = harness.predict_knet(
                api, model, args, test["true_state"], test["observations"],
                test["initial_state"], device, batch_size,
            )
    if tuple(prediction.shape) != tuple(test["true_state"].shape):
        raise ValueError(
            f"{item['model_type']} seed {item['seed']} prediction has shape "
            f"{tuple(prediction.shape)}, expected {tuple(test['true_state'].shape)}"
        )
    if not torch.isfinite(prediction).all().item():
        raise ValueError(f"Non-finite predictions for {item['model_type']} seed {item['seed']}")
    if item["model_type"] == "mamba":
        parameter_count = benchmark.count_parameters(model)
    else:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(model_record["parameter_count"]):
        raise RuntimeError(
            f"Parameter-count mismatch for {item['model_type']} seed {item['seed']}: "
            f"{parameter_count} != {model_record['parameter_count']}"
        )
    return harness.calculate_metrics(torch, test["true_state"], prediction, args.dt), parameter_count, batch_size


def _write_new(path: Path, writer) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer(stream)


def main() -> None:
    harness.assert_fresh_path(OUTPUT_DIR)
    items = _recorded_inputs()
    api = harness.project_imports()
    torch = api["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Independent evaluation requires CUDA: the recorded Mamba implementation "
            "is CUDA-only. No evaluation outputs were created."
        )
    device = torch.device("cuda")
    test, test_hash = harness.load_test_payload(api)
    results = []
    for item in items:
        metrics, parameter_count, batch_size = _evaluate_one(api, item, test, device)
        results.append({
            "seed": item["seed"],
            "model_type": item["model_type"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "test_set_sha256": test_hash,
            "parameter_count": parameter_count,
            **metrics,
            "evaluation_batch_size": batch_size,
            "evaluation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
    if harness.verify_test_hash() != test_hash:
        raise RuntimeError("Test-set file changed during independent evaluation")
    for item in items:
        if harness.sha256_file(item["checkpoint"]) != item["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed during evaluation: {item['checkpoint']}")

    comparison = []
    for result, item in zip(results, items):
        stored_metrics = item["model_record"]["metrics"]
        for metric in METRICS:
            stored = float(stored_metrics[metric])
            recomputed = float(result[metric])
            absolute = abs(recomputed - stored)
            relative = absolute / abs(stored) if stored else (0.0 if absolute == 0 else math.inf)
            comparison.append({
                "seed": result["seed"],
                "model_type": result["model_type"],
                "metric": metric,
                "stored_metric": stored,
                "recomputed_metric": recomputed,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "atol": harness.EVAL_ATOL,
                "rtol": harness.EVAL_RTOL,
                "pass": math.isclose(
                    recomputed, stored, rel_tol=harness.EVAL_RTOL, abs_tol=harness.EVAL_ATOL
                ),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    _write_new(
        OUTPUT_DIR / "checkpoint_evaluations.json",
        lambda stream: json.dump({"evaluations": results}, stream, indent=2, sort_keys=True, allow_nan=False),
    )
    comparison_fields = list(comparison[0])
    _write_new(
        OUTPUT_DIR / "stored_metric_comparison.csv",
        lambda stream: _csv_write(stream, comparison_fields, comparison),
    )
    provenance = {
        "git_commit": harness.source_provenance()["git_revision"],
        "corrected_harness_sha256": harness.sha256_file(Path(harness.__file__).resolve()),
        "source_file_sha256": harness.source_provenance()["source_file_sha256"],
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(device),
        "test_set_sha256": test_hash,
        "checkpoints": [
            {"seed": item["seed"], "model_type": item["model_type"],
             "sha256": item["checkpoint_sha256"]}
            for item in items
        ],
        "historical_record_note": (
            "Historical experiment records retain their original harness hash; "
            "they were not rewritten."
        ),
    }
    _write_new(
        OUTPUT_DIR / "provenance.json",
        lambda stream: json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False),
    )
    print(f"Independent evaluation written to {OUTPUT_DIR}")
    print(f"Metric comparisons passing: {sum(row['pass'] for row in comparison)}/{len(comparison)}")


def _csv_write(stream, fieldnames, rows) -> None:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
