#!/usr/bin/env python3
"""Run one validation-only, one-seed pilot for acceleration group loss weight 2.

Reads only the immutable fixed train/validation pool. The shared test payload is
never opened by this script. Outputs use exclusive creation in a new directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments import mamba_loss_ablation_v1 as loss_runner  # noqa: E402

from radar_tracking.cli import benchmark_mamba as benchmark  # noqa: E402
from radar_tracking.config import simparams_class  # noqa: E402


POOL = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_train_val.pt"
BASELINE_DIR = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
CONFIG = PROJECT / "configs/paper_v4_mamba_improvement/loss_acceleration_weight_2_pilot.json"
OUTPUT = PROJECT / "artifacts/results/synthetic/paper_v4_mamba_improvement_v1/pilot_loss_accel_w2"
EXPECTED_POOL_SHA256 = "4ac64b0cd80e14f6ec845f81b81ac9c9392234fd8f986f67380f3608d77a2733"
SEED = 11


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_validation(model, val_tensors, device, repeats: int = 3) -> dict:
    # One unmeasured warm-up; timed calls include the same transfers and metric
    # calculations for baseline and pilot. CUDA is synchronized around timings.
    loss_runner.evaluate(model, val_tensors, device, batch_size=16)
    durations = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        loss_runner.evaluate(model, val_tensors, device, batch_size=16)
        sync(device)
        durations.append(time.perf_counter() - start)
    return {
        "device": str(device),
        "warmup_passes": 1,
        "timed_passes": repeats,
        "mean_full_validation_seconds": sum(durations) / len(durations),
        "seconds_per_trajectory": sum(durations) / len(durations) / len(val_tensors["state"]),
        "raw_seconds": durations,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {OUTPUT}")
    if not POOL.is_file():
        raise FileNotFoundError(POOL)
    if sha256(POOL) != EXPECTED_POOL_SHA256:
        raise RuntimeError("Fixed train/validation pool hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("The current Mamba implementation requires CUDA")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["training_overrides"]["seeds"] != [SEED]:
        raise RuntimeError("Pilot config seed differs from this runner")
    device = torch.device("cuda")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "config.json").write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    payload = torch.load(POOL, map_location="cpu", weights_only=True)
    train = {
        "state": payload["train"]["state"],
        "obs": payload["train"]["observations"],
        "init": payload["train"]["initial_state"],
    }
    val = {
        "state": payload["validation"]["state"],
        "obs": payload["validation"]["observations"],
        "init": payload["validation"]["initial_state"],
    }
    if tuple(train["state"].shape) != (819, 6, 100) or tuple(val["state"].shape) != (204, 6, 100):
        raise RuntimeError("Unexpected fixed pool shape")

    baseline_record = json.loads((BASELINE_DIR / f"seed_{SEED}" / "run_metrics.json").read_text())
    baseline_ckpt = BASELINE_DIR / f"seed_{SEED}" / "best_mamba_tracker.pt"
    if sha256(baseline_ckpt) != baseline_record["mamba"]["checkpoint_sha256"]:
        raise RuntimeError("Baseline checkpoint digest does not match its run record")

    simparams = simparams_class()
    simparams.seq_len = 100
    benchmark.set_seed(SEED)
    baseline_config = benchmark.MambaTrainConfig(**baseline_record["mamba"]["config"])
    baseline_model = benchmark.build_mamba_model(simparams, baseline_config, device)
    baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True))
    baseline_val_metrics, _ = loss_runner.evaluate(baseline_model, val, device, batch_size=16)
    baseline_val_timing = timed_validation(baseline_model, val, device)
    baseline_parameters = sum(parameter.numel() for parameter in baseline_model.parameters())
    del baseline_model
    torch.cuda.empty_cache()

    api = {"torch": torch, "benchmark": benchmark, "simparams": simparams}
    run_dir = OUTPUT / f"seed_{SEED}"
    run_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    pilot_record = loss_runner.train_one(
        api=api,
        seed=SEED,
        variant="M2_accel_w2",
        accel_weight=2.0,
        train=train,
        val=val,
        state_std=None,
        run_dir=run_dir,
        device=device,
    )
    wall_seconds = time.perf_counter() - started

    pilot_config = benchmark.MambaTrainConfig(
        seed=SEED, epochs=180, batch_size=16, eval_batch_size=16,
        learning_rate=1e-3, weight_decay=1e-4, patience=25, min_delta=1e-6,
        embed_dim=128, hidden_dim=256, num_layers=3, d_state=16, d_conv=4,
        expand=2, dropout=0.08, encoder_direction="forward",
        checkpoint_name="best_mamba_tracker.pt",
    )
    pilot_model = benchmark.build_mamba_model(simparams, pilot_config, device)
    pilot_model.load_state_dict(torch.load(run_dir / "best_mamba_tracker.pt", map_location=device, weights_only=True))
    pilot_val_timing = timed_validation(pilot_model, val, device)
    pilot_parameters = sum(parameter.numel() for parameter in pilot_model.parameters())

    source_paths = [
        Path(__file__).resolve(),
        Path(loss_runner.__file__).resolve(),
        PROJECT / "src/radar_tracking/cli/benchmark_mamba.py",
        PROJECT / "src/radar_tracking/models/tracker_model.py",
        PROJECT / "src/radar_tracking/models/bimamba_encoder.py",
        PROJECT / "src/radar_tracking/models/prediction_head.py",
        PROJECT / "src/radar_tracking/data/data_generation.py",
        CONFIG,
    ]
    source_hashes = {str(path.relative_to(PROJECT)): sha256(path) for path in source_paths}
    summary = {
        "experiment": config["experiment"],
        "status": "pilot_complete_validation_only",
        "test_accessed": False,
        "seed": SEED,
        "train_count": len(train["state"]),
        "validation_count": len(val["state"]),
        "sequence_length": 100,
        "train_validation_pool_sha256": sha256(POOL),
        "baseline_checkpoint_sha256": sha256(baseline_ckpt),
        "pilot_checkpoint_sha256": sha256(run_dir / "best_mamba_tracker.pt"),
        "parameter_count": {"baseline": baseline_parameters, "pilot": pilot_parameters},
        "baseline_validation_metrics": baseline_val_metrics,
        "pilot_validation_metrics": pilot_record["validation_metrics_at_selected_checkpoint"],
        "pilot_training": {
            "epochs_run": pilot_record["config"]["actual_epochs"],
            "best_epoch": pilot_record["best_epoch"],
            "best_validation_objective": pilot_record["best_validation_objective"],
            "duration_seconds_recorded": pilot_record["training_duration_seconds"],
            "wall_seconds_including_recording": wall_seconds,
        },
        "validation_inference_timing": {
            "baseline": baseline_val_timing,
            "pilot": pilot_val_timing,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "source_sha256": source_hashes,
    }
    write_new(OUTPUT / "pilot_summary.json", summary)
    print(json.dumps({
        "output": str(OUTPUT),
        "best_epoch": pilot_record["best_epoch"],
        "baseline_val": baseline_val_metrics,
        "pilot_val": pilot_record["validation_metrics_at_selected_checkpoint"],
        "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
