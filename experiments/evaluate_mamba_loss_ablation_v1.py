#!/usr/bin/env python3
"""Final shared-test evaluation for variants selected on validation only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))

ROOT = PROJECT / "artifacts/results/synthetic/mamba_loss_ablation_v1"
BASELINE = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
TEST_PATH = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"
SEEDS = (11, 22, 33, 44, 55)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    selection_path = ROOT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    variants = selection.get("selected_variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("selection.json must list validation-selected variants")
    allowed = {"M1_standardized", "M2_accel_w4", "M2_accel_w8"}
    if set(variants) - allowed:
        raise ValueError(f"Unsupported selected variant(s): {variants}")
    out_dir = ROOT / "final_test_evaluation"
    out_dir.mkdir(parents=True, exist_ok=False)

    from radar_tracking.cli import benchmark_mamba as benchmark
    from radar_tracking.config import simparams_class
    from mamba_loss_ablation_v1 import prediction_metrics

    expected_test_hash = json.loads((BASELINE / "fixed_pool_manifest.json").read_text())[
        "shared_test_sha256"
    ]
    actual_test_hash = sha256(TEST_PATH)
    if actual_test_hash != expected_test_hash:
        raise RuntimeError("Shared test hash differs from the preserved baseline manifest")
    payload = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    truth = payload["true_state"]
    obs = payload["observations"]
    init = payload["initial_state"]
    if tuple(truth.shape) != (1024, 6, 100) or tuple(obs.shape) != (1024, 2, 100):
        raise ValueError("Unexpected shared test tensor shape")
    if not all(torch.isfinite(x).all().item() for x in (truth, obs, init)):
        raise ValueError("Shared test contains non-finite values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The existing Mamba implementation requires CUDA")
    simparams = simparams_class()
    evaluated = ["M0_original", *variants]
    for variant in evaluated:
        variant_dir = out_dir / variant
        variant_dir.mkdir()
        for seed in SEEDS:
            if variant == "M0_original":
                record_path = BASELINE / f"seed_{seed}" / "run_metrics.json"
                ckpt = BASELINE / f"seed_{seed}" / "best_mamba_tracker.pt"
                cfg = json.loads(record_path.read_text())["mamba"]["config"]
            else:
                record_path = ROOT / variant / f"seed_{seed}" / "run_metrics.json"
                ckpt = ROOT / variant / f"seed_{seed}" / "best_mamba_tracker.pt"
                cfg = json.loads(record_path.read_text())["config"]
            model_fields = {
                "seed", "embed_dim", "hidden_dim", "num_layers", "d_state",
                "d_conv", "expand", "dropout", "encoder_direction",
                "eval_batch_size", "epochs", "batch_size", "learning_rate",
                "weight_decay", "patience", "min_delta", "checkpoint_name",
            }
            config = benchmark.MambaTrainConfig(**{
                key: value for key, value in cfg.items() if key in model_fields
            })
            benchmark.set_seed(seed)
            model = benchmark.build_mamba_model(simparams, config, device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, truth.shape[0], config.eval_batch_size):
                    end = min(start + config.eval_batch_size, truth.shape[0])
                    pred = model(
                        observations=obs[start:end].to(device),
                        init_state=init[start:end].to(device),
                    )["state_estimates"]
                    chunks.append(pred.cpu())
            metrics = prediction_metrics(torch.cat(chunks), truth)
            write_new(variant_dir / f"seed_{seed}.json", {
                "seed": seed,
                "variant": variant,
                "checkpoint_sha256": sha256(ckpt),
                "test_dataset_sha256": actual_test_hash,
                "evaluation_batch_size": config.eval_batch_size,
                "metrics": metrics,
            })
            print(f"{variant} seed={seed} position={metrics['position_rmse_physical']:.2f} "
                  f"velocity={metrics['velocity_rmse_physical']:.2f} "
                  f"acceleration={metrics['acceleration_rmse_physical']:.2f}", flush=True)

    write_new(out_dir / "evaluation_manifest.json", {
        "selected_variants_from_validation": variants,
        "evaluated_variants": evaluated,
        "selection_record_sha256": sha256(selection_path),
        "test_dataset_sha256": actual_test_hash,
        "test_sample_count": int(truth.shape[0]),
        "test_accessed_after_selection": True,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
    })
    print(f"Final test evaluation complete: {out_dir}")


if __name__ == "__main__":
    main()
