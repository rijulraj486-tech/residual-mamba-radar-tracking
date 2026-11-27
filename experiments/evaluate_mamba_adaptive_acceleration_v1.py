#!/usr/bin/env python3
"""Evaluate validation-selected acceleration features on the shared test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))

from mamba_adaptive_acceleration_v1 import make_model_class, metrics, sha256, json_new
from mamba_adaptive_acceleration_features import adaptive_causal_acceleration_features

ROOT = PROJECT / "artifacts/results/synthetic/mamba_adaptive_acceleration_v1"
FIXED_ROOT = PROJECT / "artifacts/results/synthetic/mamba_acceleration_feature_ablation_v1"
BASELINE = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
TEST_PATH = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"
SEEDS = (11, 22, 33, 44, 55)


def main() -> None:
    selection_path = ROOT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected_adaptive_variants")
    known = {"A3_adaptive_narrow", "A3_adaptive_broad"}
    if not selection.get("validation_only") or selection.get("test_data_opened"):
        raise ValueError("selection record must certify validation-only selection")
    if not isinstance(selected, list) or set(selected) - known:
        raise ValueError("selection.json must contain selected known variants")
    out = ROOT / "final_test_evaluation"
    out.mkdir(parents=True, exist_ok=False)

    from radar_tracking.cli import benchmark_mamba as benchmark
    from radar_tracking.models.tracker_model import BiMambaTrackerModel

    expected_hash = json.loads((BASELINE / "fixed_pool_manifest.json").read_text())[
        "shared_test_sha256"
    ]
    actual_hash = sha256(TEST_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError("Shared test file hash does not match the preserved baseline")
    test = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    truth, observations, init = test["true_state"], test["observations"], test["initial_state"]
    if tuple(truth.shape) != (1024, 6, 100) or tuple(observations.shape) != (1024, 2, 100):
        raise ValueError("Unexpected shared test tensor shape")
    if not all(torch.isfinite(x).all().item() for x in (truth, observations, init)):
        raise ValueError("Shared test tensors must be finite")
    if not torch.cuda.is_available():
        raise RuntimeError("The baseline Mamba implementation requires CUDA")
    device = torch.device("cuda")
    tracker_class = make_model_class(BiMambaTrackerModel)
    from mamba_acceleration_feature_ablation_v1 import make_model_class as make_fixed_class
    fixed_tracker_class = make_fixed_class(BiMambaTrackerModel)
    simparams = benchmark.simparams_class()
    variants = ["A0_existing_reference", "A1_ema_alpha025", "A1_ema_alpha050", *selected]

    for variant in variants:
        variant_out = out / variant
        variant_out.mkdir()
        for seed in SEEDS:
            if variant == "A0_existing_reference":
                record = json.loads((BASELINE / f"seed_{seed}" / "run_metrics.json").read_text())["mamba"]
                config_values = record["config"]
                feature = None
                checkpoint = BASELINE / f"seed_{seed}" / "best_mamba_tracker.pt"
                model_cls = benchmark.build_mamba_model
            elif variant.startswith("A1_"):
                record = json.loads((FIXED_ROOT / variant / f"seed_{seed}" / "run_metrics.json").read_text())
                config_values = record["config"]
                feature = record["feature"]
                checkpoint = FIXED_ROOT / variant / f"seed_{seed}" / "best_mamba_tracker.pt"
                model_cls = None
            else:
                record = json.loads((ROOT / variant / f"seed_{seed}" / "run_metrics.json").read_text())
                config_values = record["config"]
                feature = record["feature"]
                checkpoint = ROOT / variant / f"seed_{seed}" / "best_mamba_tracker.pt"
                model_cls = None
            fields = {
                "seed", "epochs", "batch_size", "eval_batch_size", "learning_rate",
                "weight_decay", "patience", "min_delta", "embed_dim", "hidden_dim",
                "num_layers", "d_state", "d_conv", "expand", "dropout",
                "encoder_direction", "checkpoint_name",
            }
            config = benchmark.MambaTrainConfig(**{
                k: v for k, v in config_values.items() if k in fields
            })
            benchmark.set_seed(seed)
            if variant == "A0_existing_reference":
                model = model_cls(simparams, config, device)
            elif variant.startswith("A1_"):
                model = fixed_tracker_class(
                    task="radar_benchmark", state_dim=6, obs_dim=2, dt=simparams.dt,
                    embed_dim=config.embed_dim, hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers, d_state=config.d_state,
                    d_conv=config.d_conv, expand=config.expand, dropout=config.dropout,
                    encoder_direction="forward", acceleration_feature=feature,
                ).to(device)
            else:
                model = tracker_class(
                    task="radar_benchmark", state_dim=6, obs_dim=2, dt=simparams.dt,
                    embed_dim=config.embed_dim, hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers, d_state=config.d_state,
                    d_conv=config.d_conv, expand=config.expand, dropout=config.dropout,
                    encoder_direction="forward", acceleration_feature=feature,
                ).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            parameter_count = sum(p.numel() for p in model.parameters())
            if parameter_count != 619014:
                raise RuntimeError(f"Parameter count changed for {variant}/{seed}: {parameter_count}")
            model.eval()
            predictions = []
            with torch.no_grad():
                for start in range(0, truth.shape[0], config.eval_batch_size):
                    end = min(start + config.eval_batch_size, truth.shape[0])
                    predictions.append(model(
                        observations=observations[start:end].to(device),
                        init_state=init[start:end].to(device),
                    )["state_estimates"].cpu())
            result = metrics(torch.cat(predictions), truth)
            if variant.startswith("A3_"):
                position = model.measurement_to_cartesian(observations.to(device))
                feature_values = feature
                _, coefficients = adaptive_causal_acceleration_features(
                    position, init[:, [1, 4], 0].to(device), init[:, [2, 5], 0].to(device),
                    observations[:, 0, :].to(device), observations[:, 1, :].to(device),
                    simparams.dt,
                    sigma_range=feature_values["sigma_range"],
                    sigma_bearing=feature_values["sigma_bearing"],
                    alpha_min=feature_values["alpha_min"],
                    alpha_max=feature_values["alpha_max"],
                    reliability_scale=feature_values["reliability_scale"],
                )
                result["adaptive_alpha"] = {
                    "mean_by_step": coefficients.mean(dim=0).cpu().tolist(),
                    "min": float(coefficients.min().item()),
                    "max": float(coefficients.max().item()),
                    "mean": float(coefficients.mean().item()),
                    "std": float(coefficients.std(unbiased=False).item()),
                }
            json_new(variant_out / f"seed_{seed}.json", {
                "variant": variant,
                "seed": seed,
                "feature": feature or {"method": "existing_second_difference"},
                "parameter_count": parameter_count,
                "checkpoint_sha256": sha256(checkpoint),
                "test_dataset_sha256": actual_hash,
                "metrics": result,
            })
            print(f"{variant} seed={seed} position={result['position_rmse_physical']:.2f} "
                  f"velocity={result['velocity_rmse_physical']:.2f} "
                  f"acceleration={result['acceleration_rmse_physical']:.2f}", flush=True)

    json_new(out / "evaluation_manifest.json", {
        "selection_sha256": sha256(selection_path),
        "selected_variants": selected,
        "evaluated_variants": variants,
        "test_dataset_sha256": actual_hash,
        "test_sample_count": int(truth.shape[0]),
        "test_accessed_after_validation_selection": True,
        "parameter_count": 619014,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
    })
    print(f"Final test evaluation complete: {out}")


if __name__ == "__main__":
    main()
