#!/usr/bin/env python3
"""Freeze adaptive feature selection using validation data only."""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "experiments")]
from mamba_adaptive_acceleration_v1 import OUTPUT, BASELINE, SEEDS, sha256, json_new

FIXED_ROOT = PROJECT / "artifacts/results/synthetic/mamba_acceleration_feature_ablation_v1"
ADAPTIVE = ("A3_adaptive_narrow", "A3_adaptive_broad")
REFERENCE = ("A0_existing_reference", "A1_ema_alpha025", "A1_ema_alpha050")
GROUPS = ("position", "velocity", "acceleration")


def val_metrics(variant, seed):
    if variant == "A0_existing_reference":
        path = OUTPUT / variant / f"seed_{seed}" / "run_metrics.json"
        return json.loads(path.read_text())["validation_metrics_at_selected_checkpoint"]
    if variant.startswith("A1_"):
        path = FIXED_ROOT / variant / f"seed_{seed}" / "run_metrics.json"
        return json.loads(path.read_text())["validation_metrics_at_selected_checkpoint"]
    path = OUTPUT / variant / f"seed_{seed}" / "run_metrics.json"
    return json.loads(path.read_text())["validation_metrics_at_selected_checkpoint"]


def main():
    if not OUTPUT.is_dir():
        raise FileNotFoundError("Train/validation artifacts are missing")
    names = (*REFERENCE, *ADAPTIVE)
    means = {}
    for name in names:
        metrics = [val_metrics(name, seed) for seed in SEEDS]
        means[name] = {
            group: sum(row[f"{group}_rmse_physical"] for row in metrics) / len(metrics)
            for group in GROUPS
        }
    frontier = []
    for candidate in names:
        dominated = any(
            all(means[other][g] <= means[candidate][g] for g in GROUPS)
            and any(means[other][g] < means[candidate][g] for g in GROUPS)
            for other in names if other != candidate
        )
        if not dominated:
            frontier.append(candidate)
    selected = [name for name in ADAPTIVE if name in frontier]
    record = {
        "selection_rule": "five-seed mean validation Pareto frontier over position, velocity, acceleration RMSE; lower is better; strict dominance requires no worse in all and better in at least one",
        "validation_only": True,
        "test_data_opened": False,
        "seeds": list(SEEDS),
        "mean_validation_metrics": means,
        "pareto_frontier": frontier,
        "selected_adaptive_variants": selected,
        "fixed_references": list(REFERENCE),
        "selection_script_sha256": sha256(Path(__file__).resolve()),
    }
    json_new(OUTPUT / "selection.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
