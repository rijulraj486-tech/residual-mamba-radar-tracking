#!/usr/bin/env python3
"""Aggregate paired test results and diagnose response at true acceleration changes.

Run only after validation selection and the frozen test evaluation exist.
Ground truth is used here strictly for post-selection diagnostics.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "experiments")]
from mamba_adaptive_acceleration_v1 import make_model_class, sha256
from mamba_adaptive_acceleration_features import adaptive_causal_acceleration_features

ROOT = PROJECT / "artifacts/results/synthetic/mamba_adaptive_acceleration_v1"
TEST_PATH = PROJECT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"
BASELINE = PROJECT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1"
FIXED = PROJECT / "artifacts/results/synthetic/mamba_acceleration_feature_ablation_v1"
SEEDS = (11, 22, 33, 44, 55)
SCALE = 1000.0
GROUPS = ("position", "velocity", "acceleration")
COMPARATORS = ("A0_existing_reference", "A1_ema_alpha025", "A1_ema_alpha050")


def mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return {"mean": mean, "std": math.sqrt(variance), "n": len(values)}


def result_path(variant: str, seed: int) -> Path:
    return ROOT / "final_test_evaluation" / variant / f"seed_{seed}.json"


def aggregate() -> dict:
    selection = json.loads((ROOT / "selection.json").read_text())
    variants = [*COMPARATORS, *selection["selected_adaptive_variants"]]
    records = {
        v: {s: json.loads(result_path(v, s).read_text()) for s in SEEDS}
        for v in variants
    }
    metric_keys = {
        "position_rmse_physical": "position_rmse_physical",
        "velocity_rmse_physical": "velocity_rmse_physical",
        "acceleration_rmse_physical": "acceleration_rmse_physical",
        "six_state_mse_scaled_units": "six_state_mse_scaled_units",
    }
    summary = {}
    for v in variants:
        ms = [records[v][s]["metrics"] for s in SEEDS]
        item = {key: mean_std([m[src] for m in ms]) for key, src in metric_keys.items()}
        item["per_axis_rmse_physical"] = {
            axis: mean_std([m["per_state_rmse_physical"][i] for m in ms])
            for i, axis in enumerate(("x", "vx", "ax", "y", "vy", "ay"))
        }
        item["temporal_windows"] = []
        for w in range(10):
            win_rows = [m["temporal_windows"][w] for m in ms]
            row = {
                "start_step_1based": win_rows[0]["start_step_1based"],
                "end_step_1based": win_rows[0]["end_step_1based"],
            }
            for metric in (*GROUPS, "six_state_rmse_physical"):
                row[f"{metric}_mean_std"] = mean_std([x[f"{metric}_rmse_physical"] if metric != "six_state_rmse_physical" else x[metric] for x in win_rows])
            row["six_state_mse_scaled_units_mean_std"] = mean_std([
                x["six_state_rmse_physical"] ** 2 / SCALE ** 2 for x in win_rows
            ])
            item["temporal_windows"].append(row)
        item["early_steps_1_10"] = item["temporal_windows"][0]
        if v.startswith("A3_"):
            alpha = [records[v][s]["metrics"]["adaptive_alpha"] for s in SEEDS]
            item["adaptive_alpha"] = {
                "across_sequence_values": {
                    key: mean_std([a[key] for a in alpha]) for key in ("mean", "std", "min", "max")
                },
                "mean_by_step_across_seeds": [
                    mean_std([a["mean_by_step"][t] for a in alpha]) for t in range(100)
                ],
            }
        item["parameter_count"] = sorted({records[v][s]["parameter_count"] for s in SEEDS})
        item["checkpoint_sha256_by_seed"] = {str(s): records[v][s]["checkpoint_sha256"] for s in SEEDS}
        summary[v] = item

    paired = {}
    for candidate in selection["selected_adaptive_variants"]:
        paired[candidate] = {}
        for comparator in COMPARATORS:
            deltas = {k: [] for k in metric_keys}
            axes = {axis: [] for axis in ("x", "vx", "ax", "y", "vy", "ay")}
            for seed in SEEDS:
                cm = records[candidate][seed]["metrics"]
                bm = records[comparator][seed]["metrics"]
                for k, src in metric_keys.items():
                    deltas[k].append(cm[src] - bm[src])
                for i, axis in enumerate(axes):
                    axes[axis].append(cm["per_state_rmse_physical"][i] - bm["per_state_rmse_physical"][i])
            temporal = []
            for w in range(10):
                row = {"start_step_1based": records[candidate][SEEDS[0]]["metrics"]["temporal_windows"][w]["start_step_1based"],
                       "end_step_1based": records[candidate][SEEDS[0]]["metrics"]["temporal_windows"][w]["end_step_1based"]}
                for metric in (*GROUPS, "six_state_mse_scaled_units"):
                    dv = []
                    for seed in SEEDS:
                        cw = records[candidate][seed]["metrics"]["temporal_windows"][w]
                        bw = records[comparator][seed]["metrics"]["temporal_windows"][w]
                        if metric == "six_state_mse_scaled_units":
                            cv = cw["six_state_rmse_physical"] ** 2 / SCALE ** 2
                            bv = bw["six_state_rmse_physical"] ** 2 / SCALE ** 2
                        else:
                            cv = cw[f"{metric}_rmse_physical"]
                            bv = bw[f"{metric}_rmse_physical"]
                        dv.append(cv - bv)
                    row[metric] = {**mean_std(dv), "per_seed": dict(zip(map(str, SEEDS), dv))}
                temporal.append(row)
            paired[candidate][comparator] = {
                **{k: {**mean_std(v), "per_seed": dict(zip(map(str, SEEDS), v)), "improved_seeds": sum(x < 0 for x in v)} for k, v in deltas.items()},
                "per_axis_rmse_physical": {axis: {**mean_std(v), "per_seed": dict(zip(map(str, SEEDS), v))} for axis, v in axes.items()},
                "temporal_window_paired_differences_candidate_minus_reference": temporal,
            }
    training = {}
    for v in selection["selected_adaptive_variants"]:
        runs = [json.loads((ROOT / v / f"seed_{s}" / "run_metrics.json").read_text()) for s in SEEDS]
        training[v] = {
            "duration_seconds": mean_std([r["training_duration_seconds"] for r in runs]),
            "duration_seconds_by_seed": {str(s): r["training_duration_seconds"] for s, r in zip(SEEDS, runs)},
            "best_epoch_by_seed": {str(s): r["best_epoch"] for s, r in zip(SEEDS, runs)},
            "actual_epochs_by_seed": {str(s): r["config"]["actual_epochs"] for s, r in zip(SEEDS, runs)},
            "parameter_count": sorted({r["parameter_count"] for r in runs}),
        }
    return {
        "selection_sha256": sha256(ROOT / "selection.json"),
        "test_sha256": json.loads((ROOT / "final_test_evaluation/evaluation_manifest.json").read_text())["test_dataset_sha256"],
        "seeds": list(SEEDS), "summaries": summary, "paired_differences_candidate_minus_reference": paired,
        "training": training,
    }


def acceleration_change_diagnostic() -> dict:
    """Measure whether low adaptive alpha coincides with real acceleration changes."""
    selection = json.loads((ROOT / "selection.json").read_text())
    selected = selection["selected_adaptive_variants"]
    if not selected:
        return {"status": "no adaptive variant selected; diagnostic not applicable"}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to load the selected Mamba models")
    from radar_tracking.cli import benchmark_mamba as benchmark
    from radar_tracking.models.tracker_model import BiMambaTrackerModel
    from mamba_acceleration_feature_ablation_v1 import make_model_class as make_fixed_class

    test = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    truth, observations, init = test["true_state"], test["observations"], test["initial_state"]
    device = torch.device("cuda")
    adaptive_class = make_model_class(BiMambaTrackerModel)
    fixed_class = make_fixed_class(BiMambaTrackerModel)
    variants = [*COMPARATORS, *selected]

    # Target acceleration-change magnitude is used only after selection to stratify diagnostic errors.
    target_acc = truth[:, (2, 5), :] * SCALE
    jerk = torch.linalg.vector_norm(target_acc[:, :, 2:] - target_acc[:, :, 1:-1], dim=1)
    q1, q2 = torch.quantile(jerk.flatten(), torch.tensor([1 / 3, 2 / 3]))
    masks = (jerk <= q1, (jerk > q1) & (jerk < q2), jerk >= q2)
    bin_names = ("low_true_acceleration_change", "middle_true_acceleration_change", "high_true_acceleration_change")
    alpha_result = {}
    feature_error = {}
    if selected:
        variant = selected[0]
        run = json.loads((ROOT / variant / "seed_11/run_metrics.json").read_text())
        feat = run["feature"]
        pos = observations.clone()
        # range/bearing to Cartesian via the existing model measurement conversion
        reference_config = json.loads((ROOT / variant / "seed_11/run_metrics.json").read_text())["config"]
        config = benchmark.MambaTrainConfig(**{k: v for k, v in reference_config.items() if k in {
            "seed", "epochs", "batch_size", "eval_batch_size", "learning_rate", "weight_decay", "patience", "min_delta", "embed_dim", "hidden_dim", "num_layers", "d_state", "d_conv", "expand", "dropout", "encoder_direction", "checkpoint_name"
        }})
        model = adaptive_class(task="radar_benchmark", state_dim=6, obs_dim=2, dt=benchmark.simparams_class().dt,
            embed_dim=config.embed_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers,
            d_state=config.d_state, d_conv=config.d_conv, expand=config.expand, dropout=config.dropout,
            encoder_direction="forward", acceleration_feature=feat).to(device)
        model.load_state_dict(torch.load(ROOT / variant / "seed_11/best_mamba_tracker.pt", map_location=device, weights_only=True))
        pos = model.measurement_to_cartesian(observations.to(device))
        alpha = adaptive_causal_acceleration_features(
            pos, init[:, (1, 4), 0].to(device), init[:, (2, 5), 0].to(device),
            observations[:, 0, :].to(device), observations[:, 1, :].to(device),
            benchmark.simparams_class().dt, sigma_range=feat["sigma_range"], sigma_bearing=feat["sigma_bearing"],
            alpha_min=feat["alpha_min"], alpha_max=feat["alpha_max"], reliability_scale=feat["reliability_scale"]
        )[1].cpu()[:, 2:]
        alpha_result = {
            "overall_mean": float(alpha.mean().item()),
            "overall_std": float(alpha.std(unbiased=False).item()),
            "mean_by_true_change_bin": {
                name: float(alpha[mask].mean().item()) for name, mask in zip(bin_names, masks)
            },
            "true_change_quantiles_mps2_per_sample_interval": [float(q1), float(q2)],
            "interpretation": "A lower alpha in the high-change group indicates that the rule applies stronger smoothing where genuine acceleration changes occur; it is a post-selection diagnostic only and does not feed the estimator.",
        }

    error_accum = {v: {name: {"sse": 0.0, "count": 0, "bias_sum": 0.0} for name in bin_names} for v in variants}
    for variant in variants:
        for seed in SEEDS:
            if variant == "A0_existing_reference":
                baseline_record = json.loads((BASELINE / f"seed_{seed}/run_metrics.json").read_text())["mamba"]
                cfgv = baseline_record["config"]
                config = benchmark.MambaTrainConfig(**cfgv)
                model = benchmark.build_mamba_model(benchmark.simparams_class(), config, device)
                checkpoint = BASELINE / f"seed_{seed}/best_mamba_tracker.pt"
            else:
                root = FIXED if variant.startswith("A1_") else ROOT
                run = json.loads((root / variant / f"seed_{seed}/run_metrics.json").read_text())
                cfgv = run["config"]
                config = benchmark.MambaTrainConfig(**{k: v for k, v in cfgv.items() if k in {
                    "seed", "epochs", "batch_size", "eval_batch_size", "learning_rate", "weight_decay", "patience", "min_delta", "embed_dim", "hidden_dim", "num_layers", "d_state", "d_conv", "expand", "dropout", "encoder_direction", "checkpoint_name"
                }})
                feat = run["feature"]
                if variant.startswith("A1_"):
                    model = make_fixed_class(BiMambaTrackerModel)(task="radar_benchmark", state_dim=6, obs_dim=2, dt=benchmark.simparams_class().dt,
                        embed_dim=config.embed_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers, d_state=config.d_state,
                        d_conv=config.d_conv, expand=config.expand, dropout=config.dropout, encoder_direction="forward", acceleration_feature=feat).to(device)
                else:
                    model = adaptive_class(task="radar_benchmark", state_dim=6, obs_dim=2, dt=benchmark.simparams_class().dt,
                        embed_dim=config.embed_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers, d_state=config.d_state,
                        d_conv=config.d_conv, expand=config.expand, dropout=config.dropout, encoder_direction="forward", acceleration_feature=feat).to(device)
                checkpoint = root / variant / f"seed_{seed}/best_mamba_tracker.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            pred = []
            with torch.no_grad():
                for start in range(0, truth.shape[0], config.eval_batch_size):
                    end = min(start + config.eval_batch_size, truth.shape[0])
                    pred.append(model(observations=observations[start:end].to(device), init_state=init[start:end].to(device))["state_estimates"][:, (2, 5), :].cpu())
            err = torch.cat(pred) * SCALE - target_acc
            for name, mask in zip(bin_names, masks):
                m = mask.unsqueeze(1).expand(-1, 2, -1)
                values = err[:, :, 2:][m]
                slot = error_accum[variant][name]
                slot["sse"] += float(values.square().sum().item())
                slot["bias_sum"] += float(values.sum().item())
                slot["count"] += values.numel()
    for variant in variants:
        feature_error[variant] = {}
        for name in bin_names:
            x = error_accum[variant][name]
            feature_error[variant][name] = {
                "acceleration_rmse_mps2": math.sqrt(x["sse"] / x["count"]),
                "acceleration_bias_mps2": x["bias_sum"] / x["count"],
                "scalar_state_count": x["count"],
            }
    return {
        "test_dataset_sha256": sha256(TEST_PATH),
        "selection_sha256": sha256(ROOT / "selection.json"),
        "ground_truth_use": "post-selection stratification only",
        "true_acceleration_change_definition": "Euclidean norm of consecutive true x/y acceleration difference, in m/s^2 across adjacent samples (dt=1 s); bins are pooled test-set tertiles",
        "adaptive_alpha": alpha_result,
        "model_acceleration_error_by_true_change_bin": feature_error,
    }


def main() -> None:
    summary = aggregate()
    (ROOT / "final_test_evaluation/paired_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    diagnostic = acceleration_change_diagnostic()
    (ROOT / "final_test_evaluation/acceleration_change_diagnostic.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "summary_path": str(ROOT / "final_test_evaluation/paired_summary.json"),
        "diagnostic_path": str(ROOT / "final_test_evaluation/acceleration_change_diagnostic.json"),
        "selected": json.loads((ROOT / "selection.json").read_text())["selected_adaptive_variants"],
        "paired": summary["paired_differences_candidate_minus_reference"],
        "diagnostic_alpha": diagnostic["adaptive_alpha"],
        "diagnostic_errors": diagnostic["model_acceleration_error_by_true_change_bin"],
    }, indent=2))


if __name__ == "__main__":
    main()
