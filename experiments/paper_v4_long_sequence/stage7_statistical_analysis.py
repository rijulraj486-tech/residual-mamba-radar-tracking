#!/usr/bin/env python3
"""Seed-level statistical analysis of the frozen Stage 6 evaluation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
STAGE6 = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage6_matched_length_test_evaluation_v1"
STAGE5B = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1"
OUTPUT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence/stage7_statistical_analysis_v1"
METRICS = {
    "position_rmse_m": {"label": "Position RMSE", "unit": "m"},
    "velocity_rmse_mps": {"label": "Velocity RMSE", "unit": "m/s"},
    "acceleration_rmse_mps2": {"label": "Acceleration RMSE", "unit": "m/s²"},
    "six_state_mse_scaled_units": {"label": "Six-state MSE", "unit": "scaled-state units"},
}
LENGTHS = (200, 400)
SEEDS = (11, 22, 33, 44, 55)
COLORS = {"M0": "#315A9A", "M1": "#D45B32"}
T95_DF4 = 2.7764451051977987


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")
    return actual


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    if confidence != 0.95:
        raise ValueError("Only the predeclared 95% t interval is implemented")
    n = len(values)
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(n))
    if n != 5:
        raise ValueError("This report's t critical value is set for five seed pairs")
    return mean - T95_DF4 * sem, mean + T95_DF4 * sem


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def paired_t_test(m1: np.ndarray, m0: np.ndarray) -> tuple[float, float]:
    d = np.asarray(m1 - m0, dtype=float)
    sd = float(np.std(d, ddof=1))
    t_value = float(np.mean(d) / (sd / math.sqrt(len(d)))) if sd else math.copysign(math.inf, float(np.mean(d)))
    df = len(d) - 1
    # Two-sided Student t tail for df=4, expressed as the regularized beta
    # I_{df/(df+t^2)}(df/2, 1/2), whose df=4 form is analytic.
    x = df / (df + t_value * t_value)
    root = math.sqrt(max(0.0, 1.0 - x))
    p = 1.0 - 1.5 * root + 0.5 * root**3
    return t_value, min(1.0, max(0.0, p))


def exact_signed_rank_p(differences: np.ndarray) -> float:
    """Two-sided exact sign-flip Wilcoxon p-value conditional on |differences|."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = average_ranks(np.abs(d))
    observed = abs(float(np.sum(np.sign(d) * ranks)))
    candidates = []
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        candidates.append(abs(float(np.dot(np.asarray(signs), ranks))))
    return sum(value >= observed - 1e-12 for value in candidates) / len(candidates)


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_path = STAGE6 / "stage6_evaluation_report.json"
    stage6_hash = verify_sidecar(report_path)
    report = load_json(report_path)
    if report["status"] != "PASS" or report["evaluation_matrix"]["completed_evaluations"] != 20:
        raise ValueError("Stage 6 report is not a complete PASS evaluation")
    # Confirm all files that Stage 6 recorded as output artifacts are unchanged.
    for filename, expected in report["output_artifact_hashes"].items():
        if filename == "plots":
            for relative, digest in expected.items():
                if sha256(PROJECT / relative) != digest:
                    raise ValueError(f"Stage 6 plot hash mismatch: {relative}")
        elif sha256(STAGE6 / filename) != expected:
            raise ValueError(f"Stage 6 output hash mismatch: {filename}")

    selection_path = STAGE5B / "checkpoint_selection_manifest.json"
    summary_path = STAGE5B / "stage5b_confirmation_summary.json"
    selection_hash = verify_sidecar(selection_path)
    summary_hash = verify_sidecar(summary_path)
    selection, summary = load_json(selection_path), load_json(summary_path)
    if selection_hash != report["checkpoint_integrity"]["selection_manifest_sha256"]:
        raise ValueError("Stage 5B selection manifest hash differs from Stage 6 provenance")
    if summary_hash != report["checkpoint_integrity"]["stage5B_summary_sha256"]:
        raise ValueError("Stage 5B summary hash differs from Stage 6 provenance")
    if selection.get("status") != "FROZEN_BEFORE_TEST_ACCESS" or selection.get("test_accessed") is not False:
        raise ValueError("Stage 5B checkpoint selection was not frozen before test access")

    selected = {(int(r["sequence_length"]), r["variant"], int(r["seed"])): r for r in selection["selected_checkpoints"]}
    trained = {(int(r["sequence_length"]), r["variant"], int(r["confirmation_seed"])): r for r in summary["run_summaries"]}
    stage6 = {(int(r["sequence_length"]), r["variant"], int(r["seed"])): r for r in report["per_seed_results"]}
    expected_keys = {(length, variant, seed) for length in LENGTHS for variant in ("M0", "M1") for seed in SEEDS}
    if set(selected) != expected_keys or set(trained) != expected_keys or set(stage6) != expected_keys:
        raise ValueError("Checkpoint or Stage 6 result matrix does not equal the expected 20 rows")

    pair_records = []
    for length in LENGTHS:
        for seed in SEEDS:
            m0_key, m1_key = (length, "M0", seed), (length, "M1", seed)
            a, b = trained[m0_key], trained[m1_key]
            if a["initial_trainable_weights_sha256"] != b["initial_trainable_weights_sha256"]:
                raise ValueError(f"Initial weights do not match for T={length}, seed={seed}")
            if a["training_config"]["initialization_seed"] != b["training_config"]["initialization_seed"]:
                raise ValueError(f"Initialization seeds do not match for T={length}, seed={seed}")
            if a["training_config"]["dataloader_shuffle_seed"] != b["training_config"]["dataloader_shuffle_seed"]:
                raise ValueError(f"Loader seeds do not match for T={length}, seed={seed}")
            epoch_a = {int(x["epoch"]): x["dataloader_order_sha256"] for x in a["epoch_history"]}
            epoch_b = {int(x["epoch"]): x["dataloader_order_sha256"] for x in b["epoch_history"]}
            common_epochs = sorted(set(epoch_a) & set(epoch_b))
            expected_common = list(range(1, min(max(epoch_a, default=0), max(epoch_b, default=0)) + 1))
            if common_epochs != expected_common or any(epoch_a[e] != epoch_b[e] for e in common_epochs):
                raise ValueError(f"Per-epoch loader order hashes do not match over shared epochs for T={length}, seed={seed}")
            for key in (m0_key, m1_key):
                selected_record, result = selected[key], stage6[key]
                if result["status"] != "PASS":
                    raise ValueError(f"Stage 6 result failed: {key}")
                if selected_record["checkpoint_sha256"] != result["checkpoint_sha256"]:
                    raise ValueError(f"Selected checkpoint differs from evaluated checkpoint: {key}")
                if selected_record["config_sha256"] != result["config_sha256"]:
                    raise ValueError(f"Selected config differs from evaluated config: {key}")
                checkpoint_path = PROJECT / result["checkpoint_path"]
                config_path = PROJECT / result["config_path"]
                training_record = PROJECT / result["training_record_path"]
                if sha256(checkpoint_path) != result["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint file hash changed: {key}")
                if sha256(config_path) != result["config_sha256"]:
                    raise ValueError(f"Config file hash changed: {key}")
                if sha256(training_record) != result["training_record_sha256"]:
                    raise ValueError(f"Training record hash changed: {key}")
                if key[1] == "M1":
                    stats_path = PROJECT / result["feature_statistics_path"]
                    expected_stats_hash = selected_record["feature_statistics_sha256"]
                    if expected_stats_hash != result["feature_statistics_sha256"] or sha256(stats_path) != expected_stats_hash:
                        raise ValueError(f"M1 train-only feature statistics provenance mismatch: {key}")
            row0, row1 = stage6[m0_key], stage6[m1_key]
            for field in ("test_file_sha256", "test_manifest_sha256", "dataset_manifest_sha256"):
                if row0[field] != row1[field]:
                    raise ValueError(f"M0/M1 test data provenance differs ({field}) at T={length}")
            data_record = report["dataset_integrity"]["lengths"][str(length)]
            if row0["test_file_sha256"] != data_record["test_file_sha256"]:
                raise ValueError(f"Stage 6 evaluation file differs from dataset provenance at T={length}")
            if row0["trajectory_count"] != 1024 or row1["trajectory_count"] != 1024:
                raise ValueError(f"Unexpected test trajectory count at T={length}")
            pair_records.append({
                "sequence_length": length,
                "seed": seed,
                "initial_trainable_weights_sha256": a["initial_trainable_weights_sha256"],
                "data_loader_shuffle_seed": a["training_config"]["dataloader_shuffle_seed"],
                "matched_epoch_order_hash_count": len(common_epochs),
                "checkpoint_M0_sha256": stage6[m0_key]["checkpoint_sha256"],
                "checkpoint_M1_sha256": stage6[m1_key]["checkpoint_sha256"],
                "config_M0_sha256": stage6[m0_key]["config_sha256"],
                "config_M1_sha256": stage6[m1_key]["config_sha256"],
                "test_file_sha256": row0["test_file_sha256"],
                "test_manifest_sha256": row0["test_manifest_sha256"],
                "trajectory_ids_sha256": data_record["trajectory_ids_sha256"],
                "same_test_trajectories_confirmed": True,
                "paired_training_initialization_and_loader_order_confirmed": True,
            })

    for relative, expected in report["source_hashes"].items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError(f"Stage 6 recorded source hash changed: {relative}")
    for length in LENGTHS:
        data_record = report["dataset_integrity"]["lengths"][str(length)]
        if data_record["trajectory_count"] != 1024:
            raise ValueError(f"Dataset trajectory count is invalid at T={length}")
        for path_key, hash_key in (("bundle_manifest_path", "bundle_manifest_sha256"), ("test_manifest_path", "test_manifest_sha256"), ("test_file_path", "test_file_sha256")):
            if sha256(PROJECT / data_record[path_key]) != data_record[hash_key]:
                raise ValueError(f"Current dataset file does not match Stage 6 record: {path_key}, T={length}")
    checks = {
        "stage6_report_sidecar_verified": True,
        "all_stage6_output_hashes_verified": True,
        "stage5b_summary_and_selection_sidecars_verified": True,
        "all_20_selected_checkpoint_and_config_hashes_verified": True,
        "all_20_training_record_hashes_verified": True,
        "all_20_source_hashes_verified": True,
        "all_10_seed_pairs_shared_initial_weights_and_epoch_loader_orders": True,
        "M0_M1_test_file_manifest_and_trajectory_id_sets_match_at_both_lengths": True,
        "test_trajectory_count_each_length": 1024,
        "per_trajectory_predictions_or_errors_available": False,
    }
    provenance = {
        "stage6_report_path": str(report_path.relative_to(PROJECT)),
        "stage6_report_sha256": stage6_hash,
        "stage5b_selection_manifest_path": str(selection_path.relative_to(PROJECT)),
        "stage5b_selection_manifest_sha256": selection_hash,
        "stage5b_summary_path": str(summary_path.relative_to(PROJECT)),
        "stage5b_summary_sha256": summary_hash,
        "dataset_provenance": report["dataset_integrity"]["lengths"],
        "paired_seed_provenance": pair_records,
        "validation_checks": checks,
    }
    return report, provenance, stage6


def compute_statistics(stage6: dict[tuple[int, str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    raw_test_records = []
    for length in LENGTHS:
        for metric, spec in METRICS.items():
            m0 = np.array([stage6[(length, "M0", seed)]["metrics"][metric] for seed in SEEDS], dtype=float)
            m1 = np.array([stage6[(length, "M1", seed)]["metrics"][metric] for seed in SEEDS], dtype=float)
            diff = m1 - m0
            pct = 100.0 * diff / m0
            for i, seed in enumerate(SEEDS):
                seed_rows.append({
                    "sequence_length": length, "metric": metric, "seed": seed,
                    "M0": m0[i], "M1": m1[i],
                    "difference_M1_minus_M0": diff[i],
                    "percentage_difference_M1_vs_M0": pct[i],
                    "percentage_improvement_positive_favors_M1": -pct[i],
                })
            diff_mean, diff_sd = mean_sd(diff)
            pct_mean, pct_sd = mean_sd(pct)
            diff_ci = t_interval(diff)
            pct_ci = t_interval(pct)
            d_z = diff_mean / diff_sd if diff_sd else math.nan
            hedges_j = 1.0 - 3.0 / (4 * (len(diff) - 1) - 1)
            t_stat, t_p = paired_t_test(m1, m0)
            w_p = exact_signed_rank_p(diff)
            nonzero = diff[diff != 0]
            ranks = average_ranks(np.abs(nonzero))
            w_plus = float(np.sum(ranks[nonzero > 0]))
            w_minus = float(np.sum(ranks[nonzero < 0]))
            test_record = {
                "sequence_length": length,
                "metric": metric,
                "paired_t_statistic": t_stat,
                "paired_t_df": len(diff) - 1,
                "paired_t_p_two_sided": t_p,
                "wilcoxon_signed_rank_statistic_min_rank_sum": min(w_plus, w_minus),
                "wilcoxon_exact_sign_flip_p_two_sided": w_p,
                "wilcoxon_p_method": "exact enumeration of all 2^5 sign flips on observed absolute-difference ranks; two-sided, ties use average ranks",
            }
            raw_test_records.append(test_record)
            ci_rows.extend([
                {"sequence_length": length, "metric": metric, "estimand": "mean_paired_absolute_difference_M1_minus_M0", "unit": spec["unit"], "n_seed_pairs": 5, "estimate": diff_mean, "ci_method": "two-sided Student t interval on paired seed differences; df=4", "confidence_level": 0.95, "ci_lower": diff_ci[0], "ci_upper": diff_ci[1]},
                {"sequence_length": length, "metric": metric, "estimand": "mean_paired_percentage_difference_M1_vs_M0", "unit": "%", "n_seed_pairs": 5, "estimate": pct_mean, "ci_method": "two-sided Student t interval on paired per-seed percentage differences; df=4", "confidence_level": 0.95, "ci_lower": pct_ci[0], "ci_upper": pct_ci[1]},
            ])
            test_rows.append({
                **test_record,
                "cohens_dz": d_z,
                "hedges_corrected_gz": hedges_j * d_z,
                "effect_size_definition": "mean paired difference / sample SD of paired differences; Hedges correction J(df=4) applied",
                "holm_adjusted_p_t_family": None,
                "holm_adjusted_p_wilcoxon_family": None,
                "inferential_caveat": "n=5 seed pairs; exploratory, low power, normality/exchangeability assumptions are weakly assessable",
            })
    p_t = [r["paired_t_p_two_sided"] for r in test_rows]
    p_w = [r["wilcoxon_exact_sign_flip_p_two_sided"] for r in test_rows]
    q_t, q_w = holm_adjust(p_t), holm_adjust(p_w)
    for i, row in enumerate(test_rows):
        row["holm_adjusted_p_t_family"] = q_t[i]
        row["holm_adjusted_p_wilcoxon_family"] = q_w[i]
    return seed_rows, ci_rows, test_rows


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for length in LENGTHS:
        for metric, spec in METRICS.items():
            rows = [r for r in seed_rows if r["sequence_length"] == length and r["metric"] == metric]
            d = np.array([r["difference_M1_minus_M0"] for r in rows])
            pct = np.array([r["percentage_difference_M1_vs_M0"] for r in rows])
            m0 = np.array([r["M0"] for r in rows])
            m1 = np.array([r["M1"] for r in rows])
            summary.append({
                "sequence_length": length, "metric": metric, "unit": spec["unit"], "n_seed_pairs": 5,
                "M0_mean": float(m0.mean()), "M0_sample_sd": float(m0.std(ddof=1)),
                "M1_mean": float(m1.mean()), "M1_sample_sd": float(m1.std(ddof=1)),
                "paired_difference_mean_M1_minus_M0": float(d.mean()),
                "paired_difference_sample_sd": float(d.std(ddof=1)),
                "paired_difference_median_M1_minus_M0": float(np.median(d)),
                "paired_difference_min": float(d.min()), "paired_difference_max": float(d.max()),
                "paired_percentage_difference_mean": float(pct.mean()),
                "paired_percentage_difference_sample_sd": float(pct.std(ddof=1)),
                "paired_percentage_difference_median": float(np.median(pct)),
                "seeds_improved_by_M1": int(np.sum(d < 0)),
                "improvement_direction_consistent_all_seeds": bool(np.all(d < 0)),
            })
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256(path)


def make_figures(seed_rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]], figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True)
    seed_lookup = {(r["sequence_length"], r["metric"], r["seed"]): r for r in seed_rows}
    ci_lookup = {(r["sequence_length"], r["metric"], r["estimand"]): r for r in ci_rows}
    hashes = {}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 150, "savefig.dpi": 300})
    for metric, spec in METRICS.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
        for row_idx, length in enumerate(LENGTHS):
            ax_pair, ax_diff = axes[row_idx]
            for seed_idx, seed in enumerate(SEEDS):
                row = seed_lookup[(length, metric, seed)]
                ax_pair.plot([0, 1], [row["M0"], row["M1"]], marker="o", markersize=4, color="#808080", alpha=0.8, linewidth=1)
                ax_diff.scatter(seed_idx, row["difference_M1_minus_M0"], s=38, color=COLORS["M1"] if row["difference_M1_minus_M0"] > 0 else COLORS["M0"], zorder=3)
            ax_pair.set_xticks([0, 1], ["M0", "M1"])
            ax_pair.set_ylabel(f"{spec['label']} ({spec['unit']})")
            ax_pair.set_title(f"T={length}: paired seed results")
            ax_pair.grid(axis="y", alpha=0.25)
            dci = ci_lookup[(length, metric, "mean_paired_absolute_difference_M1_minus_M0")]
            mean_diff = dci["estimate"]
            err_low, err_high = mean_diff - dci["ci_lower"], dci["ci_upper"] - mean_diff
            ax_diff.errorbar([5.5], [mean_diff], yerr=np.array([[err_low], [err_high]]), fmt="D", color="#171717", capsize=5, linewidth=1.6, zorder=4, label="Mean and 95% t CI")
            ax_diff.axhline(0, color="#555555", linewidth=1, linestyle="--")
            ax_diff.set_xlim(-0.5, 6.0)
            ax_diff.set_xticks([*range(5), 5.5], [*[str(s) for s in SEEDS], "Mean"])
            ax_diff.set_xlabel("Seed-level paired difference and mean 95% CI")
            ax_diff.set_ylabel(f"M1 − M0 ({spec['unit']})")
            ax_diff.set_title(f"T={length}: paired differences")
            ax_diff.grid(axis="y", alpha=0.25)
            ax_diff.legend(frameon=False, fontsize=8)
        fig.suptitle(f"{spec['label']}: paired Stage 6 seed analysis", y=1.01, fontsize=13)
        fig.tight_layout()
        path = figure_dir / f"paired_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        hashes[str(path.relative_to(OUTPUT))] = sha256(path)

    # Compact, unit-free comparison: positive values indicate M1 improvement.
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    metric_order = list(METRICS)
    ypos = np.arange(len(metric_order))
    offsets = {-1: -0.12, 1: 0.12}
    for length, color in zip(LENGTHS, ("#2B6F9C", "#C65D31")):
        means, lows, highs = [], [], []
        for metric in metric_order:
            ci = ci_lookup[(length, metric, "mean_paired_percentage_difference_M1_vs_M0")]
            # Convert M1-M0 to improvement sign, so positive favors M1.
            means.append(-ci["estimate"])
            lows.append(-ci["ci_upper"])
            highs.append(-ci["ci_lower"])
        xerr = np.array([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
        ax.errorbar(means, ypos + offsets[1 if length == 400 else -1], xerr=xerr, fmt="o", capsize=4, color=color, label=f"T={length}", linewidth=1.5)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(ypos, [METRICS[m]["label"] for m in metric_order])
    ax.set_xlabel("Mean paired improvement, %  (100 × (M0 − M1) / M0)")
    ax.set_title("M1 improvement by sequence length (mean and 95% t CI across five seeds)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "summary_m1_improvement_by_length.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    hashes[str(path.relative_to(OUTPUT))] = sha256(path)
    return hashes


def md_num(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}g}"


def render_report(summary: list[dict[str, Any]], seed_rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], provenance: dict[str, Any], figure_hashes: dict[str, str]) -> str:
    lines = [
        "# Stage 7: Statistical Analysis of Final Matched-Length Test Results", "",
        "**Scope:** existing Stage 6 held-out results only. No inference was rerun; no model, checkpoint, normalization statistic, or dataset was changed.", "",
        "## Provenance and pairing checks", "",
        f"Verified Stage 6 report SHA-256: `{provenance['stage6_report_sha256']}`. Verified Stage 5B selection manifest SHA-256: `{provenance['stage5b_selection_manifest_sha256']}`.", "",
        "All 20 Stage 6 evaluations passed hash checks against the Stage 5B selected checkpoint/config records, including M1 train-only feature statistics. For each of the ten same-length, same-seed M0/M1 pairs, Stage 5B records confirm matching initial trainable-weight hashes, initialization seeds, data-loader shuffle seeds, and loader-order hashes for every epoch both runs shared. Stage 6 records confirm identical test split file, manifest, and trajectory-ID-set hashes within each length; each test set contains 1,024 trajectories.", "",
        "## Methods", "",
        "The inferential unit is the training seed pair (n=5 per length), not the test trajectory. Differences are paired as M1 minus M0; percentage difference is 100 × (M1 − M0) / M0, so negative values favor M1. We report mean, sample SD, median, and all five paired observations.", "",
        "The 95% confidence intervals are two-sided Student t intervals (df=4) over the five paired seed differences; percentage-difference intervals apply the same calculation to the five per-seed percentages. These intervals describe uncertainty in the mean paired effect across seed-level training replicates under the t model. Cohen dz is the mean paired difference divided by the sample SD of paired differences; Hedges-corrected gz applies J(df=4). Paired t-tests and exact two-sided Wilcoxon signed-rank sign-flip tests are exploratory. The exact Wilcoxon p-value enumerates all 2^5 sign assignments using average ranks for ties. Holm adjustment is reported separately across the eight length-by-metric comparisons for each test family.", "",
        "With five seed pairs, distributional assumptions cannot be checked meaningfully, estimates are sensitive to individual seeds, and power is low. P-values and intervals are exploratory and should not be treated as definitive evidence. No significance claim is based on effect magnitude.", "",
        "## Paired seed summary", "",
        "Paired differences are M1 − M0; negative values favor M1.", "",
        "| T | Metric | M0 mean ± SD | M1 mean ± SD | Mean paired difference ± SD | Median difference | Mean paired % difference ± SD | Seeds improved |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['sequence_length']} | {row['metric']} | {md_num(row['M0_mean'])} ± {md_num(row['M0_sample_sd'])} | {md_num(row['M1_mean'])} ± {md_num(row['M1_sample_sd'])} | {md_num(row['paired_difference_mean_M1_minus_M0'])} ± {md_num(row['paired_difference_sample_sd'])} {row['unit']} | {md_num(row['paired_difference_median_M1_minus_M0'])} | {md_num(row['paired_percentage_difference_mean'])}% ± {md_num(row['paired_percentage_difference_sample_sd'])}% | {row['seeds_improved_by_M1']}/5 |"
        )
    lines.extend(["", "## Five paired seed results", "", "All M0, M1, absolute, and percentage results are shown below. Percentage differences are relative to that seed's M0 result.", ""])
    for length in LENGTHS:
        lines.extend([f"### T={length}", "", "| Metric | Seed | M0 | M1 | M1 − M0 | Difference (%) |", "|---|---:|---:|---:|---:|---:|"])
        for metric in METRICS:
            for row in [r for r in seed_rows if r["sequence_length"] == length and r["metric"] == metric]:
                lines.append(f"| {metric} | {row['seed']} | {md_num(row['M0'])} | {md_num(row['M1'])} | {md_num(row['difference_M1_minus_M0'])} | {md_num(row['percentage_difference_M1_vs_M0'], 4)}% |")
        lines.append("")
    lines.extend(["## Confidence intervals and effect sizes", "", "CIs are for the mean paired difference unless specified. Intervals crossing zero do not establish equivalence; intervals excluding zero are still exploratory at n=5 and across multiple endpoints.", "", "| T | Metric | Mean difference 95% CI | Mean percent difference 95% CI | Cohen dz | Hedges gz |", "|---:|---|---:|---:|---:|---:|"])
    ci_map = {(r["sequence_length"], r["metric"], r["estimand"]): r for r in ci_rows}
    tests_map = {(r["sequence_length"], r["metric"]): r for r in test_rows}
    for length in LENGTHS:
        for metric, spec in METRICS.items():
            absolute = ci_map[(length, metric, "mean_paired_absolute_difference_M1_minus_M0")]
            pct = ci_map[(length, metric, "mean_paired_percentage_difference_M1_vs_M0")]
            tr = tests_map[(length, metric)]
            lines.append(f"| {length} | {metric} | [{md_num(absolute['ci_lower'])}, {md_num(absolute['ci_upper'])}] {spec['unit']} | [{md_num(pct['ci_lower'],4)}%, {md_num(pct['ci_upper'],4)}%] | {md_num(tr['cohens_dz'])} | {md_num(tr['hedges_corrected_gz'])} |")
    lines.extend(["", "## Exploratory paired tests", "", "Two-sided p-values are shown raw and Holm-adjusted within test family across the eight metric-by-length comparisons.", "", "| T | Metric | Paired t | t p | t Holm p | Wilcoxon W | Exact Wilcoxon p | Wilcoxon Holm p |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for r in test_rows:
        lines.append(f"| {r['sequence_length']} | {r['metric']} | {r['paired_t_statistic']:.4g} (df=4) | {r['paired_t_p_two_sided']:.6g} | {r['holm_adjusted_p_t_family']:.6g} | {r['wilcoxon_signed_rank_statistic_min_rank_sum']:.4g} | {r['wilcoxon_exact_sign_flip_p_two_sided']:.6g} | {r['holm_adjusted_p_wilcoxon_family']:.6g} |")
    lines.extend(["", "## Trajectory-level inference", "", "Stage 6 retained per-checkpoint aggregate metrics and 10-step temporal-window metrics, but no per-trajectory predictions, residuals, or squared-error contributions. Identical test trajectory sets are provenance-confirmed, but paired trajectory-level outcomes cannot be calculated from the retained aggregates. A hierarchical bootstrap resampling seeds and trajectories would therefore be unsupported; no trajectory bootstrap was performed and no `trajectory_bootstrap_results.csv` is emitted.", "", "The additional artifact needed is per-trajectory predictions or, for these metrics, per-trajectory sufficient error contributions by checkpoint: position/velocity/acceleration squared-error sums and counts, plus six-state squared-error sums and counts, keyed by trajectory ID and checkpoint. This would allow paired trajectory resampling nested within seed resampling while preserving the M0/M1 pairing.", "", "## Interpretation", ""])
    for length in LENGTHS:
        lines.append(f"- At T={length}, M1 improves by seed in: " + ", ".join(f"{r['metric']} {r['seeds_improved_by_M1']}/5" for r in summary if r["sequence_length"] == length) + ".")
    lines.extend([
        "- At T=200, position RMSE is lower for M1 in 2/5 pairs; the mean paired difference is −4.54 m (95% t CI [−31.45, 22.36]) and the mean percentage difference is −0.66% (95% t CI [−4.94%, 3.62%]). This is compatible with either a modest improvement or deterioration, so position preservation/equivalence is not established. At T=400, position RMSE is lower in all five pairs (mean difference −522.06 m; 95% t CI [−661.95, −382.16]); this remains an exploratory seed-level result.",
        "- Mean paired percentage improvements are larger at T=400 for all four endpoints: position 18.83% vs 0.66%, velocity 74.40% vs 40.42%, acceleration 63.55% vs 55.58%, and six-state MSE 62.41% vs 25.04%. This cross-length comparison is descriptive; no cross-length inferential test was specified.",
        "- Paired t-tests with Holm adjustment are below 0.05 for velocity, acceleration, and six-state MSE at T=200 and all four metrics at T=400; T=200 position is not. However, exact Wilcoxon p-values are at least 0.0625 (the minimum attainable two-sided value with five nonzero pairs), and no Wilcoxon result is below 0.05 even before adjustment. Treat the t-test results as model-dependent exploratory evidence, not definitive significance. Shared initialization and loader order make the seed contrasts paired, but five pairs do not support strong population-level conclusions.",
        "", "## Figures", "",
    ])
    for path in figure_hashes:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Limitations", "", "Only five paired training seeds are available. Per-trajectory Stage 6 outputs were not retained, so uncertainty over the held-out trajectory population cannot be estimated while respecting the seed/trajectory hierarchy. The seed-level confidence intervals represent seed-replicate uncertainty conditional on the fixed Stage 2 test sets; they do not represent uncertainty from drawing a new test set. All tests are exploratory, with low power and multiple endpoints. No model selection or tuning was performed from these test results.", ""])
    return "\n".join(lines)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 7 output directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    report, provenance, stage6 = validate_inputs()
    seed_rows, ci_rows, test_rows = compute_statistics(stage6)
    summary = summarize(seed_rows)
    csv_hashes = {
        "paired_seed_statistics.csv": write_csv(OUTPUT / "paired_seed_statistics.csv", seed_rows),
        "confidence_intervals.csv": write_csv(OUTPUT / "confidence_intervals.csv", ci_rows),
        "statistical_tests.csv": write_csv(OUTPUT / "statistical_tests.csv", test_rows),
    }
    figure_hashes = make_figures(seed_rows, ci_rows, OUTPUT / "figures")
    statistical_summary = {
        "stage": "Stage 7 statistical analysis of Stage 6 matched-length test results",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {"training": False, "fine_tuning": False, "model_changes": False, "dataset_changes": False, "test_based_tuning": False},
        "metrics": METRICS,
        "lengths": list(LENGTHS),
        "seeds": list(SEEDS),
        "seed_level_summary": summary,
        "paired_seed_statistics": seed_rows,
        "confidence_intervals": ci_rows,
        "statistical_tests": test_rows,
        "multiple_testing": {"method": "Holm step-down", "families": "Eight metric-by-length comparisons separately for paired t and Wilcoxon test families"},
        "trajectory_level_analysis": {"supported": False, "reason": "Stage 6 did not retain per-trajectory predictions, residuals, or sufficient error contributions", "hierarchical_bootstrap_performed": False, "additional_artifact_required": "Per-checkpoint per-trajectory predictions or squared-error sums/counts by metric, keyed by test trajectory ID."},
        "interpretation": {
            "paired_seed_effects_are_exploratory": True,
            "five_seed_inference_limitation": True,
            "definitive_statistical_significance_claimed": False,
            "paired_t_tests_holm_below_0_05": [
                {"sequence_length": r["sequence_length"], "metric": r["metric"]}
                for r in test_rows if r["holm_adjusted_p_t_family"] < 0.05
            ],
            "wilcoxon_results_below_0_05": [
                {"sequence_length": r["sequence_length"], "metric": r["metric"]}
                for r in test_rows if r["wilcoxon_exact_sign_flip_p_two_sided"] < 0.05
            ],
            "trajectory_population_uncertainty_estimated": False,
            "T200_position_improved_seed_count": next(r["seeds_improved_by_M1"] for r in summary if r["sequence_length"] == 200 and r["metric"] == "position_rmse_m"),
            "all_mean_percentage_improvements_larger_at_T400": all(
                next(-r["paired_percentage_difference_mean"] for r in summary if r["sequence_length"] == 400 and r["metric"] == metric)
                > next(-r["paired_percentage_difference_mean"] for r in summary if r["sequence_length"] == 200 and r["metric"] == metric)
                for metric in METRICS
            ),
        },
        "validation": provenance["validation_checks"],
        "output_hashes": {**csv_hashes, "figures": figure_hashes},
    }
    json_path = OUTPUT / "stage7_statistical_summary.json"
    json_path.write_text(json.dumps(statistical_summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    json_hash = sha256(json_path)
    json_path.with_name(json_path.name + ".sha256").write_text(f"{json_hash}  {json_path.name}\n", encoding="ascii")
    markdown = render_report(summary, seed_rows, ci_rows, test_rows, provenance, figure_hashes)
    md_path = OUTPUT / "stage7_statistical_report.md"
    md_path.write_text(markdown, encoding="utf-8")
    md_hash = sha256(md_path)
    md_path.with_name(md_path.name + ".sha256").write_text(f"{md_hash}  {md_path.name}\n", encoding="ascii")
    provenance_out = {
        "stage": "Stage 7 statistical analysis provenance",
        "created_at_utc": statistical_summary["created_at_utc"],
        "status": "PASS",
        "project_relative_analysis_script": str(Path(__file__).resolve().relative_to(PROJECT)),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "source_stage6_and_stage5b": provenance,
        "validation_checks": provenance["validation_checks"],
        "output_hashes": {**csv_hashes, "figures": figure_hashes, "stage7_statistical_summary.json": json_hash, "stage7_statistical_report.md": md_hash},
        "methods": {"confidence_intervals": "Student t intervals over paired seed differences, df=4", "paired_tests": ["paired t test", "exact two-sided Wilcoxon signed-rank sign-flip enumeration"], "multiple_testing": "Holm correction separately within t and Wilcoxon families", "trajectory_bootstrap": "not performed; per-trajectory Stage 6 outcomes unavailable"},
    }
    provenance_path = OUTPUT / "provenance.json"
    provenance_path.write_text(json.dumps(provenance_out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provenance_hash = sha256(provenance_path)
    provenance_path.with_name(provenance_path.name + ".sha256").write_text(f"{provenance_hash}  {provenance_path.name}\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "output_directory": str(OUTPUT), "seed_comparisons": len(seed_rows), "ci_rows": len(ci_rows), "test_rows": len(test_rows), "trajectory_bootstrap": "not supported; no per-trajectory outcomes", "summary_sha256": json_hash, "markdown_sha256": md_hash, "provenance_sha256": provenance_hash}, indent=2))


if __name__ == "__main__":
    main()
