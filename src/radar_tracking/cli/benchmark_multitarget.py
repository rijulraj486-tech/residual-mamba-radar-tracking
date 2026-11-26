import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from radar_tracking.models.tracker_model import BiMambaTrackerModel
from radar_tracking.multitarget.association_model import (
    AssociationModelConfig,
    AssociationTrainingConfig,
    build_association_examples,
    load_association_model,
    save_association_metrics,
    train_association_model,
)
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.metrics.gospa import gospa_metric
from radar_tracking.multitarget.multitarget_data import generate_multitarget_scenarios, rb_to_xy_np
from radar_tracking.multitarget.multitarget_trackers import (
    ImmGnnTracker,
    LearnedAssociationMambaGnnTracker,
    MambaGnnTracker,
)
from radar_tracking.config import simparams_class, validate_simparams


ROOT = Path("/home/spcom41/BiMambaTracker/radar_longseq_benchmark/Radar_Tracking_Benchmark")
DEFAULT_OUTPUT_DIR = ROOT / "multitarget_outputs"
DEFAULT_ASSOC_OUTPUT_DIR = ROOT / "association_outputs"
BEST_SINGLE_TARGET_DIR = ROOT / "benchmark_outputs_embed160"
METHOD_STYLES = {
    "IMM + GNN": {"color": "#d95f02", "marker": "s"},
    "Mamba + GNN": {"color": "#1b9e77", "marker": "^"},
    "Mamba + LearnedAssoc + GNN": {"color": "#1f78b4", "marker": "D"},
}


@dataclass
class MultiTargetConfig:
    preset: str = "standard"
    seed: int = 42
    num_scenes: int = 12
    num_targets: int = 3
    seq_len: int = 60
    detection_prob: float = 0.9
    clutter_rate: float = 2.0
    gate_distance: float = 1.5
    miss_cost: float = 2.5
    max_missed: int = 3
    confirm_hits: int = 2
    gospa_cutoff: float = 2.0
    gospa_order: int = 2
    gospa_alpha: float = 2.0
    mamba_window: int = 30
    max_birth_step: int = 12
    min_track_length: int = 18
    imm_model_bank: str = "research"
    enable_learned_assoc: bool = True
    learned_cost_weight: float = 1.0
    distance_cost_weight: float = 0.35
    min_match_probability: float = 0.05


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def apply_preset(config):
    if config.preset == "research":
        config.num_scenes = max(config.num_scenes, 10)
        config.num_targets = max(config.num_targets, 4)
        config.seq_len = max(config.seq_len, 60)
        config.detection_prob = min(config.detection_prob, 0.88)
        config.clutter_rate = max(config.clutter_rate, 3.0)
        config.gate_distance = max(config.gate_distance, 1.8)
        config.miss_cost = max(config.miss_cost, 3.0)
        config.max_missed = max(config.max_missed, 4)
        config.confirm_hits = max(config.confirm_hits, 3)
        config.mamba_window = max(config.mamba_window, 40)
        config.max_birth_step = max(config.max_birth_step, 18)
        config.min_track_length = max(config.min_track_length, 20)
        config.imm_model_bank = "research"
    elif config.preset == "large":
        config.num_scenes = max(config.num_scenes, 14)
        config.num_targets = max(config.num_targets, 4)
        config.seq_len = max(config.seq_len, 75)
        config.clutter_rate = max(config.clutter_rate, 3.5)
        config.max_missed = max(config.max_missed, 4)
        config.confirm_hits = max(config.confirm_hits, 3)
        config.mamba_window = max(config.mamba_window, 45)
        config.max_birth_step = max(config.max_birth_step, 24)
        config.min_track_length = max(config.min_track_length, 24)
        config.imm_model_bank = "research"
    return config


def load_best_mamba(device):
    metrics_path = BEST_SINGLE_TARGET_DIR / "benchmark_metrics.json"
    checkpoint_path = BEST_SINGLE_TARGET_DIR / "best_mamba_tracker.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Expected tuned Mamba benchmark outputs were not found.")

    with metrics_path.open() as f:
        metrics = json.load(f)
    train_cfg = metrics["mamba_train_config"]

    model = BiMambaTrackerModel(
        task="radar_benchmark",
        state_dim=6,
        obs_dim=2,
        dt=1.0,
        embed_dim=train_cfg["embed_dim"],
        hidden_dim=train_cfg["hidden_dim"],
        num_layers=train_cfg["num_layers"],
        d_state=train_cfg["d_state"],
        d_conv=train_cfg["d_conv"],
        expand=train_cfg["expand"],
        dropout=train_cfg["dropout"],
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, train_cfg


def build_association_configs(multitarget_config):
    train_config = AssociationTrainingConfig(
        seed=int(os.environ.get("MT_ASSOC_SEED", multitarget_config.seed + 1000)),
        train_num_scenes=int(os.environ.get("MT_ASSOC_TRAIN_SCENES", 48)),
        val_num_scenes=int(os.environ.get("MT_ASSOC_VAL_SCENES", 12)),
        num_targets=int(os.environ.get("MT_ASSOC_NUM_TARGETS", max(multitarget_config.num_targets, 4))),
        seq_len=int(os.environ.get("MT_ASSOC_SEQ_LEN", max(multitarget_config.mamba_window, 45))),
        detection_prob=float(os.environ.get("MT_ASSOC_DET_PROB", multitarget_config.detection_prob)),
        clutter_rate=float(os.environ.get("MT_ASSOC_CLUTTER_RATE", max(multitarget_config.clutter_rate, 2.5))),
        max_birth_step=int(os.environ.get("MT_ASSOC_MAX_BIRTH_STEP", multitarget_config.max_birth_step)),
        min_track_length=int(os.environ.get("MT_ASSOC_MIN_TRACK_LENGTH", multitarget_config.min_track_length)),
        window_size=int(os.environ.get("MT_ASSOC_WINDOW", multitarget_config.mamba_window)),
        min_history=int(os.environ.get("MT_ASSOC_MIN_HISTORY", 2)),
        max_missed_history=int(os.environ.get("MT_ASSOC_MAX_MISSED_HISTORY", multitarget_config.max_missed)),
        negative_ratio=int(os.environ.get("MT_ASSOC_NEGATIVE_RATIO", 4)),
        batch_size=int(os.environ.get("MT_ASSOC_BATCH_SIZE", 1024)),
        epochs=int(os.environ.get("MT_ASSOC_EPOCHS", 18)),
        learning_rate=float(os.environ.get("MT_ASSOC_LR", 1e-3)),
        weight_decay=float(os.environ.get("MT_ASSOC_WEIGHT_DECAY", 1e-4)),
        patience=int(os.environ.get("MT_ASSOC_PATIENCE", 4)),
        min_delta=float(os.environ.get("MT_ASSOC_MIN_DELTA", 1e-4)),
    )
    model_config = AssociationModelConfig(
        hidden_dim=int(os.environ.get("MT_ASSOC_HIDDEN_DIM", 128)),
        num_layers=int(os.environ.get("MT_ASSOC_NUM_LAYERS", 3)),
        dropout=float(os.environ.get("MT_ASSOC_DROPOUT", 0.10)),
    )
    return model_config, train_config


def train_or_load_association_model(multitarget_config, ca_model, simparams, mamba_model, device):
    assoc_output_dir = Path(os.environ.get("MT_ASSOC_OUTPUT_DIR", str(DEFAULT_ASSOC_OUTPUT_DIR)))
    assoc_output_dir.mkdir(parents=True, exist_ok=True)
    model_config, train_config = build_association_configs(multitarget_config)
    checkpoint_path = assoc_output_dir / train_config.checkpoint_name
    force_retrain = env_bool("MT_ASSOC_FORCE_RETRAIN", False)

    if checkpoint_path.exists() and not force_retrain:
        model, checkpoint = load_association_model(checkpoint_path, device)
        metadata = {
            "status": "loaded",
            "checkpoint_path": str(checkpoint_path),
            "model_config": checkpoint["model_config"],
            "train_config": checkpoint["train_config"],
            "best_val_loss": checkpoint.get("best_val_loss"),
        }
        return model, metadata

    print("Training learned association model on cluttered multi-target scenes...")
    set_seed(train_config.seed)
    train_scenes, _ = generate_multitarget_scenarios(
        ca_model,
        simparams,
        num_scenes=train_config.train_num_scenes,
        num_targets=train_config.num_targets,
        seq_len=train_config.seq_len,
        detection_prob=train_config.detection_prob,
        clutter_rate=train_config.clutter_rate,
        seed=train_config.seed,
        max_birth_step=train_config.max_birth_step,
        min_track_length=train_config.min_track_length,
    )
    val_scenes, _ = generate_multitarget_scenarios(
        ca_model,
        simparams,
        num_scenes=train_config.val_num_scenes,
        num_targets=train_config.num_targets,
        seq_len=train_config.seq_len,
        detection_prob=train_config.detection_prob,
        clutter_rate=train_config.clutter_rate,
        seed=train_config.seed + 1,
        max_birth_step=train_config.max_birth_step,
        min_track_length=train_config.min_track_length,
    )

    train_features, train_labels = build_association_examples(
        scenes=train_scenes,
        dt=simparams.dt,
        window_size=train_config.window_size,
        gate_distance=multitarget_config.gate_distance,
        model=mamba_model,
        device=device,
        min_history=train_config.min_history,
        max_missed_history=train_config.max_missed_history,
        negative_ratio=train_config.negative_ratio,
        seed=train_config.seed,
    )
    val_features, val_labels = build_association_examples(
        scenes=val_scenes,
        dt=simparams.dt,
        window_size=train_config.window_size,
        gate_distance=multitarget_config.gate_distance,
        model=mamba_model,
        device=device,
        min_history=train_config.min_history,
        max_missed_history=train_config.max_missed_history,
        negative_ratio=train_config.negative_ratio,
        seed=train_config.seed + 1,
    )

    model, metrics, checkpoint_path = train_association_model(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        model_config=model_config,
        train_config=train_config,
        device=device,
        output_dir=assoc_output_dir,
    )
    save_association_metrics(
        output_dir=assoc_output_dir,
        metrics=metrics,
        model_config=model_config,
        train_config=train_config,
    )
    metadata = {
        "status": "trained",
        "checkpoint_path": str(checkpoint_path),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "metrics": metrics,
    }
    return model, metadata


def plot_gospa_time(time_axis, series, output_path):
    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        style = METHOD_STYLES.get(label, {})
        plt.plot(
            time_axis,
            values,
            marker=style.get("marker", "o"),
            linewidth=2,
            label=label,
            color=style.get("color"),
        )
    plt.xlabel("Time Step")
    plt.ylabel("Average GOSPA")
    plt.title("Multi-Target Tracking Under Clutter")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_summary_bars(summary, output_path):
    labels = list(summary.keys())
    values = [summary[label]["mean_gospa"] for label in labels]
    colors = [METHOD_STYLES.get(label, {}).get("color", "#6c757d") for label in labels]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    best_idx = int(np.argmin(values))
    bars[best_idx].set_hatch("//")
    for idx, value in enumerate(values):
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=11, weight="bold")
    plt.ylabel("Mean GOSPA")
    plt.title("Mean GOSPA Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_snapshot(snapshot, output_path):
    plt.figure(figsize=(9, 7))
    true_xy = snapshot["truth"]
    det_xy = snapshot["detections"]
    true_traj = snapshot["true_trajectories"]

    for target_idx in range(true_traj.shape[0]):
        plt.plot(true_traj[target_idx, 0], true_traj[target_idx, 1], linewidth=1.5, alpha=0.7)

    if len(det_xy):
        plt.scatter(det_xy[:, 0], det_xy[:, 1], marker="x", s=50, color="#7570b3", label="Detections + Clutter")
    plt.scatter(true_xy[:, 0], true_xy[:, 1], marker="o", s=90, color="black", label="Ground Truth")
    for label, estimate_xy in snapshot["estimates"].items():
        if len(estimate_xy) == 0:
            continue
        style = METHOD_STYLES.get(label, {})
        plt.scatter(
            estimate_xy[:, 0],
            estimate_xy[:, 1],
            marker=style.get("marker", "o"),
            s=70,
            color=style.get("color"),
            label=label,
        )

    plt.xlabel("x position (km)")
    plt.ylabel("y position (km)")
    plt.title(
        f"Cluttered Scene Snapshot (t={snapshot['time_idx']}, "
        f"alive={len(true_xy)}, dets={len(det_xy)})"
    )
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_tracker_factories(config, measurement_cov, device, mamba_model, association_model, dt):
    factories = {
        "IMM + GNN": lambda: ImmGnnTracker(
            dt=dt,
            measurement_cov=measurement_cov,
            gate_distance=config.gate_distance,
            miss_cost=config.miss_cost,
            max_missed=config.max_missed,
            confirm_hits=config.confirm_hits,
            model_bank=config.imm_model_bank,
        ),
        "Mamba + GNN": lambda: MambaGnnTracker(
            model=mamba_model,
            device=device,
            dt=dt,
            measurement_cov=measurement_cov,
            gate_distance=config.gate_distance,
            miss_cost=config.miss_cost,
            max_missed=config.max_missed,
            confirm_hits=config.confirm_hits,
            window_size=config.mamba_window,
        ),
    }
    if association_model is not None:
        factories["Mamba + LearnedAssoc + GNN"] = lambda: LearnedAssociationMambaGnnTracker(
            model=mamba_model,
            association_model=association_model,
            device=device,
            dt=dt,
            measurement_cov=measurement_cov,
            gate_distance=config.gate_distance,
            miss_cost=config.miss_cost,
            max_missed=config.max_missed,
            confirm_hits=config.confirm_hits,
            window_size=config.mamba_window,
            learned_cost_weight=config.learned_cost_weight,
            distance_cost_weight=config.distance_cost_weight,
            min_match_probability=config.min_match_probability,
        )
    return factories


def main():
    config = MultiTargetConfig(
        preset=os.environ.get("MT_PRESET", "standard"),
        seed=int(os.environ.get("MT_SEED", 42)),
        num_scenes=int(os.environ.get("MT_NUM_SCENES", 12)),
        num_targets=int(os.environ.get("MT_NUM_TARGETS", 3)),
        seq_len=int(os.environ.get("MT_SEQ_LEN", 60)),
        detection_prob=float(os.environ.get("MT_DET_PROB", 0.9)),
        clutter_rate=float(os.environ.get("MT_CLUTTER_RATE", 2.0)),
        gate_distance=float(os.environ.get("MT_GATE_DISTANCE", 1.5)),
        miss_cost=float(os.environ.get("MT_MISS_COST", 2.5)),
        max_missed=int(os.environ.get("MT_MAX_MISSED", 3)),
        confirm_hits=int(os.environ.get("MT_CONFIRM_HITS", 2)),
        gospa_cutoff=float(os.environ.get("MT_GOSPA_CUTOFF", 2.0)),
        gospa_order=int(os.environ.get("MT_GOSPA_ORDER", 2)),
        gospa_alpha=float(os.environ.get("MT_GOSPA_ALPHA", 2.0)),
        mamba_window=int(os.environ.get("MT_MAMBA_WINDOW", 30)),
        max_birth_step=int(os.environ.get("MT_MAX_BIRTH_STEP", 12)),
        min_track_length=int(os.environ.get("MT_MIN_TRACK_LENGTH", 18)),
        imm_model_bank=os.environ.get("MT_IMM_MODEL_BANK", "research"),
        enable_learned_assoc=env_bool("MT_ENABLE_LEARNED_ASSOC", True),
        learned_cost_weight=float(os.environ.get("MT_LEARNED_COST_WEIGHT", 1.0)),
        distance_cost_weight=float(os.environ.get("MT_DISTANCE_COST_WEIGHT", 0.35)),
        min_match_probability=float(os.environ.get("MT_MIN_MATCH_PROB", 0.05)),
    )
    config = apply_preset(config)

    output_dir = Path(os.environ.get("MT_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    simparams = simparams_class()
    validate_simparams(simparams)
    ca_model = ConstantAcceleration2DModel(simparams)

    scenes, data_meta = generate_multitarget_scenarios(
        ca_model,
        simparams,
        num_scenes=config.num_scenes,
        num_targets=config.num_targets,
        seq_len=config.seq_len,
        detection_prob=config.detection_prob,
        clutter_rate=config.clutter_rate,
        seed=config.seed,
        max_birth_step=config.max_birth_step,
        min_track_length=config.min_track_length,
    )

    measurement_cov = np.asarray(data_meta["measurement_noise"], dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mamba_model, mamba_cfg = load_best_mamba(device)

    association_model = None
    association_meta = None
    if config.enable_learned_assoc:
        association_model, association_meta = train_or_load_association_model(
            multitarget_config=config,
            ca_model=ca_model,
            simparams=simparams,
            mamba_model=mamba_model,
            device=device,
        )

    tracker_factories = make_tracker_factories(
        config=config,
        measurement_cov=measurement_cov,
        device=device,
        mamba_model=mamba_model,
        association_model=association_model,
        dt=simparams.dt,
    )

    time_len = config.seq_len
    gospa_time = {label: np.zeros(time_len, dtype=np.float64) for label in tracker_factories}
    components = {
        label: {"localization": [], "cardinality": [], "missed": [], "false": [], "gospa": []}
        for label in tracker_factories
    }
    snapshot = None

    for scene_idx, scene in enumerate(scenes):
        trackers = {label: factory() for label, factory in tracker_factories.items()}

        for time_idx in range(config.seq_len):
            detections = scene["detections"][time_idx]
            truth_positions = scene["truth_sets"][time_idx]
            estimated_positions = {}

            for label, tracker in trackers.items():
                tracker.step(detections)
                estimated_positions[label] = tracker.estimated_positions()
                metric = gospa_metric(
                    truth_positions,
                    estimated_positions[label],
                    cutoff=config.gospa_cutoff,
                    order=config.gospa_order,
                    alpha=config.gospa_alpha,
                )
                gospa_time[label][time_idx] += metric["gospa"]
                components[label]["gospa"].append(metric["gospa"])
                components[label]["localization"].append(metric["localization"])
                components[label]["cardinality"].append(metric["cardinality"])
                components[label]["missed"].append(metric["missed"])
                components[label]["false"].append(metric["false"])

            if scene_idx == 0 and time_idx == config.seq_len // 2:
                snapshot = {
                    "time_idx": time_idx,
                    "truth": truth_positions,
                    "detections": rb_to_xy_np(detections),
                    "estimates": estimated_positions,
                    "true_trajectories": scene["true_trajectories_xy"][:, :, : time_idx + 1],
                }

    for label in gospa_time:
        gospa_time[label] /= config.num_scenes

    summary = {}
    for label, values in components.items():
        summary[label] = {
            "mean_gospa": float(np.mean(values["gospa"])),
            "mean_localization": float(np.mean(values["localization"])),
            "mean_cardinality": float(np.mean(values["cardinality"])),
            "mean_missed": float(np.mean(values["missed"])),
            "mean_false": float(np.mean(values["false"])),
        }

    plot_gospa_time(
        np.arange(config.seq_len),
        gospa_time,
        output_dir / "gospa_time_comparison.png",
    )
    plot_summary_bars(summary, output_dir / "gospa_summary.png")
    if snapshot is not None:
        plot_snapshot(snapshot, output_dir / "cluttered_scene_snapshot.png")

    metrics = {
        "config": asdict(config),
        "device": str(device),
        "data_meta": data_meta,
        "mamba_single_target_config": mamba_cfg,
        "association_model": association_meta,
        "summary": summary,
        "gospa_vs_time": {key: value.tolist() for key, value in gospa_time.items()},
    }
    with (output_dir / "multitarget_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    report_lines = [
        "# Multi-Target Clutter Benchmark",
        "",
        "## Compared Pipelines",
    ]
    for label in summary:
        report_lines.append(f"- {label}")
    report_lines.extend(
        [
            "",
            "## Scenario",
            f"- Preset: {config.preset}",
            f"- Scenes: {config.num_scenes}",
            f"- Targets per scene: {config.num_targets}",
            f"- Sequence length: {config.seq_len}",
            f"- Detection probability: {config.detection_prob}",
            f"- Clutter rate: {config.clutter_rate}",
            f"- Max birth step: {config.max_birth_step}",
            f"- Minimum track length: {config.min_track_length}",
            f"- IMM model bank: {config.imm_model_bank}",
            "",
            "## Mean GOSPA",
        ]
    )
    for label, stats in summary.items():
        report_lines.append(f"- {label}: {stats['mean_gospa']:.4f}")
    report_lines.extend(["", "## Mean Components"])
    for label, stats in summary.items():
        report_lines.append(
            f"- {label} localization/cardinality: "
            f"{stats['mean_localization']:.4f} / {stats['mean_cardinality']:.4f}"
        )
    if association_meta is not None:
        report_lines.extend(
            [
                "",
                "## Learned Association",
                f"- Status: {association_meta['status']}",
                f"- Checkpoint: {association_meta['checkpoint_path']}",
            ]
        )
    (output_dir / "multitarget_report.md").write_text("\n".join(report_lines))

    print("Multi-target benchmark complete")
    for label, stats in summary.items():
        print(f"{label} mean GOSPA: {stats['mean_gospa']:.6f}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
