#!/usr/bin/env python3

"""Benchmark EKF, KalmanNet, and standalone pure BiMamba on NCLT."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from radar_tracking.cli.nclt_benchmark import NCLTEKF, evaluate_ekf_sessions
from radar_tracking.cli.nclt_bimamba_pure_experiment import (
    ExperimentConfig as PureConfig,
    NCLTPureWindowDataset,
    PureBiMambaFilterSmoother,
    build_hybrid_teacher,
    collate_windows,
    evaluate_sessions as evaluate_pure_bimamba_sessions,
    load_pt,
    make_plot as make_pure_plot,
    set_seed,
    train_model as train_pure_bimamba_model,
)
from radar_tracking.cli.nclt_tbptt_experiment import (
    SimpleNCLTKalmanNet,
    evaluate_sessions as evaluate_kalmannet_sessions,
)


def maybe_train_pure_bimamba(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    force_train: bool,
    seed: int,
    teacher_metrics: Path | None,
    teacher_checkpoint: Path | None,
) -> tuple[dict, Path]:
    metrics_path = output_dir / "nclt_bimamba_pure_metrics.json"
    checkpoint_path = output_dir / "best_nclt_bimamba_pure.pt"
    if metrics_path.exists() and checkpoint_path.exists() and not force_train:
        return json.loads(metrics_path.read_text()), checkpoint_path

    config = PureConfig(seed=seed)
    train_sessions = load_pt(data_dir / "train.pt")
    val_sessions = load_pt(data_dir / "val.pt")
    test_sessions = load_pt(data_dir / "test.pt")

    train_dataset = NCLTPureWindowDataset(train_sessions, seq_len=config.seq_len, stride=config.stride)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
    )

    model = PureBiMambaFilterSmoother(
        filter_embed_dim=config.filter_embed_dim,
        filter_hidden_dim=config.filter_hidden_dim,
        filter_layers=config.filter_layers,
        smoother_embed_dim=config.smoother_embed_dim,
        smoother_hidden_dim=config.smoother_hidden_dim,
        smoother_layers=config.smoother_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        dropout=config.dropout,
    ).to(device)
    teacher_model = None
    if teacher_metrics is not None and teacher_checkpoint is not None:
        teacher_model = build_hybrid_teacher(teacher_metrics, teacher_checkpoint, device)
    model, train_summary = train_pure_bimamba_model(
        model,
        train_loader,
        val_sessions,
        config,
        device,
        teacher_model=teacher_model,
    )
    torch.save(model.state_dict(), checkpoint_path)
    test_metrics = evaluate_pure_bimamba_sessions(model, test_sessions, device)

    plot_path = output_dir / "nclt_bimamba_pure_trajectories.png"
    make_pure_plot(test_metrics, plot_path)

    payload = {
        "config": asdict(config),
        "train_summary": train_summary,
        "test_summary": {
            "overall_rmse_m": test_metrics["overall_rmse_m"],
            "avg_gps_rmse_m": test_metrics["avg_gps_rmse_m"],
            "avg_wheel_rmse_m": test_metrics["avg_wheel_rmse_m"],
            "per_trajectory": test_metrics["per_trajectory"],
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "trajectory_plot": str(plot_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    return payload, checkpoint_path


def plot_comparison(output_path: Path, test_sessions: list[dict], ekf_metrics: dict, kn_metrics: dict, pure_metrics: dict) -> None:
    num = min(3, len(test_sessions))
    fig, axes = plt.subplots(1, num, figsize=(6 * num, 5))
    if num == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        gt = test_sessions[idx]["ground_truth"].float()
        ekf = ekf_metrics["predictions"][idx]
        kn = kn_metrics["predictions"][idx]
        bm = pure_metrics["predictions"][idx]
        ax.plot(gt[:, 0], gt[:, 1], label="Ground truth", linewidth=2)
        ax.plot(ekf[:, 0], ekf[:, 1], label="EKF", linewidth=1.2)
        ax.plot(kn[:, 0], kn[:, 1], label="KalmanNet", linewidth=1.2)
        ax.plot(bm[:, 0], bm[:, 1], label="Pure BiMamba", linewidth=1.2)
        ax.set_title(test_sessions[idx]["data_date"])
        ax.set_xlabel("X / North (m)")
        ax.set_ylabel("Y / East (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone pure BiMamba benchmark on NCLT.")
    parser.add_argument("--data-dir", default="./data/nclt_fusion")
    parser.add_argument("--output-dir", default="./nclt_benchmark_outputs_pure")
    parser.add_argument("--bimamba-dir", default="./nclt_bimamba_pure_outputs")
    parser.add_argument("--kalmannet-dir", default="./nclt_tbptt_outputs")
    parser.add_argument("--teacher-metrics", default="./nclt_bimamba_outputs_highlevel/nclt_bimamba_metrics.json")
    parser.add_argument("--teacher-checkpoint", default="./nclt_bimamba_outputs_highlevel/best_nclt_bimamba.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-train-bimamba", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("nclt_benchmark_pure.py requires CUDA.")

    set_seed(args.seed)
    device = torch.device("cuda")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bimamba_dir = Path(args.bimamba_dir)
    bimamba_dir.mkdir(parents=True, exist_ok=True)
    kalmannet_dir = Path(args.kalmannet_dir)
    test_sessions = load_pt(data_dir / "test.pt")

    pure_payload, _ = maybe_train_pure_bimamba(
        data_dir=data_dir,
        output_dir=bimamba_dir,
        device=device,
        force_train=args.force_train_bimamba,
        seed=args.seed,
        teacher_metrics=Path(args.teacher_metrics) if args.teacher_metrics else None,
        teacher_checkpoint=Path(args.teacher_checkpoint) if args.teacher_checkpoint else None,
    )

    ekf = NCLTEKF()
    ekf_metrics = evaluate_ekf_sessions(ekf, test_sessions)

    kn_metrics_path = kalmannet_dir / "nclt_tbptt_metrics.json"
    if not kn_metrics_path.exists():
        raise FileNotFoundError(f"Missing KalmanNet metrics at {kn_metrics_path}")
    kn_payload = json.loads(kn_metrics_path.read_text())
    kn_model = SimpleNCLTKalmanNet(hidden_dim=int(kn_payload["config"]["hidden_dim"])).to(device)
    kn_model.load_state_dict(torch.load(kn_payload["artifacts"]["checkpoint"], map_location=device))
    kn_eval = evaluate_kalmannet_sessions(kn_model, test_sessions, device)

    pure_model = PureBiMambaFilterSmoother(
        filter_embed_dim=int(pure_payload["config"]["filter_embed_dim"]),
        filter_hidden_dim=int(pure_payload["config"]["filter_hidden_dim"]),
        filter_layers=int(pure_payload["config"]["filter_layers"]),
        smoother_embed_dim=int(pure_payload["config"]["smoother_embed_dim"]),
        smoother_hidden_dim=int(pure_payload["config"]["smoother_hidden_dim"]),
        smoother_layers=int(pure_payload["config"]["smoother_layers"]),
        d_state=int(pure_payload["config"]["d_state"]),
        d_conv=int(pure_payload["config"]["d_conv"]),
        expand=int(pure_payload["config"]["expand"]),
        dropout=float(pure_payload["config"]["dropout"]),
    ).to(device)
    pure_model.load_state_dict(torch.load(pure_payload["artifacts"]["checkpoint"], map_location=device))
    pure_eval = evaluate_pure_bimamba_sessions(pure_model, test_sessions, device)

    comparison_plot = output_dir / "nclt_benchmark_pure_trajectories.png"
    plot_comparison(comparison_plot, test_sessions, ekf_metrics, kn_eval, pure_eval)

    benchmark_payload = {
        "seed": args.seed,
        "ekf": {
            "overall_rmse_m": ekf_metrics["overall_rmse_m"],
            "per_trajectory": ekf_metrics["per_trajectory"],
        },
        "kalmannet": {
            "overall_rmse_m": kn_eval["overall_rmse_m"],
            "per_trajectory": kn_eval["per_trajectory"],
            "source_metrics": str(kn_metrics_path),
        },
        "pure_bimamba": {
            "overall_rmse_m": pure_eval["overall_rmse_m"],
            "per_trajectory": pure_eval["per_trajectory"],
            "source_metrics": str(bimamba_dir / "nclt_bimamba_pure_metrics.json"),
            "config": pure_payload["config"],
        },
        "artifacts": {
            "comparison_plot": str(comparison_plot),
        },
    }
    metrics_path = output_dir / "nclt_benchmark_pure_metrics.json"
    metrics_path.write_text(json.dumps(benchmark_payload, indent=2))

    report_lines = [
        "# NCLT EKF vs KalmanNet vs Pure BiMamba",
        "",
        "## Overall RMSE",
        f"- EKF: {ekf_metrics['overall_rmse_m']:.3f} m",
        f"- KalmanNet: {kn_eval['overall_rmse_m']:.3f} m",
        f"- Pure BiMamba: {pure_eval['overall_rmse_m']:.3f} m",
    ]
    (output_dir / "nclt_benchmark_pure_report.md").write_text("\n".join(report_lines) + "\n")

    print("NCLT pure benchmark summary")
    print(f"EKF RMSE: {ekf_metrics['overall_rmse_m']:.3f} m")
    print(f"KalmanNet RMSE: {kn_eval['overall_rmse_m']:.3f} m")
    print(f"Pure BiMamba RMSE: {pure_eval['overall_rmse_m']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
