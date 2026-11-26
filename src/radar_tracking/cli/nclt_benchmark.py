#!/usr/bin/env python3

"""Unified NCLT benchmark for EKF, TBPTT KalmanNet, and BiMamba."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from radar_tracking.cli.nclt_bimamba_experiment import (
    ExperimentConfig as BiMambaConfig,
    NCLTFusionBiMamba,
    NCLTWindowDataset,
    collate_windows,
    evaluate_sessions as evaluate_bimamba_sessions,
    load_pt,
    make_plot as make_bimamba_plot,
    set_seed,
    train_model as train_bimamba_model,
)
from radar_tracking.cli.nclt_tbptt_experiment import (
    ExperimentConfig as KalmanNetConfig,
    NCLTWindowDataset as KalmanNetWindowDataset,
    SimpleNCLTKalmanNet,
    collate_windows as collate_kalmannet_windows,
    evaluate_sessions as evaluate_kalmannet_sessions,
    train_model as train_kalmannet_model,
)


def split_path(data_dir: Path, split: str, use_rtk: bool) -> Path:
    suffix = "_rtk" if use_rtk else ""
    path = data_dir / f"{split}{suffix}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    return path


def sequence_rmse_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> float:
    return float(torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=1).mean()).item())


class NCLTEKF:
    """Simple wheel-prediction + GPS-update EKF for NCLT fusion."""

    def __init__(self, process_diag=None, gps_var: float = 25.0):
        if process_diag is None:
            process_diag = [4.0, 4.0, 2.0, 2.0, 0.2, 0.2]
        self.Q = torch.diag(torch.tensor(process_diag, dtype=torch.float32))
        self.R = torch.eye(2, dtype=torch.float32) * gps_var
        self.H = torch.zeros(2, 6, dtype=torch.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

    @staticmethod
    def transition(state: torch.Tensor, wheel_t: torch.Tensor) -> torch.Tensor:
        vl = wheel_t[0]
        vr = wheel_t[1]
        heading = wheel_t[2]
        omega = wheel_t[3]
        v_c = 0.5 * (vl + vr)

        pred = state.clone()
        pred[0] = state[0] + v_c * torch.cos(heading)
        pred[1] = state[1] + v_c * torch.sin(heading)
        pred[2] = v_c * torch.cos(heading)
        pred[3] = v_c * torch.sin(heading)
        pred[4] = heading
        pred[5] = omega
        return pred

    @staticmethod
    def transition_jacobian() -> torch.Tensor:
        F = torch.zeros(6, 6, dtype=torch.float32)
        F[0, 0] = 1.0
        F[1, 1] = 1.0
        return F

    def run_session(self, item: dict) -> torch.Tensor:
        wheel = item["wheel"].float()
        filtered_gps = item["filtered_gps"].float()
        state = item["initial_state"].float().clone()
        cov = torch.eye(6, dtype=torch.float32) * 25.0
        identity = torch.eye(6, dtype=torch.float32)
        preds = []

        for t in range(wheel.shape[0]):
            state = self.transition(state, wheel[t])
            F = self.transition_jacobian()
            cov = F @ cov @ F.T + self.Q

            gps_t = filtered_gps[t]
            if not torch.isnan(gps_t).any():
                innovation = gps_t - (self.H @ state)
                S = self.H @ cov @ self.H.T + self.R
                K = cov @ self.H.T @ torch.linalg.inv(S)
                state = state + K @ innovation
                cov = (identity - K @ self.H) @ cov

            preds.append(state[:2].clone())

        return torch.stack(preds, dim=0)


def evaluate_ekf_sessions(ekf: NCLTEKF, sessions: list[dict]) -> dict:
    results = []
    all_pred = []
    all_gt = []

    for item in sessions:
        pred_xy = ekf.run_session(item)
        gt_xy = item["ground_truth"].float()
        rmse = sequence_rmse_xy(pred_xy, gt_xy)
        results.append(
            {
                "data_date": item["data_date"],
                "ekf_rmse_m": rmse,
                "trajectory_length": int(gt_xy.shape[0]),
            }
        )
        all_pred.append(pred_xy)
        all_gt.append(gt_xy)

    cat_pred = torch.cat(all_pred, dim=0)
    cat_gt = torch.cat(all_gt, dim=0)
    overall_rmse = sequence_rmse_xy(cat_pred, cat_gt)

    return {
        "per_trajectory": results,
        "overall_rmse_m": overall_rmse,
        "predictions": all_pred,
        "targets": all_gt,
    }


def maybe_train_bimamba(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    force_train: bool,
    seed: int,
    kalmannet_checkpoint: Path,
    kalmannet_hidden_dim: int,
    use_rtk: bool,
) -> tuple[dict, Path]:
    metrics_path = output_dir / "nclt_bimamba_metrics.json"
    checkpoint_path = output_dir / "best_nclt_bimamba.pt"
    if metrics_path.exists() and checkpoint_path.exists() and not force_train:
        return json.loads(metrics_path.read_text()), checkpoint_path

    config = BiMambaConfig(
        seed=seed,
        seq_len=60,
        stride=30,
        batch_size=16,
        epochs=12,
        lr=5e-5,
        weight_decay=1e-5,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3,
        d_state=32,
        d_conv=4,
        expand=2,
        dropout=0.08,
        grad_clip=1.0,
        patience=4,
        encoder_direction="forward",
        kalmannet_hidden_dim=kalmannet_hidden_dim,
        freeze_filter=True,
    )

    train_sessions = load_pt(split_path(data_dir, "train", use_rtk))
    val_sessions = load_pt(split_path(data_dir, "val", use_rtk))
    test_sessions = load_pt(split_path(data_dir, "test", use_rtk))
    train_dataset = NCLTWindowDataset(train_sessions, seq_len=config.seq_len, stride=config.stride)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
    )

    model = NCLTFusionBiMamba(
        kalmannet_hidden_dim=config.kalmannet_hidden_dim,
        freeze_filter=config.freeze_filter,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        dropout=config.dropout,
        encoder_direction=config.encoder_direction,
    ).to(device)
    model.load_filter_checkpoint(kalmannet_checkpoint)
    model, train_summary = train_bimamba_model(model, train_loader, val_sessions, config, device)
    torch.save(model.state_dict(), checkpoint_path)
    test_metrics = evaluate_bimamba_sessions(model, test_sessions, device)

    plot_path = output_dir / "nclt_bimamba_trajectories.png"
    make_bimamba_plot(test_metrics, plot_path)

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
            "filter_checkpoint_source": str(kalmannet_checkpoint),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    return payload, checkpoint_path


def maybe_train_kalmannet(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    force_train: bool,
    seed: int,
    use_rtk: bool,
) -> tuple[dict, Path]:
    metrics_path = output_dir / "nclt_tbptt_metrics.json"
    checkpoint_path = output_dir / "best_nclt_tbptt_kalmannet.pt"
    if metrics_path.exists() and checkpoint_path.exists() and not force_train:
        return json.loads(metrics_path.read_text()), checkpoint_path

    config = KalmanNetConfig(
        seed=seed,
        seq_len=50,
        stride=40,
        batch_size=32,
        epochs=8,
        lr=1e-3,
        weight_decay=1e-5,
        hidden_dim=128,
        detach_step=5,
        update_every=5,
        grad_clip=1.0,
        patience=4,
    )

    train_sessions = load_pt(split_path(data_dir, "train", use_rtk))
    val_sessions = load_pt(split_path(data_dir, "val", use_rtk))
    test_sessions = load_pt(split_path(data_dir, "test", use_rtk))
    train_dataset = KalmanNetWindowDataset(train_sessions, seq_len=config.seq_len, stride=config.stride)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_kalmannet_windows,
    )

    model = SimpleNCLTKalmanNet(hidden_dim=config.hidden_dim).to(device)
    model, train_summary = train_kalmannet_model(model, train_loader, val_sessions, config, device)
    torch.save(model.state_dict(), checkpoint_path)
    test_metrics = evaluate_kalmannet_sessions(model, test_sessions, device)

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
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    return payload, checkpoint_path


def plot_comparison(
    output_path: Path,
    test_sessions: list[dict],
    ekf_metrics: dict,
    kalman_metrics: dict,
    bimamba_metrics: dict,
) -> None:
    num = min(3, len(test_sessions))
    fig, axes = plt.subplots(1, num, figsize=(6 * num, 5))
    if num == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        gt = test_sessions[idx]["ground_truth"].float()
        ekf = ekf_metrics["predictions"][idx]
        bimamba = bimamba_metrics["predictions"][idx]
        ax.plot(gt[:, 0], gt[:, 1], label="Ground truth", linewidth=2)
        ax.plot(ekf[:, 0], ekf[:, 1], label="EKF", linewidth=1.2)
        ax.plot(bimamba[:, 0], bimamba[:, 1], label="BiMamba", linewidth=1.2)
        if "predictions" in kalman_metrics:
            kn = kalman_metrics["predictions"][idx]
            ax.plot(kn[:, 0], kn[:, 1], label="KalmanNet", linewidth=1.2)
        ax.set_title(test_sessions[idx]["data_date"])
        ax.set_xlabel("X / North (m)")
        ax.set_ylabel("Y / East (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    benchmark_payload: dict,
) -> None:
    report_path = output_dir / "nclt_benchmark_report.md"
    ekf = benchmark_payload["ekf"]
    kn = benchmark_payload["kalmannet"]
    bm = benchmark_payload["bimamba"]

    lines = [
        "# NCLT EKF vs KalmanNet vs BiMamba",
        "",
        "## Overall RMSE",
        f"- EKF: {ekf['overall_rmse_m']:.3f} m",
        f"- KalmanNet: {kn['overall_rmse_m']:.3f} m",
        f"- BiMamba: {bm['overall_rmse_m']:.3f} m",
        "",
        "## Per Trajectory",
    ]
    for ekf_item, kn_item, bm_item in zip(ekf["per_trajectory"], kn["per_trajectory"], bm["per_trajectory"]):
        lines.append(
            f"- {bm_item['data_date']}: EKF {ekf_item['ekf_rmse_m']:.3f} m, "
            f"KalmanNet {kn_item['kalmannet_tbptt_rmse_m']:.3f} m, "
            f"BiMamba {bm_item['bimamba_rmse_m']:.3f} m"
        )
    lines.extend(
        [
            "",
            "## Files",
            f"- Metrics JSON: `{output_dir / 'nclt_benchmark_metrics.json'}`",
            f"- Report: `{report_path}`",
            f"- Comparison plot: `{output_dir / 'nclt_benchmark_trajectories.png'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified NCLT EKF/KalmanNet/BiMamba benchmark.")
    parser.add_argument("--data-dir", default="./data/nclt_fusion")
    parser.add_argument("--output-dir", default="./nclt_benchmark_outputs")
    parser.add_argument("--bimamba-dir", default="./nclt_bimamba_outputs_big")
    parser.add_argument("--kalmannet-dir", default="./nclt_tbptt_outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use-rtk", action="store_true")
    parser.add_argument("--force-train-bimamba", action="store_true")
    parser.add_argument("--force-train-kalmannet", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("nclt_benchmark.py requires CUDA for the BiMamba portion of the benchmark.")

    set_seed(args.seed)
    device = torch.device("cuda")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bimamba_dir = Path(args.bimamba_dir)
    bimamba_dir.mkdir(parents=True, exist_ok=True)
    kalmannet_dir = Path(args.kalmannet_dir)
    kalmannet_dir.mkdir(parents=True, exist_ok=True)

    test_sessions = load_pt(split_path(data_dir, "test", args.use_rtk))

    kalmannet_payload, _ = maybe_train_kalmannet(
        data_dir=data_dir,
        output_dir=kalmannet_dir,
        device=device,
        force_train=args.force_train_kalmannet,
        seed=args.seed,
        use_rtk=args.use_rtk,
    )
    kalmannet_checkpoint = Path(kalmannet_payload["artifacts"]["checkpoint"])
    kalmannet_hidden_dim = int(kalmannet_payload["config"]["hidden_dim"])
    bimamba_payload, _ = maybe_train_bimamba(
        data_dir=data_dir,
        output_dir=bimamba_dir,
        device=device,
        force_train=args.force_train_bimamba,
        seed=args.seed,
        kalmannet_checkpoint=kalmannet_checkpoint,
        kalmannet_hidden_dim=kalmannet_hidden_dim,
        use_rtk=args.use_rtk,
    )

    ekf = NCLTEKF()
    ekf_metrics = evaluate_ekf_sessions(ekf, test_sessions)

    bimamba_model = NCLTFusionBiMamba(
        kalmannet_hidden_dim=int(bimamba_payload["config"]["kalmannet_hidden_dim"]),
        freeze_filter=bool(bimamba_payload["config"]["freeze_filter"]),
        embed_dim=bimamba_payload["config"]["embed_dim"],
        hidden_dim=bimamba_payload["config"]["hidden_dim"],
        num_layers=bimamba_payload["config"]["num_layers"],
        d_state=bimamba_payload["config"]["d_state"],
        d_conv=bimamba_payload["config"]["d_conv"],
        expand=bimamba_payload["config"]["expand"],
        dropout=bimamba_payload["config"]["dropout"],
        encoder_direction=bimamba_payload["config"]["encoder_direction"],
    ).to(device)
    bimamba_model.load_state_dict(torch.load(bimamba_payload["artifacts"]["checkpoint"], map_location=device))
    bimamba_eval = evaluate_bimamba_sessions(bimamba_model, test_sessions, device)

    kalmannet_model = SimpleNCLTKalmanNet(
        hidden_dim=kalmannet_payload["config"]["hidden_dim"]
    ).to(device)
    kalmannet_model.load_state_dict(torch.load(kalmannet_payload["artifacts"]["checkpoint"], map_location=device))
    kalmannet_eval = evaluate_kalmannet_sessions(kalmannet_model, test_sessions, device)

    comparison_plot = output_dir / "nclt_benchmark_trajectories.png"
    plot_comparison(comparison_plot, test_sessions, ekf_metrics, kalmannet_eval, bimamba_eval)

    benchmark_payload = {
        "seed": args.seed,
        "use_rtk": args.use_rtk,
        "ekf": {
            "overall_rmse_m": ekf_metrics["overall_rmse_m"],
            "per_trajectory": ekf_metrics["per_trajectory"],
        },
        "kalmannet": {
            "overall_rmse_m": kalmannet_eval["overall_rmse_m"],
            "per_trajectory": kalmannet_eval["per_trajectory"],
            "source_metrics": str(kalmannet_dir / "nclt_tbptt_metrics.json"),
        },
        "bimamba": {
            "overall_rmse_m": bimamba_eval["overall_rmse_m"],
            "per_trajectory": bimamba_eval["per_trajectory"],
            "source_metrics": str(bimamba_dir / "nclt_bimamba_metrics.json"),
            "config": bimamba_payload["config"],
        },
        "artifacts": {
            "comparison_plot": str(comparison_plot),
        },
    }
    metrics_path = output_dir / "nclt_benchmark_metrics.json"
    metrics_path.write_text(json.dumps(benchmark_payload, indent=2))
    write_report(output_dir, benchmark_payload)

    print("NCLT benchmark summary")
    print(f"EKF RMSE: {ekf_metrics['overall_rmse_m']:.3f} m")
    print(f"KalmanNet RMSE: {kalmannet_eval['overall_rmse_m']:.3f} m")
    print(f"BiMamba RMSE: {bimamba_eval['overall_rmse_m']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
