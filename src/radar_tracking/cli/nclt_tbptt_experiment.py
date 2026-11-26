#!/usr/bin/env python3

"""Standalone NCLT TBPTT experiment for wheel/GPS KalmanNet-style fusion.

This script is intentionally lightweight so it can run inside the current
project environment without the heavier Lightning/MMEngine dependency stack used
by the external reference implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pt(path: Path):
    return torch.load(path, map_location="cpu")


def derive_state_targets(item: dict) -> torch.Tensor:
    pos = item["ground_truth"].float()
    theta = item["theta_gt"].float().squeeze(-1)
    seq_len = pos.shape[0]

    vx = torch.zeros(seq_len, dtype=torch.float32)
    vy = torch.zeros(seq_len, dtype=torch.float32)
    omega = torch.zeros(seq_len, dtype=torch.float32)

    if seq_len > 1:
        deltas = pos[1:] - pos[:-1]
        vx[1:] = deltas[:, 0]
        vy[1:] = deltas[:, 1]
        theta_delta = wrap_to_pi(theta[1:] - theta[:-1])
        omega[1:] = theta_delta

    vx[0] = float(item["initial_state"][2])
    vy[0] = float(item["initial_state"][3])
    omega[0] = float(item["initial_state"][5])

    state = torch.stack([pos[:, 0], pos[:, 1], vx, vy, theta, omega], dim=1)
    return state


class NCLTWindowDataset(Dataset):
    def __init__(self, sessions: list[dict], seq_len: int = 50, stride: int = 20):
        self.windows = []
        for item in sessions:
            targets = derive_state_targets(item)
            wheel = item["wheel"].float()
            gps = item["filtered_gps"].float()
            initial = item["initial_state"].float()
            total_len = targets.shape[0]
            if total_len < seq_len:
                continue
            for start in range(0, total_len - seq_len + 1, stride):
                init_state = initial if start == 0 else targets[start - 1]
                self.windows.append(
                    {
                        "wheel": wheel[start:start + seq_len],
                        "filtered_gps": gps[start:start + seq_len],
                        "targets": targets[start:start + seq_len],
                        "initial_state": init_state.clone(),
                    }
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return self.windows[idx]


def collate_windows(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "wheel": torch.stack([item["wheel"] for item in batch], dim=0),
        "filtered_gps": torch.stack([item["filtered_gps"] for item in batch], dim=0),
        "targets": torch.stack([item["targets"] for item in batch], dim=0),
        "initial_state": torch.stack([item["initial_state"] for item in batch], dim=0),
    }


class SimpleNCLTKalmanNet(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        input_dim = 6 + 6 + 4 + 2 + 1
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()

    @staticmethod
    def transition(state: torch.Tensor, wheel_t: torch.Tensor) -> torch.Tensor:
        vl = wheel_t[:, 0]
        vr = wheel_t[:, 1]
        heading = wheel_t[:, 2]
        omega = wheel_t[:, 3]
        v_c = 0.5 * (vl + vr)

        pred = state.clone()
        pred[:, 0] = state[:, 0] + v_c * torch.cos(heading)
        pred[:, 1] = state[:, 1] + v_c * torch.sin(heading)
        pred[:, 2] = v_c * torch.cos(heading)
        pred[:, 3] = v_c * torch.sin(heading)
        pred[:, 4] = wrap_to_pi(heading)
        pred[:, 5] = omega
        return pred

    def forward_step(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
        wheel_t: torch.Tensor,
        gps_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.transition(state, wheel_t)
        gps_available = (~torch.isnan(gps_t).any(dim=1, keepdim=True)).float()
        gps_filled = torch.nan_to_num(gps_t, nan=0.0)
        innovation = (gps_filled - pred[:, :2]) * gps_available

        features = torch.cat([pred, state, wheel_t, innovation, gps_available], dim=1)
        hidden = self.gru(features, hidden)
        raw_gain = self.head(hidden)

        pos_gain = torch.sigmoid(raw_gain[:, :2])
        vel_gain = 0.25 * torch.tanh(raw_gain[:, 2:])

        correction = torch.zeros_like(pred)
        correction[:, 0] = pos_gain[:, 0] * innovation[:, 0]
        correction[:, 1] = pos_gain[:, 1] * innovation[:, 1]
        correction[:, 2] = vel_gain[:, 0] * innovation[:, 0]
        correction[:, 3] = vel_gain[:, 1] * innovation[:, 1]
        correction = correction * gps_available
        post = pred + correction
        post[:, 4] = wrap_to_pi(post[:, 4])
        return post, hidden, pred, innovation


@dataclass
class ExperimentConfig:
    seed: int = 7
    seq_len: int = 50
    stride: int = 20
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    detach_step: int = 5
    update_every: int = 5
    grad_clip: float = 1.0
    patience: int = 4
    training_label: str = "tbptt"


def weighted_state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = pred.new_tensor([1.0, 1.0, 0.05, 0.05, 0.0, 0.0]).view(1, 1, 6)
    delta = pred - target
    delta[..., 4] = wrap_to_pi(delta[..., 4])
    return ((delta ** 2) * weights).mean()


def sequence_rmse_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> float:
    return float(torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=1).mean()).item())


def evaluate_sessions(model: SimpleNCLTKalmanNet, sessions: list[dict], device: torch.device) -> dict:
    model.eval()
    results = []
    gps_rmses = []
    wheel_rmses = []
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for item in sessions:
            wheel = item["wheel"].float().to(device)
            filtered_gps = item["filtered_gps"].float().to(device)
            raw_gps = item["gps"].float()
            gt_xy = item["ground_truth"].float()
            initial_state = item["initial_state"].float().to(device).unsqueeze(0)

            hidden = torch.zeros(1, model.gru.hidden_size, device=device)
            state = initial_state

            preds_xy = []
            wheel_only_state = initial_state.clone().squeeze(0)
            wheel_only_xy = []

            for t in range(wheel.shape[0]):
                state, hidden, _, _ = model.forward_step(state, hidden, wheel[t:t + 1], filtered_gps[t:t + 1])
                preds_xy.append(state[0, :2].detach().cpu())

                wheel_only_state = model.transition(wheel_only_state.unsqueeze(0), wheel[t:t + 1]).squeeze(0)
                wheel_only_xy.append(wheel_only_state[:2].detach().cpu())

            pred_xy = torch.stack(preds_xy)
            wheel_xy = torch.stack(wheel_only_xy)

            model_rmse = sequence_rmse_xy(pred_xy, gt_xy)
            gps_rmse = sequence_rmse_xy(raw_gps, gt_xy)
            wheel_rmse = sequence_rmse_xy(wheel_xy, gt_xy)

            results.append(
                {
                    "data_date": item["data_date"],
                    "kalmannet_tbptt_rmse_m": model_rmse,
                    "gps_only_rmse_m": gps_rmse,
                    "wheel_only_rmse_m": wheel_rmse,
                    "trajectory_length": int(gt_xy.shape[0]),
                }
            )

            gps_rmses.append(gps_rmse)
            wheel_rmses.append(wheel_rmse)
            all_pred.append(pred_xy)
            all_gt.append(gt_xy)

    cat_pred = torch.cat(all_pred, dim=0)
    cat_gt = torch.cat(all_gt, dim=0)
    overall_rmse = sequence_rmse_xy(cat_pred, cat_gt)

    return {
        "per_trajectory": results,
        "overall_rmse_m": overall_rmse,
        "avg_gps_rmse_m": float(sum(gps_rmses) / len(gps_rmses)),
        "avg_wheel_rmse_m": float(sum(wheel_rmses) / len(wheel_rmses)),
        "predictions": all_pred,
        "targets": all_gt,
    }


def train_model(
    model: SimpleNCLTKalmanNet,
    train_loader: DataLoader,
    val_sessions: list[dict],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[SimpleNCLTKalmanNet, dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = None
    best_val = float("inf")
    history = []
    no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            wheel = batch["wheel"].to(device)
            gps = batch["filtered_gps"].to(device)
            targets = batch["targets"].to(device)
            state = batch["initial_state"].to(device)
            hidden = torch.zeros(state.shape[0], model.gru.hidden_size, device=device)
            optimizer.zero_grad()

            accumulated = 0.0
            steps_in_chunk = 0
            batch_loss = 0.0
            preds = []

            for t in range(targets.shape[1]):
                state, hidden, _, _ = model.forward_step(state, hidden, wheel[:, t, :], gps[:, t, :])
                preds.append(state)
                step_loss = weighted_state_loss(state.unsqueeze(1), targets[:, t:t + 1, :])
                accumulated = accumulated + step_loss
                steps_in_chunk += 1

                if (t + 1) % config.update_every == 0 or (t + 1) == targets.shape[1]:
                    loss = accumulated / max(steps_in_chunk, 1)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss += float(loss.item())
                    accumulated = 0.0
                    steps_in_chunk = 0

                if (t + 1) % config.detach_step == 0:
                    state = state.detach()
                    hidden = hidden.detach()

            epoch_loss += batch_loss
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        val_metrics = evaluate_sessions(model, val_sessions, device)
        val_rmse = val_metrics["overall_rmse_m"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_rmse_m": val_rmse})
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f} val_rmse={val_rmse:.4f} m")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"Early stopping after epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_rmse_m": best_val, "history": history}


def make_plot(test_metrics: dict, output_path: Path) -> None:
    preds = test_metrics["predictions"]
    targets = test_metrics["targets"]
    num = min(3, len(preds))
    fig, axes = plt.subplots(1, num, figsize=(6 * num, 5))
    if num == 1:
        axes = [axes]
    for ax, pred_xy, gt_xy, result in zip(axes, preds[:num], targets[:num], test_metrics["per_trajectory"][:num]):
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], label="Ground truth", linewidth=2)
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], label="TBPTT KalmanNet", linewidth=1.5)
        ax.set_title(f"{result['data_date']} RMSE={result['kalmannet_tbptt_rmse_m']:.2f} m")
        ax.set_xlabel("X / North (m)")
        ax.set_ylabel("Y / East (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    config: ExperimentConfig,
    train_summary: dict,
    test_metrics: dict,
    checkpoint_path: Path,
    plot_path: Path,
) -> None:
    metrics_path = output_dir / "nclt_tbptt_metrics.json"
    report_path = output_dir / "nclt_tbptt_report.md"

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

    lines = [
        f"# NCLT {config.training_label.upper()} KalmanNet Result",
        "",
        "## Summary",
        f"- Training label: `{config.training_label}`",
        f"- Sequence window D: `{config.seq_len}`",
        f"- Backprop/update interval k: `{config.update_every}`",
        f"- Hidden-state detach interval w: `{config.detach_step}`",
        f"- Overall {config.training_label.upper()} KalmanNet RMSE: {test_metrics['overall_rmse_m']:.3f} m",
        f"- Average GPS-only RMSE: {test_metrics['avg_gps_rmse_m']:.3f} m",
        f"- Average wheel-only RMSE: {test_metrics['avg_wheel_rmse_m']:.3f} m",
        f"- Best validation RMSE: {train_summary['best_val_rmse_m']:.3f} m",
        "",
        "## Test Trajectories",
    ]
    for item in test_metrics["per_trajectory"]:
        lines.append(
            f"- {item['data_date']}: TBPTT KalmanNet {item['kalmannet_tbptt_rmse_m']:.3f} m, "
            f"GPS-only {item['gps_only_rmse_m']:.3f} m, wheel-only {item['wheel_only_rmse_m']:.3f} m"
        )
    lines.extend(
        [
            "",
            "## Files",
            f"- Metrics JSON: `{metrics_path}`",
            f"- Report: `{report_path}`",
            f"- Checkpoint: `{checkpoint_path}`",
            f"- Trajectory plot: `{plot_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight NCLT TBPTT KalmanNet experiment.")
    parser.add_argument("--data-dir", default="./data/nclt_fusion", help="Directory containing train.pt/val.pt/test.pt.")
    parser.add_argument("--output-dir", default="./nclt_tbptt_outputs", help="Directory for outputs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--detach-step", type=int, default=5, help="Truncation length before hidden/state detach.")
    parser.add_argument("--update-every", type=int, default=5, help="Steps accumulated before backward/update.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--training-label", default="tbptt", help="Label used in saved metrics/report, e.g. tbptt or full_bptt.")
    args = parser.parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        detach_step=args.detach_step,
        update_every=args.update_every,
        grad_clip=args.grad_clip,
        patience=args.patience,
        training_label=args.training_label,
    )

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sessions = load_pt(data_dir / "train.pt")
    val_sessions = load_pt(data_dir / "val.pt")
    test_sessions = load_pt(data_dir / "test.pt")

    train_dataset = NCLTWindowDataset(train_sessions, seq_len=config.seq_len, stride=config.stride)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
    )

    model = SimpleNCLTKalmanNet(hidden_dim=config.hidden_dim).to(device)
    model, train_summary = train_model(model, train_loader, val_sessions, config, device)

    checkpoint_path = output_dir / "best_nclt_tbptt_kalmannet.pt"
    torch.save(model.state_dict(), checkpoint_path)

    test_metrics = evaluate_sessions(model, test_sessions, device)
    plot_path = output_dir / "nclt_tbptt_trajectories.png"
    make_plot(test_metrics, plot_path)
    write_report(output_dir, config, train_summary, test_metrics, checkpoint_path, plot_path)

    print("\nTest summary")
    print(f"Overall RMSE: {test_metrics['overall_rmse_m']:.3f} m")
    print(f"Average GPS-only RMSE: {test_metrics['avg_gps_rmse_m']:.3f} m")
    print(f"Average wheel-only RMSE: {test_metrics['avg_wheel_rmse_m']:.3f} m")
    for item in test_metrics["per_trajectory"]:
        print(
            f"{item['data_date']}: model={item['kalmannet_tbptt_rmse_m']:.3f} m "
            f"gps={item['gps_only_rmse_m']:.3f} m wheel={item['wheel_only_rmse_m']:.3f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
