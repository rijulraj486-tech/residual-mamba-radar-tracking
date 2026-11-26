#!/usr/bin/env python3

"""Dedicated BiMamba experiment for NCLT wheel/GPS fusion data.

This path is intentionally separate from the active single-target radar
BiMambaTracker model. The radar model expects range/bearing observations and a
radar-specific coarse-state construction, while this script adapts BiMamba to
the processed NCLT sensor-fusion tensors produced by ``nclt_preprocess.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from radar_tracking.models.bimamba_encoder import BiMambaEncoder  # noqa: E402
from radar_tracking.cli.nclt_tbptt_experiment import SimpleNCLTKalmanNet  # noqa: E402


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
        omega[1:] = wrap_to_pi(theta[1:] - theta[:-1])

    vx[0] = float(item["initial_state"][2])
    vy[0] = float(item["initial_state"][3])
    omega[0] = float(item["initial_state"][5])

    return torch.stack([pos[:, 0], pos[:, 1], vx, vy, theta, omega], dim=1)


class NCLTWindowDataset(Dataset):
    def __init__(self, sessions: list[dict], seq_len: int = 50, stride: int = 20):
        self.windows = []
        for item in sessions:
            targets = derive_state_targets(item)
            wheel = item["wheel"].float()
            gps = item["filtered_gps"].float()
            total_len = targets.shape[0]
            if total_len < seq_len:
                continue
            for start in range(0, total_len - seq_len + 1, stride):
                init_state = item["initial_state"].float() if start == 0 else targets[start - 1]
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


class ResidualSmootherHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        pos_residual_scale: float = 8.0,
        vel_residual_scale: float = 0.8,
        heading_residual_scale: float = 0.04,
        omega_residual_scale: float = 0.04,
    ):
        super().__init__()
        self.pos_residual_scale = pos_residual_scale
        self.vel_residual_scale = vel_residual_scale
        self.heading_residual_scale = heading_residual_scale
        self.omega_residual_scale = omega_residual_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 8),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(
        self,
        encoded: torch.Tensor,
        filtered: torch.Tensor,
        post_innovation: torch.Tensor,
        gps_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(encoded)
        residual = torch.zeros_like(filtered)
        gps_mask = gps_available.expand_as(post_innovation)
        pos_gain = 0.30 * torch.tanh(raw[..., 0:2])
        vel_gain = 0.12 * torch.tanh(raw[..., 2:4])
        pos_bias = 1.5 * torch.tanh(raw[..., 4:6])
        residual[..., 0:2] = pos_gain * post_innovation * gps_mask + pos_bias
        residual[..., 2:4] = vel_gain * post_innovation * gps_mask
        residual[..., 4] = self.heading_residual_scale * torch.tanh(raw[..., 6])
        residual[..., 5] = self.omega_residual_scale * torch.tanh(raw[..., 7])
        refined = filtered + residual
        refined[..., 4] = wrap_to_pi(refined[..., 4])
        return refined, residual


class NCLTFusionBiMamba(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,
        kalmannet_hidden_dim: int = 128,
        freeze_filter: bool = True,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.08,
        encoder_direction: str = "bidirectional",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.kalmannet_hidden_dim = kalmannet_hidden_dim
        self.freeze_filter = freeze_filter
        self.pos_scale = 200.0
        self.vel_scale = 5.0
        self.omega_scale = 0.5
        self.token_dim = 27
        self.filter_model = SimpleNCLTKalmanNet(hidden_dim=kalmannet_hidden_dim)
        if self.freeze_filter:
            for param in self.filter_model.parameters():
                param.requires_grad = False
        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.encoder = BiMambaEncoder(
            dim=embed_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            encoder_direction=encoder_direction,
        )
        self.output_norm = nn.LayerNorm(self.encoder.output_dim)
        self.head = ResidualSmootherHead(
            input_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
        )
        self.temporal_kernel = 21
        self.temporal_refiner = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=self.temporal_kernel,
            padding=0,
            groups=2,
            bias=False,
        )
        with torch.no_grad():
            self.temporal_refiner.weight.fill_(1.0 / float(self.temporal_kernel))
        self.temporal_blend_logit = nn.Parameter(torch.tensor(2.2))

    def load_filter_checkpoint(self, checkpoint_path: str | Path) -> None:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        self.filter_model.load_state_dict(state_dict)
        if self.freeze_filter:
            self.filter_model.eval()

    @staticmethod
    def wheel_transition(state: torch.Tensor, wheel_t: torch.Tensor) -> torch.Tensor:
        vl = wheel_t[..., 0]
        vr = wheel_t[..., 1]
        heading = wheel_t[..., 2]
        omega = wheel_t[..., 3]
        v_c = 0.5 * (vl + vr)

        pred = state.clone()
        pred[..., 0] = state[..., 0] + v_c * torch.cos(heading)
        pred[..., 1] = state[..., 1] + v_c * torch.sin(heading)
        pred[..., 2] = v_c * torch.cos(heading)
        pred[..., 3] = v_c * torch.sin(heading)
        pred[..., 4] = wrap_to_pi(heading)
        pred[..., 5] = omega
        return pred

    def build_wheel_only_sequence(self, wheel: torch.Tensor, init_state: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = wheel.shape
        coarse = torch.zeros(batch_size, seq_len, self.state_dim, device=wheel.device, dtype=wheel.dtype)
        prev = init_state
        for t in range(seq_len):
            prev = self.wheel_transition(prev, wheel[:, t, :])
            coarse[:, t, :] = prev
        return coarse

    def run_kalmannet_filter(
        self,
        wheel: torch.Tensor,
        filtered_gps: torch.Tensor,
        init_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = wheel.shape
        state = init_state
        hidden = torch.zeros(batch_size, self.filter_model.gru.hidden_size, device=wheel.device, dtype=wheel.dtype)
        filtered_states = []
        predicted_states = []
        innovations = []
        gps_available_steps = []

        context = torch.no_grad() if self.freeze_filter else nullcontext()
        with context:
            for t in range(seq_len):
                gps_t = filtered_gps[:, t, :]
                gps_available_t = (~torch.isnan(gps_t).any(dim=1, keepdim=True)).float()
                state, hidden, pred, innovation = self.filter_model.forward_step(
                    state,
                    hidden,
                    wheel[:, t, :],
                    gps_t,
                )
                filtered_states.append(state)
                predicted_states.append(pred)
                innovations.append(innovation)
                gps_available_steps.append(gps_available_t)

        filtered_states = torch.stack(filtered_states, dim=1)
        predicted_states = torch.stack(predicted_states, dim=1)
        innovations = torch.stack(innovations, dim=1)
        gps_available = torch.stack(gps_available_steps, dim=1)
        return filtered_states, predicted_states, innovations, gps_available

    def build_tokens(
        self,
        wheel: torch.Tensor,
        filtered_gps: torch.Tensor,
        init_state: torch.Tensor,
        filtered_states: torch.Tensor,
        predicted_states: torch.Tensor,
        innovations: torch.Tensor,
        gps_available: torch.Tensor,
    ) -> torch.Tensor:
        gps_available = (~torch.isnan(filtered_gps).any(dim=-1, keepdim=True)).float()
        gps_filled = torch.nan_to_num(filtered_gps, nan=0.0)
        origin = init_state[:, None, :2]
        gps_rel = gps_filled - origin
        pred_rel = predicted_states[:, :, :2] - origin
        filtered_rel = filtered_states[:, :, :2] - origin
        post_innovation = (gps_filled - filtered_states[:, :, :2]) * gps_available
        delta_filtered_pos = torch.zeros_like(filtered_rel)
        delta_filtered_pos[:, 1:, :] = filtered_rel[:, 1:, :] - filtered_rel[:, :-1, :]
        update_delta = filtered_states[:, :, 0:4] - predicted_states[:, :, 0:4]

        tokens = torch.cat(
            [
                wheel[:, :, 0:2] / self.vel_scale,
                wheel[:, :, 2:3] / math.pi,
                wheel[:, :, 3:4] / self.omega_scale,
                gps_rel / self.pos_scale,
                gps_available,
                innovations / self.pos_scale,
                post_innovation / self.pos_scale,
                pred_rel / self.pos_scale,
                predicted_states[:, :, 2:4] / self.vel_scale,
                filtered_rel / self.pos_scale,
                filtered_states[:, :, 2:4] / self.vel_scale,
                filtered_states[:, :, 4:5] / math.pi,
                filtered_states[:, :, 5:6] / self.omega_scale,
                update_delta[:, :, 0:2] / self.pos_scale,
                update_delta[:, :, 2:4] / self.vel_scale,
                delta_filtered_pos / self.pos_scale,
            ],
            dim=-1,
        )
        if tokens.shape[-1] != self.token_dim:
            raise RuntimeError(
                f"NCLT fusion token has {tokens.shape[-1]} features but expected {self.token_dim}."
            )
        return tokens

    def forward(
        self,
        wheel: torch.Tensor,
        filtered_gps: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        filtered_states, predicted_states, innovations, gps_available = self.run_kalmannet_filter(
            wheel,
            filtered_gps,
            initial_state,
        )
        tokens = self.build_tokens(
            wheel,
            filtered_gps,
            initial_state,
            filtered_states,
            predicted_states,
            innovations,
            gps_available,
        )
        embedded = self.input_proj(tokens)
        encoded = self.encoder(embedded)
        encoded = self.output_norm(encoded)
        gps_filled = torch.nan_to_num(filtered_gps, nan=0.0)
        post_innovation = (gps_filled - filtered_states[:, :, :2]) * gps_available
        states, smoother_residual = self.head(encoded, filtered_states, post_innovation, gps_available)
        pos = states[:, :, :2].transpose(1, 2)
        pad = self.temporal_kernel // 2
        pos_padded = F.pad(pos, (pad, pad), mode="replicate")
        smoothed_pos = self.temporal_refiner(pos_padded).transpose(1, 2)
        blend = torch.sigmoid(self.temporal_blend_logit)
        fused_pos = (1.0 - blend) * states[:, :, :2] + blend * smoothed_pos
        states = torch.cat([fused_pos, states[:, :, 2:]], dim=-1)
        wheel_only_states = self.build_wheel_only_sequence(wheel, initial_state)
        return {
            "tokens": tokens,
            "coarse_state": predicted_states,
            "wheel_only_state": wheel_only_states,
            "filtered_states": filtered_states,
            "encoded_tokens": encoded,
            "state_estimates": states,
            "smoother_residual": smoother_residual,
        }


@dataclass
class ExperimentConfig:
    seed: int = 7
    seq_len: int = 60
    stride: int = 30
    batch_size: int = 16
    epochs: int = 12
    lr: float = 5e-5
    weight_decay: float = 1e-5
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.08
    grad_clip: float = 1.0
    patience: int = 4
    encoder_direction: str = "bidirectional"
    kalmannet_hidden_dim: int = 64
    freeze_filter: bool = True
    temporal_only_tuning: bool = True


def weighted_state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = F.smooth_l1_loss(pred[..., 0:2], target[..., 0:2], beta=1.0)
    vel = F.smooth_l1_loss(pred[..., 2:4], target[..., 2:4], beta=0.5)
    heading_delta = wrap_to_pi(pred[..., 4] - target[..., 4])
    heading = (heading_delta ** 2).mean()
    return pos + 0.2 * vel + 0.01 * heading


def sequence_rmse_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> float:
    return float(torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=1).mean()).item())


def evaluate_sessions(model: NCLTFusionBiMamba, sessions: list[dict], device: torch.device) -> dict:
    model.eval()
    results = []
    gps_rmses = []
    wheel_rmses = []
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for item in sessions:
            wheel = item["wheel"].float().to(device).unsqueeze(0)
            filtered_gps = item["filtered_gps"].float().to(device).unsqueeze(0)
            initial_state = item["initial_state"].float().to(device).unsqueeze(0)
            gt_xy = item["ground_truth"].float()

            output = model(wheel, filtered_gps, initial_state)
            pred_xy = output["state_estimates"][0, :, :2].detach().cpu()
            wheel_xy = output["wheel_only_state"][0, :, :2].detach().cpu()

            model_rmse = sequence_rmse_xy(pred_xy, gt_xy)
            gps_rmse = sequence_rmse_xy(item["gps"].float(), gt_xy)
            wheel_rmse = sequence_rmse_xy(wheel_xy, gt_xy)

            results.append(
                {
                    "data_date": item["data_date"],
                    "bimamba_rmse_m": model_rmse,
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
    model: NCLTFusionBiMamba,
    train_loader: DataLoader,
    val_sessions: list[dict],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[NCLTFusionBiMamba, dict]:
    if config.temporal_only_tuning:
        for name, param in model.named_parameters():
            if name.startswith("temporal_refiner.") or name == "temporal_blend_logit":
                continue
            param.requires_grad = False

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    history = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial_val_metrics = evaluate_sessions(model, val_sessions, device)
    best_val = initial_val_metrics["overall_rmse_m"]
    history.append({"epoch": 0, "train_loss": None, "val_rmse_m": best_val})
    no_improve = 0
    print(f"Epoch 0: train_loss=NA val_rmse={best_val:.4f} m")

    for epoch in range(config.epochs):
        model.train()
        if model.freeze_filter:
            model.filter_model.eval()
        epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            wheel = batch["wheel"].to(device)
            gps = batch["filtered_gps"].to(device)
            targets = batch["targets"].to(device)
            initial_state = batch["initial_state"].to(device)

            optimizer.zero_grad()
            output = model(wheel, gps, initial_state)
            estimates = output["state_estimates"]
            filtered = output["filtered_states"]
            residual = output["smoother_residual"]

            state_loss = weighted_state_loss(estimates, targets)
            estimate_err = ((estimates[..., :2] - targets[..., :2]) ** 2).sum(dim=-1)
            filtered_err = ((filtered[..., :2] - targets[..., :2]) ** 2).sum(dim=-1)
            # Encourage BiMamba to outperform the KalmanNet-filtered baseline.
            improvement_margin = 1e-3
            improvement_loss = torch.relu(estimate_err - filtered_err + improvement_margin).mean()
            residual_smoothness = (
                (residual[:, 1:, :2] - residual[:, :-1, :2]).pow(2).mean()
                if residual.shape[1] > 1
                else residual.new_tensor(0.0)
            )
            residual_penalty = residual.pow(2).mean()
            loss = (
                state_loss
                + 0.25 * improvement_loss
                + 1e-4 * residual_smoothness
                + 1e-5 * residual_penalty
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        val_metrics = evaluate_sessions(model, val_sessions, device)
        val_rmse = val_metrics["overall_rmse_m"]
        scheduler.step(val_rmse)
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
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], label="NCLT BiMamba", linewidth=1.5)
        ax.set_title(f"{result['data_date']} RMSE={result['bimamba_rmse_m']:.2f} m")
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
    metrics_path = output_dir / "nclt_bimamba_metrics.json"
    report_path = output_dir / "nclt_bimamba_report.md"

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
        "# NCLT BiMamba Result",
        "",
        "## Summary",
        f"- Overall BiMamba RMSE: {test_metrics['overall_rmse_m']:.3f} m",
        f"- Average GPS-only RMSE: {test_metrics['avg_gps_rmse_m']:.3f} m",
        f"- Average wheel-only RMSE: {test_metrics['avg_wheel_rmse_m']:.3f} m",
        f"- Best validation RMSE: {train_summary['best_val_rmse_m']:.3f} m",
        "",
        "## Test Trajectories",
    ]
    for item in test_metrics["per_trajectory"]:
        lines.append(
            f"- {item['data_date']}: BiMamba {item['bimamba_rmse_m']:.3f} m, "
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
    parser = argparse.ArgumentParser(description="Run the dedicated NCLT BiMamba sensor-fusion experiment.")
    parser.add_argument("--data-dir", default="./data/nclt_fusion", help="Directory containing train.pt/val.pt/test.pt.")
    parser.add_argument("--output-dir", default="./nclt_bimamba_outputs", help="Directory for outputs.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--encoder-direction", choices=["forward", "bidirectional"], default="bidirectional")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--kalmannet-checkpoint", default="./nclt_tbptt_outputs/best_nclt_tbptt_kalmannet.pt")
    parser.add_argument("--kalmannet-hidden-dim", type=int, default=64)
    parser.add_argument("--freeze-filter", action="store_true")
    parser.add_argument("--unfreeze-filter", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "nclt_bimamba_experiment.py requires CUDA because the shared BiMamba encoder "
            "in this repository uses mamba_ssm without a CPU execution path."
        )
    if args.freeze_filter and args.unfreeze_filter:
        raise ValueError("Use only one of --freeze-filter or --unfreeze-filter.")
    freeze_filter = True
    if args.freeze_filter:
        freeze_filter = True
    if args.unfreeze_filter:
        freeze_filter = False

    config = ExperimentConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        patience=args.patience,
        encoder_direction=args.encoder_direction,
        kalmannet_hidden_dim=args.kalmannet_hidden_dim,
        freeze_filter=freeze_filter,
    )

    set_seed(config.seed)
    device = torch.device("cuda")
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
    model.load_filter_checkpoint(args.kalmannet_checkpoint)

    model, train_summary = train_model(model, train_loader, val_sessions, config, device)

    checkpoint_path = output_dir / "best_nclt_bimamba.pt"
    torch.save(model.state_dict(), checkpoint_path)

    test_metrics = evaluate_sessions(model, test_sessions, device)
    plot_path = output_dir / "nclt_bimamba_trajectories.png"
    make_plot(test_metrics, plot_path)
    write_report(output_dir, config, train_summary, test_metrics, checkpoint_path, plot_path)

    print("\nTest summary")
    print(f"Overall RMSE: {test_metrics['overall_rmse_m']:.3f} m")
    print(f"Average GPS-only RMSE: {test_metrics['avg_gps_rmse_m']:.3f} m")
    print(f"Average wheel-only RMSE: {test_metrics['avg_wheel_rmse_m']:.3f} m")
    for item in test_metrics["per_trajectory"]:
        print(
            f"{item['data_date']}: model={item['bimamba_rmse_m']:.3f} m "
            f"gps={item['gps_only_rmse_m']:.3f} m wheel={item['wheel_only_rmse_m']:.3f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
