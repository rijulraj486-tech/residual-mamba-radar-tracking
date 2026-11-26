#!/usr/bin/env python3

"""Standalone pure BiMamba filter+smoother for NCLT wheel/GPS/IMU fusion."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from radar_tracking.models.bimamba_encoder import BiMambaEncoder  # noqa: E402
from radar_tracking.cli.nclt_bimamba_experiment import NCLTFusionBiMamba  # noqa: E402


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


class NCLTPureWindowDataset(Dataset):
    def __init__(self, sessions: list[dict], seq_len: int = 60, stride: int = 30):
        self.windows = []
        for item in sessions:
            targets = derive_state_targets(item)
            wheel = item["wheel"].float()
            imu = item.get("imu")
            if imu is None:
                imu = item["imu_sensor"]
            imu = imu.float()
            gps = item["filtered_gps"].float()
            total_len = targets.shape[0]
            if total_len < seq_len:
                continue
            for start in range(0, total_len - seq_len + 1, stride):
                init_state = item["initial_state"].float() if start == 0 else targets[start - 1]
                self.windows.append(
                    {
                        "wheel": wheel[start:start + seq_len],
                        "imu": imu[start:start + seq_len],
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
        "imu": torch.stack([item["imu"] for item in batch], dim=0),
        "filtered_gps": torch.stack([item["filtered_gps"] for item in batch], dim=0),
        "targets": torch.stack([item["targets"] for item in batch], dim=0),
        "initial_state": torch.stack([item["initial_state"] for item in batch], dim=0),
    }


def body_accel_to_world(accel_body: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    ax = accel_body[..., 0]
    ay = accel_body[..., 1]
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    return torch.stack([ax * cos_h - ay * sin_h, ax * sin_h + ay * cos_h], dim=-1)


class PureFilterHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 12),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, encoded_step: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(encoded_step)
        return {
            "pos_gain": torch.sigmoid(raw[:, 0:2]),
            "vel_gain": 0.35 * torch.tanh(raw[:, 2:4]),
            "acc_gain": 0.20 * torch.tanh(raw[:, 4:6]),
            "theta_gain": torch.sigmoid(raw[:, 6:7]),
            "omega_gain": torch.sigmoid(raw[:, 7:8]),
            "log_var": torch.clamp(raw[:, 8:10], min=-2.5, max=4.0),
            "pos_bias": 1.0 * torch.tanh(raw[:, 10:12]),
        }


class PureSmootherHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
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

    def forward(self, encoded: torch.Tensor, filtered_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(encoded)
        pos_gate = torch.sigmoid(raw[..., 6:7])
        dyn_gate = torch.sigmoid(raw[..., 7:8])

        residual = torch.zeros_like(filtered_states)
        residual[..., 0:2] = pos_gate * 3.0 * torch.tanh(raw[..., 0:2])
        residual[..., 2:4] = dyn_gate * 0.6 * torch.tanh(raw[..., 2:4])
        residual[..., 4] = (0.03 * dyn_gate[..., 0]) * torch.tanh(raw[..., 4])
        residual[..., 5] = (0.03 * dyn_gate[..., 0]) * torch.tanh(raw[..., 5])

        smoothed = filtered_states + residual
        smoothed[..., 4] = wrap_to_pi(smoothed[..., 4])
        return smoothed, residual


class PureBiMambaFilterSmoother(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,
        filter_embed_dim: int = 128,
        filter_hidden_dim: int = 256,
        filter_layers: int = 2,
        smoother_embed_dim: int = 128,
        smoother_hidden_dim: int = 256,
        smoother_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pos_scale = 200.0
        self.vel_scale = 5.0
        self.acc_scale = 2.5
        self.omega_scale = 0.5
        self.theta_prior_blend = 0.85
        self.omega_prior_blend = 0.85
        self.acc_prior_scale = 0.15

        self.filter_token_dim = 23
        self.filter_input_proj = nn.Sequential(
            nn.Linear(self.filter_token_dim, filter_embed_dim),
            nn.LayerNorm(filter_embed_dim),
            nn.SiLU(),
        )
        self.filter_encoder = BiMambaEncoder(
            dim=filter_embed_dim,
            num_layers=filter_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            encoder_direction="forward",
        )
        self.filter_output_norm = nn.LayerNorm(self.filter_encoder.output_dim)
        self.filter_head = PureFilterHead(self.filter_encoder.output_dim, hidden_dim=filter_hidden_dim)

        self.smoother_token_dim = 19
        self.smoother_input_proj = nn.Sequential(
            nn.Linear(self.smoother_token_dim, smoother_embed_dim),
            nn.LayerNorm(smoother_embed_dim),
            nn.SiLU(),
        )
        self.smoother_encoder = BiMambaEncoder(
            dim=smoother_embed_dim,
            num_layers=smoother_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            encoder_direction="forward",
        )
        self.smoother_output_norm = nn.LayerNorm(self.smoother_encoder.output_dim)
        self.smoother_head = PureSmootherHead(self.smoother_encoder.output_dim, hidden_dim=smoother_hidden_dim)

    def _propagate_prior_step(self, state: torch.Tensor, wheel_t: torch.Tensor, imu_t: torch.Tensor) -> torch.Tensor:
        vl = wheel_t[:, 0]
        vr = wheel_t[:, 1]
        v_c = 0.5 * (vl + vr)
        theta_pred = wrap_to_pi(state[:, 4] + state[:, 5])
        theta_imu = imu_t[:, 2]
        omega_imu = imu_t[:, 3]
        theta_prior = wrap_to_pi(theta_pred + self.theta_prior_blend * wrap_to_pi(theta_imu - theta_pred))
        omega_prior = state[:, 5] + self.omega_prior_blend * (omega_imu - state[:, 5])

        accel_world = body_accel_to_world(imu_t[:, 0:2], theta_prior)
        vx_prior = v_c * torch.cos(theta_prior) + self.acc_prior_scale * accel_world[:, 0]
        vy_prior = v_c * torch.sin(theta_prior) + self.acc_prior_scale * accel_world[:, 1]

        prior = state.clone()
        prior[:, 0] = state[:, 0] + vx_prior
        prior[:, 1] = state[:, 1] + vy_prior
        prior[:, 2] = vx_prior
        prior[:, 3] = vy_prior
        prior[:, 4] = theta_prior
        prior[:, 5] = omega_prior
        return prior

    def _build_motion_prior(self, wheel: torch.Tensor, imu: torch.Tensor, init_state: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = wheel.shape
        prior_states = torch.zeros(batch_size, seq_len, self.state_dim, device=wheel.device, dtype=wheel.dtype)
        state = init_state.clone()
        for t in range(seq_len):
            state = self._propagate_prior_step(state, wheel[:, t, :], imu[:, t, :])
            prior_states[:, t, :] = state
        return prior_states

    def _build_filter_tokens(
        self,
        wheel: torch.Tensor,
        imu: torch.Tensor,
        filtered_gps: torch.Tensor,
        init_state: torch.Tensor,
        motion_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gps_available = (~torch.isnan(filtered_gps).any(dim=-1, keepdim=True)).float()
        gps_filled = torch.nan_to_num(filtered_gps, nan=0.0)

        origin = init_state[:, None, :2]
        gps_rel = gps_filled - origin
        prior_rel = motion_prior[:, :, :2] - origin
        prior_vel = motion_prior[:, :, 2:4]
        prior_theta = motion_prior[:, :, 4:5]
        prior_omega = motion_prior[:, :, 5:6]

        accel_world = body_accel_to_world(imu[:, :, 0:2], imu[:, :, 2])
        innovation = (gps_filled - motion_prior[:, :, :2]) * gps_available
        theta_innovation = wrap_to_pi(imu[:, :, 2:3] - prior_theta)
        omega_innovation = (imu[:, :, 3:4] - prior_omega)
        delta_prior_pos = torch.zeros_like(prior_rel)
        delta_prior_pos[:, 1:, :] = prior_rel[:, 1:, :] - prior_rel[:, :-1, :]

        tokens = torch.cat(
            [
                wheel[:, :, 0:2] / self.vel_scale,
                imu[:, :, 0:2] / self.acc_scale,
                imu[:, :, 2:3] / math.pi,
                imu[:, :, 3:4] / self.omega_scale,
                gps_rel / self.pos_scale,
                gps_available,
                innovation / self.pos_scale,
                prior_rel / self.pos_scale,
                prior_vel / self.vel_scale,
                prior_theta / math.pi,
                prior_omega / self.omega_scale,
                accel_world / self.acc_scale,
                theta_innovation / math.pi,
                omega_innovation / self.omega_scale,
                delta_prior_pos / self.pos_scale,
            ],
            dim=-1,
        )
        if tokens.shape[-1] != self.filter_token_dim:
            raise RuntimeError(f"Filter token has {tokens.shape[-1]} features but expected {self.filter_token_dim}.")
        return tokens, gps_available, gps_filled, accel_world

    def _run_filter(
        self,
        wheel: torch.Tensor,
        imu: torch.Tensor,
        gps_available: torch.Tensor,
        gps_filled: torch.Tensor,
        accel_world: torch.Tensor,
        motion_prior: torch.Tensor,
        encoded_filter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = wheel.shape
        filtered_states = []
        log_vars = []
        state_prev = motion_prior[:, 0, :].clone()

        for t in range(seq_len):
            if t == 0:
                prev_state = motion_prior[:, 0, :].clone()
                prev_state[:, 0:2] = prev_state[:, 0:2] - prev_state[:, 2:4]
                prev_state[:, 2:4] = 0.0
            else:
                prev_state = state_prev
            prior = self._propagate_prior_step(prev_state, wheel[:, t, :], imu[:, t, :])

            head_out = self.filter_head(encoded_filter[:, t, :])
            gps_mask = gps_available[:, t, :]
            innovation = (gps_filled[:, t, :] - prior[:, :2]) * gps_mask

            theta_innovation = wrap_to_pi(imu[:, t, 2:3] - prior[:, 4:5])
            omega_innovation = imu[:, t, 3:4] - prior[:, 5:6]

            post = prior.clone()
            post[:, 0:2] = prior[:, 0:2] + head_out["pos_gain"] * innovation + gps_mask * head_out["pos_bias"]
            post[:, 2:4] = prior[:, 2:4] + head_out["vel_gain"] * innovation + head_out["acc_gain"] * accel_world[:, t, :]
            post[:, 4:5] = wrap_to_pi(prior[:, 4:5] + head_out["theta_gain"] * theta_innovation)
            post[:, 5:6] = prior[:, 5:6] + head_out["omega_gain"] * omega_innovation

            filtered_states.append(post)
            log_vars.append(head_out["log_var"])
            state_prev = post

        return torch.stack(filtered_states, dim=1), torch.stack(log_vars, dim=1)

    def _build_smoother_tokens(
        self,
        filtered_states: torch.Tensor,
        wheel: torch.Tensor,
        imu: torch.Tensor,
        filtered_gps: torch.Tensor,
        log_var: torch.Tensor,
        init_state: torch.Tensor,
    ) -> torch.Tensor:
        gps_available = (~torch.isnan(filtered_gps).any(dim=-1, keepdim=True)).float()
        gps_filled = torch.nan_to_num(filtered_gps, nan=0.0)
        origin = init_state[:, None, :2]
        filtered_rel = filtered_states[:, :, :2] - origin
        gps_rel = gps_filled - origin
        post_innovation = (gps_filled - filtered_states[:, :, :2]) * gps_available
        delta_filtered_pos = torch.zeros_like(filtered_rel)
        delta_filtered_pos[:, 1:, :] = filtered_rel[:, 1:, :] - filtered_rel[:, :-1, :]

        accel_world = body_accel_to_world(imu[:, :, 0:2], imu[:, :, 2])

        tokens = torch.cat(
            [
                filtered_rel / self.pos_scale,
                filtered_states[:, :, 2:4] / self.vel_scale,
                filtered_states[:, :, 4:5] / math.pi,
                filtered_states[:, :, 5:6] / self.omega_scale,
                gps_rel / self.pos_scale,
                gps_available,
                post_innovation / self.pos_scale,
                log_var,
                wheel[:, :, 0:2] / self.vel_scale,
                accel_world / self.acc_scale,
                delta_filtered_pos / self.pos_scale,
            ],
            dim=-1,
        )
        if tokens.shape[-1] != self.smoother_token_dim:
            raise RuntimeError(f"Smoother token has {tokens.shape[-1]} features but expected {self.smoother_token_dim}.")
        return tokens

    def forward(
        self,
        wheel: torch.Tensor,
        imu: torch.Tensor,
        filtered_gps: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        motion_prior = self._build_motion_prior(wheel, imu, initial_state)
        filter_tokens, gps_available, gps_filled, accel_world = self._build_filter_tokens(
            wheel,
            imu,
            filtered_gps,
            initial_state,
            motion_prior,
        )
        filter_embedded = self.filter_input_proj(filter_tokens)
        filter_encoded = self.filter_encoder(filter_embedded)
        filter_encoded = self.filter_output_norm(filter_encoded)

        filtered_states, log_var = self._run_filter(
            wheel,
            imu,
            gps_available,
            gps_filled,
            accel_world,
            motion_prior,
            filter_encoded,
        )

        smoother_tokens = self._build_smoother_tokens(
            filtered_states,
            wheel,
            imu,
            filtered_gps,
            log_var,
            initial_state,
        )
        smoother_embedded = self.smoother_input_proj(smoother_tokens)
        smoother_encoded = self.smoother_encoder(smoother_embedded)
        smoother_encoded = self.smoother_output_norm(smoother_encoded)
        smoothed_states, smoother_residual = self.smoother_head(smoother_encoded, filtered_states)

        return {
            "motion_prior": motion_prior,
            "filter_tokens": filter_tokens,
            "filtered_states": filtered_states,
            "smoother_tokens": smoother_tokens,
            "encoded_tokens": smoother_encoded,
            "state_estimates": smoothed_states,
            "log_var_pos": log_var,
            "smoother_residual": smoother_residual,
        }


@dataclass
class ExperimentConfig:
    seed: int = 42
    seq_len: int = 20
    stride: int = 40
    batch_size: int = 8
    epochs: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-5
    filter_embed_dim: int = 64
    filter_hidden_dim: int = 128
    filter_layers: int = 1
    smoother_embed_dim: int = 64
    smoother_hidden_dim: int = 128
    smoother_layers: int = 1
    d_state: int = 8
    d_conv: int = 2
    expand: int = 2
    dropout: float = 0.05
    grad_clip: float = 1.0
    patience: int = 2
    distill_weight: float = 0.1
    warmup_frac: float = 0.1
    min_lr_scale: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.995


class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.copy_(self.backup[name])
        self.backup = {}


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float,
    min_lr_scale: float,
):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(total_steps * warmup_frac), 1)
    min_lr_scale = float(max(min(min_lr_scale, 1.0), 0.0))

    def lr_lambda(step: int) -> float:
        step = min(max(step, 0), total_steps)
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@contextmanager
def ema_eval_context(model: nn.Module, ema: EMAHelper | None):
    if ema is None:
        yield
        return
    ema.apply_shadow(model)
    try:
        yield
    finally:
        ema.restore(model)


def build_hybrid_teacher(metrics_path: Path, checkpoint_path: Path, device: torch.device) -> NCLTFusionBiMamba:
    payload = json.loads(metrics_path.read_text())
    config = payload["config"]
    teacher = NCLTFusionBiMamba(
        kalmannet_hidden_dim=int(config["kalmannet_hidden_dim"]),
        freeze_filter=bool(config["freeze_filter"]),
        embed_dim=int(config["embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        d_state=int(config["d_state"]),
        d_conv=int(config["d_conv"]),
        expand=int(config["expand"]),
        dropout=float(config["dropout"]),
        encoder_direction=str(config["encoder_direction"]),
    ).to(device)
    filter_ckpt = payload["artifacts"]["filter_checkpoint_source"]
    teacher.load_filter_checkpoint(filter_ckpt)
    teacher.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def weighted_state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = pred.new_tensor([1.2, 1.2, 0.10, 0.10, 0.05, 0.05]).view(1, 1, 6)
    delta = pred - target
    delta[..., 4] = wrap_to_pi(delta[..., 4])
    return ((delta ** 2) * weights).mean()


def position_nll(pred_xy: torch.Tensor, target_xy: torch.Tensor, log_var_xy: torch.Tensor) -> torch.Tensor:
    error_sq = (pred_xy - target_xy) ** 2
    inv_var = torch.exp(-log_var_xy)
    return 0.5 * (error_sq * inv_var + log_var_xy).mean()


def sequence_rmse_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> float:
    return float(torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=1).mean()).item())


def evaluate_sessions(model: PureBiMambaFilterSmoother, sessions: list[dict], device: torch.device) -> dict:
    model.eval()
    results = []
    gps_rmses = []
    wheel_rmses = []
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for item in sessions:
            wheel = item["wheel"].float().to(device).unsqueeze(0)
            imu = item.get("imu")
            if imu is None:
                imu = item["imu_sensor"]
            imu = imu.float().to(device).unsqueeze(0)
            filtered_gps = item["filtered_gps"].float().to(device).unsqueeze(0)
            initial_state = item["initial_state"].float().to(device).unsqueeze(0)
            raw_gps = item["gps"].float()
            gt_xy = item["ground_truth"].float()

            output = model(wheel, imu, filtered_gps, initial_state)
            pred_xy = output["state_estimates"][0, :, :2].detach().cpu()
            wheel_xy = output["motion_prior"][0, :, :2].detach().cpu()

            model_rmse = sequence_rmse_xy(pred_xy, gt_xy)
            gps_rmse = sequence_rmse_xy(raw_gps, gt_xy)
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
    model: PureBiMambaFilterSmoother,
    train_loader: DataLoader,
    val_sessions: list[dict],
    config: ExperimentConfig,
    device: torch.device,
    teacher_model: NCLTFusionBiMamba | None = None,
) -> tuple[PureBiMambaFilterSmoother, dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * max(len(train_loader), 1)
    scheduler = make_lr_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_frac=config.warmup_frac,
        min_lr_scale=config.min_lr_scale,
    )
    ema = EMAHelper(model, decay=config.ema_decay) if config.use_ema else None
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
            imu = batch["imu"].to(device)
            gps = batch["filtered_gps"].to(device)
            targets = batch["targets"].to(device)
            initial_state = batch["initial_state"].to(device)

            optimizer.zero_grad()
            output = model(wheel, imu, gps, initial_state)
            final_loss = weighted_state_loss(output["state_estimates"], targets)
            filter_loss = 0.4 * weighted_state_loss(output["filtered_states"], targets)
            nll_loss = 0.6 * position_nll(output["state_estimates"][:, :, :2], targets[:, :, :2], output["log_var_pos"])
            smooth_penalty = 5e-4 * output["smoother_residual"].pow(2).mean()
            loss = final_loss + filter_loss + nll_loss + smooth_penalty
            if teacher_model is not None and config.distill_weight > 0.0:
                with torch.no_grad():
                    teacher_output = teacher_model(wheel=wheel, filtered_gps=gps, initial_state=initial_state)
                distill_loss = weighted_state_loss(output["state_estimates"], teacher_output["state_estimates"])
                loss = loss + config.distill_weight * distill_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            epoch_loss += float(loss.item())
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        with ema_eval_context(model, ema):
            val_metrics = evaluate_sessions(model, val_sessions, device)
        val_rmse = val_metrics["overall_rmse_m"]
        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_rmse_m": val_rmse,
                "lr": lr_now,
            }
        )
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f} val_rmse={val_rmse:.4f} m lr={lr_now:.6e}")

        if val_rmse < best_val:
            best_val = val_rmse
            with ema_eval_context(model, ema):
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
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], label="Pure BiMamba", linewidth=1.5)
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
    metrics_path = output_dir / "nclt_bimamba_pure_metrics.json"
    report_path = output_dir / "nclt_bimamba_pure_report.md"

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
        "# NCLT Pure BiMamba Result",
        "",
        "## Summary",
        f"- Overall Pure BiMamba RMSE: {test_metrics['overall_rmse_m']:.3f} m",
        f"- Average GPS-only RMSE: {test_metrics['avg_gps_rmse_m']:.3f} m",
        f"- Average wheel-only RMSE: {test_metrics['avg_wheel_rmse_m']:.3f} m",
        f"- Best validation RMSE: {train_summary['best_val_rmse_m']:.3f} m",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone pure BiMamba NCLT sensor-fusion experiment.")
    parser.add_argument("--data-dir", default="./data/nclt_fusion")
    parser.add_argument("--output-dir", default="./nclt_bimamba_pure_outputs")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--filter-embed-dim", type=int, default=64)
    parser.add_argument("--filter-hidden-dim", type=int, default=128)
    parser.add_argument("--filter-layers", type=int, default=1)
    parser.add_argument("--smoother-embed-dim", type=int, default=64)
    parser.add_argument("--smoother-hidden-dim", type=int, default=128)
    parser.add_argument("--smoother-layers", type=int, default=1)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--d-conv", type=int, default=2)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--teacher-metrics", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("nclt_bimamba_pure_experiment.py requires CUDA.")

    config = ExperimentConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        filter_embed_dim=args.filter_embed_dim,
        filter_hidden_dim=args.filter_hidden_dim,
        filter_layers=args.filter_layers,
        smoother_embed_dim=args.smoother_embed_dim,
        smoother_hidden_dim=args.smoother_hidden_dim,
        smoother_layers=args.smoother_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        warmup_frac=args.warmup_frac,
        min_lr_scale=args.min_lr_scale,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        distill_weight=args.distill_weight,
    )

    set_seed(config.seed)
    device = torch.device("cuda")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sessions = load_pt(data_dir / "train.pt")
    val_sessions = load_pt(data_dir / "val.pt")
    test_sessions = load_pt(data_dir / "test.pt")

    train_dataset = NCLTPureWindowDataset(train_sessions, seq_len=config.seq_len, stride=config.stride)
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training windows were generated. Reduce --seq-len and/or --stride for this dataset split."
        )
    train_loader = DataLoader(
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
    if args.teacher_metrics and args.teacher_checkpoint:
        teacher_model = build_hybrid_teacher(
            metrics_path=Path(args.teacher_metrics),
            checkpoint_path=Path(args.teacher_checkpoint),
            device=device,
        )

    model, train_summary = train_model(model, train_loader, val_sessions, config, device, teacher_model=teacher_model)
    checkpoint_path = output_dir / "best_nclt_bimamba_pure.pt"
    torch.save(model.state_dict(), checkpoint_path)

    test_metrics = evaluate_sessions(model, test_sessions, device)
    plot_path = output_dir / "nclt_bimamba_pure_trajectories.png"
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
