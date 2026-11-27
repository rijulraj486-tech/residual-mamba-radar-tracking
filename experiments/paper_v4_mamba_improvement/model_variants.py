"""Isolated model variants for paper-v4 validation screening."""

from __future__ import annotations

import torch
from torch import nn

from radar_tracking.models.tracker_model import BiMambaTrackerModel


NORMALIZED_FEATURE_INDICES = (0, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)


class TrainNormalizedFeatureTracker(BiMambaTrackerModel):
    """Apply frozen train-only z-scores to dimensional token channels only."""

    def __init__(self, *, feature_mean: torch.Tensor, feature_std: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        if feature_mean.shape != (21,) or feature_std.shape != (21,):
            raise ValueError("Expected 21-channel feature mean and std")
        self.register_buffer("feature_mean", feature_mean.detach().clone().float())
        self.register_buffer("feature_std", feature_std.detach().clone().float())

    def build_tokens(self, observations: torch.Tensor, init_state: torch.Tensor):
        tokens, coarse = super().build_tokens(observations, init_state)
        indices = torch.as_tensor(NORMALIZED_FEATURE_INDICES, device=tokens.device)
        means = self.feature_mean.to(device=tokens.device, dtype=tokens.dtype)[indices]
        stds = self.feature_std.to(device=tokens.device, dtype=tokens.dtype)[indices].clamp_min(1e-4)
        normalized = tokens.clone()
        normalized[:, indices, :] = (tokens[:, indices, :] - means[None, :, None]) / stds[None, :, None]
        return normalized, coarse


class ResidualBranch(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.net(encoded)


class SeparateStateGroupResidualHead(nn.Module):
    """Three independent group branches; return full [x,vx,ax,y,vy,ay] state."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.position = ResidualBranch(input_dim, hidden_dim, 2)
        self.velocity = ResidualBranch(input_dim, hidden_dim, 2)
        self.acceleration = ResidualBranch(input_dim, hidden_dim, 2)

    def forward(self, encoded_state: torch.Tensor, coarse_state: torch.Tensor) -> torch.Tensor:
        position = self.position(encoded_state)  # [B,T] -> (dx,dy)
        velocity = self.velocity(encoded_state)  # [B,T] -> (dvx,dvy)
        acceleration = self.acceleration(encoded_state)  # [B,T] -> (dax,day)
        residual = torch.stack(
            (
                position[..., 0],
                velocity[..., 0],
                acceleration[..., 0],
                position[..., 1],
                velocity[..., 1],
                acceleration[..., 1],
            ),
            dim=-1,
        )
        if residual.shape != coarse_state.shape:
            raise RuntimeError(
                f"Residual shape {tuple(residual.shape)} != coarse-state shape {tuple(coarse_state.shape)}"
            )
        return coarse_state + residual


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    head = sum(parameter.numel() for parameter in model.state_head.parameters())
    return {"total": total, "encoder": encoder, "prediction_head": head}
