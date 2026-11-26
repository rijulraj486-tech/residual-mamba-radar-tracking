import math

import torch
import torch.nn as nn

from radar_tracking.models.bimamba_encoder import BiMambaEncoder
from radar_tracking.models.prediction_head import PredictionHead


def wrap_to_pi(angle):

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class BiMambaTrackerModel(nn.Module):

    def __init__(
        self,
        state_dim=6,
        obs_dim=2,
        dt=1.0,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.08,
        encoder_direction="forward",
        task="single_target_radar",
        **_,
    ):

        super().__init__()

        if task not in {"single_target_radar", "radar_benchmark"}:
            raise ValueError(
                "BiMambaTrackerModel now supports only the single-target radar task."
            )

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.encoder_direction = encoder_direction
        if self.state_dim <= 0 or self.obs_dim <= 0:
            raise ValueError("state_dim and obs_dim must be positive integers.")
        if self.dt <= 0:
            raise ValueError("dt must be positive for the single-target radar model.")

        scalar_features = 9
        self.token_dim = scalar_features + 2 * state_dim
        if self.token_dim != 21:
            raise ValueError(
                "The single-target radar token layout expects 21 features for a 6D state."
            )

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
        self.state_head = PredictionHead(
            input_dim=self.encoder.output_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

    def _model_device(self):

        return next(self.parameters()).device

    def _prepare_observations(self, observations):

        device = self._model_device()
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32, device=device)
        else:
            observations = observations.to(device=device, dtype=torch.float32)

        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        if observations.dim() != 3:
            raise ValueError(
                "observations must have shape [batch, obs_dim, seq_len] or [obs_dim, seq_len]."
            )
        if observations.shape[1] != self.obs_dim:
            raise ValueError(
                f"observations expect obs_dim={self.obs_dim}, got shape {tuple(observations.shape)}."
            )
        if observations.shape[2] < 1:
            raise ValueError("observations must contain at least one time step.")
        return observations

    def _prepare_init_state(self, init_state, batch_size):

        device = self._model_device()
        if not isinstance(init_state, torch.Tensor):
            init_state = torch.tensor(init_state, dtype=torch.float32, device=device)
        else:
            init_state = init_state.to(device=device, dtype=torch.float32)

        if init_state.dim() == 1:
            init_state = init_state.unsqueeze(0).unsqueeze(-1)
        elif init_state.dim() == 2:
            if init_state.shape[1] == self.state_dim:
                init_state = init_state.unsqueeze(-1)
            elif init_state.shape[0] == self.state_dim and init_state.shape[1] == 1:
                init_state = init_state.unsqueeze(0)
            else:
                raise ValueError(
                    "init_state with 2 dims must have shape [batch, state_dim] or [state_dim, 1]."
                )
        elif init_state.dim() != 3:
            raise ValueError(
                "init_state must have shape [batch, state_dim, 1], [batch, state_dim], or [state_dim]."
            )

        if init_state.shape[1] != self.state_dim or init_state.shape[2] != 1:
            raise ValueError(
                f"init_state expects shape [batch, {self.state_dim}, 1], got {tuple(init_state.shape)}."
            )
        if init_state.shape[0] != batch_size:
            if init_state.shape[0] == 1:
                init_state = init_state.expand(batch_size, -1, -1)
            else:
                raise ValueError(
                    "init_state batch size must match observations batch size."
                )
        return init_state

    def measurement_to_cartesian(self, observations):

        ranges = observations[:, 0, :]
        bearings = observations[:, 1, :]
        x_pos = ranges * torch.cos(bearings)
        y_pos = ranges * torch.sin(bearings)
        return torch.stack([x_pos, y_pos], dim=1)

    def coarse_state_sequence(self, observations, init_state):

        positions = self.measurement_to_cartesian(observations)
        batch_size, _, seq_len = positions.shape
        coarse = torch.zeros(
            batch_size,
            self.state_dim,
            seq_len,
            device=observations.device,
            dtype=observations.dtype,
        )

        coarse[:, 0, :] = positions[:, 0, :]
        coarse[:, 3, :] = positions[:, 1, :]

        coarse[:, 1, 0] = init_state[:, 1, 0]
        coarse[:, 4, 0] = init_state[:, 4, 0]
        if seq_len > 1:
            coarse[:, 1, 1:] = (positions[:, 0, 1:] - positions[:, 0, :-1]) / self.dt
            coarse[:, 4, 1:] = (positions[:, 1, 1:] - positions[:, 1, :-1]) / self.dt

        coarse[:, 2, 0] = init_state[:, 2, 0]
        coarse[:, 5, 0] = init_state[:, 5, 0]
        if seq_len > 1:
            coarse[:, 2, 1] = coarse[:, 2, 0]
            coarse[:, 5, 1] = coarse[:, 5, 0]
        if seq_len > 2:
            coarse[:, 2, 2:] = (coarse[:, 1, 2:] - coarse[:, 1, 1:-1]) / self.dt
            coarse[:, 5, 2:] = (coarse[:, 4, 2:] - coarse[:, 4, 1:-1]) / self.dt

        return coarse

    def build_tokens(self, observations, init_state):

        cartesian = self.measurement_to_cartesian(observations)
        coarse = self.coarse_state_sequence(observations, init_state)

        ranges = observations[:, 0, :]
        bearings = observations[:, 1, :]
        sin_bearing = torch.sin(bearings)
        cos_bearing = torch.cos(bearings)

        delta_range = torch.zeros_like(ranges)
        delta_bearing = torch.zeros_like(bearings)
        delta_x = torch.zeros_like(cartesian[:, 0, :])
        delta_y = torch.zeros_like(cartesian[:, 1, :])
        delta_range[:, 1:] = ranges[:, 1:] - ranges[:, :-1]
        delta_bearing[:, 1:] = wrap_to_pi(bearings[:, 1:] - bearings[:, :-1])
        delta_x[:, 1:] = cartesian[:, 0, 1:] - cartesian[:, 0, :-1]
        delta_y[:, 1:] = cartesian[:, 1, 1:] - cartesian[:, 1, :-1]

        init_repeat = init_state.squeeze(-1).unsqueeze(-1).expand(-1, -1, observations.shape[-1])

        token_parts = [
            ranges.unsqueeze(1),
            sin_bearing.unsqueeze(1),
            cos_bearing.unsqueeze(1),
            cartesian,
            delta_range.unsqueeze(1),
            delta_bearing.unsqueeze(1),
            delta_x.unsqueeze(1),
            delta_y.unsqueeze(1),
            coarse,
            init_repeat,
        ]
        tokens = torch.cat(token_parts, dim=1)
        if tokens.shape[1] != self.token_dim:
            raise RuntimeError(
                f"Token has {tokens.shape[1]} features but expected {self.token_dim}."
            )

        return tokens, coarse

    def forward(self, observations=None, init_state=None, history=None, measurement=None, **_):

        if observations is None:
            observations = history
        if init_state is None:
            init_state = measurement
        if observations is None or init_state is None:
            raise ValueError("Single-target radar tracking requires observations and init_state.")

        observations = self._prepare_observations(observations)
        init_state = self._prepare_init_state(init_state, observations.shape[0])

        tokens, coarse = self.build_tokens(observations, init_state)
        token_sequence = tokens.transpose(1, 2)
        coarse_sequence = coarse.transpose(1, 2)

        embedded_tokens = self.input_proj(token_sequence)
        encoded_tokens = self.encoder(embedded_tokens)
        encoded_tokens = self.output_norm(encoded_tokens)

        state_estimates = self.state_head(encoded_tokens, coarse_sequence)

        return {
            "tokens": tokens,
            "coarse_state": coarse,
            "embedded_tokens": embedded_tokens,
            "encoded_tokens": encoded_tokens,
            "state_estimates": state_estimates.transpose(1, 2),
        }
