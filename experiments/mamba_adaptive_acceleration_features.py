"""Reliability-adaptive causal acceleration feature for the Mamba ablation."""

from __future__ import annotations

import torch


def adaptive_causal_acceleration_features(
    positions: torch.Tensor,
    initial_velocity: torch.Tensor,
    initial_acceleration: torch.Tensor,
    ranges: torch.Tensor,
    bearings: torch.Tensor,
    dt: float,
    *,
    sigma_range: float,
    sigma_bearing: float,
    alpha_min: float,
    alpha_max: float,
    reliability_scale: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal acceleration and EMA alpha, both shaped [batch,2,time]/[batch,time].

    `positions` and `ranges` share distance units; `sigma_bearing` is radians.
    The supplied sensor standard deviations therefore need to be expressed in
    the same (possibly scaled) units as positions. Alpha at t=0 is set to
    `alpha_max` for a complete, reproducible time series, though no update uses
    it because the initial smoothed velocity is supplied directly.
    """
    if positions.ndim != 3 or positions.shape[1] != 2 or positions.shape[2] < 1:
        raise ValueError("positions must have shape [batch, 2, time] with time >= 1")
    batch, _, steps = positions.shape
    shape = (batch, steps)
    if tuple(initial_velocity.shape) != (batch, 2) or tuple(initial_acceleration.shape) != (batch, 2):
        raise ValueError("initial states must have shape [batch, 2]")
    if tuple(ranges.shape) != shape or tuple(bearings.shape) != shape:
        raise ValueError("ranges and bearings must have shape [batch, time]")
    tensors = (initial_velocity, initial_acceleration, ranges, bearings)
    if any(x.device != positions.device or x.dtype != positions.dtype for x in tensors):
        raise ValueError("feature inputs must share device and dtype")
    if not torch.is_floating_point(positions):
        raise TypeError("feature inputs must be floating point")
    if any(not torch.isfinite(x).all().item() for x in (positions, *tensors)):
        raise ValueError("feature inputs must be finite")
    if not float(dt) > 0:
        raise ValueError("dt must be positive")
    if sigma_range <= 0 or sigma_bearing <= 0 or reliability_scale <= 0 or eps <= 0:
        raise ValueError("noise scales, reliability_scale, and eps must be positive")
    if not (0 < alpha_min <= alpha_max <= 1):
        raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max <= 1")

    acceleration = torch.zeros_like(positions)
    acceleration[:, :, 0] = initial_acceleration
    if steps > 1:
        acceleration[:, :, 1] = initial_acceleration
    alphas = positions.new_full(shape, float(alpha_max))
    smoothed = torch.zeros_like(positions)
    smoothed[:, :, 0] = initial_velocity

    if steps > 1:
        # trace(J diag(sr^2, stheta^2) J^T) = sr^2 + (r*stheta)^2.
        variance = float(sigma_range) ** 2 + (ranges * float(sigma_bearing)).square()
        for index in range(1, steps):
            velocity = (positions[:, :, index] - positions[:, :, index - 1]) / float(dt)
            innovation = positions[:, :, index] - (
                positions[:, :, index - 1] + smoothed[:, :, index - 1] * float(dt)
            )
            innovation_variance = (variance[:, index] + variance[:, index - 1]).clamp_min(float(eps))
            normalized_energy = innovation.square().sum(dim=1) / innovation_variance
            alpha = float(alpha_min) + (float(alpha_max) - float(alpha_min)) / (
                1.0 + normalized_energy / float(reliability_scale)
            )
            alpha = alpha.clamp(min=float(alpha_min), max=float(alpha_max))
            alphas[:, index] = alpha
            smoothed[:, :, index] = (
                alpha[:, None] * velocity
                + (1.0 - alpha[:, None]) * smoothed[:, :, index - 1]
            )
        if steps > 2:
            acceleration[:, :, 2:] = (
                smoothed[:, :, 2:] - smoothed[:, :, 1:-1]
            ) / float(dt)
    return acceleration, alphas
