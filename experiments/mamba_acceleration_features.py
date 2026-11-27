"""Causal acceleration-feature helpers for the isolated ablation runner."""

from __future__ import annotations

import torch
from functools import lru_cache


_DEVICE_WEIGHT_CACHE: dict[tuple[int, int, float, str, torch.dtype], torch.Tensor] = {}


@lru_cache(maxsize=64)
def _quadratic_acceleration_weights_cpu(window: int, sample_count: int,
                                        dt: float) -> torch.Tensor:
    """Return stable trailing least-squares weights for acceleration at the endpoint."""
    if sample_count < 3 or sample_count > window:
        raise ValueError("quadratic fit requires 3 <= sample_count <= window")
    if dt <= 0:
        raise ValueError("dt must be positive")
    # Normalize time to sample steps. This avoids ill-conditioned powers of dt
    # while keeping the final derivative conversion explicit and physical.
    offsets = torch.arange(1 - sample_count, 1, dtype=torch.float64)
    design = torch.stack((torch.ones_like(offsets), offsets, offsets.square()), dim=1)
    pinv = torch.linalg.pinv(design, rtol=1e-14)
    return (2.0 / (dt * dt)) * pinv[2]


def causal_acceleration_features(
    positions: torch.Tensor,
    initial_velocity: torch.Tensor,
    initial_acceleration: torch.Tensor,
    dt: float,
    *,
    method: str,
    alpha: float | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Estimate Cartesian acceleration from current/past measured positions.

    Args:
        positions: Cartesian measured positions [batch, 2, time]. Values may be
            in physical units or a common scaled unit, but must share units with
            initial_velocity and initial_acceleration.
        initial_velocity: Initial x/y velocity [batch, 2].
        initial_acceleration: Initial x/y acceleration [batch, 2].
        dt: Sample period in seconds (or the matching time unit).
        method: ``"ema_velocity"`` or ``"trailing_quadratic"``.
        alpha: EMA update weight in (0, 1], required for EMA.
        window: Trailing sample count >= 3, required for quadratic fitting.

    Indices zero and one preserve the original initial-acceleration handling.
    For later indices, only the current and earlier positions are read.
    """
    if positions.ndim != 3 or positions.shape[1] != 2 or positions.shape[2] < 1:
        raise ValueError("positions must have shape [batch, 2, time] with time >= 1")
    batch, _, seq_len = positions.shape
    expected = (batch, 2)
    if tuple(initial_velocity.shape) != expected:
        raise ValueError(f"initial_velocity must have shape {expected}")
    if tuple(initial_acceleration.shape) != expected:
        raise ValueError(f"initial_acceleration must have shape {expected}")
    if not torch.is_floating_point(positions):
        raise TypeError("positions must use a floating-point dtype")
    if not torch.isfinite(positions).all() or not torch.isfinite(initial_velocity).all() or not torch.isfinite(initial_acceleration).all():
        raise ValueError("feature inputs must be finite")
    if not (float(dt) > 0):
        raise ValueError("dt must be positive")
    if initial_velocity.device != positions.device or initial_acceleration.device != positions.device:
        raise ValueError("feature inputs must be on the same device")
    if initial_velocity.dtype != positions.dtype or initial_acceleration.dtype != positions.dtype:
        raise ValueError("feature inputs must use the same dtype")

    acceleration = positions.new_zeros((batch, 2, seq_len))
    acceleration[:, :, 0] = initial_acceleration
    if seq_len > 1:
        acceleration[:, :, 1] = initial_acceleration

    if method == "ema_velocity":
        if alpha is None or not 0.0 < float(alpha) <= 1.0:
            raise ValueError("ema_velocity requires alpha in (0, 1]")
        velocity = positions.new_zeros((batch, 2, seq_len))
        velocity[:, :, 0] = initial_velocity
        if seq_len > 1:
            velocity[:, :, 1:] = (positions[:, :, 1:] - positions[:, :, :-1]) / float(dt)
        smoothed = positions.new_zeros((batch, 2, seq_len))
        smoothed[:, :, 0] = initial_velocity
        for index in range(1, seq_len):
            smoothed[:, :, index] = (
                float(alpha) * velocity[:, :, index]
                + (1.0 - float(alpha)) * smoothed[:, :, index - 1]
            )
        if seq_len > 2:
            acceleration[:, :, 2:] = (
                smoothed[:, :, 2:] - smoothed[:, :, 1:-1]
            ) / float(dt)
        return acceleration

    if method == "trailing_quadratic":
        if window is None or int(window) < 3:
            raise ValueError("trailing_quadratic requires window >= 3")
        window = int(window)
        weights_by_count = {}
        for count in range(3, window + 1):
            key = (window, count, float(dt), str(positions.device), positions.dtype)
            if key not in _DEVICE_WEIGHT_CACHE:
                _DEVICE_WEIGHT_CACHE[key] = _quadratic_acceleration_weights_cpu(
                    window, count, float(dt)
                ).to(device=positions.device, dtype=positions.dtype)
            weights_by_count[count] = _DEVICE_WEIGHT_CACHE[key]
        # At sequence start grow the trailing fit from three samples. The first
        # two outputs remain the supplied initial acceleration, matching M0.
        for index in range(2, seq_len):
            count = min(window, index + 1)
            weights = weights_by_count[count]
            trailing = positions[:, :, index - count + 1:index + 1]
            acceleration[:, :, index] = torch.einsum("bcn,n->bc", trailing, weights)
        return acceleration

    raise ValueError(f"Unknown acceleration feature method: {method}")
