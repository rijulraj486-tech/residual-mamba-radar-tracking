import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from mamba_acceleration_features import causal_acceleration_features
from mamba_acceleration_feature_ablation_v1 import make_model_class
from radar_tracking.models.tracker_model import BiMambaTrackerModel


def _initial(batch=2, dtype=torch.float64):
    velocity = torch.tensor([[2.0, -3.0]] * batch, dtype=dtype)
    acceleration = torch.tensor([[0.4, -0.2]] * batch, dtype=dtype)
    return velocity, acceleration


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("ema_velocity", {"alpha": 0.25}),
        ("ema_velocity", {"alpha": 0.5}),
        ("trailing_quadratic", {"window": 5}),
        ("trailing_quadratic", {"window": 9}),
    ],
)
def test_feature_is_causal(method, kwargs):
    torch.manual_seed(5)
    positions = torch.randn(3, 2, 15, dtype=torch.float64)
    init_v, init_a = _initial(batch=3)
    original = causal_acceleration_features(
        positions, init_v, init_a, 0.5, method=method, **kwargs
    )
    changed = positions.clone()
    changed[:, :, 8:] += 1000.0 * torch.randn_like(changed[:, :, 8:])
    future_changed = causal_acceleration_features(
        changed, init_v, init_a, 0.5, method=method, **kwargs
    )
    # Perturbing observations from index 8 onward cannot affect outputs 0..7.
    torch.testing.assert_close(original[:, :, :8], future_changed[:, :, :8])


@pytest.mark.parametrize("method,kwargs", [
    ("ema_velocity", {"alpha": 0.5}),
    ("trailing_quadratic", {"window": 5}),
])
def test_initial_acceleration_and_short_sequence_boundaries(method, kwargs):
    init_v, init_a = _initial(batch=1)
    for seq_len in (1, 2):
        positions = torch.randn(1, 2, seq_len, dtype=torch.float64)
        acceleration = causal_acceleration_features(
            positions, init_v, init_a, 1.25, method=method, **kwargs
        )
        assert acceleration.shape == (1, 2, seq_len)
        torch.testing.assert_close(acceleration[:, :, 0], init_a)
        if seq_len == 2:
            torch.testing.assert_close(acceleration[:, :, 1], init_a)


@pytest.mark.parametrize("window", [5, 9])
@pytest.mark.parametrize("dt", [0.25, 1.5])
def test_trailing_quadratic_recovers_acceleration_with_boundary_growth(window, dt):
    time_steps = torch.arange(13, dtype=torch.float64) * dt
    acceleration = torch.tensor([0.8, -0.35], dtype=torch.float64)
    velocity = torch.tensor([1.7, -2.2], dtype=torch.float64)
    position0 = torch.tensor([10.0, -4.0], dtype=torch.float64)
    pos = (
        position0[None, :, None]
        + velocity[None, :, None] * time_steps[None, None, :]
        + 0.5 * acceleration[None, :, None] * time_steps[None, None, :].square()
    )
    init_v = velocity[None, :]
    init_a = torch.zeros_like(init_v)
    result = causal_acceleration_features(
        pos, init_v, init_a, dt, method="trailing_quadratic", window=window
    )
    assert result.shape == (1, 2, len(time_steps))
    # Index 2 starts with the first 3-sample fit, then the trailing window grows.
    torch.testing.assert_close(
        result[:, :, 2:], acceleration[None, :, None].expand(1, 2, 11),
        atol=1e-9, rtol=1e-9,
    )


def test_feature_units_scale_linearly_and_dt_is_applied():
    dt = 0.4
    steps = torch.arange(12, dtype=torch.float64)
    accel = torch.tensor([0.6, -0.25], dtype=torch.float64)
    vel = torch.tensor([2.0, 1.1], dtype=torch.float64)
    pos = (vel[None, :, None] * (steps * dt)[None, None, :]
           + 0.5 * accel[None, :, None] * (steps * dt)[None, None, :].square())
    init_v, init_a = vel[None, :], torch.zeros((1, 2), dtype=torch.float64)
    base = causal_acceleration_features(
        pos, init_v, init_a, dt, method="trailing_quadratic", window=5
    )
    factor = 1000.0
    scaled = causal_acceleration_features(
        pos * factor, init_v * factor, init_a * factor, dt,
        method="trailing_quadratic", window=5,
    )
    torch.testing.assert_close(scaled, base * factor, atol=1e-8, rtol=1e-10)
    torch.testing.assert_close(base[:, :, 2:], accel[None, :, None].expand(1, 2, 10))


@pytest.mark.parametrize("method,kwargs", [
    ("ema_velocity", {"alpha": 0.5}),
    ("trailing_quadratic", {"window": 9}),
])
def test_finite_for_long_sequence_and_large_coordinates(method, kwargs):
    torch.manual_seed(8)
    positions = 2.5e4 + torch.randn(4, 2, 100, dtype=torch.float64) * 1e3
    init_v, init_a = _initial(batch=4)
    result = causal_acceleration_features(
        positions, init_v, init_a, 0.01, method=method, **kwargs
    )
    assert result.shape == (4, 2, 100)
    assert torch.isfinite(result).all()


def test_rejects_invalid_dt_shapes_and_configurations():
    pos = torch.zeros(1, 2, 4, dtype=torch.float64)
    init_v, init_a = _initial(batch=1)
    with pytest.raises(ValueError, match="dt"):
        causal_acceleration_features(pos, init_v, init_a, 0.0,
                                     method="ema_velocity", alpha=0.5)
    with pytest.raises(ValueError, match="alpha"):
        causal_acceleration_features(pos, init_v, init_a, 1.0,
                                     method="ema_velocity", alpha=0.0)
    with pytest.raises(ValueError, match="window"):
        causal_acceleration_features(pos, init_v, init_a, 1.0,
                                     method="trailing_quadratic", window=2)
    with pytest.raises(ValueError, match="shape"):
        causal_acceleration_features(pos[:, :, :0], init_v, init_a, 1.0,
                                     method="ema_velocity", alpha=0.5)


def test_tracker_variant_changes_only_the_two_acceleration_token_channels():
    model_args = dict(
        task="radar_benchmark", state_dim=6, obs_dim=2, dt=1.0,
        embed_dim=8, hidden_dim=16, num_layers=1, d_state=4, d_conv=2,
        expand=2, dropout=0.0, encoder_direction="forward",
    )
    original = BiMambaTrackerModel(**model_args)
    causal_cls = make_model_class(BiMambaTrackerModel)
    variant = causal_cls(**model_args, acceleration_feature={
        "method": "trailing_quadratic", "window": 5,
    })
    observations = torch.zeros(2, 2, 8)
    observations[:, 0, :] = torch.linspace(20.0, 21.0, 8)
    observations[:, 1, :] = torch.linspace(0.2, 0.3, 8)
    init_state = torch.tensor([[20.0, 0.1, 0.01, 2.0, -0.2, -0.02]]).expand(2, -1).unsqueeze(-1)
    original_tokens, original_coarse = original.build_tokens(observations, init_state)
    variant_tokens, variant_coarse = variant.build_tokens(observations, init_state)
    assert original_tokens.shape == variant_tokens.shape == (2, 21, 8)
    assert sum(p.numel() for p in original.parameters()) == sum(
        p.numel() for p in variant.parameters()
    )
    changed = {11, 14}  # token positions of coarse ax and ay
    for channel in range(21):
        if channel not in changed:
            torch.testing.assert_close(original_tokens[:, channel], variant_tokens[:, channel])
    torch.testing.assert_close(original_coarse[:, [0, 1, 3, 4], :],
                              variant_coarse[:, [0, 1, 3, 4], :])
    torch.testing.assert_close(original_coarse[:, [2, 5], :2],
                              variant_coarse[:, [2, 5], :2])
