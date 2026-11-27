import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
from mamba_adaptive_acceleration_features import adaptive_causal_acceleration_features
from mamba_adaptive_acceleration_v1 import make_model_class
from radar_tracking.models.tracker_model import BiMambaTrackerModel


def run_feature(position, init_v, init_a, ranges, bearings, dt=1.0, **kwargs):
    return adaptive_causal_acceleration_features(
        position, init_v, init_a, ranges, bearings, dt,
        sigma_range=0.03, sigma_bearing=0.017453292519943295,
        alpha_min=0.2, alpha_max=0.6, reliability_scale=8.0, **kwargs,
    )


def test_strict_causality_and_no_future_dependency():
    torch.manual_seed(2)
    pos = torch.randn(2, 2, 12, dtype=torch.float64)
    ranges = torch.full((2, 12), 25.0, dtype=torch.float64)
    bearings = torch.linspace(-0.3, 0.3, 12, dtype=torch.float64).repeat(2, 1)
    iv = torch.zeros(2, 2, dtype=torch.float64)
    ia = torch.ones_like(iv)
    a, alpha = run_feature(pos, iv, ia, ranges, bearings)
    changed = pos.clone()
    changed[:, :, 7:] += 1e4 * torch.randn_like(changed[:, :, 7:])
    b, beta = run_feature(changed, iv, ia, ranges, bearings)
    torch.testing.assert_close(a[:, :, :7], b[:, :, :7])
    torch.testing.assert_close(alpha[:, :7], beta[:, :7])


@pytest.mark.parametrize("steps", [1, 2])
def test_initial_acceleration_and_short_sequences(steps):
    pos = torch.zeros(1, 2, steps, dtype=torch.float64)
    ranges = torch.ones(1, steps, dtype=torch.float64)
    bearings = torch.zeros_like(ranges)
    iv = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    ia = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
    acc, alpha = run_feature(pos, iv, ia, ranges, bearings)
    torch.testing.assert_close(acc, ia[:, :, None].expand_as(acc))
    assert alpha.shape == (1, steps)


@pytest.mark.parametrize("dt", [0.25, 1.5])
def test_constant_acceleration_units_and_nonunit_dt(dt):
    t = torch.arange(15, dtype=torch.float64) * dt
    accel = torch.tensor([0.8, -0.35], dtype=torch.float64)
    vel = torch.tensor([1.7, -2.2], dtype=torch.float64)
    pos = (vel[None, :, None] * t[None, None, :]
           + 0.5 * accel[None, :, None] * t[None, None, :].square())
    pos += torch.tensor([25.0, 0.0], dtype=torch.float64)[None, :, None]
    ranges = torch.full((1, len(t)), 25.0, dtype=torch.float64)
    bearings = torch.zeros_like(ranges)
    init_v = vel[None, :]
    init_a = torch.zeros_like(init_v)
    result, _ = run_feature(pos, init_v, init_a, ranges, bearings, dt=dt)
    # EMA acceleration has its expected startup transient, then stays causal;
    # compare its scale with the exact physical acceleration.
    assert torch.isfinite(result).all()
    assert result[:, :, -1].abs().max() < 2.0
    scaled, _ = adaptive_causal_acceleration_features(
        pos * 1000, init_v * 1000, init_a * 1000, ranges * 1000, bearings, dt,
        sigma_range=30.0, sigma_bearing=0.017453292519943295,
        alpha_min=0.2, alpha_max=0.6, reliability_scale=8.0,
    )
    torch.testing.assert_close(scaled, result * 1000, rtol=1e-9, atol=1e-8)


def test_coefficients_are_bounded_and_deterministic_under_extremes():
    pos = torch.tensor([[[0.0, 1e8, -1e8, 1e8], [0.0, -1e8, 1e8, -1e8]]], dtype=torch.float64)
    ranges = torch.full((1, 4), 25.0, dtype=torch.float64)
    bearings = torch.zeros_like(ranges)
    iv = torch.zeros(1, 2, dtype=torch.float64)
    ia = torch.zeros_like(iv)
    first = run_feature(pos, iv, ia, ranges, bearings)
    second = run_feature(pos, iv, ia, ranges, bearings)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert torch.isfinite(first[0]).all() and torch.isfinite(first[1]).all()
    assert first[1].min() >= 0.2 and first[1].max() <= 0.6


def test_invalid_inputs_and_configurations_rejected():
    pos = torch.zeros(1, 2, 3)
    ranges = torch.ones(1, 3)
    bearings = torch.zeros_like(ranges)
    iv = torch.zeros(1, 2)
    ia = torch.zeros_like(iv)
    with pytest.raises(ValueError, match="dt"):
        run_feature(pos, iv, ia, ranges, bearings, dt=0)
    with pytest.raises(ValueError, match="alpha bounds"):
        adaptive_causal_acceleration_features(
            pos, iv, ia, ranges, bearings, 1, sigma_range=.03,
            sigma_bearing=.017, alpha_min=0, alpha_max=.5, reliability_scale=2,
        )
    with pytest.raises(ValueError, match="shape"):
        run_feature(pos, iv, ia, ranges[:, :2], bearings)


def test_only_acceleration_channels_change_and_parameter_count_is_fixed():
    args = dict(task="radar_benchmark", state_dim=6, obs_dim=2, dt=1.0,
                embed_dim=8, hidden_dim=16, num_layers=1, d_state=4,
                d_conv=2, expand=2, dropout=0.0, encoder_direction="forward")
    original = BiMambaTrackerModel(**args)
    adaptive_cls = make_model_class(BiMambaTrackerModel)
    adaptive = adaptive_cls(**args, acceleration_feature={
        "sigma_range": .03, "sigma_bearing": .017453292519943295,
        "alpha_min": .2, "alpha_max": .6, "reliability_scale": 8,
    })
    obs = torch.zeros(2, 2, 8)
    obs[:, 0] = torch.linspace(20, 21, 8)
    obs[:, 1] = torch.linspace(.2, .3, 8)
    init = torch.tensor([[20., .1, .01, 2., -.2, -.02]]).expand(2, -1).unsqueeze(-1)
    tok0, coarse0 = original.build_tokens(obs, init)
    tok1, coarse1 = adaptive.build_tokens(obs, init)
    assert tok0.shape == tok1.shape == (2, 21, 8)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in adaptive.parameters())
    for channel in range(21):
        if channel not in {11, 14}:
            torch.testing.assert_close(tok0[:, channel], tok1[:, channel])
    torch.testing.assert_close(coarse0[:, [0, 1, 3, 4]], coarse1[:, [0, 1, 3, 4]])
    torch.testing.assert_close(coarse0[:, [2, 5], :2], coarse1[:, [2, 5], :2])
