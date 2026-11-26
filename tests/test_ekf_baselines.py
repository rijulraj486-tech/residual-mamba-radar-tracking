from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_tracking.config import simparams_class
from radar_tracking.data.data_generation import Q_CA_1D, Q_CA_2D, generate_synthetic_data
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.models.ekf import extended_kf, wrap_to_pi
from radar_tracking.models.ekf_baselines import (
    assert_legacy_ekf_inputs,
    calibrate_first_observation_prior,
    correlated_first_observation_update,
    covariance_for_policy,
    estimate_qr_from_training,
)


def _params(*, batch: int = 4, seq_len: int = 4, measurement_type: str = "RANGE_BEARING"):
    args = simparams_class()
    args.dt = 0.5
    args.seq_len = seq_len
    args.dataset_size = batch
    args.meas_type = measurement_type
    args.init_type = "MEAS"
    args.noise_ref = "FIXED"
    args.range_std = 30.0
    args.bearing_std = 1.0
    args.q2_scale = 1.0
    args.r2_scale = 1.0
    return args


def test_constant_acceleration_transition_matrix_matches_state_order():
    args = _params()
    model = ConstantAcceleration2DModel(args)
    expected = torch.tensor(
        [
            [1, 0.5, 0.125, 0, 0, 0],
            [0, 1, 0.5, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0.5, 0.125],
            [0, 0, 0, 0, 1, 0.5],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(model.F, expected, rtol=0, atol=0)


def test_process_noise_covariance_matches_white_jerk_blocks():
    dt = 0.5
    q = 2.0
    expected_1d = torch.tensor(
        [
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ],
        dtype=torch.float32,
    ) * q
    torch.testing.assert_close(Q_CA_1D(dt, q), expected_1d, rtol=1e-6, atol=1e-8)
    expected_2d = torch.block_diag(expected_1d, expected_1d)
    torch.testing.assert_close(Q_CA_2D(dt, q, q), expected_2d, rtol=1e-6, atol=1e-8)


def test_range_bearing_observation_handles_quadrants_and_near_origin():
    args = _params()
    model = ConstantAcceleration2DModel(args)
    state = torch.zeros(3, 6, 1)
    state[0, 0, 0], state[0, 3, 0] = 3.0, 4.0
    state[1, 0, 0], state[1, 3, 0] = -3.0, 4.0
    state[2, 0, 0], state[2, 3, 0] = 0.0, 0.0
    observed = model.h(state).squeeze(-1)
    expected = torch.tensor(
        [[5.0, math.atan2(4, 3)], [5.0, math.atan2(4, -3)], [1e-6, 0.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(observed).all()


def test_range_bearing_jacobian_matches_central_finite_differences():
    args = _params()
    model = ConstantAcceleration2DModel(args)
    state = torch.tensor([[[12.0], [0.2], [0.0], [7.0], [-0.1], [0.0]]])
    analytic = model.H_jacobian(state)[0]
    eps = 1e-3
    numeric = torch.zeros_like(analytic)
    for state_index in (0, 3):
        plus, minus = state.clone(), state.clone()
        plus[0, state_index, 0] += eps
        minus[0, state_index, 0] -= eps
        h_plus = model.h(plus)[0, :, 0]
        h_minus = model.h(minus)[0, :, 0]
        delta = h_plus - h_minus
        delta[1] = wrap_to_pi(delta[1])
        numeric[:, state_index] = delta / (2 * eps)
    torch.testing.assert_close(analytic, numeric, rtol=2e-3, atol=2e-4)


def test_bearing_innovation_wraps_across_zero_two_pi_boundary():
    predicted = torch.tensor([2 * math.pi - 0.005])
    observed = torch.tensor([0.005])
    residual = wrap_to_pi(observed - predicted)
    torch.testing.assert_close(residual, torch.tensor([0.01]), rtol=0, atol=2e-6)
    edge = wrap_to_pi(torch.tensor([-math.pi, math.pi, 3 * math.pi]))
    assert torch.all(edge >= -math.pi) and torch.all(edge < math.pi)


def test_legacy_update_uses_joseph_covariance_and_preserves_psd_symmetry():
    args = _params(batch=2, measurement_type="FULL")
    model = ConstantAcceleration2DModel(args)
    q = torch.eye(6) * 0.01
    r = torch.eye(6) * 0.2
    filt = extended_kf(model, q, r, args)
    filt.reset(torch.zeros(2, 6, 1))
    base = torch.eye(6) * 2.0
    base[0, 1] = base[1, 0] = 0.1
    filt.P = base.unsqueeze(0).repeat(2, 1, 1)
    p_before = filt.P.clone()
    filt.update(torch.ones(2, 6))
    h = model.H.unsqueeze(0).expand(2, -1, -1)
    identity = torch.eye(6).expand(2, -1, -1)
    ikh = identity - filt.K @ h
    expected = ikh @ p_before @ ikh.transpose(1, 2) + filt.K @ filt.R @ filt.K.transpose(1, 2)
    torch.testing.assert_close(filt.P, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(filt.P, filt.P.transpose(1, 2), rtol=0, atol=1e-6)
    assert torch.linalg.eigvalsh(filt.P).min() >= -1e-6


def test_measurement_initializer_uses_two_observations_for_velocity():
    args = _params(batch=8, seq_len=3)
    args.dt = 1.0
    model = ConstantAcceleration2DModel(args)
    torch.manual_seed(31)
    random.seed(31)
    _, observations, initial, _, _ = generate_synthetic_data(model, args, snr_dB=20)
    first_returned = observations[:, :, 0]
    p1_x = first_returned[:, 0] * torch.cos(first_returned[:, 1])
    p1_y = first_returned[:, 0] * torch.sin(first_returned[:, 1])
    expected_vx = torch.clamp((p1_x - initial[:, 0, 0]) / args.dt, -300.0, 300.0)
    expected_vy = torch.clamp((p1_y - initial[:, 3, 0]) / args.dt, -300.0, 300.0)
    torch.testing.assert_close(initial[:, 1, 0], expected_vx, rtol=1e-5, atol=2e-4)
    torch.testing.assert_close(initial[:, 4, 0], expected_vy, rtol=1e-5, atol=2e-4)
    assert torch.count_nonzero(initial[:, 2, 0]) == 0
    assert torch.count_nonzero(initial[:, 5, 0]) == 0


def test_legacy_batch_and_cpu_device_assumptions_are_explicit():
    assert_legacy_ekf_inputs(4, 4, "cpu")
    with pytest.raises(ValueError, match="allocated for batch"):
        assert_legacy_ekf_inputs(3, 4, "cpu")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="requires CPU"):
            assert_legacy_ekf_inputs(4, 4, "cuda")


def test_oracle_qr_passes_actual_nonunit_scaled_covariances_unchanged():
    q_base = torch.eye(6) * 0.1
    r_base = torch.eye(2) * 0.2
    q_scale, r_scale = 2.75, 4.5
    q_generated, r_generated = q_base * q_scale, r_base * r_scale
    q_used, r_used = covariance_for_policy("oracle_QR", q_generated, r_generated)
    torch.testing.assert_close(q_used, q_generated, rtol=0, atol=0)
    torch.testing.assert_close(r_used, r_generated, rtol=0, atol=0)
    assert not torch.equal(q_used, q_generated / q_scale)
    assert not torch.equal(r_used, r_generated / r_scale)


def test_estimated_qr_requires_training_estimates_and_ignores_oracle_values():
    q_actual, r_actual = torch.eye(6) * 0.8, torch.eye(2) * 0.9
    q_train, r_train = torch.eye(6) * 0.03, torch.eye(2) * 0.04
    with pytest.raises(ValueError, match="training-only"):
        covariance_for_policy("estimated_QR", q_actual, r_actual)
    q_used, r_used = covariance_for_policy(
        "estimated_QR", q_actual, r_actual, estimated=(q_train, r_train)
    )
    torch.testing.assert_close(q_used, q_train)
    torch.testing.assert_close(r_used, r_train)


def test_training_only_qr_and_initializer_covariances_include_cross_terms():
    args = _params(batch=96, seq_len=4)
    args.dt = 1.0
    model = ConstantAcceleration2DModel(args)
    torch.manual_seed(97)
    random.seed(97)
    state, observations, initial, _, _ = generate_synthetic_data(model, args, snr_dB=20)
    q_hat, r_hat = estimate_qr_from_training(state, observations, model)
    calibrated = calibrate_first_observation_prior(state, observations, initial, model)
    assert q_hat.shape == (6, 6) and r_hat.shape == (2, 2)
    assert torch.linalg.eigvalsh(q_hat).min() >= -1e-10
    assert torch.linalg.eigvalsh(r_hat).min() >= -1e-10
    p0 = calibrated["P0"]
    assert p0.shape == (6, 6)
    assert calibrated["sample_count"] == state.shape[0]
    assert torch.linalg.eigvalsh(p0).min() >= -1e-10
    assert calibrated["first_observation_state_noise_cross_covariance"].shape == (6, 2)
    assert torch.isfinite(calibrated["prior_error_mean"]).all()


def test_correlated_first_update_stays_symmetric_and_psd():
    args = _params(batch=4)
    model = ConstantAcceleration2DModel(args)
    q = torch.eye(6) * 1e-6
    r = torch.diag(torch.tensor([0.03**2, math.radians(1.0) ** 2]))
    filt = extended_kf(model, q, r, args)
    filt.x = torch.tensor([[[20.0], [0.1], [0.0], [25.0], [-0.1], [0.0]]]).repeat(4, 1, 1)
    filt.P = (torch.eye(6) * 0.02).repeat(4, 1, 1)
    cross = torch.zeros(6, 2)
    cross[0, 0], cross[3, 1] = 1e-4, 1e-5
    measurement = model.h(filt.x).squeeze(-1)
    measurement[:, 1] = (measurement[:, 1] + 2 * math.pi - 0.001) % (2 * math.pi)
    correlated_first_observation_update(filt, measurement, cross)
    assert torch.isfinite(filt.x).all() and torch.isfinite(filt.P).all()
    torch.testing.assert_close(filt.P, filt.P.transpose(1, 2), rtol=0, atol=1e-6)
    assert torch.linalg.eigvalsh(filt.P).min() >= -1e-5
