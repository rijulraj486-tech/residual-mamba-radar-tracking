from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_tracking.config import simparams_class
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.models.ekf import extended_kf
from radar_tracking.models.noise_estimation import (
    estimate_training_noise,
    measurement_residuals,
    process_residuals,
)
from radar_tracking.models.ukf import RadarUKF


def _model(dt: float = 1.0):
    args = simparams_class()
    args.dt = dt
    args.dataset_size = 4
    args.meas_type = "RANGE_BEARING"
    return args, ConstantAcceleration2DModel(args)


def _training_data(n: int = 8, length: int = 12):
    args, model = _model()
    generator = torch.Generator().manual_seed(938)
    state = torch.zeros(n, 6, length, dtype=torch.float64)
    state[:, 0, :] = 20.0 + torch.randn(n, length, generator=generator, dtype=torch.float64) * 0.1
    state[:, 1, :] = torch.randn(n, length, generator=generator, dtype=torch.float64) * 0.01
    state[:, 2, :] = torch.randn(n, length, generator=generator, dtype=torch.float64) * 0.001
    state[:, 3, :] = 25.0 + torch.randn(n, length, generator=generator, dtype=torch.float64) * 0.1
    state[:, 4, :] = torch.randn(n, length, generator=generator, dtype=torch.float64) * 0.01
    state[:, 5, :] = torch.randn(n, length, generator=generator, dtype=torch.float64) * 0.001
    flat = state.permute(0, 2, 1).reshape(n * length, 6, 1)
    obs = model.h(flat).reshape(n, length, 2).permute(0, 2, 1).contiguous()
    obs[:, 0, :] += 0.03
    obs[:, 1, :] += math.radians(0.25)
    return args, model, state, obs


def test_process_residual_calculation_uses_x_next_minus_f_x():
    args, model, state, _ = _training_data(n=2, length=3)
    residual = process_residuals(state, model.F)
    expected = (state[:, :, 1:] - model.F.to(torch.float64) @ state[:, :, :-1]).permute(0, 2, 1)
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    assert residual.shape == (2, 2, 6)


def test_measurement_residual_wraps_bearing_and_preserves_stored_units():
    args, model = _model()
    state = torch.zeros(1, 6, 1, dtype=torch.float64)
    state[:, 0, 0] = 1.0  # 1 km in the stored state convention.
    observations = torch.tensor([[[1.05], [2 * math.pi - 0.01]]], dtype=torch.float64)
    residual = measurement_residuals(observations, state, model)
    torch.testing.assert_close(residual[0, 0, 0], torch.tensor(0.05, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(residual[0, 0, 1], torch.tensor(-0.01, dtype=torch.float64), atol=1e-12, rtol=0)
    assert residual.shape == (1, 1, 2)


def test_estimator_is_deterministic_psd_and_returns_simulator_structured_qr():
    args, model, state, obs = _training_data()
    first = estimate_training_noise(state, obs, model)
    second = estimate_training_noise(state.clone(), obs.clone(), model)
    assert first["estimator_name"] == "estimated_QR_train_only"
    assert first["training_trajectory_count"] == 8
    assert first["process_transition_count"] == 8 * 11
    assert first["measurement_residual_sample_count"] == 8 * 12
    for key in ("process_residual_covariance_full", "process_covariance_shared_block", "process_covariance_final",
                "measurement_residual_covariance_full", "measurement_covariance_final"):
        # Also proves repeated estimation is bitwise stable on deterministic input.
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        assert torch.linalg.eigvalsh(first[key]).min() >= -1e-12
    assert torch.count_nonzero(first["process_covariance_final"][:3, 3:]) == 0
    assert torch.count_nonzero(first["measurement_covariance_final"] - torch.diag(torch.diag(first["measurement_covariance_final"]))) == 0
    assert abs(float(first["measurement_residual_bias"][0]) - 0.03) < 1e-12
    assert abs(float(first["measurement_residual_bias"][1]) - math.radians(0.25)) < 1e-12


def test_estimator_api_has_no_validation_or_test_inputs():
    parameters = inspect.signature(estimate_training_noise).parameters
    assert tuple(parameters) == ("train_state", "train_observations", "model")
    _, model, state, obs = _training_data(n=3, length=4)
    with pytest.raises(TypeError):
        estimate_training_noise(state, obs, model, validation_state=state)  # type: ignore[call-arg]


def test_covariance_policy_output_is_accepted_by_ekf_and_ukf():
    args, model, state, obs = _training_data(n=8, length=12)
    args.dataset_size = state.shape[0]
    estimated = estimate_training_noise(state, obs, model)
    q = estimated["process_covariance_final"].to(torch.float32)
    r = estimated["measurement_covariance_final"].to(torch.float32)
    ekf = extended_kf(model, q, r, args)
    assert ekf.Q.shape[-2:] == (6, 6) and ekf.R.shape[-2:] == (2, 2)
    ukf = RadarUKF(model, q, r)
    mean = state[:2, :, 0].to(torch.float64)
    covariance = torch.eye(6, dtype=torch.float64).expand(2, -1, -1).clone() * 0.01
    predicted_mean, predicted_covariance = ukf.predict(mean, covariance)
    assert predicted_mean.shape == (2, 6, 1)
    assert predicted_covariance.shape == (2, 6, 6)
    assert torch.isfinite(predicted_covariance).all()
