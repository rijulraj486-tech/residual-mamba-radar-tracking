from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_tracking.config import simparams_class
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.models.ekf import extended_kf, wrap_to_pi
from radar_tracking.models.ukf import RadarUKF, UKFNumericalError


def _model(batch: int = 3, dt: float = 1.0) -> ConstantAcceleration2DModel:
    args = simparams_class()
    args.dt = dt
    args.dataset_size = batch
    args.meas_type = "RANGE_BEARING"
    return ConstantAcceleration2DModel(args)


def _ukf(batch: int = 3, dt: float = 1.0, device: str = "cpu") -> RadarUKF:
    model = _model(batch, dt)
    q = torch.eye(6, device=device, dtype=torch.float64) * 1e-6
    r = torch.diag(torch.tensor([0.03**2, math.radians(1.0) ** 2], device=device, dtype=torch.float64))
    return RadarUKF(model, q, r)


def _weighted_covariance(ukf: RadarUKF, points: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    deviations = points - mean[:, None, :]
    wc = ukf.weights_covariance.to(points)
    return torch.einsum("s,bsi,bsj->bij", wc, deviations, deviations)


def test_unscented_weights_match_frozen_parameters():
    ukf = _ukf()
    assert (ukf.alpha, ukf.beta, ukf.kappa) == (1.0, 2.0, 0.0)
    assert ukf.lambda_ == 0.0
    assert ukf.scale == 6.0
    assert ukf.weights_mean[0].item() == 0.0
    assert ukf.weights_covariance[0].item() == 2.0
    torch.testing.assert_close(ukf.weights_mean[1:], torch.full((12,), 1 / 12, dtype=torch.float64))
    torch.testing.assert_close(ukf.weights_covariance[1:], torch.full((12,), 1 / 12, dtype=torch.float64))
    torch.testing.assert_close(ukf.weights_mean.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_sigma_points_reconstruct_mean_and_covariance():
    ukf = _ukf(batch=2)
    mean = torch.tensor([[2.0, 0.1, -0.01, 5.0, -0.2, 0.02], [-3.0, 1.2, 0.1, 4.0, -0.4, 0.02]], dtype=torch.float64)
    a = torch.tensor(
        [[1.0, 0.1, 0, 0.2, 0, 0], [0, 0.5, 0.05, 0, 0.1, 0], [0, 0, 0.2, 0, 0, 0.03],
         [0, 0, 0, 1.1, 0.1, 0], [0, 0, 0, 0, 0.6, 0.1], [0, 0, 0, 0, 0, 0.25]], dtype=torch.float64
    )
    covariance = (a @ a.T).unsqueeze(0).repeat(2, 1, 1)
    points, stabilized = ukf.sigma_points(mean, covariance)
    reconstructed_mean = ukf.weighted_mean(points)
    reconstructed_covariance = _weighted_covariance(ukf, points, reconstructed_mean)
    torch.testing.assert_close(reconstructed_mean, mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(reconstructed_covariance, stabilized, rtol=1e-11, atol=1e-12)
    assert points.shape == (2, 13, 6)


def test_linear_state_prediction_matches_ekf_prediction():
    model = _model(batch=2, dt=0.75)
    q = torch.eye(6, dtype=torch.float64) * 2e-5
    r = torch.eye(2, dtype=torch.float64)
    ukf = RadarUKF(model, q, r)
    mean = torch.tensor([[[1.0], [2.0], [0.5], [-2.0], [0.1], [-0.2]], [[4.0], [-1.0], [0.2], [3.0], [0.3], [0.1]]])
    covariance = torch.eye(6).repeat(2, 1, 1) * 0.03

    predicted_mean, predicted_covariance = ukf.predict(mean, covariance)
    transition = model.F.to(torch.float64)
    expected_mean = transition @ mean.to(torch.float64)
    expected_covariance = transition @ covariance.to(torch.float64) @ transition.T + q
    torch.testing.assert_close(predicted_mean, expected_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(predicted_covariance, expected_covariance, rtol=1e-10, atol=1e-12)

    # Also compare with the unchanged legacy EKF predict method.
    args = simparams_class()
    args.dt = 0.75
    args.dataset_size = 2
    args.meas_type = "RANGE_BEARING"
    legacy = extended_kf(model, q.to(torch.float32), r.to(torch.float32), args)
    legacy.x = mean.to(torch.float32).clone()
    legacy.P = covariance.to(torch.float32).clone()
    legacy.predict()
    torch.testing.assert_close(predicted_mean.to(torch.float32), legacy.x, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(predicted_covariance.to(torch.float32), legacy.P, rtol=2e-6, atol=1e-7)


def test_linear_gaussian_update_matches_closed_form_kalman_update():
    ukf = _ukf(batch=2)
    mean = torch.tensor([[0.2, -0.1, 0.03, 0.5, 0.2, 0.01], [-0.5, 0.4, 0.02, -0.1, 0.3, 0.02]], dtype=torch.float64)
    base = torch.tensor(
        [[0.4, 0.01, 0, 0, 0, 0], [0, 0.3, 0.01, 0, 0, 0], [0, 0, 0.2, 0, 0, 0],
         [0.02, 0, 0, 0.5, 0.01, 0], [0, 0.02, 0, 0, 0.25, 0.01], [0, 0, 0, 0, 0, 0.15]], dtype=torch.float64
    )
    p = (base @ base.T).unsqueeze(0).repeat(2, 1, 1)
    h = torch.tensor([[1.0, 0.2, 0, 0, 0, 0], [0, 0, 0, 1.0, -0.1, 0]], dtype=torch.float64)
    measurement_noise = torch.diag(torch.tensor([0.04, 0.09], dtype=torch.float64))
    ukf.measurement_noise = measurement_noise
    measurement = torch.tensor([[0.7, 0.2], [-0.2, -0.4]], dtype=torch.float64)
    callback = lambda points: points @ h.T

    posterior_mean, posterior_covariance = ukf.update(
        mean, p, measurement, measurement_function=callback, angle_indices=()
    )
    s = h @ p @ h.T + measurement_noise
    gain = torch.linalg.solve(s, (p @ h.T).transpose(-1, -2)).transpose(-1, -2)
    expected_mean = mean + (gain @ (measurement - mean @ h.T).unsqueeze(-1)).squeeze(-1)
    expected_covariance = p - gain @ s @ gain.transpose(-1, -2)
    torch.testing.assert_close(posterior_mean.squeeze(-1), expected_mean, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(posterior_covariance, expected_covariance, rtol=1e-9, atol=1e-10)


def test_circular_bearing_mean_and_residual_cross_zero_boundary():
    ukf = _ukf(batch=1)
    values = torch.zeros(1, 13, 2, dtype=torch.float64)
    values[:, :, 0] = 10.0
    values[:, 1:7, 1] = 2 * math.pi - 0.01
    values[:, 7:13, 1] = 0.01
    mean = ukf.weighted_mean(values, angle_indices=(1,))
    assert abs(float(wrap_to_pi(mean[0, 1]))) < 1e-10
    residual = wrap_to_pi(torch.tensor([0.005 - (2 * math.pi - 0.005)], dtype=torch.float64))
    torch.testing.assert_close(residual, torch.tensor([0.01], dtype=torch.float64), rtol=0, atol=1e-12)


def test_nonlinear_range_bearing_measurement_prediction_is_finite_and_psd():
    model = _model(batch=1)
    ukf = _ukf(batch=1)
    mean = torch.tensor([[3.0, 0.2, 0, 4.0, -0.1, 0]], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([1e-4, 1e-5, 1e-6, 2e-4, 1e-5, 1e-6], dtype=torch.float64)).unsqueeze(0)
    z_mean, pzz, pxz, _ = ukf.measurement_moments(mean, covariance)
    assert z_mean.shape == (1, 2)
    assert abs(float(z_mean[0, 0]) - 5.0) < 1e-3
    assert abs(float(wrap_to_pi(z_mean[0, 1] - math.atan2(4.0, 3.0)))) < 1e-4
    assert torch.isfinite(pzz).all() and torch.isfinite(pxz).all()
    assert torch.linalg.eigvalsh(pzz).min() >= -1e-10


def test_cholesky_failure_policy_jitters_singular_psd_but_rejects_bad_covariance(monkeypatch):
    ukf = _ukf(batch=1)
    mean = torch.zeros(1, 6, dtype=torch.float64)
    singular = torch.zeros(1, 6, 6, dtype=torch.float64)
    points, stabilized = ukf.sigma_points(mean, singular)
    assert torch.isfinite(points).all()
    assert ukf.last_diagnostics["sigma_covariance_jitter_max"] > 0
    assert torch.linalg.eigvalsh(stabilized).min() > 0

    indefinite = torch.eye(6, dtype=torch.float64).unsqueeze(0)
    indefinite[0, 0, 0] = -0.5
    with pytest.raises(UKFNumericalError, match="materially non-PSD"):
        ukf.sigma_points(mean, indefinite)

    calls = {"count": 0}

    def always_fail(matrix):
        calls["count"] += 1
        return torch.zeros_like(matrix), torch.ones(matrix.shape[0], dtype=torch.int32, device=matrix.device)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", always_fail)
    with pytest.raises(UKFNumericalError, match="after bounded jitter"):
        ukf.sigma_points(mean, torch.eye(6, dtype=torch.float64).unsqueeze(0))
    assert calls["count"] == ukf.max_cholesky_attempts


def test_nonfinite_state_covariance_or_measurement_is_rejected():
    ukf = _ukf(batch=1)
    mean = torch.zeros(1, 6, dtype=torch.float64)
    p = torch.eye(6, dtype=torch.float64).unsqueeze(0)
    bad_mean = mean.clone()
    bad_mean[0, 0] = math.nan
    with pytest.raises(UKFNumericalError, match="non-finite"):
        ukf.sigma_points(bad_mean, p)
    bad_covariance = p.clone()
    bad_covariance[0, 0, 0] = math.inf
    with pytest.raises(UKFNumericalError, match="non-finite"):
        ukf.sigma_points(mean, bad_covariance)
    with pytest.raises(UKFNumericalError, match="Measurement contains"):
        ukf.update(mean, p, torch.tensor([[math.nan, 0.0]], dtype=torch.float64))


def test_batch_device_and_small_large_covariance_behavior():
    ukf = _ukf(batch=2)
    mean = torch.zeros(2, 6, dtype=torch.float64)
    scales = torch.logspace(-14, 12, 6, dtype=torch.float64)
    covariance = torch.diag(scales).unsqueeze(0).repeat(2, 1, 1)
    points, stabilized = ukf.sigma_points(mean, covariance)
    rec_mean = ukf.weighted_mean(points)
    rec_covariance = _weighted_covariance(ukf, points, rec_mean)
    torch.testing.assert_close(rec_covariance, stabilized, rtol=1e-8, atol=1e-22)
    assert points.shape == (2, 13, 6)
    assert points.device.type == "cpu"

    if torch.cuda.is_available():
        gpu_ukf = _ukf(batch=2, device="cuda")
        gpu_mean, gpu_cov = gpu_ukf.predict(mean.to("cuda"), covariance.to("cuda"))
        assert gpu_mean.device.type == "cuda" and gpu_cov.device.type == "cuda"
        assert torch.isfinite(gpu_mean).all() and torch.isfinite(gpu_cov).all()


def test_covariance_update_remains_symmetric_and_psd_over_sequence():
    ukf = _ukf(batch=2)
    mean = torch.tensor([[20.0, 0.1, 0.0, 25.0, -0.1, 0.0], [22.0, -0.1, 0.0, 24.0, 0.2, 0.0]], dtype=torch.float64)
    covariance = torch.eye(6, dtype=torch.float64).repeat(2, 1, 1) * 0.02
    for _ in range(12):
        mean, covariance = ukf.predict(mean, covariance)
        truth_measurement = ukf.model.h(mean).squeeze(-1)
        measurement = truth_measurement.clone()
        measurement[:, 1] = (measurement[:, 1] + 0.002) % (2 * math.pi)
        mean, covariance = ukf.update(mean, covariance, measurement)
        torch.testing.assert_close(covariance, covariance.transpose(-1, -2), rtol=0, atol=1e-10)
        assert torch.linalg.eigvalsh(covariance).min() >= -1e-8
        assert torch.isfinite(mean).all() and torch.isfinite(covariance).all()
