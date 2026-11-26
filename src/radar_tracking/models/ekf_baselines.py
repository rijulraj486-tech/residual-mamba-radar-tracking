"""Utilities for explicitly named synthetic-radar EKF baseline variants.

The historical ``extended_kf`` implementation remains unchanged. This module
contains the new covariance policies and training-only initializer calibration
used by ``calibrated_ekf_oracle``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from radar_tracking.models.ekf import wrap_to_pi


def covariance_for_policy(
    policy: str,
    oracle_q: torch.Tensor,
    oracle_r: torch.Tensor,
    *,
    estimated: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve Q/R without applying a second noise-scale conversion.

    ``oracle_q`` and ``oracle_r`` are the actual covariance matrices that
    generated the observations. ``estimated`` must have been computed from
    training data only; this function intentionally receives no validation or
    test tensors.
    """
    if policy == "oracle_QR":
        return oracle_q.detach().clone(), oracle_r.detach().clone()
    if policy == "estimated_QR":
        if estimated is None:
            raise ValueError("estimated_QR requires training-only covariance estimates")
        return estimated[0].detach().clone(), estimated[1].detach().clone()
    raise ValueError(f"Unknown covariance policy: {policy}")


def _sample_covariance(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Covariance estimation requires at least two row samples")
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.T @ centered / (samples.shape[0] - 1)


def _wrap_bearing_residual(residual: torch.Tensor) -> torch.Tensor:
    result = residual.clone()
    result[..., 1] = wrap_to_pi(result[..., 1])
    return result


def estimate_qr_from_training(
    train_state: torch.Tensor,
    train_observations: torch.Tensor,
    model,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate additive process/measurement covariance from training only.

    Q uses state-transition residuals for adjacent labeled training states;
    R uses measurement residuals against the training ground-truth state. The
    angular residual is wrapped before computing R. Returned matrices use the
    input tensors' state units (km, km/s, km/s²) and measurement units (km,
    radians).
    """
    if train_state.ndim != 3 or train_observations.ndim != 3:
        raise ValueError("Training state/observation tensors must have shape [N,D,T]")
    if train_state.shape[0] != train_observations.shape[0] or train_state.shape[2] != train_observations.shape[2]:
        raise ValueError("Training state and observation sample/time dimensions must match")
    if train_state.shape[2] < 2:
        raise ValueError("At least two training time steps are required to estimate Q")
    if not torch.isfinite(train_state).all() or not torch.isfinite(train_observations).all():
        raise ValueError("Training tensors contain non-finite values")

    dtype = torch.float64
    states = train_state.detach().to(device="cpu", dtype=dtype)
    observations = train_observations.detach().to(device="cpu", dtype=dtype)
    transition = model.F.detach().to(device="cpu", dtype=dtype)
    process_residual = states[:, :, 1:] - transition @ states[:, :, :-1]
    q_hat = _sample_covariance(process_residual.permute(0, 2, 1).reshape(-1, states.shape[1]))

    n, state_dim, seq_len = states.shape
    flat_state = states.permute(0, 2, 1).reshape(n * seq_len, state_dim, 1)
    flat_obs = observations.permute(0, 2, 1).reshape(n * seq_len, observations.shape[1])
    with torch.no_grad():
        predicted_obs = model.h(flat_state).squeeze(-1)
    measurement_residual = _wrap_bearing_residual(flat_obs - predicted_obs)
    r_hat = _sample_covariance(measurement_residual)
    return q_hat, r_hat


def calibrate_first_observation_prior(
    train_state: torch.Tensor,
    train_observations: torch.Tensor,
    train_initial_state: torch.Tensor,
    model,
) -> dict[str, torch.Tensor | int | str]:
    """Calibrate the prior immediately before the first returned observation.

    The simulator's measurement initializer uses two observations to estimate
    velocity, and the second of those is also the first observation returned
    in the sequence. Therefore this calibration uses the training-set error of
    the one-step prior ``F @ initial_state`` against the first labeled training
    state. This captures the actual two-observation initializer and the first
    transition noise together. It also records the state-error/first-measurement
    noise cross-covariance needed for a correlated first update.

    No validation or held-out test tensors are accepted by this function.
    The covariance is centered sample covariance; the empirical prior bias is
    returned for provenance but is not applied to the supplied initial mean.
    """
    if train_state.ndim != 3 or train_observations.ndim != 3 or train_initial_state.ndim != 3:
        raise ValueError("Training tensors must have [N,D,T] / [N,D,1] shapes")
    if train_state.shape[0] != train_observations.shape[0] or train_state.shape[0] != train_initial_state.shape[0]:
        raise ValueError("Training tensors must have matching sample counts")
    if train_state.shape[2] != train_observations.shape[2]:
        raise ValueError("Training state and observation sequence lengths must match")
    if train_state.shape[2] < 1:
        raise ValueError("Training sequences must contain a first observation")
    if not torch.isfinite(train_state).all() or not torch.isfinite(train_observations).all() or not torch.isfinite(train_initial_state).all():
        raise ValueError("Training tensors contain non-finite values")

    dtype = torch.float64
    states = train_state.detach().to(device="cpu", dtype=dtype)
    observations = train_observations.detach().to(device="cpu", dtype=dtype)
    initial = train_initial_state.detach().to(device="cpu", dtype=dtype)
    transition = model.F.detach().to(device="cpu", dtype=dtype)
    prior_mean = transition @ initial
    state_error = states[:, :, 0] - prior_mean[:, :, 0]
    state_covariance = _sample_covariance(state_error)

    with torch.no_grad():
        true_first_state = states[:, :, :1]
        predicted_first_measurement = model.h(true_first_state).squeeze(-1)
    observed_first_measurement = observations[:, :, 0]
    measurement_noise = _wrap_bearing_residual(observed_first_measurement - predicted_first_measurement)
    state_error_centered = state_error - state_error.mean(dim=0, keepdim=True)
    measurement_centered = measurement_noise - measurement_noise.mean(dim=0, keepdim=True)
    state_measurement_cross_covariance = (
        state_error_centered.T @ measurement_centered / (state_error.shape[0] - 1)
    )

    state_covariance = 0.5 * (state_covariance + state_covariance.T)
    min_eigenvalue = torch.linalg.eigvalsh(state_covariance).min()
    if not torch.isfinite(state_covariance).all() or min_eigenvalue < -1e-10:
        raise RuntimeError(f"Calibrated prior covariance is not PSD: min eigenvalue {min_eigenvalue}")
    return {
        "method": "sample covariance of first returned-state residual state[:, :, 0] - F @ initial_state",
        "covariance_location": "prior immediately before the first returned observation, after one transition from the provided initializer",
        "units": "stored-state units squared: km^2, (km/s)^2, (km/s^2)^2 and cross-products",
        "sample_count": int(state_error.shape[0]),
        "P0": state_covariance,
        "prior_error_mean": state_error.mean(dim=0),
        "first_observation_state_noise_cross_covariance": state_measurement_cross_covariance,
        "minimum_eigenvalue": min_eigenvalue,
    }


def assert_legacy_ekf_inputs(
    batch_size: int,
    expected_batch_size: int,
    device: torch.device | str,
) -> None:
    """Make the legacy filter's fixed-size CPU assumptions explicit."""
    if batch_size != expected_batch_size:
        raise ValueError(
            f"legacy extended_kf was allocated for batch {expected_batch_size}, got {batch_size}"
        )
    if torch.device(device).type != "cpu":
        raise ValueError("legacy extended_kf allocates CPU tensors and requires CPU evaluation inputs")


def correlated_first_observation_update(ekf, measurement: torch.Tensor, cross_covariance: torch.Tensor) -> None:
    """EKF first update when prior error correlates with reused initializer data."""
    if measurement.device.type != "cpu":
        raise ValueError("Correlated first update currently follows the legacy CPU filter")
    h_pred = ekf.h(ekf.x).squeeze(-1)
    innovation = measurement - h_pred
    if ekf.meas_type == "RANGE_BEARING":
        innovation[:, 1] = wrap_to_pi(innovation[:, 1])
    innovation = innovation.unsqueeze(-1)
    jacobian = ekf.model.H_jacobian(ekf.x)
    cross = cross_covariance.to(device=ekf.P.device, dtype=ekf.P.dtype).unsqueeze(0)
    pht = ekf.P @ jacobian.transpose(1, 2)
    cross_innovation = pht + cross
    innovation_covariance = (
        jacobian @ ekf.P @ jacobian.transpose(1, 2)
        + ekf.R
        + jacobian @ cross
        + cross.transpose(1, 2) @ jacobian.transpose(1, 2)
    )
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.transpose(1, 2))
    gain = torch.linalg.solve(innovation_covariance, cross_innovation.transpose(1, 2)).transpose(1, 2)
    ekf.x = ekf.x + gain @ innovation
    ekf.P = ekf.P - gain @ innovation_covariance @ gain.transpose(1, 2)
    ekf.P = 0.5 * (ekf.P + ekf.P.transpose(1, 2))
    ekf.K = gain
    ekf.S = innovation_covariance
    ekf.nu = innovation.squeeze(-1)


def training_split_from_pool(payload: Mapping, manifest: Mapping) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return only the fixed training split for covariance calibration."""
    split = payload.get("train")
    if not isinstance(split, Mapping):
        raise ValueError("Fixed pool has no training split")
    expected_hashes = manifest.get("tensor_sha256", {}).get("train", {})
    result = (split.get("state"), split.get("observations"), split.get("initial_state"))
    if not all(torch.is_tensor(value) for value in result):
        raise TypeError("Fixed training split must contain state, observations, and initial_state tensors")
    names = ("state", "observations", "initial_state")
    for name, value in zip(names, result):
        if tuple(value.shape) != tuple(manifest["tensor_shapes"]["train"][name]):
            raise ValueError(f"Unexpected fixed training tensor shape: {name}")
        if tensor_sha256(value) != expected_hashes[name]:
            raise RuntimeError(f"Fixed training tensor hash mismatch: {name}")
    return result


def tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib
    import json

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()
