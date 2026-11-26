"""Training-only process and range-bearing measurement covariance estimates.

The public estimator accepts only training state, observation, and model
objects. It returns raw empirical covariances and a simulator-structured
estimate in the tensors' stored units; it never receives a validation or test
split.
"""

from __future__ import annotations

from typing import Any

import torch

from radar_tracking.models.ekf import wrap_to_pi


def process_residuals(train_state: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
    """Return ``x[t+1] - F x[t]`` as ``[N,T-1,D]`` from stored states."""
    if train_state.ndim != 3:
        raise ValueError("train_state must have shape [N,D,T]")
    if train_state.shape[-1] < 2:
        raise ValueError("At least two states per trajectory are required")
    if not torch.isfinite(train_state).all() or not torch.isfinite(transition).all():
        raise ValueError("State or transition contains non-finite values")
    states = train_state.detach().to(device="cpu", dtype=torch.float64)
    F = transition.detach().to(device="cpu", dtype=torch.float64)
    if F.shape != (states.shape[1], states.shape[1]):
        raise ValueError("Transition matrix shape does not match state dimension")
    residual = states[:, :, 1:] - torch.einsum("ij,njt->nit", F, states[:, :, :-1])
    return residual.permute(0, 2, 1).contiguous()


def measurement_residuals(
    train_observations: torch.Tensor,
    train_state: torch.Tensor,
    model: Any,
) -> torch.Tensor:
    """Return state-conditioned residuals ``z-h(x)`` as ``[N,T,M]``.

    Bearing residuals are wrapped to ``[-pi,pi)`` before any summary or
    covariance is computed. Range remains in the stored range unit.
    """
    if train_state.ndim != 3 or train_observations.ndim != 3:
        raise ValueError("Training states/observations must have shape [N,D,T]")
    if train_state.shape[0] != train_observations.shape[0] or train_state.shape[2] != train_observations.shape[2]:
        raise ValueError("Training state and observation dimensions do not align")
    if train_observations.shape[1] != 2:
        raise ValueError("This estimator expects range-bearing observations")
    if not torch.isfinite(train_state).all() or not torch.isfinite(train_observations).all():
        raise ValueError("Training inputs contain non-finite values")
    states = train_state.detach().to(device="cpu", dtype=torch.float64)
    observations = train_observations.detach().to(device="cpu", dtype=torch.float64)
    n, d, t = states.shape
    flat = states.permute(0, 2, 1).reshape(n * t, d, 1)
    with torch.no_grad():
        expected = model.h(flat).squeeze(-1).reshape(n, t, 2)
    residual = observations.permute(0, 2, 1) - expected
    residual[..., 1] = wrap_to_pi(residual[..., 1])
    return residual.contiguous()


def _mean_and_covariance(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("At least two flattened residual samples are required")
    mean = samples.mean(dim=0)
    centered = samples - mean
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    return mean, 0.5 * (covariance + covariance.T)


def estimate_training_noise(
    train_state: torch.Tensor,
    train_observations: torch.Tensor,
    model: Any,
) -> dict[str, Any]:
    """Estimate Q/R from training residuals and constrain them to simulator form.

    Q raw is the full unbiased covariance of adjacent transition residuals.
    The block-shared estimate averages corresponding 3x3 x/y blocks and sets
    cross-axis blocks to zero. The selected estimate further projects onto the
    simulator's integrated-white-jerk constant-acceleration covariance shape,
    estimating its one common scale by ``tr(A^-1 S)/6``. R raw is the full
    unbiased covariance of state-conditioned observation residuals; the
    selected R is diagonal because the generator independently samples range
    and bearing noise. Biases are reported but not added to covariance.
    """
    if train_state.ndim != 3 or train_observations.ndim != 3:
        raise ValueError("Training tensors must have shape [N,D,T]")
    if train_state.shape[0] != train_observations.shape[0] or train_state.shape[2] != train_observations.shape[2]:
        raise ValueError("Training state and observation tensors do not align")
    n, d, t = train_state.shape
    if d != 6 or train_observations.shape[1] != 2:
        raise ValueError("Expected six-state range-bearing training data")

    w = process_residuals(train_state, model.F)
    w_flat = w.reshape(-1, d)
    w_bias, q_full = _mean_and_covariance(w_flat)

    # Average corresponding within-axis blocks; simulator axes share one Q.
    q_shared_block = torch.zeros_like(q_full)
    block = 0.5 * (q_full[:3, :3] + q_full[3:, 3:])
    q_shared_block[:3, :3] = block
    q_shared_block[3:, 3:] = block

    # Shape of Q_CA_2D(dt, qx=1, qy=1), matching the generator's ordering.
    dt = float(model.dt)
    dtv = torch.tensor(dt, dtype=torch.float64)
    template_1d = torch.stack(
        [torch.stack([dtv**5 / 20, dtv**4 / 8, dtv**3 / 6]),
         torch.stack([dtv**4 / 8, dtv**3 / 3, dtv**2 / 2]),
         torch.stack([dtv**3 / 6, dtv**2 / 2, dtv])]
    )
    template = torch.block_diag(template_1d, template_1d)
    scale = torch.trace(torch.linalg.solve(template, q_full)) / d
    if not torch.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Estimated structured process scale must be positive; got {float(scale)}")
    q_structured = scale * template

    v = measurement_residuals(train_observations, train_state, model)
    v_flat = v.reshape(-1, 2)
    v_bias, r_full = _mean_and_covariance(v_flat)
    r_diagonal = torch.diag(v_flat.var(dim=0, unbiased=True))

    for label, covariance in (("Q_full", q_full), ("Q_shared_block", q_shared_block),
                              ("Q_structured", q_structured), ("R_full", r_full), ("R_diagonal", r_diagonal)):
        eig = torch.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        if not torch.isfinite(covariance).all() or eig.min() < -1e-12:
            raise RuntimeError(f"{label} is not positive semidefinite: minimum eigenvalue {float(eig.min())}")

    return {
        "estimator_name": "estimated_QR_train_only",
        "method": "centered unbiased sample covariance of transition/measurement residuals; selected Q projected onto shared integrated-white-jerk CA covariance; selected R diagonal",
        "training_trajectory_count": int(n),
        "state_observation_count": int(n * t),
        "process_transition_count": int(n * (t - 1)),
        "state_units": "km, km/s, km/s^2; covariances use corresponding products",
        "observation_units": "range km, bearing rad",
        "process_residual_bias": w_bias,
        "process_residual_covariance_full": q_full,
        "process_covariance_shared_block": q_shared_block,
        "process_covariance_structured": q_structured,
        "process_structured_scale": scale,
        "process_covariance_final": q_structured,
        "measurement_residual_bias": v_bias,
        "measurement_residual_covariance_full": r_full,
        "measurement_covariance_diagonal": r_diagonal,
        "measurement_covariance_final": r_diagonal,
        "process_residual_sample_count": int(w_flat.shape[0]),
        "measurement_residual_sample_count": int(v_flat.shape[0]),
        "process_covariance_min_eigenvalue": torch.linalg.eigvalsh(q_structured).min(),
        "measurement_covariance_min_eigenvalue": torch.linalg.eigvalsh(r_diagonal).min(),
    }
