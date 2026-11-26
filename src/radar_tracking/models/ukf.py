"""Scaled unscented Kalman filter for the synthetic six-state radar model.

The filter uses state order ``[x, vx, ax, y, vy, ay]`` and an additive-noise
model. Covariance calculations run in float64 even when the input data is
float32. Cholesky factorization symmetrizes first, rejects materially
indefinite matrices, then applies geometrically increasing diagonal jitter
bounded by ``max_jitter_relative``. A covariance that still fails factorization
raises ``UKFNumericalError``; persistent failures are never hidden.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from radar_tracking.dynamics.constant_acceleration import wrap_to_2pi
from radar_tracking.models.ekf import wrap_to_pi


class UKFNumericalError(RuntimeError):
    """Raised when a covariance cannot be represented safely by the UKF."""


class RadarUKF:
    """Scaled-UT filter for ``ConstantAcceleration2DModel``.

    For n=6, alpha=1, beta=2, kappa=0 gives lambda=0 and c=n+lambda=6.
    The mean weights are Wm[0]=0 and Wm[i]=1/12 for i>0. The covariance
    weights are Wc[0]=2 and Wc[i]=1/12 for i>0. Sigma points are the mean and
    mean ± columns of chol(c P). This positive off-center mean-weight choice
    also makes the circular bearing mean well-defined except for a genuinely
    ambiguous angular distribution, which is rejected.
    """

    def __init__(
        self,
        model,
        process_noise: torch.Tensor,
        measurement_noise: torch.Tensor,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        psd_relative_tolerance: float = 1e-8,
        initial_jitter_relative: float = 1e-12,
        max_jitter_relative: float = 1e-6,
        max_cholesky_attempts: int = 7,
    ) -> None:
        self.model = model
        self.state_dim = int(model.m)
        self.measurement_dim = int(model.n)
        if self.state_dim != 6:
            raise ValueError(f"RadarUKF expects state_dim=6, got {self.state_dim}")
        if alpha <= 0 or not all(torch.isfinite(torch.tensor(v)) for v in (alpha, beta, kappa)):
            raise ValueError("Unscented-transform parameters must be finite and alpha > 0")
        if psd_relative_tolerance < 0 or initial_jitter_relative <= 0 or max_jitter_relative < initial_jitter_relative:
            raise ValueError("Invalid PSD tolerance or jitter bounds")
        if max_cholesky_attempts < 1:
            raise ValueError("max_cholesky_attempts must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.scale = self.state_dim + self.lambda_
        if self.scale <= 0:
            raise ValueError("n + lambda must be positive")
        self.psd_relative_tolerance = float(psd_relative_tolerance)
        self.initial_jitter_relative = float(initial_jitter_relative)
        self.max_jitter_relative = float(max_jitter_relative)
        self.max_cholesky_attempts = int(max_cholesky_attempts)
        self.process_noise = torch.as_tensor(process_noise).detach().clone()
        self.measurement_noise = torch.as_tensor(measurement_noise).detach().clone()
        self.num_sigma_points = 2 * self.state_dim + 1

        wm = torch.full((self.num_sigma_points,), 1.0 / (2.0 * self.scale), dtype=torch.float64)
        wc = wm.clone()
        wm[0] = self.lambda_ / self.scale
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        self.weights_mean = wm
        self.weights_covariance = wc
        self.last_diagnostics: dict[str, float] = {}

    def _noise_for_batch(self, value: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
        noise = value.to(device=device, dtype=torch.float64)
        if noise.ndim == 2:
            return noise.unsqueeze(0).expand(batch, -1, -1)
        if noise.ndim != 3 or noise.shape[0] not in (1, batch):
            raise ValueError(f"Noise covariance must have shape [D,D] or [B,D,D], got {tuple(noise.shape)}")
        if noise.shape[0] == 1:
            noise = noise.expand(batch, -1, -1)
        return noise

    def _validate_covariance(self, covariance: torch.Tensor, label: str) -> torch.Tensor:
        if covariance.ndim != 3 or covariance.shape[-1] != covariance.shape[-2]:
            raise ValueError(f"{label} must have shape [B,D,D]")
        if not torch.isfinite(covariance).all():
            raise UKFNumericalError(f"{label} contains non-finite values")
        symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
        eig = torch.linalg.eigvalsh(symmetric)
        diagonal_scale = symmetric.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).clamp_min(1e-12)
        if torch.any(eig[:, 0] < -self.psd_relative_tolerance * diagonal_scale):
            worst = float(eig[:, 0].min())
            raise UKFNumericalError(f"{label} is materially non-PSD; minimum eigenvalue {worst:.6g}")
        return symmetric

    def _cholesky_with_jitter(self, covariance: torch.Tensor, label: str) -> tuple[torch.Tensor, torch.Tensor]:
        symmetric = self._validate_covariance(covariance, label)
        batch, dim, _ = symmetric.shape
        scale = symmetric.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).clamp_min(1e-12)
        identity = torch.eye(dim, dtype=symmetric.dtype, device=symmetric.device).expand(batch, -1, -1)
        relative = self.initial_jitter_relative
        last_info = None
        while relative <= self.max_jitter_relative * (1.0 + 1e-12):
            jitter = scale * relative
            factor, info = torch.linalg.cholesky_ex(symmetric + jitter[:, None, None] * identity)
            if torch.all(info == 0):
                self.last_diagnostics[f"{label}_jitter_max"] = float(jitter.max())
                return factor, jitter
            last_info = info
            relative *= 10.0
        raise UKFNumericalError(
            f"Cholesky failed for {label} after bounded jitter up to "
            f"{self.max_jitter_relative:g} relative; info={last_info}"
        )

    def sigma_points(self, mean: torch.Tensor, covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[B, 2n+1, n]`` sigma points and the stabilized covariance."""
        if mean.ndim == 3 and mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        if mean.ndim != 2 or mean.shape[1] != self.state_dim:
            raise ValueError(f"Mean must have shape [B,{self.state_dim}] or [B,{self.state_dim},1]")
        if covariance.shape != (mean.shape[0], self.state_dim, self.state_dim):
            raise ValueError("Mean and covariance batch/state dimensions do not match")
        mean = mean.to(dtype=torch.float64)
        covariance = covariance.to(device=mean.device, dtype=torch.float64)
        if not torch.isfinite(mean).all():
            raise UKFNumericalError("State mean contains non-finite values")
        factor, jitter = self._cholesky_with_jitter(covariance, "sigma_covariance")
        stabilized = 0.5 * (covariance + covariance.transpose(-1, -2))
        stabilized = stabilized + jitter[:, None, None] * torch.eye(
            self.state_dim, dtype=mean.dtype, device=mean.device
        )
        offsets = math_sqrt(self.scale) * factor.transpose(-1, -2)
        points = torch.cat(
            [mean[:, None, :], mean[:, None, :] + offsets, mean[:, None, :] - offsets],
            dim=1,
        )
        return points, stabilized

    def weighted_mean(self, values: torch.Tensor, angle_indices: tuple[int, ...] = ()) -> torch.Tensor:
        """Weighted mean over sigma points, using circular means for angles."""
        if values.ndim != 3 or values.shape[1] != self.num_sigma_points:
            raise ValueError("Values must have shape [B,2n+1,D]")
        if not torch.isfinite(values).all():
            raise UKFNumericalError("Sigma-point values contain non-finite values")
        weights = self.weights_mean.to(device=values.device, dtype=values.dtype)
        result = torch.sum(values * weights[None, :, None], dim=1)
        for index in angle_indices:
            sin_mean = torch.sum(torch.sin(values[:, :, index]) * weights[None, :], dim=1)
            cos_mean = torch.sum(torch.cos(values[:, :, index]) * weights[None, :], dim=1)
            resultant = torch.hypot(sin_mean, cos_mean)
            if torch.any(resultant < 1e-12):
                raise UKFNumericalError(f"Circular mean is ambiguous for angle index {index}")
            result[:, index] = wrap_to_2pi(torch.atan2(sin_mean, cos_mean))
        return result

    def _deviations(self, values: torch.Tensor, mean: torch.Tensor, angle_indices: tuple[int, ...]) -> torch.Tensor:
        deviations = values - mean[:, None, :]
        for index in angle_indices:
            deviations[:, :, index] = wrap_to_pi(deviations[:, :, index])
        return deviations

    def measurement_moments(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        *,
        measurement_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        angle_indices: tuple[int, ...] = (1,),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute predicted measurement mean, covariance, and state cross-covariance."""
        points, stabilized_covariance = self.sigma_points(mean, covariance)
        if measurement_function is None:
            batch, count, _ = points.shape
            flat = points.reshape(batch * count, self.state_dim, 1)
            predicted = self.model.h(flat).squeeze(-1).reshape(batch, count, self.measurement_dim)
        else:
            predicted = measurement_function(points)
        if predicted.ndim != 3 or predicted.shape[:2] != points.shape[:2]:
            raise ValueError("Measurement function must return shape [B,2n+1,M]")
        if not torch.isfinite(predicted).all():
            raise UKFNumericalError("Predicted sigma measurements contain non-finite values")

        z_mean = self.weighted_mean(predicted, angle_indices=angle_indices)
        mean_vector = mean.squeeze(-1) if mean.ndim == 3 and mean.shape[-1] == 1 else mean
        dx = points - mean_vector.to(device=points.device, dtype=points.dtype)[:, None, :]
        dz = self._deviations(predicted, z_mean, angle_indices)
        wc = self.weights_covariance.to(device=points.device, dtype=points.dtype)
        pzz = torch.einsum("s,bsi,bsj->bij", wc, dz, dz)
        pxz = torch.einsum("s,bsi,bsj->bij", wc, dx, dz)
        pzz = 0.5 * (pzz + pzz.transpose(-1, -2))
        return z_mean, pzz, pxz, stabilized_covariance

    def predict(self, mean: torch.Tensor, covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate sigma points through the linear constant-acceleration F and add Q."""
        points, _ = self.sigma_points(mean, covariance)
        transition = self.model.F.to(device=points.device, dtype=points.dtype)
        predicted_points = torch.einsum("ij,bsj->bsi", transition, points)
        predicted_mean = self.weighted_mean(predicted_points)
        dx = predicted_points - predicted_mean[:, None, :]
        wc = self.weights_covariance.to(device=points.device, dtype=points.dtype)
        predicted_covariance = torch.einsum("s,bsi,bsj->bij", wc, dx, dx)
        process_noise = self._noise_for_batch(self.process_noise, points.shape[0], points.device)
        predicted_covariance = predicted_covariance + process_noise
        predicted_covariance = self._validate_covariance(predicted_covariance, "predicted_covariance")
        return predicted_mean.unsqueeze(-1), predicted_covariance

    def update(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor,
        *,
        measurement_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        angle_indices: tuple[int, ...] = (1,),
        state_measurement_noise_cross_covariance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update with an optional first-observation prior/noise cross-covariance.

        When C=Cov(state error, measurement noise) is supplied, a statistical
        linearization of h maps C into measurement space. This is used for the
        calibrated initializer's first returned observation, which the
        two-measurement velocity estimate has already consumed.
        """
        if mean.ndim == 2:
            mean_col = mean.unsqueeze(-1)
        elif mean.ndim == 3 and mean.shape[-1] == 1:
            mean_col = mean
        else:
            raise ValueError("Mean must have shape [B,n] or [B,n,1]")
        batch = mean_col.shape[0]
        if measurement.ndim != 2 or measurement.shape[0] != batch:
            raise ValueError("Measurement must have shape [B,M] matching state batch")
        if measurement.shape[1] != self.measurement_dim:
            raise ValueError(f"Expected measurement dimension {self.measurement_dim}")
        if not torch.isfinite(measurement).all():
            raise UKFNumericalError("Measurement contains non-finite values")

        mean_vec = mean_col.squeeze(-1).to(dtype=torch.float64)
        covariance = covariance.to(device=mean_vec.device, dtype=torch.float64)
        measurement = measurement.to(device=mean_vec.device, dtype=torch.float64)
        z_mean, pzz, pxz, stabilized_p = self.measurement_moments(
            mean_vec, covariance, measurement_function=measurement_function, angle_indices=angle_indices
        )
        measurement_noise = self._noise_for_batch(self.measurement_noise, batch, mean_vec.device)
        innovation_covariance = pzz + measurement_noise
        state_innovation_cross_covariance = pxz
        if state_measurement_noise_cross_covariance is not None:
            cross = torch.as_tensor(
                state_measurement_noise_cross_covariance, device=mean_vec.device, dtype=torch.float64
            )
            if cross.ndim == 2:
                cross = cross.unsqueeze(0).expand(batch, -1, -1)
            elif cross.shape[0] == 1:
                cross = cross.expand(batch, -1, -1)
            if cross.shape != (batch, self.state_dim, self.measurement_dim):
                raise ValueError("State/measurement noise cross-covariance has an invalid shape")
            if not torch.isfinite(cross).all():
                raise UKFNumericalError("State/measurement noise cross-covariance is non-finite")

            # Regression of nonlinear h on the state sigma cloud: H_eff^T=P^-1 P_xz.
            h_eff_t = torch.linalg.solve(stabilized_p, pxz)
            h_eff = h_eff_t.transpose(-1, -2)
            h_noise_cross = h_eff @ cross
            innovation_covariance = innovation_covariance + h_noise_cross + h_noise_cross.transpose(-1, -2)
            state_innovation_cross_covariance = pxz + cross

        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.transpose(-1, -2))
        factor, jitter = self._cholesky_with_jitter(innovation_covariance, "innovation_covariance")
        if jitter.max() > 0:
            innovation_covariance = innovation_covariance + torch.diag_embed(jitter[:, None].expand(-1, self.measurement_dim))
        gain = torch.cholesky_solve(
            state_innovation_cross_covariance.transpose(-1, -2), factor
        ).transpose(-1, -2)
        residual = measurement - z_mean
        for index in angle_indices:
            residual[:, index] = wrap_to_pi(residual[:, index])
        posterior_mean = mean_vec + (gain @ residual.unsqueeze(-1)).squeeze(-1)
        posterior_covariance = stabilized_p - gain @ innovation_covariance @ gain.transpose(-1, -2)
        posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.transpose(-1, -2))
        posterior_covariance = self._validate_covariance(posterior_covariance, "posterior_covariance")
        self.last_diagnostics["innovation_jitter_max"] = float(jitter.max())
        self.last_diagnostics["posterior_min_eigenvalue"] = float(
            torch.linalg.eigvalsh(posterior_covariance).min()
        )
        return posterior_mean.unsqueeze(-1), posterior_covariance


def math_sqrt(value: float) -> float:
    # Keep Python's scalar sqrt explicit in the sigma-point convention.
    import math

    return math.sqrt(value)
