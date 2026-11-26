"""Focused unit coverage for the fixed-pool harness metric conversion."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


HARNESS_PATH = Path(__file__).resolve().parents[1] / "experiments/five_seed_harness.py"
SPEC = importlib.util.spec_from_file_location("five_seed_harness", HARNESS_PATH)
HARNESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HARNESS)


@pytest.mark.parametrize("dt", [0.25, 2.0])
def test_metrics_convert_scaled_state_to_physical_units_without_dt_division(dt: float) -> None:
    truth = torch.zeros((1, 6, 1), dtype=torch.float64)
    estimate = torch.tensor([[[0.003], [0.006], [0.012], [0.004], [0.008], [0.016]]])

    metrics = HARNESS.calculate_metrics(torch, truth, estimate, dt)

    assert metrics["position_rmse_m"] == pytest.approx(5.0)
    assert metrics["velocity_rmse_mps"] == pytest.approx(10.0)
    assert metrics["acceleration_rmse_mps2"] == pytest.approx(20.0)
    assert metrics["six_state_mse_scaled_units"] == pytest.approx(525e-6 / 6)
    assert metrics["six_state_mse_db"] == pytest.approx(10 * math.log10(525e-6 / 6))


def test_metrics_reject_nonpositive_dt() -> None:
    values = torch.zeros((1, 6, 1))
    with pytest.raises(ValueError, match="dt must be positive"):
        HARNESS.calculate_metrics(torch, values, values, 0)
