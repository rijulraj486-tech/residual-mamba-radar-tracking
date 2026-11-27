"""Read-only descriptive analysis of the saved original T=100 tensors.
Run from the repository root with its project environment. Writes only the CSV
beside this script; it never regenerates data or runs a model.
"""
from pathlib import Path
import csv
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "T100_distribution_statistics.csv"
MANIFEST_PATH = ROOT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json"
TRAIN_VAL_PATH = ROOT / "artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_train_val.pt"
TEST_PATH = ROOT / "artifacts/results/synthetic/fixed_test_comparison_01/shared_test.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256(TRAIN_VAL_PATH) != manifest["fixed_pool_file_sha256"]:
        raise ValueError("T=100 train/validation tensor hash does not match manifest")
    if sha256(TEST_PATH) != manifest["shared_test_sha256"]:
        raise ValueError("T=100 shared test tensor hash does not match manifest")

    # weights_only prevents arbitrary pickle object loading; no model is created.
    train_val = torch.load(TRAIN_VAL_PATH, map_location="cpu", weights_only=True)
    test = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    splits = {
        "train": train_val["train"],
        "validation": train_val["validation"],
        "test": {"state": test["true_state"], "observations": test["observations"]},
    }
    scale = float(manifest["scale"])
    rows = []
    for split, data in splits.items():
        state = data["state"].to(torch.float64) * scale
        observations = data["observations"].to(torch.float64)
        if not torch.isfinite(state).all() or not torch.isfinite(observations).all():
            raise ValueError(f"Non-finite T=100 values in {split}")
        radius = torch.hypot(state[:, 0, :], state[:, 3, :]).reshape(-1)
        speed = torch.hypot(state[:, 1, :], state[:, 4, :]).reshape(-1)
        acceleration = torch.hypot(state[:, 2, :], state[:, 5, :]).reshape(-1)
        range_m = (observations[:, 0, :] * scale).reshape(-1)
        counts = {
            "near_origin_true_samples_lt_1000m": int((radius < 1000.0).sum()),
            "near_radar_samples_lt_1000m": int((range_m < 1000.0).sum()),
            "extreme_range_samples_gt_100000m": int((range_m > 100000.0).sum()),
        }
        quantities = [
            ("position_radius", radius, "m"),
            ("speed", speed, "m/s"),
            ("acceleration_magnitude", acceleration, "m/s^2"),
            ("radar_range", range_m, "m"),
        ]
        for name, values, unit in quantities:
            quantiles = torch.quantile(
                values, torch.tensor([0.0, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64)
            ).tolist()
            rows.append({
                "sequence_length": 100,
                "split": split,
                "trajectory_count": int(state.shape[0]),
                "observations_per_trajectory": int(state.shape[-1]),
                "quantity": name,
                "units": unit,
                "minimum": quantiles[0],
                "p50": quantiles[1],
                "p95": quantiles[2],
                "p99": quantiles[3],
                "maximum": quantiles[4],
                "pooled_observation_count": int(values.numel()),
                **counts,
                "source_tensor": str((TRAIN_VAL_PATH if split in ("train", "validation") else TEST_PATH).relative_to(ROOT)),
                "source_tensor_sha256": sha256(TRAIN_VAL_PATH if split in ("train", "validation") else TEST_PATH),
                "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
