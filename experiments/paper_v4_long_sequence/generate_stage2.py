#!/usr/bin/env python3
"""Generate and audit the fixed-reference Phase 3 T=200/T=400 datasets."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from radar_tracking.data import data_generation
from radar_tracking.data.data_generation import Q_CA_2D
from radar_tracking.data.independent_dataset import (
    SPLITS,
    generate_independent_dataset_bundle,
    load_and_validate_manifest_set,
    sha256_file,
    tensor_sha256,
    validate_split_tensors,
)


OUTPUT_ROOT = PROJECT / "artifacts/results/synthetic/paper_v4_long_sequence"
LENGTHS = (200, 400)
SPLIT_SIZES = {"train": 819, "validation": 204, "test": 1024}
SPLIT_SEEDS = {
    200: {"train": 202611271, "validation": 202611272, "test": 202611273},
    400: {"train": 202611274, "validation": 202611275, "test": 202611276},
}
NEAR_ORIGIN_THRESHOLD_M = 1000.0
EXTREME_RANGE_THRESHOLD_M = 100000.0
QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _verify_sidecar(path: Path) -> str:
    checksum_path = path.with_name(path.name + ".sha256")
    expected = checksum_path.read_text(encoding="ascii").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return actual


def _summary(values: torch.Tensor) -> dict[str, Any]:
    flattened = values.detach().reshape(-1).to(torch.float64).cpu()
    q = torch.quantile(flattened, torch.tensor(QUANTILES, dtype=torch.float64))
    return {
        "count": int(flattened.numel()),
        "mean": float(flattened.mean().item()),
        "sample_std": float(flattened.std(unbiased=True).item()),
        "min": float(flattened.min().item()),
        "quantiles": {
            str(probability): float(value)
            for probability, value in zip(QUANTILES, q.tolist())
        },
        "max": float(flattened.max().item()),
    }


def _distribution_summary(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload["state"] * 1000.0
    observations = payload["observations"]
    position_x, position_y = state[:, 0, :], state[:, 3, :]
    velocity_x, velocity_y = state[:, 1, :], state[:, 4, :]
    acceleration_x, acceleration_y = state[:, 2, :], state[:, 5, :]
    position_radius = torch.sqrt(position_x.square() + position_y.square())
    velocity_magnitude = torch.sqrt(velocity_x.square() + velocity_y.square())
    acceleration_magnitude = torch.sqrt(
        acceleration_x.square() + acceleration_y.square()
    )
    radar_range_m = observations[:, 0, :] * 1000.0
    near_origin = position_radius < NEAR_ORIGIN_THRESHOLD_M
    near_range = radar_range_m < NEAR_ORIGIN_THRESHOLD_M
    extreme_range = radar_range_m > EXTREME_RANGE_THRESHOLD_M

    return {
        "trajectory_count": int(state.shape[0]),
        "observation_count": int(state.shape[0] * state.shape[2]),
        "units": {
            "position": "m",
            "velocity": "m/s",
            "acceleration": "m/s^2",
            "radar_range": "m",
        },
        "position_x": _summary(position_x),
        "position_y": _summary(position_y),
        "position_radius": _summary(position_radius),
        "velocity_x": _summary(velocity_x),
        "velocity_y": _summary(velocity_y),
        "velocity_magnitude": _summary(velocity_magnitude),
        "acceleration_x": _summary(acceleration_x),
        "acceleration_y": _summary(acceleration_y),
        "acceleration_magnitude": _summary(acceleration_magnitude),
        "radar_range": _summary(radar_range_m),
        "threshold_counts": {
            "near_origin_true_position_radius_below_m": NEAR_ORIGIN_THRESHOLD_M,
            "near_origin_true_position_observations": int(near_origin.sum().item()),
            "near_origin_true_position_trajectories": int(near_origin.any(dim=1).sum().item()),
            "near_radar_range_below_m": NEAR_ORIGIN_THRESHOLD_M,
            "near_radar_range_observations": int(near_range.sum().item()),
            "near_radar_range_trajectories": int(near_range.any(dim=1).sum().item()),
            "extreme_radar_range_above_m": EXTREME_RANGE_THRESHOLD_M,
            "extreme_radar_range_observations": int(extreme_range.sum().item()),
            "extreme_radar_range_trajectories": int(extreme_range.any(dim=1).sum().item()),
        },
    }


def _verify_simulation(manifest: dict[str, Any], sequence_length: int) -> None:
    simulation = manifest["simulation"]
    profile = simulation["motion_profile"]
    expected_profile = {
        "pos_min": 20000.0,
        "pos_max": 30000.0,
        "vel_min": -200.0,
        "vel_max": 200.0,
        "accn_min": -3.0,
        "accn_max": 3.0,
        "accn_rms_q": 1.0,
    }
    if simulation["motion_profile_mode"] != "fixed_reference" or profile != expected_profile:
        raise ValueError(f"T={sequence_length}: fixed-reference motion profile mismatch")
    if simulation["dt_seconds"] != 1.0:
        raise ValueError(f"T={sequence_length}: expected dt=1 second")
    if simulation["measurement_noise_distribution"] != "gaussian":
        raise ValueError(f"T={sequence_length}: measurement noise is not Gaussian")
    if simulation["process_noise_distribution"] != "gaussian":
        raise ValueError(f"T={sequence_length}: process noise is not Gaussian")
    if simulation["range_noise_std_m"] != 30.0:
        raise ValueError(f"T={sequence_length}: range noise std changed")
    if not math.isclose(simulation["bearing_noise_std_degrees"], 1.0):
        raise ValueError(f"T={sequence_length}: bearing noise std changed")
    if simulation["q2_scale"] != 1.0 or simulation["r2_scale"] != 1.0:
        raise ValueError(f"T={sequence_length}: Q/R scale changed")
    expected_q = Q_CA_2D(1.0, 1.0, 1.0)
    actual_q = torch.tensor(simulation["process_noise_covariance_physical"])
    expected_r = torch.diag(
        torch.tensor([30.0**2, math.radians(1.0) ** 2], dtype=torch.float32)
    )
    actual_r = torch.tensor(simulation["measurement_noise_covariance_physical"])
    if not torch.allclose(actual_q, expected_q, rtol=1e-6, atol=1e-7):
        raise ValueError(f"T={sequence_length}: process-noise covariance mismatch")
    if not torch.allclose(actual_r, expected_r, rtol=1e-6, atol=1e-7):
        raise ValueError(f"T={sequence_length}: measurement-noise covariance mismatch")


def _validate_bundle(
    sequence_length: int, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_hash = _verify_sidecar(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    simulation = manifest["simulation"]
    if manifest["sequence_length_observations"] != sequence_length:
        raise ValueError(f"wrong length in {manifest_path}")
    if manifest["artifacts"]["generation_config"]["split_seeds"] != SPLIT_SEEDS[
        sequence_length
    ]:
        raise ValueError(f"T={sequence_length}: split seed record mismatch")
    if manifest["artifacts"]["generation_config"]["split_sizes"] != SPLIT_SIZES:
        raise ValueError(f"T={sequence_length}: split count mismatch")
    if manifest["artifacts"]["generation_config"]["motion_profile_mode"] != "fixed_reference":
        raise ValueError(f"T={sequence_length}: wrong generation mode")
    _verify_simulation(manifest, sequence_length)

    config_path = output_dir / manifest["artifacts"]["generation_config_file"]
    config_hash = sha256_file(config_path)
    if config_hash != manifest["artifacts"]["generation_config_file_sha256"]:
        raise ValueError(f"T={sequence_length}: generation config hash mismatch")
    canonical_config_hash = hashlib.sha256(
        json.dumps(
            manifest["artifacts"]["generation_config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if canonical_config_hash != manifest["artifacts"]["generation_config_sha256"]:
        raise ValueError(f"T={sequence_length}: canonical generation config hash mismatch")

    source_hash_checks = {}
    source_paths = {
        "data_generation.py": PROJECT / "src/radar_tracking/data/data_generation.py",
        "constant_acceleration.py": PROJECT / "src/radar_tracking/dynamics/constant_acceleration.py",
        "config.py": PROJECT / "src/radar_tracking/config.py",
        "independent_dataset.py": PROJECT / "src/radar_tracking/data/independent_dataset.py",
    }
    for name, path in source_paths.items():
        actual = sha256_file(path)
        expected = manifest["artifacts"]["source_hashes"][name]
        source_hash_checks[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise ValueError(f"T={sequence_length}: source hash mismatch for {name}")

    distributions = {}
    split_validation = {}
    for split_name in SPLITS:
        split = manifest["splits"][split_name]
        path = output_dir / split["file"]
        data_file_hash = sha256_file(path)
        if data_file_hash != split["file_sha256"]:
            raise ValueError(f"T={sequence_length}/{split_name}: split file hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("state", "observations", "initial_state", "initialization_observations"):
            tensor = payload[key]
            if list(tensor.shape) != split["tensor_shapes"][key]:
                raise ValueError(f"T={sequence_length}/{split_name}: {key} shape mismatch")
            actual_tensor_hash = tensor_sha256(tensor)
            if actual_tensor_hash != split["tensor_sha256"][key]:
                raise ValueError(f"T={sequence_length}/{split_name}: {key} hash mismatch")
        if payload["generation_seed"] != split["generation_seed"]:
            raise ValueError(f"T={sequence_length}/{split_name}: seed mismatch")
        if payload["trajectory_ids"] != split["trajectory_ids"]:
            raise ValueError(f"T={sequence_length}/{split_name}: trajectory ID mismatch")
        validate_split_tensors(
            state=payload["state"],
            observations=payload["observations"],
            initial_state=payload["initial_state"],
            initialization_observations=payload["initialization_observations"],
            trajectory_ids=payload["trajectory_ids"],
            sequence_length=sequence_length,
            dt=simulation["dt_seconds"],
            velocity_clip_mps=simulation["range_bearing_initial_velocity_clip_mps"],
        )
        if payload["state"].shape[0] != SPLIT_SIZES[split_name]:
            raise ValueError(f"T={sequence_length}/{split_name}: count mismatch")
        distributions[split_name] = _distribution_summary(payload)
        split_validation[split_name] = {
            "trajectory_count": len(payload["trajectory_ids"]),
            "state_shape": list(payload["state"].shape),
            "observations_shape": list(payload["observations"].shape),
            "initial_state_shape": list(payload["initial_state"].shape),
            "initialization_observations_shape": list(
                payload["initialization_observations"].shape
            ),
            "all_tensors_finite": True,
            "initialization_aligned": True,
            "split_file_sha256": data_file_hash,
        }

    split_manifest_paths = []
    index_path = output_dir / "manifests/index.json"
    _verify_sidecar(index_path)
    split_index = json.loads(index_path.read_text(encoding="utf-8"))
    if split_index["parent_manifest_sha256"] != manifest_hash:
        raise ValueError(f"T={sequence_length}: split manifest parent hash mismatch")
    for item in split_index["split_manifests"]:
        split_path = output_dir / item["file"]
        split_manifest_hash = _verify_sidecar(split_path)
        if split_manifest_hash != item["sha256"]:
            raise ValueError(f"T={sequence_length}: split manifest index hash mismatch")
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if split_manifest["parent_manifest_sha256"] != manifest_hash:
            raise ValueError(f"T={sequence_length}: split manifest has wrong parent")
        split_manifest_paths.append(split_path)

    return (
        {
            "sequence_length": sequence_length,
            "bundle_manifest": str(manifest_path.relative_to(PROJECT)),
            "bundle_manifest_sha256": manifest_hash,
            "generation_config_file_sha256": config_hash,
            "generation_config_canonical_sha256": canonical_config_hash,
            "generator_source_hashes": source_hash_checks,
            "simulation_verified": True,
            "split_validation": split_validation,
            "split_manifest_paths": [
                str(path.relative_to(PROJECT)) for path in split_manifest_paths
            ],
        },
        distributions,
    )


def _distribution_markdown(distributions: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Reference Sequence-Length Distribution Comparison",
        "",
        "Generated synthetic trajectories only; no model training, inference, or test-performance evaluation.",
        "",
        "Position/velocity/acceleration/range summaries pool all observations in each split. State values are converted back to physical units for this report.",
        "Near-origin threshold: true position radius or measured range below 1,000 m. Extreme-range threshold: measured radar range above 100,000 m.",
        "",
        "| Length | Split | Position radius p50 / p95 / p99 (m) | Speed p50 / p95 / p99 (m/s) | Acceleration magnitude p50 / p95 / p99 (m/s²) | Radar range p50 / p95 / p99 / max (m) | Near-origin true samples | Near radar samples | Extreme-range samples |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for length in LENGTHS:
        for split_name in SPLITS:
            row = distributions[str(length)][split_name]
            pos = row["position_radius"]["quantiles"]
            vel = row["velocity_magnitude"]["quantiles"]
            acc = row["acceleration_magnitude"]["quantiles"]
            radar = row["radar_range"]["quantiles"]
            counts = row["threshold_counts"]
            lines.append(
                "| {length} | {split} | {p50:.1f} / {p95:.1f} / {p99:.1f} | "
                "{v50:.2f} / {v95:.2f} / {v99:.2f} | {a50:.3f} / {a95:.3f} / {a99:.3f} | "
                "{r50:.1f} / {r95:.1f} / {r99:.1f} / {rmax:.1f} | {near_true} | {near_radar} | {extreme} |".format(
                    length=length,
                    split=split_name,
                    p50=pos["0.5"],
                    p95=pos["0.95"],
                    p99=pos["0.99"],
                    v50=vel["0.5"],
                    v95=vel["0.95"],
                    v99=vel["0.99"],
                    a50=acc["0.5"],
                    a95=acc["0.95"],
                    a99=acc["0.99"],
                    r50=radar["0.5"],
                    r95=radar["0.95"],
                    r99=radar["0.99"],
                    rmax=radar["1.0"],
                    near_true=counts["near_origin_true_position_observations"],
                    near_radar=counts["near_radar_range_observations"],
                    extreme=counts["extreme_radar_range_observations"],
                )
            )
    lines.append("")
    lines.append("Full summary statistics and quantiles are in `distribution_comparison.json`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dirs = {
        length: OUTPUT_ROOT / f"T{length}_fixed_profile_v1" for length in LENGTHS
    }
    incomplete = [
        str(path)
        for path in output_dirs.values()
        if path.exists() and not (path / "dataset_manifest.json").is_file()
    ]
    if incomplete:
        raise FileExistsError(
            "Refusing to overwrite incomplete existing bundle directories: "
            + ", ".join(incomplete)
        )

    results = {}
    distributions = {}
    for length in LENGTHS:
        if output_dirs[length].exists():
            print(f"Validating existing T={length} bundle without modification", flush=True)
        else:
            print(
                f"Generating T={length} with seeds {SPLIT_SEEDS[length]} into {output_dirs[length]}",
                flush=True,
            )
            generate_independent_dataset_bundle(
                output_dirs[length],
                sequence_length=length,
                split_seeds=SPLIT_SEEDS[length],
                split_sizes=SPLIT_SIZES,
                device="cpu",
                snr_db=20,
            )
        result, summary = _validate_bundle(length, output_dirs[length])
        results[str(length)] = result
        distributions[str(length)] = summary
        print(f"Validated T={length} bundle", flush=True)

    split_manifest_paths = [
        PROJECT / path
        for length in LENGTHS
        for path in results[str(length)]["split_manifest_paths"]
    ]
    if len(split_manifest_paths) != 6:
        raise ValueError(f"Expected six split manifests, found {len(split_manifest_paths)}")
    all_manifests = load_and_validate_manifest_set(split_manifest_paths)
    if len(all_manifests) != 6:
        raise ValueError("Six-manifest disjointness validation did not load six manifests")

    total_ids = [
        trajectory_id
        for manifest in all_manifests
        for split in manifest["splits"].values()
        for trajectory_id in split["trajectory_ids"]
    ]
    if len(total_ids) != len(set(total_ids)):
        raise ValueError("trajectory IDs overlap across split or sequence length")

    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "dataset generation and provenance validation only",
        "models_loaded": False,
        "training_run": False,
        "inference_run": False,
        "test_performance_evaluated": False,
        "seed_plan": {
            "policy": "distinct deterministic seeds across every split and sequence length",
            "seeds": {str(length): SPLIT_SEEDS[length] for length in LENGTHS},
        },
        "split_sizes": SPLIT_SIZES,
        "stage2_runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "length_results": results,
        "six_split_manifests_validated_together": [
            str(path.relative_to(PROJECT)) for path in split_manifest_paths
        ],
        "six_manifest_trajectory_ids_disjoint": True,
        "unique_trajectory_id_count": len(total_ids),
    }
    report_path = OUTPUT_ROOT / "stage2_consolidated_validation.json"
    report_hash = _write_json(report_path, report)
    (OUTPUT_ROOT / "stage2_consolidated_validation.json.sha256").write_text(
        f"{report_hash}  {report_path.name}\n", encoding="ascii"
    )

    distribution_report = {
        "status": "generated",
        "thresholds": {
            "near_origin_position_or_radar_range_below_m": NEAR_ORIGIN_THRESHOLD_M,
            "extreme_measured_radar_range_above_m": EXTREME_RANGE_THRESHOLD_M,
        },
        "quantiles": list(QUANTILES),
        "lengths": distributions,
    }
    distribution_path = OUTPUT_ROOT / "distribution_comparison.json"
    distribution_hash = _write_json(distribution_path, distribution_report)
    (OUTPUT_ROOT / "distribution_comparison.json.sha256").write_text(
        f"{distribution_hash}  {distribution_path.name}\n", encoding="ascii"
    )
    markdown_path = OUTPUT_ROOT / "distribution_comparison.md"
    markdown_path.write_text(_distribution_markdown(distributions), encoding="utf-8")
    markdown_hash = sha256_file(markdown_path)
    (OUTPUT_ROOT / "distribution_comparison.md.sha256").write_text(
        f"{markdown_hash}  {markdown_path.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "validation_report": str(report_path),
                "validation_report_sha256": report_hash,
                "distribution_json": str(distribution_path),
                "distribution_json_sha256": distribution_hash,
                "distribution_markdown": str(markdown_path),
                "distribution_markdown_sha256": markdown_hash,
                "unique_trajectory_id_count": len(total_ids),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
