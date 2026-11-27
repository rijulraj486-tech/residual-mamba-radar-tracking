"""Auditable independent synthetic dataset generation for Phase 3.

This module is deliberately separate from the historical benchmark harness.
It uses the opt-in fixed-reference motion profile and per-split Torch random
generators, and refuses to overwrite a prior dataset directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radar_tracking import config as radar_config
from radar_tracking.config import simparams_class
from radar_tracking.data import data_generation
from radar_tracking.data.data_generation import generate_synthetic_data, get_motion_profile
from radar_tracking.dynamics import constant_acceleration
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel


DEFAULT_SPLIT_SIZES = {"train": 819, "validation": 204, "test": 1024}
SPLITS = ("train", "validation", "test")
STATE_SCALE = 1000.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_range_bearing_initialization(
    observations: torch.Tensor,
    initial_state: torch.Tensor,
    initialization_observations: torch.Tensor,
    dt: float,
    velocity_clip: float | None,
    atol: float = 2e-5,
) -> None:
    """Check initialization against its retained two-sample observation context."""
    if initialization_observations.shape != (observations.shape[0], 2, 2):
        raise ValueError(
            "initialization_observations must have shape [N, 2, 2], got "
            f"{tuple(initialization_observations.shape)}"
        )
    # The historical generator evolves one extra sample for initialization,
    # derives the initial state from that pair, then drops its first sample.
    # Therefore context sample 1 is the first returned observation.
    if not torch.allclose(
        observations[:, :, 0], initialization_observations[:, :, 1], atol=atol, rtol=1e-5
    ):
        raise ValueError(
            "first returned observation does not match the second initialization-context observation"
        )
    r0, theta0 = initialization_observations[:, 0, 0], initialization_observations[:, 1, 0]
    r1, theta1 = initialization_observations[:, 0, 1], initialization_observations[:, 1, 1]
    px0, py0 = r0 * torch.cos(theta0), r0 * torch.sin(theta0)
    px1, py1 = r1 * torch.cos(theta1), r1 * torch.sin(theta1)
    vx = (px1 - px0) / dt
    vy = (py1 - py0) / dt
    if velocity_clip is not None:
        vx = vx.clamp(-velocity_clip / STATE_SCALE, velocity_clip / STATE_SCALE)
        vy = vy.clamp(-velocity_clip / STATE_SCALE, velocity_clip / STATE_SCALE)

    expected = torch.zeros_like(initial_state)
    expected[:, 0, 0], expected[:, 3, 0] = px0, py0
    expected[:, 1, 0], expected[:, 4, 0] = vx, vy
    if not torch.allclose(initial_state, expected, atol=atol, rtol=1e-5):
        max_error = float((initial_state - expected).abs().max().item())
        raise ValueError(
            "initial_state is not aligned with the first two range-bearing "
            f"observations (max absolute error {max_error:g})"
        )


def validate_split_tensors(
    *,
    state: torch.Tensor,
    observations: torch.Tensor,
    initial_state: torch.Tensor,
    initialization_observations: torch.Tensor,
    trajectory_ids: Sequence[str],
    sequence_length: int,
    dt: float = 1.0,
    velocity_clip_mps: float | None = 300.0,
) -> None:
    """Validate shapes, finite values, ID uniqueness, and init alignment."""
    if state.ndim != 3 or state.shape[1:] != (6, sequence_length):
        raise ValueError(
            f"state must have shape [N, 6, {sequence_length}], got {tuple(state.shape)}"
        )
    count = state.shape[0]
    if observations.shape != (count, 2, sequence_length):
        raise ValueError(
            "observations must have shape "
            f"[{count}, 2, {sequence_length}], got {tuple(observations.shape)}"
        )
    if initial_state.shape != (count, 6, 1):
        raise ValueError(
            f"initial_state must have shape [{count}, 6, 1], "
            f"got {tuple(initial_state.shape)}"
        )
    if len(trajectory_ids) != count:
        raise ValueError(f"expected {count} trajectory IDs, got {len(trajectory_ids)}")
    if len(set(trajectory_ids)) != count:
        raise ValueError("trajectory IDs must be unique within a split")
    for name, tensor in (
        ("state", state),
        ("observations", observations),
        ("initial_state", initial_state),
        ("initialization_observations", initialization_observations),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    _validate_range_bearing_initialization(
        observations,
        initial_state,
        initialization_observations,
        dt=dt,
        velocity_clip=velocity_clip_mps,
    )


def validate_trajectory_id_disjointness(manifests: Sequence[Mapping[str, Any]]) -> None:
    """Raise if IDs overlap within or between supplied dataset manifests."""
    seen: dict[str, str] = {}
    for manifest in manifests:
        dataset_id = manifest["dataset_id"]
        for split_name, split in manifest["splits"].items():
            for trajectory_id in split["trajectory_ids"]:
                prior = seen.get(trajectory_id)
                if prior is not None:
                    raise ValueError(
                        f"trajectory ID {trajectory_id!r} overlaps between "
                        f"{prior} and {dataset_id}/{split_name}"
                    )
                seen[trajectory_id] = f"{dataset_id}/{split_name}"


def _simulation_params(sequence_length: int, dataset_size: int) -> simparams_class:
    params = simparams_class()
    params.seq_len = int(sequence_length)
    params.dataset_size = int(dataset_size)
    params.dt = 1.0
    params.meas_type = "RANGE_BEARING"
    params.init_type = "MEAS"
    params.noise_ref = "FIXED"
    params.range_std = 30.0
    params.bearing_std = 1.0
    params.q2_scale = 1.0
    params.r2_scale = 1.0
    return params


def _simulation_record(params: simparams_class, profile: Mapping[str, float], snr_db: float) -> dict[str, Any]:
    return {
        "motion_model": "2D_discrete_constant_acceleration_with_white_jerk_process_noise",
        "motion_profile_mode": "fixed_reference",
        "motion_profile": dict(profile),
        "measurement_type": params.meas_type,
        "measurement_noise_distribution": data_generation.MEAS_NOISE_DIST,
        "process_noise_distribution": data_generation.PROC_NOISE_DIST,
        "noise_reference": params.noise_ref,
        "snr_db": float(snr_db),
        "range_noise_std_m": float(params.range_std),
        "bearing_noise_std_degrees": float(params.bearing_std),
        "q2_scale": float(params.q2_scale),
        "r2_scale": float(params.r2_scale),
        "dt_seconds": float(params.dt),
        "initialization_type": params.init_type,
        "range_bearing_initial_velocity_from_two_measurements": bool(
            data_generation.RB_INIT_VEL_FROM_TWO_MEAS
        ),
        "range_bearing_initial_velocity_clip_mps": data_generation.RB_INIT_VEL_CLIP,
        "state_scale": STATE_SCALE,
    }


def generate_independent_dataset_bundle(
    output_dir: str | Path,
    *,
    sequence_length: int,
    split_seeds: Mapping[str, int],
    dataset_id: str | None = None,
    split_sizes: Mapping[str, int] | None = None,
    device: str | torch.device = "cpu",
    snr_db: float = 20.0,
) -> dict[str, Any]:
    """Generate, validate, and hash one sequence-length dataset bundle.

    The directory must not exist. Stored tensors use the benchmark's training
    scale (state and range divided by 1,000; bearing left in radians). This
    function is an implementation API only; it does not run on import.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if (
        data_generation.MEAS_NOISE_DIST.lower() != "gaussian"
        or data_generation.PROC_NOISE_DIST.lower() != "gaussian"
    ):
        raise ValueError(
            "Phase 3 fixed-profile bundles require the historical Gaussian "
            "measurement and process noise distributions"
        )
    if set(split_seeds) != set(SPLITS):
        raise ValueError(f"split_seeds must contain exactly {SPLITS}")
    if len({int(split_seeds[name]) for name in SPLITS}) != len(SPLITS):
        raise ValueError("each split must have a distinct generation seed")
    sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    if set(sizes) != set(SPLITS) or any(int(sizes[name]) < 1 for name in SPLITS):
        raise ValueError(f"split_sizes must contain positive counts for {SPLITS}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    dataset_id = dataset_id or f"paper_v4_radar_T{sequence_length}_fixed_profile_v1"
    data_dir = output_dir / "splits"
    data_dir.mkdir()

    split_records: dict[str, Any] = {}
    source_files = {
        "data_generation.py": Path(data_generation.__file__).resolve(),
        "constant_acceleration.py": Path(constant_acceleration.__file__).resolve(),
        "config.py": Path(radar_config.__file__).resolve(),
        "independent_dataset.py": Path(__file__).resolve(),
    }
    profile_params = _simulation_params(sequence_length, 1)
    profile = get_motion_profile(profile_params, mode="fixed_reference")
    simulation = _simulation_record(profile_params, profile, snr_db)

    for split_name in SPLITS:
        count = int(sizes[split_name])
        seed = int(split_seeds[split_name])
        params = _simulation_params(sequence_length, count)
        model = ConstantAcceleration2DModel(params)
        split_generator = torch.Generator(device=device)
        split_generator.manual_seed(seed)
        (
            state,
            observations,
            initial_state,
            process_covariance,
            measurement_covariance,
            initialization_observations,
        ) = (
            generate_synthetic_data(
                model,
                params,
                snr_dB=snr_db,
                batch_size=count,
                motion_profile_mode="fixed_reference",
                generator=split_generator,
                device=device,
                return_initialization_observations=True,
            )
        )

        # Match benchmark tensor units: states and range are in km-derived
        # scaled units; bearing remains radians.
        state = (state / STATE_SCALE).cpu().contiguous()
        observations = observations.cpu().contiguous()
        observations[:, 0, :] /= STATE_SCALE
        initial_state = (initial_state / STATE_SCALE).cpu().contiguous()
        initialization_observations = initialization_observations.cpu().contiguous()
        initialization_observations[:, 0, :] /= STATE_SCALE
        trajectory_ids = [
            f"{dataset_id}/T{sequence_length}/{split_name}/{index:06d}"
            for index in range(count)
        ]
        validate_split_tensors(
            state=state,
            observations=observations,
            initial_state=initial_state,
            initialization_observations=initialization_observations,
            trajectory_ids=trajectory_ids,
            sequence_length=sequence_length,
            dt=params.dt,
            velocity_clip_mps=data_generation.RB_INIT_VEL_CLIP,
        )

        payload = {
            "format_version": 1,
            "dataset_id": dataset_id,
            "sequence_length": sequence_length,
            "split": split_name,
            "generation_seed": seed,
            "state": state,
            "observations": observations,
            "initial_state": initial_state,
            "initialization_observations": initialization_observations,
            "trajectory_ids": trajectory_ids,
        }
        tensor_path = data_dir / f"{split_name}.pt"
        torch.save(payload, tensor_path)
        split_records[split_name] = {
            "trajectory_count": count,
            "trajectory_id_namespace": f"{dataset_id}/T{sequence_length}/{split_name}/",
            "generation_seed": seed,
            "file": str(tensor_path.relative_to(output_dir)),
            "file_sha256": sha256_file(tensor_path),
            "trajectory_ids": trajectory_ids,
            "tensor_shapes": {
                "state": list(state.shape),
                "observations": list(observations.shape),
                "initial_state": list(initial_state.shape),
                "initialization_observations": list(initialization_observations.shape),
            },
            "tensor_sha256": {
                "state": tensor_sha256(state),
                "observations": tensor_sha256(observations),
                "initial_state": tensor_sha256(initial_state),
                "initialization_observations": tensor_sha256(initialization_observations),
            },
        }
        # Record the resolved process and measurement covariances as generated.
        if split_name == "train":
            simulation["process_noise_covariance_physical"] = process_covariance.cpu().tolist()
            simulation["measurement_noise_covariance_physical"] = measurement_covariance.cpu().tolist()

    generation_config = {
        "sequence_length": sequence_length,
        "split_sizes": sizes,
        "split_seeds": {name: int(split_seeds[name]) for name in SPLITS},
        "device": str(device),
        "snr_db": float(snr_db),
        "motion_profile_mode": "fixed_reference",
        "dataset_id": dataset_id,
    }
    config_path = output_dir / "generation_config.json"
    config_path.write_text(
        json.dumps(generation_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset_id": dataset_id,
        "status": "generated_validated",
        "sequence_length_observations": int(sequence_length),
        "state_order": ["x", "vx", "ax", "y", "vy", "ay"],
        "observation_order": ["range_scaled_by_1000", "bearing_rad"],
        "simulation": simulation,
        "splits": split_records,
        "independence": {
            "ids_disjoint_across_splits": True,
            "ids_disjoint_from_other_sequence_lengths_by_length_namespace": True,
            "ids_disjoint_from_historical_100_step_pool_by_dataset_namespace": True,
            "historical_100_step_files_modified": False,
        },
        "artifacts": {
            "generation_config": generation_config,
            "generation_config_file": config_path.name,
            "generation_config_file_sha256": sha256_file(config_path),
            "generation_config_sha256": hashlib.sha256(
                json.dumps(
                    generation_config,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source_hashes": {
                name: sha256_file(path) for name, path in source_files.items()
            },
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "platform": platform.platform(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "combined_dataset_sha256": hashlib.sha256(
                "".join(split_records[name]["file_sha256"] for name in SPLITS).encode("ascii")
            ).hexdigest(),
        },
    }
    validate_trajectory_id_disjointness([manifest])

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # A detached checksum avoids an impossible self-referential manifest hash.
    (output_dir / "dataset_manifest.json.sha256").write_text(
        f"{sha256_file(manifest_path)}  dataset_manifest.json\n", encoding="ascii"
    )

    # Keep one full bundle manifest, plus one separately checksummed manifest
    # per split. The split manifests allow a single disjointness audit across
    # train/validation/test at both sequence lengths.
    split_manifest_dir = output_dir / "manifests"
    split_manifest_dir.mkdir()
    parent_manifest_sha256 = sha256_file(manifest_path)
    split_manifest_index = {
        "dataset_id": dataset_id,
        "parent_manifest": "../dataset_manifest.json",
        "parent_manifest_sha256": parent_manifest_sha256,
        "split_manifests": [],
    }
    for split_name in SPLITS:
        split_manifest = dict(manifest)
        split_manifest["manifest_scope"] = "single_split"
        split_manifest["parent_manifest_sha256"] = parent_manifest_sha256
        split_manifest["splits"] = {split_name: split_records[split_name]}
        split_path = split_manifest_dir / f"{split_name}_manifest.json"
        split_path.write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        split_hash = sha256_file(split_path)
        (split_manifest_dir / f"{split_name}_manifest.json.sha256").write_text(
            f"{split_hash}  {split_path.name}\n", encoding="ascii"
        )
        split_manifest_index["split_manifests"].append(
            {
                "split": split_name,
                "file": str(split_path.relative_to(output_dir)),
                "sha256": split_hash,
            }
        )
    index_path = split_manifest_dir / "index.json"
    index_path.write_text(
        json.dumps(split_manifest_index, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (split_manifest_dir / "index.json.sha256").write_text(
        f"{sha256_file(index_path)}  index.json\n", encoding="ascii"
    )
    return manifest


def load_and_validate_manifest_set(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load manifests, check their detached hashes, and ensure IDs are disjoint."""
    manifests = []
    for path_like in paths:
        path = Path(path_like)
        checksum_path = path.with_name(path.name + ".sha256")
        expected = checksum_path.read_text(encoding="ascii").split()[0]
        if sha256_file(path) != expected:
            raise ValueError(f"manifest checksum mismatch: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_trajectory_id_disjointness([manifest])
        manifests.append(manifest)
    validate_trajectory_id_disjointness(manifests)
    return manifests
