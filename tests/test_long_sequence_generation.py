from __future__ import annotations

import torch
import pytest

from radar_tracking.config import simparams_class
from radar_tracking.data.data_generation import (
    generate_synthetic_data,
    get_motion_profile,
)
from radar_tracking.data.independent_dataset import (
    generate_independent_dataset_bundle,
    load_and_validate_manifest_set,
    validate_split_tensors,
    validate_trajectory_id_disjointness,
)
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel


def make_params(sequence_length: int, dataset_size: int = 2) -> simparams_class:
    params = simparams_class()
    params.seq_len = sequence_length
    params.dataset_size = dataset_size
    params.dt = 1.0
    params.meas_type = "RANGE_BEARING"
    params.init_type = "MEAS"
    params.noise_ref = "FIXED"
    params.range_std = 30.0
    params.bearing_std = 1.0
    params.q2_scale = 1.0
    params.r2_scale = 1.0
    return params


def test_default_generator_matches_explicit_historical_profile_mode() -> None:
    params = make_params(4)
    model = ConstantAcceleration2DModel(params)

    torch.manual_seed(991)
    default_result = generate_synthetic_data(
        model, params, snr_dB=20, batch_size=2, device="cpu"
    )
    torch.manual_seed(991)
    explicit_result = generate_synthetic_data(
        model,
        params,
        snr_dB=20,
        batch_size=2,
        motion_profile_mode="sequence_scaled",
        device="cpu",
    )

    for default_tensor, explicit_tensor in zip(default_result, explicit_result):
        assert torch.equal(default_tensor, explicit_tensor)

    # Preserve the former sequence-dependent scaling rule as well.
    assert get_motion_profile(make_params(100)) == {
        "pos_min": 20000.0,
        "pos_max": 30000.0,
        "vel_min": -200.0,
        "vel_max": 200.0,
        "accn_min": -3.0,
        "accn_max": 3.0,
        "accn_rms_q": 1.0,
    }
    profile_200 = get_motion_profile(make_params(200))
    assert profile_200["vel_min"] == -100.0
    assert profile_200["accn_min"] == -0.75
    assert profile_200["accn_rms_q"] == 0.25


def test_fixed_reference_profile_and_noise_covariances_are_length_invariant() -> None:
    fixed_profiles = [
        get_motion_profile(make_params(length), mode="fixed_reference")
        for length in (100, 200, 400)
    ]
    assert fixed_profiles[0] == fixed_profiles[1] == fixed_profiles[2]
    assert fixed_profiles[0] == {
        "pos_min": 20000.0,
        "pos_max": 30000.0,
        "vel_min": -200.0,
        "vel_max": 200.0,
        "accn_min": -3.0,
        "accn_max": 3.0,
        "accn_rms_q": 1.0,
    }

    covariance_pairs = []
    for length in (100, 200, 400):
        params = make_params(length)
        model = ConstantAcceleration2DModel(params)
        generator = torch.Generator(device="cpu").manual_seed(1000 + length)
        state, observations, init_state, process_covariance, measurement_covariance = (
            generate_synthetic_data(
                model,
                params,
                snr_dB=20,
                batch_size=2,
                motion_profile_mode="fixed_reference",
                generator=generator,
                device="cpu",
            )
        )
        assert state.shape == (2, 6, length)
        assert observations.shape == (2, 2, length)
        assert init_state.shape == (2, 6, 1)
        assert torch.isfinite(state).all()
        assert torch.isfinite(observations).all()
        assert torch.isfinite(init_state).all()
        covariance_pairs.append((process_covariance, measurement_covariance))

    for process_covariance, measurement_covariance in covariance_pairs[1:]:
        assert torch.equal(process_covariance, covariance_pairs[0][0])
        assert torch.equal(measurement_covariance, covariance_pairs[0][1])

    params = make_params(7)
    model = ConstantAcceleration2DModel(params)
    seeded_results = []
    for _ in range(2):
        generator = torch.Generator(device="cpu").manual_seed(777)
        seeded_results.append(
            generate_synthetic_data(
                model,
                params,
                snr_dB=20,
                motion_profile_mode="fixed_reference",
                generator=generator,
                device="cpu",
            )
        )
    for first, second in zip(*seeded_results):
        assert torch.equal(first, second)


def test_split_validation_checks_shapes_finiteness_alignment_and_ids() -> None:
    params = make_params(5, dataset_size=3)
    model = ConstantAcceleration2DModel(params)
    generator = torch.Generator(device="cpu").manual_seed(12)
    state, observations, init_state, _, _, init_observations = generate_synthetic_data(
        model,
        params,
        snr_dB=20,
        motion_profile_mode="fixed_reference",
        generator=generator,
        device="cpu",
        return_initialization_observations=True,
    )
    state /= 1000.0
    observations[:, 0, :] /= 1000.0
    init_state /= 1000.0
    init_observations[:, 0, :] /= 1000.0
    ids = [f"dataset/T5/train/{index:06d}" for index in range(3)]
    validate_split_tensors(
        state=state,
        observations=observations,
        initial_state=init_state,
        initialization_observations=init_observations,
        trajectory_ids=ids,
        sequence_length=5,
    )

    bad_init = init_state.clone()
    bad_init[:, 0, 0] += 0.1
    with pytest.raises(ValueError, match="not aligned"):
        validate_split_tensors(
            state=state,
            observations=observations,
            initial_state=bad_init,
            initialization_observations=init_observations,
            trajectory_ids=ids,
            sequence_length=5,
        )
    with pytest.raises(ValueError, match="unique"):
        validate_split_tensors(
            state=state,
            observations=observations,
            initial_state=init_state,
            initialization_observations=init_observations,
            trajectory_ids=[ids[0], ids[0], ids[2]],
            sequence_length=5,
        )
    with pytest.raises(ValueError, match="non-finite"):
        invalid_state = state.clone()
        invalid_state[0, 0, 0] = float("nan")
        validate_split_tensors(
            state=invalid_state,
            observations=observations,
            initial_state=init_state,
            initialization_observations=init_observations,
            trajectory_ids=ids,
            sequence_length=5,
        )


def test_manifest_id_validation_rejects_split_and_length_collisions() -> None:
    manifest_a = {
        "dataset_id": "phase3",
        "splits": {"train": {"trajectory_ids": ["phase3/T200/train/000001"]}},
    }
    manifest_b = {
        "dataset_id": "phase3",
        "splits": {"test": {"trajectory_ids": ["phase3/T400/test/000001"]}},
    }
    validate_trajectory_id_disjointness([manifest_a, manifest_b])
    manifest_c = {
        "dataset_id": "phase3_other",
        "splits": {"train": {"trajectory_ids": ["phase3/T200/train/000001"]}},
    }
    with pytest.raises(ValueError, match="overlaps"):
        validate_trajectory_id_disjointness([manifest_a, manifest_c])


def test_bundle_writer_records_hashes_shapes_seeds_and_disjoint_ids(tmp_path) -> None:
    manifest = generate_independent_dataset_bundle(
        tmp_path / "tiny_fixture",
        sequence_length=3,
        split_sizes={"train": 1, "validation": 1, "test": 1},
        split_seeds={"train": 101, "validation": 102, "test": 103},
        device="cpu",
    )
    assert manifest["status"] == "generated_validated"
    assert manifest["simulation"]["motion_profile_mode"] == "fixed_reference"
    assert manifest["simulation"]["dt_seconds"] == 1.0
    assert manifest["artifacts"]["source_hashes"]
    assert manifest["artifacts"]["combined_dataset_sha256"]
    assert manifest["artifacts"]["generation_config_file_sha256"]
    assert manifest["independence"]["ids_disjoint_across_splits"]
    assert all(
        row["file_sha256"] and row["tensor_sha256"]
        for row in manifest["splits"].values()
    )
    split_manifest_paths = [
        tmp_path / "tiny_fixture" / "manifests" / f"{split}_manifest.json"
        for split in ("train", "validation", "test")
    ]
    loaded = load_and_validate_manifest_set(split_manifest_paths)
    assert len(loaded) == 3
    assert all(row["dataset_id"] == manifest["dataset_id"] for row in loaded)
