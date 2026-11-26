from copy import deepcopy

import numpy as np
import torch

from radar_tracking.data.data_generation import generate_synthetic_data


SCALE = 1000.0


def scale_single_target_batch(true_state, observations, init_state, process_noise, measurement_noise):
    true_state = true_state.clone()
    observations = observations.clone()
    init_state = init_state.clone()
    process_noise = process_noise.clone()
    measurement_noise = measurement_noise.clone()

    true_state /= SCALE
    init_state /= SCALE
    process_noise /= (SCALE ** 2)
    observations[:, 0, :] /= SCALE
    measurement_noise[0, 0] /= (SCALE ** 2)

    return true_state, observations, init_state, process_noise, measurement_noise


def rb_to_xy_np(rb_measurements):
    rb_measurements = np.asarray(rb_measurements, dtype=np.float64)
    if rb_measurements.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ranges = rb_measurements[:, 0]
    bearings = rb_measurements[:, 1]
    return np.stack([ranges * np.cos(bearings), ranges * np.sin(bearings)], axis=-1)


def generate_multitarget_scenarios(
    model,
    simparams,
    num_scenes=12,
    num_targets=3,
    seq_len=60,
    detection_prob=0.9,
    clutter_rate=2.0,
    seed=42,
    range_limits=(10.0, 50.0),
    max_birth_step=12,
    min_track_length=18,
):
    rng = np.random.default_rng(seed)
    total_targets = num_scenes * num_targets

    gen_params = deepcopy(simparams)
    gen_params.seq_len = seq_len
    gen_params.dataset_size = total_targets

    true_state, observations, init_state, process_noise, measurement_noise = generate_synthetic_data(
        model,
        gen_params,
        snr_dB=20,
        batch_size=total_targets,
    )
    true_state, observations, init_state, process_noise, measurement_noise = scale_single_target_batch(
        true_state,
        observations,
        init_state,
        process_noise,
        measurement_noise,
    )

    true_state = true_state.cpu().numpy().reshape(num_scenes, num_targets, model.m, seq_len)
    observations = observations.cpu().numpy().reshape(num_scenes, num_targets, model.n, seq_len)
    init_state = init_state.cpu().numpy().reshape(num_scenes, num_targets, model.m, 1)
    process_noise = process_noise.cpu().numpy()
    measurement_noise = measurement_noise.cpu().numpy()

    scenes = []
    for scene_idx in range(num_scenes):
        true_state_scene = true_state[scene_idx]
        observations_scene = observations[scene_idx]
        init_state_scene = init_state[scene_idx]
        alive_mask = np.zeros((num_targets, seq_len), dtype=bool)
        trajectories_xy = np.full((num_targets, 2, seq_len), np.nan, dtype=np.float32)
        birth_steps = np.full(num_targets, fill_value=-1, dtype=np.int64)
        death_steps = np.full(num_targets, fill_value=-1, dtype=np.int64)
        truth_sets = []

        for target_idx in range(num_targets):
            latest_birth = min(max_birth_step, max(0, seq_len - min_track_length))
            birth = int(rng.integers(0, latest_birth + 1))
            death_min = min(seq_len, birth + min_track_length)
            death = int(rng.integers(death_min, seq_len + 1))
            alive_mask[target_idx, birth:death] = True
            birth_steps[target_idx] = birth
            death_steps[target_idx] = death
            trajectories_xy[target_idx, 0, birth:death] = true_state_scene[target_idx, 0, birth:death].astype(np.float32)
            trajectories_xy[target_idx, 1, birth:death] = true_state_scene[target_idx, 3, birth:death].astype(np.float32)

        scene = {
            "true_state": true_state_scene,
            "true_observations": observations_scene,
            "init_state": init_state_scene,
            "alive_mask": alive_mask,
            "birth_steps": birth_steps,
            "death_steps": death_steps,
            "true_trajectories_xy": trajectories_xy,
            "truth_sets": truth_sets,
            "truth_target_ids": [],
            "detections": [],
            "detection_target_ids": [],
            "detection_is_clutter": [],
        }
        for time_idx in range(seq_len):
            detections = []
            truth_positions = []
            truth_target_ids = []
            detection_target_ids = []
            detection_is_clutter = []
            for target_idx in range(num_targets):
                if not alive_mask[target_idx, time_idx]:
                    continue
                truth_positions.append(
                    true_state_scene[target_idx, [0, 3], time_idx].astype(np.float32)
                )
                truth_target_ids.append(target_idx)
                if rng.random() <= detection_prob:
                    detections.append(observations_scene[target_idx, :, time_idx].astype(np.float32))
                    detection_target_ids.append(target_idx)
                    detection_is_clutter.append(False)

            clutter_count = int(rng.poisson(clutter_rate))
            for _ in range(clutter_count):
                clutter_range = rng.uniform(range_limits[0], range_limits[1])
                clutter_bearing = rng.uniform(0.0, 2.0 * np.pi)
                detections.append(np.asarray([clutter_range, clutter_bearing], dtype=np.float32))
                detection_target_ids.append(-1)
                detection_is_clutter.append(True)

            if detections:
                det_array = np.stack(detections, axis=0).astype(np.float32)
                order = rng.permutation(det_array.shape[0])
                det_array = det_array[order]
                det_target_ids = np.asarray(detection_target_ids, dtype=np.int64)[order]
                det_is_clutter = np.asarray(detection_is_clutter, dtype=bool)[order]
            else:
                det_array = np.zeros((0, 2), dtype=np.float32)
                det_target_ids = np.zeros((0,), dtype=np.int64)
                det_is_clutter = np.zeros((0,), dtype=bool)

            if truth_positions:
                truth_array = np.stack(truth_positions, axis=0).astype(np.float32)
            else:
                truth_array = np.zeros((0, 2), dtype=np.float32)

            scene["truth_sets"].append(truth_array)
            scene["truth_target_ids"].append(np.asarray(truth_target_ids, dtype=np.int64))
            scene["detections"].append(det_array)
            scene["detection_target_ids"].append(det_target_ids)
            scene["detection_is_clutter"].append(det_is_clutter)
        scenes.append(scene)

    meta = {
        "num_scenes": num_scenes,
        "num_targets": num_targets,
        "seq_len": seq_len,
        "detection_prob": detection_prob,
        "clutter_rate": clutter_rate,
        "range_limits": list(range_limits),
        "max_birth_step": max_birth_step,
        "min_track_length": min_track_length,
        "measurement_noise": measurement_noise.tolist(),
        "process_noise": process_noise.tolist(),
        "seed": seed,
    }
    return scenes, meta
