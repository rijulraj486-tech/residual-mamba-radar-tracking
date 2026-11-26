#!/usr/bin/env python3

"""Preprocess raw NCLT sessions into a KalmanNet-ready sensor-fusion dataset.

The output format follows the practical TBPTT KalmanNet workflow used for the
Michigan NCLT evaluation in Jian Song et al., IEEE SPL 2024.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from tqdm import tqdm


DEFAULT_TRAIN_DATES = [
    "2012-01-08",
    "2012-01-15",
    "2012-01-22",
    "2012-02-02",
    "2012-02-04",
]
DEFAULT_VAL_DATES = ["2012-02-05"]
DEFAULT_TEST_DATES = ["2012-02-12"]


def calculate_hz(sensor_name: str, timestamps: np.ndarray) -> None:
    length = timestamps[-1] - timestamps[0]
    average_timestep = length / len(timestamps)
    hz = 1.0 / average_timestep
    print(f"{sensor_name} data, Hz: {hz:.3f}")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(np.float64, copy=False)
    values = values.astype(np.float64, copy=False)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    starts = np.maximum(np.arange(len(values)) - window + 1, 0)
    sums = cumulative[np.arange(1, len(values) + 1)] - cumulative[starts]
    return sums / counts


def nearest_rows(reference_times: np.ndarray, source_times: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    insert_idx = np.searchsorted(source_times, reference_times, side="left")
    insert_idx = np.clip(insert_idx, 0, len(source_times) - 1)
    prev_idx = np.clip(insert_idx - 1, 0, len(source_times) - 1)
    next_idx = insert_idx
    choose_prev = np.abs(reference_times - source_times[prev_idx]) <= np.abs(source_times[next_idx] - reference_times)
    chosen_idx = np.where(choose_prev, prev_idx, next_idx)
    return source_values[chosen_idx]


def _compute_gps_conversion_params() -> tuple[float, float, float, float]:
    lat_0 = 0.738167915410646
    lon_0 = -1.46098650670922
    re = 6378135.0
    rp = 6356750.0
    r_ns = (re * rp) ** 2 / (((re * np.cos(lat_0)) ** 2 + (rp * np.sin(lat_0)) ** 2) ** 1.5)
    r_ew = re ** 2 / np.sqrt((re * np.cos(lat_0)) ** 2 + (rp * np.sin(lat_0)) ** 2)
    return r_ns, r_ew, lat_0, lon_0


def gps_to_local_coord(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_ns, r_ew, lat_0, lon_0 = _compute_gps_conversion_params()
    x = np.sin(lat - lat_0) * r_ns
    y = np.sin(lon - lon_0) * r_ew * np.cos(lat_0)
    return x, y


def read_gps(data_dir: str, data_date: str, use_rtk: bool = False) -> np.ndarray:
    filename = "gps_rtk.csv" if use_rtk else "gps.csv"
    filepath = osp.join(data_dir, "sensor_data", data_date, filename)
    gps = np.loadtxt(filepath, delimiter=",")
    gps = np.delete(gps, np.where(gps[:, 1] < 3)[0], axis=0)
    t = gps[:, 0] / 1_000_000.0
    lat = gps[:, 3]
    lon = gps[:, 4]
    calculate_hz("GPS RTK" if use_rtk else "GPS", t)
    x, y = gps_to_local_coord(lat, lon)
    x = x + 76.50582406697139
    y = y + 108.31373031919006
    gps_data = np.vstack((t, x, y)).T
    gps_data = np.delete(gps_data, np.where((gps_data[:, 1] < -350) | (gps_data[:, 1] > 120))[0], axis=0)
    gps_data = np.delete(gps_data, np.where((gps_data[:, 2] < -750) | (gps_data[:, 2] > 150))[0], axis=0)
    return gps_data


def read_imu(data_dir: str, data_date: str) -> np.ndarray:
    filepath = osp.join(data_dir, "sensor_data", data_date, "ms25.csv")
    ms25 = np.loadtxt(filepath, delimiter=",")
    accel_x = rolling_mean(ms25[:, 4], window=50)
    accel_y = rolling_mean(ms25[:, 5], window=50)
    omega = ms25[:, 9]
    t = ms25[:, 0] / 1_000_000.0
    calculate_hz("IMU Accel and Omega", t)
    return np.vstack((t, accel_x, accel_y, omega)).T


def read_euler(data_dir: str, data_date: str) -> np.ndarray:
    filepath = osp.join(data_dir, "sensor_data", data_date, "ms25_euler.csv")
    euler = np.loadtxt(filepath, delimiter=",")
    t = euler[:, 0] / 1_000_000.0
    heading = euler[:, 3]
    calculate_hz("IMU Euler", t)
    return np.vstack((t, heading)).T


def read_wheels(data_dir: str, data_date: str) -> np.ndarray:
    filepath = osp.join(data_dir, "sensor_data", data_date, "wheels.csv")
    wheel_vel = np.loadtxt(filepath, delimiter=",")
    t = wheel_vel[:, 0] / 1_000_000.0
    robot_vel = 0.5 * (wheel_vel[:, 1] + wheel_vel[:, 2])
    left = wheel_vel[:, 1]
    right = wheel_vel[:, 2]
    calculate_hz("Wheel Odometry", t)
    return np.vstack((t, robot_vel, left, right)).T


def read_ground_truth(data_dir: str, data_date: str, truncation: int = -1) -> np.ndarray:
    filepath_cov = osp.join(data_dir, "sensor_data", data_date, "odometry_cov_100hz.csv")
    filepath_gt = osp.join(data_dir, "ground_truth", f"groundtruth_{data_date}.csv")
    # Some NCLT ground-truth files contain malformed rows; keep only valid numeric entries.
    gt = np.genfromtxt(filepath_gt, delimiter=",", invalid_raise=False)
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    if gt.shape[1] < 7:
        raise ValueError(f"Ground-truth file has unexpected column count: {filepath_gt}")
    gt = gt[:, :7]
    gt = gt[np.all(np.isfinite(gt), axis=1)]
    cov = np.loadtxt(filepath_cov, delimiter=",")
    gt = gt[2:truncation, :]
    cov = cov[:truncation, :]
    t = cov[:, 0]
    pose_gt = nearest_rows(t, gt[:, 0], gt[:, 1:])
    t = t / 1_000_000.0
    x = pose_gt[:, 0]
    y = pose_gt[:, 1]
    yaw = pose_gt[:, 5]
    calculate_hz("Ground Truth", t)
    return np.vstack((t, x, y, yaw)).T


def find_nearest_index(array: np.ndarray, time_value: float) -> int:
    diff_arr = array - time_value
    return np.where(diff_arr <= 0, diff_arr, -np.inf).argmax()


def preprocess_session(data_dir: str, data_date: str, use_rtk: bool = False) -> dict[str, torch.Tensor | str]:
    ground_truth = read_ground_truth(data_dir, data_date)
    gps_data = read_gps(data_dir, data_date, use_rtk=use_rtk)
    imu_data = read_imu(data_dir, data_date)
    euler_data = read_euler(data_dir, data_date)
    wheel_data = read_wheels(data_dir, data_date)

    x_true = ground_truth[:, 1]
    y_true = ground_truth[:, 2]
    theta_true = ground_truth[:, 3]
    true_times = ground_truth[:, 0]

    dt = 1.0
    t = np.arange(ground_truth[0, 0], ground_truth[-1, 0], dt)

    a_x = imu_data[:, 1]
    a_y = imu_data[:, 2]
    omega = imu_data[:, 3]
    theta_imu = euler_data[:, 1]

    gps_x = gps_data[:, 1]
    gps_y = gps_data[:, 2]
    v_left_wheel = wheel_data[:, 2]
    v_right_wheel = wheel_data[:, 3]

    gps_times = gps_data[:, 0]
    imu_times = imu_data[:, 0]
    euler_times = euler_data[:, 0]
    wheel_times = wheel_data[:, 0]

    wheel_counter = 0
    gps_counter = 0
    imu_counter = 0
    ground_truth_counter = 0
    euler_counter = 0
    prev_gps_counter = -1
    prev_wheel_counter = -1

    imu_sensor = []
    gps = []
    filtered_gps = []
    gt = []
    theta_gt = []
    filtered_wheel = []
    wheel = []

    initial_state = np.array([x_true[0], y_true[0], 0.0, 0.0, theta_true[0], 0.0], dtype=np.float64)

    for step in range(1, len(t)):
        ax = a_x[imu_counter]
        ay = a_y[imu_counter]
        theta = theta_imu[euler_counter]
        omega_imu = omega[imu_counter]
        imu_sensor.append(torch.tensor([ax, ay, theta, omega_imu], dtype=torch.float64))
        gps.append(torch.tensor([gps_x[gps_counter], gps_y[gps_counter]], dtype=torch.float64))
        wheel.append(
            torch.tensor([v_left_wheel[wheel_counter], v_right_wheel[wheel_counter], theta, omega_imu], dtype=torch.float64)
        )

        imu_counter = find_nearest_index(imu_times, t[step])
        euler_counter = find_nearest_index(euler_times, t[step])
        ground_truth_counter = find_nearest_index(true_times, t[step])
        gps_counter = find_nearest_index(gps_times, t[step])
        wheel_counter = find_nearest_index(wheel_times, t[step])

        if gps_counter != prev_gps_counter:
            filtered_gps.append(torch.tensor([gps_x[gps_counter], gps_y[gps_counter]], dtype=torch.float64))
            prev_gps_counter = gps_counter
        else:
            filtered_gps.append(torch.tensor([torch.nan, torch.nan], dtype=torch.float64))

        if wheel_counter != prev_wheel_counter:
            filtered_wheel.append(torch.tensor([v_left_wheel[wheel_counter], v_right_wheel[wheel_counter]], dtype=torch.float64))
            prev_wheel_counter = wheel_counter
        else:
            filtered_wheel.append(torch.tensor([torch.nan, torch.nan], dtype=torch.float64))

        gt.append(torch.tensor([x_true[ground_truth_counter], y_true[ground_truth_counter]], dtype=torch.float64))
        theta_gt.append(torch.tensor([theta_true[ground_truth_counter]], dtype=torch.float64))

    return {
        "filtered_gps": torch.vstack(filtered_gps),
        "imu": torch.vstack(imu_sensor),
        "filtered_wheel": torch.vstack(filtered_wheel),
        "theta_gt": torch.vstack(theta_gt),
        "ground_truth": torch.vstack(gt),
        "initial_state": torch.from_numpy(initial_state),
        "gps": torch.vstack(gps),
        "wheel": torch.vstack(wheel),
        "description": "filtered data for correction, else for input.",
        "data_date": data_date,
    }


def split_processed_sessions(processed: list[dict], random_split_enabled: bool, random_seed: int):
    if random_split_enabled:
        generator = torch.Generator().manual_seed(random_seed)
        return random_split(processed, [0.8, 0.1, 0.1], generator=generator)

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for item in processed:
        date = item["data_date"]
        if date in DEFAULT_TEST_DATES:
            test_dataset.append(item)
        elif date in DEFAULT_VAL_DATES:
            val_dataset.append(item)
        elif date in DEFAULT_TRAIN_DATES:
            train_dataset.append(item)
    return train_dataset, val_dataset, test_dataset


def collect_available_dates(data_dir: Path) -> list[str]:
    sensor_dir = data_dir / "sensor_data"
    gt_dir = data_dir / "ground_truth"
    required_sensor_files = [
        "gps.csv",
        "gps_rtk.csv",
        "ms25.csv",
        "ms25_euler.csv",
        "wheels.csv",
        "odometry_cov_100hz.csv",
    ]
    available = []
    for session_dir in sorted(sensor_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        if not (gt_dir / f"groundtruth_{session_dir.name}.csv").exists():
            continue
        if any(not (session_dir / name).exists() for name in required_sensor_files):
            continue
        available.append(session_dir.name)
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess NCLT into KalmanNet-ready PyTorch files.")
    parser.add_argument("--data-dir", default="./data/NCLT", help="Path containing raw NCLT files.")
    parser.add_argument("--output-dir", default="./data/nclt_fusion", help="Directory for train/val/test .pt files.")
    parser.add_argument("--use-rtk", action="store_true", help="Use gps_rtk.csv instead of gps.csv.")
    parser.add_argument("--random-split", action="store_true", help="Use random 80/10/10 split.")
    parser.add_argument("--random-seed", default=45, type=int, help="Seed for random split.")
    args = parser.parse_args()

    sensor_dir = Path(args.data_dir) / "sensor_data"
    if not sensor_dir.exists():
        raise FileNotFoundError(f"Missing sensor directory: {sensor_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    for date in collect_available_dates(Path(args.data_dir)):
        session_dir = sensor_dir / date
        if not session_dir.is_dir():
            continue
        print(date)
        processed.append(preprocess_session(args.data_dir, date, use_rtk=args.use_rtk))

    train_dataset, val_dataset, test_dataset = split_processed_sessions(
        processed,
        random_split_enabled=args.random_split,
        random_seed=args.random_seed,
    )

    suffix = "_rtk" if args.use_rtk else ""
    torch.save(list(train_dataset), output_dir / f"train{suffix}.pt")
    torch.save(list(test_dataset), output_dir / f"test{suffix}.pt")
    torch.save(list(val_dataset), output_dir / f"val{suffix}.pt")

    print("\nSaved preprocessed datasets:")
    print(output_dir / f"train{suffix}.pt")
    print(output_dir / f"val{suffix}.pt")
    print(output_dir / f"test{suffix}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
