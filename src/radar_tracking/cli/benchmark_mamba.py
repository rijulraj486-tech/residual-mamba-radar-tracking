import json
import hashlib
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from radar_tracking.data.data_generation import (
    MEAS_NOISE_DIST,
    PROC_NOISE_DIST,
    RB_INIT_VEL_CLIP,
    RB_INIT_VEL_FROM_TWO_MEAS,
    generate_synthetic_data,
    get_motion_profile,
    split_data,
)
from radar_tracking.data.nclt_loader import load_nclt_dataset, split_windows
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.models.ekf import batch_ekf, compute_lossvstime, extended_kf
from radar_tracking.models.kalman_net import KalmanNetNN
from radar_tracking.models.tracker_model import BiMambaTrackerModel
from radar_tracking.config import simparams_class, validate_simparams
from radar_tracking.training.train_validate_test import evaluate_loss_knet, train_validate_knet


SCALE = 1000.0
DEFAULT_BENCHMARK_DIR = str(
    Path(__file__).resolve().parent / "benchmark_outputs_project"
)
REFERENCE_SEQ_LEN = 100


@dataclass
class MambaTrainConfig:
    seed: int = 42
    epochs: int = 180
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    min_delta: float = 1e-6
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.08
    encoder_direction: str = "bidirectional"
    checkpoint_name: str = "best_mamba_tracker.pt"


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def env_int(name, default=None, *, minimum=None):

    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def env_float(name, default=None, *, minimum=None):

    raw = os.environ.get(name)
    if raw is None:
        return default
    value = float(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def resolve_benchmark_dir():

    benchmark_dir = os.environ.get("BENCHMARK_DIR", DEFAULT_BENCHMARK_DIR)
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.is_absolute():
        benchmark_path = Path(__file__).resolve().parent / benchmark_path
    benchmark_path.mkdir(parents=True, exist_ok=True)
    return str(benchmark_path)


def ensure_fresh_benchmark_dir(benchmark_dir):

    benchmark_path = Path(benchmark_dir)
    existing = sorted(path.name for path in benchmark_path.iterdir())
    if existing:
        raise FileExistsError(
            f"Benchmark output directory is not empty: {benchmark_path}. "
            "Choose a new BENCHMARK_DIR so existing artifacts are preserved."
        )


def sha256_file(path):

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulation_config(simparams):

    return {
        "dt": float(simparams.dt),
        "sequence_length": int(simparams.seq_len),
        "measurement_type": simparams.meas_type,
        "initialization_type": simparams.init_type,
        "process_noise_scale": float(simparams.q2_scale),
        "measurement_noise_scale": float(simparams.r2_scale),
        "noise_reference": simparams.noise_ref,
        "range_std_m": float(simparams.range_std),
        "bearing_std_degrees": float(simparams.bearing_std),
        "snr_db": 20,
        "state_scale": SCALE,
        "measurement_noise_distribution": MEAS_NOISE_DIST,
        "process_noise_distribution": PROC_NOISE_DIST,
        "range_bearing_init_velocity_from_two_measurements": RB_INIT_VEL_FROM_TWO_MEAS,
        "range_bearing_init_velocity_clip_mps": RB_INIT_VEL_CLIP,
        "motion_profile": get_motion_profile(simparams),
    }


def fixed_test_settings(simparams):

    mode = os.environ.get("SYNTHETIC_TEST_MODE", "").strip().lower()
    if mode not in {"create", "load"}:
        raise ValueError(
            "Synthetic runs require SYNTHETIC_TEST_MODE=create or load."
        )
    raw_path = os.environ.get("SYNTHETIC_TEST_DATASET", "").strip()
    if not raw_path:
        raise ValueError("SYNTHETIC_TEST_DATASET must name the shared test file.")
    dataset_path = Path(raw_path).expanduser().resolve()
    test_size = env_int("SYNTHETIC_TEST_SIZE", 1024, minimum=1)
    test_seed = env_int("SYNTHETIC_TEST_SEED", 9001, minimum=0)
    return mode, dataset_path, test_size, test_seed, simulation_config(simparams)


def validate_fixed_test_tensors(test_data, ca_model, simparams, test_size):

    expected_shapes = (
        (test_size, ca_model.m, simparams.seq_len),
        (test_size, ca_model.n, simparams.seq_len),
        (test_size, ca_model.m, 1),
        (ca_model.m, ca_model.m),
        (ca_model.n, ca_model.n),
    )
    names = (
        "true_state",
        "observations",
        "initial_state",
        "process_noise",
        "measurement_noise",
    )
    for name, tensor, expected_shape in zip(names, test_data, expected_shapes):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Shared test dataset {name} must be a tensor.")
        if tensor.layout != torch.strided:
            raise ValueError(
                f"Shared test dataset {name} must be a dense tensor; "
                f"got layout {tensor.layout}."
            )
        if tensor.device.type != "cpu":
            raise ValueError(
                f"Shared test dataset {name} must be stored on CPU; "
                f"got device {tensor.device}."
            )
        if not tensor.is_floating_point():
            raise ValueError(
                f"Shared test dataset {name} must use a real floating-point dtype; "
                f"got {tensor.dtype}."
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Shared test dataset {name} shape is {tuple(tensor.shape)}; "
                f"expected {expected_shape}."
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(
                f"Shared test dataset {name} contains non-finite values."
            )


def prepare_fixed_test_data(ca_model, simparams, benchmark_dir):

    mode, dataset_path, test_size, test_seed, expected_simulation = fixed_test_settings(simparams)
    output_path = Path(benchmark_dir).expanduser().resolve()
    if dataset_path == output_path or output_path in dataset_path.parents:
        raise ValueError(
            "SYNTHETIC_TEST_DATASET must be outside each per-experiment "
            "BENCHMARK_DIR so both runs can share it."
        )
    if mode == "create":
        if dataset_path.exists() or dataset_path.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite shared test dataset: {dataset_path}"
            )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        simparams.dataset_size = test_size
        set_seed(test_seed)
        test_data = prepare_test_data(ca_model, simparams)
        validate_fixed_test_tensors(test_data, ca_model, simparams, test_size)
        payload = {
            "format_version": 1,
            "metadata": {
                "seed": test_seed,
                "sample_count": test_size,
                "simulation_config": expected_simulation,
            },
            "true_state": test_data[0],
            "observations": test_data[1],
            "initial_state": test_data[2],
            "process_noise": test_data[3],
            "measurement_noise": test_data[4],
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{dataset_path.name}.",
                suffix=".tmp",
                dir=dataset_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            torch.save(payload, temporary_path)
            if dataset_path.exists() or dataset_path.is_symlink():
                raise FileExistsError(
                    f"Refusing to overwrite shared test dataset: {dataset_path}"
                )
            os.rename(temporary_path, dataset_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
    else:
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Shared test dataset does not exist: {dataset_path}. "
                "Run the create experiment first."
            )
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("Shared test dataset payload must be a mapping.")
        required_entries = {
            "format_version",
            "metadata",
            "true_state",
            "observations",
            "initial_state",
            "process_noise",
            "measurement_noise",
        }
        missing_entries = sorted(required_entries.difference(payload.keys()))
        if missing_entries:
            raise ValueError(
                "Shared test dataset is missing required entries: "
                + ", ".join(missing_entries)
            )
        if type(payload["format_version"]) is not int or payload["format_version"] != 1:
            raise ValueError(f"Unsupported shared test dataset format: {dataset_path}")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("Shared test dataset metadata must be a mapping.")
        required_metadata = {"seed", "sample_count", "simulation_config"}
        missing_metadata = sorted(required_metadata.difference(metadata.keys()))
        if missing_metadata:
            raise ValueError(
                "Shared test dataset metadata is missing required fields: "
                + ", ".join(missing_metadata)
            )
        if type(metadata["seed"]) is not int:
            raise ValueError("Shared test dataset metadata seed must be an integer.")
        if type(metadata["sample_count"]) is not int:
            raise ValueError(
                "Shared test dataset metadata sample_count must be an integer."
            )
        if not isinstance(metadata["simulation_config"], Mapping):
            raise ValueError(
                "Shared test dataset simulation_config metadata must be a mapping."
            )
        if metadata.get("simulation_config") != expected_simulation:
            raise ValueError(
                "Shared test dataset simulation settings do not match this run."
            )
        if int(metadata.get("sample_count", -1)) != test_size:
            raise ValueError(
                f"Shared test dataset has {metadata.get('sample_count')} samples; "
                f"SYNTHETIC_TEST_SIZE={test_size}."
            )
        if int(metadata.get("seed", -1)) != test_seed:
            raise ValueError(
                f"Shared test dataset seed is {metadata.get('seed')}; "
                f"SYNTHETIC_TEST_SEED={test_seed}."
            )
        test_data = (
            payload["true_state"],
            payload["observations"],
            payload["initial_state"],
            payload["process_noise"],
            payload["measurement_noise"],
        )

    validate_fixed_test_tensors(test_data, ca_model, simparams, test_size)

    return test_data, {
        "path": str(dataset_path),
        "sha256": sha256_file(dataset_path),
        "mode": mode,
        "seed": test_seed,
        "sample_count": test_size,
        "simulation_config": expected_simulation,
    }


def raw_mse_and_position_rmse_m(truth, estimates):

    squared_error = (estimates - truth).square()
    aggregate_mse = squared_error.mean()
    position_error = estimates[:, [0, 3], :] - truth[:, [0, 3], :]
    position_rmse_m = torch.sqrt(position_error.square().sum(dim=1).mean()) * SCALE
    return float(aggregate_mse), float(position_rmse_m)


def checkpoint_metadata(path):

    if path is None:
        return {"path": None, "sha256": None, "exists": False}
    checkpoint_path = Path(path).expanduser().resolve()
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
        "exists": checkpoint_path.is_file(),
    }


def resolve_mamba_checkpoint_path(benchmark_dir, checkpoint_name):

    raw_name = str(checkpoint_name)
    relative_path = Path(raw_name)
    windows_path = PureWindowsPath(raw_name)
    if relative_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("MAMBA_CHECKPOINT_NAME must be a relative path.")
    if ".." in relative_path.parts or ".." in windows_path.parts:
        raise ValueError(
            "MAMBA_CHECKPOINT_NAME cannot contain parent-directory traversal."
        )
    if relative_path == Path("."):
        raise ValueError("MAMBA_CHECKPOINT_NAME must name a checkpoint file.")

    output_path = Path(benchmark_dir).expanduser().resolve()
    checkpoint_path = (output_path / relative_path).resolve()
    try:
        checkpoint_path.relative_to(output_path)
    except ValueError as exc:
        raise ValueError(
            "MAMBA_CHECKPOINT_NAME must resolve inside BENCHMARK_DIR."
        ) from exc
    return str(checkpoint_path)


def require_cuda():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The clean Mamba pipeline is CUDA-only in this project because "
            "the Mamba implementation used here does not support CPU execution."
        )
    return torch.device("cuda")


def benchmark_data_mode():

    mode = os.environ.get("BENCHMARK_DATA", "synthetic").strip().lower()
    if mode not in {"synthetic", "real_world"}:
        raise ValueError("BENCHMARK_DATA must be 'synthetic' or 'real_world'.")
    return mode


def apply_mamba_env_overrides(config):

    config.seed = env_int("MAMBA_SEED", config.seed, minimum=0)
    config.epochs = env_int("MAMBA_EPOCHS", config.epochs, minimum=1)
    config.patience = env_int("MAMBA_PATIENCE", config.patience, minimum=1)
    config.batch_size = env_int("MAMBA_BATCH_SIZE", config.batch_size, minimum=1)
    config.eval_batch_size = env_int(
        "MAMBA_EVAL_BATCH_SIZE",
        config.eval_batch_size,
        minimum=1,
    )
    config.embed_dim = env_int("MAMBA_EMBED_DIM", config.embed_dim, minimum=1)
    config.hidden_dim = env_int("MAMBA_HIDDEN_DIM", config.hidden_dim, minimum=1)
    config.num_layers = env_int("MAMBA_NUM_LAYERS", config.num_layers, minimum=1)
    config.d_state = env_int("MAMBA_D_STATE", config.d_state, minimum=1)
    config.d_conv = env_int("MAMBA_D_CONV", config.d_conv, minimum=1)
    config.expand = env_int("MAMBA_EXPAND", config.expand, minimum=1)
    config.dropout = env_float("MAMBA_DROPOUT", config.dropout, minimum=0.0)
    encoder_direction = os.environ.get("MAMBA_DIRECTION", config.encoder_direction)
    config.encoder_direction = encoder_direction.strip().lower()
    config.learning_rate = env_float("MAMBA_LR", config.learning_rate, minimum=0.0)
    config.weight_decay = env_float(
        "MAMBA_WEIGHT_DECAY",
        config.weight_decay,
        minimum=0.0,
    )
    config.min_delta = env_float("MAMBA_MIN_DELTA", config.min_delta, minimum=0.0)
    checkpoint_name = os.environ.get("MAMBA_CHECKPOINT_NAME")
    if checkpoint_name is not None:
        checkpoint_name = checkpoint_name.strip()
        if not checkpoint_name:
            raise ValueError("MAMBA_CHECKPOINT_NAME cannot be empty.")
        config.checkpoint_name = checkpoint_name
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"MAMBA_DROPOUT must be in [0, 1), got {config.dropout}.")
    if config.encoder_direction not in {"forward", "bidirectional"}:
        raise ValueError(
            "MAMBA_DIRECTION must be either 'forward' or 'bidirectional'."
        )
    return config


def mamba_display_name(config):

    return "Mamba"


def should_train_mamba():

    env_mode = os.environ.get("MAMBA_MODE")
    if env_mode is not None:
        mode = env_mode.strip().lower()
        if mode not in {"auto", "train", "load"}:
            raise ValueError("MAMBA_MODE must be one of: auto, train, load")
        if mode == "train":
            return True
        if mode == "load":
            return False

    env_train = os.environ.get("MAMBA_TRAIN")
    if env_train is not None:
        return env_train.strip().lower() in {"1", "true", "yes", "y"}

    return True


def load_existing_mamba_metadata(benchmark_dir):

    metrics_path = Path(benchmark_dir) / "benchmark_metrics.json"
    if not metrics_path.exists():
        return 0, None

    try:
        metrics = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0, None

    training_info = metrics.get("mamba_training", {})
    return (
        int(training_info.get("best_epoch", 0) or 0),
        training_info.get("best_val_loss"),
    )


def scale_dataset(true_state, observations, init_state, process_noise, measurement_noise):

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


def fill_missing_xy(obs_xy):

    obs_xy = torch.as_tensor(obs_xy, dtype=torch.float32)
    valid = torch.isfinite(obs_xy).all(dim=1)
    if not valid.any():
        return torch.zeros_like(obs_xy), valid
    filled = obs_xy.clone()
    first_valid = int(valid.nonzero(as_tuple=False)[0].item())
    filled[:first_valid] = filled[first_valid]
    for idx in range(first_valid + 1, filled.shape[0]):
        if not valid[idx]:
            filled[idx] = filled[idx - 1]
    return filled, valid


def cartesian_to_polar(xy):

    xy = torch.as_tensor(xy, dtype=torch.float32)
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack((torch.sqrt(x.square() + y.square()), torch.atan2(y, x)), dim=-1)


def wrap_to_pi_tensor(angle):

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def prepare_real_world_split(path, seq_len, stride, allow_empty=False):

    state_windows, obs_windows, init_windows = [], [], []
    range_err_sq, bearing_err_sq = [], []
    for obs_xy, state in load_nclt_dataset(path):
        for win_obs, win_state in split_windows(obs_xy, state, window_size=seq_len, stride=stride):
            filled_obs, valid = fill_missing_xy(win_obs)
            filled_polar = cartesian_to_polar(filled_obs / SCALE).transpose(0, 1)
            state_bt = (win_state / SCALE).transpose(0, 1)
            gt_xy = torch.stack((win_state[:, 0], win_state[:, 3]), dim=1) / SCALE
            state_windows.append(state_bt)
            obs_windows.append(filled_polar)
            init_windows.append(state_bt[:, :1])
            if valid.any():
                raw_polar = cartesian_to_polar(win_obs[valid] / SCALE)
                gt_polar = cartesian_to_polar(gt_xy[valid])
                range_err_sq.append((raw_polar[:, 0] - gt_polar[:, 0]).square())
                bearing_err_sq.append(wrap_to_pi_tensor(raw_polar[:, 1] - gt_polar[:, 1]).square())
    if not state_windows:
        if allow_empty:
            return (
                torch.empty(0, 6, seq_len),
                torch.empty(0, 2, seq_len),
                torch.empty(0, 6, 1),
                torch.empty(0),
                torch.empty(0),
            )
        raise RuntimeError(f"No windows could be built from {path} with seq_len={seq_len}, stride={stride}.")
    return (
        torch.stack(state_windows),
        torch.stack(obs_windows),
        torch.stack(init_windows),
        torch.cat(range_err_sq) if range_err_sq else torch.zeros(1),
        torch.cat(bearing_err_sq) if bearing_err_sq else torch.zeros(1),
    )


def select_real_world_split(split, indices):

    return tuple(value[indices] for value in split[:3]) + split[3:]


def merge_real_world_splits(splits):

    non_empty = [split for split in splits if split[0].shape[0] > 0]
    if not non_empty:
        raise RuntimeError("No real-world windows are available in the configured dataset directory.")
    return (
        torch.cat([split[0] for split in non_empty], dim=0),
        torch.cat([split[1] for split in non_empty], dim=0),
        torch.cat([split[2] for split in non_empty], dim=0),
        torch.cat([split[3] for split in non_empty if split[3].numel() > 0], dim=0),
        torch.cat([split[4] for split in non_empty if split[4].numel() > 0], dim=0),
    )


def fallback_real_world_splits(train, val, test):

    merged = merge_real_world_splits((train, val, test))
    total = merged[0].shape[0]
    if total < 3:
        raise RuntimeError("Real-world benchmarking needs at least 3 windows for train/val/test fallback splitting.")
    order = torch.randperm(total)
    n_train = max(1, int(0.7 * total))
    n_val = max(1, int(0.15 * total))
    n_train = min(n_train, total - 2)
    n_val = min(n_val, total - n_train - 1)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return (
        select_real_world_split(merged, train_idx),
        select_real_world_split(merged, val_idx),
        select_real_world_split(merged, test_idx),
    )


def prepare_real_world_data(ca_model, simparams):

    stride = env_int("REALWORLD_STRIDE", max(int(simparams.seq_len) // 2, 1), minimum=1)
    data_dir = Path(os.environ.get("REALWORLD_DATA_DIR", Path(__file__).resolve().parent / "real_world_pipeline" / "data"))
    train = prepare_real_world_split(data_dir / "train.pt", simparams.seq_len, stride, allow_empty=True)
    val = prepare_real_world_split(data_dir / "val.pt", simparams.seq_len, stride, allow_empty=True)
    test = prepare_real_world_split(data_dir / "test.pt", simparams.seq_len, stride)
    fallback_used = train[0].shape[0] == 0 or val[0].shape[0] == 0
    if fallback_used:
        train, val, test = fallback_real_world_splits(train, val, test)
    q = torch.eye(ca_model.m, dtype=torch.float32) * 1e-4
    r = torch.diag(torch.tensor([
        float(torch.cat((train[3], val[3], test[3])).mean().clamp_min(1e-6)),
        float(torch.cat((train[4], val[4], test[4])).mean().clamp_min(1e-6)),
    ], dtype=torch.float32))
    metadata = {
        "data_source": f"real-world NCLT windows from {data_dir}" + (" (fallback split from available windows)" if fallback_used else ""),
        "measurement_type": "pseudo range-bearing converted from GPS xy",
        "dataset_size": int(test[0].shape[0]),
        "report_title": "Real-World NCLT Benchmark",
        "split_sizes": {"train": int(train[0].shape[0]), "val": int(val[0].shape[0]), "test": int(test[0].shape[0])},
    }
    return (train[0], train[1], train[2]), (val[0], val[1], val[2]), test[:3], q, r, metadata


def prepare_train_val_data(ca_model, simparams):

    true_state, observations, init_state, _, _ = generate_synthetic_data(
        ca_model,
        simparams,
        snr_dB=20
    )
    true_state, observations, init_state, _, _ = scale_dataset(
        true_state,
        observations,
        init_state,
        torch.eye(ca_model.m),
        torch.eye(ca_model.n)
    )
    true_state = true_state.cpu()
    observations = observations.cpu()
    init_state = init_state.cpu()
    return split_data(true_state, observations, init_state)


def prepare_test_data(ca_model, simparams):

    true_state, observations, init_state, process_noise, measurement_noise = generate_synthetic_data(
        ca_model,
        simparams,
        snr_dB=20,
        seq_mul_fact=1
    )
    true_state, observations, init_state, process_noise, measurement_noise = scale_dataset(
        true_state,
        observations,
        init_state,
        process_noise,
        measurement_noise
    )
    return (
        true_state.cpu(),
        observations.cpu(),
        init_state.cpu(),
        process_noise.cpu(),
        measurement_noise.cpu()
    )


def build_mamba_model(simparams, config, device):

    return BiMambaTrackerModel(
        task="radar_benchmark",
        state_dim=6,
        obs_dim=2,
        dt=simparams.dt,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        dropout=config.dropout,
        encoder_direction=config.encoder_direction,
    ).to(device)


def evaluate_mamba(model, x_pool, y_pool, init_state, loss_fn=None, batch_size=None):

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction="mean")
    if batch_size is None or batch_size >= x_pool.shape[0]:
        batch_size = x_pool.shape[0]

    model.eval()
    outputs_all = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for start_idx in range(0, x_pool.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, x_pool.shape[0])
            outputs = model(
                observations=y_pool[start_idx:end_idx],
                init_state=init_state[start_idx:end_idx]
            )
            x_est = outputs["state_estimates"]
            batch_target = x_pool[start_idx:end_idx]
            batch_loss = loss_fn(x_est, batch_target)
            loss_sum += batch_loss.item() * (end_idx - start_idx)
            count += end_idx - start_idx
            outputs_all.append(x_est)

    return loss_sum / max(count, 1), torch.cat(outputs_all, dim=0)


def train_mamba(model, train_loader, val_tensors, config, device, checkpoint_path):

    model_label = mamba_display_name(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    loss_fn = torch.nn.MSELoss(reduction="mean")

    best_val = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for obs_batch, init_batch, state_batch in train_loader:
            obs_batch = obs_batch.to(device)
            init_batch = init_batch.to(device)
            state_batch = state_batch.to(device)

            optimizer.zero_grad()
            outputs = model(
                observations=obs_batch,
                init_state=init_batch
            )
            loss = loss_fn(outputs["state_estimates"], state_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)
        val_loss, _ = evaluate_mamba(
            model,
            x_pool=val_tensors["state"].to(device),
            y_pool=val_tensors["obs"].to(device),
            init_state=val_tensors["init"].to(device),
            loss_fn=loss_fn,
            batch_size=config.eval_batch_size
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"{model_label} Epoch {epoch + 1:03d} "
            f"| train={10 * np.log10(train_loss):.4f} dB "
            f"| val={10 * np.log10(val_loss):.4f} dB"
        )

        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping {model_label} at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model, history, best_epoch, best_val


def train_or_load_mamba(model, train_loader, val_tensors, config, device, checkpoint_path, benchmark_dir):

    if should_train_mamba():
        model, history, best_epoch, best_val = train_mamba(
            model,
            train_loader,
            val_tensors,
            config,
            device,
            checkpoint_path,
        )
        return model, history, best_epoch, best_val, True

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Mamba checkpoint does not exist: {checkpoint}. "
            "Run with MAMBA_MODE=train first or provide MAMBA_CHECKPOINT_NAME "
            "with an existing checkpoint in BENCHMARK_DIR."
        )

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    best_epoch, best_val = load_existing_mamba_metadata(benchmark_dir)
    print(f"Loading {mamba_display_name(config)} checkpoint: {checkpoint}")
    return model, None, best_epoch, best_val, False


def plot_loss_curves(history, output_path, model_name):

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, 10 * np.log10(np.asarray(history["train_loss"])), label="Train")
    plt.plot(epochs, 10 * np.log10(np.asarray(history["val_loss"])), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (dB)")
    plt.title(f"{model_name} Tracker Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_mse_time(time_idx, series_dict, output_path):

    plt.figure(figsize=(10, 6))
    for label, values in series_dict.items():
        plt.plot(time_idx, values, marker="o", linewidth=2, label=label)
    plt.xlabel("Time Step")
    plt.ylabel("MSE (dB)")
    plt.title("Long-Sequence Radar Benchmark")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_trajectory(obs_test, ts_test, estimates, output_path):

    idx = 0
    ix = 0
    iy = 3

    gt_x = ts_test[idx, ix, :].cpu().numpy()
    gt_y = ts_test[idx, iy, :].cpu().numpy()

    r_obs = obs_test[idx, 0, :].cpu().numpy()
    theta_obs = obs_test[idx, 1, :].cpu().numpy()
    obs_x = r_obs * np.cos(theta_obs)
    obs_y = r_obs * np.sin(theta_obs)

    plt.figure(figsize=(10, 8))
    plt.scatter(obs_x, obs_y, label="Observations", alpha=0.6)
    plt.plot(gt_x, gt_y, "k-", linewidth=2, label="Ground Truth")

    styles = {
        "EKF": ("--", "green"),
        "KalmanNet": ("-.", "blue"),
        "Mamba": (":", "red"),
    }
    for label, state_est in estimates.items():
        line_style, color = styles[label]
        est_x = state_est[idx, ix, :].cpu().numpy()
        est_y = state_est[idx, iy, :].cpu().numpy()
        plt.plot(est_x, est_y, line_style, color=color, linewidth=2, label=label)

    plt.scatter(gt_x[0], gt_y[0], c="green", marker="^", s=60, label="Start")
    plt.scatter(gt_x[-1], gt_y[-1], c="red", marker="X", s=60, label="End")
    plt.xlabel("Position X (km)")
    plt.ylabel("Position Y (km)")
    plt.title("Trajectory Comparison")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_report(metrics, output_path, output_dir_label):

    knet_retrained = metrics["knet"]["retrained_for_current_context"]
    knet_fairness = (
        "same-context comparison"
        if knet_retrained or metrics["sequence_length"] == REFERENCE_SEQ_LEN
        else "cross-length generalization only"
    )
    model_name = metrics["mamba"]["model_name"]
    encoder_direction = metrics["mamba"]["encoder_direction"]

    lines = [
        f"# {metrics.get('report_title', 'Synthetic Radar Benchmark')}",
        "",
        "## Compared Methods",
        "- EKF baseline from the provided project",
        "- KalmanNet baseline evaluated from either the provided checkpoint or a same-context retraining run",
        f"- {model_name} tracker in radar-benchmark mode",
        "",
        "## Aggregate Test MSE",
        f"- EKF: {metrics['ekf']['total_mse_db']:.3f} dB",
        f"- KalmanNet: {metrics['knet']['total_mse_db']:.3f} dB",
        f"- {model_name}: {metrics['mamba']['total_mse_db']:.3f} dB",
        "",
        "## 2D Position RMSE",
        f"- EKF: {metrics['ekf']['position_rmse_m']:.3f} m",
        f"- KalmanNet: {metrics['knet']['position_rmse_m']:.3f} m",
        f"- {model_name}: {metrics['mamba']['position_rmse_m']:.3f} m",
        "",
        "## Notes",
        f"- Dataset source: {metrics['data_source']}",
        f"- Sequence length: {metrics['sequence_length']}",
        f"- Dataset size: {metrics['dataset_size']}",
        f"- Actual split sizes: {metrics['split_sizes']}",
        f"- Measurement type: {metrics['measurement_type']}",
        f"- Device: {metrics['device']}",
        f"- Mamba encoder direction: {encoder_direction}",
        f"- KalmanNet parameters: {metrics['knet']['param_count']}",
        f"- KalmanNet checkpoint source: {metrics['knet']['checkpoint_source']}",
        f"- KalmanNet fairness status: {knet_fairness}",
        f"- {model_name} parameters: {metrics['mamba']['param_count']}",
        "",
        "## Output Files",
        f"- {output_dir_label}/benchmark_metrics.json",
        f"- {output_dir_label}/mse_time_comparison.png",
        f"- {output_dir_label}/trajectory_comparison.png",
        f"- {output_dir_label}/mamba_training_curve.png",
    ]
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def count_parameters(model):

    return sum(param.numel() for param in model.parameters())


def knet_collapse_note(knet_total_db: float, reference_seq_len: int, active_seq_len: int) -> str:

    """
    Explain why a positive KalmanNet MSE can appear in the long-context setup.

    Positive dB here means the average squared estimation error is above the
    normalized 0 dB reference level. In this project, that usually indicates
    that the step-wise recurrent KalmanNet estimator did not remain stable when
    moved to a much longer context length.
    """
    if knet_total_db <= 0.0:
        return ""

    return (
        f"\n[NOTE] KalmanNet MSE = +{knet_total_db:.3f} dB.\n"
        f"  In this benchmark, a positive value means KalmanNet is no longer\n"
        f"  tracking well at seq_len={active_seq_len}.\n"
        f"  KalmanNet updates one step at a time through a recurrent hidden state,\n"
        f"  while this long-context setting is {active_seq_len/reference_seq_len:.1f}x\n"
        f"  longer than the original reference length ({reference_seq_len}).\n"
        f"  When the hidden-state dynamics are not stable enough, the learned gain\n"
        f"  estimate can drift and the filter can diverge.\n"
        f"  This is a key motivation for using a full-sequence model such as Mamba.\n"
    )


def configure_knet_training(simparams):

    long_context = int(simparams.seq_len) > 100

    default_batch = min(simparams.n_batch, 8) if long_context else simparams.n_batch
    default_lr = min(simparams.learning_rate, 5e-4) if long_context else simparams.learning_rate
    default_weight_decay = max(simparams.weight_decay, 1e-4)
    default_patience = min(simparams.stopping_criteria, 12) if long_context else simparams.stopping_criteria

    simparams.n_batch = env_int("KNET_BATCH_SIZE", default_batch, minimum=1)
    simparams.epochs = env_int("KNET_EPOCHS", simparams.epochs, minimum=1)
    simparams.learning_rate = env_float("KNET_LR", default_lr, minimum=0.0)
    simparams.weight_decay = env_float(
        "KNET_WEIGHT_DECAY",
        default_weight_decay,
        minimum=0.0,
    )
    simparams.stopping_criteria = env_int("KNET_PATIENCE", default_patience, minimum=1)


def should_train_knet(simparams):

    env_mode = os.environ.get("KNET_MODE")
    if env_mode is not None:
        mode = env_mode.strip().lower()
        if mode not in {"auto", "train", "load"}:
            raise ValueError("KNET_MODE must be one of: auto, train, load")
        if mode == "train":
            return True
        if mode == "load":
            return False

    env_train = os.environ.get("KNET_TRAIN")
    if env_train is not None:
        return env_train.strip().lower() in {"1", "true", "yes", "y"}

    return False


def resolve_knet_checkpoint_candidates(benchmark_dir, seq_len):

    env_checkpoint = os.environ.get("KNET_CHECKPOINT")
    if env_checkpoint is not None:
        raw_path = env_checkpoint.strip()
        if not raw_path:
            raise ValueError("KNET_CHECKPOINT cannot be empty when provided.")
        checkpoint_path = Path(raw_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parent / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"KNET_CHECKPOINT does not exist: {checkpoint_path}"
            )
        return [(str(checkpoint_path), f"explicit checkpoint ({checkpoint_path.name})")]

    candidate_paths = []
    project_dir = Path(__file__).resolve().parent
    benchmark_name = f"best_knet_seq{int(seq_len)}.pt"

    candidate_paths.append(Path(benchmark_dir) / benchmark_name)
    candidate_paths.append(Path(DEFAULT_BENCHMARK_DIR) / benchmark_name)
    candidate_paths.extend(sorted(project_dir.glob(f"benchmark_outputs*/{benchmark_name}")))
    candidate_paths.append(project_dir / "best_knet.pt")

    candidates = []
    seen = set()
    for candidate_path in candidate_paths:
        candidate_path = candidate_path.resolve()
        candidate_key = str(candidate_path)
        if candidate_key in seen or not candidate_path.exists():
            continue
        seen.add(candidate_key)
        if candidate_path.name == "best_knet.pt":
            source = "legacy provided best_knet.pt"
        elif candidate_path.parent.resolve() == Path(benchmark_dir).resolve():
            source = f"same-context checkpoint ({candidate_path.name})"
        else:
            source = (
                f"same-context checkpoint ({candidate_path.name}) "
                f"from {candidate_path.parent.name}"
            )
        candidates.append((candidate_key, source))

    if candidates:
        return candidates

    raise FileNotFoundError(
        "No KalmanNet checkpoint found. Retrain with KNET_TRAIN=1 or provide "
        "KNET_CHECKPOINT=/path/to/checkpoint.pt."
    )


def select_best_knet_checkpoint(
    knet,
    device,
    ts_val,
    obs_val,
    init_val,
    simparams,
    benchmark_dir,
):

    candidates = resolve_knet_checkpoint_candidates(benchmark_dir, simparams.seq_len)
    best_entry = None

    for checkpoint_path, checkpoint_source in candidates:
        checkpoint_state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        knet.load_state_dict(checkpoint_state)
        val_loss, _, _, _ = evaluate_loss_knet(
            knet,
            ts_val.to(device),
            obs_val.to(device),
            init_val.to(device),
            simparams,
        )
        print(
            f"KalmanNet candidate: {checkpoint_path} "
            f"| val={10 * np.log10(val_loss):.4f} dB"
        )
        if best_entry is None or val_loss < best_entry["val_loss"]:
            best_entry = {
                "path": checkpoint_path,
                "source": checkpoint_source,
                "state": checkpoint_state,
                "val_loss": val_loss,
            }

    if best_entry is None:
        raise RuntimeError("KalmanNet checkpoint selection failed.")

    knet.load_state_dict(best_entry["state"])
    return best_entry["path"], best_entry["source"], best_entry["state"]


def train_or_load_knet(
    ca_model,
    simparams,
    device,
    ts_train,
    obs_train,
    ts_val,
    obs_val,
    init_train,
    init_val,
    benchmark_dir,
):

    knet = KalmanNetNN()
    knet.NNBuild(ca_model, simparams)

    train_knet = should_train_knet(simparams)
    checkpoint_path = None
    checkpoint_source = ""

    if train_knet:
        checkpoint_path = os.path.join(benchmark_dir, f"best_knet_seq{int(simparams.seq_len)}.pt")
        print(f"Training KalmanNet for seq_len={simparams.seq_len}...")
        knet, _ = train_validate_knet(
            ca_model,
            knet,
            simparams,
            ts_train.to(device),
            obs_train.to(device),
            ts_val.to(device),
            obs_val.to(device),
            init_train.to(device),
            init_val.to(device),
            checkpoint_path,
        )
        if os.path.exists(checkpoint_path):
            knet_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            knet.load_state_dict(knet_state)
            checkpoint_source = f"retrained checkpoint ({os.path.basename(checkpoint_path)})"
        else:
            knet_state = knet.state_dict()
            checkpoint_source = "training attempted, no stable checkpoint was saved"
    else:
        checkpoint_path, checkpoint_source, knet_state = select_best_knet_checkpoint(
            knet,
            device,
            ts_val,
            obs_val,
            init_val,
            simparams,
            benchmark_dir,
        )
        print(f"Loading KalmanNet checkpoint: {checkpoint_path}")
        if (
            int(simparams.seq_len) != REFERENCE_SEQ_LEN
            and f"seq{int(simparams.seq_len)}" not in os.path.basename(checkpoint_path)
        ):
            print(
                f"\n[KalmanNet FAIRNESS WARNING]\n"
                f"  Loading a checkpoint trained at seq_len={REFERENCE_SEQ_LEN}\n"
                f"  but evaluating it at seq_len={simparams.seq_len}.\n"
                f"  This is not a fair same-context comparison.\n"
                f"  The resulting KalmanNet score reflects cross-length generalization,\n"
                f"  not best KalmanNet performance at this sequence length.\n"
                f"  Re-run with KNET_TRAIN=1 to retrain KalmanNet for the active context.\n"
            )

    return knet, knet_state, checkpoint_path, checkpoint_source, train_knet


def main():

    config = apply_mamba_env_overrides(MambaTrainConfig())
    model_name = mamba_display_name(config)
    benchmark_dir = resolve_benchmark_dir()
    set_seed(config.seed)

    simparams = simparams_class()
    validate_simparams(simparams)
    configure_knet_training(simparams)
    data_mode = benchmark_data_mode()

    device = require_cuda()
    ca_model = ConstantAcceleration2DModel(simparams)
    test_metadata = None
    if data_mode == "real_world":
        (ts_train, obs_train, init_train), (ts_val, obs_val, init_val), (ts_test, obs_test, init_test), q_test, r_test, data_meta = prepare_real_world_data(
            ca_model,
            simparams,
        )
    else:
        ensure_fresh_benchmark_dir(benchmark_dir)
        training_pool_size = int(simparams.dataset_size)
        (ts_test, obs_test, init_test, q_test, r_test), test_metadata = prepare_fixed_test_data(
            ca_model,
            simparams,
            benchmark_dir,
        )
        simparams.dataset_size = training_pool_size
        # Keep train/validation generation and model initialization tied to the
        # run seed, independently of fixed-test creation or loading.
        set_seed(config.seed)
        (ts_train, obs_train, init_train), (ts_val, obs_val, init_val) = prepare_train_val_data(
            ca_model,
            simparams,
        )
        data_meta = {
            "data_source": "synthetic training pool plus shared fixed test dataset",
            "measurement_type": simparams.meas_type,
            "dataset_size": training_pool_size,
            "report_title": "Synthetic Radar Benchmark",
            "split_sizes": {"train": int(ts_train.shape[0]), "val": int(ts_val.shape[0]), "test": int(ts_test.shape[0])},
        }
    simparams.dataset_size = int(ts_test.shape[0])

    train_dataset = TensorDataset(obs_train, init_train, ts_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_tensors = {"obs": obs_val, "init": init_val, "state": ts_val}

    mamba_model = build_mamba_model(simparams, config, device)
    mamba_checkpoint = resolve_mamba_checkpoint_path(
        benchmark_dir,
        config.checkpoint_name,
    )
    mamba_model, mamba_history, best_epoch, best_val, mamba_was_trained = train_or_load_mamba(
        mamba_model,
        train_loader,
        val_tensors,
        config,
        device,
        mamba_checkpoint,
        benchmark_dir,
    )
    if mamba_history is not None:
        plot_loss_curves(
            mamba_history,
            os.path.join(benchmark_dir, "mamba_training_curve.png"),
            model_name,
        )

    knet, knet_state, knet_checkpoint, knet_checkpoint_source, knet_was_retrained = train_or_load_knet(
        ca_model,
        simparams,
        device,
        ts_train,
        obs_train,
        ts_val,
        obs_val,
        init_train,
        init_val,
        benchmark_dir,
    )

    ekf = extended_kf(ca_model, q_test / simparams.q2_scale, r_test / simparams.r2_scale, simparams)
    ekf.reset(init_test)
    ekf_initial_covariance = ekf.P.clone()
    batch_ekf_eval = batch_ekf(ekf)
    ekf_estimates, _, _, _, _ = batch_ekf_eval.evaluate_batch(ts_test, obs_test, init_test)
    ekf_loss_time, ekf_total_db = compute_lossvstime(ts_test, ekf_estimates)

    _, knet_estimates, k_fro, ln_innov = evaluate_loss_knet(
        knet,
        ts_test.to(device),
        obs_test.to(device),
        init_test.to(device),
        simparams
    )
    knet_estimates = knet_estimates.detach().cpu()
    knet_loss_time, knet_total_db = compute_lossvstime(ts_test.cpu(), knet_estimates)
    knet_raw_mse, knet_position_rmse_m = raw_mse_and_position_rmse_m(ts_test, knet_estimates)
    knet_note = knet_collapse_note(float(knet_total_db), REFERENCE_SEQ_LEN, int(simparams.seq_len))

    _, mamba_estimates = evaluate_mamba(
        mamba_model,
        x_pool=ts_test.to(device),
        y_pool=obs_test.to(device),
        init_state=init_test.to(device),
        batch_size=config.eval_batch_size
    )
    mamba_estimates = mamba_estimates.detach().cpu()
    mamba_loss_time, mamba_total_db = compute_lossvstime(ts_test.cpu(), mamba_estimates)
    mamba_raw_mse, mamba_position_rmse_m = raw_mse_and_position_rmse_m(ts_test, mamba_estimates)
    ekf_raw_mse, ekf_position_rmse_m = raw_mse_and_position_rmse_m(ts_test, ekf_estimates)

    mamba_checkpoint_info = checkpoint_metadata(mamba_checkpoint)
    knet_checkpoint_info = checkpoint_metadata(knet_checkpoint)
    if not mamba_checkpoint_info["exists"]:
        raise FileNotFoundError(
            f"Mamba checkpoint is missing after evaluation: {mamba_checkpoint}"
        )
    if not knet_checkpoint_info["exists"]:
        raise FileNotFoundError(
            f"KalmanNet checkpoint is missing after evaluation: {knet_checkpoint}"
        )
    ekf_settings = {
        **simulation_config(simparams),
        "state_transition_matrix_F": ca_model.F.tolist(),
        "process_noise_matrix_Q_used": (q_test / simparams.q2_scale).tolist(),
        "measurement_noise_matrix_R_used": (r_test / simparams.r2_scale).tolist(),
        "initial_covariance_before_first_predict": ekf_initial_covariance.tolist(),
    }
    knet_train_config = {
        "mode": "train" if knet_was_retrained else "load",
        "batch_size": int(simparams.n_batch),
        "epochs": int(simparams.epochs),
        "learning_rate": float(simparams.learning_rate),
        "weight_decay": float(simparams.weight_decay),
        "stopping_criteria": int(simparams.stopping_criteria),
        "update_threshold_db": float(simparams.update_thr_db),
        "input_multiplier": int(simparams.in_mult_KNet),
        "output_multiplier": int(simparams.out_mult_KNet),
        "range_bearing_wrap_theta": bool(simparams.knet_rb_wrap_theta),
        "range_bearing_arc_length": bool(simparams.knet_rb_arc_length),
        "sequence_length": int(simparams.seq_len),
    }

    time_idx = list(range(ts_test.shape[2]))
    plot_mse_time(
        time_idx,
        {
            f"EKF ({float(ekf_total_db):.2f} dB)": ekf_loss_time.cpu().numpy(),
            f"KalmanNet ({float(knet_total_db):.2f} dB)": knet_loss_time.cpu().numpy(),
            f"{model_name} ({float(mamba_total_db):.2f} dB)": mamba_loss_time.cpu().numpy(),
        },
        os.path.join(benchmark_dir, "mse_time_comparison.png")
    )
    plot_trajectory(
        obs_test.cpu(),
        ts_test.cpu(),
        {
            "EKF": ekf_estimates.cpu(),
            "KalmanNet": knet_estimates,
            model_name: mamba_estimates
        },
        os.path.join(benchmark_dir, "trajectory_comparison.png")
    )

    metrics_path = os.path.join(benchmark_dir, "benchmark_metrics.json")
    report_path = os.path.join(benchmark_dir, "benchmark_report.md")

    metrics = {
        "device": str(device),
        "data_mode": data_mode,
        "data_source": data_meta["data_source"],
        "measurement_type": data_meta["measurement_type"],
        "report_title": data_meta["report_title"],
        "sequence_length": int(simparams.seq_len),
        "dataset_size": data_meta["dataset_size"],
        "split_sizes": data_meta["split_sizes"],
        "mamba_train_config": asdict(config),
        "mamba_training": {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val) if best_val is not None else None,
            "best_val_loss_db": float(10 * np.log10(best_val)) if best_val is not None else None,
            "mode": "train" if mamba_was_trained else "load",
        },
        "seeds": {
            "training_and_model_seed": int(config.seed),
            "fixed_test_seed": test_metadata["seed"] if test_metadata else None,
        },
        "simulation_config": simulation_config(simparams),
        "ekf_config": ekf_settings,
        "fixed_test_dataset": test_metadata,
        "evaluation_convention": {
            "aggregate_six_state_mse": "mean((estimate - truth)^2) across samples, 6 state components, and time",
            "aggregate_six_state_mse_db": "10 * log10(aggregate_six_state_mse)",
            "position_rmse_m": "sqrt(mean(dx^2 + dy^2)) after converting scaled coordinates to metres",
        },
        "paths": {
            "benchmark_dir": benchmark_dir,
            "metrics_path": metrics_path,
            "report_path": report_path,
            "mamba_checkpoint": mamba_checkpoint_info["path"],
        },
        "ekf": {
            "total_mse_db": float(ekf_total_db),
            "aggregate_six_state_mse": ekf_raw_mse,
            "position_rmse_m": ekf_position_rmse_m,
            "loss_vs_time_db": [float(value) for value in ekf_loss_time]
        },
        "knet": {
            "total_mse_db": float(knet_total_db),
            "aggregate_six_state_mse": knet_raw_mse,
            "position_rmse_m": knet_position_rmse_m,
            "loss_vs_time_db": [float(value) for value in knet_loss_time],
            "param_count": int(sum(value.numel() for value in knet_state.values())),
            "checkpoint": knet_checkpoint_info["path"],
            "checkpoint_sha256": knet_checkpoint_info["sha256"],
            "checkpoint_source": knet_checkpoint_source,
            "retrained_for_current_context": bool(knet_was_retrained),
            "train_config": knet_train_config,
            "kalman_gain_frobenius": [float(value) for value in k_fro],
            "log_innovation_norm": [float(value) for value in ln_innov]
        },
        "mamba": {
            "model_name": model_name,
            "encoder_direction": config.encoder_direction,
            "effective_encoder_direction": getattr(mamba_model.encoder, "encoder_direction", "unknown"),
            "total_mse_db": float(mamba_total_db),
            "aggregate_six_state_mse": mamba_raw_mse,
            "position_rmse_m": mamba_position_rmse_m,
            "loss_vs_time_db": [float(value) for value in mamba_loss_time],
            "param_count": int(count_parameters(mamba_model)),
            "checkpoint": mamba_checkpoint_info["path"],
            "checkpoint_sha256": mamba_checkpoint_info["sha256"],
            "train_config": asdict(config),
        }
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_report(metrics, report_path, benchmark_dir)

    print("Benchmark complete")
    print("EKF total MSE (dB):", float(ekf_total_db))
    print("KalmanNet total MSE (dB):", float(knet_total_db))
    if knet_note:
        print(knet_note, end="")
    print(f"{model_name} total MSE (dB):", float(mamba_total_db))
    print("Saved outputs to", benchmark_dir)


if __name__ == "__main__":
    main()
