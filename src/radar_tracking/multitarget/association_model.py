import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from radar_tracking.multitarget.track_utils import (
    initial_state_from_history,
    initial_state_from_measurement,
    make_ca_transition,
    rb_to_xy,
    state_to_measurement,
    wrap_to_pi,
)


POS_SCALE = 50.0
VEL_SCALE = 1.0
ACC_SCALE = 0.05
DIST_SCALE = 5.0
GATE_FLOOR = 1e-6


@dataclass
class AssociationModelConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.10


@dataclass
class AssociationTrainingConfig:
    seed: int = 123
    train_num_scenes: int = 48
    val_num_scenes: int = 12
    num_targets: int = 4
    seq_len: int = 45
    detection_prob: float = 0.9
    clutter_rate: float = 2.5
    max_birth_step: int = 10
    min_track_length: int = 16
    window_size: int = 35
    min_history: int = 2
    max_missed_history: int = 3
    negative_ratio: int = 4
    batch_size: int = 1024
    epochs: int = 18
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = 1e-4
    checkpoint_name: str = "best_assoc_model.pt"


class AssociationScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.10):
        super().__init__()
        dims = [input_dim]
        if num_layers <= 1:
            dims.append(hidden_dim)
        else:
            dims.extend([hidden_dim] * num_layers)

        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_track_state_from_history(history, dt, window_size, model=None, device=None):
    history = np.asarray(history, dtype=np.float32).reshape(-1, 2)
    if len(history) == 0:
        raise ValueError("history must contain at least one detection")
    if len(history) == 1:
        return initial_state_from_measurement(history[0])
    if len(history) < 4 or model is None:
        coarse_init = initial_state_from_history(history, dt).squeeze(-1).astype(np.float64)
        latest_xy = rb_to_xy(history[-1])
        coarse_init[0] = latest_xy[0]
        coarse_init[3] = latest_xy[1]
        return coarse_init

    hist = np.asarray(history[-window_size:], dtype=np.float32)
    obs_tensor = torch.from_numpy(hist.T).unsqueeze(0).to(device)
    init_state = torch.from_numpy(initial_state_from_history(hist, dt)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(obs_tensor, init_state)
    return outputs["state_estimates"][0, :, -1].detach().cpu().numpy().astype(np.float64)


def predict_track_state_from_history(history, dt, window_size, model=None, device=None):
    estimated = estimate_track_state_from_history(
        history=history,
        dt=dt,
        window_size=window_size,
        model=model,
        device=device,
    )
    return make_ca_transition(dt) @ estimated


def build_association_features(predicted_state, detection, history_len, missed_count, gate_distance):
    predicted_state = np.asarray(predicted_state, dtype=np.float64).reshape(-1)
    detection = np.asarray(detection, dtype=np.float64).reshape(-1)

    det_xy = rb_to_xy(detection)
    pred_xy = predicted_state[[0, 3]]
    pred_rb = state_to_measurement(predicted_state)

    delta_xy = det_xy - pred_xy
    distance = float(np.linalg.norm(delta_xy))
    delta_rb = detection - pred_rb
    delta_rb[1] = wrap_to_pi(delta_rb[1])

    features = np.asarray(
        [
            predicted_state[0] / POS_SCALE,
            predicted_state[3] / POS_SCALE,
            predicted_state[1] / VEL_SCALE,
            predicted_state[4] / VEL_SCALE,
            predicted_state[2] / ACC_SCALE,
            predicted_state[5] / ACC_SCALE,
            pred_rb[0] / POS_SCALE,
            np.sin(pred_rb[1]),
            np.cos(pred_rb[1]),
            detection[0] / POS_SCALE,
            np.sin(detection[1]),
            np.cos(detection[1]),
            det_xy[0] / POS_SCALE,
            det_xy[1] / POS_SCALE,
            delta_xy[0] / DIST_SCALE,
            delta_xy[1] / DIST_SCALE,
            distance / max(gate_distance, GATE_FLOOR),
            delta_rb[0] / DIST_SCALE,
            np.sin(delta_rb[1]),
            np.cos(delta_rb[1]),
            min(float(history_len), 50.0) / 50.0,
            min(float(missed_count), 10.0) / 10.0,
        ],
        dtype=np.float32,
    )
    return features


def compute_association_probability(
    model,
    predicted_state,
    detection,
    history_len,
    missed_count,
    gate_distance,
    device,
):
    features = build_association_features(
        predicted_state=predicted_state,
        detection=detection,
        history_len=history_len,
        missed_count=missed_count,
        gate_distance=gate_distance,
    )
    feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(feature_tensor)
        prob = torch.sigmoid(logit).item()
    return prob


def _sample_negative_indices(rng, negative_indices, max_keep):
    if len(negative_indices) <= max_keep:
        return negative_indices
    chosen = rng.choice(np.asarray(negative_indices, dtype=np.int64), size=max_keep, replace=False)
    return [int(idx) for idx in chosen]


def build_association_examples(
    scenes,
    dt,
    window_size,
    gate_distance,
    model=None,
    device=None,
    min_history=2,
    max_missed_history=3,
    negative_ratio=4,
    seed=0,
):
    rng = np.random.default_rng(seed)
    features = []
    labels = []

    for scene in scenes:
        num_targets, seq_len = scene["alive_mask"].shape
        histories = [[] for _ in range(num_targets)]
        missed_counts = np.zeros(num_targets, dtype=np.int64)

        for time_idx in range(seq_len):
            detections = np.asarray(scene["detections"][time_idx], dtype=np.float32).reshape(-1, 2)
            detection_target_ids = np.asarray(scene["detection_target_ids"][time_idx], dtype=np.int64)
            alive_target_ids = np.where(scene["alive_mask"][:, time_idx])[0].tolist()

            for target_idx in alive_target_ids:
                history = histories[target_idx]
                if len(history) < min_history or missed_counts[target_idx] > max_missed_history:
                    continue

                predicted_state = predict_track_state_from_history(
                    history=history,
                    dt=dt,
                    window_size=window_size,
                    model=model,
                    device=device,
                )

                positive_indices = []
                negative_indices = []
                pair_features = []
                for det_idx, detection in enumerate(detections):
                    pair_features.append(
                        build_association_features(
                            predicted_state=predicted_state,
                            detection=detection,
                            history_len=len(history),
                            missed_count=int(missed_counts[target_idx]),
                            gate_distance=gate_distance,
                        )
                    )
                    if det_idx < len(detection_target_ids) and detection_target_ids[det_idx] == target_idx:
                        positive_indices.append(det_idx)
                    else:
                        negative_indices.append(det_idx)

                for det_idx in positive_indices:
                    features.append(pair_features[det_idx])
                    labels.append(1.0)

                if positive_indices:
                    max_keep = max(negative_ratio * len(positive_indices), 1)
                else:
                    max_keep = min(negative_ratio, len(negative_indices))
                for det_idx in _sample_negative_indices(rng, negative_indices, max_keep):
                    features.append(pair_features[det_idx])
                    labels.append(0.0)

            matched_targets = set()
            for det_idx, target_idx in enumerate(detection_target_ids.tolist()):
                if target_idx < 0:
                    continue
                histories[target_idx].append(detections[det_idx].astype(np.float32))
                if len(histories[target_idx]) > window_size:
                    histories[target_idx] = histories[target_idx][-window_size:]
                missed_counts[target_idx] = 0
                matched_targets.add(target_idx)

            for target_idx in alive_target_ids:
                if target_idx not in matched_targets and len(histories[target_idx]) >= min_history:
                    missed_counts[target_idx] += 1

    if not features:
        raise RuntimeError("No association training examples were generated.")

    feature_tensor = torch.from_numpy(np.stack(features, axis=0).astype(np.float32))
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    return feature_tensor, label_tensor


def _compute_binary_metrics(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    labels = labels.float()

    accuracy = (preds == labels).float().mean().item()
    tp = ((preds == 1.0) & (labels == 1.0)).sum().item()
    fp = ((preds == 1.0) & (labels == 0.0)).sum().item()
    fn = ((preds == 0.0) & (labels == 1.0)).sum().item()
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "positive_rate": float(labels.mean().item()),
    }


def train_association_model(
    train_features,
    train_labels,
    val_features,
    val_labels,
    model_config,
    train_config,
    device,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AssociationScoreNet(
        input_dim=train_features.shape[1],
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    ).to(device)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
    )

    pos_count = float(train_labels.sum().item())
    neg_count = float(train_labels.shape[0] - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = output_dir / train_config.checkpoint_name
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
    }

    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    for epoch in range(train_config.epochs):
        model.train()
        running_loss = 0.0
        running_logits = []
        running_labels = []

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch_features.shape[0]
            running_logits.append(logits.detach().cpu())
            running_labels.append(batch_labels.detach().cpu())

        train_logits = torch.cat(running_logits, dim=0)
        train_labels_cpu = torch.cat(running_labels, dim=0)
        train_loss = running_loss / max(train_labels_cpu.shape[0], 1)
        train_metrics = _compute_binary_metrics(train_logits, train_labels_cpu)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features)
            val_loss = loss_fn(val_logits, val_labels).item()
        val_metrics = _compute_binary_metrics(val_logits.detach().cpu(), val_labels.detach().cpu())

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

        print(
            f"Assoc Epoch {epoch + 1:03d} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_acc={val_metrics['accuracy']:.4f} "
            f"| val_prec={val_metrics['precision']:.4f} "
            f"| val_rec={val_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss - train_config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "train_config": asdict(train_config),
                    "feature_dim": int(train_features.shape[1]),
                    "best_val_loss": float(val_loss),
                    "history": history,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= train_config.patience:
            print(f"Early stopping association model at epoch {epoch + 1}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metrics = {
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "train_examples": int(train_features.shape[0]),
        "val_examples": int(val_features.shape[0]),
        "train_positive_rate": float(train_labels.mean().item()),
        "val_positive_rate": float(val_labels.float().mean().item()),
        "history": history,
    }
    return model, metrics, checkpoint_path


def save_association_metrics(output_dir, metrics, model_config, train_config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    with (output_dir / "association_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_association_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = AssociationModelConfig(**checkpoint["model_config"])
    model = AssociationScoreNet(
        input_dim=checkpoint["feature_dim"],
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
