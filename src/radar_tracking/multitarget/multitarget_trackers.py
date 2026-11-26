from dataclasses import dataclass, field

import numpy as np

from radar_tracking.multitarget.association_model import compute_association_probability, estimate_track_state_from_history
from radar_tracking.multitarget.assignment import solve_gnn_assignment
from radar_tracking.multitarget.track_utils import (
    initial_state_from_history,
    initial_state_from_measurement,
    make_ca_transition,
    make_cv_like_transition,
    rb_to_xy,
    state_to_measurement,
    wrap_to_pi,
)


def measurement_jacobian(state):
    px = state[0]
    py = state[3]
    r2 = px * px + py * py + 1e-12
    r = np.sqrt(r2)
    H = np.zeros((2, 6), dtype=np.float64)
    H[0, 0] = px / r
    H[0, 3] = py / r
    H[1, 0] = -py / r2
    H[1, 3] = px / r2
    return H


def covariance_from_measurement(measurement_cov):
    range_var = measurement_cov[0, 0]
    bearing_var = measurement_cov[1, 1]
    return np.diag([range_var * 2.0, 0.05, 0.01, range_var * 2.0, 0.05, 0.01]).astype(np.float64)


def gaussian_likelihood(innovation, covariance):
    innovation = np.asarray(innovation, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return 1e-12
    mahal = innovation.T @ np.linalg.solve(covariance, innovation)
    norm = np.sqrt(((2.0 * np.pi) ** innovation.size) * np.exp(logdet))
    return float(np.exp(-0.5 * mahal) / max(norm, 1e-12))


@dataclass
class ImmTrack:
    track_id: int
    mode_states: list
    mode_covs: list
    mode_probs: np.ndarray
    state: np.ndarray
    predicted_state: np.ndarray
    missed: int = 0
    age: int = 1
    confirmed: bool = True
    tentative_hits: int = 1
    alive: bool = True
    predicted_mode_states: list = field(default_factory=list)
    predicted_mode_covs: list = field(default_factory=list)
    predicted_mode_probs: np.ndarray | None = None
    predicted_measurement: np.ndarray | None = None
    predicted_measurement_cov: np.ndarray | None = None


@dataclass
class MambaTrack:
    track_id: int
    state: np.ndarray
    predicted_state: np.ndarray
    measurement_history: list
    missed: int = 0
    age: int = 1
    confirmed: bool = True
    tentative_hits: int = 1
    alive: bool = True
    association_score: float = 0.0


class ImmGnnTracker:
    def __init__(
        self,
        dt,
        measurement_cov,
        gate_distance=1.5,
        miss_cost=2.5,
        max_missed=3,
        confirm_hits=2,
        model_bank="research",
    ):
        self.dt = dt
        self.measurement_cov = np.asarray(measurement_cov, dtype=np.float64)
        self.gate_distance = gate_distance
        self.miss_cost = miss_cost
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self.next_track_id = 0
        self.tracks = []
        self.models = self._build_models(dt, model_bank)
        self.transition_probs = self._build_transition_probs(len(self.models), model_bank)

    def _build_models(self, dt, model_bank):
        if model_bank == "research":
            return [
                {
                    "name": "CV-static",
                    "F": make_cv_like_transition(dt, accel_decay=0.0),
                    "Q": np.diag([8e-5, 2e-5, 5e-6, 8e-5, 2e-5, 5e-6]),
                },
                {
                    "name": "CV-soft",
                    "F": make_cv_like_transition(dt, accel_decay=0.2),
                    "Q": np.diag([2e-4, 6e-5, 1e-5, 2e-4, 6e-5, 1e-5]),
                },
                {
                    "name": "CA-low",
                    "F": make_ca_transition(dt),
                    "Q": np.diag([4e-4, 1.5e-4, 4e-5, 4e-4, 1.5e-4, 4e-5]),
                },
                {
                    "name": "CA-mid",
                    "F": make_ca_transition(dt),
                    "Q": np.diag([9e-4, 3.5e-4, 1.2e-4, 9e-4, 3.5e-4, 1.2e-4]),
                },
                {
                    "name": "CA-high",
                    "F": make_ca_transition(dt),
                    "Q": np.diag([2.0e-3, 8e-4, 3e-4, 2.0e-3, 8e-4, 3e-4]),
                },
            ]

        return [
            {
                "name": "CV-lite",
                "F": make_cv_like_transition(dt, accel_decay=0.05),
                "Q": np.diag([2e-4, 5e-5, 1e-5, 2e-4, 5e-5, 1e-5]),
            },
            {
                "name": "CA-mid",
                "F": make_ca_transition(dt),
                "Q": np.diag([6e-4, 2e-4, 5e-5, 6e-4, 2e-4, 5e-5]),
            },
            {
                "name": "CA-high",
                "F": make_ca_transition(dt),
                "Q": np.diag([1.5e-3, 6e-4, 2e-4, 1.5e-3, 6e-4, 2e-4]),
            },
        ]

    def _build_transition_probs(self, num_models, model_bank):
        if model_bank == "research" and num_models == 5:
            transition = np.full((num_models, num_models), 0.025, dtype=np.float64)
            np.fill_diagonal(transition, 0.90)
            transition[1, 2] = transition[2, 1] = 0.04
            transition[2, 3] = transition[3, 2] = 0.05
            transition[3, 4] = transition[4, 3] = 0.045
            transition = transition / transition.sum(axis=1, keepdims=True)
            return transition

        transition = np.full((num_models, num_models), 0.08, dtype=np.float64)
        np.fill_diagonal(transition, 0.84)
        return transition / transition.sum(axis=1, keepdims=True)

    def initialize_confirmed_tracks(self, measurements):
        for measurement in measurements:
            self.tracks.append(self._create_track(measurement, confirmed=True))

    def _create_track(self, measurement, confirmed):
        init_state = initial_state_from_measurement(measurement)
        init_cov = covariance_from_measurement(self.measurement_cov)
        mode_states = [init_state.copy() for _ in self.models]
        mode_covs = [init_cov.copy() for _ in self.models]
        mode_probs = np.full(len(self.models), 1.0 / len(self.models), dtype=np.float64)
        track = ImmTrack(
            track_id=self.next_track_id,
            mode_states=mode_states,
            mode_covs=mode_covs,
            mode_probs=mode_probs,
            state=init_state.copy(),
            predicted_state=init_state.copy(),
            confirmed=confirmed,
            tentative_hits=1,
        )
        self.next_track_id += 1
        return track

    def _mix_and_predict(self, track):
        num_modes = len(self.models)
        c_j = self.transition_probs.T @ track.mode_probs
        c_j = np.clip(c_j, 1e-9, None)
        mixed_states = []
        mixed_covs = []

        for mode_j in range(num_modes):
            weights = (self.transition_probs[:, mode_j] * track.mode_probs) / c_j[mode_j]
            mixed_state = np.zeros(6, dtype=np.float64)
            for mode_i in range(num_modes):
                mixed_state += weights[mode_i] * track.mode_states[mode_i]
            mixed_cov = np.zeros((6, 6), dtype=np.float64)
            for mode_i in range(num_modes):
                diff = (track.mode_states[mode_i] - mixed_state).reshape(6, 1)
                mixed_cov += weights[mode_i] * (track.mode_covs[mode_i] + diff @ diff.T)
            mixed_states.append(mixed_state)
            mixed_covs.append(mixed_cov)

        predicted_states = []
        predicted_covs = []
        for model_idx, model in enumerate(self.models):
            pred_state = model["F"] @ mixed_states[model_idx]
            pred_cov = model["F"] @ mixed_covs[model_idx] @ model["F"].T + model["Q"]
            predicted_states.append(pred_state)
            predicted_covs.append(pred_cov)

        mode_probs = c_j / np.sum(c_j)
        fused_state = np.sum(mode_probs[:, None] * np.stack(predicted_states, axis=0), axis=0)
        predicted_measurements = [state_to_measurement(state) for state in predicted_states]
        fused_measurement = np.sum(mode_probs[:, None] * np.stack(predicted_measurements, axis=0), axis=0)
        fused_meas_cov = np.zeros((2, 2), dtype=np.float64)
        for mode_idx in range(num_modes):
            H = measurement_jacobian(predicted_states[mode_idx])
            single_cov = H @ predicted_covs[mode_idx] @ H.T + self.measurement_cov
            diff = (predicted_measurements[mode_idx] - fused_measurement).reshape(2, 1)
            diff[1, 0] = wrap_to_pi(diff[1, 0])
            fused_meas_cov += mode_probs[mode_idx] * (single_cov + diff @ diff.T)
        track.predicted_mode_states = predicted_states
        track.predicted_mode_covs = predicted_covs
        track.predicted_mode_probs = mode_probs
        track.predicted_state = fused_state
        track.predicted_measurement = fused_measurement
        track.predicted_measurement_cov = fused_meas_cov

    def predict(self):
        for track in self.tracks:
            if track.alive:
                self._mix_and_predict(track)

    def _cost_matrix(self, tracks, detections):
        if len(tracks) == 0:
            return np.zeros((0, len(detections)), dtype=np.float64)
        if len(detections) == 0:
            return np.zeros((len(tracks), 0), dtype=np.float64)

        det_xy = np.asarray([rb_to_xy(det) for det in detections], dtype=np.float64)
        cost = np.full((len(tracks), len(detections)), self.miss_cost + 1.0, dtype=np.float64)
        for track_idx, track in enumerate(tracks):
            pred_xy = track.predicted_state[[0, 3]]
            distances = np.linalg.norm(det_xy - pred_xy[None, :], axis=1)
            for det_idx, distance in enumerate(distances):
                if distance <= self.gate_distance:
                    cost[track_idx, det_idx] = distance
        return cost

    def _update_assigned(self, track, measurement):
        z = np.asarray(measurement, dtype=np.float64)
        predicted_probs = track.predicted_mode_probs
        updated_states = []
        updated_covs = []
        likelihoods = []

        for pred_state, pred_cov in zip(track.predicted_mode_states, track.predicted_mode_covs):
            z_pred = state_to_measurement(pred_state)
            H = measurement_jacobian(pred_state)
            innovation = z - z_pred
            innovation[1] = wrap_to_pi(innovation[1])
            S = H @ pred_cov @ H.T + self.measurement_cov
            K = pred_cov @ H.T @ np.linalg.inv(S)
            upd_state = pred_state + K @ innovation
            IKH = np.eye(6) - K @ H
            upd_cov = IKH @ pred_cov @ IKH.T + K @ self.measurement_cov @ K.T
            updated_states.append(upd_state)
            updated_covs.append(upd_cov)
            likelihoods.append(gaussian_likelihood(innovation, S))

        updated_probs = predicted_probs * np.asarray(likelihoods, dtype=np.float64)
        if np.sum(updated_probs) <= 0.0:
            updated_probs = np.full_like(updated_probs, 1.0 / len(updated_probs))
        else:
            updated_probs /= np.sum(updated_probs)

        fused_state = np.sum(updated_probs[:, None] * np.stack(updated_states, axis=0), axis=0)
        fused_cov = np.zeros((6, 6), dtype=np.float64)
        for mode_idx in range(len(updated_states)):
            diff = (updated_states[mode_idx] - fused_state).reshape(6, 1)
            fused_cov += updated_probs[mode_idx] * (updated_covs[mode_idx] + diff @ diff.T)

        track.mode_states = updated_states
        track.mode_covs = updated_covs
        track.mode_probs = updated_probs
        track.state = fused_state
        track.predicted_state = fused_state
        track.missed = 0
        track.age += 1
        if not track.confirmed:
            track.tentative_hits += 1
            if track.tentative_hits >= self.confirm_hits:
                track.confirmed = True

    def _update_missed(self, track):
        track.mode_states = [state.copy() for state in track.predicted_mode_states]
        track.mode_covs = [cov.copy() for cov in track.predicted_mode_covs]
        track.mode_probs = track.predicted_mode_probs.copy()
        track.state = track.predicted_state.copy()
        track.missed += 1
        track.age += 1
        if track.confirmed and track.missed > self.max_missed:
            track.alive = False
        if not track.confirmed and track.missed > 1:
            track.alive = False

    def step(self, detections):
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 2)
        active_tracks = [track for track in self.tracks if track.alive]
        self.predict()

        assignment = solve_gnn_assignment(self._cost_matrix(active_tracks, detections), miss_cost=self.miss_cost)
        assigned_track_indices = set()
        for local_track_idx, det_idx in assignment["assignments"]:
            assigned_track_indices.add(local_track_idx)
            self._update_assigned(active_tracks[local_track_idx], detections[det_idx])

        for local_track_idx, track in enumerate(active_tracks):
            if local_track_idx not in assigned_track_indices:
                self._update_missed(track)

        for det_idx in assignment["unassigned_detections"]:
            self.tracks.append(self._create_track(detections[det_idx], confirmed=False))

        self.tracks = [track for track in self.tracks if track.alive]

    def estimated_positions(self):
        confirmed = [
            track.state[[0, 3]]
            for track in self.tracks
            if track.alive and track.confirmed
        ]
        if not confirmed:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack(confirmed, axis=0)


class MambaGnnTracker:
    def __init__(
        self,
        model,
        device,
        dt,
        measurement_cov,
        gate_distance=1.5,
        miss_cost=2.5,
        max_missed=3,
        confirm_hits=2,
        window_size=30,
    ):
        self.model = model
        self.device = device
        self.dt = dt
        self.gate_distance = gate_distance
        self.miss_cost = miss_cost
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self.window_size = window_size
        self.next_track_id = 0
        self.tracks = []
        self.transition = make_ca_transition(dt)

    def initialize_confirmed_tracks(self, measurements):
        for measurement in measurements:
            self.tracks.append(self._create_track(measurement, confirmed=True))

    def _create_track(self, measurement, confirmed):
        init_state = initial_state_from_measurement(measurement)
        track = MambaTrack(
            track_id=self.next_track_id,
            state=init_state.copy(),
            predicted_state=init_state.copy(),
            measurement_history=[np.asarray(measurement, dtype=np.float32)],
            confirmed=confirmed,
            tentative_hits=1,
        )
        self.next_track_id += 1
        return track

    def _estimate_state_from_history(self, history):
        return estimate_track_state_from_history(
            history=history,
            dt=self.dt,
            window_size=self.window_size,
            model=self.model,
            device=self.device,
        )

    def predict(self):
        for track in self.tracks:
            if not track.alive:
                continue
            track.predicted_state = self.transition @ track.state

    def _cost_matrix(self, tracks, detections):
        if len(tracks) == 0:
            return np.zeros((0, len(detections)), dtype=np.float64)
        if len(detections) == 0:
            return np.zeros((len(tracks), 0), dtype=np.float64)

        det_xy = np.asarray([rb_to_xy(det) for det in detections], dtype=np.float64)
        cost = np.full((len(tracks), len(detections)), self.miss_cost + 1.0, dtype=np.float64)
        for track_idx, track in enumerate(tracks):
            pred_xy = track.predicted_state[[0, 3]]
            distances = np.linalg.norm(det_xy - pred_xy[None, :], axis=1)
            for det_idx, distance in enumerate(distances):
                if distance <= self.gate_distance:
                    cost[track_idx, det_idx] = distance
        return cost

    def _update_assigned(self, track, measurement):
        track.measurement_history.append(np.asarray(measurement, dtype=np.float32))
        if len(track.measurement_history) > self.window_size:
            track.measurement_history = track.measurement_history[-self.window_size :]
        track.state = self._estimate_state_from_history(track.measurement_history)
        track.predicted_state = track.state.copy()
        track.missed = 0
        track.age += 1
        track.association_score = 1.0
        if not track.confirmed:
            track.tentative_hits += 1
            if track.tentative_hits >= self.confirm_hits:
                track.confirmed = True

    def _update_missed(self, track):
        track.state = track.predicted_state.copy()
        track.missed += 1
        track.age += 1
        track.association_score = 0.0
        if track.confirmed and track.missed > self.max_missed:
            track.alive = False
        if not track.confirmed and track.missed > 1:
            track.alive = False

    def step(self, detections):
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 2)
        active_tracks = [track for track in self.tracks if track.alive]
        self.predict()

        assignment = solve_gnn_assignment(self._cost_matrix(active_tracks, detections), miss_cost=self.miss_cost)
        assigned_track_indices = set()
        for local_track_idx, det_idx in assignment["assignments"]:
            assigned_track_indices.add(local_track_idx)
            self._update_assigned(active_tracks[local_track_idx], detections[det_idx])

        for local_track_idx, track in enumerate(active_tracks):
            if local_track_idx not in assigned_track_indices:
                self._update_missed(track)

        for det_idx in assignment["unassigned_detections"]:
            self.tracks.append(self._create_track(detections[det_idx], confirmed=False))

        self.tracks = [track for track in self.tracks if track.alive]

    def estimated_positions(self):
        confirmed = [
            track.state[[0, 3]]
            for track in self.tracks
            if track.alive and track.confirmed
        ]
        if not confirmed:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack(confirmed, axis=0)


class LearnedAssociationMambaGnnTracker(MambaGnnTracker):
    def __init__(
        self,
        model,
        association_model,
        device,
        dt,
        measurement_cov,
        gate_distance=1.5,
        miss_cost=2.5,
        max_missed=3,
        confirm_hits=2,
        window_size=30,
        learned_cost_weight=1.0,
        distance_cost_weight=0.35,
        min_match_probability=0.05,
    ):
        super().__init__(
            model=model,
            device=device,
            dt=dt,
            measurement_cov=measurement_cov,
            gate_distance=gate_distance,
            miss_cost=miss_cost,
            max_missed=max_missed,
            confirm_hits=confirm_hits,
            window_size=window_size,
        )
        self.association_model = association_model
        self.learned_cost_weight = learned_cost_weight
        self.distance_cost_weight = distance_cost_weight
        self.min_match_probability = min_match_probability

    def _cost_matrix(self, tracks, detections):
        if len(tracks) == 0:
            return np.zeros((0, len(detections)), dtype=np.float64)
        if len(detections) == 0:
            return np.zeros((len(tracks), 0), dtype=np.float64)

        det_xy = np.asarray([rb_to_xy(det) for det in detections], dtype=np.float64)
        cost = np.full((len(tracks), len(detections)), self.miss_cost + 1.0, dtype=np.float64)
        for track_idx, track in enumerate(tracks):
            pred_xy = track.predicted_state[[0, 3]]
            distances = np.linalg.norm(det_xy - pred_xy[None, :], axis=1)
            for det_idx, distance in enumerate(distances):
                if distance > self.gate_distance:
                    continue

                match_prob = compute_association_probability(
                    model=self.association_model,
                    predicted_state=track.predicted_state,
                    detection=detections[det_idx],
                    history_len=len(track.measurement_history),
                    missed_count=track.missed,
                    gate_distance=self.gate_distance,
                    device=self.device,
                )
                if match_prob < self.min_match_probability:
                    continue

                learned_cost = -np.log(np.clip(match_prob, 1e-4, 1.0 - 1e-4))
                distance_cost = distance / max(self.gate_distance, 1e-6)
                cost[track_idx, det_idx] = (
                    self.learned_cost_weight * learned_cost
                    + self.distance_cost_weight * distance_cost
                )
        return cost

    def _update_assigned(self, track, measurement):
        match_prob = compute_association_probability(
            model=self.association_model,
            predicted_state=track.predicted_state,
            detection=measurement,
            history_len=len(track.measurement_history),
            missed_count=track.missed,
            gate_distance=self.gate_distance,
            device=self.device,
        )
        super()._update_assigned(track, measurement)
        track.association_score = float(match_prob)
