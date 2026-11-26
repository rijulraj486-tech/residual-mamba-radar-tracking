import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rb_to_xy(measurement):
    measurement = np.asarray(measurement, dtype=np.float64)
    return np.asarray(
        [
            measurement[0] * np.cos(measurement[1]),
            measurement[0] * np.sin(measurement[1]),
        ],
        dtype=np.float64,
    )


def xy_to_rb(point):
    point = np.asarray(point, dtype=np.float64)
    rng = np.sqrt(point[0] ** 2 + point[1] ** 2 + 1e-12)
    bearing = np.arctan2(point[1], point[0])
    if bearing < 0.0:
        bearing += 2.0 * np.pi
    return np.asarray([rng, bearing], dtype=np.float64)


def state_to_measurement(state):
    return xy_to_rb([state[0], state[3]])


def initial_state_from_measurement(measurement):
    position = rb_to_xy(measurement)
    state = np.zeros(6, dtype=np.float64)
    state[0] = position[0]
    state[3] = position[1]
    return state


def initial_state_from_history(history, dt):
    history = np.asarray(history, dtype=np.float64).reshape(-1, 2)
    init_state = np.zeros((6, 1), dtype=np.float32)
    first_xy = rb_to_xy(history[0])
    init_state[0, 0] = first_xy[0]
    init_state[3, 0] = first_xy[1]
    if len(history) >= 2:
        second_xy = rb_to_xy(history[1])
        init_state[1, 0] = (second_xy[0] - first_xy[0]) / dt
        init_state[4, 0] = (second_xy[1] - first_xy[1]) / dt
    return init_state


def make_ca_transition(dt):
    return np.asarray(
        [
            [1.0, dt, 0.5 * dt * dt, 0.0, 0.0, 0.0],
            [0.0, 1.0, dt, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, dt, 0.5 * dt * dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_cv_like_transition(dt, accel_decay=0.1):
    return np.asarray(
        [
            [1.0, dt, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, accel_decay, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, dt, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, accel_decay],
        ],
        dtype=np.float64,
    )
