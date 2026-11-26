import numpy as np


def _hungarian(cost_matrix):
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0:
        return 0.0, []
    if n_cols < n_rows:
        raise ValueError("Hungarian solver requires columns >= rows.")

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int32)
    way = np.zeros(n_cols + 1, dtype=np.int32)

    for row_idx in range(1, n_rows + 1):
        p[0] = row_idx
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        col0 = 0

        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col_idx in range(1, n_cols + 1):
                if used[col_idx]:
                    continue
                cur = cost[row0 - 1, col_idx - 1] - u[row0] - v[col_idx]
                if cur < minv[col_idx]:
                    minv[col_idx] = cur
                    way[col_idx] = col0
                if minv[col_idx] < delta:
                    delta = minv[col_idx]
                    col1 = col_idx
            for col_idx in range(n_cols + 1):
                if used[col_idx]:
                    u[p[col_idx]] += delta
                    v[col_idx] -= delta
                else:
                    minv[col_idx] -= delta
            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [-1] * n_rows
    for col_idx in range(1, n_cols + 1):
        if p[col_idx] != 0:
            assignment[p[col_idx] - 1] = col_idx - 1

    total_cost = sum(cost[row_idx, col_idx] for row_idx, col_idx in enumerate(assignment))
    return float(total_cost), assignment


def solve_gnn_assignment(cost_matrix, miss_cost):
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    n_tracks, n_dets = cost.shape
    if n_tracks == 0:
        return {
            "assignments": [],
            "unassigned_tracks": [],
            "unassigned_detections": list(range(n_dets)),
            "total_cost": 0.0,
        }

    augmented = np.full((n_tracks, n_dets + n_tracks), fill_value=miss_cost, dtype=np.float64)
    if n_dets:
        augmented[:, :n_dets] = cost

    total_cost, chosen_cols = _hungarian(augmented)

    assignments = []
    used_dets = set()
    unassigned_tracks = []

    for track_idx, col_idx in enumerate(chosen_cols):
        if col_idx < n_dets:
            assignments.append((track_idx, col_idx))
            used_dets.add(col_idx)
        else:
            unassigned_tracks.append(track_idx)

    unassigned_detections = [det_idx for det_idx in range(n_dets) if det_idx not in used_dets]

    return {
        "assignments": assignments,
        "unassigned_tracks": unassigned_tracks,
        "unassigned_detections": unassigned_detections,
        "total_cost": float(total_cost),
    }
