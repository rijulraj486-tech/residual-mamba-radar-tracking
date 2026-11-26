import numpy as np

from radar_tracking.multitarget.assignment import solve_gnn_assignment


def gospa_metric(true_points, est_points, cutoff=2.0, order=2, alpha=2.0):
    true_points = np.asarray(true_points, dtype=np.float64).reshape(-1, 2)
    est_points = np.asarray(est_points, dtype=np.float64).reshape(-1, 2)

    n_true = len(true_points)
    n_est = len(est_points)

    if n_true == 0 and n_est == 0:
        return {
            "gospa": 0.0,
            "localization": 0.0,
            "cardinality": 0.0,
            "missed": 0,
            "false": 0,
        }

    if n_true == 0:
        cardinality = (cutoff ** order / alpha) * n_est
        return {
            "gospa": cardinality ** (1.0 / order),
            "localization": 0.0,
            "cardinality": cardinality,
            "missed": 0,
            "false": n_est,
        }

    if n_est == 0:
        cardinality = (cutoff ** order / alpha) * n_true
        return {
            "gospa": cardinality ** (1.0 / order),
            "localization": 0.0,
            "cardinality": cardinality,
            "missed": n_true,
            "false": 0,
        }

    swapped = False
    left = true_points
    right = est_points
    if n_true > n_est:
        swapped = True
        left, right = right, left
        n_true, n_est = n_est, n_true

    pairwise = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    capped = np.minimum(pairwise, cutoff) ** order
    assignment = solve_gnn_assignment(capped, miss_cost=cutoff ** order)

    localization = 0.0
    matches = 0
    for row_idx, col_idx in assignment["assignments"]:
        localization += capped[row_idx, col_idx]
        if pairwise[row_idx, col_idx] < cutoff:
            matches += 1

    cardinality = (cutoff ** order / alpha) * abs(len(est_points) - len(true_points))
    total = localization + cardinality

    if swapped:
        missed = len(est_points) - matches
        false = len(true_points) - matches
    else:
        missed = len(true_points) - matches
        false = len(est_points) - matches

    return {
        "gospa": total ** (1.0 / order),
        "localization": localization,
        "cardinality": cardinality,
        "missed": int(max(0, missed)),
        "false": int(max(0, false)),
    }
