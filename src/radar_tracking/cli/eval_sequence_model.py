import torch

from radar_tracking.cli.benchmark_mamba import prepare_real_world_split


def evaluate_sequence_model(model, dataset, seq_len, stride):
    """Evaluate ground-truth-initialized NCLT windows with a radar model.

    ``dataset`` is the prepared dataset file path. Each window is initialized
    independently from its first ground-truth state, and no predicted state is
    carried into the next window. Overlapping windows may therefore repeat
    time positions in the aggregate RMSE.
    """
    state_windows, obs_windows, init_windows, _, _ = prepare_real_world_split(
        dataset,
        seq_len,
        stride,
        allow_empty=False,
    )

    device = next(model.parameters()).device
    batch_size = 32
    total_sq = torch.tensor(0.0, device=device)
    total_n = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, obs_windows.shape[0], batch_size):
            end = min(start + batch_size, obs_windows.shape[0])
            observations = obs_windows[start:end].to(device)
            initial_states = init_windows[start:end].to(device)
            state_estimates = model(
                observations=observations,
                init_state=initial_states,
            )["state_estimates"]

            predicted_positions = state_estimates[:, [0, 3], :]
            target_positions = state_windows[start:end, [0, 3], :].to(device)
            total_sq += (predicted_positions - target_positions).pow(2).sum()
            total_n += target_positions.shape[0] * target_positions.shape[2]

    rmse_scaled = torch.sqrt(total_sq / total_n)
    rmse_meters = rmse_scaled * 1000.0
    print(f"Average RMSE: {rmse_meters.item():.4f}")
    return rmse_meters
