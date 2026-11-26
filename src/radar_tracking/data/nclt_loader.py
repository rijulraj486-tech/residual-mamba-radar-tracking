import torch


def extract_xy(x):
    return torch.as_tensor(x, dtype=torch.float32)[..., :2]


def compute_velocity_acceleration(pos):
    pos = torch.as_tensor(pos, dtype=torch.float32)
    vel = torch.diff(pos, dim=0, prepend=pos[:1])
    acc = torch.diff(vel, dim=0, prepend=vel[:1])
    return vel, acc


def adapt_observation(obs):
    obs = extract_xy(obs)
    return obs.repeat(1, 3).unsqueeze(0)


def add_observation_noise(obs, std=0.5):
    obs = extract_xy(obs)
    return obs + std * torch.randn_like(obs)


def drop_observations(obs, drop_rate=0.2):
    obs = torch.as_tensor(obs, dtype=torch.float32)
    mask = torch.rand(obs.shape[0], device=obs.device) > drop_rate
    return obs * mask.unsqueeze(1)


def split_windows(obs, state, window_size=100, stride=50):
    obs = torch.as_tensor(obs)
    state = torch.as_tensor(state)
    windows = []
    for start in range(0, obs.shape[0] - window_size + 1, stride):
        end = start + window_size
        windows.append((obs[start:end], state[start:end]))
    return windows


def build_state(xy):
    pos = extract_xy(xy)
    vel, acc = compute_velocity_acceleration(pos)
    return torch.stack((pos[:, 0], vel[:, 0], acc[:, 0], pos[:, 1], vel[:, 1], acc[:, 1]), dim=1)


def load_nclt_dataset(path):
    return [
        (extract_xy(item["filtered_gps"]), build_state(item["ground_truth"]))
        for item in torch.load(path, map_location="cpu")
    ]
