"""
Generate synthetic trajectories and observations for the Constant-Acceleration model.

Noise distributions - Gaussian and Laplacian (double-exponential) noise for measurement 
and process noise.
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import sys
import random
import math

TWO_PI = 2.0 * math.pi

# Noise selection flags, Choose: "gaussian" | "laplace"
MEAS_NOISE_DIST = "gaussian"
PROC_NOISE_DIST = "gaussian"

# compute coarse velocity from first two measurements for initialisation
RB_INIT_VEL_FROM_TWO_MEAS = True

RB_INIT_VEL_CLIP = 300.0

REFERENCE_SEQ_LEN = 100.0

def wrap_to_2pi(angle):
    return torch.remainder(angle, TWO_PI)


def get_motion_profile(sim_params):
    """
    Keep the long-context benchmark in the same physical regime as the original
    100-step setup.

    With constant-acceleration dynamics, extending the context by 10x increases
    the velocity-driven displacement by 10x and the acceleration-driven
    displacement by 100x if we keep the same initial state ranges. That makes
    the 1000-step trajectories numerically extreme and unfair to models trained
    on the original regime. We therefore shrink the velocity, acceleration, and
    process-noise scales as the sequence gets longer.
    """
    seq_len = max(float(sim_params.seq_len), 1.0)
    seq_ratio = max(seq_len / REFERENCE_SEQ_LEN, 1.0)

    vel_scale = 1.0 / seq_ratio
    acc_scale = 1.0 / (seq_ratio ** 2)

    return {
        "pos_min": 20000.0,
        "pos_max": 30000.0,
        "vel_min": -200.0 * vel_scale,
        "vel_max": 200.0 * vel_scale,
        "accn_min": -3.0 * acc_scale,
        "accn_max": 3.0 * acc_scale,
        "accn_rms_q": 1.0 * acc_scale,
    }


def Q_CA_1D(dt, q):
    dt = torch.as_tensor(dt)
    dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
    Q = torch.tensor([[dt5/20, dt4/8, dt3/6],
                      [dt4/8,  dt3/3, dt2/2],
                      [dt3/6,  dt2/2, dt     ]])
    return q * Q


def Q_CA_2D(dt, qx, qy):
    return torch.block_diag(Q_CA_1D(dt, qx),
                            Q_CA_1D(dt, qy))


def sample_laplace_like(std_dev_vec: torch.Tensor, num_batch: int, device=None):
    """
    Sample iid Laplace(0, b) noise with per-dimension std given by std_dev_vec.
    For Laplace(0,b): std = sqrt(2)*b  =>  b = std/sqrt(2)
    """
    if device is None:
        device = std_dev_vec.device
    std = std_dev_vec.to(device)                       # [D]
    b = std / torch.sqrt(torch.tensor(2.0, device=device))
    D = std.numel()

    U = torch.rand(num_batch, D, device=device) - 0.5
    X = -b * torch.sign(U) * torch.log1p(-2.0 * torch.abs(U))
    return X.unsqueeze(-1)                             # [B, D, 1]


def generate_synthetic_data(model, sim_params, snr_dB, seq_mul_fact=1, batch_size=None):

    num_batch = sim_params.dataset_size if batch_size is None else batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_profile = get_motion_profile(sim_params)

    # Initial state values
    pos_min, pos_max = motion_profile["pos_min"], motion_profile["pos_max"]
    vel_min, vel_max = motion_profile["vel_min"], motion_profile["vel_max"]
    accn_min, accn_max = motion_profile["accn_min"], motion_profile["accn_max"]

    accn_rms_q = motion_profile["accn_rms_q"]  # to compute process noise

    pos_var = (pos_max - pos_min)**2 / 12
    vel_var = (vel_max - vel_min)**2 / 12
    accn_var = (accn_max - accn_min)**2 / 12

    snr_linear = 10**(snr_dB/10)

    pos_noise_var = pos_var / snr_linear
    vel_noise_var = vel_var / snr_linear
    accn_noise_var = accn_var / snr_linear

    # error RMS
    pos_rms = pos_noise_var ** 0.5
    vel_rms = vel_noise_var ** 0.5
    accn_rms = accn_noise_var ** 0.5

    # process noise (jerk variance)
    jerk_var = (accn_rms_q ** 2) / model.dt
    process_noise = Q_CA_2D(model.dt, jerk_var, jerk_var)
    Q = process_noise * sim_params.q2_scale

    if sim_params.meas_type == 'POS_ONLY':
        std_dev_vec = torch.tensor([pos_rms, pos_rms], device=device)

    elif sim_params.meas_type == 'FULL':
        std_dev_vec = torch.tensor([pos_rms, vel_rms, accn_rms,
                                    pos_rms, vel_rms, accn_rms], device=device)

    elif sim_params.meas_type == 'RANGE_BEARING':
        if sim_params.noise_ref == "SNR":
            r_typical = ((pos_min + pos_max) / 2.0)
            range_rms = pos_rms
            bearing_rms = pos_rms / r_typical
        else:
            range_rms = sim_params.range_std
            bearing_rms = sim_params.bearing_std * torch.pi / 180.0

        std_dev_vec = torch.tensor([range_rms, bearing_rms], device=device)
        print(f"Range rms = {range_rms:.3f} m, Bearing rms = {bearing_rms*180/torch.pi:.3f} deg")

    else:
        print("model.meas_type must be FULL, POS_ONLY, or RANGE_BEARING")
        sys.exit(1)

    R = torch.diag(std_dev_vec**2) * sim_params.r2_scale

    proc_std_vec = torch.sqrt(torch.diag(Q)).to(device)  # [m]

    # Gaussian distributions (kept for gaussian mode with full covariance)
    meas_dist = MultivariateNormal(loc=torch.zeros(model.n, device=device), covariance_matrix=R.to(device))
    proc_dist = MultivariateNormal(loc=torch.zeros(model.m, device=device), covariance_matrix=Q.to(device))

    # one more iteration is used to generate an initialisation vector
    L = int(sim_params.seq_len * seq_mul_fact) + 1
    x_seq = torch.zeros(num_batch, model.m, L, device=device)
    y_seq = torch.zeros(num_batch, model.n, L, device=device)

    # init state
    x_init = torch.zeros(num_batch, model.m, 1, device=device)
    x_init[:, 0, 0] = pos_min + (pos_max - pos_min) * torch.rand(num_batch, device=device)
    x_init[:, 1, 0] = vel_min + (vel_max - vel_min) * torch.rand(num_batch, device=device)
    x_init[:, 2, 0] = accn_min + (accn_max - accn_min) * torch.rand(num_batch, device=device)
    x_init[:, 3, 0] = pos_min + (pos_max - pos_min) * torch.rand(num_batch, device=device)
    x_init[:, 4, 0] = vel_min + (vel_max - vel_min) * torch.rand(num_batch, device=device)
    x_init[:, 5, 0] = accn_min + (accn_max - accn_min) * torch.rand(num_batch, device=device)

    init_vec = None
    x_prev = x_init

    for t in range(L):
 
        # state evolution
        x_t = model.f(x_prev)

        # process noise
        if PROC_NOISE_DIST.lower() == "gaussian":
            eq = proc_dist.rsample((num_batch,)).unsqueeze(-1)  # [B,m,1]
        elif PROC_NOISE_DIST.lower() == "laplace":
            eq = sample_laplace_like(proc_std_vec, num_batch, device=device)  # [B,m,1]
        else:
            raise ValueError(f"Unknown PROC_NOISE_DIST: {PROC_NOISE_DIST}")

        x_t = x_t + eq

        # measurement
        y_t = model.h(x_t)

        # measurement noise
        meas_std_vec = torch.sqrt(torch.diag(R)).to(device)  # [n]

        if MEAS_NOISE_DIST.lower() == "gaussian":
            er = meas_dist.rsample((num_batch,)).unsqueeze(-1)  # [B,n,1]
        elif MEAS_NOISE_DIST.lower() == "laplace":
            er = sample_laplace_like(meas_std_vec, num_batch, device=device)  # [B,n,1]
        else:
            raise ValueError(f"Unknown MEAS_NOISE_DIST: {MEAS_NOISE_DIST}")

        y_t = y_t + er

        # wrap bearing to [0, 2pi) for RANGE_BEARING
        if sim_params.meas_type == 'RANGE_BEARING':
            y_t[:, 1, 0] = wrap_to_2pi(y_t[:, 1, 0])

        x_seq[:, :, t] = x_t.squeeze(-1)
        y_seq[:, :, t] = y_t.squeeze(-1)
        x_prev = x_t

        # init vector (MEAS) computed once when we have at least t>=1
        if (init_vec is None) and (sim_params.init_type == 'MEAS') and (t >= 1):
            init_vec = torch.zeros_like(x_init)

            if sim_params.meas_type in ("FULL", "POS_ONLY"):
                for idx in range(model.n):
                    col_idx = (model.H[idx, :] == 1).nonzero(as_tuple=True)[0]
                    init_vec[:, col_idx, :] = y_seq[:, idx, 0].unsqueeze(-1).unsqueeze(-1)

            elif sim_params.meas_type == "RANGE_BEARING":
                r0  = y_seq[:, 0, 0]
                th0 = y_seq[:, 1, 0]
                init_vec[:, 0, 0] = r0 * torch.cos(th0)
                init_vec[:, 3, 0] = r0 * torch.sin(th0)

                if RB_INIT_VEL_FROM_TWO_MEAS:
                    r1  = y_seq[:, 0, 1]
                    th1 = y_seq[:, 1, 1]
                    x0 = init_vec[:, 0, 0]
                    y0 = init_vec[:, 3, 0]
                    x1 = r1 * torch.cos(th1)
                    y1 = r1 * torch.sin(th1)
                    vx0 = (x1 - x0) / model.dt
                    vy0 = (y1 - y0) / model.dt
                    if RB_INIT_VEL_CLIP is not None:
                        vx0 = torch.clamp(vx0, -RB_INIT_VEL_CLIP, RB_INIT_VEL_CLIP)
                        vy0 = torch.clamp(vy0, -RB_INIT_VEL_CLIP, RB_INIT_VEL_CLIP)
                    init_vec[:, 1, 0] = vx0
                    init_vec[:, 4, 0] = vy0
                # ax, ay is initialised to zero

                init_vec = init_vec.unsqueeze(-1) if init_vec.dim() == 2 else init_vec
            else:
                print("Unknown meas_type in init")
                sys.exit(1)

        elif sim_params.init_type == 'GTS':
            init_vec = x_seq[:, :, 0].unsqueeze(-1)

    x_seq = x_seq[:, :, 1:]
    y_seq = y_seq[:, :, 1:]

    return x_seq, y_seq, init_vec, Q, R


def split_data(x_true, y_obs, init_state, train_frac=0.8, val_frac=0.2):
    N = x_true.shape[0]
    idxs = list(range(N))
    random.shuffle(idxs)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train+n_val]

    def select(idxs):
        return x_true[idxs], y_obs[idxs], init_state[idxs]

    return select(train_idx), select(val_idx)
