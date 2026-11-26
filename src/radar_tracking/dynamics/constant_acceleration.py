####################################
'''
Data model - constant acceleration
'''
####################################

import torch
import sys
import math

TWO_PI = 2.0 * math.pi

def wrap_to_2pi(angle):
    # [0, 2pi)
    return torch.remainder(angle, TWO_PI)

class ConstantAcceleration2DModel:

    def __init__(self, sim_params):

        # State: [x, vx, ax, y, vy, ay]
        self.m = 6
        self.dt = sim_params.dt
        self.meas_type = sim_params.meas_type

        # Discrete time CA dynamics
        self.F = torch.tensor([
            [1,     self.dt,   0.5*self.dt**2,  0,      0,              0],
            [0,     1,         self.dt,         0,      0,              0],
            [0,     0,         1,               0,      0,              0],
            [0,     0,         0,               1,      self.dt,        0.5*self.dt**2],
            [0,     0,         0,               0,      1,              self.dt],
            [0,     0,         0,               0,      0,              1]
        ])

        # Measurement model
        if self.meas_type == 'FULL':
            self.n = 6
            self.H = torch.eye(self.n)

        elif self.meas_type == 'POS_ONLY':
            self.n = 2
            self.H = torch.zeros(self.n, self.m)
            self.H[0, 0] = 1.0  # x
            self.H[1, 3] = 1.0  # y

        elif self.meas_type == 'RANGE_BEARING':
            self.n = 2
            self.H = None  # nonlinear

        else:
            print("Unknown meas_type")
            sys.exit(1)

        # initializations for KNet
        self.prior_Q     = torch.diag(torch.ones(self.m))
        self.prior_Sigma = torch.diag(torch.ones(self.m))
        self.prior_S     = torch.diag(torch.ones(self.n))

    def f(self, x):
        F = self.F.to(x.device)
        return F.unsqueeze(0) @ x

    def h(self, x):
        if self.meas_type in ("FULL", "POS_ONLY"):
            H = self.H.to(x.device)
            return H.unsqueeze(0) @ x

        px = x[:, 0, 0]
        py = x[:, 3, 0]
        r = torch.sqrt(px*px + py*py + 1e-12)
        theta = torch.atan2(py, px)          # [-pi, pi]
        theta = wrap_to_2pi(theta)           # [0, 2pi)

        y = torch.stack([r, theta], dim=1).unsqueeze(-1)
        return y

    def H_jacobian(self, x):
        B = x.shape[0]

        if self.meas_type in ("FULL", "POS_ONLY"):
            H = self.H.to(x.device)
            return H.unsqueeze(0).repeat(B, 1, 1)

        px = x[:, 0, 0]
        py = x[:, 3, 0]
        r2 = px*px + py*py + 1e-12
        r = torch.sqrt(r2)

        H = torch.zeros(B, 2, self.m, device=x.device)
        H[:, 0, 0] = px / r
        H[:, 0, 3] = py / r
        H[:, 1, 0] = -py / r2
        H[:, 1, 3] =  px / r2
        return H
