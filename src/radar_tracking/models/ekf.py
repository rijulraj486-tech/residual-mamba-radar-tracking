####################################
'''
EKF implementation
'''
####################################
import torch
import sys
import math


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class extended_kf:
    def __init__(self, model, Q, R, sim_params):

        self.model = model
        self.meas_type = sim_params.meas_type
        
        dataset_size = sim_params.dataset_size
        
        self.Q = torch.zeros(dataset_size, self.model.m, self.model.m)
        self.R = torch.zeros(dataset_size, self.model.n, self.model.n)

        for bidx in range(dataset_size):
            self.Q[bidx,:,:] = Q
            self.R[bidx,:,:] = R

        self.x = torch.zeros(dataset_size,self.model.m,1)
        self.P = torch.zeros(dataset_size, self.model.m, self.model.m)

    def reset(self, x_init):
        self.x[:,:,0] = x_init.squeeze(-1)
        
        if self.meas_type == "FULL":
            init_cov = self.R[0,:,:]
        elif self.meas_type == "POS_ONLY":
            init_cov = torch.eye(self.model.m)*1e4
            for i in range(self.model.n):
                cols = (self.model.H[i] == 1).nonzero(as_tuple=False)
                j = cols[0].item()
                init_cov[j,j] = self.R[0,i,i]
        elif self.meas_type == "RANGE_BEARING":
            # large uncertainty
            init_cov = torch.eye(self.model.m) * 1e4

            # use current init position to estimate (r,theta)
            px = self.x[:, 0, 0]
            py = self.x[:, 3, 0]
            r2 = px*px + py*py + 1e-12
            r = torch.sqrt(r2)

            sig_r2 = self.R[0,0,0]
            sig_th2 = self.R[0,1,1]

            # Var(x) = cos^2 th * sig_r2 + (r^2 sin^2 th) * sig_th2
            # Var(y) = sin^2 th * sig_r2 + (r^2 cos^2 th) * sig_th2
            th = torch.atan2(py, px)

            var_x = (torch.cos(th)**2) * sig_r2 + (r**2) * (torch.sin(th)**2) * sig_th2
            var_y = (torch.sin(th)**2) * sig_r2 + (r**2) * (torch.cos(th)**2) * sig_th2

            for b_idx in range(self.x.shape[0]):
                init_cov_b = init_cov.clone()
                init_cov_b[0,0] = var_x[b_idx].item()
                init_cov_b[3,3] = var_y[b_idx].item()
                self.P[b_idx,:,:] = init_cov_b
            self.K = 0; self.S = 0; self.nu = 0
            return
        else:
            print("Unknown meas_type")
            sys.exit(-1)

        
        for b_idx in range(x_init.shape[0]):
            self.P[b_idx,:,:] = init_cov             
        
        self.K = 0
        self.S = 0
        self.nu = 0

    def f(self, x):
        return self.model.f(x)

    def h(self, x):
        return self.model.h(x)

    def predict(self):
        self.x = self.f(self.x)
        self.P = self.model.F @ self.P @ self.model.F.T + self.Q

    def update(self, z):
        z_pred = self.h(self.x).squeeze(-1)   # [B,n]
        H = self.model.H_jacobian(self.x)     # [B,n,m]
        y = (z - z_pred)                      # [B,n]

        # wrap bearing residual to [-pi, pi)
        if self.meas_type == "RANGE_BEARING":
            y[:, 1] = wrap_to_pi(y[:, 1])

        y = y.unsqueeze(-1)                   # [B,n,1]
        self.nu = y.squeeze(-1)  
        
        temp = torch.matmul(H, self.P)
        self.S = torch.matmul(temp, torch.transpose(H,1,2)) + self.R
        
        temp = torch.matmul(self.P, torch.transpose(H,1,2))
        self.K = torch.matmul(temp, torch.linalg.inv(self.S))
        
        self.x = self.x + torch.matmul(self.K, y)
        
        I = torch.eye(self.model.m).expand_as(self.P)                 # [B, m, m]
        IKH = I - torch.matmul(self.K, H)                                           # [B, m, m]
        KRKT = torch.matmul(self.K, torch.matmul(self.R, self.K.transpose(1, 2)))   # [B, m, m]
        self.P = torch.matmul(torch.matmul(IKH, self.P), IKH.transpose(1, 2)) + KRKT
           
    
class batch_ekf:
    def __init__(self, ekf_class):  
        self.ekf_obj = ekf_class

    def evaluate_batch(self, true_state, measurements, init_state):
        batch_size, obs_dim, seq_len = measurements.shape
        state_dim = self.ekf_obj.model.m

        estimates = torch.zeros(batch_size, state_dim, seq_len)
        P_seq     = torch.zeros(batch_size, state_dim, state_dim, seq_len)
        K_seq     = torch.zeros(batch_size, state_dim, obs_dim, seq_len)
        S_seq     = torch.zeros(batch_size, obs_dim, obs_dim, seq_len)
        NU_seq    = torch.zeros(batch_size, obs_dim, seq_len)
        
        self.ekf_obj.reset(init_state)
 
        for t_idx in range(seq_len):
            self.ekf_obj.predict()
            self.ekf_obj.update(measurements[:, :, t_idx])
        
            estimates[:, :, t_idx] = self.ekf_obj.x.squeeze(-1)
            P_seq[:, :, :, t_idx]  = self.ekf_obj.P
            K_seq[:, :, :, t_idx]  = self.ekf_obj.K
            S_seq[:, :, :, t_idx]  = self.ekf_obj.S
            NU_seq[:, :, t_idx]    = self.ekf_obj.nu

        return estimates, P_seq, K_seq, S_seq, NU_seq

def compute_lossvstime(truestate_dataset, ekf_estimates):
    ekf_loss_fn = torch.nn.MSELoss(reduction='mean')
    
    total_loss_dB = 10*torch.log10(ekf_loss_fn(truestate_dataset, ekf_estimates))
    
    seq_len = ekf_estimates.shape[2]
    
    loss_vs_time_dB = torch.zeros(seq_len)
    for idx in range(seq_len):
        loss_vs_time_dB[idx] = 10*torch.log10(ekf_loss_fn(truestate_dataset[:,:,idx], ekf_estimates[:,:,idx]))
        
    return loss_vs_time_dB, total_loss_dB
        
def mse_per_dim(est, true):
    # est/true: [B, m, T]
    mse_dim = torch.mean((est - true)**2, dim=(0, 2))   # [m]
    mse_dim_db = 10*torch.log10(mse_dim)
    return mse_dim_db
        
    
    
    

