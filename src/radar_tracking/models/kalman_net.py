####################################
'''
KalmanNet neural network code adopted from KalmanNet github repo, modifications are done 
on the original code
'''
####################################

import torch
import torch.nn as nn
import torch.nn.functional as func
import math

TWO_PI = 2.0 * math.pi

def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    def _sanitize(self, tensor, clamp_value=100.0):
        tensor = torch.nan_to_num(
            tensor,
            nan=0.0,
            posinf=clamp_value,
            neginf=-clamp_value,
        )
        return torch.clamp(tensor, -clamp_value, clamp_value)

    def _sanitize_hidden(self, tensor):
        return self._sanitize(tensor, clamp_value=50.0)

    def _safe_normalize(self, tensor):
        tensor = self._sanitize(tensor, clamp_value=20.0)
        return func.normalize(tensor, p=2, dim=1, eps=1e-12, out=None)
        
    def apply_film(self, x, gamma, beta, gamma_scale=1, beta_scale=1):
        """
        x, gamma, beta: [1, B, D]
        Bound modulation:
          gamma = 1 + gamma_scale * tanh(gamma)
          beta  = beta_scale * tanh(beta)
        """
        
        g = 1+gamma_scale * torch.tanh(gamma)
        b = beta_scale * torch.tanh(beta)
        
        return x * g + b

        
    def UpdateFilmWt(self, controller_out):
        """
        controller_out: [batch_size, 2*(film_fc7_dim + film_fc5_dim + film_fc6_dim)]
        Produces gamma/beta for FC7, FC5, FC6 as [1, batch_size, dim]
        """
        assert controller_out.dim() == 2, f"controller_out must be [B, D], got {controller_out.shape}"
        B, D = controller_out.shape
        assert B == self.batch_size, f"Batch mismatch: controller_out B={B}, expected {self.batch_size}"
    
        fc7_dim = getattr(self, "film_fc7_dim", self.d_output_FC7)
        fc5_dim = getattr(self, "film_fc5_dim", self.d_output_FC5)
        fc6_dim = getattr(self, "film_fc6_dim", self.d_output_FC6)
    
        expected_D = 2 * (fc7_dim + fc5_dim + fc6_dim)
        assert D == expected_D, f"controller_out dim={D}, expected {expected_D} = 2*(fc7_dim+fc5_dim+fc6_dim)"
    
        split_sizes = [fc7_dim, fc7_dim, fc5_dim, fc5_dim, fc6_dim, fc6_dim]
        gamma7, beta7, gamma5, beta5, gamma6, beta6 = torch.split(controller_out, split_sizes, dim=1)
    
        self.gamma_fc7 = gamma7.unsqueeze(0)
        self.beta_fc7  = beta7.unsqueeze(0)
    
        self.gamma_fc5 = gamma5.unsqueeze(0)
        self.beta_fc5  = beta5.unsqueeze(0)
    
        self.gamma_fc6 = gamma6.unsqueeze(0)
        self.beta_fc6  = beta6.unsqueeze(0)

        
    
    def NNBuild(self, SysModel, args):

        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        # Number of neurons in the 1st hidden layer
        #H1_KNet = (SysModel.m + SysModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        #H2_KNet = (SysModel.m * SysModel.n) * 1 * (4)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)
        
        self.meas_type = args.meas_type

        # ------------------------------------------------------------
        # RANGE_BEARING residual handling toggles
        #
        # Two independent features that can be enabled/disabled:
        #   1) wrap bearing residuals to [-pi, pi)
        #   2) convert bearing residual (rad) to tangential displacement (m)
        # ------------------------------------------------------------
        self.rb_wrap_theta = getattr(args, "knet_rb_wrap_theta", True)
        self.rb_arc_length = getattr(args, "knet_rb_arc_length", True)
        # The benchmark scales meters -> kilometers, so positions live around
        # 20-30 km and range can exceed 40 km. The previous 20.0 clamp was
        # cutting off valid states/measurements and made KalmanNet unusable.
        self.state_clip_value = 100.0
        self.gain_clip_value = 5.0
        self.measurement_clip_value = 100.0
        
        # ------------------------------------------------------------
        # FiLM (Feature-wise Linear Modulation)
        #
        # FiLM layers are constructed below but are intentionally
        # DISABLED for the current benchmark (self.film = False).
        #
        # Reason: the current project scope is a clean single-target
        # benchmark of EKF / KalmanNet / Mamba. The FiLM controller was
        # explored as an extension for adaptive noise estimation but is
        # not part of the reported results.
        #
        # To enable FiLM in a future experiment:
        #   1. Set self.film = True here.
        #   2. Call self.UpdateFilmWt(controller_output) before each
        #      KalmanNet step.
        #   3. Retrain from scratch, because FiLM changes the network's
        #      effective weights.
        # ------------------------------------------------------------
        self.film               = False   # disabled for current benchmark scope
        self.num_film_layers    = 3       # FC5, FC6, FC7 (unused while film=False)
        
        self.film_fc7_dim       = self.d_output_FC7
        self.film_fc5_dim       = self.d_output_FC5
        self.film_fc6_dim       = self.d_output_FC6
        
        # Identity FiLM params (on device)
        self.gamma_fc7 = torch.zeros(1, self.batch_size, self.film_fc7_dim, device=self.device)
        self.beta_fc7  = torch.zeros(1, self.batch_size, self.film_fc7_dim, device=self.device)
        
        self.gamma_fc5 = torch.zeros(1, self.batch_size, self.film_fc5_dim, device=self.device)
        self.beta_fc5  = torch.zeros(1, self.batch_size, self.film_fc5_dim, device=self.device)
        
        self.gamma_fc6 = torch.zeros(1, self.batch_size, self.film_fc6_dim, device=self.device)
        self.beta_fc6  = torch.zeros(1, self.batch_size, self.film_fc6_dim, device=self.device)
                
    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):

        self.seq_len_input = 1 # KNet calculates time-step by time-step
        self.batch_size = args.n_batch # Batch size

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        # GRU to track Q
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
       
        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU()).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)
        
        # Initialize final layer weights to be very small
        # This ensures K starts near 0, making the filter initially stable (pure prediction)        
        with torch.no_grad():
            self.FC2[-1].weight.mul_(0.001)
            self.FC2[-1].bias.zero_()

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU()).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU()).to(self.device)
        
        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU()).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU()).to(self.device)
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU()).to(self.device)        

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        
        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        """
        input M1_0 (torch.tensor): 1st moment of x at time 0 [batch_size, m, 1]
        """
        self.T = T

        self.m1x_posterior = self._sanitize(M1_0.to(self.device), clamp_value=self.state_clip_value)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self._sanitize(
            self.h(self.m1x_posterior),
            clamp_value=self.measurement_clip_value,
        )

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self._sanitize(
            self.f(self.m1x_posterior),
            clamp_value=self.state_clip_value,
        )

        # Predict the 1-st moment of y
        self.m1y = self._sanitize(
            self.h(self.m1x_prior),
            clamp_value=self.measurement_clip_value,
        )

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        y = self._sanitize(y, clamp_value=self.measurement_clip_value)
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(y,2) - torch.squeeze(self.y_previous,2) 
        obs_innov_diff = torch.squeeze(y,2) - torch.squeeze(self.m1y,2)
        if self.meas_type == "RANGE_BEARING":
            # Optionally wrap bearing residuals (rad)
            if self.rb_wrap_theta:
                obs_diff[:, 1] = wrap_to_pi(obs_diff[:, 1])
                obs_innov_diff[:, 1] = wrap_to_pi(obs_innov_diff[:, 1])

            # Optionally convert bearing residual (rad) -> tangential displacement (m)
            if self.rb_arc_length:
                # Cap the range factor in the normalized benchmark regime so the
                # tangentialized bearing residual does not dominate the network
                # features at long sequence lengths.
                r_scale = torch.clamp(torch.squeeze(y, 2)[:, 0], min=1.0, max=5.0)  # [B]
                obs_diff[:, 1] = obs_diff[:, 1] * r_scale
                obs_innov_diff[:, 1] = obs_innov_diff[:, 1] * r_scale
        
        
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.m1x_posterior,2) - torch.squeeze(self.m1x_posterior_previous,2)
        fw_update_diff = torch.squeeze(self.m1x_posterior,2) - torch.squeeze(self.m1x_prior_previous,2)
        
        obs_diff = self._safe_normalize(obs_diff)
        obs_innov_diff = self._safe_normalize(obs_innov_diff)
        fw_evol_diff = self._safe_normalize(fw_evol_diff)
        fw_update_diff = self._safe_normalize(fw_update_diff)
        

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = self._sanitize(
            torch.reshape(KG, (self.batch_size, self.m, self.n)),
            clamp_value=self.gain_clip_value,
        )

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        y = self._sanitize(y, clamp_value=self.measurement_clip_value)

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y  # [batch_size, n, 1]
        if self.meas_type == "RANGE_BEARING":
            # Optionally wrap bearing residuals (rad)
            if self.rb_wrap_theta:
                dy[:, 1, 0] = wrap_to_pi(dy[:, 1, 0])

            # Optionally convert bearing residual (rad) -> tangential displacement (m)
            if self.rb_arc_length:
                # Match the cap used in step_KGain_est so feature scaling stays
                # consistent between gain estimation and state correction.
                r_scale = torch.clamp(y[:, 0, 0], min=1.0, max=5.0)
                dy[:, 1, 0] = dy[:, 1, 0] * r_scale

        # Safety clamp: prevents one extreme innovation component from causing
        # a catastrophic state update in the long-context setting.
        dy = self._sanitize(torch.clamp(dy, -10.0, 10.0), clamp_value=10.0)

        # Compute the 1-st posterior moment
        INOV = self._sanitize(
            torch.bmm(self.KGain, dy),
            clamp_value=self.state_clip_value,
        )
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self._sanitize(
            self.m1x_prior + INOV,
            clamp_value=self.state_clip_value,
        )

        epsilon = y - self._sanitize(
            self.h(self.m1x_posterior),
            clamp_value=self.measurement_clip_value,
        )
        if self.meas_type == "RANGE_BEARING":
            # Optionally wrap bearing residuals (rad)
            if self.rb_wrap_theta:
                epsilon[:, 1, 0] = wrap_to_pi(epsilon[:, 1, 0])

            # Optionally convert bearing residual (rad) -> tangential displacement (m)
            if self.rb_arc_length:
                r_scale = torch.clamp(y[:, 0, 0], min=1.0, max=5.0)
                epsilon[:, 1, 0] = epsilon[:, 1, 0] * r_scale


        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = self._sanitize(y, clamp_value=self.measurement_clip_value)
        
        # Scalars for controller (scale cues)
        dy_mag = torch.log(torch.norm(dy.squeeze(2), dim=1, p=2) + 1e-12)        # [B]
        eps_mag = torch.log(torch.norm(epsilon.squeeze(2), dim=1, p=2) + 1e-12)  # [B]

        dy_mag = self._sanitize(dy_mag.detach(), clamp_value=20.0)
        eps_mag = self._sanitize(eps_mag.detach(), clamp_value=20.0)
        
        # return
        return self.m1x_posterior, dy_mag, eps_mag, self.KGain

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_update_diff
        out_FC5 = self._sanitize(self.FC5(in_FC5), clamp_value=20.0)
        
        ######## FiLM layer FC5 #################
        if self.film is True:
            out_FC5 = self.apply_film(out_FC5, self.gamma_fc5, self.beta_fc5)
        
        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)
        out_Q = self._sanitize_hidden(out_Q)
        self.h_Q = self._sanitize_hidden(self.h_Q)

        # FC 6
        in_FC6 = fw_evol_diff
        out_FC6 = self._sanitize(self.FC6(in_FC6), clamp_value=20.0)
        
        ######## FiLM layer FC6 #################
        if self.film is True:
            out_FC6 = self.apply_film(out_FC6, self.gamma_fc6, self.beta_fc6)
        
        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)
        out_Sigma = self._sanitize_hidden(out_Sigma)
        self.h_Sigma = self._sanitize_hidden(self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self._sanitize(self.FC1(in_FC1), clamp_value=20.0)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self._sanitize(self.FC7(in_FC7), clamp_value=20.0)
        
        ######## FiLM layer FC7 #################
        if self.film is True:
            out_FC7 = self.apply_film(out_FC7, self.gamma_fc7, self.beta_fc7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        out_S = self._sanitize_hidden(out_S)
        self.h_S = self._sanitize_hidden(self.h_S)
        
        #Added to tap S matrix
        self.out_S = out_S
        
        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self._sanitize(self.FC2(in_FC2), clamp_value=self.gain_clip_value)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self._sanitize(self.FC3(in_FC3), clamp_value=20.0)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self._sanitize(self.FC4(in_FC4), clamp_value=20.0)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = self._sanitize_hidden(out_FC4)

        return out_FC2
    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = self._sanitize(y.to(self.device), clamp_value=self.measurement_clip_value)
        return self.KNet_step(y)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1,1, -1).repeat(self.seq_len_input,self.batch_size, 1) # batch size expansion

    def save_internal_state(self):
        self._saved_state = {
            'm1x_posterior': self.m1x_posterior.clone(),
            'm1x_posterior_previous': self.m1x_posterior_previous.clone(),
            'm1x_prior_previous': self.m1x_prior_previous.clone(),
            'y_previous': self.y_previous.clone(),
            'h_S': self.h_S.clone(),
            'h_Sigma': self.h_Sigma.clone(),
            'h_Q': self.h_Q.clone(),
        }
    
    def restore_internal_state(self):
        state = self._saved_state
        self.m1x_posterior = state['m1x_posterior']
        self.m1x_posterior_previous = state['m1x_posterior_previous']
        self.m1x_prior_previous = state['m1x_prior_previous']
        self.y_previous = state['y_previous']
        self.h_S = state['h_S']
        self.h_Sigma = state['h_Sigma']
        self.h_Q = state['h_Q']

