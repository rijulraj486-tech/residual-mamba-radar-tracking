import os
import torch


# Parameters used for the simulation
class simparams_class:
    def __init__(self):
        self.use_cuda           = torch.cuda.is_available()
        self.dt                 = 1
        self.n_batch            = 64
        self.in_mult_KNet       = 5 #determines the number of paramters Knet
        self.out_mult_KNet      = 40 #determines the number of paramters Knet
        self.learning_rate      = 1e-3
        self.weight_decay       = 1e-4
        # Default to the original long-sequence reference setting.
        self.seq_len            = int(os.environ.get("SIM_SEQ_LEN", 100))
        self.q2_scale           = 1
        self.r2_scale           = 1
        self.meas_type          = "RANGE_BEARING" #FULL/POS_ONLY/RANGE_BEARING
        self.init_type          = "MEAS" #MEAS/GTS
        self.stopping_criteria  = 20
        self.update_thr_db      = 0.01
        self.dataset_size       = int(os.environ.get("SIM_DATASET_SIZE", 512))
        self.epochs             = 1000
        self.train_knet         = True
        self.knet_name          = "best_knet.pt"
        self.knet_rb_wrap_theta = os.environ.get("KNET_RB_WRAP_THETA", "1").strip().lower() in ("1", "true", "yes", "y")
        self.knet_rb_arc_length = os.environ.get("KNET_RB_ARC_LENGTH", "1").strip().lower() in ("1", "true", "yes", "y")
        self.noise_ref          = "FIXED" #SNR/FIXED
        self.range_std          = 30 #in m
        self.bearing_std        = 1 # in degrees
      

def validate_simparams(sim_params):
    assert sim_params.meas_type in ("FULL", "POS_ONLY", "RANGE_BEARING"), "Invalid measurement type argument"
    assert sim_params.init_type in ("MEAS", "GTS"), "Invalid init_type type argument"
    assert sim_params.noise_ref in ("FIXED", "SNR"), "Invalid noise_ref type argument"

        
