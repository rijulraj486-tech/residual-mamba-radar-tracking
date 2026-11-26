import numpy as np
import torch
import copy


def _grads_are_finite(model):
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def _tensor_is_finite(value):
    return torch.isfinite(value).all().item()


def evaluate_loss_knet(kalmannet, x_pool, y_pool, init_val, args, loss_fn=torch.nn.MSELoss(reduction='mean')):
    # Evaluate loss on a dataset
    kalmannet.eval()
    kalmannet.film = False
    eval_seq_len = x_pool.shape[2]
    Kfro_list = []
    log_norm_innov_list = []
    
    with torch.no_grad():
        kalmannet.batch_size = x_pool.shape[0]
        kalmannet.init_hidden_KNet()
        kalmannet.InitSequence(init_val, eval_seq_len)
        x_est = torch.zeros_like(x_pool)
        
        # Iterate through time steps (Validation uses full sequence length)
        for tidx in range(eval_seq_len):
            temp_in = y_pool[:, :, tidx].unsqueeze(2)
            temp, log_norm_innov, _, K_t = kalmannet(temp_in)
            x_est[:, :, tidx] = temp.squeeze(-1)
            
            # Frobenius norm, averaged over batch, for debug
            Kfro = torch.norm(K_t.reshape(K_t.shape[0], -1), dim=1)  # [B]
            Kfro_list.append(Kfro.mean().item())
            log_norm_innov_list.append(log_norm_innov.mean().item())
        
        # Calculate validation loss on the full sequence
        loss = loss_fn(x_est, x_pool)
        
        return loss.item(), x_est, Kfro_list, log_norm_innov_list

def train_validate_knet(model, kalmannet, args, x_train, y_train, x_val, y_val, init_train, init_val, filename):
 
    # Long-context KalmanNet training is substantially less stable than the
    # original short-sequence setup, so we use a lower effective LR once the
    # sequence length moves beyond the reference regime.
    if args.seq_len > 100:
        seq_ratio = max(float(args.seq_len) / 100.0, 1.0)
        long_seq_lr = args.learning_rate / (seq_ratio ** 2)
        long_seq_lr = max(long_seq_lr, 1e-5)
    else:
        long_seq_lr = args.learning_rate

    optimizer = torch.optim.Adam(kalmannet.parameters(),
                                 lr=long_seq_lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8,
        min_lr=1e-7
    )
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    # Early stopping params
    patience = args.stopping_criteria 
    min_delta_db = args.update_thr_db 

    best_val_loss = float('inf')
    best_epoch = -1
    best_model = None

    no_improve_counter = 0
    dataset_size = x_train.shape[0]
    n_batches = max(1, dataset_size // args.n_batch)
    val_loss_traindB = []
    
    # ------------------------------------------------------------------------
    # TBPTT Parameters 
    # k: Detach gradients every k steps (prevents explosion)
    # w: Update weights every w steps (sliding window)
    # Recommended values from paper: k=2, w=4 (or w=8)
    # ------------------------------------------------------------------------
    k_steps = 2
    w_steps = 4 
    if args.seq_len > 200:
        k_steps = 1
        w_steps = 2

    for epoch in range(args.epochs):
        kalmannet.train()
        kalmannet.film = False
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches_updated = 0 

        nan_steps_count = 0 
        
        # Full pass through dataset
        for i in range(0, dataset_size, args.n_batch):
            idxs = permutation[i:i + args.n_batch]
            ts_batch, obs_batch, init_batch = x_train[idxs], y_train[idxs], init_train[idxs]

            kalmannet.batch_size = ts_batch.shape[0]
            kalmannet.init_hidden_KNet()
            kalmannet.InitSequence(init_batch, args.seq_len)

            optimizer.zero_grad()
            
            # Accumulators for the sliding window
            current_window_loss = 0.0
            window_steps = 0
            batch_total_loss = 0.0
            
            debug_vec = torch.zeros_like(ts_batch)
            
            # Iterate over sequence length (Time T)
            seq_len = ts_batch.shape[2]
            
            for tidx in range(seq_len):
                # Forward Step
                # Input: [Batch, Dim, 1]
                kalmannet.save_internal_state()
                temp, _, _, _ = kalmannet(obs_batch[:, :, tidx].unsqueeze(2))
                if (
                    not _tensor_is_finite(temp)
                    or not _tensor_is_finite(kalmannet.m1x_posterior)
                    or not _tensor_is_finite(kalmannet.KGain)
                ):
                    kalmannet.restore_internal_state()
                    nan_steps_count += 1
                    continue
                
                target = ts_batch[:, :, tidx]
                
                # Calculate step loss
                step_loss = loss_fn(temp.squeeze(-1), target)
                
                if not torch.isfinite(step_loss):
                    kalmannet.restore_internal_state()
                    nan_steps_count += 1
                    continue 

                current_window_loss += step_loss
                window_steps += 1
                
                # TBPTT: Detach Gradients (k)
                # Detach hidden state every k steps
                # tidx is 0 indexed, so check (tidx + 1)
                if (tidx + 1) % k_steps == 0:
                    # Architecture 2 Hidden States
                    if hasattr(kalmannet, 'h_S'):
                        kalmannet.h_S = kalmannet.h_S.detach()
                    if hasattr(kalmannet, 'h_Sigma'):
                        kalmannet.h_Sigma = kalmannet.h_Sigma.detach()
                    if hasattr(kalmannet, 'h_Q'):
                        kalmannet.h_Q = kalmannet.h_Q.detach()
                    
                    # Architecture 1 Hidden State
                    if hasattr(kalmannet, 'h_gru'):
                        kalmannet.h_gru = kalmannet.h_gru.detach()
                        
                    if hasattr(kalmannet, 'm1x_posterior'):
                        kalmannet.m1x_posterior = kalmannet.m1x_posterior.detach()
                    if hasattr(kalmannet, 'm1x_posterior_previous'):
                        kalmannet.m1x_posterior_previous = kalmannet.m1x_posterior_previous.detach()
                    if hasattr(kalmannet, 'm1x_prior_previous'):
                        kalmannet.m1x_prior_previous = kalmannet.m1x_prior_previous.detach()
                    if hasattr(kalmannet, 'y_previous'):
                        kalmannet.y_previous = kalmannet.y_previous.detach()

                # Update weights every w steps OR at the end of the sequence
                if (tidx + 1) % w_steps == 0 or (tidx + 1) == seq_len:
                    
                    if window_steps > 0:
                        # Average loss over the window
                        loss_to_backprop = current_window_loss / window_steps
                        
                        if not torch.isfinite(loss_to_backprop) or loss_to_backprop.item() > 1e6:
                            optimizer.zero_grad()
                            nan_steps_count += window_steps
                            current_window_loss = 0.0
                            window_steps = 0
                            continue

                        loss_to_backprop.backward()

                        if not _grads_are_finite(kalmannet):
                            optimizer.zero_grad()
                            nan_steps_count += window_steps
                            current_window_loss = 0.0
                            window_steps = 0
                            continue
                        
                        # Gradient Clipping
                        torch.nn.utils.clip_grad_norm_(kalmannet.parameters(), max_norm=0.1)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # total batch loss for info only
                        batch_total_loss += loss_to_backprop.item()
                    
                    # Reset
                    current_window_loss = 0.0
                    window_steps = 0
            
            # End of batch
            if seq_len > 0:
                epoch_loss += batch_total_loss / (seq_len / w_steps) # Approx average
                batches_updated += 1
            else:
                print(f"  Epoch {epoch+1} Batch {i//args.n_batch}: All steps NaN or empty.")

        # Compute average epoch loss
        if batches_updated > 0:
            epoch_loss /= batches_updated
        else:
            epoch_loss = float('nan') 

        # Validation (Always on full sequence length, no truncation)
        val_loss, _, _, _ = evaluate_loss_knet(kalmannet, x_val, y_val, init_val, args, loss_fn)
        if np.isfinite(val_loss):
            scheduler.step(val_loss)
        
        train_db = 10*np.log10(epoch_loss) if epoch_loss > 0 else -100
        val_db = 10*np.log10(val_loss) if np.isfinite(val_loss) and val_loss > 0 else float('nan')
        
        print(f"Epoch {epoch+1}, Train Loss(dB): {train_db:.6f}, Val Loss(dB): {val_db:.6f}")
        
        if nan_steps_count > 0:
            print(f"  Epoch {epoch+1}: Found {nan_steps_count} NaN steps.")

        val_loss_traindB.append(val_db)
        
        # Improvement check
        current_best_db = 10*np.log10(best_val_loss) if best_val_loss != float('inf') else float('inf')
        
        if val_db < current_best_db - min_delta_db:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model = copy.deepcopy(kalmannet.state_dict())
            torch.save(best_model, filename)
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        # Early stopping condition
        if no_improve_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without validation improvement.")
            break

    print(f"\nBest Validation Loss(dB): {10*np.log10(best_val_loss):.6f} at Epoch {best_epoch}")
    if best_model is not None:
        kalmannet.load_state_dict(best_model)
        print(f"Best model loaded from memory and saved to {filename}.")
    else:
        print("Warning: no improvement observed, final weights used.")

    return kalmannet, val_loss_traindB
