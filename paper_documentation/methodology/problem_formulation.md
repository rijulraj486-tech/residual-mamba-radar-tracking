# Problem formulation

## State and dynamics

The implemented target state is `x_t = [p_x,v_x,a_x,p_y,v_y,a_y]^T`, with position in metres, velocity in m/s, acceleration in m/s² in the physical simulator; saved Stage 2 state coordinates and range are divided by 1,000, while bearing stays in radians. State tensors are `[N,6,T]`; returned initializer is `[N,6,1]`; radar observation tensor is `[N,2,T]` in `[range/1000,bearing_rad]` order. The explicit state order and `dt=1 s` are in `src/radar_tracking/dynamics/constant_acceleration.py` and Stage 2 generation manifests.

For `dt=Δt`, each axis follows constant acceleration:

\[
[p_{t+1},v_{t+1},a_{t+1}]^T =
\begin{bmatrix}1&Δt&Δt^2/2\\0&1&Δt\\0&0&1\end{bmatrix}[p_t,v_t,a_t]^T+w_t.
\]

In the implemented coordinate order the exact 6×6 matrix is

\[F=\begin{bmatrix}1&Δt&Δt^2/2&0&0&0\\0&1&Δt&0&0&0\\0&0&1&0&0&0\\0&0&0&1&Δt&Δt^2/2\\0&0&0&0&1&Δt\\0&0&0&0&0&1\end{bmatrix}.\]

It is not a constant-velocity model.

The process disturbance is white jerk with intensity `q` per axis. `Q_CA_1D(dt,q)=q[[dt^5/20,dt^4/8,dt^3/6],[dt^4/8,dt^3/3,dt²/2],[dt³/6,dt²/2,dt]]`; `Q=diag_block(Q_CA_1D,Q_CA_1D)` in `[x,vx,ax,y,vy,ay]` order, then multiplied by `q2_scale`. Gaussian process noise is selected by the module constant `PROC_NOISE_DIST="gaussian"`. At `dt=1`, `q=1`, the physical covariance block is `[[.05,.125,1/6],[.125,1/3,.5],[1/6,.5,1]]`, before scaling.

Initial x/y position are independent uniform draws on `[20,000,30,000]`; velocity on `[-200,200]` m/s and acceleration on `[-3,3]` m/s² for fixed reference. Initial components are drawn independently by the generator. For legacy `sequence_scaled`, velocity bounds scale by `100/T` and acceleration bounds and jerk RMS by `(100/T)^2` when `T>100`. Exact generator and manifest settings govern each recorded dataset.

## Radar measurement

The nonlinear observation is `h(x)=[r,θ]^T`, `r=sqrt(p_x²+p_y²+1e-12)`, `θ=atan2(p_y,p_x)` wrapped into `[0,2π)`. Stage 2 stores `r/1000` and θ radians. Measurement noise is Gaussian. Manifest physical covariance is `diag(30² m²,(π/180)² rad²)` (T=200 example), with stored range scaling applied; no range-dependent noise increase is applied. Noise covariance may be multiplied by `r2_scale`.

At first two generated observations, `MEAS` initialization converts the first range/bearing to Cartesian position and estimates vx/vy from the difference of the two Cartesian positions divided by dt; velocity components are clipped to ±300 m/s and acceleration initialized to zero. The second observation is the first returned observation (returned truth/observation sequence drops the extra initialization step). Initialization therefore shares an observation with the sequence model. This alignment is explicitly checked in Stage 2 manifests.

## Estimation target and leakage boundaries

M0/M1 receive the full causal observation prefix as an `[N,2,T]` tensor and the measurement-derived initial state. They output an absolute six-state estimate at each aligned index, implemented as the coarse state plus a learned per-time correction. Training labels are simulator states `[N,6,T]`; validation labels select checkpoints; test ground truth is scoring-only in Stages 4, 6, and 8. The networks do not take ground truth at inference. They do use the fixed initializer (including its initial velocity/acceleration values) in feature construction. Stage 5B records test split not opened; Stage 6 records frozen checkpoint evaluation only.

The exact reported metric scaling is summarized in Stage 9: joint x/y RMSE for each state group in physical units; six-state MSE in stored scaled-state units. See Stage 9 metric consistency audit for formulas and index alignment.
