# Implementation pipeline and source map

1. Synthetic dynamics and radar observations: `src/radar_tracking/dynamics/constant_acceleration.py`, `src/radar_tracking/data/data_generation.py`, `src/radar_tracking/data/independent_dataset.py`.
2. Fixed-reference Stage 2 datasets: `experiments/paper_v4_long_sequence/generate_stage2.py`, immutable bundles under `artifacts/results/synthetic/paper_v4_long_sequence/T{200,400}_fixed_profile_v1/`.
3. M0/M1 neural model: `src/radar_tracking/models/tracker_model.py`, `bimamba_encoder.py`, `prediction_head.py`; M1 and training orchestration in `experiments/paper_v4_long_sequence/stage5a_matched_length_screen.py` and `stage5b_matched_length_confirmation.py`.
4. Frozen preflight/evaluation/statistics/baselines: `stage3_frozen_preflight.py`, `stage4_frozen_test_evaluation.py`, `stage6_matched_length_test_evaluation.py`, `stage7_statistical_analysis.py`, `stage8_long_sequence_evaluation.py`, `stage8_baseline_audit.py`.
5. Classical methods: `src/radar_tracking/models/ekf.py`, `ukf.py`, `ekf_baselines.py`; KalmanNet in `kalman_net.py`.
6. Historical T=100: `experiments/five_seed_harness.py`, `experiments/evaluate_five_seed.py`, fixed pool manifest and seed folders under `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/`.

See `source_code_map.csv` for key files and responsibilities. The project also contains multi-target and NCLT work; these are outside the single-target synthetic radar result scope. The NCLT wheel/GPS EKF is a different sensing task and is excluded.
