# Reproducibility checklist

- [ ] Verify source/config/data/checkpoint hashes against manifests before loading.
- [ ] Preserve train/validation/test trajectory-ID isolation; do not inspect test until checkpoint choice and calibration are frozen.
- [ ] Use fixed Stage 2 splits and record device, software versions, all seed streams, loader-order hashes, and training times.
- [ ] Fit M1 feature statistics on training tokens only and persist them with training split hash.
- [ ] Select checkpoints by validation six-state MSE only.
- [ ] Score all models at aligned full-horizon indices with the shared metric implementation.
- [ ] Label Stage 4 as T=100 extrapolation, Stage 5A as screening, Stage 5B as confirmation, Stage 6 as held-out test.
- [ ] Keep deterministic classical filter scores as single results; separate oracle-Q/R and train-estimated-Q/R.
- [ ] Label KalmanNet as T=100 trained cross-length transfer and disclose missing internal clip counters.
- [ ] Record provenance references exactly; disclose Stage 8 Stage 6 hash pointer discrepancy.
- [ ] Use a separate output directory for any reproduction. No command in this guide is permission to overwrite historical records.
