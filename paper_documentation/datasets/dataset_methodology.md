# Dataset methodology

## Source and splits

The original T=100 fixed-pool experiment is recorded in `artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/fixed_pool_manifest.json`; it uses a shared 1,024-sample test pool, generation seed 20261126, shared test seed 9001, 80/20 train/validation split (819/204 from the 1,024 generated pool), plus shared 1,024 test trajectories. The manifest's recorded split construction uses Python `random.shuffle`; consult each run record for training/validation data use. The manifest records disjoint train and validation index sets from the generated pool. Test data are separately generated and hash-recorded, but a cross-split trajectory-ID comparison is not present in retained original manifests: NOT VERIFIED FROM AVAILABLE PROJECT ARTIFACTS.

Stage 2 T=200 and T=400 are independent fixed-reference bundles with 819 train, 204 validation, 1,024 test trajectories each. Per-split seeds: T=200 train/validation/test = 202611271/272/273; T=400 = 202611274/275/276. Manifest validation confirms disjoint trajectory IDs across all six split manifests and between lengths. Tensors: observations `[N,2,T]`, states `[N,6,T]`, initializer `[N,6,1]`, initialization observations `[N,2,2]`. Artifacts report finite tensors and initialization alignment. IDs and SHA-256 are indexed in `dataset_manifest_index.csv` and `dataset_hashes.csv`.

## Fixed-reference profile

`get_motion_profile(..., mode="fixed_reference")` resolves scales at reference horizon 100 regardless of requested T: initial position bounds 20–30 km, velocity ±200 m/s, acceleration ±3 m/s², jerk RMS 1 m/s² (process intensity computed as RMS²/dt). Legacy `sequence_scaled` reduces velocity by 100/T and acceleration/jerk RMS by `(100/T)^2` for T>100. Fixed-reference therefore keeps initial state bounds and per-step process law fixed while allowing repeated integration for more steps; accumulated displacement, speed/position/range distributions, and near-origin / extreme-range counts change with horizon. It is not a stationary distribution and these synthetic trajectories are not evidence of physical realism.

## Noise and range caveat

Stage 2 uses Gaussian measurement errors with fixed 30 m range standard deviation and 1° bearing standard deviation in physical units (manifest covariance); bearing noise does not shrink with range and range error does not grow with range. Thus transverse position error implied by angular noise grows approximately with range. Long-horizon test sets include very large ranges and speeds, and fixed noise assumptions over that range do not model a radar's range-dependent SNR, detection, clutter, or resolution. Do not call this realistic operational radar simulation.

## Distribution evidence

See `dataset_statistics.csv` for actual pooled quantiles and counts. T=400 test medians: position radius 137,718 m; speed 1,659 m/s; acceleration magnitude 14.70 m/s²; radar range 137,730 m; 99th-percentile range 1,443,189 m and max 2,231,837 m. T=200 test medians are 44,294 m, 595.5 m/s, 10.37 m/s², and 44,295 m range; max range 532,729 m. These are very high simulated speeds/ranges. The discrepancy between initial bounds and pooled values is accumulated motion over 200/400 steps, not altered initial bounds.

T=100 distribution comparison report with equivalent quantiles was not found: `NOT VERIFIED FROM AVAILABLE PROJECT ARTIFACTS`. Raw dataset generation files for the original pool are not included in the inventory; exact file hashes and IDs are in its fixed-pool manifest. Stage 2 T=200/T=400 distribution computation is in `distribution_comparison.json` and its Markdown summary.
