# Radar Tracking

## Project overview

This repository contains an independent radar-tracking research and implementation migration. The project studies neural and classical tracking methods with synthetic and real-world observations, while keeping model architecture and training behavior separate from package setup.

## Research and implementation scope

The implementation scope includes:

- Single-target radar tracking with range and bearing observations.
- Multi-target radar tracking with data generation, track association, and GOSPA-style evaluation.
- Optional NCLT real-world evaluation using vehicle trajectory and GPS-related datasets.
- EKF, KalmanNet, and Mamba-based sequence-model experiment implementations.

## Project directory structure

```text
src/radar_tracking/
  cli/          Experiment and evaluation entry points
  data/         Synthetic and NCLT data loading/preprocessing utilities
  dynamics/     Tracking dynamics implementations
  metrics/      GPS RMSE and GOSPA-style metric implementations
  models/       EKF, KalmanNet, and Mamba-based models
  multitarget/  Multi-target generation, tracking, and association code
  training/     Training and validation utilities
archive/         Archived source scripts and migration-era experiments
artifacts/       Checkpoints, figures, logs, and result outputs
docs/reports/    Project reports and documentation
migration_records/
  phase1_manifest.csv
  phase1_manifest.json
notebooks/       Research notebooks
pyproject.toml
requirements.txt
README.md
```

A `configs/` directory and `tests/` directory are not present in the current project root.

## Single-target radar tracking

Single-target support includes synthetic range/bearing data generation, constant-acceleration dynamics, EKF baselines, KalmanNet variants, and a BiMamba sequence-model implementation. Evaluation utilities provide loss and RMSE-oriented summaries.

## Multi-target radar tracking

Multi-target support includes synthetic multi-object scenarios, track state estimation, association-model training, assignment, track management, and multi-target performance metrics. Existing project files are preserved; import and entry-point validation remain part of later migration work.

## Optional NCLT real-world evaluation

The project contains optional NCLT download, preprocessing, loading, and evaluation utilities. NCLT data is not included with this package, and separate dataset acquisition and storage setup are required before real-world runs.

## Main technology stack

- Python 3.12
- PyTorch
- NumPy
- Matplotlib
- Mamba-SSM
- causal-conv1d
- tqdm

## Python environment and dependency information

The inspected baseline environment was:

- Python 3.12.3
- PyTorch 2.4.0+cu121
- Mamba-SSM 2.2.0
- causal-conv1d 1.4.0
- NumPy 2.4.4
- Matplotlib 3.10.8

`tqdm` is also used by source code and is declared as a dependency. No additional package versions were invented for it.

### CUDA versus CPU builds

The existing environment uses a CUDA-specific PyTorch build, `torch 2.4.0+cu121`. It must be installed from the official PyTorch CUDA 12.1 index before the remaining dependencies. PyTorch remains declared as `torch==2.4.0` in `pyproject.toml`; the wheel variant is selected explicitly during environment setup.

Mamba-SSM and causal-conv1d may require a compatible NVIDIA CUDA toolkit and compiler toolchain. NumPy remains pinned to 2.4.4. NumPy/PyTorch interoperability has not been validated and requires a runtime check before relying on these versions together.

## Initial setup guidance

From the repository root, create or activate a Python 3.12 environment and run the following sequence:

```bash
# 1. Install the CUDA 12.1 PyTorch build first.
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Install the remaining dependencies. requirements.txt does not contain PyTorch.
python -m pip install -r requirements.txt

# 3. Install the project itself without dependency resolution replacing PyTorch.
python -m pip install -e . --no-deps
```

The `--no-deps` editable installation is intentional: it prevents pip from resolving or replacing the already selected CUDA PyTorch wheel while registering the `radar_tracking` source package. Do not run a generic or CPU-only PyTorch installation afterward. This setup documents configuration only; installation and runtime behavior have not been tested by this migration step.

## Status

Training and evaluation commands will be documented after package configuration and import validation are complete.
