import json
from pathlib import Path

import torch

from radar_tracking.cli.benchmark_multitarget import (
    DEFAULT_ASSOC_OUTPUT_DIR,
    MultiTargetConfig,
    apply_preset,
    build_association_configs,
    load_best_mamba,
    train_or_load_association_model,
)
from radar_tracking.dynamics.constant_acceleration import ConstantAcceleration2DModel
from radar_tracking.config import simparams_class, validate_simparams


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simparams = simparams_class()
    validate_simparams(simparams)
    ca_model = ConstantAcceleration2DModel(simparams)

    config = apply_preset(MultiTargetConfig(preset="research"))
    mamba_model, _ = load_best_mamba(device)
    association_model, association_meta = train_or_load_association_model(
        multitarget_config=config,
        ca_model=ca_model,
        simparams=simparams,
        mamba_model=mamba_model,
        device=device,
    )
    model_config, train_config = build_association_configs(config)

    output_dir = Path(association_meta["checkpoint_path"]).parent if association_meta else DEFAULT_ASSOC_OUTPUT_DIR
    summary = {
        "device": str(device),
        "association_model": association_meta,
        "model_config": model_config.__dict__,
        "train_config": train_config.__dict__,
        "parameter_count": int(sum(param.numel() for param in association_model.parameters())),
    }
    with (output_dir / "association_report.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Association training complete")
    print(f"Checkpoint: {association_meta['checkpoint_path']}")
    print(f"Saved summary to {output_dir / 'association_report.json'}")


if __name__ == "__main__":
    main()
