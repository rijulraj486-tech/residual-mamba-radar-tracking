import numpy as np
import torch


def print_average_rmse(path="real_world_pipeline/data/test.pt"):
    rmses = []
    for item in torch.load(path, map_location="cpu"):
        gt = item["ground_truth"][:, :2].cpu().numpy()
        gps = item["filtered_gps"][:, :2].cpu().numpy()
        mask = ~np.isnan(gps).any(axis=1)
        if mask.any():
            err = gps[mask] - gt[mask]
            rmses.append(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    print(f"Average RMSE: {np.mean(rmses):.4f}")


if __name__ == "__main__":
    print_average_rmse()
