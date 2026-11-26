import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaResidualBlock(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):

        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def _sequence_mixer(self, x):

        if x.is_cuda:
            return self.mamba(x)

        raise RuntimeError(
            "BiMambaEncoder requires CUDA because the Mamba block in this "
            "project does not support CPU execution."
        )

    def forward(self, x):

        x = x + self.dropout(self._sequence_mixer(self.norm(x)))
        x = x + self.dropout(self.ff(self.ff_norm(x)))

        return x


class BiMambaEncoder(nn.Module):
    """Forward-only Mamba encoder kept under the old name for compatibility."""

    def __init__(
        self,
        dim,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        encoder_direction="forward",
    ):

        super().__init__()

        if encoder_direction not in {"forward", "bidirectional"}:
            raise ValueError(
                "encoder_direction must be either 'forward' or 'bidirectional'."
            )

        self.encoder_direction = "forward"
        self.forward_layers = nn.ModuleList([
            MambaResidualBlock(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.output_dim = dim
        self.fusion_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):

        forward_state = x
        for layer in self.forward_layers:
            forward_state = layer(forward_state)

        return self.fusion_norm(forward_state)
