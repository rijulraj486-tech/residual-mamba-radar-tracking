import torch.nn as nn


class PredictionHead(nn.Module):

    def __init__(self, input_dim, state_dim=2, hidden_dim=128):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

    def forward(self, encoded_state, last_history_state):

        predicted_delta = self.net(encoded_state)

        return last_history_state + predicted_delta
