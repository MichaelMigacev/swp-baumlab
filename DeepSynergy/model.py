import torch
import torch.nn as nn

class DeepSynergyModel(nn.Module):
    def __init__(self, input_size):
        super(DeepSynergyModel, self).__init__()

        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, 8192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1)  # Nur 1 Output: Synergiewert
        )

    def forward(self, x):
        return self.model(x)
