from __future__ import annotations

import torch.nn as nn
from .mps_layer import MPSLayer


class CNNWithMPS(nn.Module):
    def __init__(self, input_len: int = 5000, num_classes: int = 256, bond_dim: int = 8):
        super().__init__()
        self.input_len = input_len
        self.num_classes = num_classes
        self.bond_dim = bond_dim
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.AvgPool1d(10),
            nn.Conv1d(16, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(10),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mps = MPSLayer(32, num_classes, bond_dim)

    def forward(self, x):
        x = self.pool(self.features(x)).squeeze(-1)
        return self.mps(x)
