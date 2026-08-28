from __future__ import annotations

import torch.nn as nn


class LightweightCNN(nn.Module):
    def __init__(self, input_len: int = 5000, num_classes: int = 256):
        super().__init__()
        self.input_len = input_len
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.AvgPool1d(10),
            nn.Conv1d(16, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(10),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).squeeze(-1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
