from __future__ import annotations

import math
import torch
import torch.nn as nn


class MPSLayer(nn.Module):
    """Compact, differentiable Matrix Product State classifier head."""

    def __init__(self, input_dim: int, num_classes: int, bond_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bond_dim = bond_dim
        self.num_sites = max(1, math.ceil(math.log2(input_dim)))
        self.site_encoder = nn.Linear(input_dim, 2 * self.num_sites)
        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn(bond_dim, 2, bond_dim) * 0.01)
            for _ in range(self.num_sites)
        ])
        self.output = nn.Linear(bond_dim, num_classes)

    def forward(self, x):
        encoded = self.site_encoder(x).view(x.shape[0], self.num_sites, 2)
        encoded = torch.softmax(encoded, dim=-1)
        state = torch.ones(x.shape[0], self.bond_dim, device=x.device, dtype=x.dtype)
        for site, core in enumerate(self.cores):
            state = torch.einsum("bi,ijk,bj->bk", state, core, encoded[:, site])
        return self.output(state)
