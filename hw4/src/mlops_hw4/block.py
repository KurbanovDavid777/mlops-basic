"""Residual block: the basic building unit of the model."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Computes ``y = x + Linear(ReLU(Linear(x)))``; input and output shapes match."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))
