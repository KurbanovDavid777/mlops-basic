"""MLP classifier built from residual blocks."""

import torch
from torch import nn

from mlops_hw4.block import ResidualBlock


class MLPClassifier(nn.Module):
    """Input projection -> N residual blocks -> linear head returning logits."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input_proj(x)))
