import pytest
import torch

from mlops_hw4.block import ResidualBlock


@pytest.mark.unit
def test_block_keeps_shape() -> None:
    block = ResidualBlock(dim=16)
    out = block(torch.randn(4, 16))
    assert out.shape == (4, 16)


@pytest.mark.unit
def test_block_is_identity_when_branch_is_zero() -> None:
    block = ResidualBlock(dim=8)
    for layer in (block.fc1, block.fc2):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    x = torch.randn(3, 8)
    assert torch.equal(block(x), x)
