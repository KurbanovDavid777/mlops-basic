import pytest
import torch

from mlops_hw4.model import MLPClassifier


@pytest.mark.unit
def test_model_output_shape() -> None:
    model = MLPClassifier(in_features=8, hidden_dim=16, num_classes=3)
    logits = model(torch.randn(5, 8))
    assert logits.shape == (5, 3)


@pytest.mark.unit
def test_model_rejects_zero_blocks() -> None:
    with pytest.raises(ValueError):
        MLPClassifier(in_features=8, hidden_dim=16, num_classes=3, num_blocks=0)


@pytest.mark.unit
def test_model_has_requested_number_of_blocks() -> None:
    model = MLPClassifier(in_features=8, hidden_dim=16, num_classes=3, num_blocks=4)
    assert len(model.blocks) == 4
