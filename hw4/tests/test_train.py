import math

import pytest
import torch
from torch import nn

from mlops_hw4.data import make_dataloader, make_synthetic_dataset
from mlops_hw4.model import MLPClassifier
from mlops_hw4.train import evaluate, train_step


@pytest.mark.integration
def test_model_consumes_batches_from_dataloader() -> None:
    loader = make_dataloader(make_synthetic_dataset(n_features=8, n_classes=3))
    model = MLPClassifier(in_features=8, hidden_dim=16, num_classes=3)
    features, labels = next(iter(loader))
    logits = model(features)
    assert logits.shape == (features.shape[0], 3)
    assert labels.shape[0] == features.shape[0]


@pytest.mark.integration
def test_train_step_updates_weights_and_returns_finite_loss() -> None:
    torch.manual_seed(0)
    model = MLPClassifier(in_features=8, hidden_dim=16, num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    features, labels = next(iter(make_dataloader(make_synthetic_dataset())))
    before = [p.detach().clone() for p in model.parameters()]

    loss = train_step(model, features, labels, optimizer, nn.CrossEntropyLoss())

    assert math.isfinite(loss)
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


@pytest.mark.integration
def test_evaluate_returns_accuracy_in_unit_interval() -> None:
    torch.manual_seed(0)
    model = MLPClassifier(in_features=8, hidden_dim=16, num_classes=3)
    loader = make_dataloader(make_synthetic_dataset(), shuffle=False)
    assert 0.0 <= evaluate(model, loader) <= 1.0
