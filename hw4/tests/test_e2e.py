from pathlib import Path

import pytest
import torch

from mlops_hw4.data import make_dataloader, make_synthetic_dataset
from mlops_hw4.model import MLPClassifier
from mlops_hw4.train import evaluate, train


@pytest.mark.e2e
def test_training_reduces_loss_and_learns_the_task() -> None:
    torch.manual_seed(0)
    dataset = make_synthetic_dataset(n_samples=256, n_features=8, n_classes=3)
    loader = make_dataloader(dataset, batch_size=32)
    model = MLPClassifier(in_features=8, hidden_dim=32, num_classes=3)

    history = train(model, loader, epochs=10, lr=1e-2)

    assert history[-1] < history[0]
    assert evaluate(model, make_dataloader(dataset, shuffle=False)) > 0.9


@pytest.mark.e2e
def test_saved_model_reproduces_predictions(tmp_path: Path) -> None:
    torch.manual_seed(0)
    dataset = make_synthetic_dataset()
    model = MLPClassifier(in_features=8, hidden_dim=16, num_classes=3)
    train(model, make_dataloader(dataset), epochs=2)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    restored = MLPClassifier(in_features=8, hidden_dim=16, num_classes=3)
    restored.load_state_dict(torch.load(path))
    x = dataset.tensors[0][:10]

    model.eval()
    restored.eval()
    assert torch.equal(model(x), restored(x))
