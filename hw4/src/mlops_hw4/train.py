"""Training utilities: one optimization step, evaluation and the training loop."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_step(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Run one optimization step on a batch and return the loss value."""
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(features), labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Return classification accuracy of the model over the loader."""
    model.eval()
    correct = 0
    total = 0
    for features, labels in loader:
        predictions = model(features).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-2,
) -> list[float]:
    """Train with Adam and cross-entropy; return the mean loss of every epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []
    for _ in range(epochs):
        losses = [
            train_step(model, features, labels, optimizer, loss_fn)
            for features, labels in loader
        ]
        history.append(sum(losses) / len(losses))
    return history
