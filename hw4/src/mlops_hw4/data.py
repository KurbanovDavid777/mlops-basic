"""Synthetic dataset and dataloader: Gaussian blobs, one blob per class."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic_dataset(
    n_samples: int = 256,
    n_features: int = 8,
    n_classes: int = 3,
    seed: int = 0,
) -> TensorDataset:
    """Return ``n_samples`` points around ``n_classes`` random centers."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, n_features, generator=generator) * 3.0
    labels = torch.arange(n_samples) % n_classes
    noise = torch.randn(n_samples, n_features, generator=generator) * 0.5
    return TensorDataset(centers[labels] + noise, labels)


def make_dataloader(
    dataset: TensorDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    """Wrap a dataset into a DataLoader with a seeded shuffling order."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )
