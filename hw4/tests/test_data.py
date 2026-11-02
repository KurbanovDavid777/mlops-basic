import pytest
import torch

from mlops_hw4.data import make_dataloader, make_synthetic_dataset


@pytest.mark.unit
def test_dataset_shapes_and_dtypes() -> None:
    dataset = make_synthetic_dataset(n_samples=30, n_features=4, n_classes=3)
    features, labels = dataset.tensors
    assert features.shape == (30, 4)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert set(labels.tolist()) == {0, 1, 2}


@pytest.mark.unit
def test_dataset_is_deterministic_for_same_seed() -> None:
    first = make_synthetic_dataset(seed=7).tensors[0]
    second = make_synthetic_dataset(seed=7).tensors[0]
    assert torch.equal(first, second)


@pytest.mark.unit
def test_dataloader_batches() -> None:
    dataset = make_synthetic_dataset(n_samples=100, n_features=8)
    loader = make_dataloader(dataset, batch_size=32)
    batches = list(loader)
    assert len(batches) == 4  # 32 + 32 + 32 + 4
    assert batches[0][0].shape == (32, 8)
    assert batches[-1][0].shape[0] == 4
