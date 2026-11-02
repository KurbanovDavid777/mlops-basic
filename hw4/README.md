# HW4: tests for a PyTorch pipeline

A tiny PyTorch classification pipeline and a test suite that covers it from unit tests to end-to-end.

## Pipeline

| Module | Content |
| ------ | ------- |
| `block.py` | `ResidualBlock`: `x + Linear(ReLU(Linear(x)))` |
| `model.py` | `MLPClassifier`: input projection, N residual blocks, linear head |
| `data.py` | synthetic Gaussian-blob dataset and a seeded `DataLoader` |
| `train.py` | `train_step`, `evaluate` and the `train` loop (Adam, cross-entropy) |

## Tests

Tests are grouped with pytest marks (registered in `pyproject.toml`):

| Mark | What it checks | Files |
| ---- | -------------- | ----- |
| `unit` | one component in isolation: shapes, dtypes, determinism, input validation | `test_block.py`, `test_model.py`, `test_data.py` |
| `integration` | components working together: model reads dataloader batches, a training step updates weights, evaluation accuracy is valid | `test_train.py` |
| `e2e` | the whole pipeline: training lowers the loss and reaches high accuracy; a saved model reproduces predictions | `test_e2e.py` |

## Usage

```bash
uv sync
uv run pytest                 # all tests
uv run pytest -m unit         # only one mark
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
