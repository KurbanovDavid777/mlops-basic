# HW1: uv + Ruff + ty

Minimal Python project with [uv](https://docs.astral.sh/uv/) (package/project manager),
[Ruff](https://docs.astral.sh/ruff/) (linter and formatter) and [ty](https://docs.astral.sh/ty/) (type checker).

The project contains a single function, `mean` (`src/mlops_hw1/__init__.py`), covered by pytest tests.

## Usage

```bash
uv sync                       # create .venv and install dependencies
uv run pytest                 # run tests
uv run ruff check .           # lint
uv run ruff format --check .  # check formatting
uv run ty check               # type check
```
