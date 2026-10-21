# HW3: data versioning and pipelines with DVC

A minimal [DVC](https://dvc.org/) project: versioned data, a local remote and a two-stage pipeline.

## Pipeline

```
data/raw.csv  ->  prepare  ->  data/processed.csv  ->  stats  ->  metrics.json
```

| Stage | Command | Description |
| ----- | ------- | ----------- |
| `prepare` | `python src/prepare.py data/raw.csv data/processed.csv` | min-max normalizes the temperature column |
| `stats` | `python src/stats.py data/processed.csv metrics.json` | computes row count, mean, min and max temperature |

## Layout

- `data/raw.csv` - input data, tracked by DVC (git stores only `data/raw.csv.dvc`)
- `dvc.yaml` / `dvc.lock` - pipeline definition and the exact versions of its inputs and outputs
- `.dvc/config` - DVC configuration with the default remote
- `src/` - pipeline stage scripts

## Reproduce

```bash
uv sync                  # install dependencies, including DVC
uv run dvc pull          # fetch data from the remote
uv run dvc repro         # run the pipeline (only stages with changed inputs are re-run)
uv run dvc metrics show  # print metrics.json
uv run dvc dag           # show the stage graph
```

The remote is a local directory (`../../mlops-dvc-storage` relative to this folder),
so `dvc pull` works on the machine where the remote lives.
