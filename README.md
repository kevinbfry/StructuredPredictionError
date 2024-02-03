# StructuredPredictionError

Structured Data Prediction Error Estimation

## Install

Project dependencies are in `pyproject.toml` file.
We recommend installing `Mambaforge`, which is a conda installation with `mamba` installed by default and set to use `conda-forge` as the default channel.
Can be installed via `poetry` (recommended to install within a virtual environment):

```
mamba update -y conda mamba
mamba env create
conda activate spe
poetry config virtualenvs.create false --local
poetry install --no-root
```

## Tests

TODO: please fix...

```
pytest spe
```
