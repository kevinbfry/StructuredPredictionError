# StructuredPredictionError

Structured Data Prediction Error Estimation

## Install

Project dependencies are in `pyproject.toml` file.

Can be installed via `poetry` (recommended to install within a virtual environment):

```
conda create --name spe_env
conda activate spe_env
conda install poetry
poetry install
```

Additionally, a modified version of scikit-learn must be installed:

```
pip install git+https://github.com/kevinbfry/scikit-learn.git
```

## Tests

```
pytest spe
```
