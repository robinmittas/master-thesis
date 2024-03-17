# Nucleus cytosol velocity reproducibility

Reproducibility repository for the nucleus-cytosol model.

## Installation

### Developer installation

```bash
conda create -n ncv-py39 python=3.9 --yes && conda activate ncv-py39
pip install -e ".[dev]"
pre-commit install
```

Jupyter lab and the corresponding kernel can be installed with

```bash
pip install jupyterlab ipywidgets
python -m ipykernel install --user --name ncv-py39 --display-name "ncv-py39"
```

## Project structure

This repository is dividied into five directories: `data/`, `figures/`, `notebooks/`, `src/`

-   **data/**: The `data/` direcory collects downloaded data not stored in the project directory.
-   **figures/**: The `figure/` directory is used for saving figures in a central location.
-   **notebooks/**: The `notebooks/` directory includes analysis notebooks.
-   **src/**: The `src/` directory can be used to implement code made availabe through the `dynamic_grn` package.
