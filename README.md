# Process-aware self-auditing inference of biomedical Raman signal classification

Python code for submitted manuscript.

## Quickstart

## Installation

Create and activate a virtual environment, then install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

On Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

## Dataset
The working dataset is publicly available at: https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra/data


## Layout

- `src/raman_morpho/` importable library code
- `scripts/` runnable entry points
- `notebooks/` notebooks (multiple versions)
