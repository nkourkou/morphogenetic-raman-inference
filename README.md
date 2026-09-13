# Process-aware inference of biomedical Raman spectra classification

Python implementation of band-resolved negotiated inference for Raman spectra. Band-specific classifiers provide class evidence to an iterative consensus process under three regimes: Plain, H1, and H2. The analyses examine predictive performance, Decision Geometry (DG), and the association between early relative DG (eREDG) and prediction correctness.

## Installation

Requires Python 3.10 or later. From the directory containing `pyproject.toml`, install the package and its dependencies:

```bash
python -m pip install -e .
```

Dependencies are NumPy, pandas, SciPy, scikit-learn, and Matplotlib. An isolated Python environment is recommended. Dependency versions are not pinned; retain the environment used for any reported results.

## Data

Datasets are not included or downloaded automatically. Supply the files locally using `--data-dir`.

### Melanoma

Point to the `cells-raman-spectra` directory containing `dataset_i`. The loader searches recursively for CSV files below `dataset_i`; each file's immediate parent directory supplies its label. CSVs must contain numeric spectra without a header, one spectrum per row and 2,090 values per spectrum.

The loader constructs an axis from 100 to 4,278 cm⁻¹ and rounds its column labels to integer wavenumbers. Labels `A` and `G` form the melanoma group, `HPM` and `HF` the normal-skin group, and `ZAM` the disease-related group. Serum, `DMEM`, and labels ending in `-S` are excluded. Seven inference bands cover 400–2,000 cm⁻¹.

### Diabetes

The data directory must contain `innerArm.csv`, `thumbNail.csv`, `vein.csv`, and `earLobe.csv`. Each file requires:

- `patientID` and `has_DM2` columns, with disease labels coded as 0 or 1;
- spectral columns named `Var1`, `Var2`, and so on;
- exactly one row with `patientID` equal to `ramanShift`, containing the Raman axis.

Spectra are restricted to 400–1,800 cm⁻¹. Patient identifiers are used to keep patients separate between training and evaluation sets.

### Bacteria

Provide one copy of each of these files within the data directory or its subdirectories:

- `X_2018clinical.npy` and `y_2018clinical.npy`;
- `X_2019clinical.npy` and `y_2019clinical.npy`;
- `wavenumbers.npy`.

Spectral arrays have shape `(n_samples, n_wavenumbers)`; label and axis arrays are one-dimensional. Supply the intended five-class clinical subsets: the loader does not select those classes from a larger dataset. Clinical 2018 is split into training and validation sets; clinical 2019 is the external evaluation set. Seven equal-width inference bands span 400 cm⁻¹ to the smaller of 1,800 cm⁻¹ and the supplied axis maximum.

All three loaders apply sample-wise L2 normalization, followed by training-fitted feature standardization within classifier pipelines. Melanoma and bacteria normalization uses the supplied full spectra; diabetes normalization follows spectral restriction. Full-spectrum baselines use all columns retained by their loader, not necessarily only the inference bands.

## Running analyses

```bash
python -m morphoraman melanoma --data-dir "data/cells-raman-spectra" --analysis robustness --output-dir "results/melanoma"
python -m morphoraman diabetes --data-dir "data/diabetes" --analysis audit --output-dir "results/diabetes"
python -m morphoraman bacteria --data-dir "data/bacteria" --analysis audit --output-dir "results/bacteria"
```

Each command loads the dataset, fits its initial models, and runs one analysis. Use a separate process for each analysis because the dataset modules retain working state.

| Analysis | Datasets | Purpose |
| --- | --- | --- |
| `robustness` | All three | Mixed/spiky perturbation sweeps, performance and DG |
| `audit` | All three | Fixed-direction eREDG AUC and outcome distributions |
| `baselines` | Melanoma, bacteria | Classifier comparisons |
| `partition` | Melanoma, diabetes | Sensitivity to spectral partitioning |
| `conflict` | Melanoma | H1 band-resolved conflict localization |
| `summary` | Melanoma | Repeated-split quantitative summaries |
| `spectra_plot` | All three | Descriptive spectral plots |

Figure-saving routines write to `--output-dir`, which defaults to `figures`. Add `--show` for interactive plots, including plots without a save operation. Existing figures with matching filenames may be overwritten.

```bash
python -m morphoraman --help
python -m unittest discover -s tests
```

