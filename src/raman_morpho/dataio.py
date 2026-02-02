"""Dataset loading and 3-class construction."""

from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

def load_cells_raman_dataset(
    base_dir: str = "cells-raman-spectra",
    excluded_labels: set[str] | None = None,
    expected_points: int = 2090,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load nested CSVs into a single dataframe.

    Returns
    -------
    df_all : DataFrame with columns: Label, Kind, and spectral columns as strings.
    wn     : wavenumber axis (float array) with `expected_points` points
    spec_cols : list of spectral column names (strings) in same order as wn
    """
    if excluded_labels is None:
        excluded_labels = {"serum", "DMEM"}

    pattern = os.path.join(base_dir, "dataset_i", "**", "*.csv")
    rows = []

    wn = np.linspace(100, 4278, expected_points)
    spec_cols = [str(int(round(v))) for v in wn]

    for file in glob.glob(pattern, recursive=True):
        path_parts = file.split(os.path.sep)
        label = path_parts[-2]  # folder name
        kind = os.path.splitext(path_parts[-1])[0]

        if label in excluded_labels:
            continue

        arr = pd.read_csv(file, header=None).values
        if arr.shape[1] != expected_points:
            raise ValueError(f"{label}/{kind}: {arr.shape[1]} points found, expected {expected_points}")

        arr_norm = normalize(arr, axis=1)
        for i in range(arr_norm.shape[0]):
            row = {"Label": label, "Kind": kind}
            row.update(dict(zip(spec_cols, arr_norm[i])))
            rows.append(row)

    df_all = pd.DataFrame(rows)
    return df_all, wn, spec_cols


def make_3class_problem(
    df_all: pd.DataFrame,
    *,
    label_col: str = "Label",
    kind_col: str = "Kind",
    melanoma_set = ("A", "G"),
    normal_skin_set = ("HPM", "HF"),
    disease_related_set = ("ZAM",),
    excluded_base = ("DMEM",),
    exclude_serum_suffix: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder, dict]:
    """Filter to 3-class labels and return (df3, spectra, wn_sorted, label_encoder, label_map)."""
    assert label_col in df_all.columns
    assert kind_col in df_all.columns

    labels_raw = df_all[label_col].astype(str).to_numpy()

    if exclude_serum_suffix:
        mask_no_serum = np.array([not lab.endswith("-S") for lab in labels_raw], dtype=bool)
        df0 = df_all.loc[mask_no_serum].reset_index(drop=True)
    else:
        df0 = df_all.copy()

    labels_raw0 = df0[label_col].astype(str).to_numpy()
    base_label = np.array([lab[:-2] if lab.endswith("-S") else lab for lab in labels_raw0], dtype=object)

    y3_str = np.full(shape=base_label.shape, fill_value="exclude", dtype=object)
    y3_str[np.isin(base_label, list(melanoma_set))] = "melanoma"
    y3_str[np.isin(base_label, list(normal_skin_set))] = "normal_skin"
    y3_str[np.isin(base_label, list(disease_related_set))] = "disease_related"

    mask3 = (y3_str != "exclude") & (~np.isin(base_label, list(excluded_base)))
    df3 = df0.loc[mask3].reset_index(drop=True)
    y3_str = y3_str[mask3]

    non_spec_cols = {label_col, kind_col}
    spec_cols = [c for c in df3.columns if c not in non_spec_cols and np.issubdtype(df3[c].dtype, np.number)]
    spectra = df3[spec_cols].to_numpy(dtype=np.float64)

    wn = np.array([float(c) for c in spec_cols], dtype=float)
    sort_idx = np.argsort(wn)
    wn = wn[sort_idx]
    spectra = spectra[:, sort_idx]

    le3 = LabelEncoder()
    y3 = le3.fit_transform(y3_str)
    label_map = dict(zip(le3.classes_, range(len(le3.classes_))))

    return df3, spectra, wn, y3, le3, label_map
