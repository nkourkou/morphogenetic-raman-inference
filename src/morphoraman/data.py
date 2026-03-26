from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize


@dataclass(slots=True)
class DatasetBundle:
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    wn: np.ndarray
    label_encoder: LabelEncoder
    label_map: dict[str, int]
    base_labels: np.ndarray
    spectral_columns: list[str]


def load_cells_raman_dataset(
    base_dir: str | os.PathLike,
    *,
    excluded_labels: Iterable[str] = ("serum", "DMEM"),
    expected_points: int = 2090,
    normalize_rows: bool = True,
) -> pd.DataFrame:
    base_dir = str(base_dir)
    pattern = os.path.join(base_dir, "dataset_i", "**", "*.csv")
    rows: list[dict] = []

    wn = np.linspace(100, 4278, expected_points)
    spec_col_names = [str(int(round(v))) for v in wn]

    for file in glob.glob(pattern, recursive=True):
        path_parts = file.split(os.path.sep)
        label = path_parts[-2]
        kind = Path(file).stem

        if label in set(excluded_labels):
            continue

        arr = pd.read_csv(file, header=None).values
        if arr.shape[1] != expected_points:
            raise ValueError(
                f"{label}/{kind}: found {arr.shape[1]} points, expected {expected_points}"
            )

        arr_use = normalize(arr, axis=1) if normalize_rows else arr

        for i in range(arr_use.shape[0]):
            row = {"Label": label, "Kind": kind}
            row.update(dict(zip(spec_col_names, arr_use[i])))
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No CSV files found under {pattern!r}. Check the dataset directory."
        )

    return pd.DataFrame(rows)


def prepare_three_class_dataset(
    df_all: pd.DataFrame,
    *,
    label_col: str = "Label",
    kind_col: str = "Kind",
) -> DatasetBundle:
    if label_col not in df_all.columns:
        raise KeyError(f"Missing {label_col!r} column.")
    if kind_col not in df_all.columns:
        raise KeyError(f"Missing {kind_col!r} column.")

    labels_raw = df_all[label_col].astype(str).to_numpy()
    mask_no_serum = np.array([not lab.endswith("-S") for lab in labels_raw], dtype=bool)

    df0 = df_all.loc[mask_no_serum].reset_index(drop=True)
    labels_raw0 = df0[label_col].astype(str).to_numpy()
    base_label = np.array(
        [lab[:-2] if lab.endswith("-S") else lab for lab in labels_raw0],
        dtype=object,
    )

    melanoma_set = {"A", "G"}
    normal_skin_set = {"HPM", "HF"}
    disease_related_set = {"ZAM"}
    excluded_base = {"DMEM"}

    y3_str = np.full(shape=base_label.shape, fill_value="exclude", dtype=object)
    y3_str[np.isin(base_label, list(melanoma_set))] = "melanoma"
    y3_str[np.isin(base_label, list(normal_skin_set))] = "normal_skin"
    y3_str[np.isin(base_label, list(disease_related_set))] = "disease_related"

    mask3 = (y3_str != "exclude") & (~np.isin(base_label, list(excluded_base)))

    df3 = df0.loc[mask3].reset_index(drop=True)
    y3_str = y3_str[mask3]
    base_label3 = base_label[mask3]

    non_spec_cols = {label_col, kind_col}
    spec_cols = [
        c for c in df3.columns
        if c not in non_spec_cols and np.issubdtype(df3[c].dtype, np.number)
    ]

    X = df3[spec_cols].to_numpy(dtype=np.float64)
    wn = np.array([float(c) for c in spec_cols], dtype=float)
    sort_idx = np.argsort(wn)
    wn = wn[sort_idx]
    X = X[:, sort_idx]
    spec_cols_sorted = [spec_cols[i] for i in sort_idx]

    le = LabelEncoder()
    y = le.fit_transform(y3_str)
    label_map = dict(zip(le.classes_, range(len(le.classes_))))

    return DatasetBundle(
        df=df3,
        X=X,
        y=y,
        wn=wn,
        label_encoder=le,
        label_map=label_map,
        base_labels=base_label3,
        spectral_columns=spec_cols_sorted,
    )


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
