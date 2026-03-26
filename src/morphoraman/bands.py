from __future__ import annotations

import numpy as np


def build_band_indices(
    wn: np.ndarray,
    bands: list[tuple[float, float]],
) -> list[np.ndarray]:
    band_indices: list[np.ndarray] = []
    for k, (lo, hi) in enumerate(bands):
        if k < len(bands) - 1:
            idx = np.where((wn >= lo) & (wn < hi))[0]
        else:
            idx = np.where((wn >= lo) & (wn <= hi))[0]
        if idx.size == 0:
            raise ValueError(f"No points in band {lo}-{hi} cm^-1.")
        band_indices.append(idx)

    all_idx = np.concatenate(band_indices)
    if np.unique(all_idx).size != all_idx.size:
        raise ValueError("Band definitions overlap.")

    return band_indices


def make_equalwidth_bands(
    wn: np.ndarray,
    k: int,
    *,
    lo: float = 400.0,
    hi: float = 2000.0,
) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    edges = np.linspace(lo, hi, k + 1)
    bands = [(float(edges[i]), float(edges[i + 1])) for i in range(k)]
    return bands, build_band_indices(wn, bands)
