"""Spectral band definitions and index building."""
from __future__ import annotations
import numpy as np

DEFAULT_BANDS_FP = [
    (400, 700),
    (700, 900),
    (900, 1100),
    (1100, 1300),
    (1300, 1500),
    (1500, 1700),
    (1700, 2000),
]

def build_band_indices(wn: np.ndarray, bands=DEFAULT_BANDS_FP) -> list[np.ndarray]:
    """Return list of index arrays, half-open bins, last inclusive."""
    wn = np.asarray(wn, dtype=float)
    band_indices = []
    for k, (lo, hi) in enumerate(bands):
        idx = np.where((wn >= lo) & (wn < hi))[0] if k < len(bands) - 1 else np.where((wn >= lo) & (wn <= hi))[0]
        if idx.size == 0:
            raise ValueError(f"No points in band {lo}-{hi} cm^-1.")
        band_indices.append(idx)

    # overlap check
    all_idx = np.concatenate(band_indices)
    if np.unique(all_idx).size != all_idx.size:
        raise ValueError("Band definitions overlap.")

    return band_indices
