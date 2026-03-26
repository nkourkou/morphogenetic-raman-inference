from __future__ import annotations

from typing import Optional

import numpy as np


def apply_random_band_damage(
    band_logits0: np.ndarray | list[np.ndarray],
    damage_fraction: float,
    *,
    mode: str = "mixed",
    rng: Optional[np.random.Generator] = None,
    noise_scale: float | None = None,
    use_floor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)

    if isinstance(band_logits0, list):
        l0 = np.stack([np.asarray(v, dtype=float) for v in band_logits0], axis=0)
    else:
        l0 = np.asarray(band_logits0, dtype=float)

    if l0.ndim != 2:
        raise ValueError(f"Expected (K,C), got {l0.shape}")

    k_bands, c_classes = l0.shape
    frac = float(np.clip(damage_fraction, 0.0, 1.0))
    n_damaged = int(np.floor(frac * k_bands + 1e-9)) if use_floor else int(np.round(frac * k_bands))

    idx_all = np.arange(k_bands)
    rng.shuffle(idx_all)
    damaged_idx = idx_all[:n_damaged]

    damaged = l0.copy()
    frozen_flags = np.zeros(k_bands, dtype=bool)

    if noise_scale is None:
        s = float(np.std(l0))
        noise_scale = s if s > 1e-8 else 1.0

    if n_damaged == 0:
        return damaged, frozen_flags, damaged_idx

    if mode == "mixed":
        damaged[damaged_idx, :] = rng.normal(0.0, noise_scale, size=(n_damaged, c_classes))
    elif mode == "spiky":
        damaged[damaged_idx, :] = rng.normal(0.0, noise_scale, size=(n_damaged, c_classes))
        n_spikes = max(1, int(np.ceil(0.3 * n_damaged)))
        spike_idx = damaged_idx[:n_spikes]
        damaged[spike_idx, :] = rng.normal(0.0, 3.0 * noise_scale, size=(n_spikes, c_classes))
    elif mode == "frozen":
        frozen_flags[damaged_idx] = True
        damaged[damaged_idx, :] = rng.normal(0.0, noise_scale, size=(n_damaged, c_classes))
    else:
        raise ValueError("mode must be 'mixed', 'spiky', or 'frozen'")

    return damaged, frozen_flags, damaged_idx


apply_band_damage = apply_random_band_damage


def damage_selected_bands(
    l0: np.ndarray,
    bands_to_damage: list[int],
    *,
    mode: str,
    rng: np.random.Generator,
    noise_scale: float,
    spike_scale_mult: float = 6.0,
) -> np.ndarray:
    l = np.asarray(l0, dtype=float).copy()
    _, c_classes = l.shape

    if mode == "spiky":
        for k in bands_to_damage:
            l[k] = rng.normal(0.0, spike_scale_mult * noise_scale, size=c_classes)
    elif mode == "mixed":
        for k in bands_to_damage:
            if rng.random() < 0.35:
                l[k] = rng.normal(0.0, spike_scale_mult * noise_scale, size=c_classes)
            else:
                l[k] = rng.normal(0.0, noise_scale, size=c_classes)
    else:
        raise ValueError("mode must be 'mixed' or 'spiky'")
    return l
