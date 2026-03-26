from __future__ import annotations

from dataclasses import dataclass, field
import os
import random
from typing import Sequence


DEFAULT_BANDS: list[tuple[float, float]] = [
    (400.0, 700.0),
    (700.0, 900.0),
    (900.0, 1100.0),
    (1100.0, 1300.0),
    (1300.0, 1500.0),
    (1500.0, 1700.0),
    (1700.0, 2000.0),
]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 1
    test_size: float = 0.20
    expected_points: int = 2090
    base_dir: str = "cells-raman-spectra"

    alpha: float = 0.3
    lam: float = 1.0
    temperature: float = 1e-2
    tau: float = 0.05
    max_iter: int = 30
    diff_amp: float = 0.25
    mute_weight: float = 0.0
    weight_clip: tuple[float, float] = (0.0, 5.0)

    damage_fractions: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)
    damage_modes: tuple[str, ...] = ("mixed", "spiky")
    regimes: tuple[str, ...] = ("plain", "H1", "H2")

    naive_weighting: str = "uniform"
    morpho_weighting: str = "confidence"
    bands: Sequence[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_BANDS))
