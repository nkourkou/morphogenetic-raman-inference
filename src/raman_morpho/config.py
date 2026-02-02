"""Project-wide configuration (seed, plotting defaults)."""
import os
import random
import numpy as np

SEED = 1

def set_global_seed(seed: int = SEED) -> np.random.Generator:
    """Set python/numpy seed and return a NumPy Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
