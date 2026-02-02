"""Train full-spectrum and per-band LogisticRegression models without leakage."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_full_model(X_train: np.ndarray, y_train: np.ndarray, *, seed: int = 1) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            multi_class="auto",
            solver="lbfgs",
            n_jobs=-1,
            random_state=seed
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_band_models(X_train: np.ndarray, y_train: np.ndarray, band_indices: list[np.ndarray], *, seed: int = 1):
    band_pipes = []
    for idx in band_indices:
        pipe_k = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                multi_class="auto",
                solver="lbfgs",
                n_jobs=-1,
                random_state=seed
            ))
        ])
        pipe_k.fit(X_train[:, idx], y_train)
        band_pipes.append(pipe_k)
    return band_pipes

def decision_function_band_pipes(X: np.ndarray, band_pipes, band_indices: list[np.ndarray]) -> np.ndarray:
    N = X.shape[0]
    K = len(band_indices)
    # infer C from first model
    first = band_pipes[0].decision_function(X[:, band_indices[0]])
    C = first.shape[1] if first.ndim == 2 else 1
    out = np.zeros((N, K, C), dtype=float)
    out[:, 0, :] = first
    for k in range(1, K):
        out[:, k, :] = band_pipes[k].decision_function(X[:, band_indices[k]])
    return out
