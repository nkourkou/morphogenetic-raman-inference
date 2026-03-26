from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(slots=True)
class FullModelResult:
    pipeline: Pipeline
    logits_test: np.ndarray


def fit_full_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 1,
) -> FullModelResult:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="auto",
                    solver="lbfgs",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    logits = pipe.decision_function(X_test)
    return FullModelResult(pipeline=pipe, logits_test=logits)


def fit_band_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    band_indices: list[np.ndarray],
    *,
    seed: int = 1,
) -> tuple[np.ndarray, list[Pipeline]]:
    k_bands = len(band_indices)
    n_classes = len(np.unique(y_train))
    band_logits_test = np.zeros((X_test.shape[0], k_bands, n_classes), dtype=float)
    band_pipes: list[Pipeline] = []

    for k, idx in enumerate(band_indices):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        multi_class="auto",
                        solver="lbfgs",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X_train[:, idx], y_train)
        band_logits_test[:, k, :] = pipe.decision_function(X_test[:, idx])
        band_pipes.append(pipe)

    return band_logits_test, band_pipes


def fit_auxiliary_full_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 1,
) -> dict[str, float]:
    pipe_svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, random_state=seed)),
        ]
    )
    clf_rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    pipe_svm.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)

    p_svm = pipe_svm.predict_proba(X_test)
    y_pred_svm = np.argmax(p_svm, axis=1)

    p_rf = clf_rf.predict_proba(X_test)
    y_pred_rf = np.argmax(p_rf, axis=1)

    return {
        "acc_svm": float(accuracy_score(y_test, y_pred_svm)),
        "auc_svm": float(roc_auc_score(y_test, p_svm, multi_class="ovr", average="macro")),
        "acc_rf": float(accuracy_score(y_test, y_pred_rf)),
        "auc_rf": float(roc_auc_score(y_test, p_rf, multi_class="ovr", average="macro")),
    }
