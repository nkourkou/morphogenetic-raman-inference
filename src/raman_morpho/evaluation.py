"""Evaluation utilities: accuracy/AUC sweeps under damage."""
from __future__ import annotations
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from .morpho import softmax, global_logits_from_bands, morphogenetic_consensus_one, apply_random_band_damage, compute_dg_index

def evaluate_morphogenetic_system(
    full_logits_test: np.ndarray,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    damage_fractions: List[float],
    *,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    damage_mode: str = "mixed",
    naive_weighting: str = "uniform",
    morpho_weighting: str = "confidence",
    regime: str = "plain",
    run_id: int = 0,
    weight_clip: tuple = (0.0, 5.0),
    auc_multi_class: str = "ovr",
    seed: int = 1,
) -> Dict:
    N, K, C = band_logits_test.shape

    results = {
        "damage_fraction": [],
        "acc_full": [],
        "acc_naive": [],
        "acc_morpho": [],
        "auc_full": [],
        "auc_naive": [],
        "auc_morpho": [],
        "dg_index_mean": [],
        "dg_index_std": [],
        "regime": [],
        "damage_mode": [],
        "run_id": [],
    }

    y_pred_full_clean = np.argmax(full_logits_test, axis=1)
    acc_full_clean = accuracy_score(y_test, y_pred_full_clean)
    P_full = softmax(full_logits_test)
    try:
        auc_full_clean = roc_auc_score(y_test, P_full, multi_class=auc_multi_class, average="macro")
    except ValueError:
        auc_full_clean = float("nan")

    # clean naive baseline
    y_pred_naive_clean = []
    P_naive_clean = np.zeros((N, C), dtype=float)
    for i in range(N):
        L = band_logits_test[i]
        if naive_weighting == "uniform":
            w_naive = np.ones(K, dtype=float)
        elif naive_weighting == "confidence":
            w_naive = np.clip(np.max(softmax(L), axis=1), 0.0, 0.999)
        else:
            raise ValueError(f"Unknown naive_weighting: {naive_weighting}")
        l_avg = global_logits_from_bands(L, w_naive)
        y_pred_naive_clean.append(int(np.argmax(l_avg)))
        P_naive_clean[i] = softmax(l_avg)
    acc_naive_clean = accuracy_score(y_test, y_pred_naive_clean)
    try:
        auc_naive_clean = roc_auc_score(y_test, P_naive_clean, multi_class=auc_multi_class, average="macro")
    except ValueError:
        auc_naive_clean = float("nan")

    acc_full, auc_full = acc_full_clean, auc_full_clean

    for frac in damage_fractions:
        y_pred_naive, y_pred_morpho, dg_indices = [], [], []
        P_naive = np.zeros((N, C), dtype=float)
        P_morpho = np.zeros((N, C), dtype=float)

        for i in range(N):
            L0 = band_logits_test[i]
            local_seed = (run_id * 1_000_000) + (10_000 * int(round(frac * 100))) + i + seed
            rng = np.random.default_rng(local_seed)

            damaged_L0, frozen_flags, _ = apply_random_band_damage(
                L0, damage_fraction=frac, mode=damage_mode, rng=rng
            )

            # naive aggregation on damaged logits
            if naive_weighting == "uniform":
                w_naive = np.ones(K, dtype=float)
            else:
                w_naive = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)

            l_avg = global_logits_from_bands(damaged_L0, w_naive)
            y_pred_naive.append(int(np.argmax(l_avg)))
            P_naive[i] = softmax(l_avg)

            # morpho initial weights
            if morpho_weighting == "uniform":
                w_morpho = np.ones(K, dtype=float)
            else:
                w_morpho = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)

            res = morphogenetic_consensus_one(
                damaged_L0,
                frozen_flags=frozen_flags,
                band_weights0=w_morpho,
                regime=regime,
                weight_clip=weight_clip,
                alpha=alpha,
                lam=lam,
                T=T,
                tau=tau,
                max_iter=max_iter,
                use_js=True,
                return_trajectories=True,
                rng=rng
            )
            y_pred_morpho.append(int(res["y_hat"]))
            P_morpho[i] = res["p_final"]
            dg_indices.append(compute_dg_index(res["l_hist"]))

        acc_naive = accuracy_score(y_test, y_pred_naive)
        acc_morpho = accuracy_score(y_test, y_pred_morpho)

        try:
            auc_naive = roc_auc_score(y_test, P_naive, multi_class=auc_multi_class, average="macro")
        except ValueError:
            auc_naive = float("nan")
        try:
            auc_morpho = roc_auc_score(y_test, P_morpho, multi_class=auc_multi_class, average="macro")
        except ValueError:
            auc_morpho = float("nan")

        results["damage_fraction"].append(float(frac))
        results["acc_full"].append(float(acc_full))
        results["acc_naive"].append(float(acc_naive))
        results["acc_morpho"].append(float(acc_morpho))
        results["auc_full"].append(float(auc_full))
        results["auc_naive"].append(float(auc_naive))
        results["auc_morpho"].append(float(auc_morpho))
        results["dg_index_mean"].append(float(np.mean(dg_indices)))
        results["dg_index_std"].append(float(np.std(dg_indices)))
        results["regime"].append(regime)
        results["damage_mode"].append(damage_mode)
        results["run_id"].append(int(run_id))

    return results
