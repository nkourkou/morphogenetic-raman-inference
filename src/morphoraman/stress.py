from __future__ import annotations

import numpy as np
import pandas as pd

from .morphogenesis import compute_dg_index, js_div, kl_div, morphogenetic_consensus_one, softmax
from .perturbation import apply_random_band_damage


def stress_history_from_l_hist(
    l_hist: np.ndarray,
    *,
    use_js: bool = True,
    local_coupling: bool = True,
) -> np.ndarray:
    l_hist = np.asarray(l_hist, dtype=float)
    tp1, k_bands, _ = l_hist.shape

    if local_coupling:
        neighbours: list[list[int]] = []
        for k in range(k_bands):
            neigh: list[int] = []
            if k - 1 >= 0:
                neigh.append(k - 1)
            if k + 1 < k_bands:
                neigh.append(k + 1)
            neighbours.append(neigh)
    else:
        neighbours = [[j for j in range(k_bands) if j != k] for k in range(k_bands)]

    div = js_div if use_js else kl_div
    stress = np.zeros((tp1, k_bands), dtype=float)

    for t in range(tp1):
        l_now = l_hist[t]
        for k in range(k_bands):
            neigh = neighbours[k]
            if not neigh:
                continue
            l_neigh_avg = np.mean(l_now[neigh, :], axis=0)
            stress[t, k] = div(softmax(l_now[k, :]), softmax(l_neigh_avg))
    return stress


def summarize_stress_features(
    stress_hist: np.ndarray,
    step_dg_hist: np.ndarray,
    *,
    tau: float,
) -> dict[str, np.ndarray | float]:
    stress = np.asarray(stress_hist, dtype=float)
    step_dg = np.asarray(step_dg_hist, dtype=float)

    stress_auc = stress.sum(axis=0)
    stress_max = stress.max(axis=0)
    dg_band = step_dg.sum(axis=0)

    tp1, k_bands = stress.shape
    t_relax = np.full(k_bands, np.nan, dtype=float)
    for k in range(k_bands):
        below = stress[:, k] <= tau
        for t in range(tp1):
            if below[t] and np.all(below[t:]):
                t_relax[k] = float(t)
                break

    return {
        "stress_auc_global": float(stress_auc.sum()),
        "stress_max_global": float(stress_max.max()),
        "dg_total_from_steps": float(dg_band.sum()),
        "stress_auc_per_band": stress_auc,
        "stress_max_per_band": stress_max,
        "dg_per_band": dg_band,
        "t_relax_per_band": t_relax,
    }


def compute_band_summaries(
    stress_hist: np.ndarray,
    step_dg_hist: np.ndarray,
    *,
    tau: float,
) -> dict[str, np.ndarray]:
    stress = np.asarray(stress_hist, dtype=float)
    step_dg = np.asarray(step_dg_hist, dtype=float)
    tp1, k_bands = stress.shape

    stress_auc = stress.sum(axis=0)
    stress_max = stress.max(axis=0)
    dg_band = step_dg.sum(axis=0)

    t_relax = np.full(k_bands, np.nan, dtype=float)
    for k in range(k_bands):
        below = stress[:, k] <= tau
        for t in range(tp1):
            if below[t] and np.all(below[t:]):
                t_relax[k] = float(t)
                break

    return {
        "stress_auc": stress_auc,
        "stress_max": stress_max,
        "dg_band": dg_band,
        "t_relax": t_relax,
    }


def detect_lesions(
    stress_auc: np.ndarray,
    dg_band: np.ndarray,
    *,
    top_k: int = 2,
    use_dg: bool = True,
) -> np.ndarray:
    k_bands = len(stress_auc)
    lesion = np.zeros(k_bands, dtype=bool)

    idx_stress = np.argsort(stress_auc)[::-1][: max(1, top_k)]
    lesion[idx_stress] = True

    if use_dg:
        idx_dg = np.argsort(dg_band)[::-1][: max(1, top_k)]
        lesion[idx_dg] = True

    return lesion


def build_mechanism_table(
    *,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    damage_mode: str,
    regime: str,
    damage_fraction: float = 0.4,
    run_id: int = 0,
    tau: float = 0.05,
    max_iter: int = 30,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    morpho_weighting: str = "confidence",
    weight_clip: tuple[float, float] = (0.0, 5.0),
) -> pd.DataFrame:
    n_samples, k_bands, _ = band_logits_test.shape
    rows: list[dict] = []

    for i in range(n_samples):
        band_logits0_i = [band_logits_test[i, k, :] for k in range(k_bands)]
        seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i
        rng = np.random.default_rng(seed)

        damaged_logits0_i, frozen_flags_i, damaged_idx = apply_random_band_damage(
            band_logits0_i, damage_fraction=damage_fraction, mode=damage_mode, rng=rng
        )

        if morpho_weighting == "confidence":
            w0 = np.array([np.max(softmax(l)) for l in damaged_logits0_i], dtype=float)
            w0 = np.clip(w0, 0.0, 0.999)
        else:
            w0 = np.ones(k_bands, dtype=float)

        res = morphogenetic_consensus_one(
            damaged_logits0_i,
            frozen_flags=frozen_flags_i,
            band_weights0=w0,
            regime=regime,
            weight_clip=weight_clip,
            alpha=alpha,
            lam=lam,
            T=T,
            tau=tau,
            max_iter=max_iter,
            use_js=True,
            return_trajectories=True,
            record_stress=True,
            rng=rng,
        )

        y_hat = int(res["y_hat"])
        correct = int(y_hat == int(y_test[i]))

        l_hist = res["l_hist"]
        dg_total = compute_dg_index(l_hist)
        t_steps = l_hist.shape[0] - 1
        k_early = max(1, int(round(0.25 * t_steps)))
        dg_early = compute_dg_index(l_hist[: k_early + 1])

        feat = summarize_stress_features(res["stress_hist"], res["step_dg_hist"], tau=tau)

        row = {
            "i": i,
            "damage_mode": damage_mode,
            "regime": regime,
            "damage_fraction": float(damage_fraction),
            "y_true": int(y_test[i]),
            "y_pred": y_hat,
            "true_label": label_encoder.inverse_transform([int(y_test[i])])[0],
            "pred_label": label_encoder.inverse_transform([y_hat])[0],
            "correct": correct,
            "DG": float(dg_total),
            "DG_early": float(dg_early),
            "REDG": float(dg_early / (dg_total + 1e-12)),
            "stress_auc_global": feat["stress_auc_global"],
            "stress_max_global": feat["stress_max_global"],
            "dg_total_from_steps": feat["dg_total_from_steps"],
            "damaged_idx": ",".join(map(str, damaged_idx.tolist())),
        }

        for k in range(k_bands):
            row[f"stress_auc_b{k}"] = float(feat["stress_auc_per_band"][k])
            row[f"stress_max_b{k}"] = float(feat["stress_max_per_band"][k])
            row[f"dg_b{k}"] = float(feat["dg_per_band"][k])
            row[f"t_relax_b{k}"] = (
                float(feat["t_relax_per_band"][k])
                if np.isfinite(feat["t_relax_per_band"][k])
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)
