from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from .morphogenesis import (
    compute_dg_index,
    global_logits_from_bands,
    morphogenetic_consensus_one,
    softmax,
)
from .perturbation import apply_band_damage
from .stress import stress_history_from_l_hist


def macro_auc_ovr(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _confidence_weights(l: np.ndarray | list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(l, dtype=float)
    if arr.ndim == 2:
        conf = np.max(softmax(arr), axis=1)
    else:
        conf = np.array([np.max(softmax(v)) for v in l], dtype=float)
    return np.clip(conf, 0.0, 0.999)


def evaluate_morphogenetic_system(
    *,
    full_logits_test: np.ndarray,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    damage_fractions: Iterable[float],
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
    weight_clip: tuple[float, float] = (0.0, 5.0),
    auc_multi_class: str = "ovr",
) -> dict[str, list[float]]:
    n_samples, k_bands, n_classes = band_logits_test.shape

    results: dict[str, list[float]] = {
        "damage_fraction": [],
        "acc_full": [],
        "acc_naive": [],
        "acc_morpho": [],
        "auc_full": [],
        "auc_naive": [],
        "auc_morpho": [],
        "dg_index_mean": [],
        "dg_index_std": [],
    }

    y_pred_full_clean = np.argmax(full_logits_test, axis=1)
    acc_full_clean = accuracy_score(y_test, y_pred_full_clean)
    p_full = softmax(full_logits_test)
    try:
        auc_full_clean = roc_auc_score(y_test, p_full, multi_class=auc_multi_class, average="macro")
    except ValueError:
        auc_full_clean = np.nan

    y_pred_naive_clean = []
    p_naive_clean = np.zeros((n_samples, n_classes), dtype=float)
    for i in range(n_samples):
        l = band_logits_test[i]
        if naive_weighting == "uniform":
            w_naive = np.ones(k_bands, dtype=float)
        elif naive_weighting == "confidence":
            w_naive = _confidence_weights(l)
        else:
            raise ValueError(f"Unknown naive_weighting: {naive_weighting}")

        l_avg = global_logits_from_bands(l, w_naive)
        y_pred_naive_clean.append(int(np.argmax(l_avg)))
        p_naive_clean[i] = softmax(l_avg)

    acc_naive_clean = accuracy_score(y_test, y_pred_naive_clean)
    try:
        auc_naive_clean = roc_auc_score(y_test, p_naive_clean, multi_class=auc_multi_class, average="macro")
    except ValueError:
        auc_naive_clean = np.nan

    acc_full = acc_full_clean
    auc_full = auc_full_clean

    for frac in damage_fractions:
        y_pred_naive: list[int] = []
        y_pred_morpho: list[int] = []
        dg_indices: list[float] = []
        p_naive = np.zeros((n_samples, n_classes), dtype=float)
        p_morpho = np.zeros((n_samples, n_classes), dtype=float)

        for i in range(n_samples):
            l0 = band_logits_test[i]
            seed = (run_id * 1_000_000) + (10_000 * int(round(frac * 100))) + i + 1
            rng = np.random.default_rng(seed)

            damaged_l0, frozen_flags, _ = apply_band_damage(
                l0,
                damage_fraction=frac,
                mode=damage_mode,
                rng=rng,
            )

            if naive_weighting == "uniform":
                w_naive = np.ones(k_bands, dtype=float)
            elif naive_weighting == "confidence":
                w_naive = _confidence_weights(damaged_l0)
            else:
                raise ValueError(f"Unknown naive_weighting: {naive_weighting}")

            l_avg = global_logits_from_bands(damaged_l0, w_naive)
            y_pred_naive.append(int(np.argmax(l_avg)))
            p_naive[i] = softmax(l_avg)

            if morpho_weighting == "uniform":
                w_morpho = np.ones(k_bands, dtype=float)
            elif morpho_weighting == "confidence":
                w_morpho = _confidence_weights(damaged_l0)
            else:
                raise ValueError(f"Unknown morpho_weighting: {morpho_weighting}")

            res = morphogenetic_consensus_one(
                damaged_l0,
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
                rng=rng,
            )

            y_pred_morpho.append(int(res["y_hat"]))
            p_morpho[i] = res["p_final"]
            dg_indices.append(compute_dg_index(res["l_hist"]))

        acc_naive = accuracy_score(y_test, y_pred_naive)
        acc_morpho = accuracy_score(y_test, y_pred_morpho)

        try:
            auc_naive = roc_auc_score(y_test, p_naive, multi_class=auc_multi_class, average="macro")
        except ValueError:
            auc_naive = np.nan

        try:
            auc_morpho = roc_auc_score(y_test, p_morpho, multi_class=auc_multi_class, average="macro")
        except ValueError:
            auc_morpho = np.nan

        results["damage_fraction"].append(float(frac))
        results["acc_full"].append(float(acc_full))
        results["acc_naive"].append(float(acc_naive))
        results["acc_morpho"].append(float(acc_morpho))
        results["auc_full"].append(float(auc_full))
        results["auc_naive"].append(float(auc_naive))
        results["auc_morpho"].append(float(auc_morpho))
        results["dg_index_mean"].append(float(np.mean(dg_indices)))
        results["dg_index_std"].append(float(np.std(dg_indices)))

    return results


def normalize_dg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dg_norm" in df.columns:
        return df
    df["dg_norm"] = np.nan
    for (reg, dmg), dfg in df.groupby(["regime", "damage_mode"]):
        ref = dfg.loc[dfg["damage_fraction"] == 0.0, "dg_index_mean"]
        if len(ref) != 1:
            raise ValueError(f"Expected one zero-damage DG for {reg}, {dmg}")
        ref_val = float(ref.iloc[0])
        df.loc[(df["regime"] == reg) & (df["damage_mode"] == dmg), "dg_norm"] = (
            dfg["dg_index_mean"] / ref_val
        )
    return df


def compute_dg_dataset(
    *,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    damage_fraction: float = 0.4,
    damage_modes: Iterable[str] = ("mixed", "spiky"),
    regimes: Iterable[str] = ("plain", "H1", "H2"),
    run_id: int = 0,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple[float, float] = (0.0, 5.0),
    morpho_weighting: str = "confidence",
) -> pd.DataFrame:
    n_samples, k_bands, _ = band_logits_test.shape
    rows: list[dict] = []

    for dmg in damage_modes:
        for reg in regimes:
            for i in range(n_samples):
                l0_list = [band_logits_test[i, k, :] for k in range(k_bands)]
                y_true = int(y_test[i])

                seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i
                rng = np.random.default_rng(seed)

                ld_list, frozen_flags, damaged_idx = apply_band_damage(
                    l0_list,
                    damage_fraction=damage_fraction,
                    mode=dmg,
                    rng=rng,
                )

                w0 = np.ones(k_bands, dtype=float) if morpho_weighting == "uniform" else _confidence_weights(ld_list)

                res = morphogenetic_consensus_one(
                    ld_list,
                    frozen_flags=frozen_flags,
                    band_weights0=w0,
                    regime=reg,
                    weight_clip=weight_clip,
                    alpha=alpha,
                    lam=lam,
                    T=T,
                    tau=tau,
                    max_iter=max_iter,
                    use_js=True,
                    return_trajectories=True,
                )

                y_hat = int(res["y_hat"])
                correct = int(y_hat == y_true)
                dg = float(compute_dg_index(res["l_hist"]))
                t_steps = max(res["l_hist"].shape[0] - 1, 1)
                dg_norm = dg / (t_steps * k_bands)

                rows.append(
                    {
                        "i": i,
                        "damage_mode": dmg,
                        "regime": reg,
                        "damage_fraction": float(damage_fraction),
                        "y_true": y_true,
                        "y_hat": y_hat,
                        "correct": correct,
                        "DG": dg,
                        "DG_norm": float(dg_norm),
                        "damaged_idx": tuple(sorted(damaged_idx.tolist())),
                    }
                )

    return pd.DataFrame(rows)


def compute_dg_and_stress_table(
    *,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    damage_fraction: float = 0.4,
    damage_modes: Iterable[str] = ("mixed", "spiky"),
    regimes: Iterable[str] = ("plain", "H1", "H2"),
    run_id: int = 0,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple[float, float] = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    k_early: int = 3,
) -> pd.DataFrame:
    n_samples, k_bands, _ = band_logits_test.shape
    rows: list[dict] = []

    for dmg in damage_modes:
        for reg in regimes:
            for i in range(n_samples):
                l0_list = [band_logits_test[i, k, :] for k in range(k_bands)]
                y_true = int(y_test[i])

                seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i
                rng = np.random.default_rng(seed)

                ld_list, frozen_flags, damaged_idx = apply_band_damage(
                    l0_list,
                    damage_fraction=damage_fraction,
                    mode=dmg,
                    rng=rng,
                )

                w0 = np.ones(k_bands, dtype=float) if morpho_weighting == "uniform" else _confidence_weights(ld_list)

                res = morphogenetic_consensus_one(
                    ld_list,
                    frozen_flags=frozen_flags,
                    band_weights0=w0,
                    regime=reg,
                    weight_clip=weight_clip,
                    alpha=alpha,
                    lam=lam,
                    T=T,
                    tau=tau,
                    max_iter=max_iter,
                    use_js=True,
                    return_trajectories=True,
                )

                l_hist = res["l_hist"]
                dg = float(compute_dg_index(l_hist))
                diffs = l_hist[1:] - l_hist[:-1]
                step_norms = np.linalg.norm(diffs, axis=2).sum(axis=1)
                k_use = min(k_early, step_norms.shape[0])
                dg_early = float(np.sum(step_norms[:k_use]))

                stress = stress_history_from_l_hist(l_hist, use_js=True, local_coupling=True)
                stress0 = stress[0, :]
                stress0_mean = float(np.mean(stress0))
                stress0_max = float(np.max(stress0))

                y_hat = int(res["y_hat"])
                correct = int(y_hat == y_true)

                rows.append(
                    {
                        "i": i,
                        "damage_mode": dmg,
                        "regime": reg,
                        "damage_fraction": float(damage_fraction),
                        "y_true": y_true,
                        "y_hat": y_hat,
                        "correct": correct,
                        "DG": dg,
                        "DG_early": dg_early,
                        "stress0_mean": stress0_mean,
                        "stress0_max": stress0_max,
                        "damaged_idx": tuple(sorted(damaged_idx.tolist())),
                    }
                )

    return pd.DataFrame(rows)


def stratified_auc(df: pd.DataFrame, *, score_col: str, stress_col: str) -> pd.DataFrame:
    out: list[dict] = []
    for (dmg, reg), dfr in df.groupby(["damage_mode", "regime"]):
        q = pd.qcut(dfr[stress_col], q=4, labels=[1, 2, 3, 4], duplicates="drop")
        dfr = dfr.copy()
        dfr["q"] = q

        for qq in sorted(dfr["q"].dropna().unique()):
            sub = dfr[dfr["q"] == qq]
            y_wrong = 1 - sub["correct"].to_numpy(dtype=int)
            x = sub[score_col].to_numpy(dtype=float)
            auc = np.nan if len(np.unique(y_wrong)) < 2 else float(roc_auc_score(y_wrong, x))
            out.append(
                {
                    "damage_mode": dmg,
                    "regime": reg,
                    "quartile": int(qq),
                    "n": int(len(sub)),
                    f"AUC({score_col}→wrong)": auc,
                }
            )
    return pd.DataFrame(out)


def auc_by_group(df: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (dmg, reg), dfr in df.groupby(["damage_mode", "regime"]):
        y_wrong = 1 - dfr["correct"].to_numpy(dtype=int)
        x = dfr[score_col].to_numpy(dtype=float)
        auc = np.nan if len(np.unique(y_wrong)) < 2 else float(roc_auc_score(y_wrong, x))
        rows.append({"damage_mode": dmg, "regime": reg, f"AUC({score_col}→wrong)": auc})
    return pd.DataFrame(rows).sort_values(["damage_mode", "regime"]).reset_index(drop=True)


def add_redg_features(df: pd.DataFrame, *, eps: float = 1e-12) -> pd.DataFrame:
    out = df.copy()
    if "DG" not in out.columns or "DG_early" not in out.columns:
        raise KeyError("Expected columns 'DG' and 'DG_early'.")
    out["REDG"] = out["DG_early"] / (out["DG"] + eps)
    return out


def add_eredg(
    df: pd.DataFrame,
    *,
    dg_col: str = "DG",
    dg_early_col: str = "DG_early",
    out_redg_col: str = "REDG",
    out_eredg_col: str = "eREDG",
    eps: float = 1e-12,
    gate_by: str = "per_damage_mode_and_regime",
    gate_q: float = 0.20,
    hard_delta: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if dg_col not in df.columns or dg_early_col not in df.columns:
        raise KeyError(f"Missing required columns: {dg_col}, {dg_early_col}")

    df[out_redg_col] = df[dg_early_col] / (df[dg_col] + eps)
    gate_rows: list[dict] = []

    if hard_delta is not None:
        df["DG_delta"] = float(hard_delta)
        df["DG_active"] = (df[dg_col] > float(hard_delta)).astype(int)
        gate_rows.append({"group": "global", "DG_delta": float(hard_delta), "gate_q": None, "n": int(len(df))})
    elif gate_by == "global":
        delta = float(np.quantile(df[dg_col].to_numpy(dtype=float), gate_q))
        df["DG_delta"] = delta
        df["DG_active"] = (df[dg_col] > delta).astype(int)
        gate_rows.append({"group": "global", "DG_delta": delta, "gate_q": gate_q, "n": int(len(df))})
    elif gate_by == "per_damage_mode_and_regime":
        df["DG_delta"] = np.nan
        df["DG_active"] = 0
        for (dmg, reg), idx in df.groupby(["damage_mode", "regime"]).groups.items():
            sub = df.loc[idx, dg_col].to_numpy(dtype=float)
            delta = float(np.quantile(sub, gate_q))
            df.loc[idx, "DG_delta"] = delta
            df.loc[idx, "DG_active"] = (df.loc[idx, dg_col] > delta).astype(int)
            gate_rows.append(
                {
                    "group": f"{dmg}|{reg}",
                    "damage_mode": dmg,
                    "regime": reg,
                    "DG_delta": delta,
                    "gate_q": gate_q,
                    "n": int(len(idx)),
                }
            )
    else:
        raise ValueError("gate_by must be 'global' or 'per_damage_mode_and_regime'")

    df[out_eredg_col] = df[out_redg_col].where(df["DG_active"] == 1, np.nan)
    return df, pd.DataFrame(gate_rows)


def auc_with_sign_fix(y: np.ndarray, s: np.ndarray) -> tuple[float, int, np.ndarray]:
    auc_pos = roc_auc_score(y, s)
    auc_neg = roc_auc_score(y, -s)
    if auc_neg > auc_pos:
        return float(auc_neg), -1, -s
    return float(auc_pos), +1, s


def bootstrap_auc_ci(
    y: np.ndarray,
    s: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, int, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    auc_hat, sign, s_use = auc_with_sign_fix(y, s)

    aucs: list[float] = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        yb = y[b]
        sb = s_use[b]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, sb)))

    auc_arr = np.asarray(aucs, dtype=float)
    lo, hi = np.quantile(auc_arr, [0.025, 0.975])
    return auc_hat, float(lo), float(hi), int(sign), auc_arr


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))
