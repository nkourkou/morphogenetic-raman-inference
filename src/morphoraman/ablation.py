from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .evaluation import macro_auc_ovr
from .morphogenesis import compute_dg_index, morphogenetic_consensus_one, softmax
from .perturbation import damage_selected_bands


def per_band_ablation(
    *,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    bands: list[tuple[float, float]],
    damage_modes: tuple[str, ...] = ("mixed", "spiky"),
    regimes: tuple[str, ...] = ("plain", "H1", "H2"),
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple[float, float] = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    run_id: int = 0,
    spike_scale_mult: float = 6.0,
) -> pd.DataFrame:
    n_samples, k_bands, n_classes = band_logits_test.shape

    baseline: dict[str, dict[str, float]] = {}
    for reg in regimes:
        y_hat = np.zeros(n_samples, dtype=int)
        p_hat = np.zeros((n_samples, n_classes), dtype=float)
        dg_vals = np.zeros(n_samples, dtype=float)

        for i in range(n_samples):
            l0_list = [band_logits_test[i, k, :] for k in range(k_bands)]
            frozen = np.zeros(k_bands, dtype=bool)
            w0 = (
                np.ones(k_bands, dtype=float)
                if morpho_weighting == "uniform"
                else np.clip(np.array([np.max(softmax(l)) for l in l0_list]), 0.0, 0.999)
            )
            res = morphogenetic_consensus_one(
                l0_list,
                frozen_flags=frozen,
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
            y_hat[i] = int(res["y_hat"])
            p_hat[i] = np.asarray(res["p_final"], dtype=float)
            dg_vals[i] = compute_dg_index(res["l_hist"])

        baseline[reg] = {
            "acc": float(accuracy_score(y_test, y_hat)),
            "auc": float(macro_auc_ovr(y_test, p_hat)),
            "dg_mean": float(np.mean(dg_vals)),
        }

    rows: list[dict] = []
    for dmg in damage_modes:
        for reg in regimes:
            for k_ablate in range(k_bands):
                y_hat = np.zeros(n_samples, dtype=int)
                p_hat = np.zeros((n_samples, n_classes), dtype=float)
                dg_vals = np.zeros(n_samples, dtype=float)

                for i in range(n_samples):
                    l0 = band_logits_test[i].copy()
                    s = float(np.std(l0))
                    noise_scale = s if s > 1e-8 else 1.0
                    seed = (run_id * 1_000_000) + (1000 * (k_ablate + 1)) + i
                    rng = np.random.default_rng(seed)

                    ld = damage_selected_bands(
                        l0,
                        [k_ablate],
                        mode=dmg,
                        rng=rng,
                        noise_scale=noise_scale,
                        spike_scale_mult=spike_scale_mult,
                    )
                    ld_list = [ld[j, :] for j in range(k_bands)]
                    frozen = np.zeros(k_bands, dtype=bool)
                    w0 = (
                        np.ones(k_bands, dtype=float)
                        if morpho_weighting == "uniform"
                        else np.clip(np.array([np.max(softmax(l)) for l in ld_list]), 0.0, 0.999)
                    )

                    res = morphogenetic_consensus_one(
                        ld_list,
                        frozen_flags=frozen,
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

                    y_hat[i] = int(res["y_hat"])
                    p_hat[i] = np.asarray(res["p_final"], dtype=float)
                    dg_vals[i] = compute_dg_index(res["l_hist"])

                acc = float(accuracy_score(y_test, y_hat))
                auc = float(macro_auc_ovr(y_test, p_hat))
                dg_mean = float(np.mean(dg_vals))
                acc0 = baseline[reg]["acc"]
                auc0 = baseline[reg]["auc"]

                rows.append(
                    {
                        "damage_mode": dmg,
                        "regime": reg,
                        "band": k_ablate,
                        "band_range": f"{bands[k_ablate][0]}–{bands[k_ablate][1]}",
                        "acc": acc,
                        "auc": auc,
                        "dg_mean": dg_mean,
                        "delta_acc": float(acc - acc0),
                        "delta_auc": float(auc - auc0),
                        "baseline_acc": float(acc0),
                        "baseline_auc": float(auc0),
                        "baseline_dg_mean": float(baseline[reg]["dg_mean"]),
                    }
                )
    return pd.DataFrame(rows)


def band_pair_ablation(
    *,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    bands: list[tuple[float, float]],
    damage_modes: tuple[str, ...] = ("mixed", "spiky"),
    regimes: tuple[str, ...] = ("plain", "H1", "H2"),
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple[float, float] = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    run_id: int = 0,
    spike_scale_mult: float = 6.0,
) -> pd.DataFrame:
    n_samples, k_bands, n_classes = band_logits_test.shape

    baseline: dict[str, dict[str, float]] = {}
    for reg in regimes:
        y_hat = np.zeros(n_samples, dtype=int)
        p_hat = np.zeros((n_samples, n_classes), dtype=float)
        for i in range(n_samples):
            l0 = band_logits_test[i]
            l0_list = [l0[k, :] for k in range(k_bands)]
            frozen = np.zeros(k_bands, dtype=bool)
            w0 = (
                np.ones(k_bands, dtype=float)
                if morpho_weighting == "uniform"
                else np.clip(np.array([np.max(softmax(l)) for l in l0_list]), 0.0, 0.999)
            )
            res = morphogenetic_consensus_one(
                l0_list,
                frozen_flags=frozen,
                band_weights0=w0,
                regime=reg,
                weight_clip=weight_clip,
                alpha=alpha,
                lam=lam,
                T=T,
                tau=tau,
                max_iter=max_iter,
                use_js=True,
                return_trajectories=False,
            )
            y_hat[i] = int(res["y_hat"])
            p_hat[i] = np.asarray(res["p_final"], dtype=float)
        baseline[reg] = {
            "acc": float(accuracy_score(y_test, y_hat)),
            "auc": float(macro_auc_ovr(y_test, p_hat)),
        }

    rows: list[dict] = []
    for dmg in damage_modes:
        for reg in regimes:
            for a in range(k_bands):
                for b in range(k_bands):
                    y_hat = np.zeros(n_samples, dtype=int)
                    p_hat = np.zeros((n_samples, n_classes), dtype=float)
                    dg_vals = np.zeros(n_samples, dtype=float)

                    for i in range(n_samples):
                        l0 = band_logits_test[i].copy()
                        s = float(np.std(l0))
                        noise_scale = s if s > 1e-8 else 1.0
                        seed = (run_id * 1_000_000) + (10_000 * (a + 1)) + (100 * (b + 1)) + i
                        rng = np.random.default_rng(seed)

                        ld = damage_selected_bands(
                            l0,
                            sorted({a, b}),
                            mode=dmg,
                            rng=rng,
                            noise_scale=noise_scale,
                            spike_scale_mult=spike_scale_mult,
                        )
                        ld_list = [ld[k, :] for k in range(k_bands)]
                        frozen = np.zeros(k_bands, dtype=bool)
                        w0 = (
                            np.ones(k_bands, dtype=float)
                            if morpho_weighting == "uniform"
                            else np.clip(np.array([np.max(softmax(l)) for l in ld_list]), 0.0, 0.999)
                        )

                        res = morphogenetic_consensus_one(
                            ld_list,
                            frozen_flags=frozen,
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
                        y_hat[i] = int(res["y_hat"])
                        p_hat[i] = np.asarray(res["p_final"], dtype=float)
                        dg_vals[i] = compute_dg_index(res["l_hist"])

                    acc = float(accuracy_score(y_test, y_hat))
                    auc = float(macro_auc_ovr(y_test, p_hat))
                    dg_mean = float(np.mean(dg_vals))
                    acc0 = baseline[reg]["acc"]
                    auc0 = baseline[reg]["auc"]

                    rows.append(
                        {
                            "damage_mode": dmg,
                            "regime": reg,
                            "band_a": a,
                            "band_b": b,
                            "band_a_range": f"{bands[a][0]}–{bands[a][1]}",
                            "band_b_range": f"{bands[b][0]}–{bands[b][1]}",
                            "acc": acc,
                            "auc": auc,
                            "dg_mean": dg_mean,
                            "delta_acc": float(acc - acc0),
                            "delta_auc": float(auc - auc0),
                        }
                    )
    return pd.DataFrame(rows)
