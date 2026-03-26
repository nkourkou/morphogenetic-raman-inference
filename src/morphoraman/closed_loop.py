from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .morphogenesis import morphogenetic_consensus_one, softmax
from .perturbation import apply_random_band_damage
from .stress import compute_band_summaries, detect_lesions


def closed_loop_consensus_one(
    band_logits0: list[np.ndarray],
    *,
    damage_fraction: float,
    damage_mode: str,
    regime: str,
    run_seed: int,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple[float, float] = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    top_k: int = 2,
    use_dg_in_lesion: bool = True,
    mute_weight: float = 0.0,
) -> dict:
    rng = np.random.default_rng(run_seed)

    damaged_logits0, frozen_flags0, damaged_idx = apply_random_band_damage(
        band_logits0, damage_fraction=damage_fraction, mode=damage_mode, rng=rng
    )
    k_bands = len(damaged_logits0)

    if morpho_weighting == "confidence":
        w0 = np.array([np.max(softmax(l)) for l in damaged_logits0], dtype=float)
        w0 = np.clip(w0, 0.0, 0.999)
    else:
        w0 = np.ones(k_bands, dtype=float)

    res1 = morphogenetic_consensus_one(
        damaged_logits0,
        frozen_flags=frozen_flags0,
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

    summ = compute_band_summaries(res1["stress_hist"], res1["step_dg_hist"], tau=tau)
    lesion_flags = detect_lesions(
        summ["stress_auc"],
        summ["dg_band"],
        top_k=top_k,
        use_dg=use_dg_in_lesion,
    )

    frozen_flags2 = frozen_flags0.copy()
    frozen_flags2[lesion_flags] = True
    w2 = w0.copy()
    w2[lesion_flags] = mute_weight

    res2 = morphogenetic_consensus_one(
        damaged_logits0,
        frozen_flags=frozen_flags2,
        band_weights0=w2,
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

    return {
        "res1": res1,
        "res2": res2,
        "lesion_flags": lesion_flags,
        "damaged_idx": damaged_idx,
        "summ1": summ,
        "w0": w0,
        "w2": w2,
        "frozen2": frozen_flags2,
    }


def evaluate_closed_loop(
    *,
    band_logits_test: np.ndarray,
    y_test: np.ndarray,
    bands: list[tuple[float, float]],
    damage_fraction: float = 0.4,
    damage_mode: str = "spiky",
    regime: str = "H1",
    run_id: int = 0,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple[float, float] = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    top_k: int = 2,
    use_dg_in_lesion: bool = True,
    mute_weight: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_samples, k_bands, _ = band_logits_test.shape
    rows: list[dict] = []
    lesion_counts = np.zeros(k_bands, dtype=int)

    for i in range(n_samples):
        l0 = [band_logits_test[i, k, :] for k in range(k_bands)]
        seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i

        out = closed_loop_consensus_one(
            l0,
            damage_fraction=damage_fraction,
            damage_mode=damage_mode,
            regime=regime,
            run_seed=seed,
            alpha=alpha,
            lam=lam,
            T=T,
            tau=tau,
            max_iter=max_iter,
            weight_clip=weight_clip,
            morpho_weighting=morpho_weighting,
            top_k=top_k,
            use_dg_in_lesion=use_dg_in_lesion,
            mute_weight=mute_weight,
        )

        y1 = int(out["res1"]["y_hat"])
        y2 = int(out["res2"]["y_hat"])
        ytrue = int(y_test[i])

        lesion = out["lesion_flags"]
        lesion_counts += lesion.astype(int)

        rows.append(
            {
                "i": i,
                "y_true": ytrue,
                "y_hat_pass1": y1,
                "y_hat_pass2": y2,
                "correct_pass1": int(y1 == ytrue),
                "correct_pass2": int(y2 == ytrue),
                "damage_mode": damage_mode,
                "regime": regime,
                "damage_fraction": float(damage_fraction),
                "lesion_flags": tuple(np.where(lesion)[0].tolist()),
            }
        )

    df = pd.DataFrame(rows)
    lesion_freq = pd.DataFrame(
        {
            "band": np.arange(k_bands),
            "band_range": [f"{lo}-{hi}" for lo, hi in bands],
            "lesion_frequency": lesion_counts / len(df),
        }
    )
    _ = accuracy_score(df["y_true"], df["y_hat_pass1"])
    _ = accuracy_score(df["y_true"], df["y_hat_pass2"])
    return df, lesion_freq
