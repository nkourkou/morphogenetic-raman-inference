from __future__ import annotations

from typing import Optional

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    squeeze = False
    if logits.ndim == 1:
        logits = logits[None, :]
        squeeze = True
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    out = exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-12)
    return out[0] if squeeze else out


stable_softmax = softmax


def _as_kc(x: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(x, list):
        x = np.stack([np.asarray(v, dtype=float) for v in x], axis=0)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected (K,C) or list-of-(C,), got shape={x.shape}")
    return x


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)


def energy_bands(
    band_logits: np.ndarray | list[np.ndarray],
    band_logits0: np.ndarray | list[np.ndarray],
    weights: np.ndarray,
    *,
    lam: float = 1.0,
    use_js: bool = True,
    local_coupling: bool = True,
) -> float:
    l_now = _as_kc(band_logits)
    l_ref = _as_kc(band_logits0)
    if l_now.shape != l_ref.shape:
        raise ValueError(f"Shape mismatch: {l_now.shape} vs {l_ref.shape}")

    k_bands, _ = l_now.shape
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != k_bands:
        raise ValueError(f"weights must have shape ({k_bands},), got {w.shape}")

    p = softmax(l_now)
    p0 = softmax(l_ref)

    if use_js:
        m = 0.5 * (p + p0)
        e_data = 0.5 * np.sum(w * np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12)), axis=1))
        e_data += 0.5 * np.sum(w * np.sum(p0 * (np.log(p0 + 1e-12) - np.log(m + 1e-12)), axis=1))
    else:
        e_data = np.sum(w * np.sum(p * (np.log(p + 1e-12) - np.log(p0 + 1e-12)), axis=1))

    e_cons = 0.0
    if k_bands >= 2:
        if local_coupling:
            for k in range(k_bands - 1):
                if use_js:
                    e_cons += js_div(p[k], p[k + 1])
                else:
                    e_cons += kl_div(p[k], p[k + 1]) + kl_div(p[k + 1], p[k])
        else:
            for k in range(k_bands):
                for j in range(k + 1, k_bands):
                    if use_js:
                        e_cons += js_div(p[k], p[j])
                    else:
                        e_cons += kl_div(p[k], p[j]) + kl_div(p[j], p[k])

    return float(e_data + lam * e_cons)


def global_logits_from_bands(
    band_logits: np.ndarray | list[np.ndarray],
    weights: np.ndarray,
    *,
    normalize_weights: bool = True,
) -> np.ndarray:
    l_now = _as_kc(band_logits)
    k_bands, _ = l_now.shape
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != k_bands:
        raise ValueError(f"weights must have shape ({k_bands},), got {w.shape}")
    if normalize_weights:
        w = w / (np.sum(w) + 1e-12)
    return (w[:, None] * l_now).sum(axis=0)


def global_unsortedness(l_global: np.ndarray) -> float:
    p = softmax(np.asarray(l_global, dtype=float))
    return float(1.0 - np.max(p))


def compute_dg_index(l_hist: np.ndarray) -> float:
    l_hist = np.asarray(l_hist, dtype=float)
    if l_hist.ndim != 3:
        raise ValueError(f"Expected (T+1,K,C), got {l_hist.shape}")
    if l_hist.shape[0] < 2:
        return 0.0
    diffs = l_hist[1:] - l_hist[:-1]
    step_norms = np.linalg.norm(diffs, axis=2)
    return float(step_norms.sum())


def _build_neighbours(k_bands: int, local_coupling: bool) -> list[list[int]]:
    if local_coupling:
        out: list[list[int]] = []
        for k in range(k_bands):
            neigh: list[int] = []
            if k - 1 >= 0:
                neigh.append(k - 1)
            if k + 1 < k_bands:
                neigh.append(k + 1)
            out.append(neigh)
        return out
    return [[j for j in range(k_bands) if j != k] for k in range(k_bands)]


def morphogenetic_consensus_one(
    band_logits0: np.ndarray | list[np.ndarray],
    *,
    frozen_flags: Optional[np.ndarray] = None,
    band_weights0: Optional[np.ndarray] = None,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    use_js: bool = True,
    return_trajectories: bool = False,
    record_stress: bool = False,
    regime: str = "plain",
    local_coupling: bool = True,
    mute_weight: float = 0.0,
    diff_amp: float = 0.25,
    weight_clip: tuple[float, float] = (0.0, 5.0),
    rng: Optional[np.random.Generator] = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(42)

    l0 = _as_kc(band_logits0)
    k_bands, c_classes = l0.shape

    if frozen_flags is None:
        frozen_flags = np.zeros(k_bands, dtype=bool)
    frozen_flags = np.asarray(frozen_flags, dtype=bool)
    if frozen_flags.shape != (k_bands,):
        raise ValueError(f"frozen_flags must have shape ({k_bands},), got {frozen_flags.shape}")

    l_current = l0.copy()

    if band_weights0 is None:
        w = np.max(softmax(l0), axis=1)
        w = np.clip(w, 0.0, 1.0)
    else:
        w = np.asarray(band_weights0, dtype=float).copy()
        if w.shape != (k_bands,):
            raise ValueError(f"band_weights0 must have shape ({k_bands},), got {w.shape}")

    wmin, wmax = float(weight_clip[0]), float(weight_clip[1])
    neighbours = _build_neighbours(k_bands, local_coupling)
    div = js_div if use_js else kl_div

    e_hist: list[float] = []
    u_hist: list[float] = []
    l_global_hist: list[np.ndarray] = []
    l_hist: list[np.ndarray] = []
    stress_hist: list[np.ndarray] = []
    step_dg_hist: list[np.ndarray] = []

    if return_trajectories:
        l_hist.append(l_current.copy())

    e_prev = energy_bands(l_current, l0, w, lam=lam, use_js=use_js, local_coupling=local_coupling)
    l_global = global_logits_from_bands(l_current, w)
    u_prev = global_unsortedness(l_global)
    e_hist.append(e_prev)
    u_hist.append(u_prev)
    l_global_hist.append(l_global.copy())

    if record_stress:
        s0 = np.zeros(k_bands, dtype=float)
        for k in range(k_bands):
            neigh = neighbours[k]
            if not neigh:
                continue
            w_neigh = w[neigh]
            z = float(np.sum(w_neigh)) + 1e-12
            l_neigh_avg = np.zeros(c_classes, dtype=float)
            for j, wj in zip(neigh, w_neigh):
                l_neigh_avg += float(wj) * l_current[j]
            l_neigh_avg /= z
            s0[k] = div(softmax(l_current[k]), softmax(l_neigh_avg))
        stress_hist.append(s0)

    for _ in range(max_iter):
        l_old = l_current.copy()
        w_old = w.copy()

        l_prop = l_current.copy()
        w_prop = w.copy()

        for k in range(k_bands):
            if frozen_flags[k]:
                continue

            neigh = neighbours[k]
            if not neigh:
                continue

            w_neigh = w[neigh]
            z = float(np.sum(w_neigh)) + 1e-12
            l_neigh_avg = np.zeros(c_classes, dtype=float)
            for j, wj in zip(neigh, w_neigh):
                l_neigh_avg += float(wj) * l_current[j]
            l_neigh_avg /= z

            stress = div(softmax(l_current[k]), softmax(l_neigh_avg))
            reg = regime.lower()

            if reg == "plain":
                l_prop[k] = (1.0 - alpha) * l_current[k] + alpha * l_neigh_avg
            elif reg == "h1":
                if stress > tau:
                    w_prop[k] = mute_weight
                    l_prop[k] = l_neigh_avg
                else:
                    l_prop[k] = (1.0 - alpha) * l_current[k] + alpha * l_neigh_avg
            elif reg == "h2":
                if stress > tau:
                    w_prop[k] = mute_weight
                    l_prop[k] = l_neigh_avg
                else:
                    l_prop[k] = (1.0 - alpha) * l_current[k] + alpha * l_neigh_avg
                    w_prop[k] = min(wmax, max(wmin, w_prop[k] * (1.0 + diff_amp)))
            else:
                raise ValueError(f"Unknown regime: {regime}")

        w_prop = np.clip(w_prop, wmin, wmax)
        e_new = energy_bands(l_prop, l0, w_prop, lam=lam, use_js=use_js, local_coupling=local_coupling)

        accept = True
        if (e_new > e_prev) and (T > 0.0):
            prob = float(np.exp(-(e_new - e_prev) / T))
            if rng.random() >= prob:
                accept = False

        if accept:
            l_current = l_prop
            w = w_prop
            e_prev = e_new
        else:
            l_current = l_old
            w = w_old

        if record_stress or return_trajectories:
            step_dg_hist.append(np.linalg.norm(l_current - l_old, axis=1))

        if record_stress:
            s = np.zeros(k_bands, dtype=float)
            for k in range(k_bands):
                neigh = neighbours[k]
                if not neigh:
                    continue
                w_neigh = w[neigh]
                z = float(np.sum(w_neigh)) + 1e-12
                l_neigh_avg = np.zeros(c_classes, dtype=float)
                for j, wj in zip(neigh, w_neigh):
                    l_neigh_avg += float(wj) * l_current[j]
                l_neigh_avg /= z
                s[k] = div(softmax(l_current[k]), softmax(l_neigh_avg))
            stress_hist.append(s)

        l_global = global_logits_from_bands(l_current, w)
        u_prev = global_unsortedness(l_global)
        e_hist.append(e_prev)
        u_hist.append(u_prev)
        l_global_hist.append(l_global.copy())

        if return_trajectories:
            l_hist.append(l_current.copy())

    l_final = global_logits_from_bands(l_current, w)
    y_hat = int(np.argmax(l_final))
    p_final = softmax(l_final)

    out: dict[str, np.ndarray | float | int] = {
        "y_hat": y_hat,
        "p_final": p_final,
        "band_logits_final": l_current.copy(),
        "weights_final": w.copy(),
    }

    if return_trajectories:
        out["E_hist"] = np.asarray(e_hist, dtype=float)
        out["U_hist"] = np.asarray(u_hist, dtype=float)
        out["l_global_hist"] = np.stack(l_global_hist, axis=0)
        out["l_hist"] = np.stack(l_hist, axis=0)

    if record_stress:
        out["stress_hist"] = np.stack(stress_hist, axis=0) if stress_hist else np.empty((0, k_bands))
        out["step_dg_hist"] = np.stack(step_dg_hist, axis=0) if step_dg_hist else np.empty((0, k_bands))

    return out
