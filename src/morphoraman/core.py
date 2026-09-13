import numpy as np
SEED = 1
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from typing import List, Union

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-12)

def _as_KC(x: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
    if isinstance(x, list):
        x = np.stack([np.asarray(v, dtype=float) for v in x], axis=0)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f'Expected (K,C) or list-of-(C,), got {x.shape}')
    return x

def kl_div(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def js_div(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)

def energy_bands(band_logits: Union[List[np.ndarray], np.ndarray], band_logits0: Union[List[np.ndarray], np.ndarray], weights: np.ndarray, lam: float=1.0, use_js: bool=True, local_coupling: bool=True) -> float:
    L = _as_KC(band_logits)
    L0 = _as_KC(band_logits0)
    if L.shape != L0.shape:
        raise ValueError(f'band_logits and band_logits0 must match, got {L.shape} vs {L0.shape}')
    K, C = L.shape
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != K:
        raise ValueError(f'weights must have shape ({K},), got {w.shape}')
    P = softmax(L)
    P0 = softmax(L0)
    if use_js:
        M = 0.5 * (P + P0)
        E_data = 0.5 * np.sum(w * np.sum(P * (np.log(P + 1e-12) - np.log(M + 1e-12)), axis=1))
        E_data += 0.5 * np.sum(w * np.sum(P0 * (np.log(P0 + 1e-12) - np.log(M + 1e-12)), axis=1))
    else:
        E_data = np.sum(w * np.sum(P * (np.log(P + 1e-12) - np.log(P0 + 1e-12)), axis=1))
    E_cons = 0.0
    if K >= 2:
        if local_coupling:
            for k in range(K - 1):
                if use_js:
                    E_cons += js_div(P[k], P[k + 1])
                else:
                    E_cons += kl_div(P[k], P[k + 1]) + kl_div(P[k + 1], P[k])
        else:
            for k in range(K):
                for j in range(k + 1, K):
                    if use_js:
                        E_cons += js_div(P[k], P[j])
                    else:
                        E_cons += kl_div(P[k], P[j]) + kl_div(P[j], P[k])
    return float(E_data + lam * E_cons)

def global_logits_from_bands(band_logits: Union[List[np.ndarray], np.ndarray], weights: np.ndarray, normalize_weights: bool=True) -> np.ndarray:
    L = _as_KC(band_logits)
    K, C = L.shape
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != K:
        raise ValueError(f'weights must have shape ({K},), got {w.shape}')
    if normalize_weights:
        w = w / (np.sum(w) + 1e-12)
    return (w[:, None] * L).sum(axis=0)

def global_unsortedness(l_global: np.ndarray) -> float:
    p = softmax(np.asarray(l_global, dtype=float))
    return float(1.0 - np.max(p))

def center_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    return logits - logits.mean(axis=-1, keepdims=True)


def compute_dg_steps(l_hist: np.ndarray) -> np.ndarray:
    l_hist = np.asarray(l_hist, dtype=float)
    if l_hist.ndim != 3:
        raise ValueError(f'Expected l_hist shape (T+1, K, C), got {l_hist.shape}')
    centered = center_logits(l_hist)
    return np.linalg.norm(np.diff(centered, axis=0), axis=2)


def compute_dg_index(l_hist: np.ndarray) -> float:
    return float(compute_dg_steps(l_hist).sum())
from typing import Optional, Dict, Union
import numpy as np

def morphogenetic_consensus_one(band_logits0: Union[np.ndarray, list], frozen_flags: Optional[np.ndarray]=None, band_weights0: Optional[np.ndarray]=None, alpha: float=0.3, lam: float=1.0, T: float=0.01, tau: float=0.05, max_iter: int=30, use_js: bool=True, return_trajectories: bool=False, regime: str='plain', local_coupling: bool=True, mute_weight: float=0.0, diff_amp: float=0.25, weight_clip: tuple=(0.0, 5.0), rng: Optional[np.random.Generator]=None) -> Dict:
    if rng is None:
        rng = np.random.default_rng(SEED)
    L0 = _as_KC(band_logits0)
    K, C = L0.shape
    if frozen_flags is None:
        frozen_flags = np.zeros(K, dtype=bool)
    frozen_flags = np.asarray(frozen_flags, dtype=bool)
    if frozen_flags.shape != (K,):
        raise ValueError(f'frozen_flags must have shape ({K},), got {frozen_flags.shape}')
    L = L0.copy()
    if band_weights0 is None:
        w0 = np.max(softmax(L0), axis=1)
        w = np.clip(w0, 0.0, 1.0)
    else:
        w = np.asarray(band_weights0, dtype=float).copy()
        if w.shape != (K,):
            raise ValueError(f'band_weights0 must have shape ({K},), got {w.shape}')
    wmin, wmax = (float(weight_clip[0]), float(weight_clip[1]))
    if local_coupling:
        neighbours = []
        for k in range(K):
            neigh = []
            if k - 1 >= 0:
                neigh.append(k - 1)
            if k + 1 < K:
                neigh.append(k + 1)
            neighbours.append(neigh)
    else:
        neighbours = [[j for j in range(K) if j != k] for k in range(K)]
    div = js_div if use_js else kl_div
    E_hist, U_hist, l_global_hist = ([], [], [])
    l_hist = []
    if return_trajectories:
        l_hist.append(L.copy())
    E_prev = energy_bands(L, L0, w, lam=lam, use_js=use_js, local_coupling=local_coupling)
    l_global = global_logits_from_bands(L, w)
    U_prev = global_unsortedness(l_global)
    E_hist.append(E_prev)
    U_hist.append(U_prev)
    l_global_hist.append(l_global.copy())
    for _t in range(max_iter):
        L_old = L.copy()
        w_old = w.copy()
        L_prop = L.copy()
        w_prop = w.copy()
        for k in range(K):
            if frozen_flags[k]:
                continue
            neigh = neighbours[k]
            if len(neigh) == 0:
                continue
            w_neigh = w[neigh]
            Z = float(np.sum(w_neigh)) + 1e-12
            l_neigh_avg = (w_neigh[:, None] * L[neigh]).sum(axis=0) / Z
            stress = div(softmax(L[k]), softmax(l_neigh_avg))
            r = regime.lower()
            if r == 'plain':
                L_prop[k] = (1.0 - alpha) * L[k] + alpha * l_neigh_avg
            elif r == 'h1':
                if stress > tau:
                    w_prop[k] = mute_weight
                    L_prop[k] = l_neigh_avg
                else:
                    L_prop[k] = (1.0 - alpha) * L[k] + alpha * l_neigh_avg
            elif r == 'h2':
                if stress > tau:
                    w_prop[k] = mute_weight
                    L_prop[k] = l_neigh_avg
                else:
                    L_prop[k] = (1.0 - alpha) * L[k] + alpha * l_neigh_avg
                    w_prop[k] = np.clip(w_prop[k] * (1.0 + diff_amp), wmin, wmax)
            else:
                raise ValueError(f"Unknown regime: {regime} (use 'plain', 'H1', or 'H2')")
        w_prop = np.clip(w_prop, wmin, wmax)
        E_new = energy_bands(L_prop, L0, w_prop, lam=lam, use_js=use_js, local_coupling=local_coupling)
        accept = True
        if E_new > E_prev and T > 0.0:
            prob = float(np.exp(-(E_new - E_prev) / T))
            if rng.random() >= prob:
                accept = False
        if accept:
            L, w, E_prev = (L_prop, w_prop, E_new)
        else:
            L, w = (L_old, w_old)
        l_global = global_logits_from_bands(L, w)
        U_prev = global_unsortedness(l_global)
        E_hist.append(E_prev)
        U_hist.append(U_prev)
        l_global_hist.append(l_global.copy())
        if return_trajectories:
            l_hist.append(L.copy())
    l_final = global_logits_from_bands(L, w)
    y_hat = int(np.argmax(l_final))
    p_final = softmax(l_final)
    out = {'y_hat': y_hat, 'p_final': p_final, 'weights_final': w, 'band_logits_final': L}
    if return_trajectories:
        out['E_hist'] = np.asarray(E_hist, dtype=float)
        out['U_hist'] = np.asarray(U_hist, dtype=float)
        out['l_global_hist'] = np.stack(l_global_hist, axis=0)
        out['l_hist'] = np.stack(l_hist, axis=0)
    return out
from typing import Tuple, Optional, Union
import numpy as np

def apply_random_band_damage(band_logits0: Union[np.ndarray, list], damage_fraction: float, mode: str='mixed', *, rng: Optional[np.random.Generator]=None, noise_scale: Optional[float]=None, use_floor: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(SEED)
    if isinstance(band_logits0, list):
        L0 = np.stack([np.asarray(v, dtype=float) for v in band_logits0], axis=0)
    else:
        L0 = np.asarray(band_logits0, dtype=float)
    if L0.ndim != 2:
        raise ValueError(f'band_logits0 must be (K,C) or list-of-(C,), got {L0.shape}')
    K, C = L0.shape
    frac = float(np.clip(damage_fraction, 0.0, 1.0))
    n_damaged = int(np.floor(frac * K + 1e-09)) if use_floor else int(np.round(frac * K))
    idx_all = np.arange(K)
    rng.shuffle(idx_all)
    damaged_idx = idx_all[:n_damaged]
    damaged = L0.copy()
    frozen_flags = np.zeros(K, dtype=bool)
    if noise_scale is None:
        s = float(np.std(L0))
        noise_scale = s if s > 1e-08 else 1.0
    if n_damaged == 0:
        return (damaged, frozen_flags, damaged_idx)
    if mode == 'mixed':
        damaged[damaged_idx, :] = rng.normal(0.0, noise_scale, size=(n_damaged, C))
    elif mode == 'spiky':
        damaged[damaged_idx, :] = rng.normal(0.0, noise_scale, size=(n_damaged, C))
        n_spikes = max(1, int(np.ceil(0.3 * n_damaged)))
        spike_idx = damaged_idx[:n_spikes]
        damaged[spike_idx, :] = rng.normal(0.0, 3.0 * noise_scale, size=(n_spikes, C))
    elif mode == 'frozen':
        frozen_flags[damaged_idx] = True
        damaged[damaged_idx, :] = rng.normal(0.0, noise_scale, size=(n_damaged, C))
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mixed', 'spiky', or 'frozen'.")
    return (damaged, frozen_flags, damaged_idx)

def consensus_with_stress(band_logits0: List[np.ndarray], frozen_flags: np.ndarray, band_weights0: Optional[np.ndarray]=None, alpha: float=0.3, lam: float=1.0, T: float=0.01, tau: float=0.05, max_iter: int=30, use_js: bool=True, return_trajectories: bool=False, record_stress: bool=False, regime: str='plain', local_coupling: bool=True, mute_weight: float=0.0, diff_amp: float=0.25, weight_clip: tuple=(0.0, 5.0), rng: Optional[np.random.Generator]=None) -> Dict:
    if rng is None:
        rng = np.random.default_rng(42)
    K = len(band_logits0)
    if K == 0:
        raise ValueError('band_logits0 is empty')
    C = int(np.asarray(band_logits0[0]).shape[0])
    frozen_flags = np.asarray(frozen_flags, dtype=bool)
    if frozen_flags.shape != (K,):
        raise ValueError(f'frozen_flags must have shape ({K},), got {frozen_flags.shape}')
    band_logits = [np.asarray(l0, dtype=float).copy() for l0 in band_logits0]
    if band_weights0 is None:
        w0 = np.array([np.max(softmax(l0)) for l0 in band_logits0], dtype=float)
        weights = np.clip(w0, 0.0, 1.0)
    else:
        weights = np.asarray(band_weights0, dtype=float).copy()
        if weights.shape != (K,):
            raise ValueError(f'band_weights0 must have shape ({K},), got {weights.shape}')
    wmin, wmax = (float(weight_clip[0]), float(weight_clip[1]))
    if local_coupling:
        neighbours = []
        for k in range(K):
            neigh = []
            if k - 1 >= 0:
                neigh.append(k - 1)
            if k + 1 < K:
                neigh.append(k + 1)
            neighbours.append(neigh)
    else:
        neighbours = [[j for j in range(K) if j != k] for k in range(K)]
    div = js_div if use_js else kl_div
    E_hist, U_hist, l_global_hist = ([], [], [])
    l_hist = []
    stress_hist = []
    step_dg_hist = []
    if return_trajectories:
        l_hist.append(np.stack(band_logits, axis=0))
    E_prev = energy_bands(band_logits, band_logits0, weights, lam=lam, use_js=use_js, local_coupling=local_coupling)
    l_global = global_logits_from_bands(band_logits, weights)
    U_prev = global_unsortedness(l_global)
    E_hist.append(E_prev)
    U_hist.append(U_prev)
    l_global_hist.append(l_global.copy())
    if record_stress:
        s0 = np.zeros(K, dtype=float)
        for k in range(K):
            neigh = neighbours[k]
            if len(neigh) == 0:
                s0[k] = 0.0
                continue
            w_neigh = weights[neigh]
            Z = float(np.sum(w_neigh)) + 1e-12
            l_neigh_avg = np.zeros(C, dtype=float)
            for j, wj in zip(neigh, w_neigh):
                l_neigh_avg += float(wj) * band_logits[j]
            l_neigh_avg /= Z
            s0[k] = div(softmax(band_logits[k]), softmax(l_neigh_avg))
        stress_hist.append(s0)
    for _t in range(max_iter):
        band_logits_old = [l.copy() for l in band_logits]
        weights_old = weights.copy()
        band_logits_prop = [l.copy() for l in band_logits]
        weights_prop = weights.copy()
        for k in range(K):
            if frozen_flags[k]:
                continue
            neigh = neighbours[k]
            if len(neigh) == 0:
                continue
            w_neigh = weights[neigh]
            Z = float(np.sum(w_neigh)) + 1e-12
            l_neigh_avg = np.zeros(C, dtype=float)
            for j, wj in zip(neigh, w_neigh):
                l_neigh_avg += float(wj) * band_logits[j]
            l_neigh_avg /= Z
            stress = div(softmax(band_logits[k]), softmax(l_neigh_avg))
            if regime.lower() == 'plain':
                band_logits_prop[k] = (1.0 - alpha) * band_logits[k] + alpha * l_neigh_avg
            elif regime.lower() == 'h1':
                if stress > tau:
                    weights_prop[k] = mute_weight
                    band_logits_prop[k] = l_neigh_avg
                else:
                    band_logits_prop[k] = (1.0 - alpha) * band_logits[k] + alpha * l_neigh_avg
            elif regime.lower() == 'h2':
                if stress > tau:
                    weights_prop[k] = mute_weight
                    band_logits_prop[k] = l_neigh_avg
                else:
                    band_logits_prop[k] = (1.0 - alpha) * band_logits[k] + alpha * l_neigh_avg
                    weights_prop[k] = min(wmax, max(wmin, weights_prop[k] * (1.0 + diff_amp)))
            else:
                raise ValueError(f'Unknown regime: {regime}')
        weights_prop = np.clip(weights_prop, wmin, wmax)
        E_new = energy_bands(band_logits_prop, band_logits0, weights_prop, lam=lam, use_js=use_js, local_coupling=local_coupling)
        accept = True
        if E_new > E_prev and T > 0.0:
            prob = float(np.exp(-(E_new - E_prev) / T))
            if rng.random() >= prob:
                accept = False
        if accept:
            band_logits = band_logits_prop
            weights = weights_prop
            E_prev = E_new
        else:
            band_logits = band_logits_old
            weights = weights_old
        if record_stress or return_trajectories:
            L_new = np.stack(band_logits, axis=0)
            L_old = np.stack(band_logits_old, axis=0)
            step_dg_hist.append(np.linalg.norm(center_logits(L_new) - center_logits(L_old), axis=1))
        if record_stress:
            s = np.zeros(K, dtype=float)
            for k in range(K):
                neigh = neighbours[k]
                if len(neigh) == 0:
                    s[k] = 0.0
                    continue
                w_neigh = weights[neigh]
                Z = float(np.sum(w_neigh)) + 1e-12
                l_neigh_avg = np.zeros(C, dtype=float)
                for j, wj in zip(neigh, w_neigh):
                    l_neigh_avg += float(wj) * band_logits[j]
                l_neigh_avg /= Z
                s[k] = div(softmax(band_logits[k]), softmax(l_neigh_avg))
            stress_hist.append(s)
        l_global = global_logits_from_bands(band_logits, weights)
        U_prev = global_unsortedness(l_global)
        E_hist.append(E_prev)
        U_hist.append(U_prev)
        l_global_hist.append(l_global.copy())
        if return_trajectories:
            l_hist.append(np.stack(band_logits, axis=0))
    l_final = global_logits_from_bands(band_logits, weights)
    y_hat = int(np.argmax(l_final))
    p_final = softmax(l_final)
    out = {'y_hat': y_hat, 'p_final': p_final, 'band_logits_final': band_logits, 'weights_final': weights}
    if return_trajectories:
        out['E_hist'] = np.asarray(E_hist, dtype=float)
        out['U_hist'] = np.asarray(U_hist, dtype=float)
        out['l_global_hist'] = np.stack(l_global_hist, axis=0)
        out['l_hist'] = np.stack(l_hist, axis=0)
    if record_stress:
        out['stress_hist'] = np.stack(stress_hist, axis=0)
        out['step_dg_hist'] = np.stack(step_dg_hist, axis=0)
    return out
