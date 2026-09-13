from .core import center_logits, compute_dg_steps
from .auditing import correctness_auc, summarize_audit, plot_audit
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
SEED = 1
DATA_DIR = Path('.')
from .core import softmax, _as_KC, kl_div, js_div, energy_bands, global_logits_from_bands, global_unsortedness, compute_dg_index, morphogenetic_consensus_one, apply_random_band_damage, consensus_with_stress

def proba_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    P = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    P = P / np.sum(P, axis=-1, keepdims=True)
    return P

def ensure_proba(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 0.0, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True)
    return P

def macro_auc_ovr(y_true: np.ndarray, P: np.ndarray) -> float:
    P = ensure_proba(P)
    labels = np.arange(P.shape[1])
    try:
        return float(roc_auc_score(y_true, P, multi_class='ovr', average='macro', labels=labels))
    except ValueError as e:
        return np.nan

def evaluate_morphogenetic_system_bact5(full_logits_test: np.ndarray, band_logits_test: np.ndarray, y_test: np.ndarray, damage_fractions: List[float], *, regime: str='plain', damage_mode: str='mixed', alpha: float=0.3, lam: float=1.0, T: float=0.01, tau: float=0.543007, max_iter: int=30, naive_weighting: str='uniform', morpho_weighting: str='confidence', run_id: int=0, weight_clip: tuple=(0.0, 5.0)):
    N, K, C = band_logits_test.shape
    results = {'damage_fraction': [], 'acc_full': [], 'bal_acc_full': [], 'acc_naive': [], 'bal_acc_naive': [], 'acc_morpho': [], 'bal_acc_morpho': [], 'auc_full': [], 'auc_naive': [], 'auc_morpho': [], 'dg_index_mean': [], 'dg_index_std': [], 'regime': [], 'damage_mode': []}
    P_full = proba_from_logits(full_logits_test)
    y_full = np.argmax(P_full, axis=1)
    acc_full = accuracy_score(y_test, y_full)
    bal_acc_full = balanced_accuracy_score(y_test, y_full)
    auc_full = macro_auc_ovr(y_test, P_full)
    for frac in damage_fractions:
        y_pred_naive = []
        y_pred_morpho = []
        P_naive = np.zeros((N, C), dtype=float)
        P_morpho = np.zeros((N, C), dtype=float)
        dg_values = []
        for i in range(N):
            L0 = band_logits_test[i]
            seed = run_id * 1000000 + 10000 * int(round(frac * 100)) + i + SEED
            rng_i = np.random.default_rng(seed)
            damaged_L0, frozen_flags, _ = apply_random_band_damage(L0, damage_fraction=frac, mode=damage_mode, rng=rng_i)
            if naive_weighting == 'uniform':
                w_naive = np.ones(K, dtype=float)
            elif naive_weighting == 'confidence':
                w_naive = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)
            else:
                raise ValueError('Unknown naive_weighting')
            l_naive = global_logits_from_bands(damaged_L0, w_naive)
            P_naive[i] = proba_from_logits(l_naive)
            y_pred_naive.append(int(np.argmax(P_naive[i])))
            if morpho_weighting == 'uniform':
                w_morpho = np.ones(K, dtype=float)
            elif morpho_weighting == 'confidence':
                w_morpho = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)
            else:
                raise ValueError('Unknown morpho_weighting')
            res = morphogenetic_consensus_one(damaged_L0, frozen_flags=frozen_flags, band_weights0=w_morpho, regime=regime, weight_clip=weight_clip, alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter, use_js=True, return_trajectories=True, rng=rng_i)
            y_pred_morpho.append(int(res['y_hat']))
            P_morpho[i] = ensure_proba(np.asarray(res['p_final']).reshape(1, -1))[0]
            dg_values.append(compute_dg_index(res['l_hist']))
        y_pred_naive = np.asarray(y_pred_naive)
        y_pred_morpho = np.asarray(y_pred_morpho)
        results['damage_fraction'].append(float(frac))
        results['acc_full'].append(float(acc_full))
        results['bal_acc_full'].append(float(bal_acc_full))
        results['acc_naive'].append(float(accuracy_score(y_test, y_pred_naive)))
        results['bal_acc_naive'].append(float(balanced_accuracy_score(y_test, y_pred_naive)))
        results['acc_morpho'].append(float(accuracy_score(y_test, y_pred_morpho)))
        results['bal_acc_morpho'].append(float(balanced_accuracy_score(y_test, y_pred_morpho)))
        results['auc_full'].append(float(auc_full))
        results['auc_naive'].append(macro_auc_ovr(y_test, P_naive))
        results['auc_morpho'].append(macro_auc_ovr(y_test, P_morpho))
        results['dg_index_mean'].append(float(np.mean(dg_values)))
        results['dg_index_std'].append(float(np.std(dg_values)))
        results['regime'].append(regime)
        results['damage_mode'].append(damage_mode)
    return pd.DataFrame(results)

def add_eREDG(df: pd.DataFrame, *, dg_col: str='DG', dg_early_col: str='DG_early', out_redg_col: str='REDG', out_eredg_col: str='eREDG', eps: float=1e-12, gate_by: str='per_damage_mode_and_regime', gate_q: float=0.2, hard_delta: float=None):
    df = df.copy()
    assert dg_col in df.columns
    assert dg_early_col in df.columns
    df[out_redg_col] = df[dg_early_col] / (df[dg_col] + eps)
    gate_rows = []
    if hard_delta is not None:
        df['DG_delta'] = float(hard_delta)
        df['DG_active'] = (df[dg_col] > float(hard_delta)).astype(int)
        gate_rows.append({'group': 'global', 'DG_delta': float(hard_delta), 'gate_q': None, 'n': int(len(df))})
    elif gate_by == 'global':
        delta = float(np.quantile(df[dg_col].to_numpy(dtype=float), gate_q))
        df['DG_delta'] = delta
        df['DG_active'] = (df[dg_col] > delta).astype(int)
        gate_rows.append({'group': 'global', 'DG_delta': delta, 'gate_q': gate_q, 'n': int(len(df))})
    elif gate_by == 'per_damage_mode_and_regime':
        df['DG_delta'] = np.nan
        df['DG_active'] = 0
        for (dmg, reg), idx in df.groupby(['damage_mode', 'regime']).groups.items():
            sub = df.loc[idx, dg_col].to_numpy(dtype=float)
            delta = float(np.quantile(sub, gate_q))
            df.loc[idx, 'DG_delta'] = delta
            df.loc[idx, 'DG_active'] = (df.loc[idx, dg_col] > delta).astype(int)
            gate_rows.append({'group': f'{dmg}|{reg}', 'damage_mode': dmg, 'regime': reg, 'DG_delta': delta, 'gate_q': gate_q, 'n': int(len(idx))})
    else:
        raise ValueError("gate_by must be 'global' or 'per_damage_mode_and_regime'")
    df[out_eredg_col] = df[out_redg_col].where(df['DG_active'] == 1, np.nan)
    return (df, pd.DataFrame(gate_rows))

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

def kl_div(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def js_div(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)

def confidence_weights_from_logits(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    return np.clip(np.max(softmax(L), axis=1), 0.0, 0.999)

def initial_agent_stress(L: np.ndarray, *, weights=None, use_js: bool=True, local_coupling: bool=True) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    K, C = L.shape
    if weights is None:
        weights = confidence_weights_from_logits(L)
    weights = np.asarray(weights, dtype=float)
    div = js_div if use_js else kl_div
    stress = np.zeros(K, dtype=float)
    for k in range(K):
        if local_coupling:
            neigh = []
            if k - 1 >= 0:
                neigh.append(k - 1)
            if k + 1 < K:
                neigh.append(k + 1)
        else:
            neigh = [j for j in range(K) if j != k]
        if len(neigh) == 0:
            stress[k] = 0.0
            continue
        w_neigh = weights[neigh]
        Z = np.sum(w_neigh) + 1e-12
        l_neigh_avg = (w_neigh[:, None] * L[neigh]).sum(axis=0) / Z
        stress[k] = div(softmax(L[k]), softmax(l_neigh_avg))
    return stress

def clean_stress_distribution(band_logits: np.ndarray, *, weighting: str='confidence') -> np.ndarray:
    band_logits = np.asarray(band_logits, dtype=float)
    N, K, C = band_logits.shape
    values = []
    for i in range(N):
        L = band_logits[i]
        if weighting == 'confidence':
            w = confidence_weights_from_logits(L)
        elif weighting == 'uniform':
            w = np.ones(K, dtype=float)
        else:
            raise ValueError("weighting must be 'confidence' or 'uniform'")
        s = initial_agent_stress(L, weights=w)
        values.extend(s.tolist())
    return np.asarray(values, dtype=float)
from sklearn.metrics import roc_auc_score

def cliffs_delta(x, y):
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
    return (gt - lt) / (len(x) * len(y))


def compute_audit_table_bact5_corrected(*, damage_fraction: float=0.4, damage_modes=('mixed', 'spiky'), regimes=('plain', 'H1', 'H2'), k_early: int=3, run_id: int=0, tau: float=None, alpha: float=0.3, lam: float=1.0, T: float=0.01, max_iter: int=30, morpho_weighting: str='confidence', weight_clip=(0.0, 5.0)):
    if tau is None:
        if 'TAU_BACT_FINAL' in globals():
            tau = TAU_BACT_FINAL
        elif 'tau_bact_final' in globals():
            tau = tau_bact_final
        else:
            raise ValueError('Please provide tau, e.g. tau=TAU_BACT_FINAL')
    N, K, C = band_logits_test.shape
    rows = []
    mode_code = {'mixed': 0, 'spiky': 1, 'frozen': 2}
    for dmg in damage_modes:
        for reg in regimes:
            for i in range(N):
                L0 = band_logits_test[i]
                y_true = int(y_test[i])
                seed = SEED + run_id * 1000000 + 100000 * mode_code[dmg] + 10000 * int(round(damage_fraction * 100)) + i
                rng_i = np.random.default_rng(seed)
                damaged_L0, frozen_flags, damaged_idx = apply_random_band_damage(L0, damage_fraction=damage_fraction, mode=dmg, rng=rng_i)
                if morpho_weighting == 'uniform':
                    w0 = np.ones(K, dtype=float)
                else:
                    w0 = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)
                res = morphogenetic_consensus_one(damaged_L0, frozen_flags=frozen_flags, band_weights0=w0, regime=reg, weight_clip=weight_clip, alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter, use_js=True, return_trajectories=True, rng=rng_i)
                l_hist = res['l_hist']
                dg = float(compute_dg_index(l_hist))
                step_norms = compute_dg_steps(l_hist).sum(axis=1)
                k_use = min(k_early, step_norms.shape[0])
                dg_early = float(np.sum(step_norms[:k_use]))
                p_final = np.asarray(res['p_final'], dtype=float)
                p_final = p_final / (p_final.sum() + 1e-12)
                y_hat = int(res['y_hat'])
                correct = int(y_hat == y_true)
                p_sorted = np.sort(p_final)[::-1]
                max_conf = float(p_sorted[0])
                margin = float(p_sorted[0] - p_sorted[1])
                entropy = float(-np.sum(p_final * np.log(p_final + 1e-12)))
                rows.append({'i': i, 'run_id': run_id, 'damage_mode': dmg, 'regime': reg, 'damage_fraction': float(damage_fraction), 'y_true': y_true, 'y_hat': y_hat, 'correct': correct, 'DG': dg, 'DG_early': dg_early, 'REDG': dg_early / (dg + 1e-12), 'max_confidence': max_conf, 'margin': margin, 'entropy': entropy, 'damaged_idx': tuple(sorted(damaged_idx.tolist()))})
    return pd.DataFrame(rows)

def prepare(data_dir):
    global BANDS_BACT, BAND_HI, BAND_LO, C, DATA_DIR, Dict, K, LabelEncoder, List, LogisticRegression, NORMALIZE_ROWS, N_test, N_val, Optional, P_full_test, P_full_val, Path, Pipeline, RandomForestClassifier, SEED, SPLITS, SVC, StandardScaler, TAU_BACT_FINAL, TAU_BACT_Q, Tuple, X_clin18, X_clin19, X_test, X_train, X_val, _, accuracy_score, all_idx, auc_full_test, balanced_accuracy_score, band_indices, band_logits_bact_cal, band_logits_test, band_logits_val, band_pipes, classification_report, confusion_matrix, counts18, counts19, covered, edges, encoded, find_file, fp_mask, full_logits_test, full_logits_val, hi, i, idx, k, label_counts, le5, lo, load_bacteria_numpy, morphogenetic_consensus_one, normalize, np, original, os, pd, pipe_full, pipe_k, plt, random, rng, roc_auc_score, stress_bact_cal, train_test_split, wn, y_clin18, y_clin18_raw, y_clin19, y_clin19_raw, y_pred_full_test, y_pred_full_val, y_test, y_train, y_val
    random.seed(SEED)
    np.random.seed(SEED)
    import os
    import random
    from pathlib import Path
    from typing import List, Dict, Tuple, Optional
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    SEED = 1
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 400, 'axes.grid': True, 'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})
    DATA_DIR = Path(data_dir)
    SPLITS = {'train': ('X_reference.npy', 'y_reference.npy'), 'val': ('X_finetune.npy', 'y_finetune.npy'), 'test': ('X_test.npy', 'y_test.npy'), 'clinical2018': ('X_2018clinical.npy', 'y_2018clinical.npy'), 'clinical2019': ('X_2019clinical.npy', 'y_2019clinical.npy')}

    def find_file(root: Path, filename: str) -> Path:
        matches = list(root.rglob(filename))
        if len(matches) == 0:
            raise FileNotFoundError(f'Could not find {filename} under {root}')
        if len(matches) > 1:
            pass
        return matches[0]

    def load_bacteria_numpy(split: str, root: Path=DATA_DIR, mmap: bool=False):
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split}. Choose from {list(SPLITS)}')
        X_file, y_file = SPLITS[split]
        X_path = find_file(root, X_file)
        y_path = find_file(root, y_file)
        wn_path = find_file(root, 'wavenumbers.npy')
        mmap_mode = 'r' if mmap else None
        X = np.load(X_path, mmap_mode=mmap_mode)
        y = np.load(y_path).astype(int)
        wn = np.load(wn_path).astype(float)
        return (X, y, wn)
    X_clin18, y_clin18_raw, wn = load_bacteria_numpy('clinical2018', DATA_DIR, mmap=True)
    X_clin19, y_clin19_raw, _ = load_bacteria_numpy('clinical2019', DATA_DIR, mmap=True)

    def label_counts(y, name):
        tab = pd.Series(y).value_counts().sort_index()
        return tab
    counts18 = label_counts(y_clin18_raw, 'clinical2018')
    counts19 = label_counts(y_clin19_raw, 'clinical2019')
    le5 = LabelEncoder()
    le5.fit(np.concatenate([y_clin18_raw, y_clin19_raw]))
    y_clin18 = le5.transform(y_clin18_raw)
    y_clin19 = le5.transform(y_clin19_raw)
    for original, encoded in zip(le5.classes_, le5.transform(le5.classes_)):
        pass
    X_clin18 = np.asarray(X_clin18, dtype=np.float64)
    X_clin19 = np.asarray(X_clin19, dtype=np.float64)
    NORMALIZE_ROWS = True
    if NORMALIZE_ROWS:
        X_clin18 = normalize(X_clin18, axis=1)
        X_clin19 = normalize(X_clin19, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_clin18, y_clin18, test_size=0.2, stratify=y_clin18, random_state=SEED)
    X_test = X_clin19
    y_test = y_clin19
    BAND_LO = 400.0
    BAND_HI = min(1800.0, float(wn.max()))
    edges = np.linspace(BAND_LO, BAND_HI, 8)
    BANDS_BACT = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]
    band_indices = []
    for k, (lo, hi) in enumerate(BANDS_BACT):
        if k < len(BANDS_BACT) - 1:
            idx = np.where((wn >= lo) & (wn < hi))[0]
        else:
            idx = np.where((wn >= lo) & (wn <= hi))[0]
        if idx.size == 0:
            raise ValueError(f'No points in band {lo:.1f}-{hi:.1f} cm^-1.')
        band_indices.append(idx)
    all_idx = np.concatenate(band_indices)
    if np.unique(all_idx).size != all_idx.size:
        raise ValueError('Band definitions overlap.')
    fp_mask = (wn >= BAND_LO) & (wn <= BAND_HI)
    covered = np.zeros_like(fp_mask, dtype=bool)
    for idx in band_indices:
        covered[idx] = True
    for k, (lo, hi) in enumerate(BANDS_BACT):
        pass
    pipe_full = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=3000, solver='lbfgs', n_jobs=-1, random_state=SEED))])
    pipe_full.fit(X_train, y_train)
    full_logits_val = pipe_full.decision_function(X_val)
    full_logits_test = pipe_full.decision_function(X_test)
    P_full_val = pipe_full.predict_proba(X_val)
    P_full_test = pipe_full.predict_proba(X_test)
    y_pred_full_val = np.argmax(P_full_val, axis=1)
    y_pred_full_test = np.argmax(P_full_test, axis=1)
    auc_full_test = roc_auc_score(y_test, P_full_test, multi_class='ovr', average='macro', labels=np.arange(P_full_test.shape[1]))
    K = len(band_indices)
    C = len(np.unique(y_train))
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]
    band_logits_val = np.zeros((N_val, K, C), dtype=float)
    band_logits_test = np.zeros((N_test, K, C), dtype=float)
    band_pipes = []
    for k, idx in enumerate(band_indices):
        pipe_k = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=3000, solver='lbfgs', n_jobs=-1, random_state=SEED))])
        pipe_k.fit(X_train[:, idx], y_train)
        band_logits_val[:, k, :] = pipe_k.decision_function(X_val[:, idx])
        band_logits_test[:, k, :] = pipe_k.decision_function(X_test[:, idx])
        band_pipes.append(pipe_k)
    TAU_BACT_Q = 0.975
    band_logits_bact_cal = band_logits_val
    stress_bact_cal = clean_stress_distribution(band_logits_bact_cal, weighting='confidence')
    TAU_BACT_FINAL = float(np.quantile(stress_bact_cal, TAU_BACT_Q))
    morphogenetic_consensus_one = consensus_with_stress
    return auc_full_test

def baselines():
    global P_rf, P_svm, baseline_rows, clf_rf, df_baselines_bact5, pipe_svm, y_pred_rf, y_pred_svm
    baseline_rows = []
    baseline_rows.append({'model': 'Logistic regression', 'split': 'external_test', 'accuracy': accuracy_score(y_test, y_pred_full_test), 'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_full_test), 'macro_auc_ovr': roc_auc_score(y_test, P_full_test, multi_class='ovr', average='macro')})
    pipe_svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', probability=True, random_state=SEED))])
    pipe_svm.fit(X_train, y_train)
    P_svm = pipe_svm.predict_proba(X_test)
    y_pred_svm = np.argmax(P_svm, axis=1)
    baseline_rows.append({'model': 'Linear SVM', 'split': 'external_test', 'accuracy': accuracy_score(y_test, y_pred_svm), 'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_svm), 'macro_auc_ovr': roc_auc_score(y_test, P_svm, multi_class='ovr', average='macro')})
    clf_rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    P_rf = clf_rf.predict_proba(X_test)
    y_pred_rf = np.argmax(P_rf, axis=1)
    baseline_rows.append({'model': 'Random forest', 'split': 'external_test', 'accuracy': accuracy_score(y_test, y_pred_rf), 'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_rf), 'macro_auc_ovr': roc_auc_score(y_test, P_rf, multi_class='ovr', average='macro')})
    df_baselines_bact5 = pd.DataFrame(baseline_rows)
    return df_baselines_bact5

def plot_bact5_accuracy_dgl(df):
    damage_modes = ['mixed', 'spiky']
    regimes = ['plain', 'H1', 'H2']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    for col, dmg in enumerate(damage_modes):
        dfd = df[df['damage_mode'] == dmg].sort_values('damage_fraction')
        ax = axes[0, col]
        df0 = dfd[dfd['regime'] == 'plain']
        ax.plot(df0['damage_fraction'], df0['acc_full'], '--o', label='Full model')
        ax.plot(df0['damage_fraction'], df0['acc_naive'], '-o', label='Naive band avg')
        for reg in regimes:
            dfr = dfd[dfd['regime'] == reg]
            ax.plot(dfr['damage_fraction'], dfr['acc_morpho'], '-o', label=f'Negotiated ({reg})')
        ax.set_title(f'Accuracy vs damage: {dmg}')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax = axes[1, col]
        for reg in regimes:
            dfr = dfd[dfd['regime'] == reg]
            ax.plot(dfr['damage_fraction'], dfr['dg_norm'], '-o', label=f'DGL ({reg})')
        ax.set_title(f'DGL vs damage: {dmg}')
        ax.set_xlabel('Damage fraction')
        ax.set_ylabel('DGL = DG / DG$_0$')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('External 5-class bacteria/yeast Raman benchmark', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig('Bacteria5_Accuracy_DGL_composite.png', dpi=600, bbox_inches='tight')
    plt.show()

def robustness():
    global all_rows, damage_fractions, damage_modes, df_res_bact5, df_tmp, dfg, dmg, mask, ref, ref_val, reg, regimes
    damage_fractions = [0.0, 0.2, 0.4, 0.6]
    damage_modes = ['mixed', 'spiky']
    regimes = ['plain', 'H1', 'H2']
    all_rows = []
    for dmg in damage_modes:
        for reg in regimes:
            df_tmp = evaluate_morphogenetic_system_bact5(full_logits_test=full_logits_test, band_logits_test=band_logits_test, y_test=y_test, damage_fractions=damage_fractions, regime=reg, damage_mode=dmg, alpha=0.3, lam=1.0, T=0.01, tau=0.543007, max_iter=30, naive_weighting='uniform', morpho_weighting='confidence', run_id=0)
            all_rows.append(df_tmp)
    df_res_bact5 = pd.concat(all_rows, ignore_index=True)
    df_res_bact5['dg_norm'] = np.nan
    for (reg, dmg), dfg in df_res_bact5.groupby(['regime', 'damage_mode']):
        ref = dfg.loc[dfg['damage_fraction'] == 0.0, 'dg_index_mean']
        if len(ref) != 1:
            raise ValueError(f'Expected one zero-damage row for {reg}, {dmg}')
        ref_val = float(ref.iloc[0])
        mask = (df_res_bact5['regime'] == reg) & (df_res_bact5['damage_mode'] == dmg)
        df_res_bact5.loc[mask, 'dg_norm'] = df_res_bact5.loc[mask, 'dg_index_mean'] / ref_val
    plot_bact5_accuracy_dgl(df_res_bact5)
    return df_res_bact5

def audit(n_runs=10):
    global df_eredg_runs_bact5, df_eredg_samples_bact5
    samples = []
    for run_id in range(n_runs):
        table = compute_audit_table_bact5_corrected(
            damage_fraction=0.4, damage_modes=('mixed', 'spiky'),
            regimes=('plain', 'H1', 'H2'), run_id=run_id, k_early=3,
            alpha=0.3, lam=1.0, T=0.01, tau=TAU_BACT_FINAL, max_iter=30,
            weight_clip=(0.0, 5.0), morpho_weighting='confidence',
        )
        table, _ = add_eREDG(table, gate_q=0.2)
        samples.append(table)
    df_eredg_samples_bact5 = pd.concat(samples, ignore_index=True)
    df_eredg_runs_bact5 = summarize_audit(df_eredg_samples_bact5, 'run_id')
    plot_audit(df_eredg_runs_bact5, df_eredg_samples_bact5, 'Bacteria')
    return df_eredg_runs_bact5, df_eredg_samples_bact5

def spectra_plot():
    global X_fp, cls, fp_mask, mask, mean_spec, std_spec, wn_fp
    plt.figure(figsize=(10, 5))
    fp_mask = (wn >= BAND_LO) & (wn <= BAND_HI)
    wn_fp = wn[fp_mask]
    X_fp = X_clin18[:, fp_mask]
    for cls in np.unique(y_clin18):
        mask = y_clin18 == cls
        mean_spec = X_fp[mask].mean(axis=0)
        std_spec = X_fp[mask].std(axis=0)
        plt.plot(wn_fp, mean_spec, label=f'class {cls}')
        plt.fill_between(wn_fp, mean_spec - std_spec, mean_spec + std_spec, alpha=0.15)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Normalized intensity (a.u.)')
    plt.title('Mean Raman spectra per bacteria/yeast class, clinical2018')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Bacteria5_Mean_Raman_per_class.png', dpi=600, bbox_inches='tight')
    plt.show()
