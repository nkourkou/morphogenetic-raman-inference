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
apply_band_damage = apply_random_band_damage
from typing import List, Dict

def evaluate_morphogenetic_system(full_logits_test: np.ndarray, band_logits_test: np.ndarray, y_test: np.ndarray, damage_fractions: List[float], alpha: float=0.3, lam: float=1.0, T: float=0.01, tau: float=0.05, max_iter: int=30, damage_mode: str='mixed', naive_weighting: str='uniform', morpho_weighting: str='confidence', regime: str='plain', run_id: int=0, weight_clip: tuple=(0.0, 5.0), auc_multi_class: str='ovr') -> Dict:
    N, K, C = band_logits_test.shape
    results = {'damage_fraction': [], 'acc_full': [], 'acc_naive': [], 'acc_morpho': [], 'auc_full': [], 'auc_naive': [], 'auc_morpho': [], 'dg_index_mean': [], 'dg_index_std': []}
    y_pred_full_clean = np.argmax(full_logits_test, axis=1)
    acc_full_clean = accuracy_score(y_test, y_pred_full_clean)
    P_full = softmax(full_logits_test)
    try:
        auc_full_clean = roc_auc_score(y_test, P_full, multi_class=auc_multi_class, average='macro')
    except ValueError:
        auc_full_clean = np.nan
    y_pred_naive_clean = []
    P_naive_clean = np.zeros((N, C), dtype=float)
    for i in range(N):
        L = band_logits_test[i]
        if naive_weighting == 'uniform':
            w_naive = np.ones(K, dtype=float)
        elif naive_weighting == 'confidence':
            w_naive = np.clip(np.max(softmax(L), axis=1), 0.0, 0.999)
        else:
            raise ValueError(f'Unknown naive_weighting: {naive_weighting}')
        l_avg = global_logits_from_bands(L, w_naive)
        y_pred_naive_clean.append(int(np.argmax(l_avg)))
        P_naive_clean[i] = softmax(l_avg)
    acc_naive_clean = accuracy_score(y_test, y_pred_naive_clean)
    try:
        auc_naive_clean = roc_auc_score(y_test, P_naive_clean, multi_class=auc_multi_class, average='macro')
    except ValueError:
        auc_naive_clean = np.nan
    acc_full = acc_full_clean
    auc_full = auc_full_clean
    for frac in damage_fractions:
        y_pred_naive, y_pred_morpho, dg_indices = ([], [], [])
        P_naive = np.zeros((N, C), dtype=float)
        P_morpho = np.zeros((N, C), dtype=float)
        for i in range(N):
            L0 = band_logits_test[i]
            seed = run_id * 1000000 + 10000 * int(round(frac * 100)) + i + SEED
            rng = np.random.default_rng(seed)
            damaged_L0, frozen_flags, _ = apply_random_band_damage(L0, damage_fraction=frac, mode=damage_mode, rng=rng)
            if naive_weighting == 'uniform':
                w_naive = np.ones(K, dtype=float)
            elif naive_weighting == 'confidence':
                w_naive = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)
            else:
                raise ValueError(f'Unknown naive_weighting: {naive_weighting}')
            l_avg = global_logits_from_bands(damaged_L0, w_naive)
            y_pred_naive.append(int(np.argmax(l_avg)))
            P_naive[i] = softmax(l_avg)
            if morpho_weighting == 'uniform':
                w_morpho = np.ones(K, dtype=float)
            elif morpho_weighting == 'confidence':
                w_morpho = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)
            else:
                raise ValueError(f'Unknown morpho_weighting: {morpho_weighting}')
            res = morphogenetic_consensus_one(damaged_L0, frozen_flags=frozen_flags, band_weights0=w_morpho, regime=regime, weight_clip=weight_clip, alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter, use_js=True, return_trajectories=True, rng=rng)
            y_pred_morpho.append(int(res['y_hat']))
            P_morpho[i] = res['p_final']
            dg_indices.append(compute_dg_index(res['l_hist']))
        acc_naive = accuracy_score(y_test, y_pred_naive)
        acc_morpho = accuracy_score(y_test, y_pred_morpho)
        try:
            auc_naive = roc_auc_score(y_test, P_naive, multi_class=auc_multi_class, average='macro')
        except ValueError:
            auc_naive = np.nan
        try:
            auc_morpho = roc_auc_score(y_test, P_morpho, multi_class=auc_multi_class, average='macro')
        except ValueError:
            auc_morpho = np.nan
        results['damage_fraction'].append(float(frac))
        results['acc_full'].append(float(acc_full))
        results['acc_naive'].append(float(acc_naive))
        results['acc_morpho'].append(float(acc_morpho))
        results['auc_full'].append(float(auc_full))
        results['auc_naive'].append(float(auc_naive))
        results['auc_morpho'].append(float(auc_morpho))
        results['dg_index_mean'].append(float(np.mean(dg_indices)))
        results['dg_index_std'].append(float(np.std(dg_indices)))
    return results

def stress_history_from_l_hist(l_hist: np.ndarray, use_js: bool=True, local_coupling: bool=True) -> np.ndarray:
    l_hist = np.asarray(l_hist, dtype=float)
    Tp1, K, C = l_hist.shape
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
    stress = np.zeros((Tp1, K), dtype=float)
    for t in range(Tp1):
        L = l_hist[t]
        for k in range(K):
            neigh = neighbours[k]
            if len(neigh) == 0:
                stress[t, k] = 0.0
                continue
            l_neigh_avg = np.mean(L[neigh, :], axis=0)
            stress[t, k] = div(softmax(L[k, :]), softmax(l_neigh_avg))
    return stress

def compute_dg_and_stress_table(*, damage_fraction: float=0.4, damage_modes=('mixed', 'spiky'), regimes=('plain', 'H1', 'H2'), run_id: int=0, alpha: float=0.3, lam: float=1.0, T: float=0.01, tau: float=0.05, max_iter: int=30, weight_clip: tuple=(0.0, 5.0), morpho_weighting: str='confidence', k_early: int=3):
    N, K, C = band_logits_test.shape
    rows = []
    for dmg in damage_modes:
        for reg in regimes:
            for i in range(N):
                L0_list = [band_logits_test[i, k, :] for k in range(K)]
                y_true = int(y_test[i])
                seed = run_id * 1000000 + 10000 * int(round(damage_fraction * 100)) + i
                rng = np.random.default_rng(seed)
                Ld_list, frozen_flags, damaged_idx = apply_band_damage(L0_list, damage_fraction=damage_fraction, mode=dmg, rng=rng)
                if morpho_weighting == 'uniform':
                    w0 = np.ones(K, dtype=float)
                else:
                    conf = np.array([np.max(softmax(l)) for l in Ld_list], dtype=float)
                    w0 = np.clip(conf, 0.0, 0.999)
                res = morphogenetic_consensus_one(Ld_list, frozen_flags=frozen_flags, band_weights0=w0, regime=reg, weight_clip=weight_clip, alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter, use_js=True, return_trajectories=True)
                l_hist = res['l_hist']
                dg = float(compute_dg_index(l_hist))
                step_norms = compute_dg_steps(l_hist).sum(axis=1)
                k_use = min(k_early, step_norms.shape[0])
                dg_early = float(np.sum(step_norms[:k_use]))
                stress = stress_history_from_l_hist(l_hist, use_js=True, local_coupling=True)
                stress0 = stress[0, :]
                stress0_mean = float(np.mean(stress0))
                stress0_max = float(np.max(stress0))
                y_hat = int(res['y_hat'])
                correct = int(y_hat == y_true)
                rows.append({'i': i, 'damage_mode': dmg, 'regime': reg, 'damage_fraction': float(damage_fraction), 'y_true': y_true, 'y_hat': y_hat, 'correct': correct, 'DG': dg, 'DG_early': dg_early, 'stress0_mean': stress0_mean, 'stress0_max': stress0_max, 'damaged_idx': tuple(sorted(damaged_idx.tolist()))})
    return pd.DataFrame(rows)
from typing import List, Dict, Tuple, Optional

def add_eREDG(df: pd.DataFrame, *, dg_col: str='DG', dg_early_col: str='DG_early', out_redg_col: str='REDG', out_eredg_col: str='eREDG', eps: float=1e-12, gate_by: str='per_damage_mode_and_regime', gate_q: float=0.2, hard_delta: float=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    assert dg_col in df.columns, f'Missing {dg_col}'
    assert dg_early_col in df.columns, f'Missing {dg_early_col}'
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
    gate_table = pd.DataFrame(gate_rows)
    return (df, gate_table)

def prepare(data_dir):
    global BANDS, C, Dict, EXCLUDED_LABELS, EXPECTED_POINTS, K, LabelEncoder, List, LogisticRegression, M, N, N_test, Optional, Pipeline, StandardScaler, StratifiedKFold, Tuple, X_test, X_train, acc_full, accuracy_score, all_idx, arr, arr_norm, band_indices, band_logits_test, band_pipes, base_dir, base_label, base_label3, c, confusion_matrix, covered, df0, df3, df_all, disease_related_set, excluded_base, file, fp_mask, full_logits_test, gap_points, glob, hi, i, idx, k, kind, kind_col, lab, label, label_col, label_map_3class, labels_raw, labels_raw0, le3, lo, mask3, mask_no_serum, melanoma_set, mpl, non_spec_cols, normal_skin_set, normalize, np, os, path_parts, pattern, pd, pipe_full, pipe_k, plt, roc_auc_score, row, rows, sort_idx, spec_col_names, spec_cols, spec_cols_sorted, spectra, train_test_split, unique_idx, v, wn, y, y3, y3_str, y_pred_full, y_test, y_train
    random.seed(SEED)
    np.random.seed(SEED)
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import List, Tuple, Optional, Dict
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 400, 'axes.grid': True})
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13, 'figure.titlesize': 17})
    import glob
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    base_dir = str(data_dir)
    EXCLUDED_LABELS = {'serum', 'DMEM'}
    EXPECTED_POINTS = 2090
    pattern = os.path.join(base_dir, 'dataset_i', '**', '*.csv')
    rows = []
    wn = np.linspace(100, 4278, EXPECTED_POINTS)
    spec_col_names = [str(int(round(v))) for v in wn]
    for file in glob.glob(pattern, recursive=True):
        path_parts = file.split(os.path.sep)
        label = path_parts[-2]
        kind = os.path.splitext(path_parts[-1])[0]
        if label in EXCLUDED_LABELS:
            continue
        arr = pd.read_csv(file, header=None).values
        if arr.shape[1] != EXPECTED_POINTS:
            raise ValueError(f'{label}/{kind}: {arr.shape[1]} points found, expected {EXPECTED_POINTS}')
        arr_norm = normalize(arr, axis=1)
        for i in range(arr_norm.shape[0]):
            row = {'Label': label, 'Kind': kind}
            row.update(dict(zip(spec_col_names, arr_norm[i])))
            rows.append(row)
    df_all = pd.DataFrame(rows)
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    assert 'Label' in df_all.columns, "Expected column 'Label' in df_all."
    assert 'Kind' in df_all.columns, "Expected column 'Kind' in df_all."
    label_col = 'Label'
    kind_col = 'Kind'
    labels_raw = df_all[label_col].astype(str).to_numpy()
    mask_no_serum = np.array([not lab.endswith('-S') for lab in labels_raw], dtype=bool)
    df0 = df_all.loc[mask_no_serum].reset_index(drop=True)
    labels_raw0 = df0[label_col].astype(str).to_numpy()
    base_label = np.array([lab[:-2] if lab.endswith('-S') else lab for lab in labels_raw0], dtype=object)
    melanoma_set = {'A', 'G'}
    normal_skin_set = {'HPM', 'HF'}
    disease_related_set = {'ZAM'}
    excluded_base = {'DMEM'}
    y3_str = np.full(shape=base_label.shape, fill_value='exclude', dtype=object)
    y3_str[np.isin(base_label, list(melanoma_set))] = 'melanoma'
    y3_str[np.isin(base_label, list(normal_skin_set))] = 'normal_skin'
    y3_str[np.isin(base_label, list(disease_related_set))] = 'disease_related'
    mask3 = (y3_str != 'exclude') & ~np.isin(base_label, list(excluded_base))
    df3 = df0.loc[mask3].reset_index(drop=True)
    y3_str = y3_str[mask3]
    base_label3 = base_label[mask3]
    non_spec_cols = {label_col, kind_col}
    spec_cols = [c for c in df3.columns if c not in non_spec_cols and np.issubdtype(df3[c].dtype, np.number)]
    spectra = df3[spec_cols].to_numpy(dtype=np.float64)
    N, M = spectra.shape
    wn = np.array([float(c) for c in spec_cols], dtype=float)
    sort_idx = np.argsort(wn)
    wn = wn[sort_idx]
    spectra = spectra[:, sort_idx]
    spec_cols_sorted = [spec_cols[i] for i in sort_idx]
    le3 = LabelEncoder()
    y3 = le3.fit_transform(y3_str)
    label_map_3class = dict(zip(le3.classes_, range(len(le3.classes_))))
    BANDS = [(400, 700), (700, 900), (900, 1100), (1100, 1300), (1300, 1500), (1500, 1700), (1700, 2000)]
    band_indices = []
    for k, (lo, hi) in enumerate(BANDS):
        idx = np.where((wn >= lo) & (wn < hi))[0] if k < len(BANDS) - 1 else np.where((wn >= lo) & (wn <= hi))[0]
        if idx.size == 0:
            raise ValueError(f'No points in band {lo}-{hi} cm^-1. Check wn axis and band definitions.')
        band_indices.append(idx)
    if wn.min() > BANDS[0][0] or wn.max() < BANDS[-1][1]:
        pass
    all_idx = np.concatenate(band_indices)
    unique_idx = np.unique(all_idx)
    if unique_idx.size != all_idx.size:
        raise ValueError('Band definitions overlap (some wn points assigned to >1 band).')
    fp_mask = (wn >= BANDS[0][0]) & (wn <= BANDS[-1][1])
    covered = np.zeros_like(fp_mask, dtype=bool)
    for idx in band_indices:
        covered[idx] = True
    gap_points = np.where(fp_mask & ~covered)[0]
    for k, (lo, hi) in enumerate(BANDS):
        pass
    y = y3
    X_train, X_test, y_train, y_test = train_test_split(spectra, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe_full = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=SEED))])
    pipe_full.fit(X_train, y_train)
    full_logits_test = pipe_full.decision_function(X_test)
    K = len(band_indices)
    N_test = X_test.shape[0]
    C = full_logits_test.shape[1]
    band_logits_test = np.zeros((N_test, K, C), dtype=float)
    band_pipes = []
    for k, idx in enumerate(band_indices):
        pipe_k = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=SEED))])
        pipe_k.fit(X_train[:, idx], y_train)
        band_logits_test[:, k, :] = pipe_k.decision_function(X_test[:, idx])
        band_pipes.append(pipe_k)
    y_pred_full = np.argmax(full_logits_test, axis=1)
    acc_full = accuracy_score(y_test, y_pred_full)
    return acc_full
from scipy.stats import spearmanr

def _ensure_dg_norm(df_res_all: pd.DataFrame) -> pd.DataFrame:
    df = df_res_all.copy()
    if 'dg_norm' in df.columns:
        return df
    df['dg_norm'] = np.nan
    for (reg, dmg), dfg in df.groupby(['regime', 'damage_mode']):
        ref = dfg.loc[dfg['damage_fraction'] == 0.0, 'dg_index_mean']
        if len(ref) != 1:
            raise ValueError(f'Expected exactly one zero-damage DG for {reg}, {dmg}')
        ref_val = float(ref.iloc[0])
        df.loc[(df['regime'] == reg) & (df['damage_mode'] == dmg), 'dg_norm'] = dfg['dg_index_mean'] / ref_val
    return df

def plot_composite_accuracy_dgl(df_res_all: pd.DataFrame, *, highlight_regime: str='H1', use_auc: bool=False, normalize_dg: bool=True, show_dg_std: bool=True, title: str='Accuracy–Load decoupling under structured damage'):
    df = _ensure_dg_norm(df_res_all) if normalize_dg else df_res_all.copy()
    damage_modes = ['mixed', 'spiky']
    regimes = ['plain', 'H1', 'H2']
    oracle_label = 'Full model (oracle)'
    naive_label = 'Naive band avg'
    perf_y_oracle = 'auc_full' if use_auc else 'acc_full'
    perf_y_naive = 'auc_naive' if use_auc else 'acc_naive'
    perf_y_morpho = 'auc_morpho' if use_auc else 'acc_morpho'
    dg_y = 'dg_norm' if normalize_dg else 'dg_index_mean'
    dg_yerr = 'dg_index_std'
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    axA, axB = (axes[0, 0], axes[0, 1])
    axC, axD = (axes[1, 0], axes[1, 1])

    def draw_perf(ax, dmg):
        dfd = df[df['damage_mode'] == dmg].copy()
        dfd = dfd.sort_values('damage_fraction')
        df0 = dfd[dfd['regime'] == 'plain']
        ax.plot(df0['damage_fraction'], df0[perf_y_oracle], '--', label=oracle_label)
        ax.plot(df0['damage_fraction'], df0[perf_y_naive], '-o', label=naive_label)
        for reg in regimes:
            dfr = dfd[dfd['regime'] == reg]
            lw = 3.0 if reg.lower() == highlight_regime.lower() else 2.0
            mk = 'o' if reg.lower() == highlight_regime.lower() else 's'
            ax.plot(dfr['damage_fraction'], dfr[perf_y_morpho], marker=mk, linewidth=lw, label=f'Morphogenetic ({reg})')
        ax.set_title(f"{('AUC' if use_auc else 'Accuracy')} vs Damage ({dmg})")
        ax.set_ylabel('Macro AUC (OVR)' if use_auc else 'Accuracy')
        ax.grid(True, alpha=0.3)

    def draw_dg(ax, dmg):
        dfd = df[df['damage_mode'] == dmg].copy()
        dfd = dfd.sort_values('damage_fraction')
        for reg in regimes:
            dfr = dfd[dfd['regime'] == reg]
            lw = 3.0 if reg.lower() == highlight_regime.lower() else 2.0
            mk = 'o' if reg.lower() == highlight_regime.lower() else 's'
            ax.plot(dfr['damage_fraction'], dfr[dg_y], marker=mk, linewidth=lw, label=f'DGL ({reg})' if normalize_dg else f'DG ({reg})')
            if show_dg_std and (not normalize_dg) and (dg_yerr in dfr.columns):
                ax.fill_between(dfr['damage_fraction'].to_numpy(), (dfr[dg_y] - dfr[dg_yerr]).to_numpy(), (dfr[dg_y] + dfr[dg_yerr]).to_numpy(), alpha=0.15)
        ax.set_title(f"{('DGL (normalized DG)' if normalize_dg else 'DG')} vs Damage ({dmg})")
        ax.set_xlabel('Damage fraction (bands corrupted)')
        ax.set_ylabel('DGL / DGL₀' if normalize_dg else 'Mean DG')
        ax.grid(True, alpha=0.3)
    draw_perf(axA, 'mixed')
    draw_perf(axB, 'spiky')
    draw_dg(axC, 'mixed')
    draw_dg(axD, 'spiky')
    axA.text(0.01, 0.98, 'A', transform=axA.transAxes, va='top', ha='left', fontsize=14, fontweight='bold')
    axB.text(0.01, 0.98, 'B', transform=axB.transAxes, va='top', ha='left', fontsize=14, fontweight='bold')
    axC.text(0.01, 0.98, 'C', transform=axC.transAxes, va='top', ha='left', fontsize=14, fontweight='bold')
    axD.text(0.01, 0.98, 'D', transform=axD.transAxes, va='top', ha='left', fontsize=14, fontweight='bold')
    handles_labels = []
    for ax in [axA, axB, axC, axD]:
        handles_labels.extend(list(zip(*ax.get_legend_handles_labels())))
    seen = set()
    handles, labels = ([], [])
    for h, l in handles_labels:
        if l not in seen:
            seen.add(l)
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.show()

def robustness():
    global all_rows, damage_fractions, damage_modes, df0, df_reg, df_res_all, df_res_norm, df_spearman, dfd, dfg, dfr, dmg, fname, out_dir, p_norm, p_raw, ref, ref_val, reg, regimes, results_reg, rho_norm, rho_raw, rows
    damage_fractions = [0.0, 0.2, 0.4, 0.6]
    regimes = ['plain', 'H1', 'H2']
    damage_modes = ['mixed', 'spiky']
    all_rows = []
    for dmg in damage_modes:
        for reg in regimes:
            results_reg = evaluate_morphogenetic_system(full_logits_test=full_logits_test, band_logits_test=band_logits_test, y_test=y_test, regime=reg, damage_fractions=damage_fractions, alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30, damage_mode=dmg, naive_weighting='uniform', morpho_weighting='confidence')
            df_reg = pd.DataFrame(results_reg)
            df_reg['regime'] = reg
            df_reg['damage_mode'] = dmg
            all_rows.append(df_reg)
    df_res_all = pd.concat(all_rows, ignore_index=True)
    for dmg in damage_modes:
        plt.figure(figsize=(9, 5))
        dfd = df_res_all[df_res_all['damage_mode'] == dmg].copy()
        df0 = dfd[dfd['regime'] == 'plain'].copy()
        plt.plot(df0['damage_fraction'], df0['acc_full'], '-o', label='Full model (oracle)')
        plt.plot(df0['damage_fraction'], df0['acc_naive'], '-o', label='Naive band avg')
        for reg in regimes:
            dfr = dfd[dfd['regime'] == reg]
            plt.plot(dfr['damage_fraction'], dfr['acc_morpho'], '-o', label=f'Morphogenetic ({reg})')
        plt.xlabel('Damage fraction (bands corrupted)')
        plt.ylabel('Accuracy')
        plt.title(f'Robustness under band perturbations – damage_mode={dmg}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = f'Robustness-{dmg}.png'
        out_dir = '.'
        plt.savefig(os.path.join(out_dir, fname), dpi=600, bbox_inches='tight')
        plt.show()
    for dmg in damage_modes:
        plt.figure(figsize=(9, 5))
        dfd = df_res_all[df_res_all['damage_mode'] == dmg].copy()
        for reg in regimes:
            dfr = dfd[dfd['regime'] == reg]
            plt.plot(dfr['damage_fraction'], dfr['dg_index_mean'], '-o', label=f'DG ({reg})')
        plt.xlabel('Damage fraction (bands corrupted)')
        plt.ylabel('Mean Decision Geometry (DG)')
        plt.title(f'Decision Geometry vs perturbation – damage_mode={dmg}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = f'DG_vs_pertrubation-{dmg}.png'
        out_dir = '.'
        plt.savefig(os.path.join(out_dir, fname), dpi=600, bbox_inches='tight')
        plt.show()
    for dmg in damage_modes:
        plt.figure(figsize=(9, 5))
        dfd = df_res_all[df_res_all['damage_mode'] == dmg].copy()
        df0 = dfd[dfd['regime'] == 'plain'].copy()
        plt.plot(df0['damage_fraction'], df0['auc_full'], '-o', label='Full model (oracle)')
        plt.plot(df0['damage_fraction'], df0['auc_naive'], '-o', label='Naive band avg')
        for reg in regimes:
            dfr = dfd[dfd['regime'] == reg]
            plt.plot(dfr['damage_fraction'], dfr['auc_morpho'], '-o', label=f'Morphogenetic ({reg})')
        plt.xlabel('Damage fraction (bands corrupted)')
        plt.ylabel('Macro AUC (OVR)')
        plt.title(f'AUC under band perturbations – damage_mode={dmg}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        fname = f'auc_vs_damage_mode-{dmg}.png'
        out_dir = '.'
        plt.savefig(os.path.join(out_dir, fname), dpi=600, bbox_inches='tight')
        plt.show()
    df_res_norm = df_res_all.copy()
    df_res_norm['dg_norm'] = np.nan
    for (reg, dmg), dfg in df_res_norm.groupby(['regime', 'damage_mode']):
        ref = dfg.loc[dfg['damage_fraction'] == 0.0, 'dg_index_mean']
        if len(ref) != 1:
            raise ValueError(f'Expected exactly one zero-damage DG for {reg}, {dmg}')
        ref_val = float(ref.iloc[0])
        df_res_norm.loc[(df_res_norm['regime'] == reg) & (df_res_norm['damage_mode'] == dmg), 'dg_norm'] = dfg['dg_index_mean'] / ref_val
    rows = []
    for dmg in df_res_norm['damage_mode'].unique():
        for reg in df_res_norm['regime'].unique():
            dfg = df_res_norm[(df_res_norm['damage_mode'] == dmg) & (df_res_norm['regime'] == reg)].sort_values('damage_fraction')
            rho_raw, p_raw = spearmanr(dfg['damage_fraction'], dfg['dg_index_mean'])
            rho_norm, p_norm = spearmanr(dfg['damage_fraction'], dfg['dg_norm'])
            rows.append({'damage_mode': dmg, 'regime': reg, 'rho_DG_vs_damage': rho_raw, 'p_raw': p_raw, 'rho_DGnorm_vs_damage': rho_norm, 'p_norm': p_norm})
    df_spearman = pd.DataFrame(rows)
    df_spearman.sort_values(['damage_mode', 'regime'])
    plot_composite_accuracy_dgl(df_res_all, highlight_regime='H1', use_auc=False, normalize_dg=True, show_dg_std=False, title='Accuracy–Load decoupling under structured damage (3-class Raman)')
    return df_res_all

def stable_softmax(z):
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[None, :]
        squeeze = True
    else:
        squeeze = False
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    p = e / (np.sum(e, axis=1, keepdims=True) + 1e-12)
    return p[0] if squeeze else p

def make_equalwidth_bands(wn, k, lo=400.0, hi=2000.0):
    edges = np.linspace(lo, hi, k + 1)
    bands = [(float(edges[i]), float(edges[i + 1])) for i in range(k)]
    band_indices = []
    for i, (blo, भी) in enumerate(bands):
        if i < k - 1:
            idx = np.where((wn >= blo) & (wn < भी))[0]
        else:
            idx = np.where((wn >= blo) & (wn <= भी))[0]
        if idx.size == 0:
            raise ValueError(f'No points found in band {i}: {blo:.1f}-{भी:.1f} cm^-1')
        band_indices.append(idx)
    return (bands, band_indices)

def fit_band_models_for_partition(X_train, X_test, y_train, band_indices, seed=42):
    K = len(band_indices)
    n_classes = len(np.unique(y_train))
    band_logits_test = np.zeros((X_test.shape[0], K, n_classes), dtype=float)
    band_pipes = []
    for k, idx in enumerate(band_indices):
        pipe_k = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=seed))])
        pipe_k.fit(X_train[:, idx], y_train)
        band_logits_test[:, k, :] = pipe_k.decision_function(X_test[:, idx])
        band_pipes.append(pipe_k)
    return (band_logits_test, band_pipes)

def evaluate_partition(band_logits_test, y_test, *, partition_label, damage_fraction=0.0, damage_mode=None, alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30, diff_amp=0.25, mute_weight=0.0, weight_clip=(0.0, 5.0), seed_base=0):
    rows = []
    N, K, C = band_logits_test.shape
    naive_logits = []
    for i in range(N):
        L0 = [band_logits_test[i, k, :] for k in range(K)]
        if damage_fraction > 0 and damage_mode is not None:
            rng = np.random.default_rng(seed_base + i)
            Ld, frozen_flags, damaged_idx = apply_band_damage(L0, damage_fraction=damage_fraction, mode=damage_mode, rng=rng)
            L_use = np.stack(Ld, axis=0)
        else:
            L_use = np.stack(L0, axis=0)
        g = np.mean(L_use, axis=0)
        naive_logits.append(g)
    naive_logits = np.vstack(naive_logits)
    naive_probs = stable_softmax(naive_logits)
    naive_pred = np.argmax(naive_probs, axis=1)
    rows.append({'partition': partition_label, 'damage_mode': 'clean' if damage_mode is None else damage_mode, 'damage_fraction': float(damage_fraction), 'method': 'naive_band', 'acc': accuracy_score(y_test, naive_pred), 'auc': roc_auc_score(y_test, naive_probs, multi_class='ovr', average='macro'), 'DG_mean': np.nan})
    for regime in ['plain', 'H1', 'H2']:
        pred_probs = []
        pred_labels = []
        dg_vals = []
        for i in range(N):
            L0 = [band_logits_test[i, k, :] for k in range(K)]
            if damage_fraction > 0 and damage_mode is not None:
                rng_damage = np.random.default_rng(seed_base + 10000 + i)
                Ld, frozen_flags, damaged_idx = apply_band_damage(L0, damage_fraction=damage_fraction, mode=damage_mode, rng=rng_damage)
            else:
                Ld = [x.copy() for x in L0]
                frozen_flags = np.zeros(K, dtype=bool)
            w0 = np.array([np.max(stable_softmax(v)) for v in Ld], dtype=float)
            w0 = np.clip(w0, 0.0, 0.999)
            res = morphogenetic_consensus_one(band_logits0=Ld, frozen_flags=frozen_flags, band_weights0=w0, alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter, use_js=True, return_trajectories=True, regime=regime, local_coupling=True, mute_weight=mute_weight, diff_amp=diff_amp, weight_clip=weight_clip, rng=np.random.default_rng(seed_base + 100000 + i))
            pred_probs.append(res['p_final'])
            pred_labels.append(res['y_hat'])
            dg_vals.append(float(compute_dg_index(res['l_hist'])))
        pred_probs = np.vstack(pred_probs)
        pred_labels = np.array(pred_labels)
        rows.append({'partition': partition_label, 'damage_mode': 'clean' if damage_mode is None else damage_mode, 'damage_fraction': float(damage_fraction), 'method': regime, 'acc': accuracy_score(y_test, pred_labels), 'auc': roc_auc_score(y_test, pred_probs, multi_class='ovr', average='macro'), 'DG_mean': float(np.mean(dg_vals))})
    return pd.DataFrame(rows)

def partition():
    global K_use, band_defs, band_indices_use, band_logits_test_use, band_pipes_use, bands_use, df_clean, df_mixed04, df_part_sens, hi, j, lo, morphogenetic_consensus_one, results
    morphogenetic_consensus_one = consensus_with_stress
    results = []
    band_defs = {}
    for K_use in [5, 7, 9]:
        bands_use, band_indices_use = make_equalwidth_bands(wn, K_use, lo=400.0, hi=2000.0)
        band_defs[K_use] = bands_use
        for j, (lo, hi) in enumerate(bands_use):
            pass
        band_logits_test_use, band_pipes_use = fit_band_models_for_partition(X_train, X_test, y_train, band_indices_use, seed=SEED)
        df_clean = evaluate_partition(band_logits_test_use, y_test, partition_label=f'K={K_use}', damage_fraction=0.0, damage_mode=None, alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30)
        results.append(df_clean)
        df_mixed04 = evaluate_partition(band_logits_test_use, y_test, partition_label=f'K={K_use}', damage_fraction=0.4, damage_mode='mixed', alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30, seed_base=K_use * 1000)
        results.append(df_mixed04)
    df_part_sens = pd.concat(results, ignore_index=True)
    return df_part_sens
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def macro_auc_from_probs(y_true, probs):
    return roc_auc_score(y_true, probs, multi_class='ovr', average='macro')

def baselines():
    from scipy.special import softmax
    global N_SPLITS, P_lr, P_rf, P_svm, SEEDS, TEST_SIZE, X, X_test, X_train, band_logits_test, band_models, clf_rf, df_rep, df_rep_band, frozen_flags, global_logits_naive, i, idx, k, l0, m, morphogenetic_consensus_one, out, pipe_band, pipe_lr, pipe_svm, pred_labels, pred_probs, probs_naive, regime_name, rows, rows_band, seed, split_id, sss, summary, summary_band, test_idx, train_idx, v, w0, y, y_test, y_train, yhat_lr, yhat_naive, yhat_rf, yhat_svm
    morphogenetic_consensus_one = consensus_with_stress
    X = spectra
    y = y3
    N_SPLITS = 10
    TEST_SIZE = 0.2
    SEEDS = list(range(100, 100 + N_SPLITS))
    rows = []
    for split_id, seed in enumerate(SEEDS, start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = (X[train_idx], X[test_idx])
        y_train, y_test = (y[train_idx], y[test_idx])
        pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=seed))])
        pipe_svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', probability=True, random_state=seed))])
        clf_rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1)
        pipe_lr.fit(X_train, y_train)
        pipe_svm.fit(X_train, y_train)
        clf_rf.fit(X_train, y_train)
        P_lr = pipe_lr.predict_proba(X_test)
        P_svm = pipe_svm.predict_proba(X_test)
        P_rf = clf_rf.predict_proba(X_test)
        yhat_lr = np.argmax(P_lr, axis=1)
        yhat_svm = np.argmax(P_svm, axis=1)
        yhat_rf = np.argmax(P_rf, axis=1)
        rows.append({'split': split_id, 'method': 'logistic_full', 'acc': accuracy_score(y_test, yhat_lr), 'auc': roc_auc_score(y_test, P_lr, multi_class='ovr', average='macro')})
        rows.append({'split': split_id, 'method': 'svm_full', 'acc': accuracy_score(y_test, yhat_svm), 'auc': roc_auc_score(y_test, P_svm, multi_class='ovr', average='macro')})
        rows.append({'split': split_id, 'method': 'rf_full', 'acc': accuracy_score(y_test, yhat_rf), 'auc': roc_auc_score(y_test, P_rf, multi_class='ovr', average='macro')})
    df_rep = pd.DataFrame(rows)
    summary = df_rep.groupby('method')[['acc', 'auc']].agg(['mean', 'std']).round(3)
    X = spectra
    y = y3
    N_SPLITS = 10
    TEST_SIZE = 0.2
    SEEDS = list(range(100, 100 + N_SPLITS))
    rows_band = []
    for split_id, seed in enumerate(SEEDS, start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = (X[train_idx], X[test_idx])
        y_train, y_test = (y[train_idx], y[test_idx])
        band_models = []
        for idx in band_indices:
            pipe_band = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed))])
            pipe_band.fit(X_train[:, idx], y_train)
            band_models.append(pipe_band)
        band_logits_test = np.stack([m.decision_function(X_test[:, idx]) for m, idx in zip(band_models, band_indices)], axis=1)
        global_logits_naive = band_logits_test.mean(axis=1)
        probs_naive = softmax(global_logits_naive, axis=1)
        yhat_naive = np.argmax(probs_naive, axis=1)
        rows_band.append({'split': split_id, 'method': 'naive_band', 'acc': accuracy_score(y_test, yhat_naive), 'auc': macro_auc_from_probs(y_test, probs_naive)})
        for regime_name in ['plain', 'H1', 'H2']:
            pred_probs = []
            pred_labels = []
            for i in range(len(y_test)):
                l0 = [band_logits_test[i, k, :] for k in range(band_logits_test.shape[1])]
                w0 = np.array([np.max(softmax(v)) for v in l0], dtype=float)
                frozen_flags = np.zeros(len(l0), dtype=bool)
                out = morphogenetic_consensus_one(band_logits0=l0, frozen_flags=frozen_flags, band_weights0=w0, alpha=0.35, lam=0.7, T=0.01, tau=0.12, max_iter=30, use_js=True, return_trajectories=False, record_stress=False, regime=regime_name, local_coupling=True, mute_weight=0.0, diff_amp=0.25, weight_clip=(0.0, 5.0), rng=np.random.default_rng(seed + i))
                pred_probs.append(out['p_final'])
                pred_labels.append(out['y_hat'])
            pred_probs = np.vstack(pred_probs)
            pred_labels = np.array(pred_labels)
            rows_band.append({'split': split_id, 'method': regime_name, 'acc': accuracy_score(y_test, pred_labels), 'auc': macro_auc_from_probs(y_test, pred_probs)})
    df_rep_band = pd.DataFrame(rows_band)
    summary_band = df_rep_band.groupby('method')[['acc', 'auc']].agg(['mean', 'std']).round(3)
    return (summary, summary_band)
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
    return float((gt - lt) / (len(x) * len(y)))

def audit(n_splits=10):
    global band_logits_test, y_test, morphogenetic_consensus_one
    global df_eredg_runs, df_eredg_samples_all
    saved_logits, saved_y = band_logits_test, y_test
    saved_consensus = morphogenetic_consensus_one
    morphogenetic_consensus_one = consensus_with_stress
    samples = []
    try:
        for split_id, seed in enumerate(range(100, 100 + n_splits), start=1):
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, test_idx = next(splitter.split(spectra, y3))
            models = []
            for idx in band_indices:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed)),
                ])
                model.fit(spectra[train_idx][:, idx], y3[train_idx])
                models.append(model)
            band_logits_test = np.stack([
                model.decision_function(spectra[test_idx][:, idx])
                for model, idx in zip(models, band_indices)
            ], axis=1)
            y_test = y3[test_idx]
            table = compute_dg_and_stress_table(
                damage_fraction=0.4, damage_modes=('mixed',),
                regimes=('plain', 'H1', 'H2'), run_id=seed, k_early=3,
                alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30,
                weight_clip=(0.0, 5.0), morpho_weighting='confidence',
            )
            table, _ = add_eREDG(table, gate_q=0.2)
            table['split'] = split_id
            table['seed'] = seed
            table['sample_index'] = test_idx[table['i'].to_numpy(dtype=int)]
            samples.append(table)
    finally:
        band_logits_test, y_test = saved_logits, saved_y
        morphogenetic_consensus_one = saved_consensus
    df_eredg_samples_all = pd.concat(samples, ignore_index=True)
    df_eredg_runs = summarize_audit(df_eredg_samples_all, 'split')
    plot_audit(df_eredg_runs, df_eredg_samples_all, 'Melanoma')
    return df_eredg_runs, df_eredg_samples_all

def morphogenetic_consensus_one_audit(band_logits0, frozen_flags=None, band_weights0=None, alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30, use_js=True, regime='H1', local_coupling=True, mute_weight=0.0, diff_amp=0.25, weight_clip=(0.0, 5.0), rng=None):
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
        w = np.clip(np.max(softmax(L0), axis=1), 0.0, 1.0)
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
    l_hist = [L.copy()]
    weight_hist = [w.copy()]
    E_hist = []
    l_global_hist = []
    stress_hist = []
    mute_hist = []
    step_dg_hist = []
    E_prev = energy_bands(L, L0, w, lam=lam, use_js=use_js, local_coupling=local_coupling)
    l_global = global_logits_from_bands(L, w)
    E_hist.append(E_prev)
    l_global_hist.append(l_global.copy())
    for t in range(max_iter):
        L_old = L.copy()
        w_old = w.copy()
        L_prop = L.copy()
        w_prop = w.copy()
        stress_t = np.zeros(K, dtype=float)
        mute_t = np.zeros(K, dtype=bool)
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
            stress_t[k] = stress
            r = regime.lower()
            if r == 'plain':
                L_prop[k] = (1.0 - alpha) * L[k] + alpha * l_neigh_avg
            elif r == 'h1':
                if stress > tau:
                    w_prop[k] = mute_weight
                    L_prop[k] = l_neigh_avg
                    mute_t[k] = True
                else:
                    L_prop[k] = (1.0 - alpha) * L[k] + alpha * l_neigh_avg
            elif r == 'h2':
                if stress > tau:
                    w_prop[k] = mute_weight
                    L_prop[k] = l_neigh_avg
                    mute_t[k] = True
                else:
                    L_prop[k] = (1.0 - alpha) * L[k] + alpha * l_neigh_avg
                    w_prop[k] = np.clip(w_prop[k] * (1.0 + diff_amp), wmin, wmax)
            else:
                raise ValueError("regime must be 'plain', 'H1', or 'H2'")
        w_prop = np.clip(w_prop, wmin, wmax)
        E_new = energy_bands(L_prop, L0, w_prop, lam=lam, use_js=use_js, local_coupling=local_coupling)
        accept = True
        if E_new > E_prev and T > 0.0:
            prob = float(np.exp(-(E_new - E_prev) / T))
            if rng.random() >= prob:
                accept = False
        if accept:
            L_new = L_prop
            w_new = w_prop
            E_prev = E_new
        else:
            L_new = L_old
            w_new = w_old
            mute_t[:] = False
        step_dg = np.linalg.norm(center_logits(L_new) - center_logits(L), axis=1)
        L = L_new
        w = w_new
        l_global = global_logits_from_bands(L, w)
        stress_hist.append(stress_t.copy())
        mute_hist.append(mute_t.copy())
        step_dg_hist.append(step_dg.copy())
        l_hist.append(L.copy())
        weight_hist.append(w.copy())
        E_hist.append(E_prev)
        l_global_hist.append(l_global.copy())
    l_hist = np.asarray(l_hist, dtype=float)
    weight_hist = np.asarray(weight_hist, dtype=float)
    stress_hist = np.asarray(stress_hist, dtype=float)
    mute_hist = np.asarray(mute_hist, dtype=bool)
    step_dg_hist = np.asarray(step_dg_hist, dtype=float)
    l_final = global_logits_from_bands(L, w)
    p_final = softmax(l_final)
    y_hat = int(np.argmax(l_final))
    return {'y_hat': y_hat, 'p_final': p_final, 'weights_final': w.copy(), 'band_logits_final': L.copy(), 'l_hist': l_hist, 'weight_hist': weight_hist, 'stress_hist': stress_hist, 'mute_hist': mute_hist, 'step_dg_hist': step_dg_hist, 'band_dg': step_dg_hist.sum(axis=0), 'E_hist': np.asarray(E_hist, dtype=float), 'l_global_hist': np.asarray(l_global_hist, dtype=float)}

def run_h1_conflict_audit(damage_mode='mixed', damage_fraction=0.4, run_id=0, alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30, weight_clip=(0.0, 5.0), morpho_weighting='confidence'):
    N, K, C = band_logits_test.shape
    rows = []
    audit_runs = []
    for i in range(N):
        L0 = band_logits_test[i].copy()
        seed = run_id * 1000000 + 10000 * int(round(damage_fraction * 100)) + i + SEED
        rng_i = np.random.default_rng(seed)
        damaged_L0, frozen_flags, damaged_idx = apply_band_damage(L0, damage_fraction=damage_fraction, mode=damage_mode, rng=rng_i)
        if morpho_weighting == 'uniform':
            w0 = np.ones(K, dtype=float)
        elif morpho_weighting == 'confidence':
            w0 = np.clip(np.max(softmax(damaged_L0), axis=1), 0.0, 0.999)
        else:
            raise ValueError("morpho_weighting must be 'uniform' or 'confidence'")
        res = morphogenetic_consensus_one_audit(damaged_L0, frozen_flags=frozen_flags, band_weights0=w0, regime='H1', weight_clip=weight_clip, alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter, use_js=True, rng=rng_i)
        y_hat = int(res['y_hat'])
        y_true_i = int(y_test[i])
        correct_i = int(y_hat == y_true_i)
        audit_runs.append(res)
        for k, (lo, hi) in enumerate(BANDS):
            rows.append({'sample_i': i, 'damage_mode': damage_mode, 'damage_fraction': float(damage_fraction), 'band': k, 'band_label': f'B{k + 1}', 'band_range': f'{lo}-{hi}', 'y_true': y_true_i, 'y_hat': y_hat, 'correct': correct_i, 'was_damaged': int(k in set(damaged_idx.tolist())), 'mute_frequency': float(np.mean(res['mute_hist'][:, k])), 'stress_mean': float(np.mean(res['stress_hist'][:, k])), 'stress_auc': float(np.sum(res['stress_hist'][:, k])), 'stress_max': float(np.max(res['stress_hist'][:, k])), 'band_dg': float(res['band_dg'][k]), 'final_weight': float(res['weights_final'][k])})
    df_audit = pd.DataFrame(rows)
    df_sample = df_audit.groupby('sample_i').agg(y_true=('y_true', 'first'), y_hat=('y_hat', 'first'), correct=('correct', 'first'), total_mute=('mute_frequency', 'sum'), total_stress_auc=('stress_auc', 'sum'), total_band_dg=('band_dg', 'sum')).reset_index()
    acc = accuracy_score(df_sample['y_true'], df_sample['y_hat'])
    return (df_audit, df_sample, audit_runs)

def plot_h1_conflict_localization(df_audit, damage_mode='mixed', damage_fraction=0.4, save_path='melanoma_H1_conflict_localization.png'):
    K = len(BANDS)
    x = np.arange(K)
    band_labels = [f'B{k + 1}\n{lo}-{hi}' for k, (lo, hi) in enumerate(BANDS)]
    dfp = df_audit[(df_audit['damage_mode'] == damage_mode) & (df_audit['damage_fraction'] == damage_fraction)].copy()
    if dfp.empty:
        raise ValueError('No rows found for selected damage_mode and damage_fraction.')
    df_band_all = dfp.groupby(['band', 'band_label', 'band_range']).agg(mute_frequency=('mute_frequency', 'mean'), stress_auc=('stress_auc', 'mean'), stress_mean=('stress_mean', 'mean'), stress_max=('stress_max', 'mean'), band_dg=('band_dg', 'mean'), damaged_frequency=('was_damaged', 'mean')).reset_index().sort_values('band')
    df_band_corr = dfp.groupby(['band', 'correct']).agg(stress_auc=('stress_auc', 'mean'), band_dg=('band_dg', 'mean'), mute_frequency=('mute_frequency', 'mean')).reset_index()

    def get_metric_by_correct(metric, correct_value):
        tmp = df_band_corr[df_band_corr['correct'] == correct_value].sort_values('band')
        arr = np.full(K, np.nan)
        for _, r in tmp.iterrows():
            arr[int(r['band'])] = float(r[metric])
        return arr
    stress_correct = get_metric_by_correct('stress_auc', 1)
    stress_wrong = get_metric_by_correct('stress_auc', 0)
    dg_correct = get_metric_by_correct('band_dg', 1)
    dg_wrong = get_metric_by_correct('band_dg', 0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axA, axB, axC, axD = axes.ravel()
    classes = np.unique(y_test)
    for cls in classes:
        mask = y_test == cls
        mean_spec = np.mean(X_test[mask], axis=0)
        try:
            label = le3.inverse_transform([cls])[0]
        except Exception:
            label = str(cls)
        axA.plot(wn, mean_spec, linewidth=1.8, label=label)
    ymin, ymax = axA.get_ylim()
    for k, (lo, hi) in enumerate(BANDS):
        axA.axvspan(lo, hi, alpha=0.08)
        axA.text((lo + hi) / 2, ymax, f'B{k + 1}', ha='center', va='top', fontsize=10)
    axA.set_xlim(BANDS[0][0], BANDS[-1][1])
    axA.set_title('[A] Mean melanoma Raman spectra and macro-bands')
    axA.set_xlabel('Raman shift (cm$^{-1}$)')
    axA.set_ylabel('Normalized intensity')
    axA.legend(frameon=False)
    axB.bar(x, 100.0 * df_band_all['mute_frequency'].to_numpy())
    axB.set_xticks(x)
    axB.set_xticklabels(band_labels)
    axB.set_ylabel('Muted update frequency (%)')
    axB.set_title('[B] H1 stress-gated muting by macro-band')
    axB2 = axB.twinx()
    axB2.plot(x, 100.0 * df_band_all['damaged_frequency'].to_numpy(), 'o--', linewidth=1.5, label='Damaged frequency')
    axB2.set_ylabel('Damaged frequency (%)')
    axB2.set_ylim(0, 100)
    width = 0.38
    axC.bar(x - width / 2, stress_correct, width, label='Correct')
    axC.bar(x + width / 2, stress_wrong, width, label='Incorrect')
    axC.set_xticks(x)
    axC.set_xticklabels(band_labels)
    axC.set_ylabel('Mean cumulative stress')
    axC.set_title('[C] Conflict load by prediction outcome')
    axC.legend(frameon=False)
    axD.bar(x - width / 2, dg_correct, width, label='Correct')
    axD.bar(x + width / 2, dg_wrong, width, label='Incorrect')
    axD.set_xticks(x)
    axD.set_xticklabels(band_labels)
    axD.set_ylabel('Mean band-level DG contribution')
    axD.set_title('[D] Band contribution to Decision Geometry')
    axD.legend(frameon=False)
    fig.suptitle(f'H1 conflict localization in melanoma Raman classification ({damage_mode}, damage fraction={damage_fraction})', y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    return (df_band_all, df_band_corr)

def conflict():
    global K, band_labels, delta_stress, df_band_all_mixed, df_band_corr, df_band_corr_mixed, df_h1_audit_mixed, df_h1_sample_mixed, h1_runs_mixed, hi, k, lo, rank, ratio_stress, stress_correct, stress_incorrect
    df_h1_audit_mixed, df_h1_sample_mixed, h1_runs_mixed = run_h1_conflict_audit(damage_mode='mixed', damage_fraction=0.4, run_id=0, alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30, weight_clip=(0.0, 5.0), morpho_weighting='confidence')
    df_band_all_mixed, df_band_corr_mixed = plot_h1_conflict_localization(df_h1_audit_mixed, damage_mode='mixed', damage_fraction=0.4, save_path='melanoma_H1_conflict_localization_mixed_04.png')
    df_band_corr = df_h1_audit_mixed.groupby(['band', 'correct']).agg(stress_auc=('stress_auc', 'mean'), band_dg=('band_dg', 'mean'), mute_frequency=('mute_frequency', 'mean')).reset_index()
    K = len(BANDS)
    stress_correct = df_band_corr[df_band_corr['correct'] == 1].sort_values('band')['stress_auc'].to_numpy(dtype=float)
    stress_incorrect = df_band_corr[df_band_corr['correct'] == 0].sort_values('band')['stress_auc'].to_numpy(dtype=float)
    delta_stress = stress_incorrect - stress_correct
    ratio_stress = stress_incorrect / (stress_correct + 1e-12)
    band_labels = [f'B{k + 1}\n{lo}-{hi}' for k, (lo, hi) in enumerate(BANDS)]
    rank = pd.DataFrame({'band': [f'B{k + 1}' for k in range(K)], 'range_cm-1': [f'{lo}-{hi}' for lo, hi in BANDS], 'stress_correct': stress_correct, 'stress_incorrect': stress_incorrect, 'delta_stress_incorrect_minus_correct': delta_stress, 'stress_ratio_incorrect_over_correct': ratio_stress}).sort_values('delta_stress_incorrect_minus_correct', ascending=False)
    plt.figure(figsize=(8, 4.2))
    plt.bar(np.arange(K), delta_stress)
    plt.axhline(0, linewidth=1)
    plt.xticks(np.arange(K), band_labels)
    plt.ylabel('Excess cumulative stress\nIncorrect − correct')
    plt.title('Error-associated spectral conflict by Raman macro-band')
    plt.tight_layout()
    plt.savefig('melanoma_error_associated_conflict_delta_stress.png', dpi=600, bbox_inches='tight')
    plt.show()
    return rank
from collections import defaultdict

def cliffs_delta_master(x, y):
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

def summarize_one_audit_condition(df_e, *, dataset, repeat_id, damage_fraction, damage_mode, regime):
    d = df_e[(df_e['damage_mode'] == damage_mode) & (df_e['regime'] == regime)].copy()
    if len(d) == 0:
        return None
    accuracy = float(d['correct'].mean())
    dg_mean = float(d['DG'].mean())
    dg_sd = float(d['DG'].std(ddof=1))
    dg_median = float(d['DG'].median())
    dg_q1 = float(d['DG'].quantile(0.25))
    dg_q3 = float(d['DG'].quantile(0.75))
    active = d[np.isfinite(d['eREDG'])].copy()
    n_total = len(d)
    n_active = len(active)
    active_fraction = n_active / n_total if n_total else np.nan
    if n_active == 0:
        return {'dataset': dataset, 'repeat_id': repeat_id, 'damage_fraction': damage_fraction, 'damage_mode': damage_mode, 'regime': regime, 'accuracy': accuracy, 'DG_mean': dg_mean, 'DG_sd': dg_sd, 'DG_median': dg_median, 'DG_q1': dg_q1, 'DG_q3': dg_q3, 'n_total': n_total, 'n_active': 0, 'active_fraction': 0.0, 'eREDG_median': np.nan, 'eREDG_q1': np.nan, 'eREDG_q3': np.nan, 'eREDG_IQR': np.nan, 'eREDG_correct_median': np.nan, 'eREDG_incorrect_median': np.nan, 'AUC_eREDG_raw': np.nan, 'cliffs_delta_raw': np.nan, 'abs_cliffs_delta': np.nan, 'n_correct_active': 0, 'n_incorrect_active': 0}
    x = active['eREDG'].to_numpy(dtype=float)
    ycorr = active['correct'].to_numpy(dtype=int)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    correct_vals = active.loc[active['correct'] == 1, 'eREDG'].to_numpy(dtype=float)
    incorrect_vals = active.loc[active['correct'] == 0, 'eREDG'].to_numpy(dtype=float)
    if len(np.unique(ycorr)) >= 2:
        auc_raw = correctness_auc(ycorr, x)
    else:
        auc_raw = np.nan
    delta = cliffs_delta_master(correct_vals, incorrect_vals)
    return {'dataset': dataset, 'repeat_id': repeat_id, 'damage_fraction': float(damage_fraction), 'damage_mode': damage_mode, 'regime': regime, 'accuracy': accuracy, 'DG_mean': dg_mean, 'DG_sd': dg_sd, 'DG_median': dg_median, 'DG_q1': dg_q1, 'DG_q3': dg_q3, 'n_total': int(n_total), 'n_active': int(n_active), 'active_fraction': float(active_fraction), 'eREDG_median': float(med), 'eREDG_q1': float(q1), 'eREDG_q3': float(q3), 'eREDG_IQR': float(q3 - q1), 'eREDG_correct_median': float(np.median(correct_vals)) if len(correct_vals) else np.nan, 'eREDG_incorrect_median': float(np.median(incorrect_vals)) if len(incorrect_vals) else np.nan, 'AUC_eREDG_raw': auc_raw, 'cliffs_delta_raw': delta, 'abs_cliffs_delta': abs(delta) if np.isfinite(delta) else np.nan, 'n_correct_active': int(len(correct_vals)), 'n_incorrect_active': int(len(incorrect_vals))}

def make_master_summary(df_runs, pool_all, pool_correct, pool_incorrect):
    group_cols = ['dataset', 'damage_mode', 'damage_fraction', 'regime']
    rows = []
    for key, g in df_runs.groupby(group_cols):
        dataset, damage_mode, damage_fraction, regime = key
        pool_key = (damage_fraction, damage_mode, regime)
        all_arrays = pool_all.get(pool_key, [])
        corr_arrays = pool_correct.get(pool_key, [])
        wrong_arrays = pool_incorrect.get(pool_key, [])
        all_vals = np.concatenate([x for x in all_arrays if len(x)]) if any((len(x) for x in all_arrays)) else np.array([])
        corr_vals = np.concatenate([x for x in corr_arrays if len(x)]) if any((len(x) for x in corr_arrays)) else np.array([])
        wrong_vals = np.concatenate([x for x in wrong_arrays if len(x)]) if any((len(x) for x in wrong_arrays)) else np.array([])
        if len(all_vals):
            q1, med, q3 = np.quantile(all_vals, [0.25, 0.5, 0.75])
        else:
            q1 = med = q3 = np.nan
        rows.append({'dataset': dataset, 'damage_mode': damage_mode, 'damage_fraction': damage_fraction, 'regime': regime, 'accuracy_mean': g['accuracy'].mean(), 'accuracy_sd': g['accuracy'].std(ddof=1), 'DG_mean': g['DG_mean'].mean(), 'DG_between_repeat_sd': g['DG_mean'].std(ddof=1), 'eREDG_median': med, 'eREDG_q1': q1, 'eREDG_q3': q3, 'eREDG_correct_median': np.median(corr_vals) if len(corr_vals) else np.nan, 'eREDG_incorrect_median': np.median(wrong_vals) if len(wrong_vals) else np.nan, 'audit_AUC_raw_mean': g['AUC_eREDG_raw'].mean(), 'audit_AUC_raw_sd': g['AUC_eREDG_raw'].std(ddof=1), 'cliffs_delta_mean': g['cliffs_delta_raw'].mean(), 'cliffs_delta_sd': g['cliffs_delta_raw'].std(ddof=1), 'active_fraction_mean': g['active_fraction'].mean(), 'active_fraction_sd': g['active_fraction'].std(ddof=1), 'n_valid_AUC': g['AUC_eREDG_raw'].notna().sum()})
    out = pd.DataFrame(rows)
    out['DG_ratio_vs_d0'] = np.nan
    for (dataset, dmg, reg), idx in out.groupby(['dataset', 'damage_mode', 'regime']).groups.items():
        rows_idx = list(idx)
        baseline = out.loc[rows_idx].loc[out.loc[rows_idx, 'damage_fraction'] == 0.0, 'DG_mean']
        if len(baseline) == 1:
            base = float(baseline.iloc[0])
            out.loc[rows_idx, 'DG_ratio_vs_d0'] = out.loc[rows_idx, 'DG_mean'] / (base + 1e-12)
    return out.sort_values(['dataset', 'damage_mode', 'damage_fraction', 'regime']).reset_index(drop=True)

def summary():
    global DAMAGE_FRACTIONS_MASTER, DAMAGE_MODES_MASTER, GATE_Q_MASTER, K_EARLY_MASTER, N_SPLITS_MASTER, REGIMES_MASTER, SEEDS_MASTER, TEST_SIZE_MASTER, X, X_test_local, X_train_local, _, band_logits_local, band_logits_test, band_models_local, df_dg_master, df_e_master, df_master_melanoma, df_master_runs_melanoma, dmg, eredg_correct_pool_melanoma, eredg_incorrect_pool_melanoma, eredg_pool_melanoma, frac, idx, key, m, master_rows_melanoma, morphogenetic_consensus_one, pipe, reg, row, seed, split_id, sss, sub, test_idx, train_idx, y, y_test, y_test_local, y_train_local
    morphogenetic_consensus_one = consensus_with_stress
    DAMAGE_FRACTIONS_MASTER = [0.0, 0.2, 0.4, 0.6]
    DAMAGE_MODES_MASTER = ['mixed', 'spiky']
    REGIMES_MASTER = ['plain', 'H1', 'H2']
    GATE_Q_MASTER = 0.2
    X = spectra
    y = y3
    N_SPLITS_MASTER = 10
    TEST_SIZE_MASTER = 0.2
    SEEDS_MASTER = list(range(100, 100 + N_SPLITS_MASTER))
    K_EARLY_MASTER = 3
    master_rows_melanoma = []
    eredg_pool_melanoma = defaultdict(list)
    eredg_correct_pool_melanoma = defaultdict(list)
    eredg_incorrect_pool_melanoma = defaultdict(list)
    for split_id, seed in enumerate(SEEDS_MASTER, start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE_MASTER, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        X_train_local = X[train_idx]
        X_test_local = X[test_idx]
        y_train_local = y[train_idx]
        y_test_local = y[test_idx]
        band_models_local = []
        for idx in band_indices:
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed))])
            pipe.fit(X_train_local[:, idx], y_train_local)
            band_models_local.append(pipe)
        band_logits_local = np.stack([m.decision_function(X_test_local[:, idx]) for m, idx in zip(band_models_local, band_indices)], axis=1)
        band_logits_test = band_logits_local
        y_test = y_test_local
        for frac in DAMAGE_FRACTIONS_MASTER:
            df_dg_master = compute_dg_and_stress_table(damage_fraction=frac, damage_modes=DAMAGE_MODES_MASTER, regimes=REGIMES_MASTER, run_id=seed, alpha=0.3, lam=1.0, T=0.01, tau=0.05, max_iter=30, weight_clip=(0.0, 5.0), morpho_weighting='confidence', k_early=K_EARLY_MASTER)
            df_e_master, _ = add_eREDG(df_dg_master, gate_by='per_damage_mode_and_regime', gate_q=GATE_Q_MASTER)
            for dmg in DAMAGE_MODES_MASTER:
                for reg in REGIMES_MASTER:
                    row = summarize_one_audit_condition(df_e_master, dataset='melanoma', repeat_id=split_id, damage_fraction=frac, damage_mode=dmg, regime=reg)
                    if row is not None:
                        master_rows_melanoma.append(row)
                    sub = df_e_master[(df_e_master['damage_mode'] == dmg) & (df_e_master['regime'] == reg)].dropna(subset=['eREDG'])
                    key = (frac, dmg, reg)
                    eredg_pool_melanoma[key].append(sub['eREDG'].to_numpy(dtype=float))
                    eredg_correct_pool_melanoma[key].append(sub.loc[sub['correct'] == 1, 'eREDG'].to_numpy(dtype=float))
                    eredg_incorrect_pool_melanoma[key].append(sub.loc[sub['correct'] == 0, 'eREDG'].to_numpy(dtype=float))
    df_master_runs_melanoma = pd.DataFrame(master_rows_melanoma)
    df_master_melanoma = make_master_summary(df_master_runs_melanoma, eredg_pool_melanoma, eredg_correct_pool_melanoma, eredg_incorrect_pool_melanoma)
    return df_master_melanoma

def spectra_plot():
    global class_names, cls, fp_mask, label, mask, mean_spec, spectra_fp, std_spec, wn_fp
    class_names = le3.inverse_transform(np.unique(y3))
    plt.figure(figsize=(10, 5))
    for cls in np.unique(y3):
        mask = y3 == cls
        mean_spec = spectra[mask].mean(axis=0)
        std_spec = spectra[mask].std(axis=0)
        label = le3.inverse_transform([cls])[0]
        plt.plot(wn, mean_spec, label=label)
        plt.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, alpha=0.2)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Normalized intensity (a.u.)')
    plt.title('Mean Raman spectra per class (±1 SD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Mean_Raman.png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    fp_mask = (wn >= 400) & (wn <= 2000)
    wn_fp = wn[fp_mask]
    spectra_fp = spectra[:, fp_mask]
    plt.figure(figsize=(10, 5))
    for cls in np.unique(y3):
        mask = y3 == cls
        mean_spec = spectra_fp[mask].mean(axis=0)
        std_spec = spectra_fp[mask].std(axis=0)
        label = le3.inverse_transform([cls])[0]
        plt.plot(wn_fp, mean_spec, label=label)
        plt.fill_between(wn_fp, mean_spec - std_spec, mean_spec + std_spec, alpha=0.2)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Normalized intensity (a.u.)')
    plt.title('Mean Raman spectra per class (400–2000 cm$^{-1}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Mean_Raman_per_class.png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
