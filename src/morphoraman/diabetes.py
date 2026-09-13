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

def stress_history_from_l_hist(l_hist: np.ndarray, use_js: bool=True, local_coupling: bool=True):
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

def compute_dg_and_stress_table(*, band_logits_test: np.ndarray, y_test: np.ndarray, damage_fraction: float=0.4, damage_modes=('mixed', 'spiky'), regimes=('plain', 'H1', 'H2'), run_id: int=0, alpha: float=0.3, lam: float=1.0, T: float=0.01, tau: float=0.05, max_iter: int=30, weight_clip: tuple=(0.0, 5.0), morpho_weighting: str='confidence', k_early: int=3):
    band_logits_test = np.asarray(band_logits_test, dtype=float)
    y_test = np.asarray(y_test)
    if band_logits_test.ndim != 3:
        raise ValueError(f'band_logits_test must have shape (N,K,C), got {band_logits_test.shape}')
    N, K, C = band_logits_test.shape
    rows = []
    for dmg in damage_modes:
        for reg in regimes:
            for i in range(N):
                L0_list = [band_logits_test[i, k, :].copy() for k in range(K)]
                y_true = int(y_test[i])
                seed = run_id * 1000000 + 10000 * int(round(damage_fraction * 100)) + 100 * i
                rng = np.random.default_rng(seed)
                Ld_list, frozen_flags, damaged_idx = apply_band_damage(L0_list, damage_fraction=damage_fraction, mode=dmg, rng=rng)
                if morpho_weighting == 'uniform':
                    w0 = np.ones(K, dtype=float)
                else:
                    conf = np.array([np.max(softmax(l)) for l in Ld_list], dtype=float)
                    w0 = np.clip(conf, 0.0, 0.999)
                res = morphogenetic_consensus_one(band_logits0=Ld_list, frozen_flags=frozen_flags, band_weights0=w0, regime=reg, weight_clip=weight_clip, alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter, use_js=True, return_trajectories=True, rng=rng)
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
                rows.append({'i': i, 'damage_mode': dmg, 'regime': reg, 'damage_fraction': float(damage_fraction), 'DG': dg, 'DG_early': dg_early, 'stress0_mean': stress0_mean, 'stress0_max': stress0_max, 'correct': correct, 'y_true': y_true, 'y_hat': y_hat})
    df_out = pd.DataFrame(rows)
    if len(df_out) > 0:
        df_out['DG_norm'] = df_out['DG'] / (1.0 + df_out['stress0_mean'])
    else:
        df_out['DG_norm'] = pd.Series(dtype=float)
    return df_out

def prepare(data_dir):
    global BANDS, C, Dict, GroupShuffleSplit, K, LabelEncoder, List, LogisticRegression, N_test, Optional, Pipeline, StandardScaler, StratifiedKFold, Tuple, X_crop, X_norm, X_raw, X_test, X_train, acc_full, accuracy_score, all_idx, axis_row, band_indices, band_logits_test, band_pipes, base_label2, base_label3, c, confusion_matrix, covered, crop_mask, dec_k, df2, df3, df_all, df_site, dm_negative_values, dm_positive_values, file_path, filename, fp_mask, full_decision_test, full_logits_test, gap_points, group_col, groups, groups2, groups_test, groups_train, gss, hi, i, idx, k, kind_col, label_col, label_col_raw, label_map_2class, label_map_3class, labels, labels_raw, le2, le3, lo, mask2, mask_axis, mpl, non_spec_cols, normalize, np, os, patient_col_raw, patient_ids, pd, pipe_full, pipe_k, plt, raw_spec_cols, roc_auc_score, row, rows, site, site_files, sort_idx, spec_col_names, spec_cols, spec_cols_sorted, spectra, test_idx, train_idx, train_test_split, unique_idx, valid_rows, w, wn, wn_crop, wn_full_site, wn_reference, y, y2, y2_str, y3, y_pred_full, y_test, y_train
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
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    site_files = {site: str(Path(data_dir) / filename) for site, filename in {'innerArm': 'innerArm.csv', 'thumbNail': 'thumbNail.csv', 'vein': 'vein.csv', 'earLobe': 'earLobe.csv'}.items()}
    label_col_raw = 'has_DM2'
    patient_col_raw = 'patientID'
    rows = []
    wn_reference = None
    for site, file_path in site_files.items():
        df_site = pd.read_csv(file_path)
        raw_spec_cols = [c for c in df_site.columns if c.startswith('Var')]
        raw_spec_cols = sorted(raw_spec_cols, key=lambda x: int(x.replace('Var', '')))
        mask_axis = df_site[patient_col_raw].astype(str).str.strip().str.lower() == 'ramanshift'
        if mask_axis.sum() != 1:
            raise ValueError(f"{site}: expected exactly one 'ramanShift' row, found {mask_axis.sum()}")
        axis_row = df_site.loc[mask_axis, raw_spec_cols].iloc[0]
        wn_full_site = pd.to_numeric(axis_row, errors='coerce').to_numpy(dtype=float)
        if np.isnan(wn_full_site).any():
            raise ValueError(f'{site}: Raman axis row contains non-numeric values')
        if wn_reference is None:
            wn_reference = wn_full_site.copy()
        elif not np.allclose(wn_reference, wn_full_site, atol=1e-08, rtol=0):
            raise ValueError(f'{site}: Raman axis differs from previous files')
        df_site = df_site.loc[~mask_axis].copy().reset_index(drop=True)
        labels = df_site[label_col_raw].astype(str).str.strip().to_numpy()
        patient_ids = df_site[patient_col_raw].astype(str).str.strip().to_numpy()
        X_raw = df_site[raw_spec_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        valid_rows = ~pd.isna(df_site[label_col_raw]).to_numpy()
        labels = labels[valid_rows]
        patient_ids = patient_ids[valid_rows]
        X_raw = X_raw[valid_rows]
        crop_mask = (wn_reference >= 400.0) & (wn_reference <= 1800.0)
        wn_crop = wn_reference[crop_mask]
        X_crop = X_raw[:, crop_mask]
        X_norm = normalize(X_crop, axis=1)
        spec_col_names = [f'{w:.6f}' for w in wn_crop]
        for i in range(X_norm.shape[0]):
            row = {'Label': labels[i], 'Kind': site, 'patientID': patient_ids[i]}
            row.update(dict(zip(spec_col_names, X_norm[i])))
            rows.append(row)
    df_all = pd.DataFrame(rows)
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    assert 'Label' in df_all.columns
    assert 'Kind' in df_all.columns
    assert 'patientID' in df_all.columns
    label_col = 'Label'
    kind_col = 'Kind'
    group_col = 'patientID'
    labels_raw = df_all[label_col].astype(str).str.strip().to_numpy()
    dm_positive_values = {'1', '1.0'}
    dm_negative_values = {'0', '0.0'}
    y2_str = np.full(shape=labels_raw.shape, fill_value='exclude', dtype=object)
    y2_str[np.isin(labels_raw, list(dm_positive_values))] = 'dm2'
    y2_str[np.isin(labels_raw, list(dm_negative_values))] = 'non_dm2'
    mask2 = y2_str != 'exclude'
    df2 = df_all.loc[mask2].reset_index(drop=True)
    y2_str = y2_str[mask2]
    base_label2 = labels_raw[mask2]
    groups2 = df2[group_col].to_numpy()
    non_spec_cols = {label_col, kind_col, group_col}
    spec_cols = [c for c in df2.columns if c not in non_spec_cols and np.issubdtype(df2[c].dtype, np.number)]
    spectra = df2[spec_cols].to_numpy(dtype=np.float64)
    wn = np.array([float(c) for c in spec_cols], dtype=float)
    sort_idx = np.argsort(wn)
    wn = wn[sort_idx]
    spectra = spectra[:, sort_idx]
    spec_cols_sorted = [spec_cols[i] for i in sort_idx]
    le2 = LabelEncoder()
    y2 = le2.fit_transform(y2_str)
    label_map_2class = dict(zip(le2.classes_, range(len(le2.classes_))))
    df3 = df2.copy()
    y3 = y2.copy()
    le3 = le2
    label_map_3class = label_map_2class.copy()
    base_label3 = base_label2.copy()
    BANDS = [(400, 600), (600, 800), (800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800)]
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
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np
    y = y2
    groups = groups2
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(spectra, y, groups=groups))
    X_train, X_test = (spectra[train_idx], spectra[test_idx])
    y_train, y_test = (y[train_idx], y[test_idx])
    groups_train, groups_test = (groups[train_idx], groups[test_idx])
    pipe_full = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=SEED))])
    pipe_full.fit(X_train, y_train)
    full_decision_test = pipe_full.decision_function(X_test)
    full_logits_test = np.column_stack([-full_decision_test, full_decision_test])
    K = len(band_indices)
    N_test = X_test.shape[0]
    C = 2
    band_logits_test = np.zeros((N_test, K, C), dtype=float)
    band_pipes = []
    for k, idx in enumerate(band_indices):
        pipe_k = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=SEED))])
        pipe_k.fit(X_train[:, idx], y_train)
        dec_k = pipe_k.decision_function(X_test[:, idx])
        band_logits_test[:, k, :] = np.column_stack([-dec_k, dec_k])
        band_pipes.append(pipe_k)
    y_pred_full = pipe_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    df3 = df2.copy()
    y3 = y2.copy()
    le3 = le2
    label_map_3class = label_map_2class.copy()
    base_label3 = base_label2.copy()
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

def fit_band_models_for_partition(X_train, X_test, y_train, band_indices, seed=0):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    K = len(band_indices)
    N_test = X_test.shape[0]
    C = len(np.unique(y_train))
    band_logits_test = np.zeros((N_test, K, C), dtype=float)
    band_pipes = []
    for k, idx in enumerate(band_indices):
        pipe_k = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed))])
        pipe_k.fit(X_train[:, idx], y_train)
        dec = pipe_k.decision_function(X_test[:, idx])
        if dec.ndim == 1:
            logits = np.column_stack([-dec, dec])
        else:
            logits = dec
        band_logits_test[:, k, :] = logits
        band_pipes.append(pipe_k)
    return (band_logits_test, band_pipes)
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
    rows.append({'partition': partition_label, 'damage_mode': 'clean' if damage_mode is None else damage_mode, 'damage_fraction': float(damage_fraction), 'method': 'naive_band', 'acc': accuracy_score(y_test, naive_pred), 'auc': roc_auc_score(y_test, naive_probs[:, 1]), 'DG_mean': np.nan})
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
        rows.append({'partition': partition_label, 'damage_mode': 'clean' if damage_mode is None else damage_mode, 'damage_fraction': float(damage_fraction), 'method': regime, 'acc': accuracy_score(y_test, pred_labels), 'auc': roc_auc_score(y_test, pred_probs[:, 1]), 'DG_mean': float(np.mean(dg_vals))})
    return pd.DataFrame(rows)

def partition():
    global K_use, band_defs, band_indices_use, band_logits_test_use, band_pipes_use, bands_use, df_clean, df_mixed04, df_part_sens, hi, j, lo, morphogenetic_consensus_one, results
    morphogenetic_consensus_one = consensus_with_stress
    results = []
    band_defs = {}
    for K_use in [5, 7, 9]:
        bands_use, band_indices_use = make_equalwidth_bands(wn, K_use, lo=400.0, hi=1800.0)
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
from sklearn.model_selection import StratifiedGroupKFold
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

def audit(n_splits=5):
    global df_eredg_runs, df_eredg_samples_all, morphogenetic_consensus_one
    saved_consensus = morphogenetic_consensus_one
    morphogenetic_consensus_one = consensus_with_stress
    samples = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    try:
        for split_id, (train_idx, test_idx) in enumerate(splitter.split(spectra, y3, groups2), start=1):
            seed = 100 + split_id
            if set(groups2[train_idx]) & set(groups2[test_idx]):
                raise ValueError('Patient overlap between training and evaluation sets')
            logits = []
            for idx in band_indices:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed)),
                ])
                model.fit(spectra[train_idx][:, idx], y3[train_idx])
                decision = model.decision_function(spectra[test_idx][:, idx])
                logits.append(np.column_stack([-decision, decision]))
            table = compute_dg_and_stress_table(
                band_logits_test=np.stack(logits, axis=1), y_test=y3[test_idx],
                damage_fraction=0.4, damage_modes=('mixed', 'spiky'),
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
        morphogenetic_consensus_one = saved_consensus
    df_eredg_samples_all = pd.concat(samples, ignore_index=True)
    df_eredg_runs = summarize_audit(df_eredg_samples_all, 'split')
    plot_audit(df_eredg_runs, df_eredg_samples_all, 'Diabetes')
    return df_eredg_runs, df_eredg_samples_all

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
    fp_mask = (wn >= 400) & (wn <= 1800)
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
    plt.title('Mean Raman spectra per class (400–1800 cm$^{-1}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Mean_Raman_per_class.png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
