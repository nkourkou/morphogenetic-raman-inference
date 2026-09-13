import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def correctness_auc(correct, eredg):
    correct = np.asarray(correct)
    eredg = np.asarray(eredg, dtype=float)
    if correct.shape != eredg.shape or correct.ndim != 1:
        raise ValueError('Correctness and eREDG must be matching 1D arrays')
    valid = np.isfinite(eredg) & pd.notna(correct)
    y, score = correct[valid], eredg[valid]
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Correctness must be coded as 0 or 1')
    if np.unique(y).size < 2:
        return float('nan')
    return float(roc_auc_score(y, score))


def summarize_audit(samples, repeat_col):
    rows = []
    for key, group in samples.groupby([repeat_col, 'damage_mode', 'regime']):
        active = group.loc[np.isfinite(group['eREDG'])].dropna(subset=['correct'])
        correct = active.loc[active['correct'] == 1, 'eREDG'].to_numpy()
        incorrect = active.loc[active['correct'] == 0, 'eREDG'].to_numpy()
        auc = correctness_auc(active['correct'], active['eREDG'])
        delta = 2.0 * auc - 1.0 if np.isfinite(auc) else np.nan
        rows.append({
            repeat_col: key[0], 'damage_mode': key[1], 'regime': key[2],
            'auc_eREDG_to_correct': auc, 'cliffs_delta': delta,
            'n_total': len(group), 'n_active': len(active),
            'n_correct': len(correct), 'n_incorrect': len(incorrect),
        })
    return pd.DataFrame(rows)


def plot_audit(runs, samples, dataset, damage_mode='mixed'):
    regimes = ('plain', 'H1', 'H2')
    runs = runs[runs['damage_mode'] == damage_mode]
    samples = samples[samples['damage_mode'] == damage_mode]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for i, regime in enumerate(regimes):
        values = runs.loc[runs['regime'] == regime, 'auc_eREDG_to_correct'].dropna().to_numpy()
        if len(values):
            axes[0].scatter(i + np.linspace(-0.08, 0.08, len(values)), values)
            axes[0].errorbar(i, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0,
                             fmt='o', capsize=4)
    axes[0].axhline(0.5, linestyle='--', color='gray')
    axes[0].set(xticks=range(3), xticklabels=regimes, ylim=(0, 1),
                ylabel='AUC(eREDG → correctness)', title=f'[A] Audit AUC ({damage_mode})')
    values, positions, labels = [], [], []
    for i, regime in enumerate(regimes):
        for outcome, label in [(1, 'correct'), (0, 'incorrect')]:
            mask = (samples['regime'] == regime) & (samples['correct'] == outcome)
            values.append(samples.loc[mask, 'eREDG'].dropna().to_numpy())
            positions.append(i * 3 + 1 - outcome)
            labels.append(f'{regime}\n{label}')
    axes[1].boxplot(values, positions=positions, widths=0.65, showfliers=False)
    axes[1].set(xticks=positions, xticklabels=labels, ylabel='eREDG (unflipped)',
                title=f'[B] eREDG by outcome ({damage_mode})')
    axes[1].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(f'{dataset}_eREDG_fixed_direction_{damage_mode}.png', dpi=600, bbox_inches='tight')
    plt.show()
    return fig
