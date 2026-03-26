from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def savefig(fig, path: str | Path, *, dpi: int = 600) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_mean_spectra_per_class(
    *,
    X: np.ndarray,
    y: np.ndarray,
    wn: np.ndarray,
    label_encoder,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in np.unique(y):
        mask = y == cls
        mean_spec = X[mask].mean(axis=0)
        std_spec = X[mask].std(axis=0)
        label = label_encoder.inverse_transform([cls])[0]
        ax.plot(wn, mean_spec, label=label)
        ax.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, alpha=0.2)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.set_title("Mean Raman spectra per class (±1 SD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_composite_accuracy_dgl(
    df_res_all: pd.DataFrame,
    *,
    highlight_regime: str = "H1",
    use_auc: bool = False,
    normalize_dg: bool = True,
    title: str = "Accuracy–Load decoupling under structured damage",
) -> plt.Figure:
    df = df_res_all.copy()
    if normalize_dg and "dg_norm" not in df.columns:
        df["dg_norm"] = np.nan
        for (reg, dmg), dfg in df.groupby(["regime", "damage_mode"]):
            ref = dfg.loc[dfg["damage_fraction"] == 0.0, "dg_index_mean"]
            ref_val = float(ref.iloc[0])
            df.loc[(df["regime"] == reg) & (df["damage_mode"] == dmg), "dg_norm"] = dfg["dg_index_mean"] / ref_val

    damage_modes = ["mixed", "spiky"]
    regimes = ["plain", "H1", "H2"]

    perf_y_oracle = "auc_full" if use_auc else "acc_full"
    perf_y_naive = "auc_naive" if use_auc else "acc_naive"
    perf_y_morpho = "auc_morpho" if use_auc else "acc_morpho"
    dg_y = "dg_norm" if normalize_dg else "dg_index_mean"

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    def draw_perf(ax, dmg: str):
        dfd = df[df["damage_mode"] == dmg].sort_values("damage_fraction")
        df0 = dfd[dfd["regime"] == "plain"]
        ax.plot(df0["damage_fraction"], df0[perf_y_oracle], "--", label="Full model (oracle)")
        ax.plot(df0["damage_fraction"], df0[perf_y_naive], "-o", label="Naive band avg")
        for reg in regimes:
            dfr = dfd[dfd["regime"] == reg]
            lw = 3.0 if reg.lower() == highlight_regime.lower() else 2.0
            mk = "o" if reg.lower() == highlight_regime.lower() else "s"
            ax.plot(dfr["damage_fraction"], dfr[perf_y_morpho], marker=mk, linewidth=lw, label=f"Morphogenetic ({reg})")
        ax.set_title(f"{'AUC' if use_auc else 'Accuracy'} vs Damage ({dmg})")
        ax.set_ylabel("Macro AUC (OVR)" if use_auc else "Accuracy")
        ax.grid(True, alpha=0.3)

    def draw_dg(ax, dmg: str):
        dfd = df[df["damage_mode"] == dmg].sort_values("damage_fraction")
        for reg in regimes:
            dfr = dfd[dfd["regime"] == reg]
            lw = 3.0 if reg.lower() == highlight_regime.lower() else 2.0
            mk = "o" if reg.lower() == highlight_regime.lower() else "s"
            ax.plot(dfr["damage_fraction"], dfr[dg_y], marker=mk, linewidth=lw, label=f"DGL ({reg})")
        ax.set_title(f"{'DGL' if normalize_dg else 'DG'} vs Damage ({dmg})")
        ax.set_xlabel("Damage fraction (bands corrupted)")
        ax.set_ylabel("DGL / DGL₀" if normalize_dg else "Mean DG")
        ax.grid(True, alpha=0.3)

    draw_perf(axA, "mixed")
    draw_perf(axB, "spiky")
    draw_dg(axC, "mixed")
    draw_dg(axD, "spiky")

    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontsize=14, fontweight="bold")

    handles_labels = []
    for ax in [axA, axB, axC, axD]:
        handles_labels.extend(list(zip(*ax.get_legend_handles_labels())))
    seen = set()
    handles = []
    labels = []
    for h, l in handles_labels:
        if l not in seen:
            seen.add(l)
            handles.append(h)
            labels.append(l)

    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


def plot_ablation(df_ablate: pd.DataFrame, *, value: str, title: str, ylabel: str, bands: list[tuple[float, float]]) -> plt.Figure:
    regimes = ["plain", "H1", "H2"]
    damage_modes = ["mixed", "spiky"]
    k_bands = df_ablate["band"].nunique()
    x = np.arange(k_bands)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, dmg in zip(axes, damage_modes):
        dfd = df_ablate[df_ablate["damage_mode"] == dmg].copy()
        for reg in regimes:
            dfr = dfd[dfd["regime"] == reg].sort_values("band")
            ax.plot(x, dfr[value].to_numpy(), "-o", label=reg)
        ax.set_xticks(x)
        ax.set_xticklabels([f"B{k}\n{bands[k][0]}-{bands[k][1]}" for k in range(k_bands)], fontsize=9)
        ax.set_title(f"{title} ({dmg})")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Ablated band")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def plot_pair_heatmaps(df_pair: pd.DataFrame, *, k_bands: int, damage_modes=("mixed", "spiky"), regimes=("plain", "H1", "H2"), bands=None) -> list[plt.Figure]:
    figs = []
    xt = [f"B{k}\n{bands[k][0]}-{bands[k][1]}" if bands else f"B{k}" for k in range(k_bands)]
    for dmg in damage_modes:
        for reg in regimes:
            dfr = df_pair[(df_pair["damage_mode"] == dmg) & (df_pair["regime"] == reg)].copy()
            mats = {}
            for label, value in [("ΔAccuracy", "delta_acc"), ("ΔMacro AUC", "delta_auc"), ("DG (mean)", "dg_mean")]:
                m = np.full((k_bands, k_bands), np.nan, dtype=float)
                for _, r in dfr.iterrows():
                    m[int(r["band_a"]), int(r["band_b"])] = float(r[value])
                mats[label] = m

            fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
            for ax, (ttl, m) in zip(axes, mats.items()):
                im = ax.imshow(m, aspect="auto")
                ax.set_title(f"{ttl}\nmode={dmg}, regime={reg}")
                ax.set_xticks(np.arange(k_bands)); ax.set_xticklabels(xt, fontsize=8)
                ax.set_yticks(np.arange(k_bands)); ax.set_yticklabels(xt, fontsize=8)
                ax.set_xlabel("Band b"); ax.set_ylabel("Band a")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            figs.append(fig)
    return figs


def plot_dg_distributions(df_dg: pd.DataFrame, *, metric="DG", damage_modes=("mixed", "spiky"), regimes=("plain", "H1", "H2")) -> list[plt.Figure]:
    figs = []
    for dmg in damage_modes:
        fig = plt.figure(figsize=(12.5, 3.9))
        for j, reg in enumerate(regimes, start=1):
            ax = plt.subplot(1, 3, j)
            dfr = df_dg[(df_dg["damage_mode"] == dmg) & (df_dg["regime"] == reg)].copy()
            dg_corr = dfr[dfr["correct"] == 1][metric].to_numpy()
            dg_wrong = dfr[dfr["correct"] == 0][metric].to_numpy()
            ax.hist(dg_corr, bins=20, alpha=0.6, label="Correct")
            ax.hist(dg_wrong, bins=20, alpha=0.6, label="Incorrect")
            ax.set_title(f"{reg}\n(mode={dmg})")
            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
            if j == 1:
                ax.legend(frameon=False)
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"DG distributions stratified by correctness — {metric}", y=1.02)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_redg_distributions(df: pd.DataFrame, *, damage_modes=("mixed", "spiky"), regimes=("plain", "H1", "H2")) -> list[plt.Figure]:
    figs = []
    for dmg in damage_modes:
        fig = plt.figure(figsize=(12.5, 3.9))
        for j, reg in enumerate(regimes, start=1):
            ax = plt.subplot(1, 3, j)
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            redg_corr = dfr[dfr["correct"] == 1]["REDG"].to_numpy()
            redg_wrong = dfr[dfr["correct"] == 0]["REDG"].to_numpy()
            ax.hist(redg_corr, bins=20, alpha=0.6, label="Correct")
            ax.hist(redg_wrong, bins=20, alpha=0.6, label="Incorrect")
            ax.set_title(f"{reg}\n(mode={dmg})")
            ax.set_xlabel("REDG")
            ax.set_ylabel("Count")
            if j == 1:
                ax.legend(frameon=False)
            ax.grid(True, alpha=0.25)
        fig.suptitle("REDG distributions stratified by correctness", y=1.02)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_roc_wrong(df: pd.DataFrame, *, score_col: str, damage_modes=("mixed", "spiky"), regimes=("plain", "H1", "H2")) -> list[plt.Figure]:
    figs = []
    for dmg in damage_modes:
        fig = plt.figure(figsize=(6.2, 5.0))
        ax = plt.gca()
        for reg in regimes:
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            y_wrong = 1 - dfr["correct"].to_numpy(dtype=int)
            x = dfr[score_col].to_numpy(dtype=float)
            if len(np.unique(y_wrong)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_wrong, x)
            auc = roc_auc_score(y_wrong, x)
            ax.plot(fpr, tpr, label=f"{reg} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", label="Chance")
        ax.set_title(f"ROC: {score_col} → wrong\n(mode={dmg})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_roc_redg_correctness(df: pd.DataFrame, *, damage_modes=("mixed", "spiky"), regimes=("plain", "H1", "H2")) -> tuple[list[plt.Figure], pd.DataFrame]:
    figs = []
    auc_rows = []
    for dmg in damage_modes:
        fig = plt.figure(figsize=(6.2, 5.0))
        ax = plt.gca()
        for reg in regimes:
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            y = dfr["correct"].to_numpy(dtype=int)
            if len(np.unique(y)) < 2:
                continue
            x = dfr["REDG"].to_numpy(dtype=float)
            auc_pos = roc_auc_score(y, x)
            auc_neg = roc_auc_score(y, -x)
            if auc_neg > auc_pos:
                x_use = -x
                auc = auc_neg
                sign = "-"
            else:
                x_use = x
                auc = auc_pos
                sign = "+"
            fpr, tpr, _ = roc_curve(y, x_use)
            ax.plot(fpr, tpr, label=f"{reg} (AUC={auc:.3f}, score={sign}REDG)")
            auc_rows.append({"damage_mode": dmg, "regime": reg, "auc_redg_to_correct": float(auc), "sign": sign})
        ax.plot([0, 1], [0, 1], "--", label="Chance")
        ax.set_title(f"ROC: REDG predicts correctness\n(mode={dmg})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        figs.append(fig)
    return figs, pd.DataFrame(auc_rows).sort_values(["damage_mode", "regime"]).reset_index(drop=True)


def final_eredg_panel(df: pd.DataFrame, *, damage_modes=("mixed", "spiky"), regimes=("plain", "H1", "H2"), score_col="eREDG") -> list[plt.Figure]:
    figs = []
    for dmg in damage_modes:
        fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
        for j, reg in enumerate(regimes):
            ax = axes[j]
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            dfr = dfr[np.isfinite(dfr[score_col].to_numpy(dtype=float))]
            corr = dfr[dfr["correct"] == 1][score_col].to_numpy(dtype=float)
            wrong = dfr[dfr["correct"] == 0][score_col].to_numpy(dtype=float)
            ax.hist(corr, bins=18, alpha=0.65, label="Correct")
            ax.hist(wrong, bins=18, alpha=0.65, label="Incorrect")
            ax.set_title(reg)
            ax.set_xlabel(score_col)
            ax.set_ylabel("Count" if j == 0 else "")
            if j == 0:
                ax.legend(frameon=False)
            ax.grid(True, alpha=0.25)

        ax = axes[3]
        for reg in regimes:
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            dfr = dfr[np.isfinite(dfr[score_col].to_numpy(dtype=float))]
            y = dfr["correct"].to_numpy(dtype=int)
            x = dfr[score_col].to_numpy(dtype=float)
            if len(np.unique(y)) < 2:
                continue
            if np.allclose(x, x[0]):
                x_use = x
                auc = 0.5
            else:
                auc_pos = roc_auc_score(y, x)
                auc_neg = roc_auc_score(y, -x)
                if auc_neg > auc_pos:
                    x_use = -x
                    auc = auc_neg
                else:
                    x_use = x
                    auc = auc_pos
            fpr, tpr, _ = roc_curve(y, x_use)
            ax.plot(fpr, tpr, label=f"{reg} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", label="Chance")
        ax.set_title("ROC: score → correctness")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        figs.append(fig)
    return figs
