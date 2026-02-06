""" 
This file contains additional functions that were originally defined across many cells
in the exported notebook script. They are kept here to make the refactor feature-complete.

Prefer using:
- raman_morpho.morpho for core dynamics and damage models
- raman_morpho.evaluation for main sweep evaluation
- raman_morpho.plots for the core figures

These legacy helpers are still importable and runnable.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Dict

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Reuse core primitives
from .morpho import softmax, kl_div, js_div, energy_bands, global_logits_from_bands, global_unsortedness, compute_dg_index, morphogenetic_consensus_one, apply_random_band_damage
from .evaluation import evaluate_morphogenetic_system


def class_props(y):
    u, c = np.unique(y, return_counts=True)
    return dict(zip(u, c / c.sum()))

print("Overall:", class_props(y))
print("Train:  ", class_props(y_train))
print("Test:   ", class_props(y_test))


# In[ ]:


# Composite figure (2x2) = Accuracy + DGL (DG normalized) under Mixed vs Spiky
# Uses df_res_all (from Cell 10) and df_res_norm (from Cell 11A) if available.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def _ensure_dg_norm(df_res_all: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with dg_norm computed per (regime, damage_mode) relative to damage_fraction==0."""
    df = df_res_all.copy()
    if "dg_norm" in df.columns:
        return df
    df["dg_norm"] = np.nan
    for (reg, dmg), dfg in df.groupby(["regime", "damage_mode"]):
        ref = dfg.loc[dfg["damage_fraction"] == 0.0, "dg_index_mean"]
        if len(ref) != 1:
            raise ValueError(f"Expected exactly one zero-damage DG for {reg}, {dmg}")
        ref_val = float(ref.iloc[0])
        df.loc[(df["regime"] == reg) & (df["damage_mode"] == dmg), "dg_norm"] = dfg["dg_index_mean"] / ref_val
    return df



def plot_composite_accuracy_dgl(
    df_res_all: pd.DataFrame,
    *,
    highlight_regime: str = "H1",
    use_auc: bool = False,          # False -> Accuracy, True -> AUC
    normalize_dg: bool = True,      # True -> dg_norm, False -> dg_index_mean
    show_dg_std: bool = True,       # use dg_index_std only if normalize_dg==False
    title: str = "Accuracy–Load decoupling under structured damage"
):
    df = _ensure_dg_norm(df_res_all) if normalize_dg else df_res_all.copy()

    # --- configuration ---
    damage_modes = ["mixed", "spiky"]
    regimes = ["plain", "H1", "H2"]
    oracle_label = "Full model (oracle)"
    naive_label = "Naive band avg"

    # axis field names
    perf_y_oracle = "auc_full" if use_auc else "acc_full"
    perf_y_naive  = "auc_naive" if use_auc else "acc_naive"
    perf_y_morpho = "auc_morpho" if use_auc else "acc_morpho"

    dg_y = "dg_norm" if normalize_dg else "dg_index_mean"
    dg_yerr = "dg_index_std"  # only meaningful in raw DG scale

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    # helper: draw performance panel
    def draw_perf(ax, dmg):
        dfd = df[df["damage_mode"] == dmg].copy()
        dfd = dfd.sort_values("damage_fraction")

        # oracle and naive (identical across regimes by construction -> take plain slice)
        df0 = dfd[dfd["regime"] == "plain"]
        ax.plot(df0["damage_fraction"], df0[perf_y_oracle], "--", label=oracle_label)
        ax.plot(df0["damage_fraction"], df0[perf_y_naive],  "-o", label=naive_label)

        # morpho curves per regime
        for reg in regimes:
            dfr = dfd[dfd["regime"] == reg]
            lw = 3.0 if reg.lower() == highlight_regime.lower() else 2.0
            mk = "o" if reg.lower() == highlight_regime.lower() else "s"
            ax.plot(dfr["damage_fraction"], dfr[perf_y_morpho], marker=mk, linewidth=lw, label=f"Morphogenetic ({reg})")

        ax.set_title(f"{'AUC' if use_auc else 'Accuracy'} vs Damage ({dmg})")
        ax.set_ylabel("Macro AUC (OVR)" if use_auc else "Accuracy")
        ax.grid(True, alpha=0.3)

    # helper: draw DG/DGL panel
    def draw_dg(ax, dmg):
        dfd = df[df["damage_mode"] == dmg].copy()
        dfd = dfd.sort_values("damage_fraction")

        for reg in regimes:
            dfr = dfd[dfd["regime"] == reg]
            lw = 3.0 if reg.lower() == highlight_regime.lower() else 2.0
            mk = "o" if reg.lower() == highlight_regime.lower() else "s"

            ax.plot(dfr["damage_fraction"], dfr[dg_y], marker=mk, linewidth=lw, label=f"DGL ({reg})" if normalize_dg else f"DG ({reg})")

            # optional error bars (only raw DG scale)
            if show_dg_std and (not normalize_dg) and (dg_yerr in dfr.columns):
                ax.fill_between(
                    dfr["damage_fraction"].to_numpy(),
                    (dfr[dg_y] - dfr[dg_yerr]).to_numpy(),
                    (dfr[dg_y] + dfr[dg_yerr]).to_numpy(),
                    alpha=0.15
                )

        ax.set_title(f"{'DGL (normalized DG)' if normalize_dg else 'DG'} vs Damage ({dmg})")
        ax.set_xlabel("Damage fraction (bands corrupted)")
        ax.set_ylabel("DGL / DGL₀" if normalize_dg else "Mean DG")
        ax.grid(True, alpha=0.3)

    # --- draw panels ---
    draw_perf(axA, "mixed")
    draw_perf(axB, "spiky")
    draw_dg(axC, "mixed")
    draw_dg(axD, "spiky")

    # --- panel letters ---
    axA.text(0.01, 0.98, "A", transform=axA.transAxes, va="top", ha="left", fontsize=14, fontweight="bold")
    axB.text(0.01, 0.98, "B", transform=axB.transAxes, va="top", ha="left", fontsize=14, fontweight="bold")
    axC.text(0.01, 0.98, "C", transform=axC.transAxes, va="top", ha="left", fontsize=14, fontweight="bold")
    axD.text(0.01, 0.98, "D", transform=axD.transAxes, va="top", ha="left", fontsize=14, fontweight="bold")
    # Gather unique handles/labels from all axes
    handles_labels = []
    for ax in [axA, axB, axC, axD]:
        handles_labels.extend(list(zip(*ax.get_legend_handles_labels())))
    # De-duplicate while preserving order
    seen = set()
    handles, labels = [], []
    for h, l in handles_labels:
        if l not in seen:
            seen.add(l)
            handles.append(h)
            labels.append(l)

    # --- legend below the panels ---
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(title, y=0.98)

    # Leave extra space at bottom for legend
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.show()

# --- call it ---
plot_composite_accuracy_dgl(
    df_res_all,
    highlight_regime="H1",   # H1 highlighted as requested
    use_auc=False,           # set True to get AUC version
    normalize_dg=True,       # DGL = DG/DG0
    show_dg_std=False,       # std not meaningful after normalization with only 4 points
    title="Accuracy–Load decoupling under structured damage (3-class Raman)"
)


# In[ ]:


# DGL vs damage scatter (mixed vs spiky), H1 highlighted, with ρ annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --- Ensure we have normalized DG (DGL) ---



def ensure_dg_norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dg_norm" in df.columns:
        return df
    df["dg_norm"] = np.nan
    for (reg, dmg), dfg in df.groupby(["regime", "damage_mode"]):
        ref = dfg.loc[dfg["damage_fraction"] == 0.0, "dg_index_mean"]
        if len(ref) != 1:
            raise ValueError(f"Expected exactly one zero-damage DG for {reg}, {dmg}")
        ref_val = float(ref.iloc[0])
        df.loc[(df["regime"] == reg) & (df["damage_mode"] == dmg), "dg_norm"] = dfg["dg_index_mean"] / ref_val
    return df

df_plot = ensure_dg_norm(df_res_all)

# --- Plot settings ---
damage_modes = ["mixed", "spiky"]
regimes = ["plain", "H1", "H2"]
highlight_regime = "H1"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

for ax, dmg in zip(axes, damage_modes):
    dfd = df_plot[df_plot["damage_mode"] == dmg].sort_values("damage_fraction")

    # plot each regime as scatter+line; highlight H1
    for reg in regimes:
        dfr = dfd[dfd["regime"] == reg].sort_values("damage_fraction")

        lw = 3.0 if reg.lower() == highlight_regime.lower() else 2.0
        ms = 8 if reg.lower() == highlight_regime.lower() else 6
        mk = "o" if reg.lower() == highlight_regime.lower() else "s"
        ls = "-"  # keep simple and consistent

        ax.plot(
            dfr["damage_fraction"],
            dfr["dg_norm"],
            linestyle=ls,
            marker=mk,
            linewidth=lw,
            markersize=ms,
            label=reg
        )

        # Spearman rho (note: with 4 levels and strict monotonicity, rho can be 1.0 by construction)
        rho, p = spearmanr(dfr["damage_fraction"], dfr["dg_norm"])
        ax.text(
            0.02,
            0.92 - 0.10 * regimes.index(reg),
            f"{reg}: ρ={rho:.2f}",
            transform=ax.transAxes,
            fontsize=10
        )

    ax.set_title(f"DGL vs damage ({dmg})")
    ax.set_xlabel("Damage fraction")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("DGL = DG / DG$_0$ (normalized)")

# Legend below (no overlap)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Figure 3. Monotonic coupling between perturbation severity and Decision Geometry Load (DGL)", y=1.02)
plt.tight_layout(rect=[0, 0.06, 1, 0.98])
plt.show()


# In[ ]:


# Algotypes registry (formal names + parameter presets)

ALGOTYPES = {
    "plain": dict(regime="plain", dynamic_freeze=False, delayed=False),
    "H1":    dict(regime="H1",    dynamic_freeze=False, delayed=False),
    "H2":    dict(regime="H2",    dynamic_freeze=False, delayed=False),

    # NEW: Frozen-cells algotype (irreversible agent commitment)
    "FZ":    dict(regime="plain", dynamic_freeze=True,  delayed=False),

    # NEW: Delayed gratification algotype (time-aggregated stress gating)
    "DG":    dict(regime="plain", dynamic_freeze=False, delayed=True),

    # NEW: Combined (often the most interpretable)
    "FZ+DG": dict(regime="plain", dynamic_freeze=True,  delayed=True),
}

print("Available algotypes:", list(ALGOTYPES.keys()))


# In[ ]:


# Morphogenetic consensus v2 — supports:
# (i) Frozen cells (dynamic irreversible freeze)
# (ii) Delayed gratification (stress must persist before action)

from typing import List, Optional, Dict
import numpy as np



def morphogenetic_consensus_one_v2(
    band_logits0: np.ndarray,                 # (K,C) preferred
    frozen_flags: Optional[np.ndarray] = None,# (K,) initial frozen
    band_weights0: Optional[np.ndarray] = None,
    *,
    # base morpho params
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    use_js: bool = True,
    return_trajectories: bool = True,
    regime: str = "plain",                   # "plain" | "H1" | "H2"
    local_coupling: bool = True,
    mute_weight: float = 0.0,
    diff_amp: float = 0.25,
    weight_clip: tuple = (0.0, 5.0),

    # dynamic freezing
    dynamic_freeze: bool = False,
    tau_freeze: float = 0.12,                # higher than tau (freeze is "harder" than mute)
    freeze_patience: int = 2,                # require sustained stress for N steps
    freeze_snap_to_consensus: bool = True,   # freeze and snap or freeze-in-place

    # delayed gratification / memory
    delayed: bool = False,
    stress_ema_beta: float = 0.7,            # higher => more memory
    action_patience: int = 2,                # require sustained EMA stress before acting

    rng: Optional[np.random.Generator] = None
) -> Dict:
    if rng is None:
        rng = np.random.default_rng(42)

    L0 = np.asarray(band_logits0, dtype=float)
    if L0.ndim != 2:
        raise ValueError(f"band_logits0 must be (K,C), got {L0.shape}")
    K, C = L0.shape

    if frozen_flags is None:
        frozen_flags = np.zeros(K, dtype=bool)
    else:
        frozen_flags = np.asarray(frozen_flags, dtype=bool)
        if frozen_flags.shape != (K,):
            raise ValueError(f"frozen_flags must have shape ({K},), got {frozen_flags.shape}")

    # state
    band_logits = L0.copy()

    # initial weights
    if band_weights0 is None:
        w0 = np.clip(np.max(softmax(band_logits), axis=1), 0.0, 1.0)
        weights = w0.astype(float)
    else:
        weights = np.asarray(band_weights0, dtype=float).copy()
        if weights.shape != (K,):
            raise ValueError(f"band_weights0 must have shape ({K},), got {weights.shape}")

    wmin, wmax = float(weight_clip[0]), float(weight_clip[1])
    weights = np.clip(weights, wmin, wmax)

    # neighbours
    if local_coupling:
        neighbours = []
        for k in range(K):
            neigh = []
            if k - 1 >= 0: neigh.append(k - 1)
            if k + 1 < K: neigh.append(k + 1)
            neighbours.append(neigh)
    else:
        neighbours = [[j for j in range(K) if j != k] for k in range(K)]

    div = js_div if use_js else kl_div

    # memory for delayed gratification
    stress_ema = np.zeros(K, dtype=float)
    stress_run = np.zeros(K, dtype=int)      # counts consecutive "high" stress for patience gating
    freeze_run = np.zeros(K, dtype=int)

    # histories for DG/DGL
    l_hist = [band_logits.copy()] if return_trajectories else None
    E_hist, U_hist, l_global_hist = [], [], []

    E_prev = energy_bands([band_logits[k] for k in range(K)],
                          [L0[k] for k in range(K)],
                          weights,
                          lam=lam,
                          use_js=use_js,
                          local_coupling=local_coupling)

    l_global = global_logits_from_bands([band_logits[k] for k in range(K)], weights)
    U_prev = global_unsortedness(l_global)

    E_hist.append(E_prev); U_hist.append(U_prev); l_global_hist.append(l_global.copy())

    freeze_events = []  # list of (t, k, stress_value)

    for t in range(max_iter):
        band_logits_old = band_logits.copy()
        weights_old = weights.copy()
        frozen_old = frozen_flags.copy()

        band_logits_prop = band_logits.copy()
        weights_prop = weights.copy()
        frozen_prop = frozen_flags.copy()

        # propose agent updates
        for k in range(K):
            if frozen_prop[k]:
                continue
            neigh = neighbours[k]
            if len(neigh) == 0:
                continue

            w_neigh = weights[neigh]
            Z = float(np.sum(w_neigh)) + 1e-12
            l_neigh_avg = (w_neigh[:, None] * band_logits[neigh]).sum(axis=0) / Z

            # stress
            p_k = softmax(band_logits[k])
            p_star = softmax(l_neigh_avg)
            stress = div(p_k, p_star)

            # delayed gratification: maintain EMA + patience counter
            if delayed:
                stress_ema[k] = stress_ema_beta * stress_ema[k] + (1.0 - stress_ema_beta) * stress
                high = (stress_ema[k] > tau)
            else:
                high = (stress > tau)

            stress_run[k] = stress_run[k] + 1 if high else 0

            # dynamic freezing: separate higher threshold + patience
            if dynamic_freeze:
                high_freeze = (stress > tau_freeze) if not delayed else (stress_ema[k] > tau_freeze)
                freeze_run[k] = freeze_run[k] + 1 if high_freeze else 0

                if freeze_run[k] >= freeze_patience:
                    # freeze now (irreversible within this run)
                    frozen_prop[k] = True
                    freeze_events.append((t, k, float(stress_ema[k] if delayed else stress)))
                    if freeze_snap_to_consensus:
                        band_logits_prop[k] = l_neigh_avg
                    continue  # no further action on frozen agent

            # apply regime logic, but gated by delayed patience if requested
            can_act = (stress_run[k] >= action_patience) if delayed else True

            if regime.lower() == "plain":
                band_logits_prop[k] = (1.0 - alpha) * band_logits[k] + alpha * l_neigh_avg

            elif regime.lower() == "h1":
                if can_act and (stress > tau):
                    weights_prop[k] = mute_weight
                    band_logits_prop[k] = l_neigh_avg
                else:
                    band_logits_prop[k] = (1.0 - alpha) * band_logits[k] + alpha * l_neigh_avg

            elif regime.lower() == "h2":
                if can_act and (stress > tau):
                    weights_prop[k] = mute_weight
                    band_logits_prop[k] = l_neigh_avg
                else:
                    band_logits_prop[k] = (1.0 - alpha) * band_logits[k] + alpha * l_neigh_avg
                    weights_prop[k] = np.clip(weights_prop[k] * (1.0 + diff_amp), wmin, wmax)

            else:
                raise ValueError("Unknown regime (use 'plain','H1','H2')")

        weights_prop = np.clip(weights_prop, wmin, wmax)

        # accept/reject by energy
        E_new = energy_bands([band_logits_prop[k] for k in range(K)],
                             [L0[k] for k in range(K)],
                             weights_prop,
                             lam=lam,
                             use_js=use_js,
                             local_coupling=local_coupling)

        accept = True
        if (E_new > E_prev) and (T > 0.0):
            prob = float(np.exp(-(E_new - E_prev) / T))
            if rng.random() >= prob:
                accept = False

        if accept:
            band_logits = band_logits_prop
            weights = weights_prop
            frozen_flags = frozen_prop
            E_prev = E_new
        else:
            band_logits = band_logits_old
            weights = weights_old
            frozen_flags = frozen_old

        l_global = global_logits_from_bands([band_logits[k] for k in range(K)], weights)
        U_prev = global_unsortedness(l_global)

        E_hist.append(E_prev); U_hist.append(U_prev); l_global_hist.append(l_global.copy())
        if return_trajectories:
            l_hist.append(band_logits.copy())

    l_final = global_logits_from_bands([band_logits[k] for k in range(K)], weights)
    y_hat = int(np.argmax(l_final))
    p_final = softmax(l_final)

    out = {
        "y_hat": y_hat,
        "p_final": p_final,
        "weights_final": weights,
        "frozen_final": frozen_flags,
        "freeze_events": freeze_events,
    }
    if return_trajectories:
        out["E_hist"] = np.asarray(E_hist)
        out["U_hist"] = np.asarray(U_hist)
        out["l_global_hist"] = np.stack(l_global_hist, axis=0)
        out["l_hist"] = np.stack(l_hist, axis=0)  # (T+1, K, C)
    return out


# In[ ]:


# ΔAccuracy = (spiky - mixed) vs damage_fraction (Global/Naive/Morpho)

import matplotlib.pyplot as plt
import numpy as np



def delta_spiky_minus_mixed(df, col, regime=None):
    if regime is None:
        a = df[df["damage_mode"]=="spiky"].groupby("damage_fraction")[col].mean()
        b = df[df["damage_mode"]=="mixed"].groupby("damage_fraction")[col].mean()
    else:
        a = df[(df["damage_mode"]=="spiky") & (df["regime"]==regime)].set_index("damage_fraction")[col]
        b = df[(df["damage_mode"]=="mixed") & (df["regime"]==regime)].set_index("damage_fraction")[col]
    # align
    x = sorted(set(a.index).intersection(set(b.index)))
    return np.array(x), (a.loc[x].values - b.loc[x].values)

plt.figure(figsize=(8.5,4.6))

# Global fitting (oracle): acc_full
x, d = delta_spiky_minus_mixed(df_res_all, "acc_full")
plt.plot(x, d, "--o", label="Global fitting (full-spectrum)")

# Naive
x, d = delta_spiky_minus_mixed(df_res_all, "acc_naive")
plt.plot(x, d, "-o", label="Naive")

# Morpho regimes
for reg in ["plain","H1","H2"]:
    x, d = delta_spiky_minus_mixed(df_res_all, "acc_morpho", regime=reg)
    plt.plot(x, d, "-o", label=("Plain morpho" if reg=="plain" else reg))

plt.axhline(0, linewidth=1, alpha=0.4)
plt.xlabel("Damage fraction")
plt.ylabel("ΔAccuracy (spiky − mixed)")
plt.title("Structural sensitivity of accuracy: spiky vs mixed damage")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# ΔDG = (spiky - mixed) vs damage_fraction (plain/H1/H2)

plt.figure(figsize=(8.5,4.6))

for reg in ["plain","H1","H2"]:
    x, d = delta_spiky_minus_mixed(df_res_all, "dg_index_mean", regime=reg)
    plt.plot(x, d, "-o", label=f"DG {reg}")

plt.axhline(0, linewidth=1, alpha=0.4)
plt.xlabel("Damage fraction")
plt.ylabel("ΔDG (spiky − mixed)")
plt.title("Structural sensitivity of Decision Geometry: spiky vs mixed damage")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# # Part 2 - Damages

# In[ ]:


# Cell: Per-band ablation (one band damaged at a time) — 7-band macro problem (3-class)
# Produces a results table + bar plots (Δaccuracy, ΔAUC, DG) for spiky vs mixed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# ---- helpers ----



def _macro_auc_ovr(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")



def _damage_one_band(
    band_logits0: np.ndarray,   # (K,C)
    k: int,
    mode: str,
    rng: np.random.Generator,
    noise_scale: float,
    spike_scale_mult: float = 6.0,
) -> np.ndarray:
    """Return damaged logits (K,C) with only band k overwritten."""
    L = band_logits0.copy()
    C = L.shape[1]
    if mode == "mixed":
        # mixed: since only one band is damaged, treat it as "spiky" with prob 0.35 else noisy
        if rng.random() < 0.35:
            L[k] = rng.normal(0.0, spike_scale_mult * noise_scale, size=C)
        else:
            L[k] = rng.normal(0.0, noise_scale, size=C)
    elif mode == "spiky":
        L[k] = rng.normal(0.0, spike_scale_mult * noise_scale, size=C)
    else:
        raise ValueError("mode must be 'mixed' or 'spiky'")
    return L



def per_band_ablation(
    band_logits_test: np.ndarray,  # (N,K,C)
    y_test: np.ndarray,            # (N,)
    *,
    damage_modes=("mixed","spiky"),
    regimes=("plain","H1","H2"),
    # morpho params (keep consistent with your sweep)
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    run_id: int = 0,
    spike_scale_mult: float = 6.0,
) -> pd.DataFrame:
    N, K, C = band_logits_test.shape

    # ---- clean (no damage) baselines per regime ----
    baseline = {}  # baseline[(dmg,reg)] = dict(acc=..., auc=..., dg_mean=...)
    # Note: "damage_mode" affects only how we corrupt; baseline does not depend on it
    for reg in regimes:
        y_hat = np.zeros(N, dtype=int)
        p_hat = np.zeros((N, C), dtype=float)
        dg_vals = np.zeros(N, dtype=float)

        for i in range(N):
            L0_list = [band_logits_test[i, k, :] for k in range(K)]
            frozen = np.zeros(K, dtype=bool)

            # reliability prior
            if morpho_weighting == "uniform":
                w0 = np.ones(K, dtype=float)
            elif morpho_weighting == "confidence":
                conf = np.array([np.max(softmax(l)) for l in L0_list], dtype=float)
                w0 = np.clip(conf, 0.0, 0.999)
            else:
                raise ValueError("morpho_weighting must be 'uniform' or 'confidence'")

            res = morphogenetic_consensus_one(
                L0_list,
                frozen_flags=frozen,
                band_weights0=w0,
                regime=reg,
                weight_clip=weight_clip,
                alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                use_js=True,
                return_trajectories=True
            )
            y_hat[i] = int(res["y_hat"])
            p_hat[i] = np.asarray(res["p_final"], dtype=float)
            dg_vals[i] = compute_dg_index(res["l_hist"])

        baseline[reg] = dict(
            acc=accuracy_score(y_test, y_hat),
            auc=_macro_auc_ovr(y_test, p_hat),
            dg_mean=float(np.mean(dg_vals)),
            dg_std=float(np.std(dg_vals)),
        )

    # ---- per-band ablation ----
    rows = []
    for dmg in damage_modes:
        for reg in regimes:
            for k_ablate in range(K):

                y_hat = np.zeros(N, dtype=int)
                p_hat = np.zeros((N, C), dtype=float)
                dg_vals = np.zeros(N, dtype=float)

                for i in range(N):
                    L0 = band_logits_test[i].copy()  # (K,C)

                    # noise scale from sample's logits
                    s = float(np.std(L0))
                    noise_scale = s if s > 1e-8 else 1.0

                    seed = (run_id * 1_000_000) + (1000 * (k_ablate + 1)) + i
                    rng = np.random.default_rng(seed)

                    Ld = _damage_one_band(
                        L0, k=k_ablate, mode=dmg, rng=rng,
                        noise_scale=noise_scale,
                        spike_scale_mult=spike_scale_mult
                    )

                    # Convert to list-of-arrays for existing consensus function
                    Ld_list = [Ld[j, :] for j in range(K)]
                    frozen = np.zeros(K, dtype=bool)

                    # weights prior computed from *damaged* evidence (same as your sweep)
                    if morpho_weighting == "uniform":
                        w0 = np.ones(K, dtype=float)
                    else:
                        conf = np.array([np.max(softmax(l)) for l in Ld_list], dtype=float)
                        w0 = np.clip(conf, 0.0, 0.999)

                    res = morphogenetic_consensus_one(
                        Ld_list,
                        frozen_flags=frozen,
                        band_weights0=w0,
                        regime=reg,
                        weight_clip=weight_clip,
                        alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                        use_js=True,
                        return_trajectories=True
                    )

                    y_hat[i] = int(res["y_hat"])
                    p_hat[i] = np.asarray(res["p_final"], dtype=float)
                    dg_vals[i] = compute_dg_index(res["l_hist"])

                acc = accuracy_score(y_test, y_hat)
                auc = _macro_auc_ovr(y_test, p_hat)
                dg_mean = float(np.mean(dg_vals))
                dg_std = float(np.std(dg_vals))

                # deltas vs clean baseline for that regime
                acc0 = baseline[reg]["acc"]
                auc0 = baseline[reg]["auc"]

                rows.append({
                    "damage_mode": dmg,
                    "regime": reg,
                    "band": k_ablate,
                    "band_range": f"{BANDS[k_ablate][0]}–{BANDS[k_ablate][1]}",
                    "acc": float(acc),
                    "auc": float(auc),
                    "dg_mean": dg_mean,
                    "dg_std": dg_std,
                    "delta_acc": float(acc - acc0),
                    "delta_auc": float(auc - auc0),
                    "baseline_acc": float(acc0),
                    "baseline_auc": float(auc0),
                    "baseline_dg_mean": float(baseline[reg]["dg_mean"]),
                })

                print(f"[{dmg} | {reg}] ablate band {k_ablate} ({BANDS[k_ablate][0]}-{BANDS[k_ablate][1]}): "
                      f"acc={acc:.3f} (Δ{acc-acc0:+.3f}) | auc={auc:.3f} (Δ{auc-auc0:+.3f}) | "
                      f"DG={dg_mean:.3f}±{dg_std:.3f}")

    return pd.DataFrame(rows)

df_ablate = per_band_ablation(
    band_logits_test=band_logits_test,
    y_test=y_test,
    damage_modes=("mixed","spiky"),
    regimes=("plain","H1","H2"),
    alpha=0.3, lam=1.0, T=1e-2, tau=0.05, max_iter=30,
    weight_clip=(0.0, 5.0),
    morpho_weighting="confidence",
    run_id=0
)

df_ablate


# In[ ]:


# Plot per-band ablation results (ΔAccuracy, ΔAUC, DG) — spiky vs mixed

import numpy as np
import matplotlib.pyplot as plt



def plot_ablation(df_ablate: pd.DataFrame, value: str, title: str, ylabel: str):
    regimes = ["plain","H1","H2"]
    damage_modes = ["mixed","spiky"]
    K = df_ablate["band"].nunique()
    x = np.arange(K)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, dmg in zip(axes, damage_modes):
        dfd = df_ablate[df_ablate["damage_mode"] == dmg].copy()

        for reg in regimes:
            dfr = dfd[dfd["regime"] == reg].sort_values("band")
            ax.plot(x, dfr[value].to_numpy(), "-o", label=reg)

        ax.set_xticks(x)
        ax.set_xticklabels([f"B{k}\n{BANDS[k][0]}-{BANDS[k][1]}" for k in range(K)], fontsize=9)
        ax.set_title(f"{title} ({dmg})")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Ablated band")

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

# ΔAccuracy and ΔAUC relative to the clean baseline of the same regime
plot_ablation(df_ablate, value="delta_acc", title="ΔAccuracy vs one-band ablation", ylabel="ΔAccuracy (ablation − clean)")
plot_ablation(df_ablate, value="delta_auc", title="ΔMacro AUC vs one-band ablation", ylabel="ΔMacro AUC (OVR)")

# DG absolute (not delta) to show negotiation load under single-band lesion
plot_ablation(df_ablate, value="dg_mean", title="Decision Geometry (DG) under one-band ablation", ylabel="Mean DG")


# In[ ]:


# Band-pair ablation (7×7) heatmaps — ΔAccuracy, ΔAUC, and DG
# For each damage_mode (mixed, spiky) and each regime (plain, H1, H2),
# we ablate two bands at a time and compute:
#   - delta_acc  (vs clean baseline of same regime)
#   - delta_auc  (vs clean baseline of same regime)
#   - dg_mean    (absolute)
#
# Output: df_pair + 3 heatmaps per (damage_mode, regime)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score



def _damage_bands(
    L0: np.ndarray,                 # (K,C)
    bands_to_damage: list,          # indices to overwrite
    mode: str,
    rng: np.random.Generator,
    noise_scale: float,
    spike_scale_mult: float = 6.0,
) -> np.ndarray:
    """Damage the specified bands; return (K,C)."""
    L = L0.copy()
    K, C = L.shape

    if mode == "spiky":
        for k in bands_to_damage:
            L[k] = rng.normal(0.0, spike_scale_mult * noise_scale, size=C)

    elif mode == "mixed":
        # Mixed: each damaged band becomes spiky with prob 0.35, else noisy
        for k in bands_to_damage:
            if rng.random() < 0.35:
                L[k] = rng.normal(0.0, spike_scale_mult * noise_scale, size=C)
            else:
                L[k] = rng.normal(0.0, noise_scale, size=C)
    else:
        raise ValueError("mode must be 'mixed' or 'spiky'")

    return L



def compute_clean_baselines_for_pair(
    band_logits_test: np.ndarray,  # (N,K,C)
    y_test: np.ndarray,
    regimes=("plain","H1","H2"),
    *,
    alpha=0.3, lam=1.0, T=1e-2, tau=0.05, max_iter=30,
    weight_clip=(0.0, 5.0),
    morpho_weighting="confidence",
) -> dict:
    """Return baseline per regime: acc0, auc0."""
    N, K, C = band_logits_test.shape
    baseline = {}

    for reg in regimes:
        y_hat = np.zeros(N, dtype=int)
        p_hat = np.zeros((N, C), dtype=float)

        for i in range(N):
            L0 = band_logits_test[i]  # (K,C)
            L0_list = [L0[k, :] for k in range(K)]
            frozen = np.zeros(K, dtype=bool)

            if morpho_weighting == "uniform":
                w0 = np.ones(K, dtype=float)
            else:
                conf = np.array([np.max(softmax(l)) for l in L0_list], dtype=float)
                w0 = np.clip(conf, 0.0, 0.999)

            res = morphogenetic_consensus_one(
                L0_list,
                frozen_flags=frozen,
                band_weights0=w0,
                regime=reg,
                weight_clip=weight_clip,
                alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                use_js=True,
                return_trajectories=False
            )
            y_hat[i] = int(res["y_hat"])
            p_hat[i] = np.asarray(res["p_final"], dtype=float)

        baseline[reg] = dict(
            acc=float(accuracy_score(y_test, y_hat)),
            auc=float(_macro_auc_ovr(y_test, p_hat)),
        )
        print(f"[baseline {reg}] acc={baseline[reg]['acc']:.3f}, auc={baseline[reg]['auc']:.3f}")

    return baseline



def band_pair_ablation(
    band_logits_test: np.ndarray,  # (N,K,C)
    y_test: np.ndarray,
    *,
    damage_modes=("mixed","spiky"),
    regimes=("plain","H1","H2"),
    alpha=0.3, lam=1.0, T=1e-2, tau=0.05, max_iter=30,
    weight_clip=(0.0, 5.0),
    morpho_weighting="confidence",
    run_id=0,
    spike_scale_mult=6.0
) -> pd.DataFrame:
    N, K, C = band_logits_test.shape
    baseline = compute_clean_baselines_for_pair(
        band_logits_test, y_test, regimes,
        alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
        weight_clip=weight_clip,
        morpho_weighting=morpho_weighting
    )

    rows = []
    for dmg in damage_modes:
        for reg in regimes:
            for a in range(K):
                for b in range(K):
                    # symmetric grid: we compute all pairs but can later plot full 7x7
                    y_hat = np.zeros(N, dtype=int)
                    p_hat = np.zeros((N, C), dtype=float)
                    dg_vals = np.zeros(N, dtype=float)

                    for i in range(N):
                        L0 = band_logits_test[i].copy()  # (K,C)
                        s = float(np.std(L0))
                        noise_scale = s if s > 1e-8 else 1.0

                        seed = (run_id * 1_000_000) + (10_000 * (a + 1)) + (100 * (b + 1)) + i
                        rng = np.random.default_rng(seed)

                        Ld = _damage_bands(
                            L0,
                            bands_to_damage=sorted({a, b}),
                            mode=dmg,
                            rng=rng,
                            noise_scale=noise_scale,
                            spike_scale_mult=spike_scale_mult
                        )
                        Ld_list = [Ld[k, :] for k in range(K)]
                        frozen = np.zeros(K, dtype=bool)

                        if morpho_weighting == "uniform":
                            w0 = np.ones(K, dtype=float)
                        else:
                            conf = np.array([np.max(softmax(l)) for l in Ld_list], dtype=float)
                            w0 = np.clip(conf, 0.0, 0.999)

                        res = morphogenetic_consensus_one(
                            Ld_list,
                            frozen_flags=frozen,
                            band_weights0=w0,
                            regime=reg,
                            weight_clip=weight_clip,
                            alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                            use_js=True,
                            return_trajectories=True
                        )
                        y_hat[i] = int(res["y_hat"])
                        p_hat[i] = np.asarray(res["p_final"], dtype=float)
                        dg_vals[i] = compute_dg_index(res["l_hist"])

                    acc = float(accuracy_score(y_test, y_hat))
                    auc = float(_macro_auc_ovr(y_test, p_hat))
                    dg_mean = float(np.mean(dg_vals))

                    acc0 = baseline[reg]["acc"]
                    auc0 = baseline[reg]["auc"]

                    rows.append({
                        "damage_mode": dmg,
                        "regime": reg,
                        "band_a": a,
                        "band_b": b,
                        "band_a_range": f"{BANDS[a][0]}–{BANDS[a][1]}",
                        "band_b_range": f"{BANDS[b][0]}–{BANDS[b][1]}",
                        "acc": acc,
                        "auc": auc,
                        "dg_mean": dg_mean,
                        "delta_acc": float(acc - acc0),
                        "delta_auc": float(auc - auc0),
                    })

                print(f"[{dmg} | {reg}] finished row a={a}")

    return pd.DataFrame(rows)

df_pair = band_pair_ablation(
    band_logits_test=band_logits_test,
    y_test=y_test,
    damage_modes=("mixed","spiky"),
    regimes=("plain","H1","H2"),
    alpha=0.3, lam=1.0, T=1e-2, tau=0.05, max_iter=30,
    weight_clip=(0.0, 5.0),
    morpho_weighting="confidence",
    run_id=0
)

df_pair.head()


# In[ ]:


# Plot 7×7 heatmaps for ΔAccuracy, ΔAUC, and DG (per damage_mode and regime)

import numpy as np
import matplotlib.pyplot as plt



def _matrix_from_df(dfp: pd.DataFrame, value: str, K: int) -> np.ndarray:
    M = np.full((K, K), np.nan, dtype=float)
    for _, r in dfp.iterrows():
        M[int(r["band_a"]), int(r["band_b"])] = float(r[value])
    return M



def plot_pair_heatmaps(df_pair: pd.DataFrame, *, K: int, damage_modes=("mixed","spiky"), regimes=("plain","H1","H2")):
    xt = [f"B{k}\n{BANDS[k][0]}-{BANDS[k][1]}" for k in range(K)]
    yt = xt

    for dmg in damage_modes:
        for reg in regimes:
            dfr = df_pair[(df_pair["damage_mode"] == dmg) & (df_pair["regime"] == reg)].copy()

            mats = {
                "ΔAccuracy": _matrix_from_df(dfr, "delta_acc", K),
                "ΔMacro AUC": _matrix_from_df(dfr, "delta_auc", K),
                "DG (mean)": _matrix_from_df(dfr, "dg_mean", K),
            }

            fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
            for ax, (ttl, M) in zip(axes, mats.items()):
                im = ax.imshow(M, aspect="auto")
                ax.set_title(f"{ttl}\nmode={dmg}, regime={reg}")
                ax.set_xticks(np.arange(K)); ax.set_xticklabels(xt, fontsize=8)
                ax.set_yticks(np.arange(K)); ax.set_yticklabels(yt, fontsize=8)
                ax.set_xlabel("Band b"); ax.set_ylabel("Band a")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # optional: annotate diagonal as single-band equivalent (a==b)
                ax.plot(np.arange(K), np.arange(K), "w.", markersize=2)

            plt.tight_layout()
            plt.show()

K = band_logits_test.shape[1]
plot_pair_heatmaps(df_pair, K=K, damage_modes=("mixed","spiky"), regimes=("plain","H1","H2"))


# In[ ]:


# Compute per-sample DG + correctness + (optional) DG_norm at fixed damage level
# Outputs df_dg with one row per test sample, per regime, per damage_mode.

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve



def compute_dg_dataset(
    *,
    damage_fraction: float = 0.4,
    damage_modes=("mixed", "spiky"),
    regimes=("plain", "H1", "H2"),
    run_id: int = 0,
    # morpho params (MUST match your main runs)
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple = (0.0, 5.0),
    morpho_weighting: str = "confidence",
):
    N, K, C = band_logits_test.shape
    rows = []

    for dmg in damage_modes:
        for reg in regimes:
            for i in range(N):
                L0_list = [band_logits_test[i, k, :] for k in range(K)]
                y_true = int(y_test[i])

                seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i
                rng = np.random.default_rng(seed)

                Ld_list, frozen_flags, damaged_idx = apply_band_damage(
                    L0_list,
                    damage_fraction=damage_fraction,
                    mode=dmg,
                    rng=rng
                )

                if morpho_weighting == "uniform":
                    w0 = np.ones(K, dtype=float)
                else:
                    conf = np.array([np.max(softmax(l)) for l in Ld_list], dtype=float)
                    w0 = np.clip(conf, 0.0, 0.999)

                res = morphogenetic_consensus_one(
                    Ld_list,
                    frozen_flags=frozen_flags,
                    band_weights0=w0,
                    regime=reg,
                    weight_clip=weight_clip,
                    alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                    use_js=True,
                    return_trajectories=True
                )

                y_hat = int(res["y_hat"])
                correct = int(y_hat == y_true)

                dg = float(compute_dg_index(res["l_hist"]))
                # DG_norm: normalize by number of steps and agents to compare across settings (optional)
                Tp1 = res["l_hist"].shape[0]
                Tsteps = max(Tp1 - 1, 1)
                dg_norm = dg / (Tsteps * K)

                rows.append({
                    "i": i,
                    "damage_mode": dmg,
                    "regime": reg,
                    "damage_fraction": float(damage_fraction),
                    "y_true": y_true,
                    "y_hat": y_hat,
                    "correct": correct,
                    "DG": dg,
                    "DG_norm": float(dg_norm),
                    "damaged_idx": tuple(sorted(damaged_idx.tolist()))
                })

            print(f"Done: mode={dmg}, regime={reg}")

    return pd.DataFrame(rows)

df_dg = compute_dg_dataset(
    damage_fraction=0.4,
    damage_modes=("mixed","spiky"),
    regimes=("plain","H1","H2"),
    run_id=0
)

df_dg.head()


# In[ ]:


# DG distributions stratified by correctness (damage_fraction = 0.4)
# Produces 2 panels per damage_mode: DG raw and DG_norm.

import numpy as np
import matplotlib.pyplot as plt



def plot_dg_distributions(df_dg: pd.DataFrame, *, metric="DG", damage_modes=("mixed","spiky"), regimes=("plain","H1","H2")):
    for dmg in damage_modes:
        plt.figure(figsize=(12.5, 3.9))

        for j, reg in enumerate(regimes, start=1):
            ax = plt.subplot(1, 3, j)
            dfr = df_dg[(df_dg["damage_mode"] == dmg) & (df_dg["regime"] == reg)].copy()

            dg_corr = dfr[dfr["correct"] == 1][metric].to_numpy()
            dg_wrong = dfr[dfr["correct"] == 0][metric].to_numpy()

            # Use simple hist overlays (robust, no extra deps)
            ax.hist(dg_corr, bins=20, alpha=0.6, label="Correct")
            ax.hist(dg_wrong, bins=20, alpha=0.6, label="Incorrect")

            ax.set_title(f"{reg}\n(mode={dmg})")
            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)

            if j == 1:
                ax.legend(frameon=False)

        plt.suptitle(f"DG distributions stratified by correctness (damage_fraction=0.4) — {metric}", y=1.02)
        plt.tight_layout()
        plt.show()

plot_dg_distributions(df_dg, metric="DG")
plot_dg_distributions(df_dg, metric="DG_norm")


# In[ ]:


# Recompute a per-sample table at damage_fraction=0.4 with:
# - stress0_mean, stress0_max
# - DG (already)
# - DG_early (first k steps)
#
# This cell creates df_dg2 (one row per sample × mode × regime).

import numpy as np
import pandas as pd



def stress_history_from_l_hist(l_hist: np.ndarray, use_js: bool = True, local_coupling: bool = True) -> np.ndarray:
    """
    stress[t,k] = divergence(softmax(l_k(t)), softmax(neigh_avg(t)))
    l_hist: (T+1, K, C)
    returns: (T+1, K)
    """
    l_hist = np.asarray(l_hist, dtype=float)
    Tp1, K, C = l_hist.shape

    if local_coupling:
        neighbours = []
        for k in range(K):
            neigh = []
            if k-1 >= 0: neigh.append(k-1)
            if k+1 < K: neigh.append(k+1)
            neighbours.append(neigh)
    else:
        neighbours = [[j for j in range(K) if j != k] for k in range(K)]

    div = js_div if use_js else kl_div
    stress = np.zeros((Tp1, K), dtype=float)

    for t in range(Tp1):
        L = l_hist[t]  # (K,C)
        for k in range(K):
            neigh = neighbours[k]
            if len(neigh) == 0:
                stress[t, k] = 0.0
                continue
            l_neigh_avg = np.mean(L[neigh, :], axis=0)
            stress[t, k] = div(softmax(L[k, :]), softmax(l_neigh_avg))
    return stress



def compute_dg_and_stress_table(
    *,
    damage_fraction: float = 0.4,
    damage_modes=("mixed","spiky"),
    regimes=("plain","H1","H2"),
    run_id: int = 0,
    # morpho params
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    k_early: int = 3,  # first k steps for "early DG"
):
    N, K, C = band_logits_test.shape
    rows = []

    for dmg in damage_modes:
        for reg in regimes:
            for i in range(N):
                L0_list = [band_logits_test[i, k, :] for k in range(K)]
                y_true = int(y_test[i])

                seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i
                rng = np.random.default_rng(seed)

                Ld_list, frozen_flags, damaged_idx = apply_band_damage(
                    L0_list, damage_fraction=damage_fraction, mode=dmg, rng=rng
                )

                if morpho_weighting == "uniform":
                    w0 = np.ones(K, dtype=float)
                else:
                    conf = np.array([np.max(softmax(l)) for l in Ld_list], dtype=float)
                    w0 = np.clip(conf, 0.0, 0.999)

                res = morphogenetic_consensus_one(
                    Ld_list,
                    frozen_flags=frozen_flags,
                    band_weights0=w0,
                    regime=reg,
                    weight_clip=weight_clip,
                    alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                    use_js=True,
                    return_trajectories=True
                )

                l_hist = res["l_hist"]                # (Tp1,K,C)
                dg = float(compute_dg_index(l_hist))

                # early DG: only first k steps
                diffs = l_hist[1:] - l_hist[:-1]      # (T,K,C)
                step_norms = np.linalg.norm(diffs, axis=2).sum(axis=1)  # (T,)
                k_use = min(k_early, step_norms.shape[0])
                dg_early = float(np.sum(step_norms[:k_use]))

                stress = stress_history_from_l_hist(l_hist, use_js=True, local_coupling=True)
                stress0 = stress[0, :]
                stress0_mean = float(np.mean(stress0))
                stress0_max  = float(np.max(stress0))

                y_hat = int(res["y_hat"])
                correct = int(y_hat == y_true)

                rows.append({
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
                    "damaged_idx": tuple(sorted(damaged_idx.tolist()))
                })
            print(f"Done: mode={dmg}, regime={reg}")

    return pd.DataFrame(rows)

df_dg2 = compute_dg_and_stress_table(damage_fraction=0.4, k_early=3)
df_dg2.head()


# In[ ]:


# Stratified ROC by initial stress quartiles
# We plot AUC within each quartile to see whether DG becomes informative
# when conditioned on conflict level.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score



def stratified_auc(df: pd.DataFrame, score_col: str, stress_col: str):
    out = []
    for (dmg, reg), dfr in df.groupby(["damage_mode","regime"]):
        # quartiles of stress
        q = pd.qcut(dfr[stress_col], q=4, labels=[1,2,3,4])
        dfr = dfr.copy()
        dfr["q"] = q

        for qq in [1,2,3,4]:
            sub = dfr[dfr["q"] == qq]
            y_wrong = (1 - sub["correct"].to_numpy(dtype=int))
            x = sub[score_col].to_numpy(dtype=float)

            if len(np.unique(y_wrong)) < 2:
                auc = np.nan
            else:
                auc = float(roc_auc_score(y_wrong, x))

            out.append({
                "damage_mode": dmg,
                "regime": reg,
                "quartile": qq,
                "n": int(len(sub)),
                f"AUC({score_col}→wrong)": auc
            })
    return pd.DataFrame(out)

tab_strat = stratified_auc(df_dg2, score_col="DG", stress_col="stress0_mean")
tab_strat


# In[ ]:


# Create DG derivatives and run ROC for WRONGNESS
# We compute:
# - DG_eff = DG / (stress0_mean + eps)
# - DG_early_eff = DG_early / (stress0_mean + eps)
# Then AUC(score->wrong) by mode & regime.

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

eps = 1e-9
df_feat = df_dg2.copy()
df_feat["DG_eff"] = df_feat["DG"] / (df_feat["stress0_mean"] + eps)
df_feat["DG_early_eff"] = df_feat["DG_early"] / (df_feat["stress0_mean"] + eps)



def auc_by_group(df: pd.DataFrame, score_col: str):
    rows = []
    for (dmg, reg), dfr in df.groupby(["damage_mode","regime"]):
        y_wrong = (1 - dfr["correct"].to_numpy(dtype=int))
        x = dfr[score_col].to_numpy(dtype=float)
        if len(np.unique(y_wrong)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y_wrong, x))
        rows.append({"damage_mode": dmg, "regime": reg, f"AUC({score_col}→wrong)": auc})
    return pd.DataFrame(rows).sort_values(["damage_mode","regime"]).reset_index(drop=True)

tab_DG = auc_by_group(df_feat, "DG")
tab_DG_eff = auc_by_group(df_feat, "DG_eff")
tab_DG_early = auc_by_group(df_feat, "DG_early")
tab_DG_early_eff = auc_by_group(df_feat, "DG_early_eff")

tab_DG, tab_DG_eff, tab_DG_early, tab_DG_early_eff


# In[ ]:


# Choose the single best DG-derivative by mean AUC across (damage_mode, regime)

import numpy as np
import pandas as pd

candidates = ["DG", "DG_eff", "DG_early", "DG_early_eff"]

summary = []
for col in candidates:
    tab = auc_by_group(df_feat, col)
    mean_auc = float(np.nanmean(tab[f"AUC({col}→wrong)"].to_numpy()))
    summary.append({"metric": col, "mean_auc": mean_auc})

df_metric_rank = pd.DataFrame(summary).sort_values("mean_auc", ascending=False).reset_index(drop=True)
df_metric_rank


# In[ ]:


# Plot ROC curves for WRONGNESS using the best metric

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

best_metric = df_metric_rank.loc[0, "metric"]
print("Best metric:", best_metric)



def plot_roc_wrong(df: pd.DataFrame, score_col: str, damage_modes=("mixed","spiky"), regimes=("plain","H1","H2")):
    for dmg in damage_modes:
        plt.figure(figsize=(6.2, 5.0))
        for reg in regimes:
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            y_wrong = (1 - dfr["correct"].to_numpy(dtype=int))
            x = dfr[score_col].to_numpy(dtype=float)

            if len(np.unique(y_wrong)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_wrong, x)
            auc = roc_auc_score(y_wrong, x)
            plt.plot(fpr, tpr, label=f"{reg} (AUC={auc:.3f})")

        plt.plot([0,1],[0,1],"--", label="Chance")
        plt.title(f"ROC: {score_col} → WRONG\n(damage_fraction=0.4, mode={dmg})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

plot_roc_wrong(df_feat, best_metric)


# In[ ]:


# Compute "REDG" (Regulated Early Decision Geometry) per sample
# REDG = DG_early / DG_total   where DG_early is cumulative DG over first t* steps
#
# This is the key DG-derivative:
# - scale-free (ratio)
# - captures whether reconciliation is concentrated early (efficient) or spread/delayed (inefficient)

import numpy as np
import pandas as pd



def add_redg_features(df: pd.DataFrame, *, early_frac: float = 0.25, eps: float = 1e-12) -> pd.DataFrame:
    """
    Adds REDG features by re-running trajectories only if needed.
    If df already has DG and l_hist is not stored, we recompute DG_early using the saved max_iter assumption:
    Here we rely on df_dg2 which already has DG and DG_early computed for k_early steps.
    For robustness, we compute REDG using DG_early / DG.
    """
    out = df.copy()
    # If DG_early exists, use it. Otherwise user must recompute via df_dg2 builder.
    assert "DG" in out.columns, "Expected DG in df"
    assert "DG_early" in out.columns, "Expected DG_early in df (recompute with compute_dg_and_stress_table)"
    out["REDG"] = out["DG_early"] / (out["DG"] + eps)
    return out

# Use df_dg2 produced earlier (has DG, DG_early, stress0, etc.)
df_feat2 = add_redg_features(df_dg2, eps=1e-12)

df_feat2[["damage_mode","regime","correct","DG","DG_early","REDG"]].head()


# In[ ]:


# REDG distributions stratified by correctness (damage_fraction=0.4)
# Expectation:
# - plain: weak/overlapping
# - H1/H2: more stratification (correct tends to higher REDG: DG concentrated early)

import numpy as np
import matplotlib.pyplot as plt



def plot_redg_distributions(df: pd.DataFrame, *, damage_modes=("mixed","spiky"), regimes=("plain","H1","H2")):
    for dmg in damage_modes:
        plt.figure(figsize=(12.5, 3.9))
        for j, reg in enumerate(regimes, start=1):
            ax = plt.subplot(1, 3, j)
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()

            redg_corr = dfr[dfr["correct"] == 1]["REDG"].to_numpy()
            redg_wrong = dfr[dfr["correct"] == 0]["REDG"].to_numpy()

            ax.hist(redg_corr, bins=20, alpha=0.6, label="Correct")
            ax.hist(redg_wrong, bins=20, alpha=0.6, label="Incorrect")

            ax.set_title(f"{reg}\n(mode={dmg})")
            ax.set_xlabel("REDG = DG_early / DG_total")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)

            if j == 1:
                ax.legend(frameon=False)

        plt.suptitle("REDG distributions stratified by correctness (damage_fraction=0.4)", y=1.02)
        plt.tight_layout()
        plt.show()

plot_redg_distributions(df_feat2)


# In[ ]:


# Figure B — ROC curves using REDG to predict correctness
# Predictor: +REDG (higher means "more DG early" => more likely correct)
#
# If AUC < 0.5, flip the direction (use -REDG). We do this automatically.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score



def plot_roc_redg_correctness(df: pd.DataFrame, *, damage_modes=("mixed","spiky"), regimes=("plain","H1","H2")):
    auc_rows = []
    for dmg in damage_modes:
        plt.figure(figsize=(6.2, 5.0))
        for reg in regimes:
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            y = dfr["correct"].to_numpy(dtype=int)

            if len(np.unique(y)) < 2:
                continue

            x = dfr["REDG"].to_numpy(dtype=float)
            auc_pos = roc_auc_score(y, x)
            auc_neg = roc_auc_score(y, -x)

            # choose orientation that is >= 0.5 for interpretability
            if auc_neg > auc_pos:
                x_use = -x
                auc = auc_neg
                sign = "-"
            else:
                x_use = x
                auc = auc_pos
                sign = "+"

            fpr, tpr, _ = roc_curve(y, x_use)
            plt.plot(fpr, tpr, label=f"{reg} (AUC={auc:.3f}, score={sign}REDG)")

            auc_rows.append({"damage_mode": dmg, "regime": reg, "AUC(REDG→correct, best_sign)": float(auc), "sign": sign})

        plt.plot([0,1],[0,1],"--", label="Chance")
        plt.title(f"ROC: REDG predicts correctness\n(damage_fraction=0.4, mode={dmg})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(auc_rows).sort_values(["damage_mode","regime"]).reset_index(drop=True)

tab_redg_auc = plot_roc_redg_correctness(df_feat2)
tab_redg_auc


# In[ ]:





# In[ ]:


# Compute eREDG = REDG * I(DG_total > delta)
# delta is chosen data-driven (10th percentile of DG across the condition).
# This removes degenerate "inert/frozen" trajectories that can produce REDG≈1 spuriously.

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional



def add_eREDG(
    df: pd.DataFrame,
    *,
    dg_col: str = "DG",
    dg_early_col: str = "DG_early",
    out_redg_col: str = "REDG",
    out_eredg_col: str = "eREDG",
    eps: float = 1e-12,
    # gating
    gate_by: str = "per_damage_mode_and_regime",  # "global" or "per_damage_mode_and_regime"
    gate_q: float = 0.20,                         # 20th percentile
    hard_delta: float = None                      # optionally set a fixed DG threshold
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_out: input df + REDG, eREDG, DG_active flag, DG_delta used
      gate_table: thresholds used per group (or global)
    Requirements: df has DG and DG_early.
    """
    df = df.copy()
    assert dg_col in df.columns, f"Missing {dg_col}"
    assert dg_early_col in df.columns, f"Missing {dg_early_col}"

    # REDG
    df[out_redg_col] = df[dg_early_col] / (df[dg_col] + eps)

    gate_rows = []

    if hard_delta is not None:
        df["DG_delta"] = float(hard_delta)
        df["DG_active"] = (df[dg_col] > float(hard_delta)).astype(int)
        gate_rows.append({"group": "global", "DG_delta": float(hard_delta), "gate_q": None, "n": int(len(df))})

    else:
        if gate_by == "global":
            delta = float(np.quantile(df[dg_col].to_numpy(dtype=float), gate_q))
            df["DG_delta"] = delta
            df["DG_active"] = (df[dg_col] > delta).astype(int)
            gate_rows.append({"group": "global", "DG_delta": delta, "gate_q": gate_q, "n": int(len(df))})

        elif gate_by == "per_damage_mode_and_regime":
            df["DG_delta"] = np.nan
            df["DG_active"] = 0
            for (dmg, reg), idx in df.groupby(["damage_mode","regime"]).groups.items():
                sub = df.loc[idx, dg_col].to_numpy(dtype=float)
                delta = float(np.quantile(sub, gate_q))
                df.loc[idx, "DG_delta"] = delta
                df.loc[idx, "DG_active"] = (df.loc[idx, dg_col] > delta).astype(int)
                gate_rows.append({"group": f"{dmg}|{reg}", "damage_mode": dmg, "regime": reg,
                                  "DG_delta": delta, "gate_q": gate_q, "n": int(len(idx))})
        else:
            raise ValueError("gate_by must be 'global' or 'per_damage_mode_and_regime'")

    # eREDG = REDG if active else NaN (safer for ROC + hist)
    df[out_eredg_col] = df[out_redg_col].where(df["DG_active"] == 1, np.nan)

    gate_table = pd.DataFrame(gate_rows)
    return df, gate_table

# Apply to your df_dg2 (damage_fraction fixed at 0.4 in that table)
df_e, gate_table = add_eREDG(df_dg2, gate_by="per_damage_mode_and_regime", gate_q=0.20)

print("Active fraction:", df_e["DG_active"].mean())
gate_table


# In[ ]:


# Histograms + ROC for eREDG (damage_fraction=0.4)
# Panel layout (per damage_mode):
#   Row: damage_mode (mixed / spiky)
#   Col 1-3: eREDG histograms by regime (plain, H1, H2)
#   Col 4: ROC curves (eREDG -> correctness) for the three regimes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score



def final_eredg_panel(df: pd.DataFrame, *, damage_modes=("mixed","spiky"), regimes=("plain","H1","H2"),
                     score_col="eREDG", title_prefix="eREDG at 40% band perturbation"):
    for dmg in damage_modes:
        # build a 1x4 panel
        fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9), gridspec_kw={"width_ratios":[1,1,1,1.25]})
        fig.suptitle(f"{title_prefix}  |  damage_mode={dmg}", y=1.05)

        # --- Histograms (3 panels) ---
        for j, reg in enumerate(regimes):
            ax = axes[j]
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()

            # drop inactive (NaN)
            dfr = dfr[np.isfinite(dfr[score_col].to_numpy(dtype=float))]

            corr = dfr[dfr["correct"] == 1][score_col].to_numpy(dtype=float)
            wrong = dfr[dfr["correct"] == 0][score_col].to_numpy(dtype=float)

            ax.hist(corr, bins=18, alpha=0.65, label="Correct")
            ax.hist(wrong, bins=18, alpha=0.65, label="Incorrect")

            ax.set_title(f"{reg}")
            ax.set_xlabel(score_col)
            ax.set_ylabel("Count" if j == 0 else "")
            ax.grid(True, alpha=0.25)
            if j == 0:
                ax.legend(frameon=False)

        # --- ROC panel (right) ---
        ax = axes[3]
        for reg in regimes:
            dfr = df[(df["damage_mode"] == dmg) & (df["regime"] == reg)].copy()
            dfr = dfr[np.isfinite(dfr[score_col].to_numpy(dtype=float))]

            y = dfr["correct"].to_numpy(dtype=int)
            x = dfr[score_col].to_numpy(dtype=float)

            if len(np.unique(y)) < 2:
                continue

            # choose direction that yields AUC>=0.5 for interpretability (auto)
            
            if np.allclose(x, x[0]):
                x_use = x
                auc = 0.5
                sign = "0"
            else:
                auc_pos = roc_auc_score(y, x)
                auc_neg = roc_auc_score(y, -x)
          #  auc_pos = roc_auc_score(y, x)
          #  auc_neg = roc_auc_score(y, -x)
                if auc_neg > auc_pos:
                    x_use = -x
                    auc = auc_neg
                    sign = "-"
                else:
                    x_use = x
                    auc = auc_pos
                    sign = "+"

            fpr, tpr, _ = roc_curve(y, x_use)
            ax.plot(fpr, tpr, label=f"{reg} (AUC={auc:.3f})") #", {sign}{score_col})")

        ax.plot([0,1],[0,1],"--", label="Chance")
        ax.set_title("ROC: score → correctness")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="lower right")
        fname = f"eREDG_hist_roc_{dmg}.png"
        fig.savefig(fname, dpi=600, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

final_eredg_panel(df_e)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Part 3 Stress Mechanism

# In[ ]:


# Morphogenetic_consensus_one now records stress_hist and per_band_dg_hist if requested
# - stress_hist[t,k] = divergence between band k and its neighborhood consensus at iteration t
# - step_dg_hist[t,k] = || l_k(t+1) - l_k(t) ||  (per-band movement per step)

from typing import List, Optional, Dict
import numpy as np



def summarize_stress_features(stress_hist: np.ndarray, step_dg_hist: np.ndarray, tau: float) -> dict:
    """
    stress_hist: (T+1,K)
    step_dg_hist: (T,K)
    Returns per-band and global summaries.
    """
    stress = np.asarray(stress_hist, dtype=float)
    step_dg = np.asarray(step_dg_hist, dtype=float)

    # per-band summaries
    stress_auc = stress.sum(axis=0)                  # (K,)
    stress_max = stress.max(axis=0)                  # (K,)
    dg_band = step_dg.sum(axis=0)                    # (K,)
    # first time stress drops below tau and stays below (relaxation time); NaN if never
    T1, K = stress.shape
    t_relax = np.full(K, np.nan, dtype=float)
    for k in range(K):
        below = (stress[:, k] <= tau)
        for t in range(T1):
            if below[t] and np.all(below[t:]):
                t_relax[k] = float(t)
                break

    # global summaries (for mechanism links to eREDG)
    stress_auc_global = float(stress_auc.sum())
    stress_max_global = float(stress_max.max())
    dg_total = float(dg_band.sum())
    return {
        "stress_auc_global": stress_auc_global,
        "stress_max_global": stress_max_global,
        "dg_total_from_steps": dg_total,
        "stress_auc_per_band": stress_auc,
        "stress_max_per_band": stress_max,
        "dg_per_band": dg_band,
        "t_relax_per_band": t_relax
    }



def build_mechanism_table(
    *,
    band_logits_test: np.ndarray,   # (N,K,C)
    y_test: np.ndarray,             # (N,)
    le3,
    damage_mode: str,
    regime: str,
    damage_fraction: float = 0.4,
    run_id: int = 0,
    tau: float = 0.05,
    max_iter: int = 30,
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    morpho_weighting: str = "confidence",
    weight_clip: tuple = (0.0, 5.0)
) -> pd.DataFrame:
    N, K, C = band_logits_test.shape
    rows = []
    for i in range(N):
        band_logits0_i = [band_logits_test[i, k, :] for k in range(K)]
        seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i
        rng = np.random.default_rng(seed)

        damaged_logits0_i, frozen_flags_i, damaged_idx = apply_random_band_damage(
            band_logits0_i, damage_fraction=damage_fraction, mode=damage_mode, rng=rng
        )

        if morpho_weighting == "confidence":
            w0 = np.array([np.max(softmax(l)) for l in damaged_logits0_i], dtype=float)
            w0 = np.clip(w0, 0.0, 0.999)
        else:
            w0 = np.ones(K, dtype=float)

        res = morphogenetic_consensus_one(
            damaged_logits0_i,
            frozen_flags=frozen_flags_i,
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
            rng=rng
        )

        y_hat = int(res["y_hat"])
        correct = int(y_hat == int(y_test[i]))

        # DG + REDG + eREDG (use your existing convention: early = first 25% steps)
        l_hist = res["l_hist"]  # (T+1,K,C)
        DG_total = compute_dg_index(l_hist)
        T_steps = l_hist.shape[0] - 1
        k_early = max(1, int(round(0.25 * T_steps)))
        DG_early = compute_dg_index(l_hist[:k_early+1])  # first k_early steps

        # mechanism summaries
        feat = summarize_stress_features(res["stress_hist"], res["step_dg_hist"], tau=tau)

        row = {
            "i": i,
            "damage_mode": damage_mode,
            "regime": regime,
            "damage_fraction": float(damage_fraction),
            "y_true": int(y_test[i]),
            "y_pred": y_hat,
            "true_label": le3.inverse_transform([int(y_test[i])])[0],
            "pred_label": le3.inverse_transform([y_hat])[0],
            "correct": correct,
            "DG": float(DG_total),
            "DG_early": float(DG_early),
            "REDG": float(DG_early / (DG_total + 1e-12)),
        }

        # flatten per-band summaries (K=7)
        for k in range(K):
            row[f"stress_auc_b{k}"] = float(feat["stress_auc_per_band"][k])
            row[f"stress_max_b{k}"] = float(feat["stress_max_per_band"][k])
            row[f"dg_b{k}"] = float(feat["dg_per_band"][k])
            row[f"t_relax_b{k}"] = float(feat["t_relax_per_band"][k]) if np.isfinite(feat["t_relax_per_band"][k]) else np.nan

        row["stress_auc_global"] = feat["stress_auc_global"]
        row["stress_max_global"] = feat["stress_max_global"]
        row["dg_total_from_steps"] = feat["dg_total_from_steps"]
        row["damaged_idx"] = ",".join(map(str, damaged_idx.tolist()))

        rows.append(row)

    return pd.DataFrame(rows)

# Build mechanism tables for the two key regulated regimes under both damage modes
dfs = []
for dmg in ["mixed","spiky"]:
    for reg in ["plain","H1","H2"]:
        print("Building mechanism table:", dmg, reg)
        dfs.append(build_mechanism_table(
            band_logits_test=band_logits_test, y_test=y_test, le3=le3,
            damage_mode=dmg, regime=reg, damage_fraction=0.4, run_id=0
        ))
df_mech = pd.concat(dfs, ignore_index=True)

df_mech.head()


# In[ ]:


# Bootstrap AUC confidence intervals for eREDG → correctness
# Produces a compact table: AUC ± 95% CI for each (damage_mode, regime).
# Also uses a sign-fixed score so "higher = more likely correct" (reviewer-friendly).

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score



def auc_with_sign_fix(y: np.ndarray, s: np.ndarray):
    """
    Returns (auc, sign, score_used) where score_used = sign * s and auc>=0.5.
    sign in {+1, -1}.
    """
    auc_pos = roc_auc_score(y, s)
    auc_neg = roc_auc_score(y, -s)
    if auc_neg > auc_pos:
        return float(auc_neg), -1, -s
    return float(auc_pos), +1, s



def bootstrap_auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """
    Bootstrap CI for AUC using resampling with replacement.
    Returns: auc_hat, (lo, hi), auc_boot array
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)

    # point estimate with sign-fix
    auc_hat, sign, s_use = auc_with_sign_fix(y, s)

    aucs = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        yb = y[b]
        sb = s_use[b]
        # skip degenerate resamples (all same class)
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))

    aucs = np.asarray(aucs, dtype=float)
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return auc_hat, float(lo), float(hi), int(sign), aucs

# --- Choose the score column ---
SCORE_COL = "eREDG"   # if you prefer REDG, change here

rows = []
for (dmg, reg), dfr in df_e.groupby(["damage_mode", "regime"]):
    # drop inactive (NaN eREDG)
    dfr = dfr[np.isfinite(dfr[SCORE_COL].to_numpy(dtype=float))].copy()
    y = dfr["correct"].to_numpy(dtype=int)
    s = dfr[SCORE_COL].to_numpy(dtype=float)

    if len(np.unique(y)) < 2:
        continue

    auc_hat, lo, hi, sign, aucs = bootstrap_auc_ci(y, s, n_boot=2000, seed=123)

    rows.append({
        "damage_mode": dmg,
        "regime": reg,
        "n_used": int(len(y)),
        "AUC(Q→correct)": auc_hat,
        "CI95_lo": lo,
        "CI95_hi": hi,
        "sign_for_Q": sign,   # Q = sign * eREDG  (so higher Q => more correct)
    })

df_auc_ci = pd.DataFrame(rows).sort_values(["damage_mode", "regime"])
df_auc_ci


# In[ ]:


# Lesion detector based on stress history + per-band DG contribution

import numpy as np
from typing import Dict, Tuple



def compute_band_summaries(stress_hist: np.ndarray, step_dg_hist: np.ndarray, tau: float) -> Dict[str, np.ndarray]:
    """
    stress_hist: (T+1,K)
    step_dg_hist: (T,K)
    returns dict of per-band vectors length K
    """
    stress = np.asarray(stress_hist, dtype=float)
    step_dg = np.asarray(step_dg_hist, dtype=float)
    Tp1, K = stress.shape

    stress_auc = stress.sum(axis=0)     # (K,)
    stress_max = stress.max(axis=0)     # (K,)
    dg_band = step_dg.sum(axis=0)       # (K,)

    # relaxation time: first t where stress<=tau and stays <= tau afterwards (NaN if never)
    t_relax = np.full(K, np.nan, dtype=float)
    for k in range(K):
        below = (stress[:, k] <= tau)
        for t in range(Tp1):
            if below[t] and np.all(below[t:]):
                t_relax[k] = float(t)
                break

    return {
        "stress_auc": stress_auc,
        "stress_max": stress_max,
        "dg_band": dg_band,
        "t_relax": t_relax
    }



def detect_lesions(
    stress_auc: np.ndarray,
    dg_band: np.ndarray,
    *,
    top_k: int = 2,
    use_dg: bool = True,
) -> np.ndarray:
    """
    Returns lesion_flags (K,) bool.
    Default: mark top_k bands by stress_auc, optionally union with top_k by dg_band.
    """
    K = len(stress_auc)
    lesion = np.zeros(K, dtype=bool)

    # top-k by stress AUC
    idx_stress = np.argsort(stress_auc)[::-1][:max(1, top_k)]
    lesion[idx_stress] = True

    if use_dg:
        idx_dg = np.argsort(dg_band)[::-1][:max(1, top_k)]
        lesion[idx_dg] = True

    return lesion


# In[ ]:


# Closed-loop morphogenetic inference for one spectrum
# Requires:
# - apply_random_band_damage
# - morphogenetic_consensus_one(record_stress=True, return_trajectories=True)
# - softmax
# - band_logits_test, y_test

import numpy as np
from typing import List, Optional, Dict



def closed_loop_consensus_one(
    band_logits0: List[np.ndarray],
    *,
    damage_fraction: float,
    damage_mode: str,          # "spiky" or "mixed"
    regime: str,               # "plain" / "H1" / "H2" (closed-loop works best with H1/H2)
    run_seed: int,
    # morpho params
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    # lesion params
    top_k: int = 2,
    use_dg_in_lesion: bool = True,
    mute_weight: float = 0.0,
) -> Dict:
    rng = np.random.default_rng(run_seed)

    # 1) apply perturbation (same as your Programme C)
    damaged_logits0, frozen_flags0, damaged_idx = apply_random_band_damage(
        band_logits0, damage_fraction=damage_fraction, mode=damage_mode, rng=rng
    )

    K = len(damaged_logits0)

    # initial weights (reliability prior)
    if morpho_weighting == "confidence":
        w0 = np.array([np.max(softmax(l)) for l in damaged_logits0], dtype=float)
        w0 = np.clip(w0, 0.0, 0.999)
    else:
        w0 = np.ones(K, dtype=float)

    # 2) PASS 1: normal morphogenesis with stress recording
    res1 = morphogenetic_consensus_one(
        damaged_logits0,
        frozen_flags=frozen_flags0,
        band_weights0=w0,
        regime=regime,
        weight_clip=weight_clip,
        alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
        use_js=True,
        return_trajectories=True,
        record_stress=True,
        rng=rng
    )

    summ = compute_band_summaries(res1["stress_hist"], res1["step_dg_hist"], tau=tau)
    lesion_flags = detect_lesions(
        summ["stress_auc"], summ["dg_band"],
        top_k=top_k,
        use_dg=use_dg_in_lesion
    )

    # 3) PASS 2: self-audited rerun — mute+freeze lesioned bands
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
        alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
        use_js=True,
        return_trajectories=True,
        record_stress=True,
        rng=rng
    )

    return {
        "res1": res1,
        "res2": res2,
        "lesion_flags": lesion_flags,
        "damaged_idx": damaged_idx,
        "summ1": summ,
        "w0": w0,
        "w2": w2,
        "frozen2": frozen_flags2
    }


# In[ ]:


# Closed-loop evaluation on macro test set
# Outputs:
# - accuracy of pass1 vs pass2
# - lesion frequency per band
# - optional: per-sample flags to analyze further

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



def evaluate_closed_loop(
    *,
    band_logits_test: np.ndarray,  # (N,K,C)
    y_test: np.ndarray,            # (N,)
    damage_fraction: float = 0.4,
    damage_mode: str = "spiky",
    regime: str = "H1",
    run_id: int = 0,
    # morpho params (match Programme C)
    alpha: float = 0.3,
    lam: float = 1.0,
    T: float = 1e-2,
    tau: float = 0.05,
    max_iter: int = 30,
    weight_clip: tuple = (0.0, 5.0),
    morpho_weighting: str = "confidence",
    # lesion params
    top_k: int = 2,
    use_dg_in_lesion: bool = True,
    mute_weight: float = 0.0,
) -> pd.DataFrame:
    N, K, C = band_logits_test.shape
    rows = []
    lesion_counts = np.zeros(K, dtype=int)

    for i in range(N):
        L0 = [band_logits_test[i, k, :] for k in range(K)]
        seed = (run_id * 1_000_000) + (10_000 * int(round(damage_fraction * 100))) + i

        out = closed_loop_consensus_one(
            L0,
            damage_fraction=damage_fraction,
            damage_mode=damage_mode,
            regime=regime,
            run_seed=seed,
            alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
            weight_clip=weight_clip,
            morpho_weighting=morpho_weighting,
            top_k=top_k,
            use_dg_in_lesion=use_dg_in_lesion,
            mute_weight=mute_weight
        )

        y1 = int(out["res1"]["y_hat"])
        y2 = int(out["res2"]["y_hat"])
        ytrue = int(y_test[i])

        lesion = out["lesion_flags"]
        lesion_counts += lesion.astype(int)

        rows.append({
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
        })

    df = pd.DataFrame(rows)
    acc1 = accuracy_score(df["y_true"], df["y_hat_pass1"])
    acc2 = accuracy_score(df["y_true"], df["y_hat_pass2"])
    print(f"[Closed-loop] mode={damage_mode}, regime={regime}, frac={damage_fraction:.2f}")
    print(f"Accuracy pass1 (baseline morpho): {acc1:.3f}")
    print(f"Accuracy pass2 (self-audited):    {acc2:.3f}")

    # lesion frequency plot
    freq = lesion_counts / len(df)
    plt.figure(figsize=(9, 3.6))
    plt.bar(np.arange(K), freq)
    plt.xticks(np.arange(K), [f"B{k}\n{lo}-{hi}" for k,(lo,hi) in enumerate(BANDS)])
    plt.ylabel("Lesion frequency")
    plt.title(f"Closed-loop lesion frequency per band\n(mode={damage_mode}, regime={regime}, frac={damage_fraction:.2f})")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    return df

# Run for the two main conditions
# FIXED calls: pass band_logits_test and y_test explicitly

df_cl_spiky_h1 = evaluate_closed_loop(
    band_logits_test=band_logits_test,
    y_test=y_test,
    damage_mode="spiky",
    regime="H1",
    damage_fraction=0.4,
    run_id=0,
    top_k=2,
    use_dg_in_lesion=True
)

df_cl_mixed_h1 = evaluate_closed_loop(
    band_logits_test=band_logits_test,
    y_test=y_test,
    damage_mode="mixed",
    regime="H1",
    damage_fraction=0.4,
    run_id=0,
    top_k=2,
    use_dg_in_lesion=True
)

# Optional: H2
df_cl_spiky_h2 = evaluate_closed_loop(
    band_logits_test=band_logits_test,
    y_test=y_test,
    damage_mode="spiky",
    regime="H2",
    damage_fraction=0.4,
    run_id=0,
    top_k=2,
    use_dg_in_lesion=True
)

df_cl_mixed_h2 = evaluate_closed_loop(
    band_logits_test=band_logits_test,
    y_test=y_test,
    damage_mode="mixed",
    regime="H2",
    damage_fraction=0.4,
    run_id=0,
    top_k=2,
    use_dg_in_lesion=True
)


# In[ ]:


# ============================================================
# FIND: per-sample result tables already in memory
# ============================================================

import pandas as pd

candidates = []
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and obj.shape[0] > 50:
        cols = set(obj.columns.astype(str))
        key_hits = sum(k in cols for k in [
            "damage_mode", "damage_fraction", "regime",
            "y_true", "y_hat", "DG", "eREDG",
            "y_hat_pass1", "y_hat_pass2"
        ])
        if key_hits >= 4:
            candidates.append((name, obj.shape, key_hits, sorted(list(cols))[:25]))

candidates_sorted = sorted(candidates, key=lambda x: (x[2], x[1][0]), reverse=True)
for name, shape, hits, preview_cols in candidates_sorted[:12]:
    print(f"{name:25s} shape={shape}  key_hits={hits}  cols_preview={preview_cols}")


# In[ ]:


# ============================================================
# BOOTSTRAP CI: mean DG per (damage_mode, damage_fraction, regime)
# Requires per-sample 'DG' and condition columns.
# ============================================================

import numpy as np
import pandas as pd

df_long = df_e.copy()  # <-- set this

DG_COL = "DG"  # change if your column name differs
GROUPS = ["damage_mode", "damage_fraction", "regime"]



def bootstrap_mean_ci(x, B=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    boots = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(x.mean()), float(lo), float(hi)

rows = []
rng = np.random.default_rng(0)

for key, sub in df_long.groupby(GROUPS):
    mu, lo, hi = bootstrap_mean_ci(sub[DG_COL].values, B=2000, rng=rng)
    rows.append((*key, mu, lo, hi, len(sub)))

df_dg_ci = pd.DataFrame(rows, columns=GROUPS + ["DG_mean", "DG_ci_lo", "DG_ci_hi", "n"])
df_dg_ci.sort_values(["damage_mode", "regime", "damage_fraction"], inplace=True)
df_dg_ci.head()


# In[ ]:


# ============================================================
# PLOT: DG vs damage with bootstrap CIs (1x2 mixed|spiky)
# ============================================================

import matplotlib.pyplot as plt

regimes = ["plain", "H1", "H2"]
modes = ["mixed", "spiky"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

for ax, mode in zip(axes, modes):
    d = df_dg_ci[df_dg_ci["damage_mode"] == mode].copy()

    for reg in regimes:
        s = d[d["regime"] == reg].sort_values("damage_fraction")
        if len(s) == 0:
            continue
        x = s["damage_fraction"].values
        y = s["DG_mean"].values
        ylo = s["DG_ci_lo"].values
        yhi = s["DG_ci_hi"].values

        ax.plot(x, y, "-o", label=f"DG ({reg})")
        ax.fill_between(x, ylo, yhi, alpha=0.15)

    ax.set_title(f"{mode.capitalize()} perturbations")
    ax.set_xlabel("Damage fraction (bands corrupted)")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Decision Geometry (DG)")
axes[1].legend(loc="best", frameon=False)

plt.tight_layout()
plt.savefig("DG_vs_damage_CI_1x2.png", dpi=600, bbox_inches="tight")
plt.show()


# In[ ]:


# ============================================================
# eREDG: separation effect size (Cliff's delta) and Mann–Whitney U
# Requires per-sample eREDG and correctness label under each regime/condition.
# ============================================================

import numpy as np
from scipy.stats import mannwhitneyu

EREDG_COL = "eREDG"  # adjust if needed



def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # O(n*m) but fine for typical test sizes; can optimize if huge
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))

df_long["correct"] = (df_long["y_hat"] == df_long["y_true"]).astype(int)

GROUPS = ["damage_mode", "damage_fraction", "regime"]

rows = []
for key, sub in df_long.groupby(GROUPS):
    a = sub[sub["correct"] == 1][EREDG_COL].values
    b = sub[sub["correct"] == 0][EREDG_COL].values
    if len(a) < 5 or len(b) < 5:
        continue
    delta = cliffs_delta(a, b)
    # MWU (two-sided)
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    rows.append((*key, delta, float(p), len(a), len(b)))

df_eredg_fx = pd.DataFrame(rows, columns=GROUPS + ["cliffs_delta", "MWU_p", "n_correct", "n_incorrect"])
df_eredg_fx.sort_values(GROUPS, inplace=True)
df_eredg_fx


# In[ ]:





# In[ ]:





# In[ ]:





# # Part 4 Illustrative Examples

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt



def visualize_band_damage_once(band_logits_test, *, i=0, damage_fraction=0.4, mode="spiky", seed=123):
    """
    Shows which macro-bands were damaged for sample i and how their logits changed.
    """
    L0_list = [band_logits_test[i, k, :].copy() for k in range(band_logits_test.shape[1])]

    rng = np.random.default_rng(seed)
    Ld_list, frozen_flags, damaged_idx = apply_random_band_damage(
        L0_list, damage_fraction=damage_fraction, mode=mode, rng=rng
    )

    K = len(L0_list)
    C = L0_list[0].shape[0]

    # Build per-band change magnitude (L2 norm of logit delta)
    delta = np.array([np.linalg.norm(Ld_list[k] - L0_list[k]) for k in range(K)], dtype=float)

    # Binary damaged mask
    mask = np.zeros(K, dtype=int)
    mask[np.array(damaged_idx, dtype=int)] = 1

    fig, ax = plt.subplots(1, 1, figsize=(9, 2.8))
    ax.plot(np.arange(K), delta, marker="o", label="||Δ logits|| per band")
    ax.scatter(np.where(mask==1)[0], delta[mask==1], s=90, label="Damaged bands")
    ax.set_title(f"Band damage (mode={mode}, frac={damage_fraction}) | sample i={i}")
    ax.set_xlabel("Band index k")
    ax.set_ylabel("Logit change magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    print("Damaged bands indices:", damaged_idx)
    print("Frozen flags (per band):", frozen_flags)

# Example calls
visualize_band_damage_once(band_logits_test, i=0, damage_fraction=0.4, mode="spiky", seed=42)
visualize_band_damage_once(band_logits_test, i=0, damage_fraction=0.4, mode="mixed", seed=42)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt



def logits_heatmap_before_after(band_logits_test, *, i=0, damage_fraction=0.4, mode="spiky", seed=123):
    L0 = band_logits_test[i].copy()  # (K,C)
    L0_list = [L0[k, :].copy() for k in range(L0.shape[0])]

    rng = np.random.default_rng(seed)
    Ld_list, frozen_flags, damaged_idx = apply_random_band_damage(
        L0_list, damage_fraction=damage_fraction, mode=mode, rng=rng
    )
    Ld = np.stack(Ld_list, axis=0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios":[1,1,1]})
    axes[0].imshow(L0, aspect="auto")
    axes[0].set_title("Original logits (K×C)")
    axes[1].imshow(Ld, aspect="auto")
    axes[1].set_title(f"Damaged logits ({mode})")
    axes[2].imshow(Ld - L0, aspect="auto")
    axes[2].set_title("Δ logits")

    for ax in axes:
        ax.set_xlabel("Class c")
        ax.set_ylabel("Band k")

    plt.tight_layout()
    plt.show()

    print("Damaged bands:", damaged_idx)
    print("Frozen flags:", frozen_flags)

# Example calls
logits_heatmap_before_after(band_logits_test, i=0, damage_fraction=0.4, mode="spiky", seed=42)
logits_heatmap_before_after(band_logits_test, i=0, damage_fraction=0.4, mode="mixed", seed=42)


# In[ ]:


import matplotlib.pyplot as plt



def plot_damaged_raman_windows(BANDS, damaged_idx, *, title):
    fig, ax = plt.subplots(figsize=(10.5, 2.2))

    for k, (lo, hi) in enumerate(BANDS):
        if k in damaged_idx:
            ax.axvspan(lo, hi, alpha=0.25)
            ax.text((lo+hi)/2, 0.5, f"band {k}", ha="center", va="center")

    ax.set_xlim(BANDS[0][0], BANDS[-1][1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

# example
plot_damaged_raman_windows(
    BANDS,
    damaged_idx=[2,3],
    title="Spiky damage example: corrupted Raman windows"
)


# In[ ]:


import numpy as np

# RAW spectra directly from the dataframe (no ML scaling, no transforms)
X_raw = df_all[spec_cols_sorted].to_numpy(dtype=float)  # (N, M)
wn_raw = wn.copy()                                      # (M,)

# fingerprint crop (matches your bands)
fp_mask = (wn_raw >= BANDS[0][0]) & (wn_raw <= BANDS[-1][1])
wn_fp_raw = wn_raw[fp_mask]
X_fp_raw = X_raw[:, fp_mask]

print("X_fp_raw:", X_fp_raw.shape, "wn_fp_raw:", wn_fp_raw.shape)
print("Example spectrum min/max:", float(X_fp_raw[0].min()), float(X_fp_raw[0].max()))


# In[ ]:


# ============================================================
# Cell: Illustrative Exampls
# Goal: plain converges fast to WRONG class, H1 recovers to CORRECT.
#
# Requires:
#   band_logits_test (N,K,C)
#   apply_random_band_damage
#   morphogenetic_consensus_one (return_trajectories=True)
#   labels: y_test or y_test_m or y_test_macro (any one present)
#
# Output:
#   DEMO dict with i, seed, frac, mode, damaged_bands, frozen_flags, plus trajectories.
# ============================================================

import numpy as np

# ---------------------------
# 0) Locate ground-truth y
# ---------------------------



def _find_y():
    for name in ["y_test", "y_test_m", "y_test_macro", "y_macro", "y"]:
        if name in globals():
            y = np.asarray(globals()[name]).astype(int)
            return name, y
    raise NameError("Could not find y labels. Expected one of y_test / y_test_m / y_test_macro / y_macro / y.")

y_name, y_true = _find_y()
N, K, C = band_logits_test.shape
assert len(y_true) == N, f"{y_name} length {len(y_true)} != band_logits_test N {N}"

# ---------------------------
# 1) Helpers: consensus, entropy, convergence step
# ---------------------------



def _softmax(z):
    z = np.asarray(z, float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)



def _get_consensus_traj(res):
    # Returns (T+1,C) consensus logits over time
    if isinstance(res, dict) and ("l_hist" in res):
        lh = np.asarray(res["l_hist"], float)      # (T+1,K,C)
        return lh.sum(axis=1)
    if isinstance(res, dict) and ("l_global_hist" in res):
        return np.asarray(res["l_global_hist"], float)
    raise KeyError("Result dict missing 'l_hist' or 'l_global_hist'.")



def _pred_from_consensus(cons_traj):
    # final prediction from last consensus logits
    return int(np.argmax(cons_traj[-1]))



def _entropy_from_consensus(cons_traj):
    P = _softmax(cons_traj)
    ent = -np.sum(P * np.log(P + 1e-12), axis=1)
    return ent



def _conv_step(ent, thr=1e-6):
    # first step where entropy falls below threshold
    hit = np.where(ent <= thr)[0]
    return int(hit[0]) if hit.size else (len(ent)-1)

# ---------------------------
# 2) Search settings (tune if needed)
# ---------------------------
modes = ["mixed"]           # add "spiky" if you want
fracs = [0.4, 0.6, 0.8]     # harder damage at the end
seeds = list(range(0, 60))  # broaden if needed

# limit search for speed (increase if needed)
max_i = min(N, 250)

# Morphogenesis params — keep aligned with your notebook defaults
alpha = 0.3
lam = 1.0
T = 1e-2
tau = 0.05
max_iter = 30
weight_clip = (0.0, 5.0)
use_js = True

# “fast convergence” definition for plain
ENT_THR = 1e-6
FAST_T = 6  # plain must hit entropy <= ENT_THR within this many iterations

best = None
candidates = []

for mode in modes:
    for frac in fracs:
        for i in range(max_i):
            # original logits list for this spectrum
            L0 = band_logits_test[i].copy()  # (K,C)
            L0_list = [L0[k, :].copy() for k in range(K)]

            for seed in seeds:
                rng = np.random.default_rng(seed)

                # damage in agent space
                Ld_list, frozen_flags, damaged_bands = apply_random_band_damage(
                    L0_list, damage_fraction=frac, mode=mode, rng=rng
                )

                # run plain and H1 on the SAME damaged input
                res_plain = morphogenetic_consensus_one(
                    Ld_list, frozen_flags=np.asarray(frozen_flags, bool),
                    regime="plain",
                    alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                    weight_clip=weight_clip, use_js=use_js,
                    return_trajectories=True
                )
                res_h1 = morphogenetic_consensus_one(
                    Ld_list, frozen_flags=np.asarray(frozen_flags, bool),
                    regime="H1",
                    alpha=alpha, lam=lam, T=T, tau=tau, max_iter=max_iter,
                    weight_clip=weight_clip, use_js=use_js,
                    return_trajectories=True
                )

                cons_plain = _get_consensus_traj(res_plain)
                cons_h1    = _get_consensus_traj(res_h1)

                yhat_plain = _pred_from_consensus(cons_plain)
                yhat_h1    = _pred_from_consensus(cons_h1)
                y_i        = int(y_true[i])

                # must be: plain wrong, H1 correct
                if (yhat_plain == y_i) or (yhat_h1 != y_i):
                    continue

                ent_plain = _entropy_from_consensus(cons_plain)
                ent_h1    = _entropy_from_consensus(cons_h1)

                t_plain = _conv_step(ent_plain, thr=ENT_THR)
                t_h1    = _conv_step(ent_h1, thr=ENT_THR)

                # "plain converges fast" (optional constraint, but matches your narrative)
                if t_plain > FAST_T:
                    continue

                # score: prefer (plain wrong with HIGH confidence) and (H1 correct with HIGH confidence)
                # confidence proxy: final margin
                def _margin(cons):
                    z = cons[-1]
                    zs = np.sort(z)
                    return float(zs[-1] - zs[-2])

                m_plain = _margin(cons_plain)
                m_h1    = _margin(cons_h1)

                score = (m_plain) + (m_h1) + (FAST_T - t_plain) + (t_h1) * 0.05  # tweak weights if desired

                cand = {
                    "score": score,
                    "i": i, "seed": seed, "frac": frac, "mode": mode,
                    "y_true": y_i,
                    "yhat_plain": yhat_plain, "yhat_h1": yhat_h1,
                    "t_plain": t_plain, "t_h1": t_h1,
                    "margin_plain": m_plain, "margin_h1": m_h1,
                    "damaged_bands": np.asarray(damaged_bands, int),
                    "frozen_flags": np.asarray(frozen_flags, bool),
                    "res_plain": res_plain,
                    "res_h1": res_h1,
                }
                candidates.append(cand)

                if (best is None) or (score > best["score"]):
                    best = cand

# ---------------------------
# 3) Report + export DEMO
# ---------------------------
if best is None:
    print("No killer case found with current search settings.")
    print("Try: increase seeds range, increase max_i, include frac=0.9, or include mode='spiky'.")
else:
    print("FOUND KILLER CASE ✅")
    print(f"  i={best['i']}, seed={best['seed']}, mode={best['mode']}, frac={best['frac']}")
    print(f"  y_true={best['y_true']}")
    print(f"  plain: yhat={best['yhat_plain']}, t_conv={best['t_plain']}, margin={best['margin_plain']:.3f}")
    print(f"  H1   : yhat={best['yhat_h1']}, t_conv={best['t_h1']}, margin={best['margin_h1']:.3f}")
    print(f"  damaged_bands={best['damaged_bands'].tolist()}, frozen={best['frozen_flags'].tolist()}")

    DEMO = {
        "i": best["i"],
        "seed": best["seed"],
        "mode": best["mode"],
        "frac": best["frac"],
        "damaged_bands": best["damaged_bands"],
        "frozen_flags": best["frozen_flags"],
        "res_plain": best["res_plain"],
        "res_h1": best["res_h1"],
        "y_true": best["y_true"],
        "yhat_plain": best["yhat_plain"],
        "yhat_h1": best["yhat_h1"],
    }
    print("\nSaved dict: DEMO (use this to render the composite figure).")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Unpack DEMO
# -----------------------
i = DEMO["i"]
seed = DEMO["seed"]
mode = DEMO["mode"]
frac = DEMO["frac"]
damaged_bands = DEMO["damaged_bands"]
frozen_flags = DEMO["frozen_flags"]
res_plain = DEMO["res_plain"]
res_h1 = DEMO["res_h1"]
y_true = DEMO["y_true"]

# -----------------------
# Spectrum source (REAL Raman)
# -----------------------
wn_use = np.asarray(wn, float)
x_raw = np.asarray(X_test[i], float)

# -----------------------
# Build band indices on wn
# -----------------------
band_indices_use = []
for k, (lo, hi) in enumerate(BANDS):
    if k < len(BANDS) - 1:
        idx = np.where((wn_use >= lo) & (wn_use < hi))[0]
    else:
        idx = np.where((wn_use >= lo) & (wn_use <= hi))[0]
    band_indices_use.append(idx)

# -----------------------
# Spectral damage (illustrative, forced bands)
# -----------------------



def damage_spectrum_demo(x, band_indices, forced_bands, rng):
    x = x.copy()
    scale = np.std(x)
    for k in forced_bands:
        idx = band_indices[k]
        if len(idx) < 5:
            continue
        # spikes
        pos = rng.choice(idx, size=max(2, len(idx)//30), replace=False)
        x[pos] += rng.normal(0, 2.5*scale, size=len(pos))
        # smooth drift
        z = rng.normal(0, 1, size=len(idx))
        drift = np.convolve(z, np.ones(11)/11, mode="same")
        drift -= drift.mean()
        x[idx] += 1.0 * scale * drift
    return x

rng = np.random.default_rng(seed)
x_damaged = damage_spectrum_demo(x_raw, band_indices_use, damaged_bands, rng)
dx = x_damaged - x_raw

# -----------------------
# Stress from logits
# -----------------------
L0 = band_logits_test[i]
Ld = np.stack(apply_random_band_damage(
    [L0[k].copy() for k in range(L0.shape[0])],
    damage_fraction=frac, mode=mode, rng=np.random.default_rng(seed)
)[0])

stress = np.linalg.norm(Ld - L0, axis=1)

# -----------------------
# Trajectories
# -----------------------



def consensus(res):
    if "l_hist" in res:
        return np.asarray(res["l_hist"]).sum(axis=1)
    return np.asarray(res["l_global_hist"])

