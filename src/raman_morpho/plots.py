"""Plotting helpers (keep matplotlib code out of core algorithms)."""
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_vs_damage(df_res_all: pd.DataFrame, *, out_dir: str = "outputs", fname_prefix: str = "Robustness"):
    os.makedirs(out_dir, exist_ok=True)
    regimes = sorted(df_res_all["regime"].unique())
    damage_modes = sorted(df_res_all["damage_mode"].unique())

    for dmg in damage_modes:
        plt.figure(figsize=(9, 5))
        dfd = df_res_all[df_res_all["damage_mode"] == dmg].copy()

        df0 = dfd[dfd["regime"] == regimes[0]].copy()
        plt.plot(df0["damage_fraction"], df0["acc_full"],  "-o", label="Full model (oracle)")
        plt.plot(df0["damage_fraction"], df0["acc_naive"], "-o", label="Naive band avg")

        for reg in regimes:
            dfr = dfd[dfd["regime"] == reg]
            plt.plot(dfr["damage_fraction"], dfr["acc_morpho"], "-o", label=f"Morphogenetic ({reg})")

        plt.xlabel("Damage fraction (bands corrupted)")
        plt.ylabel("Accuracy")
        plt.title(f"Robustness under band perturbations – damage_mode={dmg}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_prefix}-{dmg}.png"), dpi=600, bbox_inches="tight")
        plt.close()
