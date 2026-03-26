from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from morphoraman.bands import build_band_indices
from morphoraman.config import ExperimentConfig, set_global_seed
from morphoraman.data import load_cells_raman_dataset, prepare_three_class_dataset, split_dataset
from morphoraman.evaluation import (
    add_eredg,
    add_redg_features,
    auc_by_group,
    bootstrap_auc_ci,
    compute_dg_and_stress_table,
)
from morphoraman.io_utils import ensure_dir, save_dataframe
from morphoraman.models import fit_band_models
from morphoraman.plotting import (
    final_eredg_panel,
    plot_dg_distributions,
    plot_redg_distributions,
    plot_roc_redg_correctness,
    plot_roc_wrong,
    savefig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", default="outputs/eredg")
    args = parser.parse_args()

    cfg = ExperimentConfig(base_dir=args.data_dir)
    set_global_seed(cfg.seed)
    outdir = ensure_dir(args.outdir)

    df_all = load_cells_raman_dataset(cfg.base_dir, expected_points=cfg.expected_points)
    bundle = prepare_three_class_dataset(df_all)
    band_indices = build_band_indices(bundle.wn, list(cfg.bands))

    x_train, x_test, y_train, y_test = split_dataset(bundle.X, bundle.y, test_size=cfg.test_size, seed=cfg.seed)
    band_logits_test, _ = fit_band_models(x_train, y_train, x_test, band_indices, seed=cfg.seed)

    df_dg2 = compute_dg_and_stress_table(
        band_logits_test=band_logits_test,
        y_test=y_test,
        damage_fraction=0.4,
        damage_modes=("mixed", "spiky"),
        regimes=("plain", "H1", "H2"),
        run_id=0,
        alpha=cfg.alpha,
        lam=cfg.lam,
        T=cfg.temperature,
        tau=cfg.tau,
        max_iter=cfg.max_iter,
        weight_clip=cfg.weight_clip,
        morpho_weighting=cfg.morpho_weighting,
        k_early=3,
    )
    save_dataframe(df_dg2, outdir / "dg_stress_table.csv")

    df_feat = df_dg2.copy()
    eps = 1e-9
    df_feat["DG_eff"] = df_feat["DG"] / (df_feat["stress0_mean"] + eps)
    df_feat["DG_early_eff"] = df_feat["DG_early"] / (df_feat["stress0_mean"] + eps)
    save_dataframe(df_feat, outdir / "dg_feature_table.csv")

    metric_rank = []
    for col in ["DG", "DG_eff", "DG_early", "DG_early_eff"]:
        tab = auc_by_group(df_feat, score_col=col)
        mean_auc = float(np.nanmean(tab[f"AUC({col}→wrong)"].to_numpy()))
        metric_rank.append({"metric": col, "mean_auc": mean_auc})
    df_metric_rank = pd.DataFrame(metric_rank).sort_values("mean_auc", ascending=False)
    save_dataframe(df_metric_rank, outdir / "metric_rank.csv")

    best_metric = df_metric_rank.iloc[0]["metric"]
    for i, fig in enumerate(plot_roc_wrong(df_feat, score_col=best_metric), start=1):
        savefig(fig, outdir / f"roc_wrong_{i:02d}.png")

    df_redg = add_redg_features(df_dg2)
    save_dataframe(df_redg, outdir / "redg_table.csv")
    for i, fig in enumerate(plot_dg_distributions(df_dg2, metric="DG"), start=1):
        savefig(fig, outdir / f"dg_distribution_{i:02d}.png")
    for i, fig in enumerate(plot_redg_distributions(df_redg), start=1):
        savefig(fig, outdir / f"redg_distribution_{i:02d}.png")
    redg_figs, redg_auc = plot_roc_redg_correctness(df_redg)
    save_dataframe(redg_auc, outdir / "redg_auc.csv")
    for i, fig in enumerate(redg_figs, start=1):
        savefig(fig, outdir / f"redg_roc_{i:02d}.png")

    df_e, gate_table = add_eredg(df_dg2, gate_by="per_damage_mode_and_regime", gate_q=0.20)
    save_dataframe(df_e, outdir / "eredg_table.csv")
    save_dataframe(gate_table, outdir / "eredg_gates.csv")

    rows = []
    for (dmg, reg), dfr in df_e.groupby(["damage_mode", "regime"]):
        dfr = dfr[np.isfinite(dfr["eREDG"].to_numpy(dtype=float))]
        y = dfr["correct"].to_numpy(dtype=int)
        s = dfr["eREDG"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        auc_hat, lo, hi, sign, _ = bootstrap_auc_ci(y, s, n_boot=2000, seed=123)
        rows.append({
            "damage_mode": dmg,
            "regime": reg,
            "n_used": int(len(y)),
            "auc_q_to_correct": auc_hat,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "sign_for_q": sign,
        })
    save_dataframe(pd.DataFrame(rows), outdir / "eredg_auc_bootstrap.csv")

    for i, fig in enumerate(final_eredg_panel(df_e), start=1):
        savefig(fig, outdir / f"eredg_panel_{i:02d}.png")

    print(f"Saved eREDG analysis outputs to {outdir}")


if __name__ == "__main__":
    main()
