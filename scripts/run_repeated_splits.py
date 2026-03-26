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
from sklearn.model_selection import StratifiedShuffleSplit

from morphoraman.bands import build_band_indices
from morphoraman.config import ExperimentConfig, set_global_seed
from morphoraman.data import load_cells_raman_dataset, prepare_three_class_dataset
from morphoraman.evaluation import add_eredg, cliffs_delta, compute_dg_and_stress_table
from morphoraman.io_utils import ensure_dir, save_dataframe
from morphoraman.models import fit_band_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", default="outputs/repeated_splits")
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    cfg = ExperimentConfig(base_dir=args.data_dir)
    set_global_seed(cfg.seed)
    outdir = ensure_dir(args.outdir)

    df_all = load_cells_raman_dataset(cfg.base_dir, expected_points=cfg.expected_points)
    bundle = prepare_three_class_dataset(df_all)
    band_indices = build_band_indices(bundle.wn, list(cfg.bands))

    x = bundle.X
    y = bundle.y
    n_splits = args.n_splits
    seeds = list(range(100, 100 + n_splits))

    rows = []
    for split_id, seed in enumerate(seeds, start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(x, y))

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        band_logits_test, _ = fit_band_models(x_train, y_train, x_test, band_indices, seed=seed)

        df_dg2 = compute_dg_and_stress_table(
            band_logits_test=band_logits_test,
            y_test=y_test,
            damage_fraction=0.4,
            damage_modes=("mixed",),
            regimes=("plain", "H1", "H2"),
            run_id=seed,
            alpha=cfg.alpha,
            lam=cfg.lam,
            T=cfg.temperature,
            tau=cfg.tau,
            max_iter=cfg.max_iter,
            weight_clip=cfg.weight_clip,
            morpho_weighting=cfg.morpho_weighting,
            k_early=3,
        )
        df_e, _ = add_eredg(df_dg2, gate_by="per_damage_mode_and_regime", gate_q=0.20)

        for dmg in ("mixed",):
            for reg in ("plain", "H1", "H2"):
                dfr = df_e[(df_e["damage_mode"] == dmg) & (df_e["regime"] == reg)].copy()
                dfr = dfr[np.isfinite(dfr["eREDG"].to_numpy(dtype=float))]
                if len(dfr) == 0:
                    continue

                y_corr = dfr["correct"].to_numpy(dtype=int)
                x_score = dfr["eREDG"].to_numpy(dtype=float)
                if len(np.unique(y_corr)) < 2:
                    auc = np.nan
                    score_sign = "NA"
                else:
                    from sklearn.metrics import roc_auc_score
                    auc_pos = roc_auc_score(y_corr, x_score)
                    auc_neg = roc_auc_score(y_corr, -x_score)
                    if auc_neg > auc_pos:
                        auc = float(auc_neg)
                        score_sign = "-"
                    else:
                        auc = float(auc_pos)
                        score_sign = "+"

                a = dfr.loc[dfr["correct"] == 1, "eREDG"].to_numpy(dtype=float)
                b = dfr.loc[dfr["correct"] == 0, "eREDG"].to_numpy(dtype=float)
                delta = cliffs_delta(a, b)

                rows.append({
                    "split": split_id,
                    "seed": seed,
                    "damage_mode": dmg,
                    "regime": reg,
                    "n_active": len(dfr),
                    "auc_eredg_to_correct": auc,
                    "score_sign": score_sign,
                    "cliffs_delta": delta,
                    "n_correct": int(np.sum(dfr["correct"] == 1)),
                    "n_incorrect": int(np.sum(dfr["correct"] == 0)),
                })

    df_runs = pd.DataFrame(rows)
    save_dataframe(df_runs, outdir / "repeated_split_eredg_runs.csv")
    summary = (
        df_runs.groupby(["damage_mode", "regime"])[["auc_eredg_to_correct", "cliffs_delta", "n_active"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.to_csv(outdir / "repeated_split_eredg_summary.csv")
    print(f"Saved repeated-split robustness outputs to {outdir}")


if __name__ == "__main__":
    main()
