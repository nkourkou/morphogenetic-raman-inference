#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from raman_morpho.config import set_global_seed, SEED
from raman_morpho.dataio import load_cells_raman_dataset, make_3class_problem
from raman_morpho.bands import build_band_indices
from raman_morpho.models import train_full_model, train_band_models, decision_function_band_pipes
from raman_morpho.evaluation import evaluate_morphogenetic_system
from raman_morpho.plots import plot_accuracy_vs_damage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="cells-raman-spectra")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--damage", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6])
    args = ap.parse_args()

    set_global_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    df_all, _wn_raw, _spec_cols = load_cells_raman_dataset(args.base_dir)
    df3, spectra, wn, y, le3, label_map = make_3class_problem(df_all)

    band_indices = build_band_indices(wn)

    X_train, X_test, y_train, y_test = train_test_split(
        spectra, y, test_size=args.test_size, stratify=y, random_state=SEED
    )

    full_model = train_full_model(X_train, y_train, seed=SEED)
    full_logits_test = full_model.decision_function(X_test)

    band_pipes = train_band_models(X_train, y_train, band_indices, seed=SEED)
    band_logits_test = decision_function_band_pipes(X_test, band_pipes, band_indices)

    regimes = ["plain", "H1", "H2"]
    damage_modes = ["mixed", "spiky"]
    all_rows = []

    for dmg in damage_modes:
        for reg in regimes:
            res = evaluate_morphogenetic_system(
                full_logits_test=full_logits_test,
                band_logits_test=band_logits_test,
                y_test=y_test,
                damage_fractions=list(args.damage),
                regime=reg,
                damage_mode=dmg,
                naive_weighting="uniform",
                morpho_weighting="confidence",
                seed=SEED,
            )
            df = pd.DataFrame(res)
            all_rows.append(df)

    df_res_all = pd.concat(all_rows, ignore_index=True)
    df_res_all.to_csv(os.path.join(args.out_dir, "results_sweep.csv"), index=False)

    plot_accuracy_vs_damage(df_res_all, out_dir=args.out_dir)

    print("Saved:", os.path.join(args.out_dir, "results_sweep.csv"))
    print("Figures in:", args.out_dir)

if __name__ == "__main__":
    main()
