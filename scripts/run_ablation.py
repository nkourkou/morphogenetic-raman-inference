#!/usr/bin/env python
from __future__ import annotations
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split

from raman_morpho.config import set_global_seed, SEED
from raman_morpho.dataio import load_cells_raman_dataset, make_3class_problem
from raman_morpho.bands import build_band_indices
from raman_morpho.models import train_band_models, decision_function_band_pipes
from raman_morpho.legacy import per_band_ablation, plot_ablation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="cells-raman-spectra")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--damage", type=float, default=0.4)
    ap.add_argument("--mode", default="spiky", choices=["spiky","mixed","frozen"])
    args = ap.parse_args()

    set_global_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    df_all, _wn_raw, _ = load_cells_raman_dataset(args.base_dir)
    df3, spectra, wn, y, le3, _map = make_3class_problem(df_all)
    band_indices = build_band_indices(wn)

    X_train, X_test, y_train, y_test = train_test_split(
        spectra, y, test_size=args.test_size, stratify=y, random_state=SEED
    )

    band_pipes = train_band_models(X_train, y_train, band_indices, seed=SEED)
    band_logits_test = decision_function_band_pipes(X_test, band_pipes, band_indices)

    df_ab = per_band_ablation(band_logits_test, y_test, damage_fraction=args.damage, mode=args.mode, seed=SEED)
    out_csv = os.path.join(args.out_dir, f"ablation_{args.mode}_d{args.damage:.2f}.csv")
    df_ab.to_csv(out_csv, index=False)

    fig = plot_ablation(df_ab, title=f"Per-band ablation ({args.mode}, damage={args.damage})")
    figpath = os.path.join(args.out_dir, f"ablation_{args.mode}_d{args.damage:.2f}.png")
    fig.savefig(figpath, dpi=600, bbox_inches="tight")

    print("Saved:", out_csv)
    print("Figure:", figpath)

if __name__ == "__main__":
    main()
