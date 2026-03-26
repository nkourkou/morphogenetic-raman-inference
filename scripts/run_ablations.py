from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morphoraman.ablation import band_pair_ablation, per_band_ablation
from morphoraman.bands import build_band_indices
from morphoraman.config import ExperimentConfig, set_global_seed
from morphoraman.data import load_cells_raman_dataset, prepare_three_class_dataset, split_dataset
from morphoraman.io_utils import ensure_dir, save_dataframe
from morphoraman.models import fit_band_models
from morphoraman.plotting import plot_ablation, plot_pair_heatmaps, savefig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", default="outputs/ablations")
    args = parser.parse_args()

    cfg = ExperimentConfig(base_dir=args.data_dir)
    set_global_seed(cfg.seed)
    outdir = ensure_dir(args.outdir)

    df_all = load_cells_raman_dataset(cfg.base_dir, expected_points=cfg.expected_points)
    bundle = prepare_three_class_dataset(df_all)
    band_indices = build_band_indices(bundle.wn, list(cfg.bands))

    x_train, x_test, y_train, y_test = split_dataset(bundle.X, bundle.y, test_size=cfg.test_size, seed=cfg.seed)
    band_logits_test, _ = fit_band_models(x_train, y_train, x_test, band_indices, seed=cfg.seed)

    df_ablate = per_band_ablation(
        band_logits_test=band_logits_test,
        y_test=y_test,
        bands=list(cfg.bands),
        alpha=cfg.alpha,
        lam=cfg.lam,
        T=cfg.temperature,
        tau=cfg.tau,
        max_iter=cfg.max_iter,
        weight_clip=cfg.weight_clip,
        morpho_weighting=cfg.morpho_weighting,
        run_id=0,
    )
    save_dataframe(df_ablate, outdir / "per_band_ablation.csv")

    for value, title, ylabel, fname in [
        ("delta_acc", "ΔAccuracy vs one-band ablation", "ΔAccuracy (ablation − clean)", "per_band_delta_acc.png"),
        ("delta_auc", "ΔMacro AUC vs one-band ablation", "ΔMacro AUC (OVR)", "per_band_delta_auc.png"),
        ("dg_mean", "Decision Geometry under one-band ablation", "Mean DG", "per_band_dg.png"),
    ]:
        fig = plot_ablation(df_ablate, value=value, title=title, ylabel=ylabel, bands=list(cfg.bands))
        savefig(fig, outdir / fname)

    df_pair = band_pair_ablation(
        band_logits_test=band_logits_test,
        y_test=y_test,
        bands=list(cfg.bands),
        alpha=cfg.alpha,
        lam=cfg.lam,
        T=cfg.temperature,
        tau=cfg.tau,
        max_iter=cfg.max_iter,
        weight_clip=cfg.weight_clip,
        morpho_weighting=cfg.morpho_weighting,
        run_id=0,
    )
    save_dataframe(df_pair, outdir / "band_pair_ablation.csv")
    figs = plot_pair_heatmaps(df_pair, k_bands=len(cfg.bands), bands=list(cfg.bands))
    for i, fig in enumerate(figs, start=1):
        savefig(fig, outdir / f"pair_heatmap_{i:02d}.png")
    print(f"Saved ablation outputs to {outdir}")


if __name__ == "__main__":
    main()
