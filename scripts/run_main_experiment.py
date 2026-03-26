from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from sklearn.metrics import accuracy_score

from morphoraman.bands import build_band_indices
from morphoraman.config import ExperimentConfig, set_global_seed
from morphoraman.data import load_cells_raman_dataset, prepare_three_class_dataset, split_dataset
from morphoraman.evaluation import evaluate_morphogenetic_system, normalize_dg
from morphoraman.io_utils import ensure_dir, save_dataframe
from morphoraman.models import fit_auxiliary_full_baselines, fit_band_models, fit_full_model
from morphoraman.plotting import plot_composite_accuracy_dgl, savefig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", default="outputs/main")
    args = parser.parse_args()

    cfg = ExperimentConfig(base_dir=args.data_dir)
    set_global_seed(cfg.seed)
    outdir = ensure_dir(args.outdir)

    df_all = load_cells_raman_dataset(cfg.base_dir, expected_points=cfg.expected_points)
    bundle = prepare_three_class_dataset(df_all)
    band_indices = build_band_indices(bundle.wn, list(cfg.bands))

    x_train, x_test, y_train, y_test = split_dataset(bundle.X, bundle.y, test_size=cfg.test_size, seed=cfg.seed)
    full_res = fit_full_model(x_train, y_train, x_test, seed=cfg.seed)
    band_logits_test, _ = fit_band_models(x_train, y_train, x_test, band_indices, seed=cfg.seed)

    baselines = fit_auxiliary_full_baselines(x_train, y_train, x_test, y_test, seed=cfg.seed)
    pd.DataFrame([{
        "baseline_full_acc": float(accuracy_score(y_test, full_res.logits_test.argmax(axis=1))),
        **baselines,
    }]).to_csv(outdir / "full_model_baselines.csv", index=False)

    rows = []
    for damage_mode in cfg.damage_modes:
        for regime in cfg.regimes:
            res = evaluate_morphogenetic_system(
                full_logits_test=full_res.logits_test,
                band_logits_test=band_logits_test,
                y_test=y_test,
                damage_fractions=cfg.damage_fractions,
                alpha=cfg.alpha,
                lam=cfg.lam,
                T=cfg.temperature,
                tau=cfg.tau,
                max_iter=cfg.max_iter,
                damage_mode=damage_mode,
                naive_weighting=cfg.naive_weighting,
                morpho_weighting=cfg.morpho_weighting,
                regime=regime,
                run_id=0,
                weight_clip=cfg.weight_clip,
            )
            df_reg = pd.DataFrame(res)
            df_reg["regime"] = regime
            df_reg["damage_mode"] = damage_mode
            rows.append(df_reg)

    df_res = pd.concat(rows, ignore_index=True)
    df_res = normalize_dg(df_res)
    save_dataframe(df_res, outdir / "robustness_sweep.csv")

    fig = plot_composite_accuracy_dgl(df_res, highlight_regime="H1", normalize_dg=True)
    savefig(fig, outdir / "composite_accuracy_dgl.png")
    print(f"Saved tables and figures to {outdir}")


if __name__ == "__main__":
    main()
