from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morphoraman.bands import build_band_indices
from morphoraman.closed_loop import evaluate_closed_loop
from morphoraman.config import ExperimentConfig, set_global_seed
from morphoraman.data import load_cells_raman_dataset, prepare_three_class_dataset, split_dataset
from morphoraman.io_utils import ensure_dir, save_dataframe
from morphoraman.models import fit_band_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", default="outputs/closed_loop")
    args = parser.parse_args()

    cfg = ExperimentConfig(base_dir=args.data_dir)
    set_global_seed(cfg.seed)
    outdir = ensure_dir(args.outdir)

    df_all = load_cells_raman_dataset(cfg.base_dir, expected_points=cfg.expected_points)
    bundle = prepare_three_class_dataset(df_all)
    band_indices = build_band_indices(bundle.wn, list(cfg.bands))

    x_train, x_test, y_train, y_test = split_dataset(bundle.X, bundle.y, test_size=cfg.test_size, seed=cfg.seed)
    band_logits_test, _ = fit_band_models(x_train, y_train, x_test, band_indices, seed=cfg.seed)

    for damage_mode in ("spiky", "mixed"):
        for regime in ("H1", "H2"):
            df_eval, df_lesions = evaluate_closed_loop(
                band_logits_test=band_logits_test,
                y_test=y_test,
                bands=list(cfg.bands),
                damage_fraction=0.4,
                damage_mode=damage_mode,
                regime=regime,
                run_id=0,
                alpha=cfg.alpha,
                lam=cfg.lam,
                T=cfg.temperature,
                tau=cfg.tau,
                max_iter=cfg.max_iter,
                weight_clip=cfg.weight_clip,
                morpho_weighting=cfg.morpho_weighting,
                top_k=2,
                use_dg_in_lesion=True,
            )
            save_dataframe(df_eval, outdir / f"closed_loop_eval_{damage_mode}_{regime}.csv")
            save_dataframe(df_lesions, outdir / f"closed_loop_lesions_{damage_mode}_{regime}.csv")
    print(f"Saved closed-loop outputs to {outdir}")


if __name__ == "__main__":
    main()
