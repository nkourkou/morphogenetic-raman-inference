from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morphoraman.config import ExperimentConfig, set_global_seed
from morphoraman.data import load_cells_raman_dataset, prepare_three_class_dataset
from morphoraman.io_utils import ensure_dir
from morphoraman.plotting import plot_mean_spectra_per_class, savefig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", default="outputs/mean_spectra")
    args = parser.parse_args()

    cfg = ExperimentConfig(base_dir=args.data_dir)
    set_global_seed(cfg.seed)

    outdir = ensure_dir(args.outdir)
    df_all = load_cells_raman_dataset(cfg.base_dir, expected_points=cfg.expected_points)
    bundle = prepare_three_class_dataset(df_all)

    fig = plot_mean_spectra_per_class(X=bundle.X, y=bundle.y, wn=bundle.wn, label_encoder=bundle.label_encoder)
    savefig(fig, outdir / "mean_raman_per_class.png")
    print(f"Saved: {outdir / 'mean_raman_per_class.png'}")


if __name__ == "__main__":
    main()
