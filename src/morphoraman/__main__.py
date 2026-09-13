import argparse
import importlib
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("melanoma", "diabetes", "bacteria"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--analysis", default="robustness")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        parser.error(f"Data directory does not exist: {data_dir}")
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    module = importlib.import_module(f"morphoraman.{args.dataset}")
    supported = ("robustness", "baselines", "partition", "audit", "conflict", "summary", "spectra_plot")
    if args.analysis not in supported or not hasattr(module, args.analysis):
        parser.error(f"Unsupported analysis for {args.dataset}: {args.analysis}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_dir = Path.cwd()
    try:
        os.chdir(output_dir)
        module.prepare(data_dir)
        result = getattr(module, args.analysis)()
        if result is not None:
            print(result)
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main()
