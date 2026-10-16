"""Run the whole pipeline on synthetic data. Needs no DHS data and no downloads.

Run from a clean clone:
    python -m src.demo

Steps: make the synthetic data (if data/demo/ is empty), predict the shares, add uncertainty ranges and the
do-not-trust mask, write the EAE market-reality layer. Everything lands in outputs/demo/.
Then it checks the result against the hidden true shares that only the synthetic data has.
"""
import argparse

import numpy as np
import pandas as pd
import rasterio

from src import export_eae, predict, uncertainty
from src.config import load_config, repo_path, run_paths
from src.make_demo_data import write_demo
from src.train import CLASSES


def score_against_truth(out_dir, demo_dir) -> dict:
    """Mean absolute error of each predicted share vs the hidden true share, population-weighted, populated cells only."""
    cells = pd.read_csv(demo_dir / "demo_features_1km.csv")
    grid = pd.read_csv(demo_dir / "demo_grid_cells.csv")[["cell_id", "row", "col"]]
    cells = cells.merge(grid, on="cell_id", validate="one_to_one")
    result = {}
    for cls in CLASSES:
        with rasterio.open(out_dir / f"demo_share_{cls}.tif") as src:
            pred = src.read(1)[cells["row"], cells["col"]]
        ok = pred != predict.NODATA
        err = np.abs(pred[ok] - cells[f"true_share_{cls}"][ok])
        result[cls] = float(np.average(err, weights=cells["population_total"][ok]))
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--regenerate", action="store_true", help="rebuild data/demo/ even if it exists")
    args = parser.parse_args(argv)
    cfg = load_config()
    paths = run_paths(cfg, "demo", demo=True)
    if args.regenerate or not paths["train"].exists():
        print("Making the synthetic data ...")
        write_demo(repo_path(cfg["paths"]["demo"]), cfg["model"]["seed"])

    for step in (predict, uncertainty, export_eae):
        print(f"\n=== {step.__name__} ===")
        step.main(["--demo"])

    errors = score_against_truth(paths["out_dir"], repo_path(cfg["paths"]["demo"]))
    print("\nMean absolute error vs the hidden true shares (only possible because the data is synthetic):")
    for cls, e in errors.items():
        print(f"  {cls:9s} {e:.3f}")
    print("A planted rule is far easier than real life: on Rwanda, spatial cross-validation error is about 0.15, "
          "not this small.")
    print(f"\nDone. Open the .tif files in {paths['out_dir']} (QGIS, or rasterio). "
          "Every file is tagged as synthetic demo data.")


if __name__ == "__main__":
    main()
