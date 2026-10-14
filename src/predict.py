"""Predict grid / solar / unserved shares for every 1 km cell and write the maps.

Run (after src.features):
    python -m src.predict --country rwanda
    python -m src.predict --demo            # synthetic data in data/demo/, writes to outputs/demo/

Model: logistic regression on the core features (the simplest model, tied for best in spatial CV).
Cells get features from the same circle radius used for the clusters (model.buffer_m in config.yaml).

Inputs:
    data/restricted/features/<country>_train_table.csv    cluster features and shares (RESTRICTED)
    data/processed/<country>_features_1km.csv, _grid_cells.csv, _grid_spec.json

Outputs (aggregated, model-derived, safe to publish):
    outputs/<country>_share_grid.tif, _share_solar.tif, _share_unserved.tif
        1 km, shares between 0 and 1 that sum to 1, no-data -9999 where a feature is missing
        or the cell's own population is below model.min_cell_population (parks, lakes). Each file carries its sources, licences, method and limitations as tags
        and in a .json next to it.
    outputs/predicted_district_shares_<country>.csv       population-weighted shares per district

The maps are MODEL ESTIMATES, not survey results. Uncertainty ranges are a separate step.
"""
import argparse
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.config import check_restricted_is_ignored, load_config, repo_path, run_paths
from src.train import CLASSES, make_logistic_model

NODATA = -9999

LIMITATIONS = (
    "Modelled estimate from 560 survey clusters, not a census. Training label is the MAIN lighting "
    "source (DHS hv262), so households using grid plus solar count as grid. DHS cluster locations are "
    "displaced by up to 2 km (urban) and 5 km (rural). Solar versus unserved is separated less well than "
    "grid versus non-grid. Night lights and satellite layers cannot see small solar systems. "
    "Unserved here is mostly torch users (about 26% nationally), not the official access gap (about 12%)."
)

DEMO_NOTICE = ("SYNTHETIC DEMO DATA: made up by src/make_demo_data.py to show that the pipeline runs. It is not "
               "Rwanda, not DHS data, and the numbers mean nothing for any real place.")


# ---------------------------------------------------------------- pure helpers
def predict_cells(train: pd.DataFrame, cells: pd.DataFrame, features: list, log_features: list,
                  radius_label: str, seed: int) -> np.ndarray:
    """Fit on clusters, predict every cell. Cells with a missing feature get NaN.

    `cells` has the neighbourhood columns `<feature>_r<radius_label>` (from src.features).
    """
    columns = {f"{f}_r{radius_label}": f for f in features}
    missing = [c for c in columns if c not in cells.columns]
    if missing:
        raise KeyError(f"Not in the 1 km table: {missing}. Is model.buffer_m one of buffers_m in config.yaml?")
    X = cells[list(columns)].rename(columns=columns)
    ok = X.notna().all(axis=1).to_numpy()
    out = np.full((len(cells), len(CLASSES)), np.nan)
    model = make_logistic_model(features, log_features, seed)
    out[ok] = model(train, X[ok])
    return out


def mask_unpopulated(shares: np.ndarray, population: np.ndarray, min_population: float) -> np.ndarray:
    """Copy of `shares` with NaN for cells whose own population is missing or below `min_population`.

    The neighbourhood features fill in parks and lakes, so without this they would get a prediction.
    """
    out = shares.copy()
    out[~(population >= min_population)] = np.nan      # NaN population fails the comparison too
    return out


def cells_to_raster(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: tuple) -> np.ndarray:
    """Put one value per cell back on the grid. NaN and cells outside the country become NODATA."""
    raster = np.full(shape, NODATA, dtype="float32")
    raster[rows, cols] = np.where(np.isnan(values), NODATA, values)
    return raster


def weighted_shares(shares: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of each share over the cells that have a prediction."""
    ok = ~np.isnan(shares).any(axis=1) & ~np.isnan(weights)
    return (shares[ok] * weights[ok, None]).sum(axis=0) / weights[ok].sum()


def district_table(districts: pd.Series, shares: np.ndarray, population: np.ndarray) -> pd.DataFrame:
    """Population-weighted shares per district."""
    rows = []
    for name in sorted(districts.dropna().unique()):
        mask = (districts == name).to_numpy()
        w = weighted_shares(shares[mask], population[mask])
        rows.append({"district": name, **{f"share_{c}": w[k] for k, c in enumerate(CLASSES)},
                     "population": np.nansum(population[mask]), "n_cells": int(mask.sum()),
                     "n_cells_predicted": int((~np.isnan(shares[mask]).any(axis=1)).sum())})
    return pd.DataFrame(rows)


def layer_sources(features: list, gee_dir) -> list:
    """Source and licence of each feature layer, read from the .json files written at download time."""
    out = []
    for f in features:
        meta = json.loads((gee_dir / f"rwanda_{f}.json").read_text())
        out.append(f"{f}: {meta['source']} [{meta['licence']}]")
    return out


def write_share_raster(path, array: np.ndarray, transform, crs: str, tags: dict) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=NODATA, compress="deflate") as dst:
        dst.write(array, 1)
        dst.update_tags(**tags)
    path.with_name(path.name + ".json").write_text(json.dumps(tags, indent=2))


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    parser.add_argument("--demo", action="store_true", help="use the synthetic data in data/demo/")
    args = parser.parse_args(argv)
    if args.demo:
        args.country = "demo"

    check_restricted_is_ignored()
    cfg = load_config()
    mcfg = cfg["model"]
    features = mcfg["feature_sets"]["core"]["baseline"]
    radius_label = f"{int(mcfg['buffer_m'] / 1000)}km"
    paths = run_paths(cfg, args.country, args.demo)
    out_dir = paths["out_dir"]
    np.random.seed(mcfg["seed"])

    train = pd.read_csv(paths["train"]).dropna(subset=features).reset_index(drop=True)
    table = pd.read_csv(paths["features"])
    grid = pd.read_csv(paths["grid_cells"])[["cell_id", "row", "col"]]
    cells = table.merge(grid, on="cell_id", how="left", validate="one_to_one")
    spec = json.loads(paths["grid_spec"].read_text())
    res = spec["resolution_m"]
    transform = from_origin(spec["xmin"], spec["ymax"], res, res)

    shares = predict_cells(train, cells, features, mcfg["log_features"], radius_label, mcfg["seed"])
    n_features_ok = int((~np.isnan(shares).any(axis=1)).sum())
    pop = cells["population_total"].to_numpy()
    shares = mask_unpopulated(shares, pop, mcfg["min_cell_population"])
    n_ok = int((~np.isnan(shares).any(axis=1)).sum())
    print(f"Fitted on {len(train)} clusters, predicted {n_ok:,} of {len(cells):,} cells "
          f"({len(cells) - n_features_ok} empty: missing feature; {n_features_ok - n_ok} empty: own population "
          f"below {mcfg['min_cell_population']} per km2, e.g. parks and lakes)")
    assert np.allclose(np.nansum(shares, axis=1)[~np.isnan(shares).any(axis=1)], 1.0)

    tags = {
        "title": "Share of households by main lighting source, 1 km",
        "country": args.country,
        "unit": "share of households (0 to 1); grid + solar + unserved = 1",
        "method": f"Multinomial logistic regression on {len(features)} features, trained on DHS cluster "
                  f"shares weighted by households; features averaged within {mcfg['buffer_m']} m circles.",
        "labels": "Rwanda DHS 2025 (DHS-8), restricted data. Only model-derived outputs are published.",
        "feature_sources": "Synthetic." if args.demo else
                           "; ".join(layer_sources(features, repo_path(cfg["gee"]["out_dir"]))),
        "year": "Labels: fieldwork June to October 2025. Features: WorldPop 2020, VIIRS 2024-2025, "
                "GHSL 2020, Malaria Atlas Project travel time (2015 cities, 2019 health), SRTM",
        "limitations": LIMITATIONS,
    }
    if args.demo:
        tags.update(labels="Synthetic clusters.", year="None, synthetic.", limitations=DEMO_NOTICE)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, cls in enumerate(CLASSES):
        raster = cells_to_raster(shares[:, k], cells["row"].to_numpy(), cells["col"].to_numpy(),
                                 (spec["height"], spec["width"]))
        write_share_raster(out_dir / f"{args.country}_share_{cls}.tif", raster, transform, spec["crs"],
                           {**tags, "band": f"share_{cls}"})
        print(f"  wrote {args.country}_share_{cls}.tif")

    # ---- sanity checks (aggregates only)
    national = weighted_shares(shares, pop)
    if args.demo:
        print("\nNational shares, % (predicted, population-weighted): " +
              ", ".join(f"{cls} {100 * national[k]:.1f}" for k, cls in enumerate(CLASSES)))
    else:
        dhs = pd.read_csv(out_dir / f"label_summary_{args.country}.csv").set_index("area").loc["national"]
        print("\nNational shares, % (predicted, population-weighted  vs  DHS household-weighted):")
        for k, cls in enumerate(CLASSES):
            print(f"  {cls:9s} {100 * national[k]:5.1f}  vs  {100 * dhs[f'share_{cls}']:5.1f}")

    by_district = district_table(cells["district"], shares, pop)
    by_district.to_csv(out_dir / f"predicted_district_shares_{args.country}.csv", index=False)
    show = by_district.set_index("district")[[f"share_{c}" for c in CLASSES]]
    print("\nHighest grid share, % :")
    print((100 * show.sort_values("share_grid", ascending=False).head(4)).round(1).to_string())
    print("Lowest grid share, % :")
    print((100 * show.sort_values("share_grid").head(4)).round(1).to_string())


if __name__ == "__main__":
    main()
