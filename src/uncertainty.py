"""Uncertainty ranges (low and high) for every share, and a "do not trust" mask.

Run (after src.features and src.predict):
    python -m src.uncertainty --country rwanda
    python -m src.uncertainty --demo        # synthetic data in data/demo/, writes to outputs/demo/

Method: split conformal prediction. The logistic model is cross-validated with spatial folds
(groups = district), so every cluster is predicted by a model that never saw its district. The
household-weighted quantile of the absolute errors gives the half-width of the range. Clusters are
binned by predicted value (errors are larger for mid-range shares) and each bin has its own width.
Range = prediction +/- half-width, clipped to 0..1, separately for grid, solar and unserved,
so the three ranges do not sum to 1.

Coverage is checked with nested spatial folds: calibrate on four folds, test on the fifth.
The "do not trust" mask has two flags: the widest model.uncertainty.flag_widest_share of cells, ranked
by their widest share range (value 1), and cell features outside the range seen at the clusters
(value 2). Both = 3, fine = 0. The width flag is a ranking, not an absolute limit: a fixed 0.4 limit
flagged 94% of cells, because cluster-level noise dominates the ranges.

Inputs:  as src.predict, plus data/restricted/features/<country>_train_table.csv (RESTRICTED)
Outputs (aggregated, safe to publish):
    outputs/<country>_share_<grid|solar|unserved>_<low|high>.tif
    outputs/<country>_do_not_trust.tif
    outputs/uncertainty_coverage_<country>.csv      coverage and mean width per share and bin
"""
import argparse
import json

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from src.config import check_restricted_is_ignored, load_config, repo_path, run_paths
from src.predict import (DEMO_NOTICE, LIMITATIONS, NODATA, cells_to_raster, layer_sources, mask_unpopulated,
                         predict_cells, write_share_raster)
from src.train import CLASSES, SHARE_COLS, cross_validate, make_logistic_model, make_splits

WIDTH_FLAG, EXTRAPOLATION_FLAG = 1, 2


# ---------------------------------------------------------------- pure helpers
def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Smallest value whose cumulative weight reaches `level` (0 to 1)."""
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return values[order][min(np.searchsorted(cum, level), len(values) - 1)]


def conformal_level(coverage: float, n: int) -> float:
    """Finite-sample correction: a little above the target, because n is small."""
    return min(1.0, coverage * (1 + 1 / n))


def assign_bins(pred: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin number 0..len(edges) for each prediction."""
    return np.searchsorted(edges, pred, side="right")


def calibrate(pred: np.ndarray, true: np.ndarray, weights: np.ndarray, coverage: float,
              n_bins: int, min_bin_clusters: int):
    """Per share: bin edges (from the predictions) and the half-width of the range in each bin.

    Returns (edges, half_width) with shapes (n_shares, n_bins - 1) and (n_shares, n_bins).
    Bins with fewer than `min_bin_clusters` clusters use the pooled half-width.
    """
    n_shares = pred.shape[1]
    edges = np.empty((n_shares, n_bins - 1))
    half = np.empty((n_shares, n_bins))
    for k in range(n_shares):
        err = np.abs(true[:, k] - pred[:, k])
        edges[k] = np.quantile(pred[:, k], np.arange(1, n_bins) / n_bins)
        bins = assign_bins(pred[:, k], edges[k])
        pooled = weighted_quantile(err, weights, conformal_level(coverage, len(err)))
        for j in range(n_bins):
            m = bins == j
            half[k, j] = (weighted_quantile(err[m], weights[m], conformal_level(coverage, int(m.sum())))
                          if m.sum() >= min_bin_clusters else pooled)
    return edges, half


def apply_interval(pred: np.ndarray, edges: np.ndarray, half: np.ndarray):
    """Low and high per share, clipped to 0..1. NaN predictions stay NaN."""
    low, high = np.full(pred.shape, np.nan), np.full(pred.shape, np.nan)
    for k in range(pred.shape[1]):
        ok = ~np.isnan(pred[:, k])
        width = half[k][assign_bins(pred[ok, k], edges[k])]
        low[ok, k] = np.clip(pred[ok, k] - width, 0, 1)
        high[ok, k] = np.clip(pred[ok, k] + width, 0, 1)
    return low, high


def nested_coverage(pred: np.ndarray, true: np.ndarray, weights: np.ndarray, splits: list,
                    coverage: float, n_bins: int, min_bin_clusters: int) -> pd.DataFrame:
    """Honest coverage: calibrate on the other folds' errors, test on the held-out fold.

    Household-weighted share of true values inside the range, per share and per bin.
    (The out-of-fold predictions of the calibration folds came from models that saw the test
    fold. That only affects the width, and slightly.)
    """
    inside, width = np.zeros(pred.shape, dtype=bool), np.zeros(pred.shape)
    bins = np.zeros(pred.shape, dtype=int)
    for train_idx, test_idx in splits:
        edges, half = calibrate(pred[train_idx], true[train_idx], weights[train_idx], coverage,
                                n_bins, min_bin_clusters)
        low, high = apply_interval(pred[test_idx], edges, half)
        inside[test_idx] = (true[test_idx] >= low) & (true[test_idx] <= high)
        width[test_idx] = high - low
        for k in range(pred.shape[1]):
            bins[test_idx, k] = assign_bins(pred[test_idx, k], edges[k])
    rows = []
    for k, cls in enumerate(CLASSES):
        for label, mask in [("all", np.ones(len(pred), dtype=bool))] + [(str(j), bins[:, k] == j) for j in range(n_bins)]:
            w = weights[mask]
            if w.sum() == 0:
                continue
            rows.append({"share": cls, "bin": label, "n_clusters": int(mask.sum()),
                         "coverage": (w * inside[mask, k]).sum() / w.sum(),
                         "mean_width": (w * width[mask, k]).sum() / w.sum()})
    return pd.DataFrame(rows)


def extrapolation_flag(cell_features: pd.DataFrame, train_features: pd.DataFrame, percentiles: list) -> np.ndarray:
    """True where any cell feature is outside the percentile range seen at the training clusters.

    Both tables have the same column names. Missing values are not flagged (those cells are empty anyway).
    """
    lo = train_features.quantile(percentiles[0] / 100)
    hi = train_features.quantile(percentiles[1] / 100)
    return ((cell_features < lo) | (cell_features > hi)).any(axis=1).to_numpy()


def do_not_trust_codes(low: np.ndarray, high: np.ndarray, extrapolated: np.ndarray, widest_share: float):
    """0 fine, 1 among the widest ranges, 2 outside the clusters' feature range, 3 both.

    Returns (codes, cut_off). Cells with no prediction get NaN. `cut_off` is the (1 - widest_share) quantile
    of each cell's widest share range; cells at or above it are flagged. Ranges take few distinct values
    (one per bin, before clipping), so ties can flag slightly more than `widest_share` of the cells.
    """
    has_prediction = ~np.isnan(low).any(axis=1)
    widest = np.where(has_prediction, (high - low).max(axis=1), np.nan)
    cut_off = float(np.quantile(widest[has_prediction], 1 - widest_share))
    codes = (widest >= cut_off) * WIDTH_FLAG + extrapolated * EXTRAPOLATION_FLAG
    return np.where(has_prediction, codes, np.nan).astype(float), cut_off


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
    mcfg, ucfg = cfg["model"], cfg["model"]["uncertainty"]
    features = mcfg["feature_sets"]["core"]["baseline"]
    radius_label = f"{int(mcfg['buffer_m'] / 1000)}km"
    seed = mcfg["seed"]
    paths = run_paths(cfg, args.country, args.demo)
    out_dir = paths["out_dir"]
    np.random.seed(seed)

    train = pd.read_csv(paths["train"]).dropna(subset=features).reset_index(drop=True)
    true, weights = train[SHARE_COLS].to_numpy(), train["n_households"].to_numpy()

    # ---- calibration on spatial out-of-fold predictions
    splits = make_splits(train, "spatial", mcfg["cv_folds"], seed)
    oof = cross_validate(train, make_logistic_model(features, mcfg["log_features"], seed), splits)
    args_cal = (ucfg["coverage"], ucfg["n_bins"], ucfg["min_bin_clusters"])
    edges, half = calibrate(oof, true, weights, *args_cal)
    check = nested_coverage(oof, true, weights, splits, *args_cal)
    out_dir.mkdir(parents=True, exist_ok=True)
    check.to_csv(out_dir / f"uncertainty_coverage_{args.country}.csv", index=False)
    print(f"Target coverage {ucfg['coverage']:.0%}. Nested spatial coverage, household-weighted:")
    print(check[check["bin"] == "all"].round(3).to_string(index=False))
    print("Half-width of the range per bin (rows: grid, solar, unserved):")
    print(pd.DataFrame(half, index=CLASSES).round(3).to_string())

    # ---- cells
    table = pd.read_csv(paths["features"])
    grid = pd.read_csv(paths["grid_cells"])[["cell_id", "row", "col"]]
    cells = table.merge(grid, on="cell_id", how="left", validate="one_to_one")
    spec = json.loads(paths["grid_spec"].read_text())
    res = spec["resolution_m"]
    transform = from_origin(spec["xmin"], spec["ymax"], res, res)

    shares = predict_cells(train, cells, features, mcfg["log_features"], radius_label, seed)
    pop = cells["population_total"].to_numpy()
    shares = mask_unpopulated(shares, pop, mcfg["min_cell_population"])
    low, high = apply_interval(shares, edges, half)

    cell_features = cells[[f"{f}_r{radius_label}" for f in features]].set_axis(features, axis=1)
    extrapolated = extrapolation_flag(cell_features, train[features], ucfg["extrapolation_percentiles"])
    codes, cut_off = do_not_trust_codes(low, high, extrapolated, ucfg["flag_widest_share"])

    has = ~np.isnan(codes)
    for flag, name in [(WIDTH_FLAG, f"widest {ucfg['flag_widest_share']:.0%} of ranges (width at or above {cut_off:.2f})"), (EXTRAPOLATION_FLAG, "features outside cluster range")]:
        flagged = has & (np.nan_to_num(codes).astype(int) & flag > 0)
        print(f"Flagged, {name}: {flagged.sum():,} of {has.sum():,} cells, "
              f"{100 * np.nansum(pop[flagged]) / np.nansum(pop[has]):.1f}% of population")
    for k, cls in enumerate(CLASSES):
        w = (high - low)[has, k]
        print(f"  {cls:9s} range width, median {np.median(w):.2f}, 90th percentile {np.quantile(w, 0.9):.2f}")

    # ---- write
    nested = check[check["bin"] == "all"].set_index("share")["coverage"]
    tags = {
        "country": args.country,
        "unit": "share of households (0 to 1)",
        "method": f"Split conformal prediction on spatial (leave-districts-out) out-of-fold errors of the "
                  f"share model, {ucfg['n_bins']} bins by predicted value, target coverage "
                  f"{ucfg['coverage']:.0%}. Range = prediction +/- bin half-width, clipped to 0..1. "
                  "Grid, solar and unserved are bounded separately, so the three ranges do not sum to 1.",
        "coverage_check": "Nested spatial coverage: " + ", ".join(f"{c} {nested[c]:.0%}" for c in CLASSES),
        "labels": "Rwanda DHS 2025 (DHS-8), restricted data. Only model-derived outputs are published.",
        "feature_sources": "Synthetic." if args.demo else
                           "; ".join(layer_sources(features, repo_path(cfg["gee"]["out_dir"]))),
        "limitations": LIMITATIONS + " The range is calibrated on cluster-level errors, which include "
                       "household sampling noise (about 25 households per cluster) and GPS displacement, so it is "
                       "wider than the uncertainty of the true cell share. Coverage holds on average across "
                       "districts, not for every cell.",
    }
    if args.demo:
        tags.update(labels="Synthetic clusters.",
                    limitations=DEMO_NOTICE + " " + tags["limitations"])
    shape, rows, cols = (spec["height"], spec["width"]), cells["row"].to_numpy(), cells["col"].to_numpy()
    for k, cls in enumerate(CLASSES):
        for side, values in [("low", low), ("high", high)]:
            write_share_raster(out_dir / f"{args.country}_share_{cls}_{side}.tif",
                               cells_to_raster(values[:, k], rows, cols, shape), transform, spec["crs"],
                               {**tags, "title": f"{side.capitalize()} end of the {ucfg['coverage']:.0%} range "
                                                 f"for the {cls} share, 1 km", "band": f"share_{cls}_{side}"})
    write_share_raster(out_dir / f"{args.country}_do_not_trust.tif", cells_to_raster(codes, rows, cols, shape),
                       transform, spec["crs"],
                       {**tags, "title": "Do-not-trust mask, 1 km", "band": "do_not_trust",
                        "unit": f"code: 0 fine, 1 among about the widest {ucfg['flag_widest_share']:.0%} of ranges (a ranking, "
                                f"widest share range at or above {cut_off:.2f}; a fixed 0.4 limit would flag 94% of cells), "
                                "2 features outside the range seen at survey clusters, 3 both"})
    print(f"\nWrote 6 range rasters and the mask to {out_dir}")


if __name__ == "__main__":
    main()
