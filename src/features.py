"""Stack the GEE (Google Earth Engine) feature layers on the 1 km grid and on DHS clusters.

Run (after src.grid, src.download.gee_features and src.labels_dhs):
    python -m src.features --country rwanda

Inputs:
    data/raw/gee/<country>_<layer>.tif                 100 m layers from Google Earth Engine
    data/processed/layers/<country>_<layer>.tif        100 m distance layers (python -m src.distance_layers)
    data/processed/<country>_grid_1km.tif, _grid_cells.csv, _grid_spec.json
    data/restricted/labels/<country>_cluster_labels.csv   RESTRICTED (has cluster coordinates)

Outputs:
    data/processed/<country>_features_1km.csv          one row per 1 km cell, for prediction
    data/restricted/features/<country>_train_table.csv one row per cluster, for training.
                                                       No coordinates, but kept restricted anyway.

Units: every layer is a MEAN of the 100 m pixels (so population is "people per 100 m pixel"),
for the cell itself and for circles around it. This keeps 1 km cells and cluster buffers
comparable. `population_total` is the real sum (people per km2), used only to weight sums.

How training and prediction features match (one radius for both, model.buffer_m in config.yaml):
    train table   <layer>          mean of 100 m pixels inside a circle of buffer_m around the cluster
    1 km table    <layer>          mean inside the cell itself
                  <layer>_r2km     mean in a 2 km circle around the cell  (prediction uses this when buffer_m = 2000)
                  <layer>_r5km     mean in a 5 km circle around the cell
The circles in the 1 km table are computed from 1 km cell means, so they approximate the exact
100 m circles used for clusters (correlation at least 0.98 for 2 km, checked on Rwanda).

DHS rule: only counts and layer averages are printed, never cluster coordinates.
"""
import argparse
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import convolve

from src.config import check_restricted_is_ignored, load_config, repo_path

NODATA = -9999
SHARES = ["share_grid", "share_solar", "share_unserved"]


# ---------------------------------------------------------------- pure array helpers
def coarsen(a: np.ndarray, factor: int, how: str = "mean") -> np.ndarray:
    """Combine `factor` x `factor` blocks of pixels. NaN pixels are ignored."""
    h, w = a.shape
    if h % factor or w % factor:
        raise ValueError(f"Array shape {a.shape} is not a multiple of {factor}")
    blocks = a.reshape(h // factor, factor, w // factor, factor).swapaxes(1, 2)
    blocks = blocks.reshape(h // factor, w // factor, factor * factor)
    all_nan = np.isnan(blocks).all(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # "mean of empty slice" for all-NaN blocks
        out = np.nansum(blocks, axis=2) if how == "sum" else np.nanmean(blocks, axis=2)
    out[all_nan] = np.nan
    return out


def focal_mean(a: np.ndarray, radius_cells: float) -> np.ndarray:
    """Mean of all non-NaN cells whose centre is within `radius_cells` of each cell."""
    r = int(np.floor(radius_cells))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    kernel = ((xx ** 2 + yy ** 2) <= radius_cells ** 2).astype(float)
    valid = ~np.isnan(a)
    total = convolve(np.where(valid, a, 0.0), kernel, mode="constant")
    count = convolve(valid.astype(float), kernel, mode="constant")
    out = np.full(a.shape, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def buffer_means(layers: dict, transform, xs, ys, radii_m) -> dict:
    """Mean of each layer inside a circle around each point (x, y in the raster CRS, metres).

    A pixel counts when its centre is inside the circle. NaN pixels are ignored.
    Points whose circle holds no valid pixel get NaN.
    """
    res = transform.a
    height, width = next(iter(layers.values())).shape
    out = {name: np.full(len(xs), np.nan) for name in layers}
    for i, (x, y, radius) in enumerate(zip(xs, ys, radii_m)):
        col, row = (x - transform.c) / res, (transform.f - y) / res    # fractional pixel position
        r_px = radius / res
        r0, r1 = max(int(np.floor(row - r_px)), 0), min(int(np.ceil(row + r_px)) + 1, height)
        c0, c1 = max(int(np.floor(col - r_px)), 0), min(int(np.ceil(col + r_px)) + 1, width)
        if r0 >= r1 or c0 >= c1:
            continue                                                   # point outside the raster
        rr, cc = np.ogrid[r0:r1, c0:c1]
        inside = (cc + 0.5 - col) ** 2 + (rr + 0.5 - row) ** 2 <= r_px ** 2
        for name, a in layers.items():
            values = a[r0:r1, c0:c1][inside]
            values = values[~np.isnan(values)]
            if values.size:
                out[name][i] = values.mean()
    return out


# ---------------------------------------------------------------- loading
def load_layers(layer_dirs: list, country: str, inside_100m: np.ndarray):
    """Read all 100 m layers from the given folders, set no-data and pixels outside the country to NaN."""
    layers, transform = {}, None
    paths = sorted(p for d in layer_dirs for p in d.glob(f"{country}_*.tif"))
    for path in paths:
        name = path.stem[len(country) + 1:]
        with rasterio.open(path) as src:
            a = src.read(1).astype("float64")
            if transform is not None and src.transform != transform:
                raise ValueError(f"{path.name} is not aligned with the other layers")
            transform = src.transform
        a[a == NODATA] = np.nan
        a[~inside_100m] = np.nan
        layers[name] = a
    if not layers:
        raise FileNotFoundError(f"No layers in {layer_dirs}. Run python -m src.download.gee_features first.")
    return layers, transform


# ---------------------------------------------------------------- main steps
def build_1km_table(layers: dict, cells: pd.DataFrame, factor: int, res_m: int, radii_m: dict) -> pd.DataFrame:
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    table = cells[["cell_id", "district", "x", "y", "lon", "lat"]].copy()
    table["population_total"] = coarsen(layers["population"], factor, "sum")[rows, cols]
    for name, a in layers.items():
        cell_mean = coarsen(a, factor, "mean")
        table[name] = cell_mean[rows, cols]
        for label, radius_m in radii_m.items():
            table[f"{name}_r{label}"] = focal_mean(cell_mean, radius_m / res_m)[rows, cols]
    return table


def build_train_table(layers: dict, transform, labels: pd.DataFrame, districts: gpd.GeoDataFrame, crs: str,
                      buffer_m: float):
    """One row per cluster: shares, household count, district and features.

    Features are means inside a circle of `buffer_m` metres, the same radius for every cluster
    (see model.buffer_m in config.yaml). The radius in the labels file is ignored.
    """
    labels = labels[labels["has_gps"] == 1].reset_index(drop=True)
    pts = gpd.GeoDataFrame(
        labels[["cluster"]], geometry=gpd.points_from_xy(labels["lon"], labels["lat"]), crs="EPSG:4326"
    ).to_crs(crs)
    xs, ys = pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()

    feats = pd.DataFrame(buffer_means(layers, transform, xs, ys, np.full(len(xs), buffer_m)))

    # district name for spatial cross-validation; displaced points near a border get the nearest one
    dist = districts[["shapeName", "geometry"]].rename(columns={"shapeName": "district"})
    joined = gpd.sjoin_nearest(pts, dist, how="left")
    joined = joined[~joined["cluster"].duplicated()].set_index("cluster")["district"]

    keep = ["cluster", "n_households", "sum_weight", "urban", "buffer_m", *SHARES, "share_grid_unreliable"]
    table = labels[keep].copy()
    table["buffer_m"] = buffer_m
    table["district"] = table["cluster"].map(joined)
    return pd.concat([table, feats], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()

    check_restricted_is_ignored()
    cfg = load_config()
    ccfg = cfg["countries"][args.country]
    res_m, feat_res_m = cfg["grid"]["resolution_m"], cfg["grid"]["feature_resolution_m"]
    factor = res_m // feat_res_m
    processed = repo_path(cfg["paths"]["processed"])

    # ---- 1 km grid
    spec = json.loads((processed / f"{args.country}_grid_spec.json").read_text())
    with rasterio.open(processed / f"{args.country}_grid_1km.tif") as src:
        cell_id = src.read(1)
    inside_100m = np.repeat(np.repeat(cell_id >= 0, factor, axis=0), factor, axis=1)
    cells = pd.read_csv(processed / f"{args.country}_grid_cells.csv")

    # GEE layers, plus distance layers if src.distance_layers has been run
    layer_dirs = [repo_path(cfg["gee"]["out_dir"]), processed / "layers"]
    layers, transform = load_layers(layer_dirs, args.country, inside_100m)
    print(f"{len(layers)} layers: {', '.join(layers)}")
    assert layers["population"].shape == (spec["height"] * factor, spec["width"] * factor)

    radii = {f"{int(m / 1000)}km": m for m in cfg["buffers_m"].values()}   # {"2km": 2000, "5km": 5000}
    table_1km = build_1km_table(layers, cells, factor, res_m, radii)
    table_1km.to_csv(processed / f"{args.country}_features_1km.csv", index=False)
    print(f"1 km table: {len(table_1km):,} cells x {table_1km.shape[1]} columns, "
          f"{int(table_1km.isna().any(axis=1).sum())} cells with a missing value")

    # ---- clusters (restricted)
    restricted = repo_path(cfg["paths"]["restricted"])
    labels = pd.read_csv(restricted / "labels" / f"{args.country}_cluster_labels.csv")
    districts = gpd.read_file(
        repo_path(cfg["boundaries"]["out_dir"]) / f"{ccfg['iso3']}_{ccfg['district_level']}.geojson"
    ).to_crs(ccfg["crs_work"])
    train = build_train_table(layers, transform, labels, districts, ccfg["crs_work"],
                              cfg["model"]["buffer_m"])
    out_dir = restricted / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / f"{args.country}_train_table.csv", index=False)
    print(f"Train table: {len(train)} clusters, {train['district'].nunique()} districts, "
          f"{int(train[list(layers)].isna().any(axis=1).sum())} clusters with a missing feature")
    print("Cluster feature means:")
    print(train[list(layers)].mean().round(2).to_string().replace("\n", "\n  ").join(["  ", ""]))


if __name__ == "__main__":
    main()
