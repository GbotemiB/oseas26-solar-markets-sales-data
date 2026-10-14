"""Build the 1 km grid for one country.

Run:  python -m src.grid --country rwanda
Needs: data/raw/boundaries/<ISO3>_ADM0.geojson and the district level (python -m src.download.boundaries)

Outputs (data/processed/, git-ignored):
    <country>_grid_1km.tif      cell_id per 1 km cell (-1 outside the country)
    <country>_grid_cells.csv    cell_id, row, col, x, y, lon, lat, district
    <country>_grid_spec.json    grid origin and size. The GEE (Google Earth Engine) script
                                reads this so 100 m features line up exactly with the 1 km grid.
"""
import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from src.config import load_config, repo_path


def build_grid(adm0: gpd.GeoDataFrame, res: int):
    """Grid that covers the country, with edges snapped to whole multiples of `res`."""
    minx, miny, maxx, maxy = adm0.total_bounds
    xmin = np.floor(minx / res) * res - res      # one cell of padding on each side
    ymax = np.ceil(maxy / res) * res + res
    xmax = np.ceil(maxx / res) * res + res
    ymin = np.floor(miny / res) * res - res
    width, height = int((xmax - xmin) / res), int((ymax - ymin) / res)
    transform = from_origin(xmin, ymax, res, res)
    inside = rasterize(
        adm0.geometry, out_shape=(height, width), transform=transform,
        fill=0, default_value=1, all_touched=True, dtype="uint8",
    ).astype(bool)
    return xmin, ymax, width, height, transform, inside


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()

    cfg = load_config()
    ccfg = cfg["countries"][args.country]
    crs, res = ccfg["crs_work"], cfg["grid"]["resolution_m"]
    bdir = repo_path(cfg["boundaries"]["out_dir"])

    adm0 = gpd.read_file(bdir / f"{ccfg['iso3']}_ADM0.geojson").to_crs(crs)
    districts = gpd.read_file(bdir / f"{ccfg['iso3']}_{ccfg['district_level']}.geojson").to_crs(crs)

    xmin, ymax, width, height, transform, inside = build_grid(adm0, res)

    # cell ids: running number over cells inside the country, -1 outside
    cell_id = np.full((height, width), -1, dtype="int32")
    cell_id[inside] = np.arange(inside.sum(), dtype="int32")

    # table of cells with centre coordinates and district
    rows, cols = np.nonzero(inside)
    x = xmin + (cols + 0.5) * res
    y = ymax - (rows + 0.5) * res
    cells = gpd.GeoDataFrame(
        {"cell_id": cell_id[rows, cols], "row": rows, "col": cols, "x": x, "y": y},
        geometry=gpd.points_from_xy(x, y), crs=crs,
    )
    dist = districts[["shapeName", "geometry"]].rename(columns={"shapeName": "district"})
    # border cells whose centre falls just outside every district get the nearest one
    cells = gpd.sjoin_nearest(cells, dist, how="left").drop(columns="index_right")
    cells = cells[~cells.cell_id.duplicated()]          # ties on a border: keep first
    ll = cells.to_crs("EPSG:4326").geometry
    cells["lon"], cells["lat"] = ll.x.round(6), ll.y.round(6)

    out = repo_path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out / f"{args.country}_grid_1km.tif", "w", driver="GTiff", height=height, width=width,
        count=1, dtype="int32", crs=crs, transform=transform, nodata=-1, compress="deflate",
    ) as dst:
        dst.write(cell_id, 1)
    pd.DataFrame(cells.drop(columns="geometry")).to_csv(out / f"{args.country}_grid_cells.csv", index=False)
    spec = {"country": args.country, "crs": crs, "resolution_m": res,
            "xmin": float(xmin), "ymax": float(ymax), "width": width, "height": height,
            "n_cells_inside": int(inside.sum())}
    (out / f"{args.country}_grid_spec.json").write_text(json.dumps(spec, indent=2))

    print(f"Grid {width} x {height} cells at {res} m, {inside.sum():,} inside {args.country}, "
          f"{cells.district.nunique()} districts")


if __name__ == "__main__":
    main()
