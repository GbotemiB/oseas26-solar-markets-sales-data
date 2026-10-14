"""Build 100 m "distance to nearest ..." layers from Gridfinder and OpenStreetMap (OSM).

Run (after src.grid, src.download.gridfinder, src.download.osm):
    python -m src.distance_layers --country rwanda

Outputs, in the same 100 m grid as the GEE layers (data/processed/layers/):
    <country>_dist_gridfinder.tif   km to nearest Gridfinder predicted grid line (data from 2019)
    <country>_dist_power_osm.tif    km to nearest OSM power line or minor line
    <country>_dist_road_major.tif   km to nearest OSM motorway, trunk, primary or secondary road
    <country>_dist_town.tif         km to nearest OSM city or town point
    <country>_dist_market.tif       km to nearest OSM marketplace point
Each has a .json next to it (source, licence, unit). src.features picks these up with the GEE layers.

Known limits (state them in the limitations statement):
    - Gridfinder is a 2019 model output; the grid has grown since.
    - OSM power lines are very incomplete in Rwanda (about 120 mapped line segments for the whole
      country), so "far from OSM power" often just means "not mapped".
    - OSM markets are only a few dozen points.
Features near the border use features from the neighbouring country too (30 km margin).
"""
import argparse
import json

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

from src.config import load_config, repo_path

PAD_PX = 300        # margin of 30 km around the grid, so features just across the border still count

OSM_QUERIES = {
    "dist_power_osm": (
        "SELECT other_tags FROM lines WHERE other_tags LIKE '%\"power\"=>\"line\"%' "
        "OR other_tags LIKE '%\"power\"=>\"minor_line\"%'"),
    "dist_road_major": "SELECT highway FROM lines WHERE highway IN ('motorway', 'trunk', 'primary', 'secondary')",
    "dist_town": "SELECT place FROM points WHERE place IN ('city', 'town')",
    "dist_market": "SELECT other_tags FROM points WHERE other_tags LIKE '%\"amenity\"=>\"marketplace\"%'",
}


def distance_to_features(geoms, transform, shape, pad_px: int = PAD_PX) -> np.ndarray:
    """Distance in km from every pixel centre to the nearest pixel touched by `geoms`.

    `geoms` must be in the raster CRS (metres). The raster is padded on every side before the
    distance is computed, so features just outside the raster are taken into account.
    """
    res = transform.a
    height, width = shape
    big_transform = from_origin(transform.c - pad_px * res, transform.f + pad_px * res, res, res)
    big_shape = (height + 2 * pad_px, width + 2 * pad_px)
    hit = rasterize(list(geoms), out_shape=big_shape, transform=big_transform,
                    fill=0, default_value=1, all_touched=True, dtype="uint8")
    if not hit.any():
        raise ValueError("No features inside the raster or its margin")
    dist_m = distance_transform_edt(hit == 0, sampling=res)
    return (dist_m[pad_px:pad_px + height, pad_px:pad_px + width] / 1000).astype("float32")


def read_osm(pbf, query: str, crs: str) -> gpd.GeoSeries:
    gdf = pyogrio.read_dataframe(pbf, sql=query, sql_dialect="OGRSQL")
    return gdf.geometry.to_crs(crs)


def read_gridfinder(gpkg, bounds_work: tuple, crs: str) -> gpd.GeoSeries:
    """Read only the lines near the country from the 725 MB world file."""
    box = gpd.GeoSeries.from_xy([bounds_work[0], bounds_work[2]], [bounds_work[1], bounds_work[3]], crs=crs)
    minx, miny, maxx, maxy = box.to_crs("EPSG:4326").total_bounds
    gdf = pyogrio.read_dataframe(gpkg, bbox=(minx, miny, maxx, maxy))
    return gdf.geometry.to_crs(crs)


def write_layer(path, array: np.ndarray, transform, crs: str, meta: dict) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=-9999, compress="deflate") as dst:
        dst.write(array, 1)
    path.with_suffix(".json").write_text(json.dumps({**meta, "unit": "km", "nodata": -9999}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()

    cfg = load_config()
    crs = cfg["countries"][args.country]["crs_work"]
    res = cfg["grid"]["feature_resolution_m"]
    spec = json.loads((repo_path(cfg["paths"]["processed"]) / f"{args.country}_grid_spec.json").read_text())
    transform = from_origin(spec["xmin"], spec["ymax"], res, res)
    shape = (spec["height"] * spec["resolution_m"] // res, spec["width"] * spec["resolution_m"] // res)
    out_dir = repo_path(cfg["paths"]["processed"]) / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    margin = PAD_PX * res
    bounds = (spec["xmin"] - margin, spec["ymax"] - shape[0] * res - margin,
              spec["xmin"] + shape[1] * res + margin, spec["ymax"] + margin)

    layers = {"dist_gridfinder": (
        read_gridfinder(repo_path(cfg["downloads"]["gridfinder_dir"]) / "grid.gpkg", bounds, crs),
        {"source": "Gridfinder predicted grid lines (Zenodo 3628142), data from 2019", "licence": "CC BY 4.0"})}
    pbf = repo_path(cfg["downloads"]["osm_dir"]) / f"{args.country}-latest.osm.pbf"
    for name, query in OSM_QUERIES.items():
        layers[name] = (read_osm(pbf, query, crs),
                        {"source": "OpenStreetMap via Geofabrik", "licence": "ODbL, credit OpenStreetMap contributors"})

    for name, (geoms, meta) in layers.items():
        dist = distance_to_features(geoms, transform, shape)
        write_layer(out_dir / f"{args.country}_{name}.tif", dist, transform, crs, meta)
        print(f"{name}: {len(geoms):,} features, distance km: mean {dist.mean():.1f}, "
              f"median {np.median(dist):.1f}, max {dist.max():.1f}")


if __name__ == "__main__":
    main()
