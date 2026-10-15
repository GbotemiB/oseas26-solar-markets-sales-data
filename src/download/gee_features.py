"""Export feature layers from Google Earth Engine (GEE) at 100 m, lined up with the 1 km grid.

Run (after `python -m src.grid`):
    earthengine authenticate                     # once per computer
    python -m src.download.gee_features --country rwanda
    python -m src.download.gee_features --layers population night_lights
    python -m src.download.gee_features --drive  # batch export to Google Drive instead

Each layer is saved to data/raw/gee/<country>_<layer>.tif with a .json next to it
(source, licence, unit). Pixels with no data are -9999.

Small layers download directly. If a layer is too big or GEE times out, rerun that layer
with --drive, then download the file from the Drive folder in config.yaml into data/raw/gee/.

DHS rule: this script never sends cluster locations to GEE. Cluster averages are computed
on your own machine in src/features.py.
"""
import argparse
import json

import ee
import requests

from src.config import load_config, repo_path

NODATA = -9999
MAX_DIRECT_BYTES = 30e6   # GEE direct download limit is about 32 MB
TILE_PX = 2048            # larger layers are downloaded in tiles of this size (about 17 MB as float32)


def target_projection(spec: dict, res: int) -> ee.Projection:
    return ee.Projection(spec["crs"], [res, 0, spec["xmin"], 0, -res, spec["ymax"]])


def aggregate(img: ee.Image, proj: ee.Projection, reducer: ee.Reducer, steps=(10,)) -> ee.Image:
    """Aggregate a fine image to the target grid. `steps` adds intermediate sizes (metres)
    so GEE does not have to combine too many pixels in one go."""
    for step in steps:
        mid = proj.atScale(step)
        img = img.reduceResolution(reducer, maxPixels=65535).reproject(mid)
    return img.reduceResolution(reducer, maxPixels=65535).reproject(proj)


def build_layers(cfg: dict, iso3: str, proj: ee.Projection, region: ee.Geometry) -> dict:
    f = cfg["gee_features"]
    res = cfg["grid"]["feature_resolution_m"]
    L = {}

    # Population: convert to density first so totals are kept when the pixel size changes.
    wp = (ee.ImageCollection("WorldPop/GP/100m/pop")
          .filter(ee.Filter.eq("country", iso3))
          .filter(ee.Filter.eq("year", f["worldpop_year"])).first())
    density = wp.divide(ee.Image.pixelArea())
    L["population"] = dict(
        image=aggregate(density, proj, ee.Reducer.mean(), steps=()).multiply(res * res),
        source=f"WorldPop/GP/100m/pop ({f['worldpop_year']})", licence="CC BY 4.0",
        unit="people per 100 m cell")

    # Night lights, two products from the same VIIRS (Visible Infrared Imaging Radiometer Suite) sensor.
    # NOAA: monthly composites with cloud masking only.
    viirs_col = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
                 .filterDate(f["viirs_start"], f["viirs_end"]).select("avg_rad"))
    viirs = viirs_col.median().max(0).setDefaultProjection(viirs_col.first().projection())
    L["night_lights_noaa"] = dict(
        image=viirs.resample("bilinear").reproject(proj),
        source=f"NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG median {f['viirs_start']} to {f['viirs_end']}",
        licence="Free to use, cite Earth Observation Group (confirm)", unit="nW/cm2/sr")

    # NASA Black Marble: daily, corrected for moonlight, atmosphere and snow, gaps filled.
    bm_col = (ee.ImageCollection("NASA/VIIRS/002/VNP46A2")
              .filterDate(f["viirs_start"], f["viirs_end"]).select("Gap_Filled_DNB_BRDF_Corrected_NTL"))
    bm = bm_col.median().max(0).setDefaultProjection(bm_col.first().projection())
    L["night_lights_bm"] = dict(
        image=bm.resample("bilinear").reproject(proj),
        source=f"NASA/VIIRS/002/VNP46A2 (Black Marble) median {f['viirs_start']} to {f['viirs_end']}",
        licence="NASA open data, cite Black Marble (confirm)", unit="nW/cm2/sr")

    # Built-up surface: GHSL (Global Human Settlement Layer), native 100 m, so it aggregates directly.
    # (Google Open Buildings is 0.5 m and too big for GEE to aggregate over a whole country.)
    ghsl = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/" + str(f["ghsl_epoch"]))
            .select("built_surface"))
    L["built_up_surface"] = dict(
        image=aggregate(ghsl, proj, ee.Reducer.sum(), steps=()),
        source=f"JRC/GHSL/P2023A/GHS_BUILT_S ({f['ghsl_epoch']})", licence="CC BY 4.0",
        unit="built-up surface in m2 per 100 m cell")

    # Travel time (Malaria Atlas Project)
    L["travel_time_city"] = dict(
        image=ee.Image("projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0").select("accessibility")
              .resample("bilinear").reproject(proj),
        source="projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0", licence="CC BY 4.0",
        unit="minutes to nearest city (50k+ people)")
    L["travel_time_health"] = dict(
        image=ee.Image("projects/malariaatlasproject/assets/accessibility/accessibility_to_healthcare/2019").select("accessibility")
              .resample("bilinear").reproject(proj),
        source="projects/malariaatlasproject/assets/accessibility/accessibility_to_healthcare/2019", licence="CC BY 4.0",
        unit="minutes to nearest health facility (motorised)")

    # Terrain (SRTM, Shuttle Radar Topography Mission)
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = aggregate(srtm.select("elevation"), proj, ee.Reducer.mean(), steps=())
    L["elevation"] = dict(image=elevation, source="USGS/SRTMGL1_003", licence="Public domain", unit="metres")
    # Slope from the 100 m elevation (slope at 30 m runs out of GEE memory)
    L["slope"] = dict(image=ee.Terrain.slope(elevation),
                      source="USGS/SRTMGL1_003 (ee.Terrain.slope on 100 m mean elevation)",
                      licence="Public domain", unit="degrees")
    return L


def download_tiled(img, name, out_path, spec, res, width, height, tile_px=TILE_PX, workers=4) -> None:
    """Download a layer that is too big for one request, in square tiles, into one GeoTIFF.

    Each tile is a direct GEE download (at most tile_px x tile_px float32 pixels). The tiles are
    written into a .part file and renamed when all are done, so a failed run leaves no half file.
    """
    import io
    import time
    from concurrent.futures import ThreadPoolExecutor

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    tiles = [(r0, c0, min(tile_px, height - r0), min(tile_px, width - c0))
             for r0 in range(0, height, tile_px) for c0 in range(0, width, tile_px)]

    def fetch(tile):
        r0, c0, h, w = tile
        tf = [res, 0, spec["xmin"] + c0 * res, 0, -res, spec["ymax"] - r0 * res]
        for attempt in range(1, 6):
            try:
                url = img.getDownloadURL({"name": f"{out_path.stem}_{r0}_{c0}", "crs": spec["crs"],
                                          "crs_transform": tf, "dimensions": f"{w}x{h}", "format": "GEO_TIFF"})
                r = requests.get(url, timeout=600)
                if r.status_code == 200:
                    with rasterio.open(io.BytesIO(r.content)) as src:
                        return tile, src.read(1)
                err = f"{r.status_code}: {r.text[:150]}"
            except Exception as exc:          # network or GEE error: wait and try again
                err = repr(exc)[:150]
            time.sleep(5 * attempt)
        raise RuntimeError(f"{name}: tile row {r0} col {c0} failed after 5 attempts ({err})")

    part = out_path.with_name(out_path.name + ".part")
    print(f"  {name}: too big for one request, downloading {len(tiles)} tiles of up to {tile_px} px")
    with rasterio.open(part, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32",
                       crs=spec["crs"], transform=from_origin(spec["xmin"], spec["ymax"], res, res),
                       nodata=NODATA, compress="deflate", tiled=True, BIGTIFF="YES") as dst, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        for n, ((r0, c0, h, w), data) in enumerate(pool.map(fetch, tiles), 1):
            dst.write(data.astype("float32"), 1, window=Window(c0, r0, w, h))
            if n % 10 == 0 or n == len(tiles):
                print(f"    {n}/{len(tiles)} tiles")
    part.rename(out_path)
    print(f"  {name}: saved {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")


def export_layer(name, layer, spec, res, out_dir, drive_folder, use_drive) -> None:
    width, height = spec["width"] * spec["resolution_m"] // res, spec["height"] * spec["resolution_m"] // res
    transform = [res, 0, spec["xmin"], 0, -res, spec["ymax"]]
    img = layer["image"].toFloat().unmask(NODATA).rename(name)
    file_stem = f"{spec['country']}_{name}"
    meta = {k: v for k, v in layer.items() if k != "image"}
    meta.update(nodata=NODATA, crs=spec["crs"], resolution_m=res)
    (out_dir / f"{file_stem}.json").write_text(json.dumps(meta, indent=2))

    too_big = width * height * 4 > MAX_DIRECT_BYTES
    if too_big and not use_drive:
        out_path = out_dir / f"{file_stem}.tif"
        if out_path.exists():
            print(f"  {name}: already there, skipping")
            return
        download_tiled(img, name, out_path, spec, res, width, height)
        return
    if use_drive:
        task = ee.batch.Export.image.toDrive(
            image=img, description=file_stem, folder=drive_folder, fileNamePrefix=file_stem,
            crs=spec["crs"], crsTransform=transform, dimensions=f"{width}x{height}", maxPixels=1e10)
        task.start()
        print(f"  {name}: Drive export started (task {task.id}). "
              f"When done, download {file_stem}.tif from Drive folder '{drive_folder}' into {out_dir}")
        return

    out_path = out_dir / f"{file_stem}.tif"
    if out_path.exists():
        print(f"  {name}: already there, skipping")
        return
    url = img.getDownloadURL({"name": file_stem, "crs": spec["crs"], "crs_transform": transform,
                              "dimensions": f"{width}x{height}", "format": "GEO_TIFF"})
    r = requests.get(url, timeout=600)
    if r.status_code != 200:
        print(f"  {name}: FAILED ({r.status_code}): {r.text[:200]}\n"
              f"    -> rerun with: --layers {name} --drive")
        return
    out_path.write_bytes(r.content)
    print(f"  {name}: saved {out_path.name} ({len(r.content) / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="rwanda")
    parser.add_argument("--layers", nargs="*", help="only these layers (default: all)")
    parser.add_argument("--drive", action="store_true", help="export to Google Drive")
    args = parser.parse_args()

    cfg = load_config()
    ee.Initialize(project=cfg["gee"]["project"])

    spec_path = repo_path(cfg["paths"]["processed"]) / f"{args.country}_grid_spec.json"
    if not spec_path.exists():
        raise SystemExit(f"Missing {spec_path}. Run `python -m src.grid --country {args.country}` first.")
    spec = json.loads(spec_path.read_text())
    res = cfg["grid"]["feature_resolution_m"]

    out_dir = repo_path(cfg["gee"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rectangle covering the whole grid, used to limit which tiles GEE has to look at
    region = ee.Geometry.Rectangle(
        [spec["xmin"], spec["ymax"] - spec["height"] * spec["resolution_m"],
         spec["xmin"] + spec["width"] * spec["resolution_m"], spec["ymax"]],
        proj=spec["crs"], geodesic=False)
    layers = build_layers(cfg, cfg["countries"][args.country]["iso3"], target_projection(spec, res), region)
    for name in args.layers or layers:
        print(f"- {name}")
        export_layer(name, layers[name], spec, res, out_dir, cfg["gee"]["drive_folder"], args.drive)


if __name__ == "__main__":
    main()
