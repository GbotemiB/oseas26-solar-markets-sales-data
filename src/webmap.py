"""Interactive web map: one self-contained HTML file built from the output rasters.

Run (after src.predict, src.uncertainty and src.export_eae):
    python -m src.webmap --country rwanda          # writes docs/map.html
    python -m src.webmap --demo                    # synthetic data, writes docs/demo_map.html

The file opens by double-click (no server, no local fetch). It needs internet only for Leaflet and the
basemap tiles. Layers: market-reality, grid, solar, unserved, do-not-trust overlay, district outlines.
Click a cell for its value, its 80% range, its household count, its do-not-trust flag and its district.

Display only: rasters are resampled to EPSG:4326 with nearest-neighbour (cell values are exact, not blended),
shown as images, and values are rounded to whole percent. The GeoTIFFs in outputs/ stay the data product.
The market-reality colour scale is fixed between 0.4 and 0.85 for display; the raster itself is not stretched.
"""
import argparse
import base64
import io
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colormaps
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.config import load_config, repo_path, run_paths
from src.predict import NODATA
from src.train import CLASSES

DST_CRS = "EPSG:4326"
MARKET_RANGE = (0.4, 0.85)                 # display colour scale only
FLAG_COLOURS = {1: (90, 90, 90, 150), 2: (160, 40, 160, 130), 3: (30, 30, 30, 190)}
LAYERS = {                                # key: (title, colormap, vmin, vmax, note)
    "market": ("Market-reality layer", "magma_r", MARKET_RANGE[0], MARKET_RANGE[1],
               "Unserved share among households without grid. 1 = off-grid solar market has not arrived, "
               "0 = it has succeeded. LOW CONFIDENCE: a hypothesis, only 8% better than a flat map at separating "
               "solar from unserved. Colour scale fixed 0.4 to 0.85 for display. Cells where fewer than 20% of households lack grid are left empty."),
    "grid": ("Grid share", "Blues", 0, 1, "Share of households whose main light is grid electricity."),
    "solar": ("Solar share", "Oranges", 0, 1, "Share whose main light is solar (lantern or home system)."),
    "unserved": ("Unserved share", "Purples", 0, 1, "Share with neither grid nor solar as main light (mostly torch users)."),
}
FLAG_TEXT = {0: "no flag", 1: "wide range (among the widest 25%)", 2: "features outside the range seen at survey clusters",
             3: "wide range and features outside the survey range"}
BANNER = "Model estimate, not a census. Trained on survey clusters (DHS 2025) and open satellite layers."
DEMO_BANNER = "SYNTHETIC DEMO DATA: a made-up country that shows the pipeline runs. Not Rwanda. Nothing here is real."


# ---------------------------------------------------------------- pure helpers
def to_percent(array: np.ndarray) -> np.ndarray:
    """Whole percent (0 to 100) as int16, -1 where the raster has no value (NODATA or NaN)."""
    out = np.full(array.shape, -1, dtype=np.int16)
    ok = (array != NODATA) & ~np.isnan(array)
    out[ok] = np.rint(100 * array[ok]).astype(np.int16)
    return out


def to_count(array: np.ndarray) -> np.ndarray:
    """Whole households as int16, -1 where there is no value."""
    out = np.full(array.shape, -1, dtype=np.int16)
    ok = (array != NODATA) & ~np.isnan(array)
    out[ok] = np.clip(np.rint(array[ok]), 0, 32000).astype(np.int16)
    return out


def count_layer(vmax: float) -> tuple:
    return ("Unserved households per cell", "Reds", 0, int(vmax),
            "Households with neither grid nor solar as main light, per 1 km cell (population scaled to the national "
            "household count). Use it with the market-reality layer: a high ratio over few households is small need. "
            "Colour scale stops at the 99th percentile.")


def colorize(array: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    """RGBA uint8 image of a share raster. No-data cells are fully transparent."""
    ok = (array != NODATA) & ~np.isnan(array)
    rgba = (colormaps[cmap](np.clip((array - vmin) / (vmax - vmin), 0, 1)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def flag_image(codes: np.ndarray) -> np.ndarray:
    """RGBA image of the do-not-trust codes: transparent where the cell is fine or empty."""
    rgba = np.zeros(codes.shape + (4,), dtype=np.uint8)
    for code, colour in FLAG_COLOURS.items():
        rgba[codes == code] = colour
    return rgba


def png_data_uri(rgba: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def resample_nearest(array: np.ndarray, src_transform, src_crs, dst_transform, dst_shape: tuple, nodata) -> np.ndarray:
    out = np.full(dst_shape, nodata, dtype=array.dtype)
    reproject(array, out, src_transform=src_transform, src_crs=src_crs, src_nodata=nodata,
              dst_transform=dst_transform, dst_crs=DST_CRS, dst_nodata=nodata, resampling=Resampling.nearest)
    return out


def ramp_css(cmap: str) -> str:
    stops = [colormaps[cmap](t) for t in np.linspace(0, 1, 6)]
    return "linear-gradient(to right, " + ", ".join(
        f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})" for r, g, b, _ in stops) + ")"


def build_payload(arrays: dict, names: list, bounds: list, banner: str, layers: dict = None) -> dict:
    """Everything the page needs, as plain JSON: bounds, grid shape, district names, popup values, text."""
    layers = LAYERS if layers is None else layers
    shape = next(iter(arrays.values())).shape
    return {"bounds": bounds, "height": shape[0], "width": shape[1], "banner": banner, "districts": names,
            "values": {key: [int(v) for v in a.ravel()] for key, a in arrays.items()},
            "layers": {k: {"title": t, "note": n, "ramp": ramp_css(c), "min": lo, "max": hi}
                       for k, (t, c, lo, hi, n) in layers.items()},
            "flags": {str(k): v for k, v in FLAG_TEXT.items()}}


def render_html(payload: dict, images: dict, outlines: dict, title: str) -> str:
    return (TEMPLATE.replace("__TITLE__", title)
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
            .replace("__IMAGES__", json.dumps(images, separators=(",", ":")))
            .replace("__OUTLINES__", json.dumps(outlines, separators=(",", ":"))))


# ---------------------------------------------------------------- build
def read_layer(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def build(country: str, demo: bool, out_path) -> dict:
    cfg = load_config()
    paths = run_paths(cfg, country, demo)
    out_dir = paths["out_dir"]
    files = {c: f"{country}_share_{c}.tif" for c in CLASSES}
    files["market"] = f"{country}_market_reality_eae_utm.tif"
    for c in CLASSES:
        files[f"{c}_lo"], files[f"{c}_hi"] = f"{country}_share_{c}_low.tif", f"{country}_share_{c}_high.tif"
    files["flag"] = f"{country}_do_not_trust.tif"
    files["hh_unserved"], files["hh"] = f"{country}_unserved_households.tif", f"{country}_households.tif"

    raw = {key: read_layer(out_dir / name) for key, name in files.items()}
    array, transform, crs = raw["grid"]
    height, width = array.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    dst_transform, dst_w, dst_h = calculate_default_transform(crs, DST_CRS, width, height,
                                                              left=left, bottom=bottom, right=right, top=top)
    shape = (dst_h, dst_w)
    geo = {key: resample_nearest(a.astype("float32"), t, c, dst_transform, shape, NODATA) for key, (a, t, c) in raw.items()}

    # district of each cell, from the grid table
    cells = pd.read_csv(paths["grid_cells"])
    names = sorted(cells["district"].dropna().unique())
    index = np.full((height, width), -1, dtype=np.int16)
    index[cells["row"].to_numpy(), cells["col"].to_numpy()] = cells["district"].map({n: i for i, n in enumerate(names)}).fillna(-1).to_numpy()
    geo_district = resample_nearest(index, transform, crs, dst_transform, shape, -1)

    west, north = dst_transform * (0, 0)
    east, south = dst_transform * (dst_w, dst_h)
    values = {key: to_percent(a) for key, a in geo.items() if key not in ("flag", "hh_unserved", "hh")}
    values["hh_unserved"], values["hh"] = to_count(geo["hh_unserved"]), to_count(geo["hh"])
    values["flag"] = np.where(geo["flag"] == NODATA, -1, np.rint(geo["flag"])).astype(np.int16)
    values["district"] = geo_district
    layers = {**LAYERS, "hh_unserved": count_layer(np.quantile(geo["hh_unserved"][geo["hh_unserved"] != NODATA], 0.99))}
    payload = build_payload(values, names, [[south, west], [north, east]], DEMO_BANNER if demo else BANNER, layers)

    images = {key: png_data_uri(colorize(geo[key], spec[1], spec[2], spec[3])) for key, spec in layers.items()}
    images["flag"] = png_data_uri(flag_image(np.where(geo["flag"] == NODATA, -1, np.rint(geo["flag"]))))

    outlines = {"type": "FeatureCollection", "features": []}
    boundary_file = repo_path(cfg["boundaries"]["out_dir"]) / "RWA_ADM2.geojson"
    if not demo and boundary_file.exists():
        districts = gpd.read_file(boundary_file)[["shapeName", "geometry"]]
        districts["geometry"] = districts.geometry.simplify(0.003)
        outlines = json.loads(districts.to_json())
    elif not demo:
        print(f"No district outlines: {boundary_file} not found")

    title = "Synthetic demo map" if demo else f"{country.capitalize()} off-grid solar access map"
    html = render_html(payload, images, outlines, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return {"path": out_path, "bytes": len(html.encode()), "cells": int((values["grid"] >= 0).sum()), "shape": shape}


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<style>
  html, body { height: 100%; margin: 0; font: 14px/1.4 system-ui, sans-serif; }
  #map { height: 100%; }
  .leaflet-image-layer { image-rendering: pixelated; image-rendering: crisp-edges; }
  #banner { position: absolute; z-index: 1000; top: 0; left: 0; right: 0; background: #7a1f1f; color: #fff;
            padding: 6px 12px; font-weight: 600; text-align: center; }
  #panel { position: absolute; z-index: 1000; top: 44px; right: 10px; width: 290px; background: #fff;
           padding: 10px 12px; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,.35); max-height: calc(100% - 70px); overflow: auto; }
  #panel h3 { margin: 0 0 6px; font-size: 15px; }
  #panel label { display: block; margin: 2px 0; cursor: pointer; }
  #panel .note { font-size: 12px; color: #444; margin: 6px 0; }
  #bar { height: 12px; border-radius: 3px; margin-top: 4px; }
  #scale { display: flex; justify-content: space-between; font-size: 11px; color: #333; }
  .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: -2px; margin-right: 4px; border: 1px solid #666; }
  select, input[type=range] { width: 100%; }
  .leaflet-popup-content { font-size: 13px; }
</style></head>
<body>
<div id="banner"></div>
<div id="map"></div>
<div id="panel">
  <h3>__TITLE__</h3>
  <div><b>Layer</b></div>
  <div id="layers"></div>
  <div id="note" class="note"></div>
  <div id="bar"></div><div id="scale"><span id="lo"></span><span id="hi"></span></div>
  <div style="margin-top:8px"><label><input type="checkbox" id="flag" checked> Do-not-trust overlay</label>
    <div class="note"><span class="swatch" style="background:rgba(90,90,90,.6)"></span>wide range
    <span class="swatch" style="background:rgba(160,40,160,.55)"></span>outside survey range
    <span class="swatch" style="background:rgba(30,30,30,.75)"></span>both</div></div>
  <label><input type="checkbox" id="outlines" checked> District outlines</label>
  <div style="margin-top:6px"><b>Opacity</b><input type="range" id="opacity" min="10" max="100" value="80"></div>
  <div style="margin-top:6px"><b>Basemap</b><select id="base"><option value="light">Light grey</option><option value="imagery">Satellite imagery</option></select></div>
  <div class="note">Click a cell for its value, range and flag. Display only: values rounded to whole percent, cells shown as 1 km squares.</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const P = __PAYLOAD__, IMAGES = __IMAGES__, OUTLINES = __OUTLINES__;
document.getElementById('banner').textContent = P.banner;
const map = L.map('map', {zoomSnap: 0.25}).fitBounds(P.bounds);
const esri = (name) => L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/' + name + '/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 16, attribution: 'Tiles &copy; Esri'});
const bases = {light: esri('Canvas/World_Light_Gray_Base'), imagery: esri('World_Imagery')};
let base = bases.light.addTo(map);
document.getElementById('base').onchange = (e) => { map.removeLayer(base); base = bases[e.target.value].addTo(map); base.bringToBack(); };

const overlays = {};
for (const key of Object.keys(P.layers)) overlays[key] = L.imageOverlay(IMAGES[key], P.bounds, {opacity: 0.8, interactive: false});
const flagLayer = L.imageOverlay(IMAGES.flag, P.bounds, {opacity: 0.9, interactive: false}).addTo(map);
const outlineLayer = L.geoJSON(OUTLINES, {style: {color: '#333', weight: 1, fill: false, opacity: 0.6}}).addTo(map);
let current = null;

const layerBox = document.getElementById('layers');
Object.keys(P.layers).forEach((key, i) => {
  const l = document.createElement('label');
  l.innerHTML = '<input type="radio" name="layer" value="' + key + '"' + (i === 0 ? ' checked' : '') + '> ' + P.layers[key].title;
  layerBox.appendChild(l);
});
function show(key) {
  if (current) map.removeLayer(overlays[current]);
  current = key;
  overlays[key].addTo(map).setOpacity(document.getElementById('opacity').value / 100);
  flagLayer.bringToFront();
  const info = P.layers[key];
  document.getElementById('note').textContent = info.note;
  document.getElementById('bar').style.background = info.ramp;
  document.getElementById('lo').textContent = info.min;
  document.getElementById('hi').textContent = info.max;
}
layerBox.onchange = (e) => show(e.target.value);
document.getElementById('opacity').oninput = (e) => overlays[current].setOpacity(e.target.value / 100);
document.getElementById('flag').onchange = (e) => e.target.checked ? flagLayer.addTo(map) : map.removeLayer(flagLayer);
document.getElementById('outlines').onchange = (e) => e.target.checked ? outlineLayer.addTo(map) : map.removeLayer(outlineLayer);
show('market');

const pct = (v) => v < 0 ? 'no data' : v + '%';
map.on('click', (e) => {
  const [[south, west], [north, east]] = P.bounds;
  const col = Math.floor((e.latlng.lng - west) / (east - west) * P.width);
  const row = Math.floor((north - e.latlng.lat) / (north - south) * P.height);
  if (row < 0 || col < 0 || row >= P.height || col >= P.width) return;
  const i = row * P.width + col, v = P.values;
  if (v.grid[i] < 0) { L.popup().setLatLng(e.latlng).setContent('No estimate here (nobody lives in this cell, or a feature is missing).').openOn(map); return; }
  const line = (name, k) => name + ': <b>' + pct(v[k][i]) + '</b> (80% range ' + v[k + '_lo'][i] + ' to ' + v[k + '_hi'][i] + '%)';
  const market = v.market[i] < 0 ? 'no value (almost everyone has grid)' : '<b>' + (v.market[i] / 100).toFixed(2) + '</b> (1 = market has not arrived, 0 = served; low confidence)';
  const d = v.district[i] >= 0 ? P.districts[v.district[i]] : 'outside districts';
  L.popup().setLatLng(e.latlng).setContent(
    '<b>' + d + '</b><br>' + line('Grid', 'grid') + '<br>' + line('Solar', 'solar') + '<br>' + line('Unserved', 'unserved') +
    '<br>Unserved households: <b>~' + v.hh_unserved[i] + '</b> of ~' + v.hh[i] + ' (range ' +
    Math.round(v.hh[i] * v.unserved_lo[i] / 100) + ' to ' + Math.round(v.hh[i] * v.unserved_hi[i] / 100) + ')' +
    '<br>Market-reality: ' + market + '<br>Do-not-trust flag: ' + P.flags[String(v.flag[i])]).openOn(map);
});
</script>
</body></html>
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    parser.add_argument("--demo", action="store_true", help="build the map from the synthetic outputs in outputs/demo/")
    parser.add_argument("--out", default=None, help="output file (default docs/map.html, or docs/demo_map.html)")
    args = parser.parse_args(argv)
    country = "demo" if args.demo else args.country
    default = "docs/demo_map.html" if args.demo else f"docs/{'map' if country == 'rwanda' else country + '_map'}.html"
    result = build(country, args.demo, repo_path(args.out if args.out else default))
    print(f"Wrote {result['path']} ({result['bytes'] / 1e6:.2f} MB, {result['shape'][1]} x {result['shape'][0]} pixels, "
          f"{result['cells']:,} cells with a value)")


if __name__ == "__main__":
    main()
