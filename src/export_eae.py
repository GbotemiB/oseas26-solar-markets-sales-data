"""Write the market-reality layer in Energy Access Explorer (EAE) format.

Run (after src.predict):
    python -m src.export_eae --country rwanda
    python -m src.export_eae --demo         # reads and writes outputs/demo/

Value per 1 km cell: unserved / (solar + unserved), the unserved share among NON-grid households.
    1 = the off-grid solar market has failed to arrive (non-grid households are mostly unserved)
    0 = the off-grid solar market has quietly succeeded (non-grid households are mostly on solar)
It is a ratio between 0 and 1 already, so it is NOT min-max stretched (a stretch would make the value
depend on which cells happen to be the extremes). Cells with almost no non-grid households
(non-grid share below eae.min_nongrid_share) are no-data: darkness there says nothing about solar.

Outputs (outputs/):
    <country>_market_reality_eae.tif         EPSG:4326 headline file (config grid.crs_export)
    <country>_market_reality_eae_utm.tif     same layer on the 1 km UTM grid, no resampling
    each with its metadata in the GeoTIFF tags and in a .json next to it

This layer is a LOW-CONFIDENCE HYPOTHESIS: solar versus unserved is only weakly predictable from open
data (see outputs/solar_vs_unserved_test_*.csv). Use it together with <country>_do_not_trust.tif.
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.config import check_restricted_is_ignored, load_config, run_paths
from src.predict import DEMO_NOTICE, NODATA, write_share_raster

DIRECTION = ("1 = the off-grid solar market has failed to arrive (non-grid households are mostly unserved); "
             "0 = the market has quietly succeeded (non-grid households are mostly on solar). "
             "Higher value = more need and more unmet demand. The challenge brief did not fix a direction, "
             "so check it before mixing this layer with EAE layers that run the other way.")


# ---------------------------------------------------------------- pure helpers
def market_reality(grid: np.ndarray, solar: np.ndarray, unserved: np.ndarray, min_nongrid: float) -> np.ndarray:
    """unserved / (solar + unserved), NODATA where any input is missing or the non-grid share is below `min_nongrid`."""
    out = np.full(grid.shape, NODATA, dtype="float32")
    valid = (grid != NODATA) & (solar != NODATA) & (unserved != NODATA)
    nongrid = solar + unserved                              # equals 1 - grid, shares sum to 1
    ok = valid & (nongrid >= min_nongrid)
    out[ok] = np.clip(unserved[ok] / nongrid[ok], 0.0, 1.0)
    return out


def to_geographic(array: np.ndarray, src_transform, src_crs, dst_crs: str):
    """Reproject to `dst_crs` with bilinear resampling that ignores no-data. Returns (array, transform)."""
    height, width = array.shape
    left, top = src_transform * (0, 0)
    right, bottom = src_transform * (width, height)
    dst_transform, dst_w, dst_h = calculate_default_transform(src_crs, dst_crs, width, height,
                                                              left=left, bottom=bottom, right=right, top=top)
    out = np.full((dst_h, dst_w), NODATA, dtype="float32")
    reproject(array, out, src_transform=src_transform, src_crs=src_crs, src_nodata=NODATA,
              dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=NODATA, resampling=Resampling.bilinear)
    out[out != NODATA] = np.clip(out[out != NODATA], 0.0, 1.0)
    return out, dst_transform


def solar_test_sentence(test: pd.DataFrame) -> str:
    """Quote the solar-versus-unserved test result, so the metadata cannot drift from the numbers."""
    row = test.set_index("model").loc["ratio model (7 core features)"]
    return (f"Held-out test (spatial CV, {int(row['n_clusters'])} clusters with non-grid households): the model's error "
            f"for the solar share among non-grid households is {row['mae']:.3f}, only {100 * row['skill']:.0f}% better than "
            f"a flat map, and it is not clearly better than night lights alone (AUC {row['auc_solar_majority']:.2f}).")


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    parser.add_argument("--demo", action="store_true", help="use the synthetic outputs in outputs/demo/")
    args = parser.parse_args(argv)
    if args.demo:
        args.country = "demo"

    check_restricted_is_ignored()
    cfg = load_config()
    ecfg, out_dir = cfg["eae"], run_paths(cfg, args.country, args.demo)["out_dir"]
    c = args.country

    layers = {}
    for cls in ["grid", "solar", "unserved"]:
        with rasterio.open(out_dir / f"{c}_share_{cls}.tif") as src:
            layers[cls] = src.read(1)
            transform, crs = src.transform, src.crs
    layer = market_reality(layers["grid"], layers["solar"], layers["unserved"], ecfg["min_nongrid_share"])

    has_shares = layers["grid"] != NODATA
    ok = layer != NODATA
    dropped = has_shares & ~ok
    print(f"{ok.sum():,} cells with a value; {dropped.sum():,} of {has_shares.sum():,} predicted cells "
          f"({100 * dropped.sum() / has_shares.sum():.1f}%) left empty: non-grid share below {ecfg['min_nongrid_share']}")
    q = np.quantile(layer[ok], [0.05, 0.25, 0.5, 0.75, 0.95])
    print("Value quantiles 5/25/50/75/95%: " + " ".join(f"{v:.2f}" for v in q))

    confidence = ("LOW. Treat as a hypothesis about where the off-grid solar market has not arrived, not a finding. "
                  + (DEMO_NOTICE if args.demo else solar_test_sentence(pd.read_csv(out_dir / f"solar_vs_unserved_test_{c}.csv"))))
    tags = {
        "title": "Market-reality layer: unserved share among non-grid households, 1 km",
        "country": c,
        "unit": "ratio 0 to 1, unserved / (solar + unserved). Not min-max stretched.",
        "direction": DIRECTION,
        "no_data": f"{NODATA}. Cells with no prediction (missing feature, or own population below "
                   f"{cfg['model']['min_cell_population']} per km2) and cells where the non-grid share is below "
                   f"{ecfg['min_nongrid_share']} (the ratio is unstable there).",
        "confidence": confidence,
        "use_with": f"{c}_do_not_trust.tif (flag codes: 1 widest ranges, 2 features outside the survey range, 3 both).",
        "method": "Multinomial logistic regression on 7 open features, trained on DHS cluster shares; layer = "
                  "predicted unserved share divided by predicted (solar + unserved) share. See the share rasters' tags.",
        "sources": "Labels: Rwanda DHS 2025 (DHS-8), restricted; only model-derived outputs are published. Features: "
                   "WorldPop 2020 [CC BY 4.0], GHSL 2020 [CC BY 4.0], NASA Black Marble VIIRS 2024-2025, Malaria Atlas Project "
                   "travel time [CC BY 4.0], SRTM [public domain]. This layer does not use the Meta Relative Wealth Index.",
        "year": "Survey fieldwork June to October 2025. Features 2015 to 2025.",
        "limitations": "Unserved here is mostly households using torches (about 26% nationally), not the official access gap "
                       "(about 12%). Training label is the MAIN lighting source. DHS cluster locations are displaced by up to "
                       "2 km (urban) and 5 km (rural). Night lights and satellite layers cannot see small solar systems. "
                       "Likely to misallocate capital toward places that look dark and remote to the model but already "
                       "have solar, and away from places with solar-like features but no market. "
                       "Not rescaled to GOGLA sales (see outputs/calibration_note_*.md).",
    }

    if args.demo:
        tags.update(sources="Synthetic.", year="None, synthetic.", limitations=DEMO_NOTICE)
    write_share_raster(out_dir / f"{c}_market_reality_eae_{ecfg['utm_file_suffix']}.tif", layer, transform, str(crs),
                       {**tags, "crs_note": "UTM zone 35S, the 1 km model grid, no resampling"})
    geo, geo_transform = to_geographic(layer, transform, crs, cfg["grid"]["crs_export"])
    write_share_raster(out_dir / f"{c}_market_reality_eae.tif", geo, geo_transform, cfg["grid"]["crs_export"],
                       {**tags, "crs_note": "Reprojected from the 1 km UTM grid with bilinear resampling, about 1 km pixels"})
    print(f"Wrote {c}_market_reality_eae.tif ({geo.shape[1]} x {geo.shape[0]}, {cfg['grid']['crs_export']}) "
          f"and {c}_market_reality_eae_{ecfg['utm_file_suffix']}.tif")


if __name__ == "__main__":
    main()
