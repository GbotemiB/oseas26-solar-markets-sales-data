"""Control totals: compare the predicted maps with independent published numbers.

Run (after src.predict and src.labels_open):
    python -m src.validate --country rwanda

Predicted shares are summed per area, weighted by population (WorldPop), and compared with:
  - EICV7 (2023/24 survey) district share with electricity as main light, with its 95% interval
  - Census 2022 sector share with "electricity" as main light
  - EICV7 and Census 2022 grid vs solar split per province
  - Census 2022 grid-connected share per sector (Table C1.5), and a rough solar proxy: electrified (C1.3) minus
    connected (C1.5), clipped at 0. "Connected" is not "grid is the main light" (connected households can light
    with something else), so its LEVEL is higher than our grid share by definition. Read the pattern
    (pearson, spearman, mae_no_bias), not the bias.
"Served" = grid + solar in our maps = 1 - unserved. This is NOT training data: the model never saw
these tables. The numbers come from different years (Census Aug 2022, EICV7 Oct 2023 to Oct 2024, DHS 2025),
and solar grew over that time, so some bias is expected, mostly in the Census comparison.

Inputs:  outputs/<country>_share_*.tif, data/processed/<country>_features_1km.csv, _grid_cells.csv,
         data/processed/open_labels/*.csv, data/raw/boundaries/RWA_ADM1/ADM3.geojson
Outputs: outputs/control_totals_{districts,sectors,provinces,summary}_<country>.csv
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.config import load_config, repo_path
from src.labels_open import norm_name
from src.predict import weighted_shares
from src.train import CLASSES


# ---------------------------------------------------------------- pure helpers
def group_shares(keys: pd.DataFrame, shares: np.ndarray, population: np.ndarray) -> pd.DataFrame:
    """Population-weighted grid / solar / unserved per group. `keys` holds the grouping columns (NaN keys are dropped).

    Cells without a prediction, or with no population, do not count. Groups left with nothing are dropped.
    """
    columns = list(keys.columns)
    rows = []
    for key, idx in keys.groupby(columns, sort=True).indices.items():
        ok = ~np.isnan(shares[idx]).any(axis=1) & ~np.isnan(population[idx])
        if population[idx][ok].sum() <= 0:
            continue
        w = weighted_shares(shares[idx], population[idx])
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(columns, key)), **{f"pred_{c}": w[k] for k, c in enumerate(CLASSES)},
                     "population": float(population[idx][ok].sum())})
    return pd.DataFrame(rows)


def compare(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray = None, ci: tuple = None) -> dict:
    """Agreement between predicted and published shares per area.

    mae: mean absolute error. bias: mean(pred - truth), positive = we predict more than published.
    mae_no_bias: mae after removing the bias, i.e. how well the pattern across areas matches.
    skill: 1 - mae / mae of a constant map (the weighted national mean of our own predictions), so
    0 means no better than a flat map and 1 means perfect. within_ci: share of areas whose prediction
    lies inside the published 95% interval (needs `ci` = (low, high)).
    """
    w = np.ones(len(pred)) if weights is None else weights
    mae = np.average(np.abs(pred - truth), weights=w)
    constant_mae = np.average(np.abs(np.average(pred, weights=w) - truth), weights=w)
    bias = np.average(pred - truth, weights=w)
    out = {"n": len(pred), "mae": mae, "bias": bias, "mae_no_bias": np.average(np.abs(pred - truth - bias), weights=w),
           "pearson": float(pd.Series(pred).corr(pd.Series(truth))),
           "spearman": float(pd.Series(pred).corr(pd.Series(truth), method="spearman")),
           "constant_mae": constant_mae, "skill": 1 - mae / constant_mae}
    if ci is not None:
        out["within_ci"] = float(((pred >= ci[0]) & (pred <= ci[1])).mean())
    return out


def add_grid_connected(sectors: pd.DataFrame, connected: pd.DataFrame) -> pd.DataFrame:
    """Join the census grid-connected share to the sector table and derive the solar proxy.

    `sectors` has district_key, sector_key and served (C1.3); `connected` has the same keys and connected (C1.5).
    solar_proxy = max(served - connected, 0). Sectors without a C1.5 row are dropped.
    """
    out = sectors.merge(connected[["district_key", "sector_key", "connected"]],
                        on=["district_key", "sector_key"], how="inner", validate="one_to_one")
    out["solar_proxy"] = (out["served"] - out["connected"]).clip(lower=0)
    return out


def match_report(left: pd.Series, right: pd.Series) -> tuple:
    """Names in `left` that have no partner in `right`, and the reverse (using normalised names)."""
    l, r = set(left), set(right)
    return sorted(l - r), sorted(r - l)


# ---------------------------------------------------------------- main
def read_cell_values(path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
        band[band == src.nodata] = np.nan
    return band[rows, cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()

    cfg = load_config()
    processed, out_dir = repo_path(cfg["paths"]["processed"]), repo_path(cfg["paths"]["outputs"])
    open_labels = processed / "open_labels"
    boundaries = repo_path(cfg["boundaries"]["out_dir"])

    table = pd.read_csv(processed / f"{args.country}_features_1km.csv", usecols=["cell_id", "district", "lon", "lat", "population_total"])
    grid = pd.read_csv(processed / f"{args.country}_grid_cells.csv")[["cell_id", "row", "col"]]
    cells = table.merge(grid, on="cell_id", how="left", validate="one_to_one")
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    shares = np.column_stack([read_cell_values(out_dir / f"{args.country}_share_{c}.tif", rows, cols) for c in CLASSES])
    pop = cells["population_total"].to_numpy()

    # province and sector of each cell centre
    points = gpd.GeoDataFrame(cells[["cell_id"]], geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs="EPSG:4326")
    for level, name in [("ADM1", "province"), ("ADM3", "sector")]:
        polygons = gpd.read_file(boundaries / f"RWA_{level}.geojson")[["shapeName", "geometry"]].rename(columns={"shapeName": name})
        joined = gpd.sjoin(points, polygons, how="left", predicate="within")
        assert len(joined) == len(points), f"{level} polygons overlap"
        cells[name] = joined[name].to_numpy()
    cells["district_key"], cells["sector_key"] = cells["district"].map(norm_name), cells["sector"].map(lambda s: norm_name(s) if isinstance(s, str) else np.nan)
    cells["province_key"] = cells["province"].map(lambda s: norm_name(s) if isinstance(s, str) else np.nan)

    summary = []

    # ---- districts vs EICV7
    eicv = pd.read_csv(open_labels / "eicv7_district_access.csv")
    eicv["district_key"] = eicv["district"].map(norm_name)
    pred = group_shares(cells[["district_key"]], shares, pop)
    d = pred.merge(eicv, on="district_key", how="inner", validate="one_to_one")
    d["pred_served"] = d["pred_grid"] + d["pred_solar"]
    d["diff"] = d["pred_served"] - d["served"]
    d["inside_ci"] = (d["pred_served"] >= d["ci_low"]) & (d["pred_served"] <= d["ci_high"])
    miss = match_report(eicv["district_key"], pred["district_key"])
    print(f"Districts matched: {len(d)} of {len(eicv)}  (unmatched in EICV7: {miss[0]}, in map: {miss[1]})")
    summary.append({"level": "district", "source": "EICV7 2023/24", "quantity": "served (grid+solar)",
                    **compare(d["pred_served"].to_numpy(), d["served"].to_numpy(), d["population"].to_numpy(),
                              (d["ci_low"].to_numpy(), d["ci_high"].to_numpy()))})
    d[["district", "pred_grid", "pred_solar", "pred_served", "served", "ci_low", "ci_high", "diff", "inside_ci", "population"]] \
        .rename(columns={"served": "eicv7_served"}).to_csv(out_dir / f"control_totals_districts_{args.country}.csv", index=False)

    # ---- sectors vs Census 2022
    census = pd.read_csv(open_labels / "census_sector_electricity.csv")
    census["district_key"], census["sector_key"] = census["district"].map(norm_name), census["sector"].map(norm_name)
    ps = group_shares(cells[["district_key", "sector_key"]], shares, pop)
    s = ps.merge(census, on=["district_key", "sector_key"], how="inner", validate="one_to_one")
    s["pred_served"] = s["pred_grid"] + s["pred_solar"]
    left_out = census.merge(s[["district_key", "sector_key"]], how="left", indicator=True).query("_merge == 'left_only'")
    print(f"Sectors matched: {len(s)} of {len(census)} census sectors "
          f"({100 * s['n_households'].sum() / census['n_households'].sum():.0f}% of census households); "
          f"unmatched: {(left_out['district'] + ' / ' + left_out['sector']).tolist()}")
    summary.append({"level": "sector", "source": "Census 2022", "quantity": "served (grid+solar)",
                    **compare(s["pred_served"].to_numpy(), s["served"].to_numpy(), s["n_households"].to_numpy())})
    g = pd.read_csv(open_labels / "census_sector_grid.csv")
    g["district_key"], g["sector_key"] = g["district"].map(norm_name), g["sector"].map(norm_name)
    sg = add_grid_connected(s, g)
    print(f"Sectors with a C1.5 grid-connected share: {len(sg)} of {len(s)}. Census national (household-weighted): "
          f"connected {np.average(sg['connected'], weights=sg['n_households']):.3f}, "
          f"solar proxy {np.average(sg['solar_proxy'], weights=sg['n_households']):.3f} (Table 8.12 solar: about 0.139)")
    summary.append({"level": "sector", "source": "Census 2022 (C1.5)", "quantity": "grid connected",
                    **compare(sg["pred_grid"].to_numpy(), sg["connected"].to_numpy(), sg["n_households"].to_numpy())})
    summary.append({"level": "sector", "source": "Census 2022 (C1.3 minus C1.5)", "quantity": "solar proxy (rough)",
                    **compare(sg["pred_solar"].to_numpy(), sg["solar_proxy"].to_numpy(), sg["n_households"].to_numpy())})
    s = s.merge(sg[["district_key", "sector_key", "connected", "solar_proxy"]], on=["district_key", "sector_key"], how="left")
    s[["district", "sector", "pred_grid", "pred_solar", "pred_served", "served", "connected", "solar_proxy", "n_households"]] \
        .rename(columns={"served": "census_served", "connected": "census_grid_connected", "solar_proxy": "census_solar_proxy"}) \
        .to_csv(out_dir / f"control_totals_sectors_{args.country}.csv", index=False)

    # ---- grid vs solar split per province
    ev = pd.read_csv(open_labels / "eicv7_province_lighting.csv")
    cp = pd.read_csv(open_labels / "census_province_lighting.csv")
    cp = cp[cp["area"] == "total"].assign(census_grid=lambda x: x["grid"] / x["total"],
                                          census_solar=lambda x: (x["solar"] + x["lantern"]) / x["total"])
    for frame in (ev, cp):
        frame["province_key"] = frame["province"].map(norm_name)
    pp = group_shares(cells[["province_key"]], shares, pop)
    p = pp.merge(ev[["province_key", "province", "share_grid", "share_solar"]], on="province_key", validate="one_to_one") \
          .merge(cp[["province_key", "census_grid", "census_solar"]], on="province_key", validate="one_to_one") \
          .rename(columns={"share_grid": "eicv7_grid", "share_solar": "eicv7_solar"})
    p.drop(columns=["province_key", "pred_unserved"]).to_csv(out_dir / f"control_totals_provinces_{args.country}.csv", index=False)
    print(f"Provinces matched: {len(p)} of 5")
    for src, label in [("eicv7", "EICV7 2023/24"), ("census", "Census 2022")]:
        for share in ["grid", "solar"]:
            summary.append({"level": "province", "source": label, "quantity": share,
                            **compare(p[f"pred_{share}"].to_numpy(), p[f"{src}_{share}"].to_numpy(), p["population"].to_numpy())})

    result = pd.DataFrame(summary)
    result.to_csv(out_dir / f"control_totals_summary_{args.country}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nAgreement with published numbers (shares, 0 to 1). bias > 0: we predict more than published.")
    print(result.drop(columns=["constant_mae"]).round(3).to_string(index=False))
    worst = d.reindex(d["diff"].abs().sort_values(ascending=False).index).head(5)
    print("\nLargest district differences (predicted served minus EICV7):")
    print(worst[["district", "pred_served", "served", "ci_low", "ci_high", "diff"]].round(3).to_string(index=False))
    print("\nGrid and solar by province (predicted vs EICV7 vs Census):")
    print(p[["province", "pred_grid", "eicv7_grid", "census_grid", "pred_solar", "eicv7_solar", "census_solar"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
