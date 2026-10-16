"""Build grid / solar / unserved shares per DHS cluster.

These shares are the training labels for the model.

Run from the repo root:
    python -m src.labels_dhs                    # all countries in config.yaml
    python -m src.labels_dhs --country rwanda   # one country

Inputs (RESTRICTED, never committed):
    data/restricted/dhs/<country>/*.ZIP    DHS download zip (household recode + GPS)

Outputs:
    data/restricted/labels/<country>_cluster_labels.csv
        One row per cluster, WITH (displaced) GPS coordinates -> stays restricted.
    outputs/label_summary_<country>.csv
        National and urban/rural weighted shares only. Aggregated -> safe to publish.

DHS rules: this script only prints aggregate numbers. It never prints household
rows or cluster coordinates.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import check_restricted_is_ignored, load_config, repo_path

# hv262 "Type of light at home" (same codes in Rwanda, Kenya, Nigeria DHS-8)
LIGHT_ELECTRICITY = 1
LIGHT_SOLAR = 2          # "Solar lantern" / "Solar lantern/standalone solar system"
LIGHT_MISSING = {97, 98, 99}   # DHS missing / don't know codes, if present

CLASSES = ["grid", "solar", "unserved"]
HOUSEHOLD_COLS = ["hv001", "hv005", "hv009", "hv025", "hv206", "hv262"]   # hv009 = household members


# ---------------------------------------------------------------- file helpers
def find_zip(dhs_dir: Path) -> Path:
    zips = sorted(p for p in dhs_dir.iterdir() if p.suffix.lower() == ".zip")
    if not zips:
        raise FileNotFoundError(f"No DHS zip found in {dhs_dir}")
    if len(zips) > 1:
        print(f"  note: several zips in {dhs_dir}, using {zips[0].name}")
    return zips[0]


def find_member(zf: zipfile.ZipFile, file_name: str) -> str:
    """Find a file inside the zip, ignoring folders and upper/lower case."""
    matches = [n for n in zf.namelist() if Path(n).name.lower() == file_name.lower()]
    if not matches:
        raise FileNotFoundError(f"{file_name} not found inside {zf.filename}")
    return matches[0]


def extract_once(zip_path: Path, member: str, out_dir: Path) -> Path:
    """Unzip one file into data/restricted/... (skip if already there)."""
    out_path = out_dir / member
    if not out_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(member, out_dir)
    return out_path


# ---------------------------------------------------------------- loading
def load_households(zip_path: Path, file_name: str, out_dir: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        member = find_member(zf, file_name)
    dta = extract_once(zip_path, member, out_dir)
    # Read only the columns we need: full files are 40-175 MB and use a lot of memory.
    hh = pd.read_stata(dta, columns=HOUSEHOLD_COLS, convert_categoricals=False)
    hh["weight"] = hh["hv005"] / 1_000_000
    return hh


def load_gps(zip_path: Path, file_name: str) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        member = find_member(zf, file_name)
    gps = gpd.read_file(f"zip://{zip_path}!{member}")
    return gps[["DHSCLUST", "URBAN_RURA", "LATNUM", "LONGNUM"]].rename(
        columns={"DHSCLUST": "cluster", "URBAN_RURA": "urban_rural",
                 "LATNUM": "lat", "LONGNUM": "lon"}
    )


# ---------------------------------------------------------------- labelling
def classify(hh: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Give each household one class: grid, solar or unserved."""
    hh = hh[hh["hv262"].notna() & ~hh["hv262"].isin(LIGHT_MISSING)].copy()
    light = hh["hv262"]
    has_elec = hh["hv206"] == 1

    if rule == "lighting":
        hh["cls"] = np.select(
            [light == LIGHT_ELECTRICITY, light == LIGHT_SOLAR],
            ["grid", "solar"], default="unserved",
        )
    elif rule == "electricity_then_lighting":
        hh = hh[hh["hv206"].notna()]
        light, has_elec = hh["hv262"], hh["hv206"] == 1
        hh["cls"] = np.select(
            [light == LIGHT_SOLAR, has_elec | (light == LIGHT_ELECTRICITY)],
            ["solar", "grid"], default="unserved",
        )
    else:
        raise ValueError(f"Unknown label_rule: {rule}")

    # Extra flag (all countries): says it has electricity but mainly lights with
    # something else -> likely an unreliable grid connection.
    hh["grid_unreliable"] = (has_elec & ~light.isin([LIGHT_ELECTRICITY, LIGHT_SOLAR])).astype(int)
    return hh


def weighted_shares(df: pd.DataFrame) -> pd.Series:
    w = df["weight"]
    out = {f"share_{c}": w[df["cls"] == c].sum() / w.sum() for c in CLASSES}
    out["share_grid_unreliable"] = (w * df["grid_unreliable"]).sum() / w.sum()
    return pd.Series(out)


def cluster_labels(hh: pd.DataFrame) -> pd.DataFrame:
    shares = hh.groupby("hv001").apply(weighted_shares, include_groups=False)
    counts = hh.groupby("hv001").agg(
        n_households=("hv001", "size"),
        sum_weight=("weight", "sum"),
        hv025=("hv025", "first"),
    )
    return counts.join(shares).reset_index().rename(columns={"hv001": "cluster"})


def mean_household_size(hh: pd.DataFrame) -> float:
    """Weighted mean number of household members (hv009). One aggregate number, used to turn population into households."""
    ok = hh["hv009"].notna()
    return float((hh.loc[ok, "weight"] * hh.loc[ok, "hv009"]).sum() / hh.loc[ok, "weight"].sum())


def household_sizes(hh: pd.DataFrame) -> dict:
    """Weighted mean household size overall, and in urban (hv025 = 1) and rural (hv025 = 2) households."""
    out = {"mean_household_size": mean_household_size(hh)}
    for code, name in [(1, "urban"), (2, "rural")]:
        out[f"{name}_household_size"] = mean_household_size(hh[hh["hv025"] == code])
    return out


def summary_table(hh: pd.DataFrame, country: str) -> pd.DataFrame:
    rows = {"national": weighted_shares(hh)}
    for code, name in [(1, "urban"), (2, "rural")]:
        rows[name] = weighted_shares(hh[hh["hv025"] == code])
    out = pd.DataFrame(rows).T
    out.insert(0, "country", country)
    out["n_households"] = [len(hh), (hh.hv025 == 1).sum(), (hh.hv025 == 2).sum()]
    return out.rename_axis("area").reset_index()


# ---------------------------------------------------------------- main
def run_country(name: str, cfg: dict, buffers: dict) -> None:
    print(f"\n=== {name} ({cfg['survey']}), rule = {cfg['label_rule']}")
    dhs_dir = repo_path(cfg["dhs_dir"])
    zip_path = find_zip(dhs_dir)

    hh = load_households(zip_path, cfg["household_file"], dhs_dir / "extracted")
    n_before = len(hh)
    hh = classify(hh, cfg["label_rule"])
    print(f"  households: {len(hh):,} used ({n_before - len(hh)} dropped for missing lighting/electricity)")

    labels = cluster_labels(hh)
    assert np.allclose(labels[[f"share_{c}" for c in CLASSES]].sum(axis=1), 1.0)

    # ---- join GPS
    gps = load_gps(zip_path, cfg["gps_file"])
    zero = (gps["lat"] == 0) & (gps["lon"] == 0)
    gps = gps[~zero]
    labels = labels.merge(gps, on="cluster", how="left")
    no_gps = labels["lat"].isna()

    # urban/rural from GPS file should agree with hv025 in the household file
    gps_urban = labels["urban_rural"].map({"U": 1, "R": 2})
    mismatch = (~no_gps & (gps_urban != labels["hv025"])).sum()

    labels["urban"] = (labels["hv025"] == 1).astype(int)
    labels["buffer_m"] = np.where(labels["urban"] == 1, buffers["urban"], buffers["rural"])
    labels["has_gps"] = (~no_gps).astype(int)
    labels = labels.drop(columns=["hv025", "urban_rural"])

    print(f"  clusters: {len(labels)} with households | GPS file: {len(gps) + zero.sum()} points, "
          f"{zero.sum()} at (0,0) dropped | clusters without usable GPS: {no_gps.sum()} "
          f"| urban/rural mismatches: {mismatch}")
    print(f"  households per cluster: median {labels.n_households.median():.0f}, "
          f"min {labels.n_households.min()}, clusters with <10 households: {(labels.n_households < 10).sum()}")

    # ---- save
    label_dir = repo_path("data/restricted/labels")
    label_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(label_dir / f"{name}_cluster_labels.csv", index=False)

    sizes = {k: round(v, 3) for k, v in household_sizes(hh).items()}
    (label_dir / f"{name}_household_size.json").write_text(json.dumps({**sizes, "n_households": len(hh)}))
    print(f"  household size, members: all {sizes['mean_household_size']:.2f}, urban {sizes['urban_household_size']:.2f}, "
          f"rural {sizes['rural_household_size']:.2f} (aggregates, used to turn population into households)")

    summary = summary_table(hh, name)
    out_dir = repo_path("outputs")
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / f"label_summary_{name}.csv", index=False)

    show = summary.set_index("area")[[f"share_{c}" for c in CLASSES] + ["share_grid_unreliable"]]
    print("  weighted shares (%):")
    print((100 * show).round(1).to_string().replace("\n", "\n    ").join(["    ", ""]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", help="one country from config.yaml (default: all)")
    args = parser.parse_args()

    check_restricted_is_ignored()
    config = load_config()
    countries = config["countries"]
    names = [args.country] if args.country else list(countries)
    for name in names:
        run_country(name, countries[name], config["buffers_m"])


if __name__ == "__main__":
    main()
