"""Plan B / validation labels: lighting-source tables from the open NISR (National Institute of
Statistics of Rwanda) publications. They are used to CHECK the maps, not to train them.

Run (after src.download.nisr):
    python -m src.labels_open

Writes tidy CSVs to data/processed/open_labels/ (public aggregates, no household data):
    eicv7_district_access.csv      share of households with electricity as main light, per district, with 95% interval
    eicv7_province_lighting.csv    grid / solar / other share per province (EICV7, 2023/24)
    census_sector_electricity.csv  share of households with "electricity" as main light, per sector (Census 2022)
    census_province_lighting.csv   household counts by grid / solar / lantern per province and area (Census 2022)

Definitions (checked against the tables):
  - EICV7 Table A5.14 "access to electricity" is 72.0% nationally = grid 49.96% + solar 22.1% (+ about 1% both).
    So it is grid plus solar, the same as 1 - unserved in our maps.
  - Census Table C1.3 "Electricity" is 61.0% nationally = grid 47.1% + solar 13.9% from Table 8.12.
    Lanterns (1.2%) are a separate column. Our solar label includes lanterns, so the census is slightly low.
  - Solar can only be split from grid at province level (EICV7 5.11, Census 8.12), not for districts or sectors.
"""
import re
import unicodedata

import pandas as pd

from src.config import load_config, repo_path
from src.xlsx import read_sheet

GRID_ROW = "Electricity from REG or other private source"


def is_number(x) -> bool:
    return isinstance(x, (int, float))


def split_row(row: list):
    """(label, numbers): the first text cell, and the numeric cells in order. The tables have an empty first column."""
    texts = [x for x in row if isinstance(x, str)]
    return (texts[0].strip() if texts else ""), [x for x in row if is_number(x)]


def norm_name(name: str) -> str:
    """Key for matching names across sources: no accents, case, spaces or the word 'Province'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text.replace("province", ""))


def find_row(rows: list, text: str, start: int = 0) -> int:
    """Index of the first row at or after `start` with a cell that equals `text`."""
    for i in range(start, len(rows)):
        if any(isinstance(x, str) and x.strip() == text for x in rows[i]):
            return i
    raise KeyError(f"No row with '{text}'")


def parse_eicv7_districts(rows: list) -> pd.DataFrame:
    """Table A5.14: % of households with electricity as main light, per district, with standard error and 95% CI.

    Percent values are converted to shares (0 to 1).
    """
    out = []
    for row in rows[find_row(rows, "District") + 1:]:
        name, n = split_row(row)
        if not n:
            break                                       # the 'Source: NISR' line ends the table
        out.append({"district": name, "served": n[0] / 100, "se": n[1] / 100,
                    "ci_low": n[2] / 100, "ci_high": n[3] / 100})
    return pd.DataFrame(out)


def parse_eicv7_provinces(rows: list) -> pd.DataFrame:
    """Table 5.11: grid, solar and other share of households by main lighting source, national and per province."""
    header = [x.strip() for x in rows[find_row(rows, "Electricity distributor")] if isinstance(x, str)]
    i_grid, i_solar = header.index("Electricity distributor"), header.index("Solar power")
    places = ["Rwanda", "City of Kigali", "Southern", "Western", "Northern", "Eastern"]
    out = []
    for row in rows[find_row(rows, "Rwanda"):]:
        name, n = split_row(row)
        if name in places and n:
            out.append({"province": name, "share_grid": n[i_grid] / 100, "share_solar": n[i_solar] / 100,
                        "share_other": 1 - (n[i_grid] + n[i_solar]) / 100})
    return pd.DataFrame(out)


def parse_census_sectors(rows: list) -> pd.DataFrame:
    """Table C1.3: share of households with 'Electricity' as main lighting source, per district and sector.

    Columns after the name: 12 shares (first is Electricity) and the total, then the household count.
    """
    start = find_row(rows, "Electricity")
    assert [x for x in rows[start - 1] if isinstance(x, str)][-1] == "Total count", "household count column moved"
    out, district = [], None
    for row in rows[start + 1:]:
        name, n = split_row(row)
        if not n:
            if name and not name.startswith("Source"):
                district = name                         # a district line has a name and no numbers
            continue
        if name in ("Rwanda", "Total"):
            continue                                    # national line and the district total lines
        out.append({"district": district, "sector": name, "served": n[0], "n_households": n[-1]})
    return pd.DataFrame(out)


def parse_census_provinces(rows: list) -> pd.DataFrame:
    """Table 8.12: household counts by grid, solar, lantern and total, per province and area (urban, rural, total)."""
    wanted = {GRID_ROW: "grid", "Solar power": "solar", "Lantern": "lantern", "Total": "total"}
    assert [x for x in rows[find_row(rows, "Urban")] if isinstance(x, str)][1:4] == ["Urban", "Rural", "Total"]
    unit, counts = None, {}
    for row in rows:
        name, n = split_row(row)
        if not n:
            unit = name if name else unit
        elif name in wanted:
            counts[(unit, wanted[name])] = n[:3]        # urban, rural, total counts
    out = []
    for unit_name in sorted({u for u, _ in counts}):
        for k, area in enumerate(["urban", "rural", "total"]):
            out.append({"province": unit_name.replace(" Province", ""), "area": area,
                        **{kind: counts[(unit_name, kind)][k] for kind in ["grid", "solar", "lantern", "total"]}})
    return pd.DataFrame(out)


def main() -> None:
    cfg = load_config()
    nisr = repo_path(cfg["nisr"]["out_dir"])
    out_dir = repo_path(cfg["paths"]["processed"]) / "open_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    eicv7, census = nisr / "eicv7_tables.xlsx", nisr / "census2022_housing.xlsx"
    tables = {
        "eicv7_district_access": parse_eicv7_districts(read_sheet(eicv7, "Table A5.14")),
        "eicv7_province_lighting": parse_eicv7_provinces(read_sheet(eicv7, "Table 5.11")),
        "census_sector_electricity": parse_census_sectors(read_sheet(census, "Table-C1.3")),
        "census_province_lighting": parse_census_provinces(read_sheet(census, "Table-8.12")),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows")

    d, p = tables["eicv7_district_access"], tables["eicv7_province_lighting"]
    s, c = tables["census_sector_electricity"], tables["census_province_lighting"]
    print(f"\nEICV7: {len(d)} districts, national grid {p.set_index('province').loc['Rwanda', 'share_grid']:.3f}, "
          f"solar {p.set_index('province').loc['Rwanda', 'share_solar']:.3f}")
    print(f"Census: {len(s)} sectors in {s['district'].nunique()} districts, "
          f"{int(s['n_households'].sum()):,} households")
    nat = c[(c["province"] == "Rwanda") & (c["area"] == "total")].iloc[0]
    print(f"Census national: grid {nat['grid'] / nat['total']:.3f}, solar {nat['solar'] / nat['total']:.3f}, "
          f"lantern {nat['lantern'] / nat['total']:.3f}")


if __name__ == "__main__":
    main()
