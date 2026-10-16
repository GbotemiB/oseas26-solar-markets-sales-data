"""Download the NISR (National Institute of Statistics of Rwanda) tables used to check the maps.

Run:  python -m src.download.nisr
Saves data/raw/nisr/<name>.xlsx plus a .json next to each with source and download date.

EICV7 (Integrated Household Living Conditions Survey, 2023/24): main indicator tables.
Census 2022 (RPHC5): Housing and Households Characteristics thematic tables.
"""
from src.config import load_config, repo_path
from src.download.common import stream_to_file

META = {
    "eicv7_tables.xlsx": {"source": "NISR EICV7 Main Indicator Report tables, 2023/24",
                          "licence": "NISR terms of use (CC BY 4.0 listed in data/SOURCES.md, to confirm)"},
    "census2022_housing.xlsx": {"source": "NISR Rwanda Population and Housing Census 2022 (RPHC5), "
                                          "Housing and Households Characteristics tables",
                                "licence": "NISR terms of use (CC BY 4.0 listed in data/SOURCES.md, to confirm)"},
}


def main() -> None:
    cfg = load_config()
    out_dir = repo_path(cfg["nisr"]["out_dir"])
    for name, url in cfg["nisr"]["files"].items():
        stream_to_file(url, out_dir / name, META[name])


if __name__ == "__main__":
    main()
