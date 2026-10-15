"""Download admin boundaries from geoBoundaries (open, mostly CC BY 4.0).

Run:  python -m src.download.boundaries --country rwanda
Saves data/raw/boundaries/<iso3>_<level>.geojson (e.g. RWA_ADM2.geojson).
"""
import argparse

import requests

from src.config import load_config, repo_path

API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{level}/"


def download(iso3: str, level: str, out_dir) -> None:
    out_path = out_dir / f"{iso3}_{level}.geojson"
    if out_path.exists():
        print(f"  {out_path.name} already there, skipping")
        return
    reply = requests.get(API.format(iso3=iso3, level=level), timeout=60)
    if reply.status_code == 404:
        print(f"  {iso3} has no {level} in geoBoundaries, skipping")
        return
    reply.raise_for_status()
    meta = reply.json()
    data = requests.get(meta["gjDownloadURL"], timeout=300)
    data.raise_for_status()
    out_path.write_bytes(data.content)
    print(f"  {out_path.name}: {meta.get('boundaryLicense', 'licence ?')}, "
          f"source year {meta.get('boundaryYearRepresented', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()
    cfg = load_config()
    iso3 = cfg["countries"][args.country]["iso3"]
    out_dir = repo_path(cfg["boundaries"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in cfg["boundaries"]["levels"]:
        download(iso3, level, out_dir)


if __name__ == "__main__":
    main()
