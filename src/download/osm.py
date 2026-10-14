"""Download an OpenStreetMap (OSM) country extract from Geofabrik.

Run:  python -m src.download.osm --country rwanda
Saves data/raw/osm/<country>-latest.osm.pbf (about 70 MB for Rwanda).

We need the .pbf (not the free shapefile bundle) because the shapefiles have no power lines.
Licence: ODbL. Anyone using the outputs must credit OpenStreetMap contributors.
"""
import argparse

from src.config import load_config, repo_path
from src.download.common import stream_to_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()
    cfg = load_config()
    region = cfg["countries"][args.country]["geofabrik"]
    stream_to_file(
        f"https://download.geofabrik.de/{region}-latest.osm.pbf",
        repo_path(cfg["downloads"]["osm_dir"]) / f"{args.country}-latest.osm.pbf",
        {"source": f"OpenStreetMap via Geofabrik ({region})", "licence": "ODbL, credit OpenStreetMap contributors"})


if __name__ == "__main__":
    main()
