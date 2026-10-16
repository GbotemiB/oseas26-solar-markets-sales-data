"""Download Gridfinder predicted grid lines (global, from 2019).

Run:  python -m src.download.gridfinder
Saves data/raw/gridfinder/grid.gpkg (725 MB, whole world). Later steps read only Rwanda from it.

Gridfinder predicts medium-voltage lines from night lights and other open data. It is a model
output from 2019, not a survey of the real network, and the Rwandan grid has grown since.
"""
from src.config import load_config, repo_path
from src.download.common import stream_to_file


def main() -> None:
    cfg = load_config()
    out_dir = repo_path(cfg["downloads"]["gridfinder_dir"])
    stream_to_file(
        cfg["downloads"]["gridfinder_url"], out_dir / "grid.gpkg",
        {"source": "Gridfinder, Zenodo record 3628142 (Arderne et al. 2020)", "licence": "CC BY 4.0",
         "note": "predicted grid lines, data from 2019"})


if __name__ == "__main__":
    main()
