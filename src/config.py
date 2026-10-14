"""Load config.yaml and resolve paths relative to the repo root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(relative: str) -> Path:
    """Turn a path from config.yaml into an absolute path."""
    return ROOT / relative


def check_restricted_is_ignored() -> None:
    """Stop if data/restricted/ is not in .gitignore (DHS terms of use)."""
    gitignore = ROOT / ".gitignore"
    text = gitignore.read_text() if gitignore.exists() else ""
    if "data/restricted/" not in text:
        raise SystemExit(
            "STOP: data/restricted/ is not in .gitignore. "
            "DHS data must never be committed. Fix .gitignore first."
        )


def run_paths(cfg: dict, country: str, demo: bool = False) -> dict:
    """Input and output locations of one run. A demo run reads only data/demo/ and writes to outputs/demo/."""
    outputs = repo_path(cfg["paths"]["outputs"])
    if demo:
        d = repo_path(cfg["paths"]["demo"])
        return {"train": d / "demo_train_table.csv", "features": d / "demo_features_1km.csv",
                "grid_cells": d / "demo_grid_cells.csv", "grid_spec": d / "demo_grid_spec.json",
                "out_dir": outputs / "demo"}
    processed = repo_path(cfg["paths"]["processed"])
    return {"train": repo_path(cfg["paths"]["restricted"]) / "features" / f"{country}_train_table.csv",
            "features": processed / f"{country}_features_1km.csv",
            "grid_cells": processed / f"{country}_grid_cells.csv",
            "grid_spec": processed / f"{country}_grid_spec.json", "out_dir": outputs}
