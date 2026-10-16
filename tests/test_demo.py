"""Tests for the synthetic demo (src/make_demo_data.py, src/demo.py). No DHS files needed.

Run from the repo root:  python -m pytest tests/test_demo.py -v
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import rasterio

import src.config as config
from src import demo
from src.config import ROOT
from src.make_demo_data import FEATURES, N_CLUSTERS, write_demo
from src.train import CLASSES, SHARE_COLS


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A copy of the repo layout in a temp folder, so the test never touches the real data/ or outputs/."""
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def test_generator_is_reproducible_and_changes_with_the_seed(tmp_path):
    write_demo(tmp_path / "a", 1)
    write_demo(tmp_path / "b", 1)
    write_demo(tmp_path / "c", 2)
    a, b, c = (pd.read_csv(tmp_path / d / "demo_train_table.csv") for d in "abc")
    assert a.equals(b)
    assert not a.equals(c)


def test_demo_data_has_the_columns_the_pipeline_needs_and_no_restricted_ones(tmp_path):
    write_demo(tmp_path, 1)
    train = pd.read_csv(tmp_path / "demo_train_table.csv")
    cells = pd.read_csv(tmp_path / "demo_features_1km.csv")
    assert len(train) == N_CLUSTERS and train["district"].nunique() >= 10
    assert train[SHARE_COLS].sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert set(FEATURES) <= set(train.columns)
    assert {f"{f}_r2km" for f in FEATURES} <= set(cells.columns)
    for name in ["demo_train_table.csv", "demo_features_1km.csv", "demo_grid_cells.csv"]:
        columns = set(pd.read_csv(tmp_path / name).columns)
        assert not columns & {"lat", "lon", "latitude", "longitude", "hv001", "hv002", "hv005"}
    assert (cells["population_total"] < 1).sum() > 0          # a lake, so the population mask has something to do


def test_whole_demo_runs_and_every_output_is_valid_and_labelled_synthetic(sandbox):
    demo.main(["--regenerate"])
    out = sandbox / "outputs" / "demo"
    shares = []
    for cls in CLASSES:
        with rasterio.open(out / f"demo_share_{cls}.tif") as src:
            shares.append(src.read(1))
            assert "SYNTHETIC" in src.tags()["limitations"]
    stack = np.stack(shares)
    valid = (stack != -9999).all(axis=0)
    assert valid.sum() > 1000 and (stack[:, valid] >= 0).all()
    assert stack[:, valid].sum(axis=0) == pytest.approx(1.0, abs=1e-4)
    assert (stack[:, ~valid] == -9999).all()                  # lake and no-data stay empty in all three

    for side in ["low", "high"]:
        assert (out / f"demo_share_unserved_{side}.tif").exists()
    with rasterio.open(out / "demo_do_not_trust.tif") as src:
        codes = src.read(1)
        assert set(np.unique(codes[codes != -9999])) <= {0, 1, 2, 3}

    with rasterio.open(out / "demo_market_reality_eae.tif") as src:
        layer = src.read(1)
        assert str(src.crs) == "EPSG:4326" and "SYNTHETIC" in src.tags()["confidence"]
        assert "direction" in src.tags()
    assert 0 <= layer[layer != -9999].min() and layer[layer != -9999].max() <= 1
    sidecar = json.loads((out / "demo_market_reality_eae.tif.json").read_text())
    assert "SYNTHETIC" in sidecar["limitations"]


def test_demo_beats_a_flat_map_against_the_hidden_truth(sandbox):
    demo.main(["--regenerate"])
    errors = demo.score_against_truth(sandbox / "outputs" / "demo", sandbox / "data" / "demo")
    cells = pd.read_csv(sandbox / "data" / "demo" / "demo_features_1km.csv")
    for cls in CLASSES:
        flat = np.average(np.abs(cells[f"true_share_{cls}"] - np.average(
            cells[f"true_share_{cls}"], weights=cells["population_total"])), weights=cells["population_total"])
        assert errors[cls] < 0.5 * flat


def test_demo_run_never_reads_restricted_or_raw_data(sandbox):
    """The sandbox has no data/restricted, data/processed or data/raw: a demo that needed them would crash."""
    demo.main(["--regenerate"])
    assert not (sandbox / "data" / "restricted").exists()
    assert not (sandbox / "data" / "processed").exists()
