"""Tests for src/predict.py. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_predict.py -v
"""
import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.predict import (NODATA, cells_to_raster, district_table, layer_sources, mask_unpopulated,
                         predict_cells, weighted_shares, write_share_raster)
from src.train import CLASSES, SHARE_COLS


def synthetic_clusters(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    grid = 1 / (1 + np.exp(-3 * x))
    solar = (1 - grid) * 0.7
    shares = np.column_stack([grid, solar, 1 - grid - solar])
    return pd.DataFrame({"n_households": rng.integers(20, 30, n), "signal": x,
                         **{c: shares[:, k] for k, c in enumerate(SHARE_COLS)}})


def synthetic_cells(signal) -> pd.DataFrame:
    return pd.DataFrame({"signal_r2km": signal})


def test_predictions_are_shares_that_sum_to_one_and_follow_the_signal():
    cells = synthetic_cells([-2.0, 0.0, 2.0])
    pred = predict_cells(synthetic_clusters(), cells, ["signal"], [], "2km", seed=1)
    assert pred.shape == (3, 3)
    assert pred.sum(axis=1) == pytest.approx(1.0)
    assert (pred >= 0).all() and (pred <= 1).all()
    assert pred[0, 0] < pred[1, 0] < pred[2, 0]         # grid share rises with the signal
    assert pred[0, 2] > pred[2, 2]                      # unserved share falls with it


def test_cells_with_a_missing_feature_are_left_empty_not_guessed():
    cells = synthetic_cells([0.5, np.nan, -0.5])
    pred = predict_cells(synthetic_clusters(), cells, ["signal"], [], "2km", seed=1)
    assert np.isnan(pred[1]).all()
    assert not np.isnan(pred[[0, 2]]).any()


def test_wrong_radius_gives_a_clear_error():
    with pytest.raises(KeyError, match="buffer_m"):
        predict_cells(synthetic_clusters(), synthetic_cells([0.0]), ["signal"], [], "3km", seed=1)


def test_prediction_is_reproducible():
    cells = synthetic_cells(np.linspace(-2, 2, 7))
    a = predict_cells(synthetic_clusters(), cells, ["signal"], [], "2km", seed=1)
    b = predict_cells(synthetic_clusters(), cells, ["signal"], [], "2km", seed=1)
    assert np.allclose(a, b)


def test_mask_unpopulated_blanks_cells_with_missing_or_tiny_population():
    shares = np.full((4, 3), 1 / 3)
    out = mask_unpopulated(shares, np.array([50.0, 0.4, np.nan, 1.0]), 1.0)
    assert not np.isnan(out[[0, 3]]).any()              # populated, and exactly at the threshold, are kept
    assert np.isnan(out[[1, 2]]).all()                  # below threshold and missing are blanked
    assert not np.isnan(shares).any()                   # input is not changed


def test_cells_to_raster_places_values_and_marks_gaps():
    values = np.array([0.2, np.nan, 0.8])
    r = cells_to_raster(values, np.array([0, 1, 2]), np.array([0, 1, 2]), (3, 3))
    assert r.dtype == np.float32
    assert r[0, 0] == pytest.approx(0.2) and r[2, 2] == pytest.approx(0.8)
    assert r[1, 1] == NODATA                            # NaN cell
    assert r[0, 2] == NODATA                            # outside the country (no cell there)


def test_weighted_shares_uses_weights_and_skips_empty_cells():
    shares = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [np.nan, np.nan, np.nan]])
    out = weighted_shares(shares, np.array([30.0, 10.0, 1000.0]))
    assert out == pytest.approx([0.75, 0.25, 0.0])      # the empty cell's big weight is ignored
    nan_weight = weighted_shares(shares[:2], np.array([30.0, np.nan]))
    assert nan_weight == pytest.approx([1.0, 0.0, 0.0])


def test_district_table_weights_by_population():
    districts = pd.Series(["A", "A", "B", "B"])
    shares = np.array([[1.0, 0, 0], [0.0, 1.0, 0], [0, 0, 1.0], [np.nan] * 3])
    pop = np.array([3.0, 1.0, 5.0, 7.0])
    t = district_table(districts, shares, pop).set_index("district")
    assert t.loc["A", "share_grid"] == pytest.approx(0.75)
    assert t.loc["B", "share_unserved"] == pytest.approx(1.0)
    assert t.loc["B", "n_cells"] == 2 and t.loc["B", "n_cells_predicted"] == 1


def test_written_raster_has_metadata_nodata_and_matches_the_grid(tmp_path):
    transform = from_origin(1000, 5000, 1000, 1000)
    array = cells_to_raster(np.array([0.5, np.nan]), np.array([0, 1]), np.array([0, 1]), (4, 4))
    path = tmp_path / "share_grid.tif"
    write_share_raster(path, array, transform, "EPSG:32735", {"unit": "share", "limitations": "test"})
    with rasterio.open(path) as src:
        assert src.shape == (4, 4) and src.nodata == NODATA and str(src.crs) == "EPSG:32735"
        assert src.transform == transform
        assert src.tags()["limitations"] == "test"
        data = src.read(1)
    assert data[0, 0] == pytest.approx(0.5) and data[1, 1] == NODATA
    assert json.loads((tmp_path / "share_grid.tif.json").read_text())["unit"] == "share"


def test_layer_sources_reads_the_download_sidecars(tmp_path):
    (tmp_path / "rwanda_population.json").write_text(json.dumps({"source": "WorldPop", "licence": "CC BY 4.0"}))
    assert layer_sources(["population"], tmp_path, "rwanda") == ["population: WorldPop [CC BY 4.0]"]
