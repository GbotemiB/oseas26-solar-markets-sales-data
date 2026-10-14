"""Tests for src/export_eae.py. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_export_eae.py -v
"""
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from src.export_eae import household_counts, market_reality, solar_test_sentence, to_geographic
from src.predict import NODATA


def shares(grid, solar, unserved):
    return [np.array(v, dtype="float32") for v in (grid, solar, unserved)]


def test_ratio_is_unserved_among_non_grid_and_stays_between_0_and_1():
    g, s, u = shares([0.5, 0.0, 0.2], [0.25, 0.5, 0.0], [0.25, 0.5, 0.8])
    out = market_reality(g, s, u, 0.05)
    assert out == pytest.approx([0.5, 0.5, 1.0])
    assert out.dtype == np.float32


def test_direction_one_means_no_solar_and_zero_means_all_solar():
    g, s, u = shares([0.1, 0.1], [0.0, 0.9], [0.9, 0.0])
    out = market_reality(g, s, u, 0.05)
    assert out[0] == 1.0 and out[1] == 0.0


def test_almost_fully_grid_cells_are_no_data():
    # binary-exact values, so the threshold comparison is not blurred by float32 rounding
    g, s, u = shares([0.99, 0.875, 0.9], [0.005, 0.0625, 0.05], [0.005, 0.0625, 0.05])
    out = market_reality(g, s, u, 0.125)
    assert out[0] == NODATA                  # non-grid share 0.01
    assert out[1] == pytest.approx(0.5)      # exactly at the threshold is kept
    assert out[2] == NODATA                  # 0.10, just below


def test_missing_inputs_give_no_data():
    g, s, u = shares([NODATA, 0.5], [0.2, NODATA], [0.3, 0.5])
    assert (market_reality(g, s, u, 0.05) == NODATA).all()


def test_reprojection_keeps_values_range_and_no_data():
    transform = from_origin(500000, 9800000, 1000, 1000)   # near Rwanda, UTM 35S
    layer = np.full((6, 6), 0.4, dtype="float32")
    layer[0, 0] = NODATA
    out, out_transform = to_geographic(layer, transform, "EPSG:32735", "EPSG:4326")
    valid = out[out != NODATA]
    assert valid.size > 0 and (NODATA in out)
    assert valid.min() >= 0.0 and valid.max() <= 1.0
    assert valid == pytest.approx(0.4, abs=1e-5)
    assert abs(out_transform.a) < 0.02       # pixels are degrees, about 1 km


def test_solar_test_sentence_quotes_the_numbers():
    test = pd.DataFrame({"model": ["ratio model (7 core features)"], "n_clusters": [472], "mae": [0.185],
                         "skill": [0.084], "auc_solar_majority": [0.644]})
    text = solar_test_sentence(test)
    assert "472" in text and "0.185" in text and "8%" in text and "0.64" in text


def test_household_counts_scale_to_the_national_total_and_keep_missing_cells_empty():
    population = np.array([100.0, 300.0, 600.0, np.nan])
    unserved = np.array([0.5, 0.1, np.nan, 0.9])
    households, unserved_households = household_counts(population, unserved, 500.0)
    assert np.nansum(households) == pytest.approx(500.0)                    # sums to the national count
    assert households[:3] == pytest.approx([50.0, 150.0, 300.0])
    assert unserved_households[:2] == pytest.approx([25.0, 15.0])
    assert np.isnan(unserved_households[2]) and np.isnan(unserved_households[3])   # no share, no population
