"""Tests for src/features.py. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_features.py -v
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from src.features import buffer_means, build_1km_table, build_train_table, coarsen, focal_mean

RES = 100
# 100 m raster, 50 x 50 pixels. Top-left corner at x=0, y=5000 (metres).
TRANSFORM = from_origin(0, 5000, RES, RES)


def test_coarsen_mean_and_sum():
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert coarsen(a, 2, "mean").tolist() == [[2.5, 4.5], [10.5, 12.5]]
    assert coarsen(a, 2, "sum").tolist() == [[10, 18], [42, 50]]


def test_coarsen_ignores_nan_and_keeps_all_nan_blocks_nan():
    a = np.ones((4, 4))
    a[0, 0] = np.nan                    # one missing pixel: mean still 1, sum 3
    a[2:, 2:] = np.nan                  # whole block missing
    mean, total = coarsen(a, 2, "mean"), coarsen(a, 2, "sum")
    assert mean[0, 0] == 1.0 and total[0, 0] == 3.0
    assert np.isnan(mean[1, 1]) and np.isnan(total[1, 1])


def test_coarsen_rejects_bad_shape():
    with pytest.raises(ValueError):
        coarsen(np.ones((5, 5)), 2)


def test_focal_mean_constant_stays_constant_even_at_edges():
    out = focal_mean(np.full((9, 9), 3.0), 2)
    assert np.allclose(out, 3.0)


def test_focal_mean_single_spike():
    a = np.zeros((9, 9))
    a[4, 4] = 13.0                      # radius 2 circle has 13 cells
    out = focal_mean(a, 2)
    assert out[4, 4] == pytest.approx(1.0)      # 13 / 13
    assert out[4, 6] == pytest.approx(1.0)      # centre of spike is exactly 2 cells away, still inside
    assert out[4, 7] == 0.0                     # 3 cells away, outside


def test_focal_mean_skips_nan_and_keeps_empty_as_nan():
    a = np.full((5, 5), np.nan)
    a[2, 2] = 4.0
    out = focal_mean(a, 1)
    assert out[2, 2] == 4.0 and out[2, 3] == 4.0
    assert np.isnan(out[0, 0])


def test_buffer_mean_of_linear_ramp_equals_centre_value():
    # value = x coordinate of pixel centre. A circle that is symmetric around its centre
    # has mean equal to the centre value (a few pixels of rounding error allowed).
    cols = np.arange(50)
    ramp = np.tile((cols + 0.5) * RES, (50, 1)).astype(float)
    out = buffer_means({"ramp": ramp}, TRANSFORM, [2500.0], [2500.0], [1000])
    assert out["ramp"][0] == pytest.approx(2500.0, abs=25)


def test_buffer_mean_uses_only_pixels_inside_the_radius():
    a = np.zeros((50, 50))
    a[20:30, 20:30] = 10.0              # 1 km square in the middle, x 2000-3000, y 2000-3000
    inside = buffer_means({"a": a}, TRANSFORM, [2500.0], [2500.0], [400])
    far = buffer_means({"a": a}, TRANSFORM, [500.0], [500.0], [400])
    assert inside["a"][0] == pytest.approx(10.0)
    assert far["a"][0] == 0.0


def test_buffer_mean_ignores_nan_and_handles_points_outside_raster():
    a = np.full((50, 50), 2.0)
    a[:, :25] = np.nan                  # left half missing
    out = buffer_means({"a": a}, TRANSFORM, [2500.0, 99999.0, 100.0], [2500.0, 99999.0, 2500.0], [1000, 1000, 50])
    assert out["a"][0] == pytest.approx(2.0)    # only the valid half is averaged
    assert np.isnan(out["a"][1])                # far outside the raster
    assert np.isnan(out["a"][2])                # circle only covers missing pixels


def test_build_1km_table_columns_and_values():
    factor = 10
    pop = np.ones((20, 20))             # 2 x 2 cells of 1 km, each 10 x 10 pixels
    elev = np.full((20, 20), 1500.0)
    cells = pd.DataFrame({"cell_id": [0, 1], "district": ["A", "B"], "row": [0, 1], "col": [0, 1],
                          "x": [500.0, 1500.0], "y": [1500.0, 500.0], "lon": [30.0, 30.1], "lat": [-2.0, -2.1]})
    table = build_1km_table({"population": pop, "elevation": elev}, cells, factor, 1000, {"2km": 2000})
    assert table.loc[0, "population_total"] == 100          # sum of 100 pixels
    assert table.loc[0, "population"] == 1.0                # mean per pixel
    assert table.loc[1, "elevation_r2km"] == 1500.0
    assert {"population_r2km", "elevation_r2km", "district"} <= set(table.columns)


def test_build_train_table_joins_features_shares_and_district():
    crs = "EPSG:32735"
    # cluster placed at UTM (30.0E, ~0S) is awkward; build the point in UTM and convert back to lon/lat
    pt = gpd.GeoSeries(gpd.points_from_xy([2500.0], [2500.0]), crs=crs).to_crs("EPSG:4326")
    labels = pd.DataFrame({
        "cluster": [1, 2], "n_households": [25, 24], "sum_weight": [25.0, 24.0], "urban": [1, 0],
        "buffer_m": [500, 500], "share_grid": [0.5, 0.0], "share_solar": [0.3, 0.2],
        "share_unserved": [0.2, 0.8], "share_grid_unreliable": [0.0, 0.0],
        "lon": [pt.x[0], pt.x[0]], "lat": [pt.y[0], pt.y[0]], "has_gps": [1, 0]})
    districts = gpd.GeoDataFrame({"shapeName": ["Test District"]}, geometry=[box(0, 0, 5000, 5000)], crs=crs)
    layers = {"a": np.full((50, 50), 7.0)}
    train = build_train_table(layers, TRANSFORM, labels, districts, crs, buffer_m=800)
    assert len(train) == 1                                  # cluster without GPS is dropped
    assert train.loc[0, "buffer_m"] == 800                  # one radius for all, not the labels file value
    row = train.iloc[0]
    assert row["cluster"] == 1 and row["a"] == pytest.approx(7.0)
    assert row["district"] == "Test District"
    assert row[["share_grid", "share_solar", "share_unserved"]].sum() == pytest.approx(1.0)
    assert "lat" not in train.columns and "lon" not in train.columns   # no coordinates in the table


def test_build_train_table_uses_the_given_radius_not_the_one_in_the_labels_file():
    crs = "EPSG:32735"
    pt = gpd.GeoSeries(gpd.points_from_xy([2500.0], [2500.0]), crs=crs).to_crs("EPSG:4326")
    labels = pd.DataFrame({
        "cluster": [1], "n_households": [25], "sum_weight": [25.0], "urban": [0], "buffer_m": [5000],
        "share_grid": [0.5], "share_solar": [0.3], "share_unserved": [0.2], "share_grid_unreliable": [0.0],
        "lon": [pt.x[0]], "lat": [pt.y[0]], "has_gps": [1]})
    districts = gpd.GeoDataFrame({"shapeName": ["D"]}, geometry=[box(0, 0, 5000, 5000)], crs=crs)
    layer = np.zeros((50, 50))
    layer[20:30, 20:30] = 10.0                              # 1 km square around the cluster
    small = build_train_table({"a": layer}, TRANSFORM, labels, districts, crs, buffer_m=400)
    large = build_train_table({"a": layer}, TRANSFORM, labels, districts, crs, buffer_m=2000)
    assert small.loc[0, "a"] == pytest.approx(10.0)         # circle entirely inside the square
    assert 0 < large.loc[0, "a"] < 10.0                     # wider circle takes in zeros as well
