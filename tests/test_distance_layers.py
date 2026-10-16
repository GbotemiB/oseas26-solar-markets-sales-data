"""Tests for src/distance_layers.py. Synthetic geometries only.

Run from the repo root:  python -m pytest tests/test_distance_layers.py -v
"""
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from src.distance_layers import distance_to_features

RES = 100
TRANSFORM = from_origin(0, 2000, RES, RES)     # 20 x 20 pixels, x 0-2000, y 0-2000
SHAPE = (20, 20)


def test_point_distance_is_zero_at_the_point_and_grows_with_distance():
    d = distance_to_features([Point(1050, 1050)], TRANSFORM, SHAPE, pad_px=5)   # centre of pixel row 9, col 10
    assert d[9, 10] == 0.0
    assert d[9, 13] == pytest.approx(0.3)           # 3 pixels = 300 m = 0.3 km
    assert d[9, 10 - 8] == pytest.approx(0.8)
    assert d[0, 10] == pytest.approx(0.9)           # 9 pixels up


def test_line_distance_is_perpendicular_distance():
    d = distance_to_features([LineString([(0, 1050), (2000, 1050)])], TRANSFORM, SHAPE, pad_px=5)
    assert d[9, :].max() == 0.0                     # the row the line runs through
    assert d[0, 5] == pytest.approx(0.9)
    assert d[19, 5] == pytest.approx(1.0)           # the line is in row 9, so row 19 is 10 pixels away


def test_feature_just_outside_the_raster_is_counted_thanks_to_the_margin():
    d = distance_to_features([Point(-250, 1050)], TRANSFORM, SHAPE, pad_px=10)   # 2.5 pixels left of the raster
    assert d[9, 0] == pytest.approx(0.3, abs=0.01)  # pixel 0 is 3 pixels from the pixel holding the point
    without = distance_to_features([Point(250, 1050)], TRANSFORM, SHAPE, pad_px=10)
    assert d[9, 0] < without[9, 10]                 # sanity: nearer than a point 8 pixels away


def test_no_feature_in_range_raises():
    with pytest.raises(ValueError):
        distance_to_features([Point(50000, 50000)], TRANSFORM, SHAPE, pad_px=5)


def test_output_shape_and_type():
    d = distance_to_features([Point(1050, 1050)], TRANSFORM, SHAPE, pad_px=5)
    assert d.shape == SHAPE and d.dtype == np.float32 and (d >= 0).all()
