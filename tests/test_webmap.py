"""Tests for src/webmap.py. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_webmap.py -v
"""
import base64
import io
import json
import re
import shutil

import numpy as np
import pytest
from PIL import Image

import src.config as config
from src import demo, webmap
from src.config import ROOT
from src.predict import NODATA


def test_to_percent_rounds_and_marks_missing_cells():
    a = np.array([[0.0, 0.456, 1.0], [NODATA, np.nan, 0.994]], dtype="float32")
    p = webmap.to_percent(a)
    assert p.tolist() == [[0, 46, 100], [-1, -1, 99]]
    assert p.dtype == np.int16


def test_to_count_rounds_clips_and_marks_missing_cells():
    a = np.array([[0.4, 12.6, 99999.0], [NODATA, np.nan, 0.0]], dtype="float32")
    assert webmap.to_count(a).tolist() == [[0, 13, 32000], [-1, -1, 0]]


def test_colorize_makes_no_data_transparent_and_clips_to_the_scale():
    a = np.array([[0.1, 0.9], [NODATA, 0.5]], dtype="float32")
    rgba = webmap.colorize(a, "magma_r", 0.3, 0.9)
    assert rgba[1, 0, 3] == 0 and rgba[0, 0, 3] == 255 and rgba[0, 1, 3] == 255
    assert tuple(rgba[0, 0, :3]) == tuple(webmap.colorize(np.array([[0.3]], dtype="float32"), "magma_r", 0.3, 0.9)[0, 0, :3])


def test_flag_image_only_colours_flagged_cells():
    rgba = webmap.flag_image(np.array([[0, 1, 2, 3, -1]]))
    assert rgba[0, 0, 3] == 0 and rgba[0, 4, 3] == 0           # fine and empty cells stay transparent
    assert all(rgba[0, k, 3] > 0 for k in (1, 2, 3))
    assert len({tuple(rgba[0, k]) for k in (1, 2, 3)}) == 3     # three distinct colours


def test_png_data_uri_round_trips():
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[1, 2] = (10, 20, 30, 255)
    uri = webmap.png_data_uri(rgba)
    image = np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(",")[1]))))
    assert image.shape == (3, 4, 4) and tuple(image[1, 2]) == (10, 20, 30, 255)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def test_demo_map_is_self_contained_labelled_and_matches_the_rasters(sandbox):
    demo.main(["--regenerate"])
    out = sandbox / "docs" / "demo_map.html"
    webmap.main(["--demo", "--out", "docs/demo_map.html"])
    html = out.read_text()
    assert "SYNTHETIC DEMO DATA" in html and "LOW CONFIDENCE" in html
    assert "fetch(" not in html and "XMLHttpRequest" not in html            # opens from file:// without a server
    assert out.stat().st_size < 1_000_000
    payload = json.loads(re.search(r"const P = (\{.*?\}), IMAGES", html, re.S).group(1))
    n = payload["height"] * payload["width"]
    for key in ["grid", "solar", "unserved", "market", "flag", "district", "grid_lo", "grid_hi", "hh", "hh_unserved"]:
        assert len(payload["values"][key]) == n
    grid, solar, unserved = (np.array(payload["values"][k]) for k in ("grid", "solar", "unserved"))
    ok = grid >= 0
    assert ok.sum() > 1000
    assert np.abs(grid[ok] + solar[ok] + unserved[ok] - 100).max() <= 2     # rounding only
    south, west = payload["bounds"][0]
    north, east = payload["bounds"][1]
    assert south < north and west < east and 29 < west < 31                 # UTM 35S demo area sits near 30 E
    assert set(np.unique(payload["values"]["flag"])) <= {-1, 0, 1, 2, 3}
    hh, hh_unserved = np.array(payload["values"]["hh"]), np.array(payload["values"]["hh_unserved"])
    assert (hh_unserved[ok] <= hh[ok] + 1).all()                            # unserved households never exceed households
    assert "hh_unserved" in payload["layers"] and "market" in payload["layers"]
