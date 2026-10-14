"""Tests for src/report.py and the one-page limitations statement. Synthetic data only.

Run from the repo root:  python -m pytest tests/test_report.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.config import ROOT
from src.report import (NODATA, cv_findings, cv_table, cv_text, mask_summary, md_table, province_table)

MODELS = ["constant (national mean)", "logistic regression", "lightgbm"]


def cv_rows(logistic_spatial, lightgbm_spatial, logistic_random=0.15):
    rows = []
    for model, random, spatial in [(MODELS[0], 0.21, 0.21), (MODELS[1], logistic_random, logistic_spatial),
                                   (MODELS[2], 0.16, lightgbm_spatial)]:
        for cv, v in [("random", random), ("spatial", spatial)]:
            rows.append({"feature_set": "core", "cv": cv, "model": model, "mean": v, "fold_min": v - 0.02, "fold_max": v + 0.02})
    rows.append({"feature_set": "plus_osm", "cv": "spatial", "model": MODELS[1], "mean": 9.9, "fold_min": 9.9, "fold_max": 9.9})
    return pd.DataFrame(rows)


def test_cv_table_sorts_by_spatial_error_and_ignores_other_feature_sets():
    t = cv_table(cv_rows(0.16, 0.17))
    assert list(t["model"]) == ["logistic regression", "lightgbm", "constant (national mean)"]
    assert t.loc[0, "spatial_cv"] == pytest.approx(0.16) and t.loc[0, "random_cv"] == pytest.approx(0.15)
    assert 9.9 not in t["spatial_cv"].to_numpy()


def test_text_says_checkpoint_not_met_when_lightgbm_is_worse():
    text = cv_text(cv_findings(cv_table(cv_rows(0.16, 0.17))))
    assert "Checkpoint not met" in text and "Checkpoint met" not in text


def test_text_says_checkpoint_met_when_lightgbm_is_better():
    text = cv_text(cv_findings(cv_table(cv_rows(0.17, 0.16))))
    assert "Checkpoint met" in text and "not met" not in text


def test_text_calls_a_big_random_vs_spatial_gap_optimistic_and_a_small_one_small():
    big = cv_text(cv_findings(cv_table(cv_rows(0.20, 0.21, logistic_random=0.15))))
    small = cv_text(cv_findings(cv_table(cv_rows(0.151, 0.21, logistic_random=0.15))))
    assert "random CV is optimistic" in big
    assert "almost the same" in small


def test_mask_summary_counts_flags_and_skips_no_data():
    codes = np.array([0, 1, 2, 3, NODATA, 0], dtype=float)
    m = mask_summary(codes)
    assert m["cells"] == 5
    assert m["width"] == pytest.approx(2 / 5) and m["extrapolation"] == pytest.approx(2 / 5)
    assert m["either"] == pytest.approx(3 / 5)


def test_province_table_is_predicted_minus_published_in_points():
    p = pd.DataFrame({"province": ["A"], "pred_grid": [0.6], "eicv7_grid": [0.5], "pred_solar": [0.1], "eicv7_solar": [0.2]})
    t = province_table(p)
    assert t.loc[0, "grid_pred_minus_eicv7"] == pytest.approx(10.0)
    assert t.loc[0, "solar_pred_minus_eicv7"] == pytest.approx(-10.0)


def test_md_table_formats_numbers_and_leaves_blanks_for_missing():
    t = md_table(pd.DataFrame({"a": ["x", "y"], "b": [0.12345, np.nan]}), {"b": "{:.2f}"})
    assert t.splitlines() == ["| a | b |", "|---|---|", "| x | 0.12 |", "| y |  |"]


def test_limitations_statement_fits_one_page_and_has_the_required_parts():
    text = (ROOT / "outputs" / "limitations_rwanda.md").read_text()
    assert len(text.split()) <= 500
    for part in ["What the model cannot see", "Where it will be wrong", "misallocate capital"]:
        assert part in text
