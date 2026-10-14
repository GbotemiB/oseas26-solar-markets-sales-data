"""Tests for src/validate.py. Synthetic data only.

Run from the repo root:  python -m pytest tests/test_validate.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.validate import compare, group_shares, match_report


def test_group_shares_weights_by_population_and_skips_empty_cells():
    keys = pd.DataFrame({"area": ["A", "A", "B", "B", "C"]})
    shares = np.array([[1.0, 0, 0], [0.0, 1.0, 0], [0, 0, 1.0], [np.nan] * 3, [0.5, 0.5, 0]])
    pop = np.array([3.0, 1.0, 5.0, 100.0, 0.0])
    t = group_shares(keys, shares, pop).set_index("area")
    assert t.loc["A", "pred_grid"] == pytest.approx(0.75) and t.loc["A", "pred_solar"] == pytest.approx(0.25)
    assert t.loc["B", "pred_unserved"] == pytest.approx(1.0) and t.loc["B", "population"] == 5.0
    assert "C" not in t.index                            # no population: dropped, not divided by zero


def test_group_shares_with_two_keys_and_missing_key():
    keys = pd.DataFrame({"d": ["A", "A", "B", None], "s": ["x", "x", "x", "x"]})
    shares = np.tile([0.2, 0.3, 0.5], (4, 1))
    t = group_shares(keys, shares, np.ones(4))
    assert sorted(zip(t["d"], t["s"])) == [("A", "x"), ("B", "x")]      # the row with a missing key is ignored
    assert t.set_index("d").loc["A", "population"] == 2.0


def test_compare_perfect_and_biased_predictions():
    truth = np.array([0.2, 0.4, 0.6, 0.8])
    perfect = compare(truth.copy(), truth)
    assert perfect["mae"] == 0 and perfect["pearson"] == pytest.approx(1.0) and perfect["skill"] == pytest.approx(1.0)
    shifted = compare(truth + 0.1, truth)
    assert shifted["bias"] == pytest.approx(0.1) and shifted["mae"] == pytest.approx(0.1)
    assert shifted["mae_no_bias"] == pytest.approx(0.0, abs=1e-12)     # right pattern, wrong level
    assert shifted["spearman"] == pytest.approx(1.0)


def test_compare_flat_prediction_has_zero_skill():
    truth = np.array([0.2, 0.4, 0.6, 0.8])
    flat = compare(np.full(4, 0.5), truth)
    assert flat["skill"] == pytest.approx(0.0)


def test_compare_weights_and_interval_coverage():
    truth = np.array([0.5, 0.5])
    out = compare(np.array([0.5, 0.9]), truth, weights=np.array([3.0, 1.0]),
                  ci=(np.array([0.4, 0.4]), np.array([0.6, 0.6])))
    assert out["mae"] == pytest.approx(0.1)                 # (3 x 0 + 1 x 0.4) / 4
    assert out["within_ci"] == pytest.approx(0.5)


def test_match_report_lists_unmatched_names():
    assert match_report(pd.Series(["a", "b"]), pd.Series(["b", "c"])) == (["a"], ["c"])
