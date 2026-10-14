"""Tests for src/solar_test.py. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_solar_test.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.solar_test import (district_bootstrap_diff, flat_model, make_ratio_model, metrics,
                            solar_among_nongrid, weighted_mae)


def table(solar, unserved, n=20):
    solar, unserved = np.array(solar, float), np.array(unserved, float)
    return pd.DataFrame({"share_solar": solar, "share_unserved": unserved,
                         "share_grid": 1 - solar - unserved, "n_households": n})


def test_target_is_solar_share_of_nongrid_with_nongrid_weights():
    t = table([0.2, 0.0, 0.0], [0.2, 0.5, 0.0])
    target, w, keep = solar_among_nongrid(t, 0.05)
    assert target[0] == pytest.approx(0.5) and target[1] == pytest.approx(0.0)
    assert w[:2] == pytest.approx([20 * 0.4, 20 * 0.5])
    assert keep.tolist() == [True, True, False]             # all-grid cluster is left out
    assert np.isnan(target[2])                              # never divides by zero


def test_non_grid_threshold_is_inclusive():
    _, _, keep = solar_among_nongrid(table([0.03, 0.05, 0.1], [0.02, 0.0, 0.0]), 0.05)
    assert keep.tolist() == [True, True, True]


def test_flat_model_is_the_weighted_mean():
    pred = flat_model(None, pd.DataFrame(index=range(3)), np.array([1.0, 0.0]), np.array([3.0, 1.0]))
    assert pred == pytest.approx([0.75, 0.75, 0.75])


def test_ratio_model_learns_a_signal_and_stays_between_zero_and_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    p = 1 / (1 + np.exp(-2 * x))
    train = pd.DataFrame({"signal": x})
    model = make_ratio_model(["signal"], [], seed=1)
    pred = model(train, pd.DataFrame({"signal": [-2.0, 0.0, 2.0]}), p, np.full(400, 25.0))
    assert pred[0] < pred[1] < pred[2]
    assert ((pred > 0) & (pred < 1)).all()


def test_ratio_model_with_uninformative_feature_predicts_the_mean():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"noise": rng.normal(size=500)})
    target = np.full(500, 0.3)
    pred = make_ratio_model(["noise"], [], seed=1)(train, pd.DataFrame({"noise": [-1.0, 1.0]}), target, np.ones(500))
    assert pred == pytest.approx([0.3, 0.3], abs=0.03)


def test_weighted_mae():
    assert weighted_mae(np.array([0.0, 1.0]), np.array([0.0, 0.5]), np.array([3.0, 1.0])) == pytest.approx(0.125)


def test_metrics_for_a_perfect_ranking_and_a_flat_prediction():
    target = np.array([0.1, 0.2, 0.8, 0.9])
    w = np.ones(4)
    flat_mae = weighted_mae(target, np.full(4, 0.5), w)
    perfect = metrics(target, target, w, flat_mae)
    assert perfect["mae"] == 0 and perfect["skill"] == pytest.approx(1.0)
    assert perfect["auc_solar_majority"] == pytest.approx(1.0) and perfect["spearman"] == pytest.approx(1.0)
    assert metrics(target, np.full(4, 0.5), w, flat_mae)["skill"] == pytest.approx(0.0)


def test_bootstrap_interval_excludes_zero_for_a_clearly_better_model():
    groups = np.repeat(np.arange(30), 10)
    rng = np.random.default_rng(2)
    good, bad = rng.uniform(0.0, 0.1, 300), rng.uniform(0.2, 0.3, 300)
    mean, lo, hi = district_bootstrap_diff(good, bad, np.ones(300), groups, 300, seed=0)
    assert mean == pytest.approx(good.mean() - bad.mean()) and hi < 0


def test_bootstrap_interval_contains_zero_for_identical_models():
    groups = np.repeat(np.arange(20), 5)
    err = np.random.default_rng(3).uniform(0, 0.3, 100)
    mean, lo, hi = district_bootstrap_diff(err, err.copy(), np.ones(100), groups, 200, seed=0)
    assert mean == 0 and lo == 0 and hi == 0


def test_bootstrap_is_reproducible_and_resamples_whole_districts():
    groups = np.repeat(np.arange(10), 4)
    a, b = np.random.default_rng(4).uniform(0, 1, 40), np.random.default_rng(5).uniform(0, 1, 40)
    r1 = district_bootstrap_diff(a, b, np.ones(40), groups, 100, seed=7)
    r2 = district_bootstrap_diff(a, b, np.ones(40), groups, 100, seed=7)
    assert r1 == r2                                         # same seed, same answer
    assert r1[1] <= r1[2]
