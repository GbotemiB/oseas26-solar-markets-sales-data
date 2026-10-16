"""Tests for src/uncertainty.py. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_uncertainty.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.uncertainty import (apply_interval, assign_bins, calibrate, conformal_level, do_not_trust_codes,
                             extrapolation_flag, nested_coverage, weighted_quantile)


def noisy_data(n=1200, noise=0.1, seed=0):
    """Predictions in 0.2..0.8 and truth = prediction + Gaussian noise, so errors are known."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.2, 0.8, size=(n, 3))
    true = pred + rng.normal(0, noise, size=(n, 3))
    return pred, true, rng.integers(20, 30, n).astype(float)


def test_weighted_quantile_equal_weights_and_heavy_weight():
    v = np.arange(1.0, 11.0)
    assert weighted_quantile(v, np.ones(10), 0.5) == 5.0
    assert weighted_quantile(v, np.ones(10), 1.0) == 10.0
    w = np.array([100.0] + [1.0] * 9)                       # one value carries almost all the weight
    assert weighted_quantile(v, w, 0.5) == 1.0


def test_conformal_level_is_above_target_but_capped_at_one():
    assert conformal_level(0.8, 100) == pytest.approx(0.808)
    assert conformal_level(0.99, 2) == 1.0


def test_assign_bins():
    assert assign_bins(np.array([0.1, 0.5, 0.9]), np.array([0.3, 0.7])).tolist() == [0, 1, 2]


def test_calibrated_width_matches_known_noise():
    pred, true, w = noisy_data(n=4000, noise=0.1)
    edges, half = calibrate(pred, true, w, coverage=0.8, n_bins=4, min_bin_clusters=20)
    assert edges.shape == (3, 3) and half.shape == (3, 4)
    assert half == pytest.approx(0.128, abs=0.015)          # 80% of |N(0, 0.1)| is 1.28 x 0.1


def test_wider_noise_in_one_bin_gives_a_wider_range_there():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0, 1, size=(3000, 1))
    noise = np.where(pred < 0.5, 0.02, 0.2) * rng.normal(size=pred.shape)
    _, half = calibrate(pred, pred + noise, np.ones(3000), 0.8, 2, 20)
    assert half[0, 1] > 3 * half[0, 0]


def test_small_bins_fall_back_to_the_pooled_width():
    pred = np.linspace(0.1, 0.9, 30).reshape(-1, 1)
    rng = np.random.default_rng(2)
    _, half = calibrate(pred, pred + rng.normal(0, 0.05, pred.shape), np.ones(30), 0.8, 4, min_bin_clusters=20)
    assert np.allclose(half[0], half[0, 0])                 # every bin has under 20 clusters: all pooled


def test_interval_is_clipped_and_nan_is_kept():
    pred = np.array([[0.02, 0.5, 0.98], [np.nan, np.nan, np.nan]])
    edges, half = np.full((3, 1), 0.5), np.full((3, 2), 0.1)
    low, high = apply_interval(pred, edges, half)
    assert low[0] == pytest.approx([0.0, 0.4, 0.88]) and high[0] == pytest.approx([0.12, 0.6, 1.0])
    assert np.isnan(low[1]).all() and np.isnan(high[1]).all()


def test_nested_coverage_is_close_to_target_on_held_out_folds():
    pred, true, w = noisy_data(n=1500, noise=0.1, seed=3)
    folds = np.arange(len(pred)) % 5
    splits = [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in range(5)]
    out = nested_coverage(pred, true, w, splits, 0.8, 4, 20)
    overall = out[out["bin"] == "all"]
    assert (overall["coverage"].between(0.76, 0.86)).all()
    assert set(out["bin"]) == {"all", "0", "1", "2", "3"}


def test_extrapolation_flag_marks_cells_outside_the_cluster_range():
    train = pd.DataFrame({"a": np.arange(101.0), "b": np.arange(101.0)})
    cells = pd.DataFrame({"a": [50.0, 150.0, np.nan, 50.0], "b": [50.0, 50.0, 50.0, -20.0]})
    assert extrapolation_flag(cells, train, [1, 99]).tolist() == [False, True, False, True]


def test_do_not_trust_codes_combine_width_ranking_and_extrapolation():
    low = np.array([[0.4, 0.4, 0.4], [0.0, 0.4, 0.4], [0.0, 0.4, 0.4], [np.nan] * 3])
    high = np.array([[0.6, 0.6, 0.6], [0.5, 0.6, 0.6], [0.5, 0.6, 0.6], [np.nan] * 3])
    codes, cut_off = do_not_trust_codes(low, high, np.array([False, False, True, True]), widest_share=0.67)
    assert 0.2 < cut_off < 0.5                              # widest range per cell: 0.2, 0.5, 0.5
    assert codes[:3].tolist() == [0.0, 1.0, 3.0]
    assert np.isnan(codes[3])                               # no prediction, no flag


def test_width_flag_marks_the_requested_share_of_cells():
    rng = np.random.default_rng(0)
    low = np.zeros((1000, 3))
    high = rng.uniform(0.1, 0.9, size=(1000, 3))
    codes, _ = do_not_trust_codes(low, high, np.zeros(1000, dtype=bool), widest_share=0.25)
    assert (codes == 1).mean() == pytest.approx(0.25, abs=0.01)
    assert (codes == 2).sum() == 0
