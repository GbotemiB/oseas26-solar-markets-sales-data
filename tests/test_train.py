"""Tests for src/train.py. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_train.py -v
"""
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from src.train import (CLASSES, SHARE_COLS, constant_model, cross_validate, expand_soft_labels,
                       make_catboost_model, make_features, make_forest_model, make_lightgbm_model,
                       make_logistic_model, make_splits, make_xgboost_model, score, slug, weighted_mae)


def synthetic_table(n_districts=10, per_district=12, seed=0) -> pd.DataFrame:
    """Clusters where the grid share rises with one feature and solar takes most of the rest."""
    rng = np.random.default_rng(seed)
    n = n_districts * per_district
    x = rng.normal(size=n)
    grid = 1 / (1 + np.exp(-3 * x))
    solar = (1 - grid) * 0.7
    shares = np.column_stack([grid, solar, 1 - grid - solar])
    return pd.DataFrame({
        "cluster": np.arange(n), "district": np.repeat([f"D{i}" for i in range(n_districts)], per_district),
        "n_households": rng.integers(20, 30, size=n), "signal": x, "noise": rng.normal(size=n),
        **{col: shares[:, k] for k, col in enumerate(SHARE_COLS)}})


def test_weighted_mae_known_values():
    true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pred = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    out = weighted_mae(true, pred, np.array([1.0, 1.0]))
    assert out["grid"] == pytest.approx(0.25) and out["solar"] == pytest.approx(0.25)
    assert out["unserved"] == 0.0 and out["mean"] == pytest.approx(0.5 / 3)
    # weights matter: the perfect cluster counts 3 times as much
    out_w = weighted_mae(true, pred, np.array([1.0, 3.0]))
    assert out_w["grid"] == pytest.approx(0.125)


def test_expand_soft_labels_shapes_and_weights():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    shares = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    Xe, ye, we = expand_soft_labels(X, shares, np.array([10.0, 20.0]))
    assert len(Xe) == len(ye) == len(we) == 3          # zero-weight rows are dropped
    assert we.sum() == pytest.approx(30.0)              # total weight = total households
    assert sorted(ye.tolist()) == [0, 1, 2]
    assert we[ye == 2].tolist() == [20.0]


def test_make_features_logs_only_requested_columns():
    t = pd.DataFrame({"lights": [0.0, np.e - 1], "slope": [5.0, 7.0]})
    X = make_features(t, ["lights", "slope"], ["lights", "not_in_table"])
    assert X["lights"].tolist() == pytest.approx([0.0, 1.0])
    assert X["slope"].tolist() == [5.0, 7.0]


def test_constant_model_returns_weighted_mean_and_sums_to_one():
    train = pd.DataFrame({"n_households": [10, 30],
                          "share_grid": [1.0, 0.0], "share_solar": [0.0, 1.0], "share_unserved": [0.0, 0.0]})
    pred = constant_model(train, pd.DataFrame(index=range(4)))
    assert pred.shape == (4, 3)
    assert pred[0].tolist() == pytest.approx([0.25, 0.75, 0.0])


def test_spatial_splits_never_share_a_district_but_random_splits_do():
    t = synthetic_table()
    for train_idx, test_idx in make_splits(t, "spatial", 5, seed=1):
        assert set(t.district.iloc[train_idx]).isdisjoint(set(t.district.iloc[test_idx]))
    shared = [set(t.district.iloc[tr]) & set(t.district.iloc[te]) for tr, te in make_splits(t, "random", 5, seed=1)]
    assert all(len(s) > 0 for s in shared)
    with pytest.raises(ValueError):
        make_splits(t, "nonsense", 5, seed=1)


def test_every_cluster_is_predicted_once_and_shares_sum_to_one():
    t = synthetic_table()
    splits = make_splits(t, "spatial", 5, seed=1)
    model = make_logistic_model(["signal", "noise"], [], seed=1)
    oof = cross_validate(t, model, splits)
    assert not np.isnan(oof).any()
    assert oof.sum(axis=1) == pytest.approx(1.0)


def test_logistic_beats_constant_when_there_is_a_real_signal():
    t = synthetic_table()
    splits = make_splits(t, "spatial", 5, seed=1)
    const = score(t, cross_validate(t, constant_model, splits), splits)
    logit = score(t, cross_validate(t, make_logistic_model(["signal", "noise"], [], seed=1), splits), splits)
    assert logit["mean"] < const["mean"]
    assert logit["fold_min"] <= logit["fold_max"]


def test_logistic_is_no_better_than_constant_on_pure_noise():
    t = synthetic_table()
    rng = np.random.default_rng(5)
    t["signal"], t["noise"] = rng.normal(size=len(t)), rng.normal(size=len(t))
    splits = make_splits(t, "spatial", 5, seed=1)
    const = score(t, cross_validate(t, constant_model, splits), splits)["mean"]
    logit = score(t, cross_validate(t, make_logistic_model(["signal", "noise"], [], seed=1), splits), splits)["mean"]
    assert logit > const * 0.9          # no big gain from random features


SMALL_LGBM = {"num_leaves": 4, "min_child_samples": 5, "n_estimators": 30, "learning_rate": 0.1}


def test_lightgbm_predictions_are_valid_shares():
    t = synthetic_table()
    splits = make_splits(t, "spatial", 5, seed=1)
    oof = cross_validate(t, make_lightgbm_model(["signal", "noise"], SMALL_LGBM, seed=1), splits)
    assert not np.isnan(oof).any()
    assert oof.sum(axis=1) == pytest.approx(1.0)
    assert (oof >= 0).all() and (oof <= 1).all()


def test_lightgbm_beats_constant_when_there_is_a_real_signal():
    t = synthetic_table()
    splits = make_splits(t, "spatial", 5, seed=1)
    const = score(t, cross_validate(t, constant_model, splits), splits)["mean"]
    lgbm = score(t, cross_validate(t, make_lightgbm_model(["signal", "noise"], SMALL_LGBM, seed=1), splits), splits)["mean"]
    assert lgbm < const


def test_lightgbm_is_reproducible_with_the_same_seed():
    t = synthetic_table()
    train, test = t.iloc[:90], t.iloc[90:]
    a = make_lightgbm_model(["signal", "noise"], SMALL_LGBM, seed=3)(train, test)
    b = make_lightgbm_model(["signal", "noise"], SMALL_LGBM, seed=3)(train, test)
    assert np.allclose(a, b)


FEATS = ["signal", "noise"]
MODEL_FACTORIES = {
    "random_forest": lambda: make_forest_model(RandomForestRegressor, FEATS, {"n_estimators": 50, "min_samples_leaf": 5}, 1),
    "extra_trees": lambda: make_forest_model(ExtraTreesRegressor, FEATS, {"n_estimators": 50, "min_samples_leaf": 5}, 1),
    "xgboost": lambda: make_xgboost_model(
        FEATS, {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1, "min_child_weight": 2}, 1),
    "catboost": lambda: make_catboost_model(FEATS, {"iterations": 60, "depth": 3, "learning_rate": 0.1}, 1),
    "elastic_net": lambda: make_logistic_model(
        FEATS, [], 1, {"solver": "saga", "l1_ratio": 0.5, "C": 0.3, "max_iter": 5000}),
}


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_other_models_give_valid_shares_and_beat_constant_on_real_signal(name):
    t = synthetic_table()
    splits = make_splits(t, "spatial", 5, seed=1)
    oof = cross_validate(t, MODEL_FACTORIES[name](), splits)
    assert not np.isnan(oof).any()
    assert oof.sum(axis=1) == pytest.approx(1.0)
    assert (oof >= 0).all() and (oof <= 1).all()
    const = score(t, cross_validate(t, constant_model, splits), splits)["mean"]
    assert score(t, oof, splits)["mean"] < const


def test_catboost_returns_probabilities_not_log_odds():
    t = synthetic_table()
    train, test = t.iloc[:90], t.iloc[90:]
    pred = make_catboost_model(FEATS, {"iterations": 60, "depth": 3, "learning_rate": 0.1}, 1)(train, test)
    # grid share is high when signal is high: predictions must follow the data scale (0-1), not log-odds
    assert np.corrcoef(pred[:, 0], test["share_grid"])[0, 1] > 0.5
    assert abs(pred[:, 0].mean() - test["share_grid"].mean()) < 0.2


def test_slug_makes_column_safe_names():
    assert slug("lightgbm (regularised)") == "lightgbm_regularised"
    assert slug("elastic-net logistic") == "elastic_net_logistic"
