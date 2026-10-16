"""Baseline (logistic regression) and tree models, random CV vs spatial block CV.

Run (after src.features):
    python -m src.train --country rwanda

Inputs:
    data/restricted/features/<country>_train_table.csv    one row per DHS cluster (RESTRICTED)

Outputs:
    outputs/cv_results_<country>.csv                       aggregate error numbers only (safe to publish)
    data/restricted/models/<country>_oof.csv               out-of-fold predictions per cluster (restricted)

Why soft labels: each cluster has three shares (grid, solar, unserved), not one class.
Logistic regression in scikit-learn wants one class per row, so each cluster becomes three rows,
one per class, weighted by households x share. This is the same likelihood as fitting the shares.

Boosted trees (LightGBM, XGBoost, CatBoost): one model per share, loss for targets between 0 and 1,
weight = households per cluster, the three predictions rescaled to sum to 1. Random forest and
extra trees predict all three shares at once. Every model gets the same folds.

Metric: household-weighted mean absolute error (MAE) of the three shares, averaged over classes.
Random CV (5 folds) is optimistic because neighbouring clusters share districts. Spatial CV
(GroupKFold, groups = district) is the honest score.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.special import expit
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import check_restricted_is_ignored, load_config, repo_path

CLASSES = ["grid", "solar", "unserved"]
SHARE_COLS = [f"share_{c}" for c in CLASSES]


# ---------------------------------------------------------------- data helpers
def make_features(table: pd.DataFrame, features: list, log_features: list) -> pd.DataFrame:
    """Pick the feature columns and apply log1p to the skewed ones (lights, population, buildings)."""
    X = table[features].copy()
    for name in log_features:
        if name in X.columns:
            X[name] = np.log1p(X[name])
    return X


def expand_soft_labels(X: pd.DataFrame, shares: np.ndarray, weights: np.ndarray):
    """Turn n clusters with 3 shares each into 3n rows with one class each.

    Row weight = household count x share. Rows with zero weight are dropped.
    """
    n = len(X)
    Xe = pd.concat([X] * len(CLASSES), ignore_index=True)
    ye = np.repeat(np.arange(len(CLASSES)), n)
    we = np.concatenate([weights * shares[:, k] for k in range(len(CLASSES))])
    keep = we > 0
    return Xe[keep], ye[keep], we[keep]


def weighted_mae(true: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> dict:
    """Household-weighted MAE per class and the mean over the three classes."""
    err = np.abs(true - pred)
    per_class = (err * weights[:, None]).sum(axis=0) / weights.sum()
    out = {c: per_class[k] for k, c in enumerate(CLASSES)}
    out["mean"] = per_class.mean()
    return out


# ---------------------------------------------------------------- models
# A model is a function (train_table, test_table) -> predicted shares, shape (n_test, 3).
def constant_model(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict the household-weighted mean shares of the training clusters everywhere."""
    w = train["n_households"].to_numpy()
    mean = (train[SHARE_COLS].to_numpy() * w[:, None]).sum(axis=0) / w.sum()
    return np.tile(mean, (len(test), 1))


def make_logistic_model(features: list, log_features: list, seed: int, params: dict = None):
    params = {"max_iter": 2000} if params is None else params
    def logistic_model(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        Xe, ye, we = expand_soft_labels(
            make_features(train, features, log_features),
            train[SHARE_COLS].to_numpy(), train["n_households"].to_numpy())
        pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **params))
        pipe.fit(Xe, ye, logisticregression__sample_weight=we)
        return pipe.predict_proba(make_features(test, features, log_features))   # columns follow CLASSES
    return logistic_model


def per_share_model(fit_predict, features: list):
    """Fit one model per share with `fit_predict(X, y, weights, X_test)`, then rescale to sum to 1."""
    def model(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        weights = train["n_households"].to_numpy()
        preds = np.column_stack([
            fit_predict(train[features], train[col].to_numpy(), weights, test[features])
            for col in SHARE_COLS])
        return preds / preds.sum(axis=1, keepdims=True)
    return model


def make_lightgbm_model(features: list, params: dict, seed: int):
    def fit_predict(X, y, w, X_test):
        model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, verbose=-1, **params)
        return model.fit(X, y, sample_weight=w).predict(X_test)
    return per_share_model(fit_predict, features)


def make_xgboost_model(features: list, params: dict, seed: int):
    def fit_predict(X, y, w, X_test):
        model = xgb.XGBRegressor(objective="reg:logistic", random_state=seed, n_jobs=1, **params)
        return model.fit(X, y, sample_weight=w).predict(X_test)
    return per_share_model(fit_predict, features)


def make_catboost_model(features: list, params: dict, seed: int):
    def fit_predict(X, y, w, X_test):
        # CrossEntropy takes fractional targets (0-1) but only exists in the classifier class
        model = CatBoostClassifier(loss_function="CrossEntropy", random_seed=seed, verbose=0,
                                   allow_writing_files=False, thread_count=1, **params)
        model.fit(Pool(X, label=y, weight=w))
        return expit(model.predict(Pool(X_test), prediction_type="RawFormulaVal"))   # log-odds -> share
    return per_share_model(fit_predict, features)


def make_forest_model(estimator_class, features: list, params: dict, seed: int):
    """Random forest or extra trees: one model for all three shares."""
    def forest_model(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = estimator_class(random_state=seed, n_jobs=-1, **params)
        model.fit(train[features], train[SHARE_COLS], sample_weight=train["n_households"])
        preds = model.predict(test[features])
        return preds / preds.sum(axis=1, keepdims=True)
    return forest_model


# ---------------------------------------------------------------- cross-validation
def make_splits(table: pd.DataFrame, scheme: str, n_folds: int, seed: int):
    if scheme == "random":
        return list(KFold(n_folds, shuffle=True, random_state=seed).split(table))
    if scheme == "spatial":
        return list(GroupKFold(n_folds).split(table, groups=table["district"]))
    raise ValueError(f"Unknown CV scheme: {scheme}")


def cross_validate(table: pd.DataFrame, model, splits: list) -> np.ndarray:
    """Out-of-fold predictions: every cluster is predicted by a model that never saw it."""
    oof = np.full((len(table), len(CLASSES)), np.nan)
    for train_idx, test_idx in splits:
        oof[test_idx] = model(table.iloc[train_idx], table.iloc[test_idx])
    return oof


def score(table: pd.DataFrame, oof: np.ndarray, splits: list) -> dict:
    """Pooled MAE over all clusters, plus the spread of the mean MAE across folds."""
    true, w = table[SHARE_COLS].to_numpy(), table["n_households"].to_numpy()
    pooled = weighted_mae(true, oof, w)
    fold_means = [weighted_mae(true[te], oof[te], w[te])["mean"] for _, te in splits]
    pooled["fold_min"], pooled["fold_max"] = min(fold_means), max(fold_means)
    return pooled


# ---------------------------------------------------------------- main
def slug(name: str) -> str:
    """Model name as a column-friendly word: 'lightgbm (regularised)' -> 'lightgbm_regularised'."""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").replace("__", "_")


def build_models(mcfg: dict, base: list, full: list, seed: int) -> dict:
    """All models for one feature set. `base` feeds the linear models, `full` the tree models."""
    log = mcfg["log_features"]
    return {
        "constant (national mean)": constant_model,
        "logistic regression": make_logistic_model(base, log, seed),
        "elastic-net logistic": make_logistic_model(base, log, seed, mcfg["elastic_net"]),
        "random forest": make_forest_model(RandomForestRegressor, full, mcfg["random_forest"], seed),
        "extra trees": make_forest_model(ExtraTreesRegressor, full, mcfg["extra_trees"], seed),
        "lightgbm": make_lightgbm_model(full, mcfg["lightgbm"], seed),
        "lightgbm (regularised)": make_lightgbm_model(full, mcfg["lightgbm_regularised"], seed),
        "xgboost": make_xgboost_model(full, mcfg["xgboost"], seed),
        "catboost": make_catboost_model(full, mcfg["catboost"], seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()

    check_restricted_is_ignored()
    cfg = load_config()
    mcfg = cfg["model"]
    seed, n_folds = mcfg["seed"], mcfg["cv_folds"]
    np.random.seed(seed)

    restricted = repo_path(cfg["paths"]["restricted"])
    table = pd.read_csv(restricted / "features" / f"{args.country}_train_table.csv")

    # feature sets whose columns are all in the train table (plus_osm needs src.distance_layers first)
    sets = {}
    for set_name, fs in mcfg["feature_sets"].items():
        missing = sorted(set(fs["baseline"] + fs["all"]) - set(table.columns))
        if missing:
            print(f"Skipping feature set '{set_name}': not in the train table: {', '.join(missing)}")
        else:
            sets[set_name] = fs
    used = sorted({f for fs in sets.values() for f in fs["all"] + fs["baseline"]})
    table = table.dropna(subset=used).reset_index(drop=True)
    print(f"{len(table)} clusters, {table['district'].nunique()} districts, "
          f"{int(table['n_households'].sum()):,} households")

    rows, oof_out = [], table[["cluster", "district"]].copy()
    for set_name, fs in sets.items():
        models = build_models(mcfg, fs["baseline"], fs["all"], seed)
        for scheme in ["random", "spatial"]:
            splits = make_splits(table, scheme, n_folds, seed)    # same folds for every model and set
            for name, model in models.items():
                oof = cross_validate(table, model, splits)
                rows.append({"feature_set": set_name, "cv": scheme, "model": name,
                             **score(table, oof, splits)})
                if name != "constant (national mean)":
                    for k, c in enumerate(CLASSES):
                        oof_out[f"pred_{set_name}_{slug(name)}_{c}_{scheme}"] = oof[:, k]

    results = pd.DataFrame(rows)
    out_dir = repo_path(cfg["paths"]["outputs"])
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_dir / f"cv_results_{args.country}.csv", index=False)
    model_dir = restricted / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    oof_out.to_csv(model_dir / f"{args.country}_oof.csv", index=False)

    print(f"\nHousehold-weighted MAE of shares (0-1 scale, lower is better), {n_folds}-fold CV")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
