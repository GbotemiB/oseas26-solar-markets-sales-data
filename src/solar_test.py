"""Test: among households without grid, does the model separate solar from unserved better than night lights?

Run (after src.features):
    python -m src.solar_test --country rwanda

Target per cluster: solar / (solar + unserved), the share of non-grid households that use solar
(one minus the market-reality ratio). Clusters with a non-grid share under 0.05 are left out, and
the rest are weighted by their number of non-grid households.

Models, all with the same spatial folds (groups = district) as src.train:
  flat            weighted mean of the training clusters
  lights only     logistic on log night lights (the "darkness" view)
  lights + pop    logistic on log night lights and log population
  our map         ratio taken from the multinomial model that makes the maps (out-of-fold)
  ratio model     the 7 core features fitted directly on this ratio

The gap to "lights only" is bootstrapped by district (resampling whole districts), because with a few
hundred clusters a difference can be noise. Decision rule: the map separates solar from unserved only
if the 95% interval of (map error - lights-only error) is entirely below zero.

Input:  data/restricted/features/<country>_train_table.csv (RESTRICTED, read only)
Output: outputs/solar_vs_unserved_test_<country>.csv (aggregate metrics only)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import check_restricted_is_ignored, load_config, repo_path
from src.train import cross_validate, make_features, make_logistic_model, make_splits


# ---------------------------------------------------------------- pure helpers
def solar_among_nongrid(table: pd.DataFrame, min_nongrid: float):
    """Target, weights and a keep-mask.

    target = solar / (solar + unserved). weight = non-grid households. Clusters with a non-grid
    share below `min_nongrid` get keep = False (the ratio is noise when almost everyone has grid).
    """
    nongrid = table["share_solar"] + table["share_unserved"]
    keep = (nongrid >= min_nongrid).to_numpy()
    target = (table["share_solar"] / nongrid.where(nongrid > 0)).to_numpy()
    weights = (table["n_households"] * nongrid).to_numpy()
    return target, weights, keep


def flat_model(train: pd.DataFrame, test: pd.DataFrame, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of the training target, everywhere. `target` and `weights` are aligned with `train`."""
    return np.full(len(test), np.average(target, weights=weights))


def make_ratio_model(features: list, log_features: list, seed: int):
    """Logistic regression for a 0-1 target: each cluster is two weighted rows (solar and not solar)."""
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
        X = make_features(train, features, log_features)
        X2 = pd.concat([X, X], ignore_index=True)
        y2 = np.r_[np.ones(len(X)), np.zeros(len(X))]
        w2 = np.r_[weights * target, weights * (1 - target)]
        keep = w2 > 0
        pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=2000))
        pipe.fit(X2[keep], y2[keep], logisticregression__sample_weight=w2[keep])
        return pipe.predict_proba(make_features(test, features, log_features))[:, 1]
    return fit_predict


def weighted_mae(true: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.abs(true - pred), weights=weights))


def district_bootstrap_diff(err_a: np.ndarray, err_b: np.ndarray, weights: np.ndarray, groups: np.ndarray,
                            n_boot: int, seed: int) -> tuple:
    """Weighted mean of (err_a - err_b), with a 95% interval from resampling whole districts.

    err_a and err_b are absolute errors of two models on the same clusters. Negative = model a is better.
    """
    rng = np.random.default_rng(seed)
    members = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    names = list(members)
    diff = err_a - err_b
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(names, size=len(names), replace=True)])
        draws[i] = np.average(diff[idx], weights=weights[idx])
    return float(np.average(diff, weights=weights)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def metrics(target: np.ndarray, pred: np.ndarray, weights: np.ndarray, flat_mae: float) -> dict:
    mae = weighted_mae(target, pred, weights)
    solar_majority = (target > 0.5).astype(int)
    auc = roc_auc_score(solar_majority, pred, sample_weight=weights) if 0 < solar_majority.sum() < len(target) else np.nan
    return {"mae": mae, "skill": 1 - mae / flat_mae,
            "spearman": float(pd.Series(pred).corr(pd.Series(target), method="spearman")), "auc_solar_majority": auc}


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    parser.add_argument("--min-nongrid", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    check_restricted_is_ignored()
    cfg = load_config()
    mcfg = cfg["model"]
    seed, core = mcfg["seed"], mcfg["feature_sets"]["core"]["baseline"]
    np.random.seed(seed)

    table = pd.read_csv(repo_path(cfg["paths"]["restricted"]) / "features" / f"{args.country}_train_table.csv")
    table = table.dropna(subset=core).reset_index(drop=True)
    target, weights, keep = solar_among_nongrid(table, args.min_nongrid)
    splits = make_splits(table, "spatial", mcfg["cv_folds"], seed)

    # our map: ratio from the multinomial model, out-of-fold on all clusters
    oof = cross_validate(table, make_logistic_model(core, mcfg["log_features"], seed), splits)
    map_pred = oof[:, 1] / (oof[:, 1] + oof[:, 2])

    ratio_models = {
        "lights only": make_ratio_model(["night_lights_bm"], mcfg["log_features"], seed),
        "lights + population": make_ratio_model(["night_lights_bm", "population"], mcfg["log_features"], seed),
        "ratio model (7 core features)": make_ratio_model(core, mcfg["log_features"], seed),
    }
    preds = {"flat": np.full(len(table), np.nan), **{name: np.full(len(table), np.nan) for name in ratio_models}}
    for train_idx, test_idx in splits:
        tr, te = train_idx[keep[train_idx]], test_idx[keep[test_idx]]
        preds["flat"][te] = flat_model(table.iloc[tr], table.iloc[te], target[tr], weights[tr])
        for name, model in ratio_models.items():
            preds[name][te] = model(table.iloc[tr], table.iloc[te], target[tr], weights[tr])
    preds["our map (multinomial)"] = map_pred

    t, w, g = target[keep], weights[keep], table["district"].to_numpy()[keep]
    preds = {name: p[keep] for name, p in preds.items()}
    flat_mae = weighted_mae(t, preds["flat"], w)
    rows = [{"model": name, "n_clusters": int(keep.sum()), **metrics(t, p, w, flat_mae)} for name, p in preds.items()]
    result = pd.DataFrame(rows)

    errors = {name: np.abs(t - p) for name, p in preds.items()}
    for ref in ["lights only", "flat"]:
        diffs = {name: district_bootstrap_diff(errors[name], errors[ref], w, g, args.n_boot, seed) for name in preds}
        result[f"mae_minus_{ref.replace(' ', '_')}"] = result["model"].map(lambda m: diffs[m][0])
        result[f"ci_low_vs_{ref.replace(' ', '_')}"] = result["model"].map(lambda m: diffs[m][1])
        result[f"ci_high_vs_{ref.replace(' ', '_')}"] = result["model"].map(lambda m: diffs[m][2])

    out_dir = repo_path(cfg["paths"]["outputs"])
    out_dir.mkdir(exist_ok=True)
    result.to_csv(out_dir / f"solar_vs_unserved_test_{args.country}.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{int(keep.sum())} of {len(table)} clusters have a non-grid share of at least {args.min_nongrid}; "
          f"target = solar / (solar + unserved), weighted by non-grid households")
    print(result[["model", "mae", "skill", "spearman", "auc_solar_majority"]].round(3).to_string(index=False))
    print("\nDifference in error vs lights only (negative = better), 95% interval from resampling districts:")
    print(result[["model", "mae_minus_lights_only", "ci_low_vs_lights_only", "ci_high_vs_lights_only"]].round(3).to_string(index=False))
    row = result.set_index("model").loc["our map (multinomial)"]
    verdict = row["ci_high_vs_lights_only"] < 0
    print(f"\nDecision rule (map beats lights only, interval below zero): {'MET' if verdict else 'NOT MET'}")


if __name__ == "__main__":
    main()
