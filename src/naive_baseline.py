"""Naive "dense and dark" index versus our market-reality layer.

Run (after src.export_eae and src.uncertainty; the cluster test needs the restricted train table):
    python -m src.naive_baseline --country rwanda
    python -m src.naive_baseline --demo         # synthetic data in data/demo/, reads and writes outputs/demo/

Naive index per 1 km cell = population x darkness (high = many people and little light, "underserved").
Darkness = 1 minus the percentile of the cell's night lights among populated cells (mid-rank for ties), so it runs
from about 0 (brightest) to about 1 (darkest) and does not depend on the units or the skew of the lights.
Population is the cell's own WorldPop count (population_total).

Two comparisons, both reported as plain numbers, not tuned:
  1. Cell level, over cells where both the naive index and our market-reality value exist: Spearman correlation,
     overlap of the top 10% (Jaccard, and share of the naive top set that is also in ours) counted in cells and in
     unserved households, how much of the naive top set our do-not-trust mask flags, and how much solar our own map
     predicts there (the "already succeeded" check).
  2. Cluster level, spatial CV (GroupKFold by district, the folds of src.train): rank agreement with the survey value
     unserved / (solar + unserved) among non-grid households (clusters with non-grid share under eae.min_nongrid_share
     are skipped, weight = non-grid households) for the naive index, night lights alone, and our model's
     out-of-fold value. The difference (ours minus naive) gets a 95% interval from resampling whole districts.
     The naive index and lights need no fitting, so only our model depends on the folds.

Inputs:  outputs/<country>_{market_reality_eae_utm,households,unserved_households,do_not_trust,share_solar}.tif,
         data/processed/<country>_features_1km.csv, _grid_cells.csv,
         data/restricted/features/<country>_train_table.csv (RESTRICTED, read only)
Outputs: outputs/naive_baseline_<country>.csv (one row per metric) and .md. Aggregates only.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import check_restricted_is_ignored, load_config, run_paths
from src.predict import DEMO_NOTICE
from src.solar_test import solar_among_nongrid
from src.train import cross_validate, make_logistic_model, make_splits
from src.validate import read_cell_values


# ---------------------------------------------------------------- pure helpers
def darkness(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """1 minus the mid-rank percentile of `values` within `reference` (NaN stays NaN).

    Mid-rank gives ties (many cells share the same dim light) the same value, centred on their rank range.
    """
    values = np.asarray(values, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float)[~np.isnan(reference)])
    out = np.full(values.shape, np.nan)
    ok = ~np.isnan(values)
    if len(ref) == 0:
        return out
    pct = (np.searchsorted(ref, values[ok], side="left") + np.searchsorted(ref, values[ok], side="right")) / (2 * len(ref))
    out[ok] = 1 - pct
    return out


def naive_index(population: np.ndarray, lights: np.ndarray, min_population: float, reference: np.ndarray = None) -> np.ndarray:
    """Population x darkness. NaN where population or lights is missing or the population is below `min_population`.

    `reference` = lights the percentile is taken against; default is the lights of the populated cells themselves.
    """
    population, lights = np.asarray(population, dtype=float), np.asarray(lights, dtype=float)
    populated = population >= min_population           # False for NaN
    if reference is None:
        reference = lights[populated]
    out = population * darkness(lights, reference)
    out[~populated] = np.nan
    return out


def top_flags(values: np.ndarray, fraction: float) -> np.ndarray:
    """True for the top `fraction` of the non-NaN values. Ties at the cut-off are all included, NaN is never flagged."""
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    flags = np.zeros(len(values), dtype=bool)
    if ok.sum() == 0:
        return flags
    flags[ok] = values[ok] >= np.quantile(values[ok], 1 - fraction)
    return flags


def overlap(a: np.ndarray, b: np.ndarray, weights: np.ndarray = None) -> dict:
    """Overlap of two boolean sets, optionally counted in `weights` (for example households) instead of cells.

    jaccard = |A and B| / |A or B|. a_in_b = share of A that is also in B. b_in_a = share of B that is also in A.
    NaN where the denominator is zero.
    """
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    both, either = w[a & b].sum(), w[a | b].sum()
    sum_a, sum_b = w[a].sum(), w[b].sum()
    return {"jaccard": both / either if either > 0 else np.nan,
            "a_in_b": both / sum_a if sum_a > 0 else np.nan,
            "b_in_a": both / sum_b if sum_b > 0 else np.nan}


def spearman(a: np.ndarray, b: np.ndarray, weights: np.ndarray = None) -> float:
    """Rank correlation over the pairs where both values exist (weighted Pearson on average ranks if `weights`).

    NaN if fewer than 3 pairs or one side is constant.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return np.nan
    ra, rb = pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy()
    w = np.ones(ok.sum()) if weights is None else np.asarray(weights, dtype=float)[ok]
    ma, mb = np.average(ra, weights=w), np.average(rb, weights=w)
    cov = np.average((ra - ma) * (rb - mb), weights=w)
    var = np.average((ra - ma) ** 2, weights=w) * np.average((rb - mb) ** 2, weights=w)
    return float(cov / np.sqrt(var)) if var > 0 else np.nan


def auc_top_group(score: np.ndarray, target: np.ndarray, fraction: float, weights: np.ndarray = None) -> float:
    """AUC of `score` for telling the top `fraction` of `target` from the rest. NaN if one group is empty."""
    score, target = np.asarray(score, dtype=float), np.asarray(target, dtype=float)
    ok = ~np.isnan(score) & ~np.isnan(target)
    label = top_flags(target[ok], fraction).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return np.nan
    w = None if weights is None else np.asarray(weights, dtype=float)[ok]
    return float(roc_auc_score(label, score[ok], sample_weight=w))


def district_bootstrap_metric_diff(score_a: np.ndarray, score_b: np.ndarray, target: np.ndarray, weights: np.ndarray,
                                   groups: np.ndarray, metric, n_boot: int, seed: int) -> tuple:
    """`metric(score_a) - metric(score_b)` and a 95% interval from resampling whole districts.

    metric(score, target, weights) -> float. Positive = a agrees better with the target than b. Draws where the
    metric is undefined are ignored.
    """
    rng = np.random.default_rng(seed)
    members = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    names = list(members)
    point = metric(score_a, target, weights) - metric(score_b, target, weights)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(names, size=len(names), replace=True)])
        draws[i] = metric(score_a[idx], target[idx], weights[idx]) - metric(score_b[idx], target[idx], weights[idx])
    if np.isnan(draws).all():
        return float(point), np.nan, np.nan
    return float(point), float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean over the entries where both exist. NaN if none."""
    ok = ~np.isnan(values) & ~np.isnan(weights)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() and weights[ok].sum() > 0 else np.nan


def cell_comparison(naive: np.ndarray, reality: np.ndarray, unserved_hh: np.ndarray, flagged: np.ndarray,
                    solar: np.ndarray, population: np.ndarray, fraction: float) -> dict:
    """Cell-level numbers. All inputs are one value per cell, NaN where missing.

    `flagged` is 1.0 where the do-not-trust mask is set, 0.0 where it is not, NaN where there is no prediction.
    The overlap tests use the cells where both naive and reality exist, each top set taken inside that shared set.
    The do-not-trust and solar checks use the top set of the naive index over all cells that have one.
    """
    shared = ~np.isnan(naive) & ~np.isnan(reality)
    out = {"n_shared": int(shared.sum()), "spearman": spearman(naive[shared], reality[shared])}
    top_naive, top_ours = top_flags(np.where(shared, naive, np.nan), fraction), top_flags(np.where(shared, reality, np.nan), fraction)
    for name, w in [("cells", None), ("households", np.nan_to_num(unserved_hh))]:
        o = overlap(top_naive, top_ours, w)
        out[f"jaccard_{name}"], out[f"naive_in_ours_{name}"] = o["jaccard"], o["a_in_b"]
    out["chance_share"] = fraction                      # expected overlap share if the two sets were unrelated
    top_all = top_flags(naive, fraction)
    has_pred = ~np.isnan(flagged)
    out["naive_top_cells"] = int(top_all.sum())
    out["naive_top_no_reality"] = float((top_all & np.isnan(reality)).sum() / max(top_all.sum(), 1))
    out["naive_top_flagged"] = weighted_mean(flagged[top_all & has_pred], np.ones((top_all & has_pred).sum()))
    out["all_flagged"] = weighted_mean(flagged[has_pred], np.ones(has_pred.sum()))
    out["naive_top_solar"] = weighted_mean(solar[top_all], population[top_all])
    out["national_solar"] = weighted_mean(solar, population)
    out["naive_top_reality"] = weighted_mean(reality[top_all], population[top_all])
    out["national_reality"] = weighted_mean(reality, population)
    return out


def conclusion(row: dict) -> str:
    """One plain sentence chosen from the numbers: does our layer differ from the naive one, and is it better."""
    parts = []
    rho, share = row["cell_spearman"], row["cell_naive_in_ours_cells"]
    if rho <= -0.3:
        parts.append("At cell level the two layers rank cells in roughly opposite order.")
    elif rho < 0.3 and share < 0.3:
        parts.append("At cell level the two layers are largely different.")
    elif rho >= 0.6 or share >= 0.6:
        parts.append("At cell level the two layers are close to each other.")
    else:
        parts.append("At cell level the two layers partly overlap.")
    d, lo, hi = row["cluster_auc_diff"], row["cluster_auc_diff_low"], row["cluster_auc_diff_high"]
    if np.isnan(d) or np.isnan(lo):
        parts.append("The held-out test could not be run.")
    elif lo > 0:
        parts.append("On held-out districts our model ranks the survey outcome better than the naive index.")
    elif hi < 0:
        parts.append("On held-out districts the naive index ranks the survey outcome better than our model.")
    else:
        parts.append("On held-out districts we cannot tell our model from the naive index (the interval for the "
                     "difference includes zero), so this test does not show that our layer is better.")
    return " ".join(parts)


# ---------------------------------------------------------------- output text
def write_markdown(path, country: str, m: dict, demo_notice: str = "") -> None:
    pct = lambda x: f"{100 * x:.0f}%"
    lights_line = ""
    if m["cluster_auc_lights"] < 0.5:
        lights_line = ("Night lights alone point the wrong way here: an AUC below 0.5 means darker clusters are LESS often "
                       "unserved (more often on solar) than brighter ones.\n\n")
    text = f"""# Naive "dense and dark" index versus our market-reality layer ({country})

{demo_notice}Naive index = population x darkness (darkness = 1 minus the percentile of night lights). We compared it with our layer
(unserved / (solar + unserved) among non-grid households) at cell level and, on held-out districts, against the survey.

**Cells** ({m['cell_n_shared']:,} cells with both values): Spearman correlation {m['cell_spearman']:.2f}. Of the naive top 10%,
{pct(m['cell_naive_in_ours_cells'])} is also in our top 10% by cells ({pct(m['cell_naive_in_ours_households'])} by unserved households;
Jaccard {m['cell_jaccard_cells']:.2f}); chance would give {pct(m['cell_chance_share'])}. Our do-not-trust mask flags
{pct(m['cell_naive_top_flagged'])} of the naive top 10% (all cells: {pct(m['cell_all_flagged'])}). Our predicted solar share there is
{pct(m['cell_naive_top_solar'])} against {pct(m['cell_national_solar'])} nationally (population-weighted).

**Held-out districts** ({m['cluster_n']} clusters, spatial cross-validation): Spearman with the survey value is
{m['cluster_spearman_naive']:.2f} naive, {m['cluster_spearman_lights']:.2f} night lights alone, {m['cluster_spearman_ours']:.2f} ours.
AUC for the top third is {m['cluster_auc_naive']:.2f}, {m['cluster_auc_lights']:.2f} and {m['cluster_auc_ours']:.2f}. Ours minus naive: AUC
{m['cluster_auc_diff']:+.2f} (95% interval {m['cluster_auc_diff_low']:+.2f} to {m['cluster_auc_diff_high']:+.2f}).

{lights_line}**Conclusion:** {conclusion(m)} Both rest on weak signals; read the layer as a hypothesis.
"""
    path.write_text(text)


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    parser.add_argument("--demo", action="store_true", help="use the synthetic data in data/demo/")
    args = parser.parse_args(argv)
    if args.demo:
        args.country = "demo"

    check_restricted_is_ignored()
    cfg = load_config()
    ncfg, mcfg = cfg["naive_baseline"], cfg["model"]
    paths = run_paths(cfg, args.country, args.demo)
    out_dir, c = paths["out_dir"], args.country
    seed = mcfg["seed"]
    min_pop, min_nongrid = mcfg["min_cell_population"], cfg["eae"]["min_nongrid_share"]
    radius = f"{int(mcfg['buffer_m'] / 1000)}km"
    rows_out = []

    def add(metric, value, n=np.nan, low=np.nan, high=np.nan, note=""):
        rows_out.append({"metric": metric, "value": value, "ci_low": low, "ci_high": high, "n": n, "note": note})

    # ---- cell level
    table = pd.read_csv(paths["features"])
    grid = pd.read_csv(paths["grid_cells"])[["cell_id", "row", "col"]]
    cells = table.merge(grid, on="cell_id", how="left", validate="one_to_one")
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    read = lambda name: read_cell_values(out_dir / f"{c}_{name}.tif", rows, cols)
    reality, unserved_hh, solar = read("market_reality_eae_utm"), read("unserved_households"), read("share_solar")
    flagged = read("do_not_trust")
    flagged = np.where(np.isnan(flagged), np.nan, (flagged > 0).astype(float))
    population = cells["population_total"].to_numpy(dtype=float)

    near = f"night_lights_bm_r{radius}"
    variants = [(ncfg["lights_column"], "cell")] if ncfg["lights_column"] in cells.columns else []
    variants += [(near, "neighbourhood")] if near in cells.columns else []
    if not variants:
        raise SystemExit(f"Neither {ncfg['lights_column']} nor {near} is in the 1 km table.")
    for k, (column, label) in enumerate(variants):
        naive = naive_index(population, cells[column].to_numpy(dtype=float), min_pop)
        result = cell_comparison(naive, reality, unserved_hh, flagged, solar, population, ncfg["top_fraction"])
        suffix = "" if k == 0 else f"_sens_{label}"
        note = f"lights = {column}" + ("" if k == 0 else " (sensitivity)")
        for key, value in result.items():
            add(f"cell_{key}{suffix}", value, n=result["n_shared"], note=note)
        if k == 0:
            cell_result = result
            print(f"Cell level, naive lights = {column}: {result['n_shared']:,} shared cells, Spearman {result['spearman']:.2f}, "
                  f"top-10% overlap {100 * result['naive_in_ours_cells']:.0f}% of cells, "
                  f"{100 * result['naive_in_ours_households']:.0f}% of unserved households")
    reference = cells.loc[population >= min_pop, near].to_numpy(dtype=float)   # cluster lights are neighbourhood means

    # ---- cluster level (aggregates only leave this function)
    check_restricted_is_ignored()
    core = mcfg["feature_sets"]["core"]["baseline"]
    train = pd.read_csv(paths["train"]).dropna(subset=core).reset_index(drop=True)
    target_solar, weights, keep = solar_among_nongrid(train, min_nongrid)
    splits = make_splits(train, "spatial", mcfg["cv_folds"], seed)
    oof = cross_validate(train, make_logistic_model(core, mcfg["log_features"], seed), splits)
    ours = oof[:, 2] / (oof[:, 1] + oof[:, 2])
    naive_c = naive_index(train["population"].to_numpy(), train["night_lights_bm"].to_numpy(), 0.0, reference)
    dark_c = darkness(train["night_lights_bm"].to_numpy(), reference)
    target = 1 - target_solar                                  # unserved / (solar + unserved)
    t, w, g = target[keep], weights[keep], train["district"].to_numpy()[keep]
    scores = {"naive": naive_c[keep], "lights": dark_c[keep], "ours": ours[keep]}
    frac = ncfg["cluster_top_fraction"]
    rank = lambda s, tt, ww: spearman(s, tt, ww)
    auc = lambda s, tt, ww: auc_top_group(s, tt, frac, ww)
    n_c = int(keep.sum())
    cluster = {"n": n_c}
    for name, s in scores.items():
        cluster[f"spearman_{name}"], cluster[f"auc_{name}"] = rank(s, t, w), auc(s, t, w)
        add(f"cluster_spearman_{name}", cluster[f"spearman_{name}"], n=n_c, note="survey unserved / (solar + unserved), spatial CV")
        add(f"cluster_auc_{name}", cluster[f"auc_{name}"], n=n_c, note="top tercile vs rest")
    for label, metric in [("spearman", rank), ("auc", auc)]:
        for other in ["naive", "lights"]:
            d, lo, hi = district_bootstrap_metric_diff(scores["ours"], scores[other], t, w, g, metric, ncfg["n_boot"], seed)
            add(f"cluster_{label}_diff" if other == "naive" else f"cluster_{label}_diff_vs_lights", d, n=n_c, low=lo, high=hi,
                note=f"ours minus {other}, 95% interval from resampling districts")
            if other == "naive":
                cluster[f"{label}_diff"], cluster[f"{label}_diff_low"], cluster[f"{label}_diff_high"] = d, lo, hi
    print(f"Cluster level: {n_c} of {len(train)} clusters, spatial CV. Spearman naive/lights/ours "
          f"{cluster['spearman_naive']:.2f}/{cluster['spearman_lights']:.2f}/{cluster['spearman_ours']:.2f}; "
          f"AUC {cluster['auc_naive']:.2f}/{cluster['auc_lights']:.2f}/{cluster['auc_ours']:.2f}; "
          f"AUC difference ours minus naive {cluster['auc_diff']:+.2f} [{cluster['auc_diff_low']:+.2f}, {cluster['auc_diff_high']:+.2f}]")

    result = pd.DataFrame(rows_out)
    result.to_csv(out_dir / f"naive_baseline_{c}.csv", index=False)
    m = {f"cell_{k}": v for k, v in cell_result.items()}
    m.update({f"cluster_{k}": v for k, v in cluster.items()})
    write_markdown(out_dir / f"naive_baseline_{c}.md", c, m, DEMO_NOTICE + "\n\n" if args.demo else "")
    print(conclusion(m))
    print(f"Wrote naive_baseline_{c}.csv and naive_baseline_{c}.md")


if __name__ == "__main__":
    main()
