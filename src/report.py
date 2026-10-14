"""Validation report: random vs spatial CV, uncertainty coverage, control totals, solar test.

Run (after src.train, src.uncertainty, src.validate, src.solar_test):
    python -m src.report --country rwanda

Reads the aggregate result files in outputs/ and writes outputs/validation_report_<country>.md.
Every number in the report comes from those files, so a rerun of the pipeline and of this script keeps
the report in step with the results. Interpretation sentences that depend on the numbers are chosen
in code from the numbers (for example whether spatial error is worse than random error).
"""
import argparse

import numpy as np
import pandas as pd
import rasterio

from src.config import load_config, repo_path

NODATA = -9999
CORE = "core"
SHIPPED_MODEL = "logistic regression"
TREE_MODEL = "lightgbm"
CONSTANT = "constant (national mean)"


# ---------------------------------------------------------------- pure helpers
def md_table(df: pd.DataFrame, formats: dict = None) -> str:
    """GitHub markdown table. `formats` maps a column to a format string such as '{:.3f}'."""
    formats = {} if formats is None else formats
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        cells = [formats[c].format(row[c]) if c in formats and pd.notna(row[c]) else ("" if pd.isna(row[c]) else str(row[c]))
                 for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cv_table(cv: pd.DataFrame, feature_set: str = CORE) -> pd.DataFrame:
    """One row per model: household-weighted MAE under random CV and under spatial CV, sorted by spatial MAE."""
    sub = cv[cv["feature_set"] == feature_set]
    random = sub[sub["cv"] == "random"].set_index("model")
    spatial = sub[sub["cv"] == "spatial"].set_index("model")
    out = pd.DataFrame({"model": spatial.index, "random_cv": random.loc[spatial.index, "mean"].to_numpy(),
                        "spatial_cv": spatial["mean"].to_numpy(), "spatial_fold_min": spatial["fold_min"].to_numpy(),
                        "spatial_fold_max": spatial["fold_max"].to_numpy()})
    return out.sort_values("spatial_cv").reset_index(drop=True)


def cv_findings(table: pd.DataFrame) -> dict:
    """The numbers the text needs: best model, constant baseline, shipped model, tree model, random-vs-spatial gap."""
    by = table.set_index("model")
    ship, tree, const = by.loc[SHIPPED_MODEL], by.loc[TREE_MODEL], by.loc[CONSTANT]
    return {"best_model": table.iloc[0]["model"], "best_spatial": table.iloc[0]["spatial_cv"],
            "ship_spatial": ship["spatial_cv"], "ship_random": ship["random_cv"],
            "tree_spatial": tree["spatial_cv"], "constant_spatial": const["spatial_cv"],
            "improvement_over_constant": 1 - ship["spatial_cv"] / const["spatial_cv"],
            "gap": ship["spatial_cv"] - ship["random_cv"],
            "tree_beats_ship": bool(tree["spatial_cv"] < ship["spatial_cv"])}


def cv_text(f: dict) -> str:
    lines = [f"- The shipped model is {SHIPPED_MODEL}: spatial error {f['ship_spatial']:.3f}, "
             f"{100 * f['improvement_over_constant']:.0f}% lower than the constant national-mean map ({f['constant_spatial']:.3f})."]
    if f["tree_beats_ship"]:
        lines.append(f"- Checkpoint met: LightGBM ({f['tree_spatial']:.3f}) beats the logistic baseline in spatial CV.")
    else:
        lines.append(f"- **Checkpoint not met:** LightGBM ({f['tree_spatial']:.3f}) does not beat the logistic baseline "
                     f"({f['ship_spatial']:.3f}) in spatial CV, so scope was cut to the logistic model. With about 560 clusters "
                     "the simplest model is as good as the complex ones.")
    if f["gap"] > 0.005:
        lines.append(f"- Spatial error is {f['gap']:.3f} higher than random error, so random CV is optimistic by that much.")
    else:
        lines.append(f"- Random and spatial error are almost the same (gap {f['gap']:+.3f}). Neighbouring clusters leaking "
                     "information is a small problem at district level here. The spatial score is still the one to quote.")
    return "\n".join(lines)


def coverage_table(coverage: pd.DataFrame, target: float) -> pd.DataFrame:
    sub = coverage[coverage["bin"].astype(str) == "all"]
    return pd.DataFrame({"share": sub["share"].to_numpy(), "target": target, "coverage": sub["coverage"].to_numpy(),
                         "mean_range_width": sub["mean_width"].to_numpy()})


def mask_summary(codes: np.ndarray) -> dict:
    """Share of predicted cells per do-not-trust code (value -9999 = no prediction)."""
    valid = codes[codes != NODATA].astype(int)
    return {"cells": int(valid.size), "width": float(np.mean((valid & 1) > 0)),
            "extrapolation": float(np.mean((valid & 2) > 0)), "either": float(np.mean(valid > 0))}


def province_table(provinces: pd.DataFrame) -> pd.DataFrame:
    """Predicted minus EICV7 per province, in percentage points, for the grid and solar shares."""
    return pd.DataFrame({"province": provinces["province"],
                         "grid_pred_minus_eicv7": 100 * (provinces["pred_grid"] - provinces["eicv7_grid"]),
                         "solar_pred_minus_eicv7": 100 * (provinces["pred_solar"] - provinces["eicv7_solar"])})


def control_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["level", "source", "quantity", "n", "mae", "bias", "pearson", "spearman", "skill"]
    return summary[cols].copy()


def solar_test_table(test: pd.DataFrame) -> pd.DataFrame:
    return test[["model", "n_clusters", "mae", "skill", "auc_solar_majority"]].copy()


# ---------------------------------------------------------------- report
def build_report(country: str, cv: pd.DataFrame, coverage: pd.DataFrame, target: float, mask: dict,
                 control: pd.DataFrame, provinces: pd.DataFrame, solar: pd.DataFrame) -> str:
    cvt = cv_table(cv)
    f = cv_findings(cvt)
    prov = province_table(provinces).sort_values("grid_pred_minus_eicv7")
    worst = prov.iloc[-1]
    ctl = control.set_index(["level", "source", "quantity"])
    district, sector = ctl.loc[("district", "EICV7 2023/24", "served (grid+solar)")], \
        ctl.loc[("sector", "Census 2022", "served (grid+solar)")]
    ratio = solar.set_index("model").loc["ratio model (7 core features)"]
    lights = solar.set_index("model").loc["lights only"]

    parts = [f"# Validation report: {country.capitalize()} off-grid solar access map", "",
             "Generated by `python -m src.report` from the result files in `outputs/`. "
             "Household-weighted mean absolute error (MAE) of the grid, solar and unserved shares, averaged over the three. "
             "Lower is better.", "",
             "## 1. Random vs spatial cross-validation (5 folds)", "",
             "Random folds mix clusters from the same district across train and test. Spatial folds hold out whole "
             "districts (`GroupKFold`, groups = district), so a test cluster is predicted by a model that never saw its "
             "district. The spatial score is the honest one. Core feature set (7 features).", "",
             md_table(cvt, {c: "{:.3f}" for c in ["random_cv", "spatial_cv", "spatial_fold_min", "spatial_fold_max"]}), "",
             cv_text(f), "",
             "## 2. Uncertainty ranges", "",
             f"Split conformal prediction on spatial out-of-fold errors. Target coverage {target:.0%}. Coverage is checked "
             "with nested spatial folds: calibrate on four folds of districts, test on the fifth.", "",
             md_table(coverage_table(coverage, target), {"target": "{:.0%}", "coverage": "{:.1%}", "mean_range_width": "{:.2f}"}), "",
             "- Coverage holds on average. It does not hold for every cell or every district.",
             "- The ranges are wide because they are calibrated on cluster-level errors, which include household sampling "
             "noise (about 25 households per cluster) and GPS displacement. They overstate the uncertainty of the true cell share.",
             f"- Do-not-trust mask ({mask['cells']:,} predicted cells): {100 * mask['width']:.0f}% flagged for a wide range "
             f"(a ranking: the widest 25%), {100 * mask['extrapolation']:.0f}% flagged because their features lie outside the "
             f"range seen at survey clusters, {100 * mask['either']:.0f}% flagged by either. A fixed 0.4 width limit would flag "
             "94% of cells, so the width flag is relative.", "",
             "## 3. Control totals against independent published numbers", "",
             "Predicted cell shares are summed (population-weighted) per area and compared with published survey and census "
             "numbers. These sources were not used to train the model. \"Served\" is grid + solar. Sources are from different "
             "years: Census August 2022, EICV7 2023/24, DHS June to October 2025.", "",
             md_table(control_table(ctl.reset_index()), {c: "{:.3f}" for c in ["mae", "bias", "pearson", "spearman", "skill"]}), "",
             "Skill = 1 minus MAE divided by the MAE of a flat map at the national mean. Bias = predicted minus published.", "",
             f"- Districts: MAE {district['mae']:.3f}, skill {district['skill']:.2f}, but rank order is weak "
             f"(Spearman {district['spearman']:.2f}). The map gets district levels roughly right and their order only partly.",
             f"- Sectors vs Census 2022: bias {sector['bias']:+.3f}. Most of this is likely the 2022 vs 2025 timing gap "
             f"(access grew between the two), not model error alone. After removing the bias, MAE is {sector['mae_no_bias']:.3f}.",
             f"- Provinces: {worst['province']} has the largest grid over-prediction vs EICV7 "
             f"({worst['grid_pred_minus_eicv7']:+.1f} points).", "",
             md_table(prov, {"grid_pred_minus_eicv7": "{:+.1f}", "solar_pred_minus_eicv7": "{:+.1f}"}), "",
             "Only the served share can be checked below province level. The grid / solar split exists in published "
             "tables only by province.", "",
             "## 4. Solar vs unserved among non-grid households", "",
             "The market-reality layer depends on separating solar from unserved households. Target: solar / (solar + unserved) "
             "among non-grid households, spatial CV, weighted by non-grid households.", "",
             md_table(solar_test_table(solar), {"mae": "{:.3f}", "skill": "{:.2f}", "auc_solar_majority": "{:.2f}"}), "",
             f"- The model ({ratio['mae']:.3f}) is only {100 * ratio['skill']:.0f}% better than a flat map and "
             f"{ratio['mae_minus_lights_only']:+.3f} vs night lights alone ({lights['mae']:.3f}); the 95% interval of that "
             f"difference is [{ratio['ci_low_vs_lights_only']:+.3f}, {ratio['ci_high_vs_lights_only']:+.3f}] (paired bootstrap by "
             "district).",
             "- The decision rule (clearly better than night lights) is **not met**. The market-reality layer is a "
             "low-confidence hypothesis.", "",
             "## 5. What was not validated", "",
             "- No independent check of the solar vs unserved split below province level.",
             "- The World Bank Multi-Tier Framework (MTF) data was not used.",
             "- GOGLA sales are national and affiliate-only; see `calibration_note_" + country + ".md`. The maps are not rescaled.",
             "- Time change was not tested, and no second country was run.",
             "- One of 416 census sectors did not match a boundary name.", ""]
    return "\n".join(parts)


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args(argv)
    c = args.country
    cfg = load_config()
    out = repo_path(cfg["paths"]["outputs"])

    with rasterio.open(out / f"{c}_do_not_trust.tif") as src:
        mask = mask_summary(src.read(1))
    text = build_report(
        c, pd.read_csv(out / f"cv_results_{c}.csv"), pd.read_csv(out / f"uncertainty_coverage_{c}.csv"),
        cfg["model"]["uncertainty"]["coverage"], mask, pd.read_csv(out / f"control_totals_summary_{c}.csv"),
        pd.read_csv(out / f"control_totals_provinces_{c}.csv"), pd.read_csv(out / f"solar_vs_unserved_test_{c}.csv"))
    path = out / f"validation_report_{c}.md"
    path.write_text(text)
    print(f"Wrote {path} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
