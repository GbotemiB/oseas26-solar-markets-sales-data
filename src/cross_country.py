"""Compare the independent country runs side by side (aggregates only).

Run (after the per-country pipeline, at least through src.naive_baseline):
    python -m src.cross_country                       # every country that has its result files
    python -m src.cross_country --countries rwanda kenya

Reads the aggregate files in outputs/ (cv_results, uncertainty_coverage, solar_vs_unserved_test,
naive_baseline, label_summary, predicted_district_shares) and writes outputs/cross_country_summary.csv and .md.
Every number comes from those files, so a rerun keeps the comparison in step with the country runs.
Each country is trained and tested on its own survey only: this is replication of the method, not transfer
of a model from one country to another.
"""
import argparse

import numpy as np
import pandas as pd

from src.config import load_config, repo_path
from src.report import cv_findings, cv_table, md_table

NEEDED = ["cv_results_{c}.csv", "uncertainty_coverage_{c}.csv", "solar_vs_unserved_test_{c}.csv",
          "naive_baseline_{c}.csv", "label_summary_{c}.csv", "predicted_district_shares_{c}.csv"]
SHARES = ["grid", "solar", "unserved"]


# ---------------------------------------------------------------- pure helpers
def country_row(name: str, cv: pd.DataFrame, coverage: pd.DataFrame, solar: pd.DataFrame, naive: pd.DataFrame,
                labels: pd.DataFrame, districts: pd.DataFrame) -> dict:
    """One row of the comparison from one country's result tables."""
    f = cv_findings(cv_table(cv))
    cov = coverage[coverage["bin"].astype(str) == "all"].set_index("share")
    ratio = solar.set_index("model").loc["ratio model (7 core features)"]
    lights = solar.set_index("model").loc["lights only"]
    nv = naive.set_index("metric")
    dhs = labels.set_index("area").loc["national"]
    weights = districts["population"].to_numpy()
    row = {"country": name,
           "clusters": int(cov.loc["grid", "n_clusters"]),
           "constant_spatial_mae": f["constant_spatial"], "logistic_spatial_mae": f["ship_spatial"],
           "logistic_random_mae": f["ship_spatial"] - f["gap"], "lightgbm_spatial_mae": f["tree_spatial"],
           "improvement_over_constant": f["improvement_over_constant"], "lightgbm_beats_logistic": bool(f["tree_beats_ship"]),
           **{f"coverage_{s}": float(cov.loc[s, "coverage"]) for s in SHARES},
           "solar_test_clusters": int(ratio["n_clusters"]), "solar_ratio_mae": float(ratio["mae"]),
           "solar_flat_mae": float(solar.set_index("model").loc["flat", "mae"]),
           "solar_lights_mae": float(lights["mae"]), "solar_minus_lights": float(ratio["mae_minus_lights_only"]),
           "solar_minus_lights_ci_low": float(ratio["ci_low_vs_lights_only"]),
           "solar_minus_lights_ci_high": float(ratio["ci_high_vs_lights_only"]),
           "decision_rule_met": bool(ratio["ci_high_vs_lights_only"] < 0),
           "cell_spearman_naive_vs_ours": float(nv.loc["cell_spearman", "value"]),
           "naive_top_decile_in_ours": float(nv.loc["cell_naive_in_ours_cells", "value"]),
           "cluster_auc_naive": float(nv.loc["cluster_auc_naive", "value"]),
           "cluster_auc_lights": float(nv.loc["cluster_auc_lights", "value"]),
           "cluster_auc_ours": float(nv.loc["cluster_auc_ours", "value"]),
           "auc_ours_minus_naive": float(nv.loc["cluster_auc_diff", "value"]),
           "auc_ours_minus_naive_low": float(nv.loc["cluster_auc_diff", "ci_low"]),
           "auc_ours_minus_naive_high": float(nv.loc["cluster_auc_diff", "ci_high"])}
    for s in SHARES:
        ok = districts[f"share_{s}"].notna().to_numpy()
        predicted = float(np.average(districts.loc[ok, f"share_{s}"], weights=weights[ok]))
        row[f"dhs_{s}"], row[f"map_{s}"], row[f"bias_{s}"] = float(dhs[f"share_{s}"]), predicted, predicted - float(dhs[f"share_{s}"])
    return row


def summary_lines(rows: pd.DataFrame) -> list:
    """Plain statements chosen from the numbers (no claim is made that the numbers do not support)."""
    met = rows.loc[rows["decision_rule_met"], "country"].tolist()
    not_met = rows.loc[~rows["decision_rule_met"], "country"].tolist()
    naive_better = rows.loc[rows["auc_ours_minus_naive_low"] > 0, "country"].tolist()
    biased = rows.loc[(rows[[f"bias_{s}" for s in SHARES]].abs() > 0.05).any(axis=1), "country"].tolist()
    beats_flat = rows.loc[rows["improvement_over_constant"] > 0, "country"].tolist()
    lines = [f"- Spatial CV error is below the flat national-mean map in {len(beats_flat)} of {len(rows)} countries "
             f"({', '.join(beats_flat) or 'none'}).",
             f"- Our model ranks the survey unserved / (solar + unserved) better than the naive \"dense and dark\" index, "
             f"with a 95% interval above zero, in {len(naive_better)} of {len(rows)} countries ({', '.join(naive_better) or 'none'})."]
    lines.append("- Solar vs unserved is clearly better than night lights alone (decision rule met) in: "
                 f"{', '.join(met) or 'none'}; not met in: {', '.join(not_met) or 'none'}.")
    if biased:
        lines.append(f"- National shares from the map differ from the survey by more than 5 points in: {', '.join(biased)}. "
                     "Quote levels for those countries with caution.")
    return lines


def comparison_table(rows: pd.DataFrame) -> pd.DataFrame:
    """The headline columns, one row per country, for the markdown table."""
    return pd.DataFrame({
        "country": rows["country"], "clusters": rows["clusters"],
        "spatial MAE (logistic)": rows["logistic_spatial_mae"], "flat map": rows["constant_spatial_mae"],
        "LightGBM": rows["lightgbm_spatial_mae"], "coverage (grid)": rows["coverage_grid"],
        "solar test: model minus lights": rows["solar_minus_lights"], "rule met": rows["decision_rule_met"].map({True: "yes", False: "no"}),
        "AUC ours": rows["cluster_auc_ours"], "AUC naive": rows["cluster_auc_naive"], "AUC lights": rows["cluster_auc_lights"]})


def bias_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Map national shares against the survey, in percentage points."""
    out = {"country": rows["country"]}
    for s in SHARES:
        out[f"{s} map %"], out[f"{s} survey %"] = 100 * rows[f"map_{s}"], 100 * rows[f"dhs_{s}"]
        out[f"{s} bias (pts)"] = 100 * rows[f"bias_{s}"]
    return pd.DataFrame(out)


def build_markdown(rows: pd.DataFrame) -> str:
    fmt3 = {c: "{:.3f}" for c in ["spatial MAE (logistic)", "flat map", "LightGBM", "solar test: model minus lights", "AUC ours", "AUC naive", "AUC lights"]}
    bias_fmt = {c: "{:.1f}" if "%" in c else "{:+.1f}" for c in bias_table(rows).columns if c != "country"}
    return "\n".join([
        "# Cross-country comparison", "",
        "Independent runs of the same pipeline, each trained and tested on its own country's DHS survey. This is a check that "
        "the method replicates, not a transfer of one model to another country. Generated by `python -m src.cross_country` "
        "from the aggregate result files in `outputs/`.", "",
        "## Headline", "", *summary_lines(rows), "",
        "## Scores (household-weighted MAE, spatial CV, held-out districts or constituencies)", "",
        md_table(comparison_table(rows), {**fmt3, "coverage (grid)": "{:.1%}"}), "",
        "Spatial-CV groups are the ADM2 units of each country (Rwanda districts, Kenya constituencies, Nigeria local government "
        "areas), so groups are smaller where the survey is sparser and the spatial score is then less strict. The solar test "
        "column is MAE of the ratio model minus MAE of night lights alone (negative = better); the rule is met when its 95% "
        "paired-bootstrap interval is below zero. AUC is for ranking clusters in the top third of unserved / (solar + unserved).", "",
        "## National shares: map against survey", "",
        md_table(bias_table(rows), bias_fmt), "",
        "Map shares are population-weighted over predicted cells. Survey shares are household-weighted DHS. A large gap means "
        "the 1 km map does not reproduce the survey level even where the cluster-level model is unbiased. In Kenya and Nigeria "
        "this was traced to aggregation, not to the model: the map weights cells by people while the survey weights households "
        "(urban households are smaller), and WorldPop places fewer people in urban cells than the survey sample does. Weighting "
        "by households would close about half of Kenya's gap and most of Nigeria's. This was diagnosed but not applied.", ""])


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--countries", nargs="*", help="default: every country in config.yaml with all result files")
    args = parser.parse_args(argv)
    cfg = load_config()
    out = repo_path(cfg["paths"]["outputs"])
    names = args.countries if args.countries else list(cfg["countries"])
    rows = []
    for c in names:
        missing = [n.format(c=c) for n in NEEDED if not (out / n.format(c=c)).exists()]
        if missing:
            print(f"  {c}: skipped, missing {missing}")
            continue
        rows.append(country_row(c, *[pd.read_csv(out / n.format(c=c)) for n in NEEDED]))
    if not rows:
        raise SystemExit("No country has all result files yet.")
    table = pd.DataFrame(rows)
    table.to_csv(out / "cross_country_summary.csv", index=False)
    text = build_markdown(table)
    (out / "cross_country_summary.md").write_text(text)
    print(f"Wrote cross_country_summary.csv and .md for {', '.join(table['country'])} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
