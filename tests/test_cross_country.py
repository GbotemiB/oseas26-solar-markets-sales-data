"""Tests for src/cross_country.py with small synthetic result tables.

Run from the repo root:  python -m pytest tests/test_cross_country.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.cross_country import bias_table, build_markdown, country_row, summary_lines


def cv_frame(logistic, lightgbm, constant, random_logistic):
    rows = [("random", "constant (national mean)", constant), ("random", "logistic regression", random_logistic),
            ("random", "lightgbm", lightgbm), ("spatial", "constant (national mean)", constant),
            ("spatial", "logistic regression", logistic), ("spatial", "lightgbm", lightgbm)]
    return pd.DataFrame([{"feature_set": "core", "cv": cv, "model": m, "grid": v, "solar": v, "unserved": v, "mean": v,
                          "fold_min": v - 0.01, "fold_max": v + 0.01} for cv, m, v in rows])


def tables(ci_high, auc_low, bias=0.0):
    coverage = pd.DataFrame({"share": ["grid", "solar", "unserved"], "bin": ["all"] * 3, "n_clusters": [100] * 3,
                             "coverage": [0.8, 0.79, 0.81], "mean_width": [0.5] * 3})
    solar = pd.DataFrame({"model": ["flat", "lights only", "ratio model (7 core features)"], "n_clusters": [90] * 3,
                          "mae": [0.25, 0.24, 0.19], "mae_minus_lights_only": [0.01, 0.0, -0.05],
                          "ci_low_vs_lights_only": [0.0, 0.0, -0.06], "ci_high_vs_lights_only": [0.02, 0.0, ci_high]})
    naive = pd.DataFrame({"metric": ["cell_spearman", "cell_naive_in_ours_cells", "cluster_auc_naive", "cluster_auc_lights",
                                     "cluster_auc_ours", "cluster_auc_diff"],
                          "value": [-0.2, 0.1, 0.5, 0.4, 0.7, 0.2], "ci_low": [np.nan] * 5 + [auc_low],
                          "ci_high": [np.nan] * 5 + [0.3]})
    labels = pd.DataFrame({"area": ["national"], "share_grid": [0.5], "share_solar": [0.2], "share_unserved": [0.3]})
    districts = pd.DataFrame({"share_grid": [0.5 + bias, 0.5 + bias], "share_solar": [0.2, 0.2], "share_unserved": [0.3 - bias, 0.3 - bias],
                              "population": [10.0, 30.0]})
    return coverage, solar, naive, labels, districts


def row(name, ci_high=-0.04, auc_low=0.1, bias=0.0, lightgbm=0.14):
    return country_row(name, cv_frame(0.16, lightgbm, 0.25, 0.158), *tables(ci_high, auc_low, bias))


def test_country_row_reads_the_headline_numbers():
    r = row("kenya")
    assert r["clusters"] == 100 and r["logistic_spatial_mae"] == pytest.approx(0.16)
    assert r["logistic_random_mae"] == pytest.approx(0.158) and r["lightgbm_beats_logistic"] is True
    assert r["improvement_over_constant"] == pytest.approx(0.36)
    assert r["decision_rule_met"] is True and r["auc_ours_minus_naive_low"] == pytest.approx(0.1)


def test_decision_rule_needs_the_whole_interval_below_zero():
    assert row("a", ci_high=0.001)["decision_rule_met"] is False
    assert row("a", ci_high=-0.001)["decision_rule_met"] is True


def test_bias_is_map_minus_survey_and_uses_population_weights():
    r = row("x", bias=0.08)
    assert r["bias_grid"] == pytest.approx(0.08) and r["bias_unserved"] == pytest.approx(-0.08)
    districts = tables(-0.04, 0.1)[4].assign(share_grid=[0.0, 1.0])
    r2 = country_row("y", cv_frame(0.16, 0.14, 0.25, 0.158), *tables(-0.04, 0.1)[:4], districts)
    assert r2["map_grid"] == pytest.approx(0.75)         # 30 of 40 population weight on the second district


def test_summary_lines_name_the_right_countries():
    rows = pd.DataFrame([row("rwanda", ci_high=0.001), row("kenya", bias=0.08)])
    text = "\n".join(summary_lines(rows))
    assert "met) in: kenya; not met in: rwanda" in text
    assert "more than 5 points in: kenya" in text
    assert "2 of 2 countries (rwanda, kenya)" in text


def test_summary_lines_say_none_when_nothing_qualifies():
    rows = pd.DataFrame([row("a", ci_high=0.01, auc_low=-0.1)])
    text = "\n".join(summary_lines(rows))
    assert "in 0 of 1 countries (none)" in text and "met) in: none" in text


def test_markdown_has_both_tables_and_no_nan():
    rows = pd.DataFrame([row("rwanda"), row("kenya", bias=0.08)])
    text = build_markdown(rows)
    assert "## Scores" in text and "## National shares" in text and "nan" not in text.lower()
    assert list(bias_table(rows)["country"]) == ["rwanda", "kenya"]
