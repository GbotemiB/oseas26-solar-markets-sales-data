"""Tests for src/naive_baseline.py and its report section. Synthetic data only, no DHS files needed.

Run from the repo root:  python -m pytest tests/test_naive_baseline.py -v
"""
import shutil

import numpy as np
import pandas as pd
import pytest

import src.config as config
from src import demo, naive_baseline
from src.config import ROOT
from src.naive_baseline import (auc_top_group, cell_comparison, conclusion, darkness, district_bootstrap_metric_diff,
                                naive_index, overlap, spearman, top_flags, weighted_mean)
from src.report import naive_text


# ---------------------------------------------------------------- index
def test_darkness_runs_from_dark_to_bright_and_is_unit_free():
    lights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    d = darkness(lights, lights)
    assert (np.diff(d) < 0).all() and d.max() < 1 and d.min() > 0
    assert darkness(lights * 100, lights * 100) == pytest.approx(d)       # only the rank matters


def test_darkness_gives_tied_lights_the_same_value_and_keeps_nan():
    d = darkness(np.array([0.0, 0.0, 0.0, 5.0, np.nan]), np.array([0.0, 0.0, 0.0, 5.0]))
    assert d[0] == d[1] == d[2] and d[0] > d[3]
    assert np.isnan(d[4])


def test_darkness_with_empty_reference_is_all_nan():
    assert np.isnan(darkness(np.array([1.0, 2.0]), np.array([np.nan]))).all()


def test_naive_index_is_high_where_dense_and_dark_and_empty_where_unpopulated_or_unlit_data_missing():
    pop = np.array([1000.0, 1000.0, 10.0, 0.0, np.nan, 500.0])
    lights = np.array([0.1, 9.0, 0.1, 0.1, 0.1, np.nan])
    idx = naive_index(pop, lights, min_population=1)
    assert idx[0] > idx[1] and idx[0] > idx[2]            # dense and dark beats dense and bright, and sparse and dark
    assert np.isnan(idx[3]) and np.isnan(idx[4]) and np.isnan(idx[5])


def test_naive_index_percentile_uses_populated_cells_only():
    pop = np.array([100.0, 100.0, 0.0])
    lights = np.array([1.0, 2.0, 100.0])                  # the unpopulated bright cell must not push the others darker
    a = naive_index(pop, lights, 1)
    b = naive_index(pop[:2], lights[:2], 1)
    assert a[:2] == pytest.approx(b)


# ---------------------------------------------------------------- sets and overlap
def test_top_flags_takes_the_top_fraction_and_ignores_nan():
    v = np.r_[np.arange(20.0), np.nan]
    f = top_flags(v, 0.1)
    assert f.sum() == 2 and f[18] and f[19] and not f[20]


def test_top_flags_includes_all_ties_at_the_cut_and_handles_empty():
    assert top_flags(np.array([1.0, 1.0, 1.0, 1.0]), 0.25).all()
    assert not top_flags(np.array([np.nan, np.nan]), 0.1).any()
    assert top_flags(np.array([]), 0.1).size == 0


def test_overlap_counts_cells_and_weights():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    o = overlap(a, b)
    assert o["jaccard"] == pytest.approx(1 / 3) and o["a_in_b"] == pytest.approx(0.5) and o["b_in_a"] == pytest.approx(0.5)
    w = overlap(a, b, np.array([8.0, 1.0, 1.0, 5.0]))
    assert w["jaccard"] == pytest.approx(8 / 10) and w["a_in_b"] == pytest.approx(8 / 9)


def test_overlap_of_empty_sets_is_nan_not_an_error():
    none = np.zeros(3, dtype=bool)
    o = overlap(none, none)
    assert np.isnan(o["jaccard"]) and np.isnan(o["a_in_b"])


# ---------------------------------------------------------------- rank metrics
def test_spearman_perfect_reversed_and_ties():
    x = np.arange(10.0)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(np.array([1.0, 1.0, 2.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 3.0, 3.0])) == pytest.approx(
        pd.Series([1.0, 1.0, 2.0, 2.0, 3.0]).corr(pd.Series([1.0, 2.0, 2.0, 3.0, 3.0]), method="spearman"))


def test_spearman_skips_nan_pairs_and_returns_nan_when_undefined():
    assert spearman(np.array([1.0, 2.0, 3.0, np.nan]), np.array([1.0, 2.0, 3.0, 9.0])) == pytest.approx(1.0)
    assert np.isnan(spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))      # constant side
    assert np.isnan(spearman(np.array([1.0, np.nan]), np.array([np.nan, 2.0])))          # no overlap


def test_weighted_spearman_with_equal_weights_matches_the_plain_one():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    assert spearman(a, b, np.full(50, 7.0)) == pytest.approx(spearman(a, b))


def test_auc_top_group_separates_a_good_score_and_flips_a_bad_one():
    target = np.arange(30.0)
    assert auc_top_group(target, target, 1 / 3) == pytest.approx(1.0)
    assert auc_top_group(-target, target, 1 / 3) == pytest.approx(0.0)
    assert auc_top_group(np.zeros(30), target, 1 / 3) == pytest.approx(0.5)


def test_auc_top_group_is_nan_when_one_group_is_empty_and_skips_nan():
    assert np.isnan(auc_top_group(np.arange(5.0), np.full(5, 0.5), 0.1))              # all tied: everyone is in the top group
    assert np.isnan(auc_top_group(np.array([np.nan, np.nan]), np.array([1.0, 2.0]), 0.5))
    score, target = np.array([1.0, 2.0, 3.0, np.nan]), np.array([1.0, 2.0, 3.0, 4.0])
    assert auc_top_group(score, target, 1 / 3) == pytest.approx(1.0)


def test_auc_uses_weights():
    target = np.array([0.0, 1.0, 0.0, 1.0])
    score = np.array([0.1, 0.9, 0.8, 0.2])               # three of four positive-negative pairs ranked right
    assert auc_top_group(score, target, 0.5, np.ones(4)) == pytest.approx(0.75)
    assert auc_top_group(score, target, 0.5, np.array([1.0, 1.0, 5.0, 1.0])) < 0.75   # the bad negative now counts more


def test_bootstrap_difference_is_positive_for_a_better_score_and_reproducible():
    rng = np.random.default_rng(1)
    n = 300
    target = rng.uniform(size=n)
    good, bad = target + rng.normal(0, 0.1, n), rng.normal(size=n)
    groups = np.repeat(np.arange(30), 10)
    metric = lambda s, t, w: spearman(s, t, w)
    d, lo, hi = district_bootstrap_metric_diff(good, bad, target, np.ones(n), groups, metric, 200, 3)
    assert lo > 0 and lo <= d <= hi
    assert (d, lo, hi) == district_bootstrap_metric_diff(good, bad, target, np.ones(n), groups, metric, 200, 3)
    d0, lo0, hi0 = district_bootstrap_metric_diff(bad, bad.copy(), target, np.ones(n), groups, metric, 200, 3)
    assert d0 == pytest.approx(0.0) and lo0 <= 0 <= hi0


def test_weighted_mean_ignores_nan_and_is_nan_when_empty():
    assert weighted_mean(np.array([1.0, np.nan, 3.0]), np.array([1.0, 5.0, 3.0])) == pytest.approx(2.5)
    assert np.isnan(weighted_mean(np.array([np.nan]), np.array([1.0])))


# ---------------------------------------------------------------- cell comparison
def cells(n=40, seed=0):
    rng = np.random.default_rng(seed)
    naive = rng.uniform(size=n)
    return {"naive": naive, "reality": naive.copy(), "unserved_hh": rng.uniform(1, 5, n), "flagged": np.zeros(n),
            "solar": np.full(n, 0.2), "population": rng.uniform(1, 5, n)}


def test_cell_comparison_identical_layers_overlap_fully():
    c = cells()
    r = cell_comparison(**c, fraction=0.1)
    assert r["spearman"] == pytest.approx(1.0) and r["jaccard_cells"] == pytest.approx(1.0)
    assert r["naive_in_ours_households"] == pytest.approx(1.0) and r["n_shared"] == 40
    assert r["naive_top_flagged"] == 0 and r["naive_top_solar"] == pytest.approx(0.2) and r["national_solar"] == pytest.approx(0.2)


def test_cell_comparison_reversed_layers_do_not_overlap_and_missing_cells_are_handled():
    c = cells()
    c["reality"] = -c["naive"]
    c["reality"][:5] = np.nan                              # cells with no market-reality value
    c["naive"][5] = np.nan
    r = cell_comparison(**c, fraction=0.1)
    assert r["spearman"] == pytest.approx(-1.0) and r["jaccard_cells"] == pytest.approx(0.0)
    assert r["n_shared"] == 34
    assert 0 <= r["naive_top_no_reality"] <= 1


def test_cell_comparison_with_no_shared_cells_gives_nan_not_an_error():
    c = cells(10)
    c["reality"][:] = np.nan
    r = cell_comparison(**c, fraction=0.1)
    assert r["n_shared"] == 0 and np.isnan(r["spearman"]) and np.isnan(r["jaccard_cells"])
    assert r["naive_top_no_reality"] == 1.0


def test_flag_share_reads_the_mask():
    c = cells()
    c["flagged"] = np.where(c["naive"] > np.quantile(c["naive"], 0.5), 1.0, 0.0)     # every high cell flagged
    r = cell_comparison(**c, fraction=0.1)
    assert r["naive_top_flagged"] == pytest.approx(1.0) and r["all_flagged"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------- wording
def row(rho=0.0, share=0.1, d=0.0, lo=-0.1, hi=0.1):
    return {"cell_spearman": rho, "cell_naive_in_ours_cells": share, "cluster_auc_diff": d,
            "cluster_auc_diff_low": lo, "cluster_auc_diff_high": hi}


def test_conclusion_says_so_when_the_naive_index_is_equal_or_better():
    assert "cannot tell" in conclusion(row())
    assert "naive index ranks the survey outcome better" in conclusion(row(d=-0.2, lo=-0.3, hi=-0.1))
    assert "our model ranks the survey outcome better" in conclusion(row(d=0.2, lo=0.1, hi=0.3))
    assert "close to each other" in conclusion(row(rho=0.8, share=0.7))
    assert "opposite" in conclusion(row(rho=-0.8))
    assert "could not be run" in conclusion(row(d=np.nan, lo=np.nan, hi=np.nan))


def test_report_section_quotes_the_numbers():
    metrics = {"cell_spearman": -0.18, "cell_naive_in_ours_cells": 0.13, "cell_naive_in_ours_households": 0.18,
               "cell_chance_share": 0.1, "cell_naive_top_flagged": 0.53, "cell_naive_top_solar": 0.23,
               "cell_national_solar": 0.20, "cluster_auc_naive": 0.48, "cluster_auc_lights": 0.39, "cluster_auc_ours": 0.70,
               "cluster_auc_diff": 0.22}
    df = pd.DataFrame({"metric": list(metrics), "value": list(metrics.values()), "ci_low": np.nan, "ci_high": np.nan, "n": 389})
    df.loc[df["metric"] == "cluster_auc_diff", ["ci_low", "ci_high"]] = [0.10, 0.34]
    text = "\n".join(naive_text(df))
    assert "-0.18" in text and "13%" in text and "0.48" in text and "+0.22" in text and "+0.10 to +0.34" in text


# ---------------------------------------------------------------- demo run
def test_demo_run_writes_aggregate_outputs_only(tmp_path, monkeypatch):
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    demo.main(["--regenerate"])
    naive_baseline.main(["--demo"])
    out = tmp_path / "outputs" / "demo"
    result = pd.read_csv(out / "naive_baseline_demo.csv")
    assert list(result.columns) == ["metric", "value", "ci_low", "ci_high", "n", "note"]
    for metric in ["cell_spearman", "cell_jaccard_cells", "cell_naive_top_solar", "cluster_auc_naive", "cluster_auc_ours",
                   "cluster_auc_diff"]:
        assert metric in set(result["metric"])
    assert result["metric"].is_unique and len(result) < 100
    text = (out / "naive_baseline_demo.md").read_text()
    assert "SYNTHETIC" in text and len(text.split()) < 300
    assert chr(8212) not in text
    assert not (tmp_path / "data" / "restricted").exists()
