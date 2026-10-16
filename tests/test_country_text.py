"""Tests for src/country_text.py and the country-neutral parts of the labels and report code.

Run from the repo root:  python -m pytest tests/test_country_text.py -v
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.country_text import household_total, labels_tag, limitations_text, read_household_size
from src.labels_dhs import household_sizes, mean_household_size
from src.export_eae import saturation_note
from src.report import control_section, cv_text, naive_text, solar_verdict

RWANDA_ONLY = ["Rwanda", "EICV", "560", "26%", "12%", "NISR", "GOGLA"]


@pytest.fixture(scope="module")
def countries():
    return load_config()["countries"]


def test_rwanda_labels_tag_is_unchanged(countries):
    assert labels_tag(countries["rwanda"]) == "Rwanda DHS 2025 (DHS-8), restricted data. Only model-derived outputs are published."


def test_rwanda_limitations_keep_the_old_wording(countries):
    text = limitations_text(countries["rwanda"], 560)
    assert text.startswith("Modelled estimate from 560 survey clusters, not a census.")
    assert text.endswith("Unserved here is mostly torch users (about 26% nationally), not the official access gap (about 12%).")


@pytest.mark.parametrize("name", ["kenya", "nigeria"])
def test_other_countries_text_has_no_rwanda_only_claims(countries, name):
    ccfg = countries[name]
    texts = [labels_tag(ccfg), limitations_text(ccfg, 1234)]
    for text in texts:
        for word in RWANDA_ONLY:
            assert word not in text, (name, word)
    assert "1234 survey clusters" in texts[1]
    assert ccfg["display_name"] in texts[0] and ccfg["survey"] in texts[0]


def test_nigeria_text_flags_the_unreliable_grid_label(countries):
    assert "counted as grid" in limitations_text(countries["nigeria"], 100)
    assert "counted as grid" not in limitations_text(countries["kenya"], 100)


def test_only_rwanda_has_external_checks(countries):
    assert [k for k, v in countries.items() if v["external_checks"]] == ["rwanda"]


def test_mean_household_size_is_weighted_and_skips_missing():
    hh = pd.DataFrame({"weight": [1.0, 3.0, 5.0], "hv009": [2, 6, np.nan]})
    assert mean_household_size(hh) == pytest.approx((1 * 2 + 3 * 6) / 4)


def test_cv_text_uses_the_given_cluster_count():
    f = {"ship_spatial": 0.2, "improvement_over_constant": 0.1, "constant_spatial": 0.22, "tree_beats_ship": False,
         "tree_spatial": 0.21, "gap": 0.0}
    assert "about 560 clusters" in cv_text(f, 560)
    assert "560" not in cv_text(f, None) and "few hundred" in cv_text(f, None)


def test_control_section_without_external_totals_says_so_and_does_not_crash():
    lines = control_section(None, None, "Kenya")
    text = "\n".join(lines)
    assert "No external control totals are available for Kenya" in text
    assert "EICV7 2023/24, DHS" not in text



def _ratio(ci_high):
    return pd.Series({"mae": 0.19, "skill": 0.22, "mae_minus_lights_only": -0.05, "ci_low_vs_lights_only": -0.06,
                      "ci_high_vs_lights_only": ci_high})


def test_solar_verdict_says_met_only_when_the_whole_interval_is_below_zero():
    lights = pd.Series({"mae": 0.24})
    met = " ".join(solar_verdict(_ratio(-0.04), lights))
    assert "**met**" in met and "not met" not in met and "only 22%" not in met
    not_met = " ".join(solar_verdict(_ratio(0.001), lights))
    assert "**not met**" in not_met and "only 22% better" in not_met


def test_saturation_note_only_when_the_layer_is_squeezed_near_one():
    assert saturation_note(0.71) == "" and saturation_note(0.89) == ""
    assert "median 0.96" in saturation_note(0.96) and "ranking, not the level" in saturation_note(0.96)


def test_naive_text_names_the_countrys_own_file():
    naive = pd.DataFrame({"metric": ["cell_spearman", "cell_naive_in_ours_cells", "cell_naive_in_ours_households", "cell_chance_share",
                                     "cell_naive_top_flagged", "cell_naive_top_solar", "cell_national_solar", "cluster_auc_ours",
                                     "cluster_auc_naive", "cluster_auc_lights", "cluster_auc_diff"],
                          "value": [-0.2, 0.1, 0.2, 0.1, 0.5, 0.2, 0.2, 0.7, 0.5, 0.4, 0.2], "ci_low": [None] * 10 + [0.1],
                          "ci_high": [None] * 10 + [0.3], "n": [100] * 11})
    text = " ".join(naive_text(naive, "nigeria"))
    assert "naive_baseline_nigeria.md" in text and "<country>" not in text


def test_household_sizes_split_urban_and_rural():
    hh = pd.DataFrame({"weight": [1.0, 1.0, 2.0, 2.0], "hv009": [2, 4, 5, 7], "hv025": [1, 1, 2, 2]})
    sizes = household_sizes(hh)
    assert sizes["urban_household_size"] == pytest.approx(3.0) and sizes["rural_household_size"] == pytest.approx(6.0)
    assert sizes["mean_household_size"] == pytest.approx((2 + 4 + 10 + 14) / 6)


def test_frame_note_is_kenya_only_and_reaches_the_limitations(countries):
    assert "north-east" in limitations_text(countries["kenya"], 10)
    assert "north-east" not in limitations_text(countries["rwanda"], 10)
    assert limitations_text(countries["rwanda"], 560).endswith("not the official access gap (about 12%).")


def test_household_total_uses_the_sourced_count_for_rwanda(countries):
    total, method = household_total(countries["rwanda"], {"households_total": 3289810}, 13_000_000.0, 4.1)
    assert total == 3289810 and "EICV7" in method


def test_household_total_for_other_countries_divides_by_household_size(countries):
    total, method = household_total(countries["kenya"], {"households_total": 3289810}, 40_000_000.0, 4.0)
    assert total == pytest.approx(10_000_000) and "4.00 members" in method


def test_household_total_is_skipped_without_a_household_size(countries):
    assert household_total(countries["nigeria"], {"households_total": 1}, 1e6, None) == (None, None)


def test_read_household_size(tmp_path):
    assert read_household_size(tmp_path / "missing.json") is None
    path = tmp_path / "kenya_household_size.json"
    path.write_text(json.dumps({"mean_household_size": 3.84, "n_households": 10}))
    assert read_household_size(path) == pytest.approx(3.84)
