"""Tests for src/calibrate.py. Synthetic numbers only.

Run from the repo root:  python -m pytest tests/test_calibrate.py -v
"""
import pandas as pd
import pytest

from src.calibrate import build_note, implied_households, position, scenario_table, survival

SALES = {2021: 100, 2022: 100, 2023: 100, 2024: 100, 2025: 100}      # thousand units


def test_survival_is_one_while_young_zero_when_old_and_partial_in_between():
    assert survival(0, 3) == 1 and survival(2, 3) == 1
    assert survival(2.5, 3) == pytest.approx(0.5)
    assert survival(3, 3) == 0 and survival(10, 3) == 0


def test_implied_households_counts_only_units_within_lifetime():
    assert implied_households(SALES, 2025, 1.0, 1.0) == 100_000                  # this year only
    assert implied_households(SALES, 2025, 2.0, 1.0) == 200_000                  # 2025 and 2024
    assert implied_households(SALES, 2025, 5.0, 1.0) == 500_000                  # all five years
    assert implied_households(SALES, 2025, 2.5, 1.0) == pytest.approx(250_000)    # 2023 counts half


def test_units_per_household_divides_and_later_years_are_ignored():
    assert implied_households(SALES, 2025, 5.0, 2.0) == 250_000
    assert implied_households({**SALES, 2026: 999}, 2025, 5.0, 1.0) == 500_000


def test_scenarios_are_ordered_low_central_high_and_share_uses_total():
    sc = {"low": {"lifetime_years": 2.0, "units_per_household": 2.0},
          "central": {"lifetime_years": 3.0, "units_per_household": 1.5},
          "high": {"lifetime_years": 5.0, "units_per_household": 1.0}}
    t = scenario_table(SALES, 2025, sc, 1_000_000)
    assert t["scenario"].tolist() == ["low", "central", "high"]
    assert t["households"].tolist() == [100_000, 200_000, 500_000]
    assert t["share_of_households"].tolist() == pytest.approx([0.1, 0.2, 0.5])
    assert t["households"].is_monotonic_increasing


def test_position():
    assert position(5, 10, 20) == "below the range" and position(25, 10, 20) == "above the range"
    assert position(10, 10, 20) == "inside the range"


def test_note_mentions_assumptions_decision_and_the_numbers():
    cfg = {"calibration": {"reference_year": 2025, "households_total": 1_000_000, "gogla_sales_thousands": SALES}}
    sc = pd.DataFrame({"scenario": ["low", "central", "high"], "lifetime_years": [2, 3, 5],
                       "units_per_household": [2, 1.5, 1], "households": [100_000.0, 200_000.0, 500_000.0],
                       "share_of_households": [0.1, 0.2, 0.5]})
    survey = pd.DataFrame({"share": [0.25], "households": [250_000.0]}, index=pd.Index(["Our map (trained on DHS 2025)"], name="source"))
    note = build_note("testland", sc, survey, cfg)
    assert "inside the range" in note and "250,000" in note and "not rescaled" in note
    assert "our assumptions" in note.lower() and "Testland" in note
