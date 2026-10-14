"""Calibration note: is the survey-based national solar total consistent with GOGLA sales for Rwanda?

Run (after src.predict and src.labels_open):
    python -m src.calibrate --country rwanda

Sales (GOGLA, solar energy kits per year) are converted to households in use. A unit sold in year y counts
fully while it is younger than the product lifetime, and partly in the last year:
    survival = clip(lifetime - (reference_year - y), 0, 1)
    households = sum(units_y x survival_y) / units_per_household
Three scenarios (low, central, high) show how much the answer depends on the assumptions. The lifetimes and
units per household are OUR assumptions, set in config.yaml. The maps are not rescaled: the gap is reported.

Outputs: outputs/calibration_note_<country>.csv, outputs/calibration_note_<country>.md
"""
import argparse

import numpy as np
import pandas as pd

from src.config import load_config, repo_path


# ---------------------------------------------------------------- pure helpers
def survival(age_years: float, lifetime_years: float) -> float:
    """1 while the unit is younger than its lifetime, falling to 0 over the last year of life."""
    return float(np.clip(lifetime_years - age_years, 0, 1))


def implied_households(sales_thousands: dict, reference_year: int, lifetime_years: float, units_per_household: float) -> float:
    """Households with a working GOGLA unit in the reference year (units sold after the reference year are ignored)."""
    units = sum(thousands * 1000 * survival(reference_year - year, lifetime_years)
                for year, thousands in sales_thousands.items() if year <= reference_year)
    return units / units_per_household


def scenario_table(sales_thousands: dict, reference_year: int, scenarios: dict, households_total: float) -> pd.DataFrame:
    rows = []
    for name in ["low", "central", "high"]:
        sc = scenarios[name]
        n = implied_households(sales_thousands, reference_year, sc["lifetime_years"], sc["units_per_household"])
        rows.append({"scenario": name, "lifetime_years": sc["lifetime_years"],
                     "units_per_household": sc["units_per_household"], "households": n,
                     "share_of_households": n / households_total})
    return pd.DataFrame(rows)


def position(value: float, low: float, high: float) -> str:
    return "below the range" if value < low else "above the range" if value > high else "inside the range"


def build_note(country: str, scenarios: pd.DataFrame, survey: pd.DataFrame, cfg: dict) -> str:
    """The one-page note in Markdown, with the numbers filled in."""
    cal = cfg["calibration"]
    low, central, high = (scenarios.set_index("scenario").loc[k, "households"] for k in ["low", "central", "high"])
    ours = survey.loc["Our map (trained on DHS 2025)", "households"]
    sales = ", ".join(f"{y}: {v}k" for y, v in cal["gogla_sales_thousands"].items())
    sc_rows = "\n".join(f"| {r.scenario} | {r.lifetime_years} | {r.units_per_household} | {r.households:,.0f} | {100 * r.share_of_households:.1f}% |"
                        for r in scenarios.itertuples())
    sv_rows = "\n".join(f"| {r.Index} | {100 * r.share:.1f}% | {r.households:,.0f} |" for r in survey.itertuples())
    return f"""# Calibration note: survey-based solar total vs GOGLA sales ({country.capitalize()})

**Question.** Is the national number of households using off-grid solar in our map consistent with what GOGLA
(Global Off-Grid Lighting Association) sales imply?

**Sales used.** GOGLA Global Off-Grid Solar Market Report 2025, Figure 19, solar energy kits sold in Rwanda
({sales}). Lanterns and solar home systems together. GOGLA affiliate companies only, so this is a lower bound
on the market. No Rwanda split by product type is given.

**Conversion.** A unit counts as a working unit while younger than its lifetime. Households = units in use
divided by units per household. The lifetimes and units per household are **our assumptions** (lanterns about
2 to 3 years, home systems about 5), because the report gives none. Reference year {cal['reference_year']}.

| Scenario | Lifetime (years) | Units per household | GOGLA-implied households | Share of {cal['households_total']:,} households |
|---|---|---|---|---|
{sc_rows}

**Survey-based totals** (share of households whose main light is solar, applied to {cal['households_total']:,} households):

| Source | Solar share | Households |
|---|---|---|
{sv_rows}

**Result.** Our map implies about {ours:,.0f} solar households. That is {position(ours, low, high)} implied by GOGLA
({low:,.0f} to {high:,.0f}), and {ours / central:.1f} times the central scenario ({central:,.0f}). The range is a
factor of {high / low:.0f} wide, so GOGLA data can only say the survey total is plausible. It cannot confirm it
to better than that.

**Decision.** The maps are **not rescaled** to GOGLA. The assumptions alone move the implied total by a factor of
{high / low:.0f}, GOGLA covers affiliates only, and the survey total already agrees with the national statistics
office survey (EICV7). A rescale would add precision the data do not have.

**Limitations.**
- Sales are units, not households, and the survival and units-per-household assumptions drive the result.
- Affiliates only: unverified products and informal sales are not counted, so true solar use is probably higher.
- Sales include replacements and upgrades, and households that bought more than one kit, which can inflate the
  household count. Appliances are reported separately and are not included.
- Sources date from different years: Census 2022, EICV7 2023/24, DHS 2025. Solar use grew and sales followed a
  subsidy boom-and-bust cycle (2025 first-half sales were low).
- Our map counts a household as solar only if solar is its MAIN light source.
"""


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--country", default="rwanda")
    args = parser.parse_args()
    cfg = load_config()
    cal = cfg["calibration"]
    out_dir = repo_path(cfg["paths"]["outputs"])
    open_labels = repo_path(cfg["paths"]["processed"]) / "open_labels"

    scenarios = scenario_table(cal["gogla_sales_thousands"], cal["reference_year"], cal["scenarios"], cal["households_total"])

    districts = pd.read_csv(out_dir / f"predicted_district_shares_{args.country}.csv")
    map_solar = np.average(districts["share_solar"], weights=districts["population"])
    dhs = pd.read_csv(out_dir / f"label_summary_{args.country}.csv").set_index("area").loc["national", "share_solar"]
    eicv7 = pd.read_csv(open_labels / "eicv7_province_lighting.csv").set_index("province").loc["Rwanda", "share_solar"]
    census = pd.read_csv(open_labels / "census_province_lighting.csv")
    census = census[(census["province"] == "Rwanda") & (census["area"] == "total")].iloc[0]
    shares = {"Our map (trained on DHS 2025)": map_solar, "DHS 2025 (survey, household-weighted)": dhs,
              "EICV7 2023/24": eicv7, "Census 2022 (solar + lantern)": (census["solar"] + census["lantern"]) / census["total"]}
    survey = pd.DataFrame({"share": shares}).rename_axis("source")
    survey["households"] = survey["share"] * cal["households_total"]

    result = pd.concat([scenarios.assign(kind="GOGLA-implied"),
                        survey.reset_index().rename(columns={"source": "scenario"}).assign(kind="survey")], ignore_index=True)
    result.to_csv(out_dir / f"calibration_note_{args.country}.csv", index=False)
    note = build_note(args.country, scenarios, survey, cfg)
    (out_dir / f"calibration_note_{args.country}.md").write_text(note)

    pd.set_option("display.width", 200)
    print(scenarios.round(3).to_string(index=False))
    print(survey.round(3).to_string())
    print(f"\nWrote calibration_note_{args.country}.csv and .md")


if __name__ == "__main__":
    main()
