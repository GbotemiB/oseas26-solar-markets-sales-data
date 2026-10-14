# Calibration note: survey-based solar total vs GOGLA sales (Rwanda)

**Question.** Is the national number of households using off-grid solar in our map consistent with what GOGLA
(Global Off-Grid Lighting Association) sales imply?

**Sales used.** GOGLA Global Off-Grid Solar Market Report 2025, Figure 19, solar energy kits sold in Rwanda
(2021: 237k, 2022: 266k, 2023: 123k, 2024: 171k, 2025: 82k). Lanterns and solar home systems together. GOGLA affiliate companies only, so this is a lower bound
on the market. No Rwanda split by product type is given.

**Conversion.** A unit counts as a working unit while younger than its lifetime. Households = units in use
divided by units per household. The lifetimes and units per household are **our assumptions** (lanterns about
2 to 3 years, home systems about 5), because the report gives none. Reference year 2025.

| Scenario | Lifetime (years) | Units per household | GOGLA-implied households | Share of 3,289,810 households |
|---|---|---|---|---|
| low | 2.5 | 1.5 | 209,667 | 6.4% |
| central | 3.5 | 1.2 | 424,167 | 12.9% |
| high | 5.0 | 1.0 | 879,000 | 26.7% |

**Survey-based totals** (share of households whose main light is solar, applied to 3,289,810 households):

| Source | Solar share | Households |
|---|---|---|
| Our map (trained on DHS 2025) | 19.7% | 646,918 |
| DHS 2025 (survey, household-weighted) | 19.6% | 644,844 |
| EICV7 2023/24 | 22.1% | 726,903 |
| Census 2022 (solar + lantern) | 15.1% | 496,359 |

**Result.** Our map implies about 646,918 solar households. That is inside the range implied by GOGLA
(209,667 to 879,000), and 1.5 times the central scenario (424,167). The range is a
factor of 4 wide, so GOGLA data can only say the survey total is plausible. It cannot confirm it
to better than that.

**Decision.** The maps are **not rescaled** to GOGLA. The assumptions alone move the implied total by a factor of
4, GOGLA covers affiliates only, and the survey total already agrees with the national statistics
office survey (EICV7). A rescale would add precision the data do not have.

**Limitations.**
- Sales are units, not households, and the survival and units-per-household assumptions drive the result.
- Affiliates only: unverified products and informal sales are not counted, so true solar use is probably higher.
- Sales include replacements and upgrades, and households that bought more than one kit, which can inflate the
  household count. Appliances are reported separately and are not included.
- Sources date from different years: Census 2022, EICV7 2023/24, DHS 2025. Solar use grew and sales followed a
  subsidy boom-and-bust cycle (2025 first-half sales were low).
- Our map counts a household as solar only if solar is its MAIN light source.
