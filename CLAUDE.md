# CLAUDE.md — OSEAS 2026: Real Off-Grid Solar Access in Rwanda

This file gives Claude Code the context for this project. Read it before doing any work.

## What this project is

Our entry for the **"Identifying Solar Markets Based on Sales Data"** challenge at
OSEAS (Open Source in Energy Access Symposium) 2026, Kigali.
Challenge repo: https://github.com/EnAccess/oseas26-solar-markets-sales-data

We have **one week** to build. The team works on a shared GitHub repo.

**Goal:** build an open, reproducible 1 km map of Rwanda that shows, for every cell:
- share of households on the **grid**
- share on **off-grid solar** (solar home system or solar lantern)
- share **unserved** (no electricity) — this is the market gap

…plus an **uncertainty range** for each share, so planners can see where the model
should not be trusted.

**Why:** Rwanda targets 100% household access by 2029. About 25% of households are
served off-grid. Night-time lights cannot see small solar systems, so solar-served and
unserved villages look the same. Our map separates them and is honest about errors.

**Challenge framing (from the official brief):** build a **market-reality layer** for
EAE that separates where the off-grid solar market has **failed to arrive** from where
it has **quietly succeeded**. Darkness only shows "no grid" (the addressable market),
not whether that market is served. The brief says:
- Survey geo-clusters (DHS, MTF) are **the label** — the only source measuring
  penetration at a location.
- GOGLA sales data is national only, so use it as a **national control total** to
  calibrate the survey-based map, never as a pixel-level layer.
- Feature analysis must be framed as **hypotheses, not findings**.
- Challenge contact: Preston Adie (OSEAS Discord).

## Outputs we must deliver

1. Penetration maps (grid / solar / unserved) at 1 km, as GeoTIFF rasters
2. Uncertainty map for each share (low and high value per cell)
3. **Market-reality layer:** one 0–1 normalised raster in EAE (Energy Access Explorer)
   format. Headline value = unserved share among non-grid households
   (1 = market has failed to arrive, 0 = market has served the area).
   **Document the direction** in the metadata. Metadata must cover source, year,
   method and known limitations.
4. Validation report: random vs spatial block cross-validation (CV)
5. Limitations statement (one page): what the model cannot see, where it will be
   wrong, and who it would misallocate capital toward
6. Calibration note: survey-based national solar total vs GOGLA-implied total
7. Public repo with open code (MIT licence), a demo that runs without DHS data, clear README

Nice to have: simple web map prototype (Leaflet or Streamlit), change over time using
older DHS surveys, a run on a second country.

## CRITICAL: data rules (DHS terms of use)

DHS (Demographic and Health Surveys) data is free but **not open**. We have approved access.

- **Never commit DHS files or GPS points.** They live only in `data/restricted/`,
  which must be in `.gitignore`. Check `.gitignore` before any commit.
- Never print, log or save raw household rows or cluster coordinates into any file
  outside `data/restricted/` (no notebooks outputs, no CSVs in `outputs/`, no test fixtures).
- Only aggregated, model-derived outputs (maps, rasters, summary stats) go public.
- Tests and the demo must use **synthetic data** in `data/demo/`, never real DHS data.
- The Meta Relative Wealth Index is **non-commercial (CC BY-NC 4.0)**. If used, also
  produce a version without it. The EAE layer uses the version without it.
- Every output file must carry metadata listing its sources and licences.

## Data

### Labels — Plan A (DHS, restricted)
Rwanda DHS 2025 (DHS-8 recode, survey code `RW` + version `91`).
Fieldwork June–October 2025. 560 clusters (198 urban, 362 rural), ~14,500 households.

Files (place in `data/restricted/dhs/`):
- `RWHR91DT.ZIP` — **Household Recode, Stata (.dta)** — main file, one row per household
- `RWGE91FL.ZIP` — GPS cluster points (shapefile) — name may differ slightly, check
- `RWGC91FL.ZIP` — Geospatial Covariates per cluster, if available (useful as a check)

Key variables (household recode):
| Variable | Meaning |
|---|---|
| `hv001` | Cluster number — joins to GPS file (`DHSCLUST`) |
| `hv002` | Household number |
| `hv005` | Household sample weight — **divide by 1,000,000** |
| `hv024` | Province |
| `hv025` | Urban (1) / rural (2) |
| `hv206` | Has electricity (yes/no) |
| `hv270` | Wealth index quintile |
| `hv262` | **Type of light at home** — 1 = electricity, 2 = solar lantern / standalone solar system, other = unserved. Same codes in RW, KE, NG |

**Label rule (settled):** see `label_rule` in `config.yaml` and `src/labels_dhs.py`.
- Rwanda, Kenya: `lighting` — grid = hv262 == 1, solar = hv262 == 2, unserved = rest.
- Nigeria: `electricity_then_lighting` — solar = hv262 == 2, grid = hv206 == 1, unserved = rest.
  About 19% of Nigerian households have electricity but mainly light with torches (power cuts),
  so hv262 alone would undercount grid. Flag `share_grid_unreliable` keeps this visible.
- Limitation: hv262 is the MAIN light source. Grid + solar households count as grid.
- Kenya only: `sh132g` "Household has a solar panel" can cross-check the solar share
  (in Rwanda sh132g is a different item — do not use).
- National weighted shares (grid / solar / unserved): RW 53.9 / 19.6 / 26.5,
  KE 56.5 / 27.9 / 15.7, NG 51.7 / 6.8 / 41.5. RW is close to EICV7 (50 / 22).
  RW "unserved" is ~26% (mostly torches), not the ~12% official access gap — explain in pitch.

GPS displacement (on purpose, for privacy): urban up to 2 km, rural up to 5 km
(1% of rural up to 10 km). So **never read single pixels at cluster points** —
use buffer means: 2 km radius for urban, 5 km for rural.

### Labels — Plan B (open, CC BY 4.0)
- Census 2022 (RPHC5), NISR (National Institute of Statistics of Rwanda): main source
  of lighting by sector (416 sectors). Check if solar is split from grid.
- EICV7 (Integrated Household Living Conditions Survey, 2023/24): lighting source by
  district (30 districts). National: 50% grid, 22% solar, 1% both.
- World Bank MTF (Multi-Tier Framework) Rwanda 2022: 223 villages, access tiers 0–5.
  Admin-unit level only, no GPS. Link to it, do not re-host raw files.

### Sales data — calibration (national control total)
- **GOGLA Sales and Impact reports** (public, semi-annual): unit sales by country and
  product type, cash vs PAYGo. Check whether Rwanda is reported on its own.
  Company-level data is confidential (three-data-point rule) — not obtainable.
- Convert cumulative sales to "households currently using solar" with product lifetimes
  (assumptions must be stated: e.g. lanterns ~2–3 years, solar home systems ~5 years)
  and units per household. Compare with the national total our map predicts.
- **Off-Grid Solar Market Trends Report (GOGLA/ESMAP):** context and sanity check.
- Stretch only: GOGLA deals database and PAYGo PERFORM (company performance measures)
  to note which companies could expand into gap areas.

### Validation sources
- EICV7 district totals and Census 2022 sector totals
- Rwanda off-grid subsidy / results-based financing records by district (ask organizers)

### Features (all open)
| Layer | Source | Notes |
|---|---|---|
| Population (100 m) | WorldPop | CC BY 4.0, also in Google Earth Engine (GEE) |
| Night-time lights | NASA Black Marble / VIIRS | via GEE |
| Predicted grid lines | Gridfinder (Zenodo record 3628142) | from 2019, grid has grown since |
| Roads, towns, markets, power lines | OpenStreetMap via Geofabrik | ODbL, must credit OSM |
| Building footprints | Google Open Buildings / Microsoft / Overture | count + area per cell |
| Travel time to cities | Malaria Atlas Project | confirm licence |
| Relative wealth (optional) | Meta RWI on HDX | non-commercial — see rules |
| Boundaries | geoBoundaries | districts, sectors |

Confirm each licence on its download page and record it in `data/SOURCES.md`.

## Repo structure

```
oseas26-rwanda-solar-access/
├── CLAUDE.md
├── README.md            # what it is, how to run, how to request DHS access
├── LICENSE              # MIT
├── environment.yml
├── config.yaml          # country, grid size (1 km), buffer sizes, paths, CRS
├── data/
│   ├── raw/             # open downloads (git-ignored, re-downloadable)
│   ├── restricted/      # DHS files + GPS (git-ignored, NEVER commit)
│   ├── processed/       # feature stack, cluster table (git-ignored)
│   ├── demo/            # small synthetic dataset (committed)
│   └── SOURCES.md       # every source, licence, download date
├── src/
│   ├── download/        # one script per open source
│   ├── grid.py          # build 1 km grid over Rwanda
│   ├── features.py      # stack features on grid; buffer means for clusters
│   ├── labels_dhs.py    # Plan A loader -> cluster shares
│   ├── labels_open.py   # Plan B loader (census, EICV7, MTF)
│   ├── train.py         # baseline, LightGBM, quantile models
│   ├── validate.py      # random vs spatial block CV, control totals
│   └── export_eae.py    # 0–1 raster + metadata
├── notebooks/           # exploration only; clear outputs before commit
├── outputs/             # maps, rasters, validation report
└── app/                 # optional web map
```

## Pipeline

1. **Download** open feature layers (`src/download/`).
2. **Grid:** 1 km grid over Rwanda. Use an equal-area or UTM CRS for distances
   (Rwanda: UTM zone 35S, EPSG:32735); export final rasters in EPSG:4326 if EAE needs it.
3. **Features:** stack all layers on the grid. For DHS clusters, compute buffer means
   (2 km urban / 5 km rural).
4. **Labels:** per cluster, weighted shares (using `hv005`): grid, solar, unserved.
   Also keep the number of households per cluster as a training weight.
5. **Train** (see Modelling).
6. **Validate** (see Validation).
7. **Calibrate** to the GOGLA-implied national solar total (rescale, or report the gap
   if the two disagree strongly).
8. **Export** penetration maps, uncertainty maps, market-reality EAE layer with metadata.

## Modelling

Only ~560 labelled clusters, so keep models small. The value is in honest validation
and uncertainty, not model complexity.

1. **Baseline:** multinomial logistic regression with 5–8 features.
2. **Main:** LightGBM, one model per share, `objective="cross_entropy"` (targets 0–1),
   `sample_weight` = households per cluster, `num_leaves=15`, `min_child_samples=10`,
   ~300 trees, learning rate 0.05. Rescale the three predictions to sum to 1.
3. **Optional spatial part:** regression kriging (Gaussian process on residuals), only
   if residuals show spatial pattern.
4. **Plan B (if no DHS):** disaggregation / learning from label proportions — predict
   per cell, sum (population-weighted) per sector/district, train so sums match totals.

## Validation and uncertainty

- **Random CV** (5 folds) vs **spatial block CV** (`GroupKFold`, groups = district).
  Report both side by side; the spatial score is the honest one.
- **Control totals:** sum predictions per district, compare with EICV7 / census.
- **Sanity checks:** Kigali mostly grid; remote areas far from grid mostly off-grid/unserved.
- **Uncertainty:** start with LightGBM quantile regression (`objective="quantile"`,
  alpha 0.1 and 0.9). Stretch: conformal prediction calibrated on spatial blocks.
- Produce a "do not trust" mask where the range is wider than a threshold (e.g. 0.4).
- Metric: household-weighted mean absolute error.
- Feature importance is reported as **hypotheses** ("may suggest"), never as causes.

## Known issues to remember

- REG (Rwanda Energy Group) and EICV7 access numbers differ (connections vs survey).
  Always state which source a number comes from.
- Gridfinder is from 2019; add OSM power lines and note this in limitations.
- Rwanda on EAE was only soft-launched (Sept 2026); export in EAE format anyway.

## Tools and conventions

- Python 3.11. Main libraries: geopandas, rasterio, rioxarray, xarray, earthengine-api,
  scikit-learn, lightgbm, pandas, matplotlib. Read DHS .dta with
  `pd.read_stata(path, convert_categoricals=False)`.
- All paths and parameters come from `config.yaml`; no hard-coded paths.
- Each pipeline step is runnable on its own: `python -m src.<step>`.
- Set random seeds for reproducibility.
- Work in branches, merge through pull requests, keep `main` runnable.
- Write simple, clear code and comments. Spell out acronyms in docs.

## Current status

- [x] DHS access approved (Rwanda 2025, including GPS)
- [x] DHS zips for RW, KE, NG in `data/restricted/dhs/<country>/`
- [x] Grid/solar variable found (`hv262`); `src/labels_dhs.py` builds cluster labels
- [x] Census 2022 and EICV7 downloaded and parsed (`src/labels_open.py`): grid/solar split only by province.
      Sector tables give served (C1.3) and grid-connected (C1.5); solar below province is only a rough proxy
      (C1.3 minus C1.5). MTF not used.
- [x] GOGLA Rwanda sales found (2025 report, Figure 19, affiliates only); calibration note written, maps not rescaled
- [x] Open question to organizers on the EAE layer direction was never answered: direction is documented as an assumption
- [x] GEE features, 1 km grid, feature stack, model comparison (random vs spatial CV)
- [x] Share maps, conformal uncertainty ranges, do-not-trust mask, control totals, solar-vs-unserved test
- [x] EAE market-reality layer (`src/export_eae.py`), worded as a low-confidence hypothesis
- [x] Synthetic demo (`python -m src.demo`), README, validation report, limitations statement
- [ ] Kenya and Nigeria runs, web map, change over time (nice to have, not started)
- [ ] Confirm licences marked "to confirm" in `data/SOURCES.md` (NISR, Gridfinder, MAP, geoBoundaries, Black Marble)
- [ ] Check DHS terms allow publishing derived 1 km rasters
- [ ] Initialise git and push through pull requests (the folder is not a git repo yet)

## Week plan

- Days 1–2: data + features (labels loaded, feature stack, labels joined to grid)
- Days 3–4: models (baseline, LightGBM + spatial CV, quantile ranges)
  — checkpoint: LightGBM beats baseline in spatial CV
- Days 5–6: checks + outputs (control totals, maps, EAE layer, optional prototype)
  — checkpoint: repo runs from a clean clone
- Day 7: limitations statement, README, demo, 5-minute pitch

If a checkpoint fails, cut scope rather than slip.