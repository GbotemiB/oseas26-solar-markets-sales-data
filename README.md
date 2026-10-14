# OSEAS 2026: Real Off-Grid Solar Access in Rwanda

Entry for the **"Identifying Solar Markets Based on Sales Data"** challenge at
OSEAS (Open Source in Energy Access Symposium) 2026, Kigali.

We build an open 1 km map that shows, for each cell, the share of households on the
**grid**, on **off-grid solar**, and **unserved**, with an uncertainty range. The
headline layer for EAE (Energy Access Explorer) shows where the off-grid solar market
has not yet arrived.

## Try it without DHS data

The demo runs the whole pipeline (predict, uncertainty ranges, do-not-trust mask, EAE layer) on a
**synthetic** 40 x 40 km country. It needs no DHS access, no downloads and no Google Earth Engine account.

```bash
conda env create -f environment.yml
conda activate oseas26
python -m src.demo
```

Outputs go to `outputs/demo/` (GeoTIFFs, each tagged as synthetic). The synthetic data is in `data/demo/`
and is rebuilt with `python -m src.make_demo_data`. The demo also prints the error against the hidden true
shares, which only synthetic data has. That error is far smaller than on real data, because the planted
rule is easy: do not read it as the accuracy of the Rwanda map.

Tests (all use synthetic data): `python -m pytest -q`

## Data access (for the real Rwanda run)

Labels come from DHS (Demographic and Health Surveys). DHS data is free but **not open**:
request access at https://dhsprogram.com/data/new-user-registration.cfm, then put the
download zips in:

```
data/restricted/dhs/rwanda/   # Rwanda DHS 2025
data/restricted/dhs/kenya/    # Kenya DHS 2022
data/restricted/dhs/nigeria/  # Nigeria DHS 2023-24
```

`data/restricted/` is git-ignored. **Never commit DHS files or GPS points.** Only model-derived,
aggregated outputs are published.

## Pipeline (Rwanda)

| Step | Command | Output |
|---|---|---|
| 1. Cluster labels | `python -m src.labels_dhs` | `data/restricted/labels/` (restricted), `outputs/label_summary_*.csv` |
| 2. Boundaries | `python -m src.download.boundaries --country rwanda` | `data/raw/boundaries/` |
| 3. 1 km grid | `python -m src.grid --country rwanda` | `data/processed/rwanda_grid_*` |
| 4. GEE features (100 m) | `earthengine authenticate` (once), then `python -m src.download.gee_features --country rwanda` | `data/raw/gee/` |
| 5. Distance layers | `python -m src.download.gridfinder`, `python -m src.download.osm --country rwanda`, `python -m src.distance_layers --country rwanda` | `data/processed/layers/` |
| 6. Feature stack | `python -m src.features --country rwanda` | `data/processed/rwanda_features_1km.csv`, `data/restricted/features/` |
| 7. Model comparison, random vs spatial CV | `python -m src.train --country rwanda` | `outputs/cv_results_rwanda.csv` |
| 8. Share maps | `python -m src.predict --country rwanda` | `outputs/rwanda_share_{grid,solar,unserved}.tif` |
| 9. Uncertainty ranges and do-not-trust mask | `python -m src.uncertainty --country rwanda` | `outputs/rwanda_share_*_{low,high}.tif`, `outputs/rwanda_do_not_trust.tif` |
| 10. Control totals | `python -m src.download.nisr`, `python -m src.labels_open`, `python -m src.validate --country rwanda` | `outputs/control_totals_*_rwanda.csv` |
| 11. Solar vs unserved test | `python -m src.solar_test --country rwanda` | `outputs/solar_vs_unserved_test_rwanda.csv` |
| 12. GOGLA calibration note | `python -m src.calibrate --country rwanda` | `outputs/calibration_note_rwanda.md` |
| 13. EAE market-reality layer | `python -m src.export_eae --country rwanda` | `outputs/rwanda_market_reality_eae.tif` (EPSG:4326), `..._utm.tif` |
| 14. Validation report | `python -m src.report --country rwanda` | `outputs/validation_report_rwanda.md` |

GEE = Google Earth Engine. The project ID is set in `config.yaml` (`gee.project`).
If a layer fails or times out, rerun it as a Drive export, e.g.
`python -m src.download.gee_features --layers building_count building_presence --drive`.
All settings (paths, radius, thresholds, GOGLA scenarios) are in `config.yaml`. Rasters are git-ignored:
publish the final ones through a release, not git.

## The market-reality layer

`rwanda_market_reality_eae.tif` holds, per 1 km cell, **unserved / (solar + unserved)**: the unserved share
among households that are not on the grid. **1 = the off-grid solar market has failed to arrive, 0 = it has
quietly succeeded.** The challenge brief did not fix a direction, so check it before mixing the layer with
others. Cells with almost no non-grid households (non-grid share below 0.05) are no-data.

Treat it as a **low-confidence hypothesis**. Separating solar from unserved is only weakly predictable from
open data (held-out error 0.185, 8% better than a flat map, not clearly better than night lights alone),
so the values are a gradient, not a clear split. Use it with `rwanda_do_not_trust.tif`.

## Reports

- `outputs/validation_report_rwanda.md`: random vs spatial CV, uncertainty coverage, control totals, solar test
  (regenerate with `python -m src.report --country rwanda`)
- `outputs/limitations_rwanda.md`: one-page limitations statement
- `outputs/calibration_note_rwanda.md`: survey-based solar total vs GOGLA sales

## Known limits

- The maps are model estimates from 560 survey clusters, not a census.
- The unserved share is mostly torch users (about 26%), not the official access gap (about 12%).
- GOGLA (Global Off-Grid Lighting Association) sales are national only and cover affiliate companies. The
  maps are **not** rescaled to them; the gap is reported in `outputs/calibration_note_rwanda.md`.
- Independent checks exist only for the served share (grid + solar) by district and sector, and for the
  grid/solar split by province. See `outputs/control_totals_*_rwanda.csv`.

## Licence

Code: MIT. Each data source keeps its own licence; see `data/SOURCES.md`.
