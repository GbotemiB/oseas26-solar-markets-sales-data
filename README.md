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

Shortcuts with `make` (run `make help` for the list):

```bash
make test       # the tests
make demo       # the synthetic demo, same as python -m src.demo
make demo-map   # the demo, then docs/demo_map.html
make pipeline   # the real Rwanda run from the features to the report and web map (needs DHS data)
```

`make pipeline` reruns only the steps whose output is missing or older than the step before it.
`make -n pipeline` previews what would run, `make -B pipeline` redoes everything. `make download` fetches the
open inputs; the Google Earth Engine layers are a separate command (see the pipeline table).

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
| 13. EAE market-reality layer | `python -m src.export_eae --country rwanda` | `outputs/rwanda_market_reality_eae.tif` (EPSG:4326), `..._utm.tif`, plus `rwanda_households.tif` and `rwanda_unserved_households.tif` (counts) |
| 14. Validation report | `python -m src.report --country rwanda` | `outputs/validation_report_rwanda.md` |
| 15. Web map | `python -m src.webmap --country rwanda` (or `--demo`) | `docs/map.html` (`docs/demo_map.html`): one self-contained file, open it in a browser |

GEE = Google Earth Engine. The project ID is set in `config.yaml` (`gee.project`).
If a layer fails or times out, rerun it as a Drive export, e.g.
`python -m src.download.gee_features --layers building_count building_presence --drive`.
All settings (paths, radius, thresholds, GOGLA scenarios) are in `config.yaml`. Rasters are git-ignored:
publish the final ones through a release, not git.

## The market-reality layer

`rwanda_market_reality_eae.tif` holds, per 1 km cell, **unserved / (solar + unserved)**: the unserved share
among households that are not on the grid. **1 = the off-grid solar market has failed to arrive, 0 = it has
quietly succeeded.** The challenge brief did not fix a direction, so check it before mixing the layer with
others. Cells where fewer than 20% of households lack grid (non-grid share below 0.2) are no-data: the ratio rests on few households there, and at a 0.05 cut-off the highest values were town edges. Read it together with `rwanda_unserved_households.tif` (households with neither grid nor solar per cell): a high ratio over few households is small need.

Treat it as a **low-confidence hypothesis**. Separating solar from unserved is only weakly predictable from
open data (held-out error 0.185, 8% better than a flat map, not clearly better than night lights alone),
so the values are a gradient, not a clear split. Use it with `rwanda_do_not_trust.tif`.

**Naive baseline** (`python -m src.naive_baseline --country rwanda`): population x darkness, the method the
challenge brief warns about. At cell level it correlates -0.18 with our layer, and only 13% of its top decile is
in our top decile (10% by chance). On held-out districts, night lights alone and the naive index rank the survey
value unserved / (solar + unserved) no better than chance (AUC 0.39 and 0.48); our model reaches 0.70 (ours minus
naive +0.22, 95% interval +0.10 to +0.34). This is still a low-confidence hypothesis: 53% of the naive top decile is
also flagged by our do-not-trust mask. Results: `outputs/naive_baseline_rwanda.{csv,md}`.

## Reports

- `outputs/validation_report_rwanda.md`: random vs spatial CV, uncertainty coverage, control totals, solar test,
  naive baseline (regenerate with `python -m src.report --country rwanda`)
- `outputs/limitations_rwanda.md`: one-page limitations statement
- `outputs/calibration_note_rwanda.md`: survey-based solar total vs GOGLA sales

## Known limits

- The maps are model estimates from 560 survey clusters, not a census.
- The unserved share is mostly torch users (about 26%), not the official access gap (about 12%).
- GOGLA (Global Off-Grid Lighting Association) sales are national only and cover affiliate companies. The
  maps are **not** rescaled to them; the gap is reported in `outputs/calibration_note_rwanda.md`.
- Independent checks: the served share (grid + solar) by district and sector, the grid-connected share by sector
  (Census C1.5), and the grid/solar split by province. Below province level solar is only checked through a rough
  proxy. See `outputs/control_totals_*_rwanda.csv` and `outputs/validation_report_rwanda.md`.

## Licence

Code: MIT. Each data source keeps its own licence; see `data/SOURCES.md`.
