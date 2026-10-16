# Limitations statement: Rwanda off-grid solar access map

**What this is.** A model estimate of the share of households on grid, on off-grid solar and unserved, per 1 km
cell, trained on 560 DHS 2025 survey clusters and open satellite layers. It is not a census. The market-reality layer
built from it is a low-confidence hypothesis.

## What the model cannot see

- **Small solar systems.** Night lights and built-up area do not show lanterns or small home systems, so solar-served and
  unserved villages can look alike.
- **Mixed households.** The label is the main light source, so a household with grid and solar counts as grid.
- **What "unserved" means.** Mostly households lighting with torches (about 26% nationally), not the official access gap
  (about 12%). Do not compare the two.
- **Grid growth.** The grid map is from 2019 and features date from 2015 to 2025. Newer lines are missed.
- **Exact locations.** DHS moves cluster points by up to 2 km (urban) and 5 km (rural). Cell values are neighbourhood
  estimates, not village facts.
- **Sales.** GOGLA data is national and covers affiliate companies only. The map is not rescaled to it; it falls inside
  the sales-implied range, which is a factor of 4 wide.

## Where it will be wrong

- **Solar versus unserved.** Held-out error is 0.185, only 8% better than a flat map and not clearly better than night
  lights alone. Nine in ten cells score between 0.45 and 0.74, so read the layer as a ranking, together with the unserved-households layer.
- **Levels.** Spatial cross-validation error is 0.154 per share (0.209 for a constant map). District rankings are weak
  (Spearman 0.39 against EICV7).
- **Southern Province.** The grid share is over-predicted by about 12 points against EICV7.
- **Unusual places.** About 19% of cells have features outside the range seen at survey clusters. Nothing is calibrated
  there.
- **Ranges.** The 80% ranges cover 79 to 81% in held-out districts on average, not for every cell, and are wide because
  they include survey noise.
- **Timing.** Sources date from 2022, 2023/24 and 2025, so part of any gap against them is real change.

## Who it could misallocate capital toward

- **Towards** places that look dark and remote to the model but already have solar: households with a system would be
  counted as need.
- **Away from** places with solar-like features but no market: a planner trusting a low value would skip them. Values
  near 0.5 carry almost no information.
- **Either way** inside the do-not-trust mask, where ranges are widest or features lie outside the survey.

## How to use it

Use it to shortlist places for a field check, not to allocate funds. Treat cells flagged in
`rwanda_do_not_trust.tif` as unknown. Refresh with the next DHS or EICV round before any investment decision.
