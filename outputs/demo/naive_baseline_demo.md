# Naive "dense and dark" index versus our market-reality layer (demo)

SYNTHETIC DEMO DATA: made up by src/make_demo_data.py to show that the pipeline runs. It is not Rwanda, not DHS data, and the numbers mean nothing for any real place.

Naive index = population x darkness (darkness = 1 minus the percentile of night lights). We compared it with our layer
(unserved / (solar + unserved) among non-grid households) at cell level and, on held-out districts, against the survey.

**Cells** (1,436 cells with both values): Spearman correlation -0.88. Of the naive top 10%,
0% is also in our top 10% by cells (0% by unserved households;
Jaccard 0.00); chance would give 10%. Our do-not-trust mask flags
49% of the naive top 10% (all cells: 69%). Our predicted solar share there is
30% against 20% nationally (population-weighted).

**Held-out districts** (123 clusters, spatial cross-validation): Spearman with the survey value is
-0.24 naive, 0.01 night lights alone, 0.14 ours.
AUC for the top third is 0.39, 0.47 and 0.62. Ours minus naive: AUC
+0.23 (95% interval -0.06 to +0.40).

Night lights alone point the wrong way here: an AUC below 0.5 means darker clusters are LESS often unserved (more often on solar) than brighter ones.

**Conclusion:** At cell level the two layers rank cells in roughly opposite order. On held-out districts we cannot tell our model from the naive index (the interval for the difference includes zero), so this test does not show that our layer is better. Both rest on weak signals; read the layer as a hypothesis.
