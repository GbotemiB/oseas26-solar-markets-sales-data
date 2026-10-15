# Naive "dense and dark" index versus our market-reality layer (nigeria)

Naive index = population x darkness (darkness = 1 minus the percentile of night lights). We compared it with our layer
(unserved / (solar + unserved) among non-grid households) at cell level and, on held-out districts, against the survey.

**Cells** (900,785 cells with both values): Spearman correlation -0.50. Of the naive top 10%,
0% is also in our top 10% by cells (0% by unserved households;
Jaccard 0.00); chance would give 10%. Our do-not-trust mask flags
68% of the naive top 10% (all cells: 53%). Our predicted solar share there is
9% against 7% nationally (population-weighted).

**Held-out districts** (861 clusters, spatial cross-validation): Spearman with the survey value is
-0.05 naive, 0.33 night lights alone, 0.38 ours.
AUC for the top third is 0.48, 0.65 and 0.66. Ours minus naive: AUC
+0.17 (95% interval +0.10 to +0.25).

**Conclusion:** At cell level the two layers rank cells in roughly opposite order. On held-out districts our model ranks the survey outcome better than the naive index. Both rest on weak signals; read the layer as a hypothesis.
