# Naive "dense and dark" index versus our market-reality layer (kenya)

Naive index = population x darkness (darkness = 1 minus the percentile of night lights). We compared it with our layer
(unserved / (solar + unserved) among non-grid households) at cell level and, on held-out districts, against the survey.

**Cells** (495,123 cells with both values): Spearman correlation -0.44. Of the naive top 10%,
3% is also in our top 10% by cells (6% by unserved households;
Jaccard 0.02); chance would give 10%. Our do-not-trust mask flags
77% of the naive top 10% (all cells: 76%). Our predicted solar share there is
38% against 29% nationally (population-weighted).

**Held-out districts** (1117 clusters, spatial cross-validation): Spearman with the survey value is
-0.13 naive, 0.23 night lights alone, 0.54 ours.
AUC for the top third is 0.41, 0.65 and 0.82. Ours minus naive: AUC
+0.41 (95% interval +0.32 to +0.50).

**Conclusion:** At cell level the two layers rank cells in roughly opposite order. On held-out districts our model ranks the survey outcome better than the naive index. Both rest on weak signals; read the layer as a hypothesis.
