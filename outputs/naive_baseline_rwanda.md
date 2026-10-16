# Naive "dense and dark" index versus our market-reality layer (rwanda)

Naive index = population x darkness (darkness = 1 minus the percentile of night lights). We compared it with our layer
(unserved / (solar + unserved) among non-grid households) at cell level and, on held-out districts, against the survey.

**Cells** (21,579 cells with both values): Spearman correlation -0.18. Of the naive top 10%,
13% is also in our top 10% by cells (18% by unserved households;
Jaccard 0.07); chance would give 10%. Our do-not-trust mask flags
53% of the naive top 10% (all cells: 44%). Our predicted solar share there is
23% against 20% nationally (population-weighted).

**Held-out districts** (389 clusters, spatial cross-validation): Spearman with the survey value is
-0.11 naive, -0.20 night lights alone, 0.33 ours.
AUC for the top third is 0.48, 0.39 and 0.70. Ours minus naive: AUC
+0.22 (95% interval +0.10 to +0.34).

Night lights alone point the wrong way here: an AUC below 0.5 means darker clusters are LESS often unserved (more often on solar) than brighter ones.

**Conclusion:** At cell level the two layers are largely different. On held-out districts our model ranks the survey outcome better than the naive index. Both rest on weak signals; read the layer as a hypothesis.
