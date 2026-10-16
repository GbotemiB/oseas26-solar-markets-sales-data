<p align="center">
  <a href="https://github.com/EnAccess/oseas26-solar-markets-sales-data">
    <img
      src="https://drive.google.com/uc?id=1gtL_p7l3HbOcCzc09A7KW5d7B5qn-BDs"
      alt="Identifying solar markets based on sales data"
      width="640"
    >
  </a>
</p>
<p align="center">
    October 26-27 | Open Source in Energy Access Symposium Hackathon | Kigali, Rwanda
</p>

---

# Identifying Solar Markets Based on Sales Data

Build a market-reality layer for the
[Energy Access Explorer (EAE)](https://www.energyaccessexplorer.org) that
distinguishes where the off-grid solar market has failed to arrive from where it
has quietly succeeded.

## Abstract and goal

Grant capital and results-based financing for off-grid solar tend to flow toward
territories that are already commercially viable, because those are the
territories that are visible in the data. Places that remain dark across every
available dataset also remain dark in most funding decisions.

The sector does not lack maps of where energy demand should exist; the Energy
Access Explorer (EAE) already provides these, running multi-criteria analysis on
the fly across normalized demand and supply layers so that planners can reweight
criteria interactively and rank priority areas from their own perspective. What
EAE currently doesn't include is a **market-reality layer**: a view of where the
off-grid solar market has failed to arrive, and a means of distinguishing that
from where it has quietly succeeded. This challenge asks teams to build that layer
for a country of their choice, integrating sales data.

VIIRS and Black Marble have a radiance detection floor well above the emissions of
a solar home system or pico-lantern, which means a village where every household
owns a solar home system registers as dark, and a village with no energy access at
all also registers as dark. The intuitive formulation of this challenge — in which
dense population combined with zero nighttime lights is taken to indicate an
underserved community — will therefore surface precisely those territories where
the off-grid sector has already done its best work. Darkness indicates the absence
of a grid connection, which defines the addressable market for distributed
renewable energy, but it says nothing about whether that market has been served.

Actual penetration must be measured from sources that measure penetration, which
is why georeferenced household survey data (DHS and MICS geo-clusters carrying
solar panel ownership and electricity source, and ESMAP Multi-Tier Framework
surveys carrying explicit Tier 0 to 5 access) forms the backbone of this
challenge. GOGLA sales data is semi-annual, affiliate-reported, and aggregated to
country and product category, with company-level detail held confidential under a
three-data-point rule and no public access to the underlying platform, so it
should be treated as a national control total against which a survey-derived
penetration surface is calibrated — rather than as a pixel-level penetration layer
in its own right.

## Expected outcomes

Three core deliverables (plus an optional prototype):

- **Predictive model of off-grid solar penetration:** an open-source model trained
  on georeferenced survey labels, with blocked cross-validation performance
  reported alongside random cross-validation performance, and feature analysis
  framed as hypotheses rather than findings.
- **EAE-conformant layer:** at least one raster, with values normalized 0 to 1 and
  direction documented, accompanied by metadata covering source, year, method, and
  known limitations.
- **Documentation and limitations statement:** a brief write-up describing the
  datasets used, the modelling approach, the validation methodology, and accuracy
  metrics, together with a one-page statement of what the model cannot see, where
  it will be wrong, and who it would misallocate capital toward.
- **Interactive prototype (optional):** a map in which priority zones are visible,
  inspectable, and traceable back to the layers that produced them.

## Required knowledge

### Stack

- **Python geospatial:** GeoPandas for vector processing; Rasterio / GDAL for
  raster analysis.
- **Machine learning:** Scikit-learn, XGBoost, LightGBM for modelling off-grid
  solar penetration.
- **Geospatial and socio-economic data:** NASA Black Marble; WorldPop or Meta High
  Resolution Settlement Layer for population; Meta Relative Wealth Index via the
  Humanitarian Data Exchange; Gridfinder or OpenStreetMap for grid infrastructure.

#### Solar market and sales data

This is the suggested group that allows a team to distinguish a village the market
has already served from a village it has never reached. Teams are encouraged to
identify more data during the hackathon. Each source is listed with the resolution
it operates at, because that determines the job it can do:

1. **GOGLA Sales and Impact Data:** unit sales by country, region, and product
   category, split between cash and PAYGo, reported semi-annually and annually by
   GOGLA affiliates. Published as public reports with accompanying tables. The
   underlying platform at data.gogla.org is restricted to companies that report
   into it, and a three-data-point rule enforces confidentiality, so
   distributor-level footprints are not obtainable. *Resolution:* national.
   *Job:* control total.
2. **Off-Grid Solar Market Trends Report (GOGLA and ESMAP):** market sizing,
   investment gap, and outlook. *Resolution:* regional and national. *Job:*
   context and sanity check.
3. **GOGLA deals database:** investment flows into off-grid solar companies from
   2012 onward. *Resolution:* company. *Job:* identifying which distributors are
   capitalized to expand into a new territory.
4. **PAYGo PERFORM:** standardized key performance indicators for PAYGo company
   performance, including portfolio quality and repayment. *Resolution:* company,
   aggregated. *Job:* identifying which distributors could absorb and deploy a
   grant.
5. **AMDA Benchmarking Africa's Minigrids:** operational and financial benchmarks
   aggregated across member developers and countries. *Resolution:* operator,
   aggregated. *Job:* context on mini-grid viability.
6. **ENERGYDATA.INFO (ESMAP / World Bank):** open data registry with API access,
   carrying existing and candidate mini-grid project datasets, grid network data,
   and the Africa Electricity Grids Explorer. *Resolution:* point and line
   geometry. *Job:* the only georeferenced market and infrastructure source in
   this group.
7. **Infrastructure data available in Energy Access Explorer.**
8. **National results-based financing and electrification programme records:**
   connection counts and subsidy disbursements published by rural electrification
   agencies and programme administrators. *Resolution:* sub-national where
   available, variable quality. *Job:* validation and ground-truthing.
9. **DHS and MICS geo-clusters, and ESMAP Multi-Tier Framework surveys:**
   household solar panel ownership, electricity source, and Tier 0 to 5 access,
   carried at cluster level with displaced GPS. *Resolution:* point. *Job:* this
   is the only source in the challenge that measures penetration at a location,
   and it is therefore the label.

### Programming languages

- Python
- SQL
- JavaScript / any other for frontend

### Helpful experiences

- Experience with Geospatial Analysis and Remote Sensing using, for example,
  Python, Google Earth Engine, PostGIS, QGIS, GeoJSON, GeoParquet, or spatial SQL.
- Experience with Machine Learning and Predictive Modelling, including feature
  engineering, model evaluation, and validation of spatially autocorrelated data.
- Experience with Data Science and Analytics using Python, Pandas and GeoPandas,
  data cleaning and integration, and statistical analysis.
- Familiarity with household survey microdata, for example DHS, MICS, LSMS, or
  Multi-Tier Framework surveys.
- Knowledge of off-grid solar markets and distribution models, including solar
  home systems, pico-solar, and results-based financing.
- Dashboard and mapping development skills.

## Person of contact supporting this challenge

- Preston Adie

## Getting started

- Questions about this challenge? Join the OSEAS Discord server: <https://community.oseas.org/> — that's where you reach the case givers and mentors, before kickoff and throughout the hackathon. 
- Introduce yourself in the `#introductions` channel and join the relevant
  channels for this challenge.
- Read the documentation:
  - [Energy Access Explorer](https://www.energyaccessexplorer.org)
  - [EAE GitHub](https://github.com/energyaccessexplorer)
  - [EAE Technical Note](https://www.wri.org/research/energy-access-explorer-data-and-methods?ap3c=IGaj6AgspJqgeKwBAGaj6AgmzCZ5Iv70Fr7H6ahniwtFr1FOgg)
