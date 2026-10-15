# Automation for the OSEAS 2026 Rwanda solar access pipeline.
#
#   make help       list the targets
#   make test       run the tests (synthetic data only)
#   make demo       run the whole pipeline on synthetic data, no DHS access needed
#   make pipeline   real Rwanda run, from the downloaded open data to the report and web map
#
# Each step is a file target: make reruns a step only when its output is missing or the output of
# the step before it is newer. Code and config edits do not trigger a rerun: to redo everything
# use `make -B pipeline`, to redo one step delete its output. `make -n pipeline` previews what would
# run. DHS data must already be in data/restricted/dhs/<country>/.

PY      ?= python
COUNTRY ?= rwanda
C       := $(COUNTRY)
OUT     := outputs
PROC    := data/processed
CFG     := config.yaml
# Census, EICV7 and GOGLA inputs exist only for Rwanda (external_checks in config.yaml); other countries skip those steps
MAP     := docs/$(if $(filter rwanda,$(C)),map,$(C)_map).html   # src.webmap names the file this way
EXT     := $(if $(filter rwanda,$(C)),$(OUT)/control_totals_summary_$(C).csv $(OUT)/calibration_note_$(C).md,)

.DEFAULT_GOAL := help
.PHONY: help env test demo demo-map download pipeline naive report webmap clean clean-demo clean-cache check-dhs compare

help: ## show this list
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables: COUNTRY=$(COUNTRY) PY=$(PY)   (example: make pipeline COUNTRY=rwanda)"

env: ## create the conda environment (then: conda activate oseas26)
	conda env create -f environment.yml

test: ## run the tests, synthetic data only
	$(PY) -m pytest -q

demo: ## run the pipeline on synthetic data (outputs/demo/)
	$(PY) -m src.demo

demo-map: demo ## build the demo web map (docs/demo_map.html)
	$(PY) -m src.webmap --demo

# ---------------------------------------------------------------- real run
check-dhs:
	@test -n "$$(ls data/restricted/dhs/$(C) 2>/dev/null)" || \
	  { echo "No DHS files in data/restricted/dhs/$(C)/. See 'Data access' in the README."; exit 1; }

download: ## download the open inputs (boundaries, Gridfinder, OSM, census); skips files already there
	$(PY) -m src.download.boundaries --country $(C)
	$(PY) -m src.download.gridfinder
	$(PY) -m src.download.osm --country $(C)
	$(if $(filter rwanda,$(C)),$(PY) -m src.download.nisr)
	@echo "Google Earth Engine layers are separate (needs 'earthengine authenticate'):"
	@echo "  $(PY) -m src.download.gee_features --country $(C)"

$(OUT)/label_summary_$(C).csv: | check-dhs
	$(PY) -m src.labels_dhs --country $(C)

$(PROC)/$(C)_grid_cells.csv:
	$(PY) -m src.grid --country $(C)

$(PROC)/layers/$(C)_dist_town.tif:
	$(PY) -m src.distance_layers --country $(C)

$(PROC)/$(C)_features_1km.csv: $(PROC)/$(C)_grid_cells.csv $(PROC)/layers/$(C)_dist_town.tif $(OUT)/label_summary_$(C).csv
	$(PY) -m src.features --country $(C)

$(OUT)/cv_results_$(C).csv: $(PROC)/$(C)_features_1km.csv
	$(PY) -m src.train --country $(C)

$(OUT)/$(C)_share_grid.tif: $(PROC)/$(C)_features_1km.csv $(OUT)/cv_results_$(C).csv
	$(PY) -m src.predict --country $(C)

$(OUT)/$(C)_do_not_trust.tif: $(OUT)/$(C)_share_grid.tif
	$(PY) -m src.uncertainty --country $(C)

$(OUT)/control_totals_summary_$(C).csv: $(OUT)/$(C)_share_grid.tif
	$(PY) -m src.labels_open
	$(PY) -m src.validate --country $(C)

$(OUT)/solar_vs_unserved_test_$(C).csv: $(OUT)/$(C)_do_not_trust.tif
	$(PY) -m src.solar_test --country $(C)

$(OUT)/calibration_note_$(C).md: $(OUT)/$(C)_share_grid.tif
	$(PY) -m src.calibrate --country $(C)

$(OUT)/$(C)_market_reality_eae.tif: $(OUT)/$(C)_do_not_trust.tif $(OUT)/solar_vs_unserved_test_$(C).csv
	$(PY) -m src.export_eae --country $(C)

$(OUT)/naive_baseline_$(C).csv: $(OUT)/$(C)_market_reality_eae.tif
	$(PY) -m src.naive_baseline --country $(C)

$(OUT)/validation_report_$(C).md: $(OUT)/cv_results_$(C).csv $(EXT) $(OUT)/solar_vs_unserved_test_$(C).csv $(OUT)/$(C)_market_reality_eae.tif $(OUT)/naive_baseline_$(C).csv
	$(PY) -m src.report --country $(C)

$(MAP): $(OUT)/$(C)_market_reality_eae.tif $(OUT)/$(C)_do_not_trust.tif
	$(PY) -m src.webmap --country $(C)

naive: $(OUT)/naive_baseline_$(C).csv ## compare our layer with the naive "dense and dark" index
report: $(OUT)/validation_report_$(C).md ## build the validation report
webmap: $(MAP) ## build the web map (docs/map.html, real data, do not publish before the DHS check)

pipeline: report webmap ## real run, every step from the features to the report and web map (after: make download)

compare: ## side-by-side comparison of the country runs (outputs/cross_country_summary.md)
	$(PY) -m src.cross_country

# ---------------------------------------------------------------- clean
clean-demo: ## delete the demo outputs and the demo web map
	rm -rf $(OUT)/demo docs/demo_map.html

clean-cache: ## delete Python caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean: clean-demo clean-cache ## clean-demo and clean-cache. Never touches data/ or the real outputs
