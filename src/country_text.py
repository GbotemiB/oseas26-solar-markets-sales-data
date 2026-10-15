"""Country-specific wording and numbers for the output metadata and the report.

Everything that names a country, a survey or a country-only fact comes from `countries.<c>` in config.yaml,
so a Kenya or Nigeria run is never labelled as Rwanda. Pure functions, no file reading except `read_household_size`.
"""
import json


def labels_tag(ccfg: dict) -> str:
    """The `labels` tag: which survey trained the model."""
    return f"{ccfg['display_name']} {ccfg['survey']}, restricted data. Only model-derived outputs are published."


def limitations_text(ccfg: dict, n_clusters: int) -> str:
    """Limitations for the share and uncertainty rasters."""
    text = (f"Modelled estimate from {n_clusters} survey clusters, not a census. Training label is the MAIN lighting "
            "source (DHS hv262), so households using grid plus solar count as grid. DHS cluster locations are "
            "displaced by up to 2 km (urban) and 5 km (rural). Solar versus unserved is separated less well than "
            "grid versus non-grid. Night lights and satellite layers cannot see small solar systems. "
            + ccfg["unserved_note"])
    notes = [ccfg["label_note"], ccfg["frame_note"]]
    return text + "".join(" " + n for n in notes if n)


def household_total(ccfg: dict, calibration: dict, population_total: float, mean_size) -> tuple:
    """National household count behind the count layers: (total, method text), or (None, None) to skip the layers.

    Rwanda has a sourced count (EICV7). Other countries use population divided by the mean household size of
    the DHS survey (one aggregate number), or skip the count layers when that number is not available.
    """
    if ccfg["external_checks"]:
        return float(calibration["households_total"]), "national count, EICV7 2023/24"
    if mean_size is None:
        return None, None
    return population_total / mean_size, f"population divided by the mean DHS household size ({mean_size:.2f} members)"


def read_household_size(path):
    """Mean household size from data/restricted/labels/<c>_household_size.json, or None if the file is missing."""
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["mean_household_size"])
