"""Make the synthetic demo dataset in data/demo/ (no DHS data, no real place).

Run:
    python -m src.make_demo_data

A made-up 40 x 40 km country with 16 districts. Every 1 km cell gets the 7 core features; a hidden rule
turns them into true grid / solar / unserved shares; about 200 fake survey clusters sample 25 households
each from those shares (so the labels are noisy, like a real survey). The true cell shares are kept in
`true_share_*` columns ONLY so the demo can show how close the pipeline gets. The real pipeline never
has them.

Files (all committed, all synthetic):
    demo_train_table.csv     one row per fake cluster: features, shares, district
    demo_features_1km.csv    one row per cell: features (2 km neighbourhood means), population, true shares
    demo_grid_cells.csv, demo_grid_spec.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter

from src.config import load_config, repo_path
from src.train import CLASSES

SIZE = 40                  # cells per side, 1 km each
DISTRICTS_PER_SIDE = 4     # 4 x 4 = 16 districts
N_CLUSTERS = 200
HOUSEHOLDS_PER_CLUSTER = 25
FEATURES = ["population", "built_up_surface", "night_lights_bm", "travel_time_city", "travel_time_health",
            "slope", "elevation"]


def softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def make_country(rng: np.random.Generator) -> pd.DataFrame:
    """One row per cell: 2 km neighbourhood features, population and the hidden true shares."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    towns = rng.uniform(5, SIZE - 5, size=(3, 2))
    dist_town = np.min([np.hypot(yy - r, xx - c) for r, c in towns], axis=0)        # km
    urban = np.exp(-dist_town / 5.0)                                                 # 1 in a town, fades out
    noise = lambda sigma: gaussian_filter(rng.normal(size=(SIZE, SIZE)), sigma)

    pop_total = 20 + 4000 * urban ** 2 * np.exp(0.3 * noise(2) / noise(2).std())
    lake = (slice(28, 32), slice(8, 13))
    pop_total[lake] = 0.2                                                            # a lake: nobody lives there
    smooth = lambda a: uniform_filter(a, size=3, mode="nearest")                     # about a 2 km neighbourhood
    lights = np.clip(2.5 * urban ** 1.5 + 0.2 * noise(1) / noise(1).std() + 0.05, 0, None)
    layers = {
        "population": smooth(pop_total) / 100,
        "built_up_surface": smooth(15 + 600 * urban ** 2 + 20 * rng.random((SIZE, SIZE))),
        "night_lights_bm": smooth(lights),
        "travel_time_city": smooth(8 * dist_town + 5 * rng.random((SIZE, SIZE))),
        "travel_time_health": smooth(4 * dist_town + 8 * rng.random((SIZE, SIZE))),
        "slope": smooth(np.abs(8 + 5 * noise(2) / noise(2).std())),
        "elevation": smooth(1500 + 150 * noise(4) / noise(4).std()),
    }
    # hidden rule: grid follows lights, solar peaks at mid-distance from town, unserved is remote and dark
    z = (layers["night_lights_bm"] - 0.3) / 0.3
    remote = (layers["travel_time_city"] - 90) / 60
    scores = np.stack([1.2 * z - 0.9, 0.6 - 0.5 * (remote - 0.2) ** 2 - 0.2 * z, 0.2 + 0.5 * remote - 0.3 * z], axis=-1)
    true = softmax(scores.reshape(-1, 3))

    side = SIZE // DISTRICTS_PER_SIDE
    district = (yy // side).astype(int) * DISTRICTS_PER_SIDE + (xx // side).astype(int)
    cells = pd.DataFrame({
        "cell_id": np.arange(SIZE * SIZE), "district": [f"District {d + 1:02d}" for d in district.ravel()],
        "row": yy.ravel().astype(int), "col": xx.ravel().astype(int),
        "population_total": pop_total.ravel(),
        **{f"{f}_r2km": layers[f].ravel() for f in FEATURES},
        **{f"true_share_{c}": true[:, k] for k, c in enumerate(CLASSES)}})
    return cells


def make_clusters(cells: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fake survey: cells are picked in proportion to population, 25 households each answer from the true shares."""
    pick = rng.choice(len(cells), size=N_CLUSTERS, replace=False, p=cells["population_total"] / cells["population_total"].sum())
    chosen = cells.iloc[pick].reset_index(drop=True)
    true = chosen[[f"true_share_{c}" for c in CLASSES]].to_numpy()
    counts = np.array([rng.multinomial(HOUSEHOLDS_PER_CLUSTER, p) for p in true])
    return pd.DataFrame({
        "cluster": np.arange(1, N_CLUSTERS + 1), "n_households": HOUSEHOLDS_PER_CLUSTER,
        **{f"share_{c}": counts[:, k] / HOUSEHOLDS_PER_CLUSTER for k, c in enumerate(CLASSES)},
        "district": chosen["district"], **{f: chosen[f"{f}_r2km"] for f in FEATURES}})


def write_demo(out_dir, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cells = make_country(rng)
    clusters = make_clusters(cells, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters.round(5).to_csv(out_dir / "demo_train_table.csv", index=False)
    cells.drop(columns=["row", "col"]).round(5).to_csv(out_dir / "demo_features_1km.csv", index=False)
    cells[["cell_id", "row", "col", "district"]].to_csv(out_dir / "demo_grid_cells.csv", index=False)
    spec = {"country": "demo (synthetic)", "crs": "EPSG:32735", "resolution_m": 1000, "xmin": 800000.0,
            "ymax": 9800000.0, "width": SIZE, "height": SIZE, "n_cells_inside": SIZE * SIZE}
    (out_dir / "demo_grid_spec.json").write_text(json.dumps(spec, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=None, help="default: model.seed in config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["demo"])
    write_demo(out_dir, cfg["model"]["seed"] if args.seed is None else args.seed)
    print(f"Wrote the synthetic demo dataset to {out_dir}")


if __name__ == "__main__":
    main()
