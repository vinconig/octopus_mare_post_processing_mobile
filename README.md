# DenoiseGPS

Code that turns raw recordings from four wearable air-quality sensors (OM_1..OM_4), walked through a Rio de Janeiro bairro, into the released dataset. Raw, noisy lat/lon + timestamp CSVs are cleaned, snapped onto the real road network (OpenStreetMap via OSMnx) with an HMM/Viterbi map matcher, aggregated per road edge, and packaged into the deposit folder `output/zenodo_curated/`.

## Setup

Python 3.11. Dependencies are pinned exactly, because reproducing the released files byte for byte depends on the numerical behaviour of these versions.

```bash
pip install -e ".[dev]"
```

## Input data

The raw per-recording exports are not part of this repository. Place them, one folder per sensor, in:

```
data/OM_1_a_raw/   data/OM_2_a_raw/   data/OM_3_a_raw/   data/OM_4_a_raw/
```

with file names of the form `OM_n_YYYYMMDD_HH_Ddurmin__RECrecid.csv`.

[PLACEHOLDER: where the raw recordings can be obtained.]

## Usage

```bash
# Run the whole pipeline, raw recordings -> output/zenodo_curated/
python scripts/main.py

# Run all tests
python -m pytest
```

Every stage is also a script of its own (`python scripts/<stage>.py --help`); each takes `--folders OM_2 ...` to restrict it to some tracks.

## Pipeline

Each stage reads the previous stage's folder and writes the next one. Folders are named `OM_n_<letter>_<stage>`, the letter encoding processing order (`a_raw` → `b_walks` → `c_prepared` → `d_matched` → `e_edge_analysis`); the names and the track list are defined once in `src/gps_route_matcher/paths.py`.

1. **`make_trip_data.py`** — validates each raw recording's timestamps, concatenates them per sensor, and cuts the stream into one CSV per walk (`data/OM_n_b_walks/`). Rows that reach no walk go to `data/OM_n_quarantine.csv` with a reason.
2. **`prepare_trips.py`** — route-independent GPS cleaning, as flag columns only (`data/OM_n_c_prepared/`); no rows are dropped.
3. **`infer_route.py`** — HMM-matches all walks of a track onto the road graph, counts directed-edge traversals and finds the highest-support path between the route's endpoints. Writes `routes/Inferred_Route_OM_n.gpx` and `routes/route_metadata/OM_n_inferred_route.json`. The endpoints are set by hand in `scripts/main.py`.
4. **`map_matching.py`** — matches every walk onto the road graph against its inferred route, with per-track settings from `config.FINAL_MATCH_OVERRIDES` and the hand-picked route edges in `curation/edge_bonus.json` (`data/OM_n_d_matched/`).
5. **`route_edge_analysis.py`** — aggregates the matched points into one row per route edge with per-edge summary statistics (`data/OM_n_e_edge_analysis/`).
6. **`package_for_zenodo.py`** — keeps the walks inside the export date window, applies the shipped column whitelist (`zenodo_columns.py`) and copies the route metadata and the road-graph snapshot (`output/zenodo_package/`).
7. **`apply_curation.py`** — applies the hand-made keep/discard/trim/cut decisions in `curation/release_curation.json` (`output/zenodo_curated/`, the folder to upload).

All tunable matching parameters live in `src/gps_route_matcher/config.py` (`MatchConfig`).

## Committed inputs

| Path | What |
| --- | --- |
| `roadnetwork/mare_openstreetmap_road_graph_2026-11-04.graphml` | The OSM road-network snapshot every edge id refers to. A missing file is an error; the pipeline never downloads a graph. |
| `borders/Bairro.kml` | Boundary of the studied area. |
| `curation/edge_bonus.json` | Hand-picked route edges that get a matching bonus. |
| `curation/release_curation.json` | Hand-made keep/discard/trim/cut decisions for the released walks. |

## Tests

`tests/` mirrors the modules in `src/gps_route_matcher/`. No test needs the network or the raw data.
