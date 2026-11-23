# octopus_mare_post_processing_mobile

Processing code for the mobile part of an environmental sensing campaign in Maré, a complex of 15 favelas in Rio de Janeiro, Brazil. Four sensing platforms (Octopus, developed by MIT) were walked along 4 predefined routes. This repository turns their raw recordings into individual walk files, cleans the timestamps and map-matches the GPS positions onto the real road network (OpenStreetMap via OSMnx) with an HMM/Viterbi map matcher. The matched walks are then aggregated per road edge and packaged into the deposit folder `output/zenodo_curated/`.

The calibration of the environmental readings and the fixed data stream of the campaign are not part of this repository; they are processed elsewhere.

The released dataset is *Participatory pedestrian and fixed-site environmental sensing data from Complexo da Maré, Rio de Janeiro* (Jansen, Bastos Moroz, Schlüter, Mora, Ratti): [doi.org/10.5281/zenodo.23105404](https://doi.org/10.5281/zenodo.23105404).

**The raw recordings are not available.** They contain locations that identify private addresses, so they cannot be shared, and the pipeline cannot be re-run from the start without them. What is public is the released dataset (the map-matched walks and per-edge aggregates) and this code. The code can be read and audited, its tests run without any data, and everything that was decided by hand (route endpoints, matching settings, release curation) is committed alongside it.

This repository is for three uses:

- **Reviewing** how the released data came from the raw recordings: read [Pipeline](#pipeline) and the code of each stage.
- **Understanding the method:** every tunable parameter is in `src/gps_route_matcher/config.py`, and the cleaning and matching logic is in `src/gps_route_matcher/`.
- **Building a similar pipeline for your own GPS recordings:** see [Adapting it to your own data](#adapting-it-to-your-own-data).

## Pipeline

`scripts/main.py` runs every stage in order. Each stage reads the previous stage's folder and writes the next one. Folders are named `OM_n_<letter>_<stage>`, the letter encoding processing order (`a_raw` → `b_walks` → `c_prepared` → `d_matched` → `e_edge_analysis`); the names and the track list are defined once in `src/gps_route_matcher/paths.py`.

1. **`make_trip_data.py`**: validates each raw recording's timestamps, concatenates them per sensor, and cuts the stream into one CSV per walk (`data/OM_n_b_walks/`). Rows that reach no walk go to `data/OM_n_quarantine.csv` with a reason.
2. **`prepare_trips.py`**: route-independent GPS cleaning, as flag columns only (`data/OM_n_c_prepared/`); no rows are dropped.
3. **`infer_route.py`**: HMM-matches all walks of a track onto the road graph, counts directed-edge traversals and finds the highest-support path between the route's endpoints. Writes `routes/Inferred_Route_OM_n.gpx` and `routes/route_metadata/OM_n_inferred_route.json`. The endpoints are set by hand in `scripts/main.py`.
4. **`map_matching.py`**: matches every walk onto the road graph against its inferred route, with per-track settings from `config.FINAL_MATCH_OVERRIDES` and the hand-picked route edges in `curation/edge_bonus.json` (`data/OM_n_d_matched/`).
5. **`route_edge_analysis.py`**: aggregates the matched points into one row per route edge with per-edge summary statistics (`data/OM_n_e_edge_analysis/`).
6. **`package_for_zenodo.py`**: keeps the walks inside the export date window, applies the shipped column whitelist (`zenodo_columns.py`) and copies the route metadata and the road-graph snapshot (`output/zenodo_package/`).
7. **`apply_curation.py`**: applies the hand-made keep/discard/trim/cut decisions in `curation/release_curation.json` (`output/zenodo_curated/`, the folder that was uploaded).

All tunable matching parameters live in `src/gps_route_matcher/config.py` (`MatchConfig`).

## Committed inputs

These are the hand-made inputs behind the release. They are judgement, not output, so they cannot be regenerated and are committed.

| Path | What |
| --- | --- |
| `roadnetwork/mare_openstreetmap_road_graph_2026-11-04.graphml` | The OSM road-network snapshot every edge id refers to. A missing file is an error; the pipeline never downloads a graph. |
| `borders/mare_outline.kml` | Outline of the studied area (Maré). |
| `routes/` | The inferred route of each track (GPX plus edge list). Later stages read `routes/route_metadata/` as required input. |
| `curation/edge_bonus.json` | Hand-picked route edges that get a matching bonus. |
| `curation/release_curation.json` | Hand-made keep/discard/trim/cut decisions for the released walks. |

## Setup and tests

Python 3.11. Dependencies are pinned exactly, because reproducing the released files byte for byte would depend on the numerical behaviour of these versions.

```bash
pip install -e ".[dev]"
python -m pytest
```

`tests/` mirrors the modules in `src/gps_route_matcher/`. No test needs the network or the raw data.

## Re-running the pipeline (requires the raw recordings)

Only possible with your own recordings in the expected format; see the next section. Place them, one folder per sensor, in `data/OM_n_a_raw/`, then:

```bash
# Run the whole pipeline, raw recordings -> output/zenodo_curated/
python scripts/main.py
```

Every stage is also a script of its own (`python scripts/<stage>.py --help`); each takes `--folders OM_2 ...` to restrict it to some tracks.

## Adapting it to your own data

The code was written for this specific campaign, so reusing it means editing a handful of campaign-specific settings, not just changing a config file. Here is everything you would have to replace.

**Input format**
- One folder per sensor, `data/<track>_a_raw/`, holding one CSV per recording. Recordings are read with an auto-detected delimiter.
- Required columns are `Timestamp`, `Latitude` and `Longitude`. The names are settings (`timestamp_col`, `lat_col`, `lon_col` in `MatchConfig`). Other columns are carried through; the sensor columns that were released (`Temperature`, `Humidity`, `PM2.5`) are listed in `src/gps_route_matcher/zenodo_columns.py`.
- File names must look like `OM_n_YYYYMMDD_HH_Ddurmin__RECrecid.csv`. The date and hour in the name are used to cross-check each recording's timestamps. The pattern is in `src/gps_route_matcher/recording_naming.py`, and the track prefix is hard-coded as `OM_<number>`.

**Campaign-specific settings**
| What | Where |
| --- | --- |
| Track list | `TRACKS` in `src/gps_route_matcher/paths.py` |
| Study-area outline | `borders/mare_outline.kml` (path in `paths.py`) |
| Road network (an OSMnx `.graphml`; never downloaded) | `ROAD_GRAPH_PATH` in `src/gps_route_matcher/roadnet.py` |
| Manual route start and end points per track | `manual_endpoints` in `scripts/main.py` (a track left out falls back to automatic start/end clustering) |
| Matching parameters, defaults and per-track overrides | `MatchConfig` and `FINAL_MATCH_OVERRIDES` in `src/gps_route_matcher/config.py` |
| Accepted timestamp range | `valid_timestamp_min` / `valid_timestamp_max` in `MatchConfig`. These are fixed to this campaign's dates. |
| Hand-picked route edges | `curation/edge_bonus.json` |
| Export date window | `_EXPORT_DATE_MIN` / `_EXPORT_DATE_MAX` in `scripts/package_for_zenodo.py` |
| Shipped columns, renames and the timestamp shift to local time | `src/gps_route_matcher/zenodo_columns.py` |
| Keep/discard/trim/cut decisions | `curation/release_curation.json`. These are specific to the released walks. |

The matching itself (`netmatch.py`, `route_inference.py`) does not depend on the sensors, so it is the part most likely to carry over. Expect to re-tune the matching parameters for your own GPS quality; the per-track overrides in `config.py` show how much they differed between the four sensors here.

## License and citation

The code is released under the [MIT License](LICENSE). To cite it, use the metadata in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button reads it), and cite the dataset via its DOI above.
