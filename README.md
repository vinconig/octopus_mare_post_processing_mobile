# DenoiseGPS

A GPS map-matching pipeline for wearable air-quality sensor recordings (OM_1..OM_4) walked through a Rio de Janeiro bairro. Raw, noisy lat/lon + timestamp CSVs are cleaned, snapped onto the real road network (OpenStreetMap via OSMnx) using an HMM/Viterbi matcher, aggregated per road edge, and rendered as interactive HTML maps.

## Setup

```bash
pip install -e ".[dev,viz,gdrive]"
```

- `dev` — pytest
- `viz` — plotly (interactive maps)
- `gdrive` — Google Drive download script (`scripts/download_gdrive_data.py`)

## Usage

```bash
# Run the full pipeline (download → trip-splitting → route inference → map matching → edge analysis → maps)
python scripts/main.py

# Run all tests
python -m pytest

# Run a single test file / test
python -m pytest tests/test_hmm.py
python -m pytest tests/test_hmm.py::test_name -q
```

There is no lint/format command configured in this repo.

## Architecture

### Two matchers, shared building blocks

The core distinction driving the whole `src/gps_route_matcher/` package is **single-route matching vs. network matching**:

- **Single-route matcher** (`candidates.py` + `hmm.py`, orchestrated by `pipeline.py`) — states are 1-D arc-length positions `s_m` on one predefined polyline (`route.py`'s `Route1D`). Used to match a recording against one already-known GPX route.
- **Network matcher** (`netmatch.py`) — states are `(edge, offset)` positions on any edge of a directed `networkx` road graph (`roadnet.py`, via OSMnx). Used both for inferring the ground-truth route from many trips (`route_inference.py`) and for the final per-trip matching in `scripts/map_matching.py`.

Both share: candidate generation logic (project GPS point onto nearby segments/edges), the Viterbi cost model (emission = squared distance / 2σ²; transition = backward-travel penalty + high-speed penalty using *real* elapsed time, not a fixed step), gap detection/segmentation (`gaps.py`), post-match interpolation in arc-length space (`interpolation.py`), and input cleaning/validation (`cleaning.py`).

All tunable parameters for both matchers live in one place: `src/gps_route_matcher/config.py` (`MatchConfig` dataclass) — GPS sigma, candidate radius/k, gap thresholds, speed/backward/reversal penalties, timestamp-outlier detection window, out-of-bounds buffer, etc.

### End-to-end pipeline (`scripts/main.py`)

Each stage reads the previous stage's output directory and writes a new one; folders are the unit of work (`OM_1_trips`, `OM_2_trips`, ...):

1. **`download_gdrive_data.py`** — pulls raw per-recording CSVs from Google Drive (OAuth; run once, then commented out).
2. **`make_trip_data.py`** — concatenates raw per-sensor CSVs (`data/OM_n_raw/` → `data/OM_n_concatenated.csv`), then splits by date + morning/evening time window into one CSV per trip (`data/OM_n_trips/`).
3. **`infer_route.py`** — HMM-matches all trips for a folder onto the OSM road graph, counts directed-edge traversals, clusters trip endpoints (or uses manually supplied lat/lon endpoints), and computes the highest-support path via Dijkstra with a `length / (count+1)^gamma` cost. Exports a GPX (`routes/Inferred_Route_*.gpx`) and route metadata (`routes/route_metadata/`). Only needs to redownload/cache the road graph once (`force_refresh` on the first folder).
4. **`map_matching.py`** — network-matches every raw trip CSV onto the road graph, writing enriched CSVs to `data/OM_n_trips_matched/` (adds matched edge, timestamp inference, match status, interpolation, etc.).
5. **`route_edge_analysis.py`** — aggregates point-level matched CSVs into one row per road-network edge (in route order) with per-edge environmental summary stats, writing to `data/OM_n_trips_edge_analysis/`. Column classification for "environmental vs. pipeline metadata" is centralized in `analysis.py`'s `_NON_ENV_COLUMNS` — new raw sensor columns are picked up automatically as long as they're not added to that set.
6. **`generate_map_info.py`** — computes map pan/zoom info.
7. **`visualization.py`** — renders interactive Plotly/Leaflet HTML maps to `output/`. `generate_interactive_map_all_data_incl_edges` is the current main data panel (single map with all trip data + edge analysis); it superseded and merged the older `generate_interactive_map_all_edge_data` / `generate_interactive_map_all_data` (kept as separate functions but no longer both driven from `main.py`).

### Filesystem conventions

- `paths.PROJECT_ROOT` is derived from the installed package location (`src/gps_route_matcher/paths.py`), not cwd — always import it rather than hardcoding paths.
- `roadnet.py` caches the downloaded OSM graph under `roadnetwork/`; `route_inference.py` reads/writes `routes/`.
- Recording filenames follow one of two conventions, parsed centrally by `recording_naming.py` (used by both `visualization.py` and `analysis.py` so the regex logic isn't duplicated):
  - `OM_n_YYYYMMDD_HH_Ddurmin__RECrecid.csv` — raw per-recording exports (`data/OM_n_raw/`)
  - `OM_n_YYYYMMDD_<morning|evening>_Ddurmin.csv` — one file per trip (`data/OM_n_trips/`)
- `data/`, `output/`, `cache/`, and `NotebooksCassiano/` are gitignored, as are all `*.csv`/`*.html` files anywhere — treat them as regenerable/local-only, never expect them to be committed.
- `credentials.json` / `token.json` (Google Drive OAuth) exist at repo root and are gitignored — never commit them.

## Tests

`tests/` mirrors the `src/gps_route_matcher/` modules (`test_hmm.py`, `test_candidates.py`, `test_cleaning.py`, `test_netmatch.py`, `test_route_inference.py`, `test_route_projection.py`, `test_interpolation.py`, `test_analysis.py`). No test touches live network/OSM downloads or Google Drive — those integration paths (`roadnet.py`, `download_gdrive_data.py`) are exercised only via `scripts/`.
