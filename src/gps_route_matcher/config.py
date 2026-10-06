"""Configuration dataclass for GPS map matching."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchConfig:
    """All tuneable parameters for the map-matching pipeline.

    Column names default to lowercase; override if your CSV uses different names
    (e.g. timestamp_col="Timestamp", lon_col="Longitude", lat_col="Latitude").
    """

    # ── Input column names ──────────────────────────────────────────────────
    timestamp_col: str = "Timestamp"
    lon_col: str = "Longitude"
    lat_col: str = "Latitude"

    # ── GPS noise / candidate generation ───────────────────────────────────
    # 1-σ GPS position error [m]. Deliberately far below the empirical residual
    # (median ~5 m, p90 ~16 m): emission is d²/(2σ²) while the network detour
    # term is |net - straight|/σ, so a small σ keeps geometry dominant over
    # routing. Raising σ to the "correct" value lets the transition term take
    # over and the matcher starts preferring shortest paths that backtrack over
    # the route actually walked. Tuned empirically — do not raise it without
    # re-checking that behaviour.
    gps_sigma_m: float = 0.3
    candidate_radius_m: float = 50.0   # max distance to consider a candidate
    candidate_k: int = 30               # max candidates per observation

    # ── Node avoidance (network matcher only) ───────────────────────────────
    # Shapely's LineString.project() clamps to [0, length], so without an inset
    # every edge meeting at a node contributes a candidate at exactly that node,
    # with an identical position and an identical emission cost. The emission
    # term then says nothing about which street the point is on, the tie falls
    # through to the transition term, and spatially distinct fixes collapse onto
    # one coordinate. Candidate offsets are therefore clamped into
    # [inset, length - inset], so a candidate never sits on a node.
    node_inset_m: float = 2.0
    # …but the inset never eats more than this share of a short edge, which keeps
    # the interval non-empty without special-casing (10% of the walk graph is
    # under 15 m, 2% under 5 m).
    max_node_inset_fraction: float = 0.05

    # ── Along-edge candidate densification (network matcher only) ───────────
    # Edges belonging to the inferred route get several candidate positions
    # instead of just their single nearest projection, so a continuous match on
    # the correct edge can win over a jump to a side street near a junction.
    # These extra positions are strictly farther from the GPS fix than that
    # edge's own nearest projection, so they never improve its best emission
    # score — they only win when continuity demands it.
    #
    # HOW THE THREE KNOBS INTERACT — change them together, not individually:
    #
    #   * route_candidates_per_edge and candidate_spacing_m set the along-edge
    #     window searched per route edge:
    #         half-window = (route_candidates_per_edge // 2) * candidate_spacing_m
    #     At the defaults that is (5 // 2) * 10 = 20 m either side of the
    #     projection. Widen it by raising either knob; raising the spacing
    #     widens the window but coarsens it, raising the count keeps the
    #     resolution and costs candidates.
    #
    #   * candidate_k must be large enough to actually hold them. Candidates
    #     from all nearby edges are pooled, sorted by distance and cut to
    #     candidate_k, so the budget needed per observation is roughly
    #         (# nearby route edges) * route_candidates_per_edge
    #         + (# nearby non-route edges)
    #     Within candidate_radius_m = 50 m of a fix there are typically 1-3
    #     route edges and 5-10 other edges, i.e. ~3*5 + 10 = 25 — hence
    #     candidate_k = 30. Set candidate_k too low and the cut silently drops
    #     the farthest densified positions first, which is exactly the
    #     densification you were paying for.
    #
    #   * Viterbi is O(candidate_k²) per observation, so candidate_k is the
    #     cost driver: ~0.3 s per 1000-point trip at k=13, ~1.5 s at k=30.
    #
    # Densification only applies when the caller passes route_edges (i.e. in
    # scripts/map_matching.py, where the inferred route already exists);
    # route inference itself runs with one candidate per edge.
    route_candidates_per_edge: int = 5
    candidate_spacing_m: float = 10.0

    # ── Timestamp inference ─────────────────────────────────────────────────
    expected_sample_interval_s: float | None = None
    infer_missing_timestamps: bool = True
    # Whether a row with no timestamp may be filled by extrapolating past the
    # first/last valid timestamp of its recording, or only by interpolating
    # between two valid ones. Off by default: the rows before a recording's
    # first valid timestamp are the device's pre-RTC-sync warm-up block, which
    # has no GPS fix either (97% of them are (0,0)), so extrapolating backwards
    # only invents trip membership for rows recorded before the walk began.
    # Interior dropouts, by contrast, are bracketed on both sides and do carry
    # real sensor readings worth recovering.
    extrapolate_missing_timestamps: bool = False

    # ── Gap / segment handling ──────────────────────────────────────────────
    max_split_gap_s: float = 300.0     # gaps > this force a new HMM segment
    max_speed_mps: float = 2.5         # ~9 km/h; used in transition penalty

    # ── Cross-segment gap bridging (network matcher only) ───────────────────
    # A gap between two HMM segments is only bridged with an inferred network
    # path when both segments have enough support and the bridge itself is
    # short; otherwise the gap is left as a genuine break in the edge sequence
    # rather than inventing a long stretch of unobserved road.
    min_segment_points_to_bridge_gap: int = 2   # segments shorter than this never bridge
    max_gap_bridge_m: float = 20.0              # bridges longer than this are left open

    # Flat cost added when a transition reverses direction on the same physical
    # street — either a backward offset move on the same directed edge, or a
    # switch to that edge's paired reverse-direction edge (u,v,key) -> (v,u,key).
    # Mirrors backward_penalty's magnitude/spirit for the network matcher, where
    # there's no single "forward" direction to check delta_s against.
    network_reversal_penalty: float = 10000.0

    # ── Transition speed penalty (network matcher) ───────────────────────────
    high_speed_penalty: float = 1000.0

    # ── Post-HMM interpolation ──────────────────────────────────────────────
    max_interpolation_gap_s: float = 300.0

    # ── Cleaning ────────────────────────────────────────────────────────────
    treat_zero_zero_as_missing: bool = True
    # Which checks run is decided by what the caller passes to
    # cleaning.prepare_for_matching, not by a flag: a `route` enables the
    # route-buffer and walk-time-window checks, a `polygon` enables the
    # bairro-membership check. (There used to be a single check_oob flag
    # gating all three; it conflated concerns that become available at
    # different pipeline stages — the route only exists after infer_route —
    # and made check_oob=True with route=None crash.)
    oob_buffer_m: float = 35.0  # max distance from the inferred route before out_of_bounds is set

    # ── Timestamp sequentiality / outlier detection ──────────────────────────
    # Primary anchor is each raw recording's own filename (sessions.py), which
    # encodes the true capture date/hour and is written at export time — after
    # the RTC has synced — while the bad rows are logged before it. The
    # absolute window below is only a loose outer backstop for recordings whose
    # filename doesn't parse; it is deliberately not a tight, hand-maintained
    # range (valid_timestamp_max used to be, and silently expired, rejecting
    # every timestamp of every recording captured after it).
    detect_timestamp_outliers: bool = True
    valid_timestamp_min: str | None = "2025-10-01"
    # None means "now + max_future_skew_h" — a recording cannot have been
    # captured in the future, and a bound derived at runtime cannot expire. Since data colletion in this project is in the past, a hardcoded expiry date is fine.
    valid_timestamp_max: str | None = "2026-05-01"
    max_future_skew_h: float = 24.0
    timestamp_outlier_mad_k: float = 3.0
    min_timestamps_for_sequentiality_check: int = 5
    # How far a row's timestamp may sit from its recording filename's declared
    # capture time before it's flagged, and how far the file's own median may
    # sit from it before the filename stops being trusted as an anchor at all
    # (some exports are named for the day they were downloaded). Generous: the
    # filename carries a whole hour, not a minute, and a recording can
    # legitimately run long.
    recording_date_tolerance_h: float = 12.0

    # ── Trip assembly (sessions.py, scripts/make_trip_data.py) ──────────────
    # A walk is a contiguous run of samples, so trips are cut on time gaps and
    # only then labelled by their own start hour. Labelling per row instead
    # would split a single walk that happens to cross a window edge.
    session_split_gap_s: float = 1800.0   # 30 min of silence ends a session
    # A walk also ends when the walking stops, which silence alone doesn't
    # capture: a device left running between the morning and afternoon walk
    # never goes quiet, so gap-cutting alone returns both walks and the four
    # idle hours between them as one session. Samples that stay within
    # stationary_radius_m for longer than max_stationary_gap_s are idle time,
    # not walk data — they end the session and are quarantined rather than
    # attributed to a route. The radius is well above GPS jitter at rest; the
    # duration is well above any pause taken mid-walk.
    stationary_radius_m: float = 25.0
    max_stationary_gap_s: float = 1200.0        # 20 min
    # Backstop for what neither rule catches — a sensor in transit still moves
    # continuously. Recorded walks run to ~75 min at the outside, so anything
    # past a few hours is something else. (One real OM_2 session runs 33 h
    # with its GPS wandering a 350 km box.)
    max_session_duration_s: float = 10800.0     # 3 h
    morning_hours: tuple[int, int] = (11, 14)   # [start, end) hour
    afternoon_hours: tuple[int, int] = (17, 20)   # [start, end) hour


@dataclass
class RouteInferenceConfig:
    """Parameters for ground-truth route inference (Steps 3–5)."""

    gamma: float = 2.0                  # frequency dominance in the edge cost
    endpoint_cluster_eps_m: float = 75.0   # radius for endpoint clustering
    endpoint_min_cluster_size: int = 3
    # Optional explicit overrides — when set, clustering is skipped.
    start_latlon: tuple[float, float] | None = None   # (lat, lon)
    end_latlon: tuple[float, float] | None = None
    start_node: int | None = None
    end_node: int | None = None
    # Optional via point for loop routes (start == end). When None and the
    # route is a loop, the densest cluster of trip midpoints is used.
    via_latlon: tuple[float, float] | None = None
    via_node: int | None = None
    # Trips whose matched length is below this are excluded from endpoint /
    # midpoint clustering (they are usually stationary recordings at a base).
    # They still contribute to edge traversal counts.
    min_trip_length_m: float = 500.0
