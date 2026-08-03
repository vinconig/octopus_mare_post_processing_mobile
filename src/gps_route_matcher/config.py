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
    gps_sigma_m: float = 0.5
    candidate_radius_m: float = 50.0   # max distance to consider a candidate
    candidate_k: int = 30               # max candidates per observation
    deduplicate_s_tolerance_m: float = 5.0  # merge candidates this close in s

    # ── Node avoidance (network matcher only) ───────────────────────────────
    # Shapely's LineString.project() clamps to [0, length], so without an inset
    # every edge meeting at a node contributes a candidate at exactly that node,
    # with an identical position and an identical emission cost. The emission
    # term then says nothing about which street the point is on, the tie falls
    # through to the transition term, and spatially distinct fixes collapse onto
    # one coordinate. Candidate offsets are therefore clamped into
    # [inset, length - inset], so a candidate never sits on a node.
    node_inset_m: float = 5.0
    # …but the inset never eats more than this share of a short edge, which keeps
    # the interval non-empty without special-casing (10% of the walk graph is
    # under 15 m, 2% under 5 m).
    max_node_inset_fraction: float = 0.25

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

    # ── Gap / segment handling ──────────────────────────────────────────────
    max_bridge_gap_s: float = 90.0     # gaps ≤ this are bridged within one segment
    max_split_gap_s: float = 300.0     # gaps > this force a new HMM segment
    max_speed_mps: float = 2.5         # ~7.2 km/h; used in transition penalty

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

    # ── HMM transition model ────────────────────────────────────────────────
    allow_backward: bool = False
    backward_penalty: float = 10000.0
    high_speed_penalty: float = 1000.0

    # ── Post-HMM interpolation ──────────────────────────────────────────────
    max_interpolation_gap_s: float = 300.0

    # ── Cleaning ────────────────────────────────────────────────────────────
    treat_zero_zero_as_missing: bool = True
    check_oob: bool = True    # run inspect_gps_oob (+ categorize_oob/inspect_gps_ootb if polygon given)
    oob_buffer_m: float = 35.0  # max distance from the inferred route before out_of_bounds is set

    # ── Timestamp sequentiality / outlier detection ──────────────────────────
    detect_timestamp_outliers: bool = True
    valid_timestamp_min: str | None = "2025-09-01"
    valid_timestamp_max: str | None = "2026-08-01"
    timestamp_outlier_mad_k: float = 3.0
    min_timestamps_for_sequentiality_check: int = 5
