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
    gps_sigma_m: float = 10.0          # 1-σ GPS position error [m]
    candidate_radius_m: float = 50.0   # max distance to consider a candidate
    candidate_k: int = 5               # max candidates per observation
    deduplicate_s_tolerance_m: float = 5.0  # merge candidates this close in s

    # ── Timestamp inference ─────────────────────────────────────────────────
    expected_sample_interval_s: float | None = None
    infer_missing_timestamps: bool = False

    # ── Gap / segment handling ──────────────────────────────────────────────
    max_bridge_gap_s: float = 90.0     # gaps ≤ this are bridged within one segment
    max_split_gap_s: float = 300.0     # gaps > this force a new HMM segment
    max_speed_mps: float = 6.0         # ~22 km/h; used in transition penalty

    # ── HMM transition model ────────────────────────────────────────────────
    allow_backward: bool = False
    backward_penalty: float = 1000.0
    high_speed_penalty: float = 1000.0

    # ── Post-HMM interpolation ──────────────────────────────────────────────
    max_interpolation_gap_s: float = 300.0

    # ── Cleaning ────────────────────────────────────────────────────────────
    treat_zero_zero_as_missing: bool = True
