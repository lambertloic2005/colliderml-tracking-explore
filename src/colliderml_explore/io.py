"""Loading helpers for ColliderML tracking data."""
from typing import Iterable

import polars as pl
from colliderml.core import load_tables, collect_tables
from colliderml.polars import explode_particles, explode_tracker_hits


def load_event_tables(
    channels: str = "ttbar",
    pileup: str = "pu0",
    objects: Iterable[str] = ("particles", "tracker_hits", "tracks"),
    max_events: int | None = 500,
    split: str = "train",
) -> dict:
    """Load event-as-row tables (one row per event, list columns inside)."""
    cfg = {
        "dataset_id": "CERN/ColliderML-Release-1",
        "channels": channels,
        "pileup": pileup,
        "objects": list(objects),
        "split": split,
        "lazy": False,
        "max_events": max_events,
    }
    return collect_tables(load_tables(cfg))


def _to_polars(df) -> pl.DataFrame:
    """Accept either pandas or polars; always return polars."""
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


def _explode_tracks(df: pl.DataFrame) -> pl.DataFrame:
    """Explode the tracks table (no helper ships for this one)."""
    list_cols = [c for c in df.columns if c != "event_id"]
    return df.explode(list_cols)


def load_flat(
    channels: str = "ttbar",
    pileup: str = "pu0",
    max_events: int | None = 500,
) -> dict[str, pl.DataFrame]:
    """Load tracking tables and explode to object-as-row polars DataFrames."""
    frames = load_event_tables(channels, pileup, max_events=max_events)
    return {
        "particles":    _to_polars(explode_particles(frames["particles"])),
        "tracker_hits": _to_polars(explode_tracker_hits(frames["tracker_hits"])),
        "tracks":       _to_polars(_explode_tracks(_to_polars(frames["tracks"]))),
    }