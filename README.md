# colliderml-tracking-explore

Exploratory plots for the ColliderML dataset, focused on the tracking
pipeline: truth particles → tracker hits → reconstructed tracks.

## Setup

````bash
uv sync
````

## Download a subset

````bash
uv run colliderml download \
    --channels ttbar --pileup pu0 \
    --objects particles,tracker_hits,tracks \
    --max-events 500
````

## Layout

- `src/colliderml_explore/` — reusable loading, derived quantities, plotting
- `notebooks/` — exploration notebooks (01 particles, 02 hits, 03 tracks, 04 matching)
- `scripts/` — one-shot scripts (download, regenerate figures)
- `data/` — local data cache (gitignored)
- `figures/` — output plots
