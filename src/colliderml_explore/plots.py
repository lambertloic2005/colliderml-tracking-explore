# ---------------- Truth: hits-per-particle and impact parameters ----------------

def hist_num_tracker_hits(df: pl.DataFrame, ax=None, label=None,
                          max_hits=25, threshold=7):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    n = df["num_tracker_hits"].to_numpy()
    ax.hist(n, bins=range(0, max_hits + 1),
            histtype="step", linewidth=1.5, label=label)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"≥ {threshold} hits (typical reco threshold)")
    ax.set_xlabel("Number of tracker hits")
    ax.set_ylabel("Count")
    ax.legend()
    return ax


def hist_impact_param(df: pl.DataFrame, ax=None, kind="d0",
                      label=None, range_=(-5, 5), bins=100):
    """Histogram perigee_d0 (transverse) or perigee_z0 (longitudinal), in mm."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    col = f"perigee_{kind}"
    vals = df[col].drop_nulls().to_numpy()
    ax.hist(vals, bins=bins, range=range_,
            histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel(f"perigee {kind} [mm]")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    if label:
        ax.legend()
    return ax


# ---------------- Hits: occupancy and resolution ----------------

def hist_hits_per_event(hits: pl.DataFrame, ax=None, label=None, bins=40):
    counts = hits.group_by("event_id").len()["len"].to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.hist(counts, bins=bins, histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel("Tracker hits per event")
    ax.set_ylabel("Events")
    if label:
        ax.legend()
    return ax


def hist_residual(hits: pl.DataFrame, ax=None, axis="x",
                  range_=(-0.2, 0.2), bins=100, label=None):
    """Digitization residual: measured - true position, in mm."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    res = (
        hits.with_columns(res=pl.col(axis) - pl.col(f"true_{axis}"))
        ["res"].to_numpy()
    )
    ax.hist(res, bins=bins, range=range_,
            histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel(f"{axis} - true_{axis} [mm]")
    ax.set_ylabel("Count")
    if label:
        ax.legend()
    return ax


def bar_hits_by_volume(hits: pl.DataFrame, ax=None):
    """Bar chart of tracker hits grouped by volume_id."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    counts = hits.group_by("volume_id").len().sort("volume_id")
    ax.bar(counts["volume_id"].to_list(), counts["len"].to_list())
    ax.set_xlabel("volume_id")
    ax.set_ylabel("Hit count")
    ax.set_title("Tracker hit occupancy by detector volume")
    return ax