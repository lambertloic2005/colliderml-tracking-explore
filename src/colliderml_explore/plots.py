"""Reusable plotting helpers for ColliderML tracking exploration."""
import matplotlib.pyplot as plt
import mplhep as hep
import polars as pl
import plotly.graph_objects as go
import plotly.express as px

def setup_style():
    """Apply ATLAS-style plotting defaults. Call once at notebook start."""
    hep.style.use("ATLAS")


# ---------------- Particle-level (truth) ----------------

def hist_pt(df: pl.DataFrame, ax=None, label=None, bins=50, pt_max=50.0):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    pt = df["pt"].to_numpy()
    ax.hist(pt, bins=bins, range=(0, pt_max),
            histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    if label:
        ax.legend()
    return ax


def hist_eta(df: pl.DataFrame, ax=None, label=None, bins=60, eta_max=4.0):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    eta = df["eta"].drop_nulls().to_numpy()
    ax.hist(eta, bins=bins, range=(-eta_max, eta_max),
            histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Count")
    if label:
        ax.legend()
    return ax


def hist2d_pt_eta(df: pl.DataFrame, ax=None, bins=(60, 50),
                  pt_max=30.0, eta_max=4.0):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    sub = df.filter(pl.col("eta").is_not_null() & (pl.col("pt") < pt_max))
    h = ax.hist2d(
        sub["eta"].to_numpy(), sub["pt"].to_numpy(),
        bins=bins, range=[[-eta_max, eta_max], [0, pt_max]],
        cmin=1,
    )
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$p_T$ [GeV]")
    plt.colorbar(h[3], ax=ax, label="Count")
    return ax


def hist_pdg_composition(df: pl.DataFrame, ax=None, top_n=10):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    pdg_names = {
        11: "e", 13: r"$\mu$", 22: r"$\gamma$",
        211: r"$\pi^\pm$", 321: r"$K^\pm$",
        2212: "p", 2112: "n",
        130: r"$K^0_L$", 310: r"$K^0_S$", 111: r"$\pi^0$",
    }
    counts = (
        df.select(pl.col("pdg_id").abs().alias("apdg"))
        .group_by("apdg").len()
        .sort("len", descending=True)
        .head(top_n)
    )
    labels = [pdg_names.get(p, str(p)) for p in counts["apdg"].to_list()]
    ax.bar(labels, counts["len"].to_list())
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(f"Top {top_n} particle species")
    return ax


# ---------------- Hit-level (detector) ----------------

def scatter_rz(hits: pl.DataFrame, event_id: int, ax=None, color_by="volume_id"):
    """rho vs z scatter for a single event — *this draws the detector*."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    evt = hits.filter(pl.col("event_id") == event_id).with_columns(
        rho=(pl.col("x") ** 2 + pl.col("y") ** 2).sqrt(),
    )
    sc = ax.scatter(
        evt["z"].to_numpy(), evt["rho"].to_numpy(),
        c=evt[color_by].to_numpy(), s=3, cmap="tab10", alpha=0.7,
    )
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(r"$\rho = \sqrt{x^2 + y^2}$ [mm]")
    ax.set_title(f"Event {event_id} — tracker hits coloured by {color_by}")
    plt.colorbar(sc, ax=ax, label=color_by)
    return ax


def scatter_xy_barrel(hits: pl.DataFrame, event_id: int,
                      ax=None, abs_z_max=500.0):
    """x-y view of barrel hits — shows concentric layers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    evt = hits.filter(
        (pl.col("event_id") == event_id) & (pl.col("z").abs() < abs_z_max)
    )
    ax.scatter(
        evt["x"].to_numpy(), evt["y"].to_numpy(),
        c=evt["layer_id"].to_numpy(), s=5, cmap="tab20", alpha=0.7,
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_title(f"Event {event_id} — barrel hits ($|z| < {abs_z_max:.0f}$ mm)")
    return ax


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

# ---------------- Interactive 3D (plotly) ----------------

def scatter3d_hits(hits: pl.DataFrame, event_id: int,
                   color_by: str = "volume_id",
                   marker_size: int = 2,
                   opacity: float = 0.7,
                   max_points: int | None = 50_000):
    """Interactive 3D scatter of all tracker hits in one event (x, y, z).

    Rotate to view from any angle. The ρ-z "side view" is recovered by
    rotating until the camera looks along the x or y axis.
    """
    evt = hits.filter(pl.col("event_id") == event_id)
    if max_points is not None and evt.height > max_points:
        evt = evt.sample(n=max_points, seed=0)

    fig = go.Figure(data=[go.Scatter3d(
        x=evt["x"].to_numpy(),
        y=evt["y"].to_numpy(),
        z=evt["z"].to_numpy(),
        mode="markers",
        marker=dict(
            size=marker_size,
            color=evt[color_by].to_numpy(),
            colorscale="Viridis",
            opacity=opacity,
            colorbar=dict(title=color_by),
        ),
        hovertemplate=(
            "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}"
            "<br>" + color_by + f"=%{{marker.color}}"
            "<extra></extra>"
        ),
    )])
    fig.update_layout(
        title=f"Event {event_id} — tracker hits (3D, by {color_by})",
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title="z [mm]",
            aspectmode="data",  # don't distort axes
        ),
        width=900, height=700,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def scatter3d_hits_by_particle(hits: pl.DataFrame, event_id: int,
                               top_n_particles: int = 10,
                               connect_hits: bool = True):
    """3D scatter highlighting the top-N particles by hit count.

    Background hits are gray; each highlighted particle gets a distinct color.
    If connect_hits=True, hits along a particle are sorted by time and
    drawn as a polyline — approximates the trajectory.
    """
    evt = hits.filter(pl.col("event_id") == event_id)
    top_particles = (
        evt.group_by("particle_id").len()
        .sort("len", descending=True)
        .head(top_n_particles)
        ["particle_id"].to_list()
    )

    fig = go.Figure()

    # Background: everything not in top_particles
    bg = evt.filter(~pl.col("particle_id").is_in(top_particles))
    fig.add_trace(go.Scatter3d(
        x=bg["x"].to_numpy(), y=bg["y"].to_numpy(), z=bg["z"].to_numpy(),
        mode="markers",
        marker=dict(size=1.5, color="lightgray", opacity=0.3),
        name="other particles",
        showlegend=True,
    ))

    palette = px.colors.qualitative.Plotly  # 10 distinct colors
    mode = "markers+lines" if connect_hits else "markers"
    for i, pid in enumerate(top_particles):
        p_hits = (
            evt.filter(pl.col("particle_id") == pid)
            .sort("time")  # chronological → traces the trajectory
        )
        fig.add_trace(go.Scatter3d(
            x=p_hits["x"].to_numpy(),
            y=p_hits["y"].to_numpy(),
            z=p_hits["z"].to_numpy(),
            mode=mode,
            marker=dict(size=4, color=palette[i % len(palette)]),
            line=dict(color=palette[i % len(palette)], width=3),
            name=f"particle {pid} ({p_hits.height} hits)",
        ))

    fig.update_layout(
        title=f"Event {event_id} — top {top_n_particles} particles (by hit count)",
        scene=dict(
            xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]",
            aspectmode="data",
        ),
        width=1000, height=750,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def scatter3d_rho_z_phi(hits: pl.DataFrame, event_id: int,
                        color_by: str = "volume_id"):
    """The ρ-z plot extended to 3D by adding φ as a third axis.

    In this representation:
      - barrel layers become *cylinders* (constant ρ),
      - endcap disks become *vertical sheets* (constant z),
      - tracks become helices that unroll along φ.
    """
    evt = hits.filter(pl.col("event_id") == event_id).with_columns(
        rho=(pl.col("x") ** 2 + pl.col("y") ** 2).sqrt(),
        phi=pl.arctan2(pl.col("y"), pl.col("x")),
    )

    fig = go.Figure(data=[go.Scatter3d(
        x=evt["z"].to_numpy(),
        y=evt["phi"].to_numpy(),
        z=evt["rho"].to_numpy(),
        mode="markers",
        marker=dict(
            size=2,
            color=evt[color_by].to_numpy(),
            colorscale="Viridis",
            opacity=0.7,
            colorbar=dict(title=color_by),
        ),
        hovertemplate=(
            "z=%{x:.1f} mm<br>φ=%{y:.2f} rad<br>ρ=%{z:.1f} mm"
            "<extra></extra>"
        ),
    )])
    fig.update_layout(
        title=f"Event {event_id} — cylindrical view (z, φ, ρ)",
        scene=dict(
            xaxis_title="z [mm]",
            yaxis_title="φ [rad]",
            zaxis_title="ρ [mm]",
        ),
        width=900, height=700,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig