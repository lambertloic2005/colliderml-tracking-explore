"""Physics-level derived quantities (pT, eta, p, q)."""
import polars as pl


def with_particle_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Add pt, p, eta, phi to a flat particles DataFrame.

    Requires columns: px, py, pz.
    """
    return (
        df.with_columns(
            pt=(pl.col("px") ** 2 + pl.col("py") ** 2).sqrt(),
            p=(pl.col("px") ** 2 + pl.col("py") ** 2 + pl.col("pz") ** 2).sqrt(),
            phi=pl.arctan2(pl.col("py"), pl.col("px")),
        )
        .with_columns(
            # eta = 0.5 * ln((p + pz) / (p - pz)), undefined when pt == 0
            eta=pl.when(pl.col("p") > pl.col("pz").abs())
            .then(
                0.5 * (
                    (pl.col("p") + pl.col("pz")) / (pl.col("p") - pl.col("pz"))
                ).log()
            )
            .otherwise(None),
        )
    )


def with_track_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Add pt, p, q, eta to a flat tracks DataFrame.

    Requires columns: qop, theta.
    """
    return df.with_columns(
        p=1.0 / pl.col("qop").abs(),
        q=pl.col("qop").sign(),
    ).with_columns(
        pt=pl.col("p") * pl.col("theta").sin(),
        # eta = -ln(tan(theta/2))
        eta=-(pl.col("theta") / 2).tan().log(),
    )

def with_hit_counts(
    particles: pl.DataFrame,
    tracker_hits: pl.DataFrame,
) -> pl.DataFrame:
    """Attach num_tracker_hits to particles by counting from tracker_hits.

    The dataset doesn't ship this column directly, so we derive it via a
    join on (event_id, particle_id). Particles with no hits get 0.
    """
    counts = (
        tracker_hits
        .group_by(["event_id", "particle_id"])
        .len()
        .rename({"len": "num_tracker_hits"})
    )
    return (
        particles
        .join(counts, on=["event_id", "particle_id"], how="left")
        .with_columns(pl.col("num_tracker_hits").fill_null(0).cast(pl.UInt16))
    )