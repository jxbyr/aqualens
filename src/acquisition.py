"""Sentinel-2 L2A acquisition from the Copernicus Data Space Ecosystem via openEO.

openEO does the heavy lifting server-side: for each requested epoch it builds
a cloud-masked median composite of every low-cloud acquisition inside the
search window, resampled to the local UTM zone, and returns one small
GeoTIFF — nothing close to a full ~1 GB L2A product is ever downloaded.

Each epoch runs as a batch job rather than a synchronous download: composites
over a large AOI routinely take longer than the 30-minute synchronous
connection limit, while batch jobs are polled and can run as long as needed.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, timedelta
from pathlib import Path

from .config import PipelineConfig

log = logging.getLogger(__name__)

OPENEO_URL = "openeo.dataspace.copernicus.eu"
COLLECTION = "SENTINEL2_L2A"
BANDS = ["B02", "B03", "B04", "B08", "B11"]
# Scene Classification Layer values excluded from composites:
# 3 cloud shadow, 8 cloud (medium prob.), 9 cloud (high prob.), 10 cirrus, 11 snow/ice
MASK_SCL = (3, 8, 9, 10, 11)


def connect(url: str = OPENEO_URL):
    """Connect and authenticate against the Copernicus openEO backend.

    Uses the OIDC device flow: a login URL is printed; sign in with a free
    Copernicus Data Space account. Works in Colab and plain terminals alike.
    """
    import openeo

    connection = openeo.connect(url)
    connection.authenticate_oidc()
    return connection


def epoch_cache_path(config: PipelineConfig, target: date) -> Path:
    aoi = config.aoi
    key = (
        f"{aoi.lat:.4f}_{aoi.lon:.4f}_{aoi.buffer_km:g}_"
        f"{config.window_days}_{config.max_cloud_cover:g}_{config.resolution_m}_v2"
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return config.cache_dir / f"s2_{target.isoformat()}_{digest}.tif"


def fetch_epoch(connection, config: PipelineConfig, target: date, force: bool = False) -> Path:
    path = epoch_cache_path(config, target)
    if path.exists() and not force:
        log.info("[%s] using cached composite %s", target, path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    half = timedelta(days=config.window_days)
    t0, t1 = target - half, target + half

    cube = connection.load_collection(
        COLLECTION,
        spatial_extent=config.aoi.bbox,
        temporal_extent=[t0.isoformat(), t1.isoformat()],
        bands=BANDS + ["SCL"],
        max_cloud_cover=config.max_cloud_cover,
    )
    # Resample to the target grid BEFORE masking and compositing: the median
    # then runs on ~36x fewer pixels than at native 10 m, which keeps large
    # AOIs within the backend's executor memory (full-resolution composites
    # of 100+ km AOIs get the workers OOM-killed).
    cube = cube.resample_spatial(
        resolution=config.resolution_m, projection=config.aoi.utm_epsg
    )
    scl = cube.band("SCL")
    cloud_mask = scl == MASK_SCL[0]
    for value in MASK_SCL[1:]:
        cloud_mask = cloud_mask | (scl == value)

    composite = (
        cube.filter_bands(BANDS)
        .mask(cloud_mask)
        .reduce_dimension(dimension="t", reducer="median")
    )

    # Backend failures are often transient ("simply try submitting again",
    # per the platform's own error message), so retry once before giving up.
    for attempt in (1, 2):
        log.info(
            "[%s] submitting batch job for %s → %s (attempt %d) ...",
            target, t0, t1, attempt,
        )
        try:
            composite.execute_batch(
                outputfile=str(path),
                out_format="GTiff",
                title=f"aqualens {target.isoformat()}",
            )
            break
        except Exception:
            if attempt == 2:
                raise
            log.warning("[%s] batch job failed; retrying once ...", target)

    log.info("[%s] downloaded %s (%.1f MB)", target, path.name, path.stat().st_size / 1e6)
    return path


def fetch_all(config: PipelineConfig, connection=None) -> dict[date, Path]:
    """Fetch composites for every configured epoch; failed epochs are skipped."""
    if connection is None:
        connection = connect()
    paths: dict[date, Path] = {}
    for target in config.dates:
        try:
            paths[target] = fetch_epoch(connection, config, target)
        except Exception:
            log.exception("[%s] acquisition failed — epoch skipped", target)
    if not paths:
        raise RuntimeError(
            "No epochs could be acquired; check the AOI, dates, and cloud-cover limit."
        )
    return paths
