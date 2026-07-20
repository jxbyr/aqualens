"""End-to-end test on synthetic GeoTIFFs — no network or credentials needed."""

import json
from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from aqualens.config import AOI, PipelineConfig
from aqualens.pipeline import run_pipeline
from aqualens.preprocessing import load_scene


def write_composite(path, water_rows, size=120, nodata_rows=0):
    """Write a 5-band GeoTIFF mimicking an openEO composite (60 m UTM pixels)."""
    rng = np.random.default_rng(42)
    transform = from_origin(500000, 5000000, 60, 60)

    green = rng.normal(800, 50, (size, size)).astype(np.float32)
    nir = rng.normal(2400, 100, (size, size)).astype(np.float32)  # land: NDWI < 0
    nir[:water_rows] = rng.normal(300, 30, (water_rows, size))    # water: NDWI > 0
    green[:water_rows] = rng.normal(900, 50, (water_rows, size))

    bands = [green * 0.9, green, green * 1.1, nir, nir * 1.2]  # B02,B03,B04,B08,B11
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=5,
        dtype="float32", crs="EPSG:32640", transform=transform, nodata=-9999.0,
    ) as dst:
        for i, band in enumerate(bands, start=1):
            band = band.copy()
            if nodata_rows:
                band[-nodata_rows:] = -9999.0
            dst.write(band.astype(np.float32), i)
    return path


def test_load_scene_geometry_and_ndwi(tmp_path):
    path = write_composite(tmp_path / "scene.tif", water_rows=40, nodata_rows=10)
    scene = load_scene(path, date(2020, 6, 1))

    assert scene.pixel_area_km2 == pytest.approx(0.0036)
    assert scene.crs == "EPSG:32640"
    assert not scene.valid[-5:].any()          # nodata rows excluded
    assert (scene.ndwi[:40][scene.valid[:40]] > 0).mean() > 0.99
    assert (scene.ndwi[50:100] < 0).all()


def test_pipeline_offline_end_to_end(tmp_path):
    scene_paths = {}
    # Shrinking water body: 60 → 45 → 30 rows of water.
    for year, rows in [(2016, 60), (2021, 45), (2026, 30)]:
        d = date(year, 8, 1)
        scene_paths[d] = write_composite(tmp_path / f"{year}.tif", water_rows=rows)

    config = PipelineConfig(
        aoi=AOI(lat=45.0, lon=59.0, buffer_km=5),
        dates=list(scene_paths),
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    result = run_pipeline(config, scene_paths=scene_paths)

    assert len(result.epochs) == 3
    areas = [e.area_km2 for e in result.epochs]
    assert areas[0] > areas[1] > areas[2]

    # True water areas: rows * 120 px * 0.0036 km².
    for epoch, rows in zip(result.epochs, [60, 45, 30]):
        assert epoch.area_km2 == pytest.approx(rows * 120 * 0.0036, rel=0.05)

    assert result.trend["rate_km2_per_year"] < 0

    out = tmp_path / "out"
    assert (out / "results.json").exists()
    assert (out / "results.csv").exists()
    assert (out / "drawdown.png").exists()
    assert len(list(out.glob("epoch_*.png"))) == 3

    payload = json.loads((out / "results.json").read_text())
    assert payload["trend"]["n_epochs"] == 3
    assert len(payload["epochs"]) == 3
