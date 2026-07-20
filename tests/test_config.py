from datetime import date

import pytest

from aqualens.config import AOI, PipelineConfig


def test_aral_sea_utm_zone_matches_original_project():
    # The GEOL0069 Aral Sea data was in EPSG:32640 (UTM 40N).
    assert AOI(lat=45.0, lon=59.0).utm_epsg == 32640


def test_bbox_is_centred_and_ordered():
    aoi = AOI(lat=45.0, lon=59.0, buffer_km=20)
    bbox = aoi.bbox
    assert bbox["west"] < aoi.lon < bbox["east"]
    assert bbox["south"] < aoi.lat < bbox["north"]
    # Longitude span must be wider than latitude span away from the equator.
    assert (bbox["east"] - bbox["west"]) > (bbox["north"] - bbox["south"])


def test_invalid_aoi_rejected():
    with pytest.raises(ValueError):
        AOI(lat=91.0, lon=0.0)
    with pytest.raises(ValueError):
        AOI(lat=0.0, lon=0.0, buffer_km=500)


def test_date_range_yearly():
    dates = PipelineConfig.date_range("2016-08-01", "2026-08-01", "yearly")
    assert len(dates) == 11
    assert dates[0] == date(2016, 8, 1)
    assert dates[-1] == date(2026, 8, 1)


def test_date_range_rejects_unknown_cadence():
    with pytest.raises(ValueError):
        PipelineConfig.date_range("2016-01-01", "2017-01-01", "weekly")


def test_config_sorts_and_dedupes_dates():
    cfg = PipelineConfig(
        aoi=AOI(lat=45.0, lon=59.0),
        dates=["2022-06-01", "2016-06-01", "2016-06-01"],
    )
    assert cfg.dates == [date(2016, 6, 1), date(2022, 6, 1)]


def test_from_yaml_with_date_range(tmp_path):
    yaml_text = """
aoi:
  lat: 45.0
  lon: 59.0
  buffer_km: 60
start: 2016-08-01
end: 2026-08-01
cadence: yearly
max_cloud_cover: 30
"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    cfg = PipelineConfig.from_yaml(path)
    assert len(cfg.dates) == 11
    assert cfg.max_cloud_cover == 30
    assert cfg.aoi.buffer_km == 60
