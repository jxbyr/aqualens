from datetime import date
from pathlib import Path

import numpy as np
import pytest

from aqualens.classify import Classification, classify_ndwi
from aqualens.config import AOI, PipelineConfig
from aqualens.pipeline import Result
from aqualens.temporal import EpochResult, summarise_epoch


def make_config(mode):
    return PipelineConfig(
        aoi=AOI(lat=25.0, lon=55.0, buffer_km=15),
        dates=["2019-01-15", "2025-01-15"],
        mode=mode,
    )


def test_mode_validation():
    assert make_config("coastal").mode == "coastal"
    with pytest.raises(ValueError):
        make_config("harbour")


def test_coastal_summary_reports_land_change():
    epochs = [
        EpochResult(date(2019, 1, 15), 100.0, 2.0, 95.0, 105.0, 99.0, 1.0),
        EpochResult(date(2025, 1, 15), 70.0, 2.0, 65.0, 75.0, 69.0, 1.0),
    ]
    trend = {
        "rate_km2_per_year": -5.0, "rate_ci95": None, "percent_change": -30.0,
        "start": "2019-01-15", "end": "2025-01-15",
    }
    coastal = Result(make_config("coastal"), epochs, trend, Path("out")).summary()
    assert "Net land change: +5.0 km²/yr" in coastal

    drawdown = Result(make_config("drawdown"), epochs, trend, Path("out")).summary()
    assert "land" not in drawdown


def test_wide_shoreline_band_flagged(synthetic_scene):
    # Fabricate a GMM output where a third of the "water" is ambiguous,
    # as on tidal flats: confident water in the top rows, p≈0.5 below them.
    p = np.zeros(synthetic_scene.ndwi.shape, dtype=np.float32)
    p[:40] = 0.99   # confident water
    p[40:70] = 0.5  # ambiguous intertidal band
    labels = np.full(synthetic_scene.ndwi.shape, 0, dtype=np.int8)
    labels[:40] = 1
    gmm = Classification("gmm", labels, p)

    epoch = summarise_epoch(synthetic_scene, gmm, classify_ndwi(synthetic_scene))
    assert "wide_shoreline_band" in epoch.qc_flags


def test_clean_scene_not_flagged(synthetic_scene):
    from aqualens.classify import classify_gmm

    gmm = classify_gmm(synthetic_scene)
    epoch = summarise_epoch(synthetic_scene, gmm, classify_ndwi(synthetic_scene))
    assert "wide_shoreline_band" not in epoch.qc_flags
    assert "fallback_ndwi_only" not in epoch.qc_flags


def test_degenerate_property():
    labels = np.zeros((4, 4), dtype=np.int8)
    p = np.full((4, 4), 0.5, dtype=np.float32)
    assert Classification("gmm", labels, None).degenerate                # no probabilities
    assert Classification("gmm", labels, p, ["poor_component_separation"]).degenerate
    assert not Classification("gmm", labels, p).degenerate


def test_epoch_figure_uses_fallback_classification(synthetic_scene, tmp_path):
    from aqualens.report import save_epoch_figure

    p = np.full(synthetic_scene.ndwi.shape, 0.5, dtype=np.float32)
    labels = np.ones(synthetic_scene.ndwi.shape, dtype=np.int8)  # bogus all-water
    gmm = Classification("gmm", labels, p, ["poor_component_separation"])

    out = save_epoch_figure(synthetic_scene, gmm, classify_ndwi(synthetic_scene), tmp_path)
    assert out.exists()


def test_degenerate_gmm_falls_back_to_ndwi(synthetic_scene):
    # A degenerate fit (components never separated): p ~ 0.5 everywhere,
    # labels tipped all-land — as seen on a mostly-ocean coastal AOI.
    p = np.full(synthetic_scene.ndwi.shape, 0.5, dtype=np.float32)
    labels = np.zeros(synthetic_scene.ndwi.shape, dtype=np.int8)
    gmm = Classification("gmm", labels, p, ["poor_component_separation"])

    ndwi = classify_ndwi(synthetic_scene)
    epoch = summarise_epoch(synthetic_scene, gmm, ndwi)

    assert "fallback_ndwi_only" in epoch.qc_flags
    assert epoch.area_km2 == pytest.approx(epoch.ndwi_area_km2)
    assert np.isfinite(epoch.sigma_km2) and epoch.sigma_km2 > 0
