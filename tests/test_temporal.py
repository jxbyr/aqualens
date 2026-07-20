from datetime import date

import numpy as np

from aqualens.classify import classify_gmm, classify_ndwi
from aqualens.temporal import EpochResult, fit_trend, summarise_epoch


def make_epoch(d, area, sigma=20.0, ndwi_area=None):
    return EpochResult(
        date=d,
        area_km2=area,
        sigma_km2=sigma,
        area_low_km2=area - 2 * sigma,
        area_high_km2=area + 2 * sigma,
        ndwi_area_km2=area if ndwi_area is None else ndwi_area,
        valid_fraction=1.0,
    )


def test_trend_recovers_known_slope():
    # 3000 km² declining by 110 km²/yr with small noise.
    rng = np.random.default_rng(0)
    epochs = [
        make_epoch(date(2016 + i, 8, 1), 3000.0 - 110.0 * i + rng.normal(0, 5))
        for i in range(11)
    ]
    trend = fit_trend(epochs)
    assert abs(trend["rate_km2_per_year"] + 110.0) < 5.0
    assert trend["rate_ci95"] is not None
    assert trend["rate_ci95"] < 20.0
    assert trend["percent_change"] < 0


def test_trend_with_two_epochs_has_no_ci():
    epochs = [make_epoch(date(2016, 8, 1), 3000.0), make_epoch(date(2026, 8, 1), 2000.0)]
    trend = fit_trend(epochs)
    assert abs(trend["rate_km2_per_year"] + 100.0) < 1.0
    assert trend["rate_ci95"] is None


def test_trend_requires_two_epochs():
    assert fit_trend([make_epoch(date(2016, 8, 1), 3000.0)]) is None


def test_summarise_epoch_consistency(synthetic_scene):
    gmm = classify_gmm(synthetic_scene)
    ndwi = classify_ndwi(synthetic_scene)
    epoch = summarise_epoch(synthetic_scene, gmm, ndwi)

    assert epoch.area_low_km2 <= epoch.area_km2 <= epoch.area_high_km2
    assert epoch.sigma_km2 > 0
    # GMM and NDWI agree on this clean synthetic scene: no divergence flag.
    assert "gmm_ndwi_divergence" not in epoch.qc_flags
    assert epoch.valid_fraction == 1.0


def test_summarise_epoch_flags_divergence(synthetic_scene):
    gmm = classify_gmm(synthetic_scene)
    ndwi = classify_ndwi(synthetic_scene, threshold=0.45)  # deliberately bad baseline
    epoch = summarise_epoch(synthetic_scene, gmm, ndwi)
    assert "gmm_ndwi_divergence" in epoch.qc_flags
