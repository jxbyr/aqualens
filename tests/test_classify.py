import numpy as np

from aqualens.classify import classify_gmm, classify_ndwi


def test_ndwi_threshold_recovers_water_region(synthetic_scene):
    result = classify_ndwi(synthetic_scene, threshold=0.0)
    water_fraction = (result.labels == 1).mean()
    assert abs(water_fraction - 0.30) < 0.01
    # Water is in the top rows, land at the bottom.
    assert (result.labels[:50] == 1).mean() > 0.99
    assert (result.labels[100:] == 0).mean() > 0.99


def test_gmm_recovers_water_area_with_probabilities(synthetic_scene):
    result = classify_gmm(synthetic_scene)
    assert result.p_water is not None
    assert result.qc_flags == []

    true_area = 0.30 * synthetic_scene.valid.size * synthetic_scene.pixel_area_km2
    area = result.water_area_km2(synthetic_scene.pixel_area_km2)
    assert abs(area - true_area) / true_area < 0.02

    # Posterior probabilities should be confident away from the boundary.
    assert result.p_water[:50].mean() > 0.95
    assert result.p_water[100:].mean() < 0.05


def test_gmm_all_land_scene_reports_no_water(synthetic_scene):
    # All-land scene: overwrite the water region. With physically seeded
    # component means the water component stays an empty ghost at high NDWI,
    # so nothing is classified as water and no degeneracy flag is needed.
    rng = np.random.default_rng(1)
    synthetic_scene.ndwi[:] = rng.normal(-0.3, 0.05, synthetic_scene.ndwi.shape).astype(
        np.float32
    )
    result = classify_gmm(synthetic_scene)
    assert (result.labels == 1).sum() == 0
    assert result.p_water.max() < 0.5


def test_gmm_handles_tiny_scene(synthetic_scene):
    synthetic_scene.valid[:] = False
    synthetic_scene.valid[:5, :5] = True
    result = classify_gmm(synthetic_scene)
    assert "too_few_valid_pixels" in result.qc_flags
    assert result.p_water is None
