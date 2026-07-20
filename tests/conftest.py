from datetime import date

import numpy as np
import pytest

from aqualens.preprocessing import Scene


@pytest.fixture
def synthetic_scene():
    """A 200x200 scene with a known water region in the top 30% of rows.

    Water NDWI ~ N(0.4, 0.05), land NDWI ~ N(-0.3, 0.05): clearly bimodal,
    like a real lake shoreline at coarse resolution.
    """
    rng = np.random.default_rng(0)
    size = 200
    water_rows = 60  # 30% of the scene

    ndwi = rng.normal(-0.3, 0.05, size=(size, size)).astype(np.float32)
    ndwi[:water_rows] = rng.normal(0.4, 0.05, size=(water_rows, size))

    return Scene(
        date=date(2020, 6, 1),
        reflectance=np.zeros((size, size, 5), dtype=np.float32),
        ndwi=ndwi,
        valid=np.ones((size, size), dtype=bool),
        pixel_area_km2=0.0036,  # 60 m pixels
    )
