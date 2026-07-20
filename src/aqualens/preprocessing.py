"""Load acquired composites into analysis-ready scenes.

Much simpler than the original notebook pipeline: NDWI is a ratio index, so
it needs no radiometric normalisation and is directly comparable across
scenes and years. Clouds were already removed server-side via the SCL mask,
so masked pixels arrive as nodata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from .acquisition import BANDS

GREEN = BANDS.index("B03")
NIR = BANDS.index("B08")


@dataclass
class Scene:
    """One epoch's composite, ready for classification."""

    date: date
    reflectance: np.ndarray  # (H, W, 5) float32, band order = acquisition.BANDS
    ndwi: np.ndarray         # (H, W) float32, zero where invalid
    valid: np.ndarray        # (H, W) bool
    pixel_area_km2: float
    crs: str = ""
    path: Path | None = None


def load_scene(path: str | Path, scene_date: date) -> Scene:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)
        nodata = src.nodata
        transform = src.transform
        crs = str(src.crs)

    if data.shape[0] != len(BANDS):
        raise ValueError(f"{path}: expected {len(BANDS)} bands, found {data.shape[0]}")

    reflectance = np.transpose(data, (1, 2, 0))
    valid = np.all(np.isfinite(reflectance), axis=-1)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.any(reflectance == nodata, axis=-1)
    valid &= np.any(reflectance > 0, axis=-1)  # all-zero pixels are empty

    green = reflectance[..., GREEN]
    nir = reflectance[..., NIR]
    denom = green + nir
    valid &= np.abs(denom) > 1e-6

    ndwi = np.zeros(green.shape, dtype=np.float32)
    ndwi[valid] = (green[valid] - nir[valid]) / denom[valid]

    # Pixel area from the geotransform (the composite is in a metric UTM CRS),
    # replacing the hardcoded pixel size of the original project.
    pixel_area_km2 = abs(transform.a * transform.e) / 1e6

    return Scene(
        date=scene_date,
        reflectance=reflectance,
        ndwi=ndwi,
        valid=valid,
        pixel_area_km2=pixel_area_km2,
        crs=crs,
        path=Path(path),
    )
