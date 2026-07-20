"""Water/land classification.

Two estimators, deliberately only two:

- NDWI threshold: fixed, parameter-free physical baseline. Free to compute
  and independent of any fitting, so it doubles as a QC check.
- GMM on NDWI: adaptive two-component mixture. Does everything K-Means did
  in the original project, plus per-pixel water probabilities — which is
  where all downstream uncertainty quantification comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture

from .preprocessing import Scene

# Flags meaning the fit cannot be trusted; downstream code then uses the
# NDWI baseline instead of this classification.
DEGENERATE_FLAGS = frozenset({"poor_component_separation", "water_component_negative_ndwi"})


@dataclass
class Classification:
    method: str
    labels: np.ndarray                 # (H, W) int8: 1 water, 0 land, -1 invalid
    p_water: np.ndarray | None = None  # (H, W) float32, defined where valid
    qc_flags: list[str] = field(default_factory=list)

    @property
    def degenerate(self) -> bool:
        return self.p_water is None or bool(DEGENERATE_FLAGS & set(self.qc_flags))

    def water_area_km2(self, pixel_area_km2: float) -> float:
        return float((self.labels == 1).sum()) * pixel_area_km2


def classify_ndwi(scene: Scene, threshold: float = 0.0) -> Classification:
    labels = np.full(scene.ndwi.shape, -1, dtype=np.int8)
    labels[scene.valid] = (scene.ndwi[scene.valid] > threshold).astype(np.int8)
    return Classification("ndwi_threshold", labels)


def classify_gmm(
    scene: Scene, random_state: int = 42, max_fit_samples: int = 200_000
) -> Classification:
    labels = np.full(scene.ndwi.shape, -1, dtype=np.int8)
    qc: list[str] = []

    x = scene.ndwi[scene.valid].reshape(-1, 1).astype(np.float64)
    if x.shape[0] < 100:
        qc.append("too_few_valid_pixels")
        return Classification("gmm", labels, None, qc)

    if x.shape[0] > max_fit_samples:
        rng = np.random.default_rng(random_state)
        fit_x = x[rng.choice(x.shape[0], max_fit_samples, replace=False)]
    else:
        fit_x = x

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=random_state,
        # Physically motivated starting points (land NDWI ~ -0.3, water ~ +0.4).
        # Random initialisation can degenerate into two near-identical
        # components when one class dominates the scene (e.g. a mostly-ocean
        # coastal AOI), classifying everything as one class.
        means_init=np.array([[-0.3], [0.4]]),
    )
    gmm.fit(fit_x)

    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())
    water = int(np.argmax(means))  # water = component with the higher NDWI mean

    separation = abs(means[1] - means[0]) / max(float(stds.mean()), 1e-6)
    if separation < 1.0:
        qc.append("poor_component_separation")  # scene may be all land or all water
    if means[water] < 0.0:
        qc.append("water_component_negative_ndwi")

    p_water = np.zeros(scene.ndwi.shape, dtype=np.float32)
    p_water[scene.valid] = gmm.predict_proba(x)[:, water].astype(np.float32)
    labels[scene.valid] = (p_water[scene.valid] > 0.5).astype(np.int8)
    return Classification("gmm", labels, p_water, qc)
