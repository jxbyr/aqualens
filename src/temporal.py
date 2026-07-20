"""Per-epoch water area with uncertainty, and the drawdown trend across epochs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy import stats

from .classify import Classification
from .preprocessing import Scene

# Bounds of the ambiguous shoreline probability band. Pixels between these
# limits are the ones the GMM genuinely cannot call, and they define the
# reported uncertainty on each epoch's area.
P_LOW, P_HIGH = 0.1, 0.9

# GMM vs NDWI-baseline relative disagreement beyond this flags the epoch.
DIVERGENCE_TOLERANCE = 0.15


@dataclass
class EpochResult:
    date: date
    area_km2: float        # GMM water area (hard labels, p > 0.5)
    sigma_km2: float
    area_low_km2: float    # confident water only (p >= 0.9)
    area_high_km2: float   # everything possibly water (p >= 0.1)
    ndwi_area_km2: float   # independent fixed-threshold check
    valid_fraction: float
    qc_flags: list[str] = field(default_factory=list)


def summarise_epoch(
    scene: Scene, gmm: Classification, ndwi: Classification
) -> EpochResult:
    px = scene.pixel_area_km2
    qc = list(gmm.qc_flags)

    valid_fraction = float(scene.valid.mean())
    if valid_fraction < 0.5:
        qc.append("low_valid_fraction")

    ndwi_area = ndwi.water_area_km2(px)

    # When the GMM is unusable — too few pixels, or a degenerate fit where the
    # two components never separated — the fixed NDWI threshold is the more
    # trustworthy estimate, so it becomes the primary area for this epoch.
    if gmm.degenerate:
        qc.append("fallback_ndwi_only")
        sigma = max(0.05 * ndwi_area, 1e-6)
        return EpochResult(
            scene.date, ndwi_area, sigma, ndwi_area, ndwi_area,
            ndwi_area, valid_fraction, qc,
        )

    p = gmm.p_water[scene.valid].astype(np.float64)
    # Hard-label area, not the sum of probabilities: over a large AOI the
    # probability tail on millions of land pixels adds up to phantom water.
    area = gmm.water_area_km2(px)
    area_low = float((p >= P_HIGH).sum()) * px
    area_high = float((p >= P_LOW).sum()) * px
    # Uncertainty combines the ambiguous shoreline band (as a ±2σ envelope),
    # the GMM/NDWI disagreement as a systematic term, and a 0.5% floor so no
    # epoch reports implausibly perfect confidence.
    sigma = max(
        (area_high - area_low) / 4.0,
        abs(area - ndwi_area) / 2.0,
        0.005 * area,
    )

    if max(area, ndwi_area) > 0 and abs(area - ndwi_area) / max(area, ndwi_area) > DIVERGENCE_TOLERANCE:
        qc.append("gmm_ndwi_divergence")

    # A wide ambiguous band relative to the water area usually means tidal
    # flats or a mixed shoreline: the measured area there swings with the
    # tide state of the scenes that entered the composite.
    if area > 0 and (area_high - area_low) > 0.25 * area:
        qc.append("wide_shoreline_band")

    return EpochResult(
        scene.date, area, sigma, area_low, area_high, ndwi_area, valid_fraction, qc
    )


def _decimal_year(d: date) -> float:
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def fit_trend(epochs: list[EpochResult]) -> dict | None:
    """Weighted least-squares drawdown trend. Returns None with < 2 epochs.

    Weights are 1/sigma^2; the slope standard error is inflated by the
    reduced chi-square when the scatter exceeds the per-epoch uncertainties.
    """
    if len(epochs) < 2:
        return None

    epochs = sorted(epochs, key=lambda e: e.date)
    t = np.array([_decimal_year(e.date) for e in epochs])
    y = np.array([e.area_km2 for e in epochs])
    s = np.array([
        e.sigma_km2 if np.isfinite(e.sigma_km2) else 0.05 * e.area_km2
        for e in epochs
    ])  # nan guard kept for results serialised by older versions
    s = np.maximum(s, max(0.005 * float(y.mean()), 1e-6))
    w = 1.0 / s**2

    sw, swx, swy = w.sum(), (w * t).sum(), (w * y).sum()
    swxx, swxy = (w * t * t).sum(), (w * t * y).sum()
    delta = sw * swxx - swx**2
    slope = (sw * swxy - swx * swy) / delta
    intercept = (swxx * swy - swx * swxy) / delta

    n = len(epochs)
    result = {
        "n_epochs": n,
        "start": epochs[0].date.isoformat(),
        "end": epochs[-1].date.isoformat(),
        "rate_km2_per_year": float(slope),
        "intercept_km2": float(intercept),
        "total_change_km2": float(slope * (t[-1] - t[0])),
        "percent_change": float((y[-1] - y[0]) / y[0] * 100.0) if y[0] else float("nan"),
        "rate_stderr": None,
        "rate_ci95": None,
        "chi2_reduced": None,
    }
    if n > 2:
        residuals = y - (slope * t + intercept)
        chi2_red = float((w * residuals**2).sum() / (n - 2))
        stderr = math.sqrt(sw / delta * max(1.0, chi2_red))
        result["rate_stderr"] = stderr
        result["rate_ci95"] = float(stats.t.ppf(0.975, n - 2) * stderr)
        result["chi2_reduced"] = chi2_red
    return result
