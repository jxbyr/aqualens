"""Run configuration: where, when, and how to monitor.

A PipelineConfig is the single input to the pipeline. Everything that would
otherwise be hardcoded (coordinates, dates, cloud limits, resolution, paths)
lives here, so pointing the tool at a new water body means changing the
config, never the code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

KM_PER_DEG_LAT = 111.32

CADENCE_MONTHS = {"monthly": 1, "quarterly": 3, "half-yearly": 6, "yearly": 12}


def _parse_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _add_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) + months
    # Clamp to day 28 so every month is a legal target; the search window
    # around each date makes the exact day irrelevant.
    return date(total // 12, total % 12 + 1, min(d.day, 28))


@dataclass
class AOI:
    """Square area of interest centred on a point."""

    lat: float
    lon: float
    buffer_km: float = 20.0

    def __post_init__(self) -> None:
        if not -84.0 <= self.lat <= 84.0:
            raise ValueError(f"lat must be within ±84° (UTM limit), got {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon must be in [-180, 180], got {self.lon}")
        if not 1.0 <= self.buffer_km <= 150.0:
            raise ValueError(f"buffer_km must be in [1, 150], got {self.buffer_km}")

    @property
    def bbox(self) -> dict[str, float]:
        """WGS84 bounding box in the mapping format openEO expects."""
        dlat = self.buffer_km / KM_PER_DEG_LAT
        dlon = self.buffer_km / (KM_PER_DEG_LAT * math.cos(math.radians(self.lat)))
        return {
            "west": self.lon - dlon,
            "south": self.lat - dlat,
            "east": self.lon + dlon,
            "north": self.lat + dlat,
        }

    @property
    def utm_epsg(self) -> int:
        """EPSG code of the local UTM zone, so pixel sizes are in metres."""
        zone = min(60, int((self.lon + 180.0) // 6.0) + 1)
        return (32600 if self.lat >= 0 else 32700) + zone


@dataclass
class PipelineConfig:
    aoi: AOI
    dates: list[date]
    window_days: int = 45          # composite window is target ± window_days
    max_cloud_cover: float = 40.0  # % scene cloud cover admitted to the composite
    resolution_m: int = 60         # output ground sampling distance
    ndwi_threshold: float = 0.0    # fixed-threshold baseline / QC check
    mode: str = "drawdown"         # "drawdown" (lakes/inland) or "coastal" (reclamation)
    out_dir: Path = Path("aqualens_output")
    cache_dir: Path = Path("aqualens_cache")

    def __post_init__(self) -> None:
        self.dates = sorted({_parse_date(d) for d in self.dates})
        if not self.dates:
            raise ValueError("at least one target date is required")
        if not 5 <= self.window_days <= 120:
            raise ValueError(f"window_days must be in [5, 120], got {self.window_days}")
        if not 0.0 < self.max_cloud_cover <= 100.0:
            raise ValueError(f"max_cloud_cover must be in (0, 100], got {self.max_cloud_cover}")
        if self.resolution_m < 10:
            raise ValueError("resolution_m below 10 m exceeds Sentinel-2 native resolution")
        if self.mode not in ("drawdown", "coastal"):
            raise ValueError(f"mode must be 'drawdown' or 'coastal', got {self.mode!r}")
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir)

    @staticmethod
    def date_range(start: date | str, end: date | str, cadence: str = "yearly") -> list[date]:
        """Expand a start/end pair into a list of target dates."""
        if cadence not in CADENCE_MONTHS:
            raise ValueError(f"cadence must be one of {sorted(CADENCE_MONTHS)}, got {cadence!r}")
        step = CADENCE_MONTHS[cadence]
        current, stop = _parse_date(start), _parse_date(end)
        dates = []
        while current <= stop:
            dates.append(current)
            current = _add_months(current, step)
        return dates

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text())
        aoi = AOI(**raw.pop("aoi"))
        if "dates" not in raw:
            raw["dates"] = cls.date_range(
                raw.pop("start"), raw.pop("end"), raw.pop("cadence", "yearly")
            )
        return cls(aoi=aoi, **raw)
