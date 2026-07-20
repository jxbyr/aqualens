"""End-to-end orchestration: coordinates + dates in, drawdown report out."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import acquisition, report
from .classify import classify_gmm, classify_ndwi
from .config import PipelineConfig
from .preprocessing import load_scene
from .temporal import EpochResult, fit_trend, summarise_epoch

log = logging.getLogger(__name__)


@dataclass
class Result:
    config: PipelineConfig
    epochs: list[EpochResult]
    trend: dict | None
    out_dir: Path
    figures: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["aqualens results", "----------------"]
        for e in self.epochs:
            flags = f"  [{', '.join(e.qc_flags)}]" if e.qc_flags else ""
            lines.append(
                f"{e.date.isoformat()}  water area {e.area_km2:8.1f} ± {e.sigma_km2:.1f} km²  "
                f"(NDWI baseline {e.ndwi_area_km2:.1f} km²){flags}"
            )
        if self.trend is not None:
            rate = self.trend["rate_km2_per_year"]
            ci = self.trend["rate_ci95"]
            ci_text = f" ± {ci:.1f} (95% CI)" if ci else ""
            lines.append("")
            lines.append(
                f"Trend {self.trend['start']} → {self.trend['end']}: "
                f"{rate:+.1f} km²/yr{ci_text}  "
                f"({self.trend['percent_change']:+.1f}% total)"
            )
            if self.config.mode == "coastal":
                lines.append(
                    f"Net land change: {-rate:+.1f} km²/yr "
                    "(positive = land gained from the sea)"
                )
        lines.append("")
        lines.append(f"Outputs written to {self.out_dir}/")
        return "\n".join(lines)


def run_pipeline(
    config: PipelineConfig,
    connection=None,
    scene_paths: dict[date, Path] | None = None,
) -> Result:
    """Run the full pipeline.

    `scene_paths` (date -> GeoTIFF) skips acquisition entirely — useful for
    testing and for re-analysing previously downloaded composites offline.
    """
    if scene_paths is None:
        scene_paths = acquisition.fetch_all(config, connection)

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs: list[EpochResult] = []
    figures: list[Path] = []
    for target in sorted(scene_paths):
        scene = load_scene(scene_paths[target], target)
        gmm = classify_gmm(scene)
        ndwi = classify_ndwi(scene, config.ndwi_threshold)
        epoch = summarise_epoch(scene, gmm, ndwi)
        epochs.append(epoch)
        figures.append(report.save_epoch_figure(scene, gmm, ndwi, out_dir))
        log.info(
            "[%s] area %.1f ± %.1f km² (NDWI %.1f km²)%s",
            target, epoch.area_km2, epoch.sigma_km2, epoch.ndwi_area_km2,
            f" flags: {epoch.qc_flags}" if epoch.qc_flags else "",
        )

    trend = fit_trend(epochs)
    figures.append(report.save_drawdown_figure(epochs, trend, out_dir, mode=config.mode))
    report.write_results(config, epochs, trend, out_dir)

    return Result(config, epochs, trend, out_dir, figures)
