"""Figures and machine-readable outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .classify import Classification
from .config import PipelineConfig
from .preprocessing import Scene
from .temporal import EpochResult, _decimal_year

WATER_CMAP = ListedColormap(["#C4874A", "#2E86AB"])  # land brown, water blue
LEGEND = [
    mpatches.Patch(color="#2E86AB", label="Water"),
    mpatches.Patch(color="#C4874A", label="Land"),
]


def save_epoch_figure(
    scene: Scene, gmm: Classification, ndwi: Classification, out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    ndwi_img = scene.ndwi.astype(float).copy()
    ndwi_img[~scene.valid] = np.nan

    # Show the classification the numbers actually used: a degenerate GMM
    # falls back to the NDWI baseline (see temporal.summarise_epoch).
    primary = ndwi if gmm.degenerate else gmm
    labels_img = primary.labels.astype(float)
    labels_img[primary.labels == -1] = np.nan

    n_panels = 2 if gmm.degenerate else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6))
    fig.suptitle(f"aqualens — {scene.date.isoformat()}", fontsize=13, fontweight="bold")

    ax = axes[0]
    im = ax.imshow(ndwi_img, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    ax.set_title("NDWI", fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.imshow(labels_img, cmap=WATER_CMAP, vmin=0, vmax=1, interpolation="nearest")
    title = f"Classification ({primary.method})"
    if gmm.degenerate:
        title += "\n(GMM degenerate — NDWI fallback used)"
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=LEGEND, loc="lower right", fontsize=8)

    if not gmm.degenerate:
        p_img = gmm.p_water.astype(float).copy()
        p_img[~scene.valid] = np.nan
        ax = axes[2]
        im = ax.imshow(p_img, cmap="RdYlBu", vmin=0, vmax=1)
        ax.set_title("P(water) — GMM posterior", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = out_dir / f"epoch_{scene.date.isoformat()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_drawdown_figure(
    epochs: list[EpochResult],
    trend: dict | None,
    out_dir: Path,
    mode: str = "drawdown",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = sorted(epochs, key=lambda e: e.date)

    t = np.array([_decimal_year(e.date) for e in epochs])
    y = np.array([e.area_km2 for e in epochs])
    yerr = np.array([e.sigma_km2 if np.isfinite(e.sigma_km2) else 0.0 for e in epochs])
    ndwi_y = np.array([e.ndwi_area_km2 for e in epochs])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        t, y, yerr=yerr, fmt="o-", color="#2166AC", capsize=4, linewidth=2,
        markersize=8, markerfacecolor="white", markeredgewidth=2,
        label="GMM water area ± σ",
    )
    ax.plot(
        t, ndwi_y, "D--", color="black", alpha=0.7, markersize=7,
        label="NDWI baseline (QC)",
    )

    if trend is not None:
        xs = np.linspace(t.min(), t.max(), 50)
        ax.plot(
            xs, trend["rate_km2_per_year"] * xs + trend["intercept_km2"],
            ":", color="#D6604D", linewidth=2,
        )
        rate = trend["rate_km2_per_year"]
        ci = trend["rate_ci95"]
        ci_text = f" ± {ci:.1f} (95% CI)" if ci else ""
        if mode == "coastal":
            ax.set_title(
                f"Coastal water area — net land change {-rate:+.1f} km²/yr{ci_text}",
                fontsize=12, fontweight="bold",
            )
        else:
            ax.set_title(
                f"Water area drawdown — trend {rate:+.1f} km²/yr{ci_text}",
                fontsize=12, fontweight="bold",
            )
    else:
        ax.set_title("Water area by epoch", fontsize=12, fontweight="bold")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Water area (km²)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    out = out_dir / "drawdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def write_results(
    config: PipelineConfig,
    epochs: list[EpochResult],
    trend: dict | None,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = sorted(epochs, key=lambda e: e.date)

    epoch_rows = []
    for e in epochs:
        row = asdict(e)
        row["date"] = e.date.isoformat()
        row["qc_flags"] = list(e.qc_flags)
        epoch_rows.append(row)

    payload = {
        "config": {
            "aoi": {"lat": config.aoi.lat, "lon": config.aoi.lon, "buffer_km": config.aoi.buffer_km},
            "dates": [d.isoformat() for d in config.dates],
            "window_days": config.window_days,
            "max_cloud_cover": config.max_cloud_cover,
            "resolution_m": config.resolution_m,
            "ndwi_threshold": config.ndwi_threshold,
        },
        "epochs": epoch_rows,
        "trend": trend,
    }

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        for row in epoch_rows:
            row = dict(row)
            row["qc_flags"] = ";".join(row["qc_flags"])
            writer.writerow(row)

    return json_path
