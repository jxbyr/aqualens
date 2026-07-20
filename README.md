# aqualens

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jxbyr/aqualens/blob/main/notebooks/quickstart.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Automated water-body monitoring from Sentinel-2 satellite imagery.

Give it geographic coordinates and a set of dates; it pulls cloud-masked
Sentinel-2 composites from the Copernicus Data Space, classifies water vs
land, and reports the water area over time with a trend and honest
uncertainty — for any lake, reservoir, inland sea, or stretch of coast.

Generalised from my [Aral Sea analysis](https://github.com/jxbyr/GEOL0069-End-Of-Year-Assignment), 
final projecct for AI for Earth Observation module, UCL. That project compared five ML classifiers;
this tool keeps the two that earned their place — a Gaussian Mixture Model
(adaptive, gives per-pixel water probabilities) cross-checked against the
parameter-free NDWI > 0 baseline — and drops the rest for efficiency.

## Case studies

| Site | Question | Result |
|------|----------|--------|
| **Aral Sea** (drawdown) | How fast is the world's most famous shrinking lake still receding? | **−65.8 ± 33.8 km²/yr** over Aug 2018–2025 (−55% total). A 2018 high-inflow outlier was automatically down-weighted by its own uncertainty. |
| **Loch Ness** (null test) | Does the tool invent drawdown where none exists? | **−0.7 ± 14.7 km²/yr** over 2021–2025 — cleanly consistent with zero, through heavy Scottish cloud and terrain shadow. |
| **Dubai coast** (coastal mode) | Can it monitor land reclamation — and fail safely on a mostly-ocean scene? | **+0.3 ± 11.2 km²/yr** net land change 2019–2025. The GMM degenerated on the ocean-dominated AOI and every epoch automatically fell back to the NDWI baseline, flagged `fallback_ndwi_only`. |

Three sites, three behaviours: a real signal detected, a stable site left
alone, and a pathological case caught by QC instead of published as fact.

## How it works

1. **Acquire** — for each target date, openEO builds a cloud-masked median
   composite (bands B02/B03/B04/B08/B11, SCL cloud mask) server-side on the
   Copernicus Data Space as a batch job, downsampled before compositing, and
   returns one small GeoTIFF in the local UTM zone. Composites are cached,
   so re-runs are free.
2. **Classify** — NDWI is computed from raw reflectance (a ratio index needs
   no normalisation, so scenes are comparable across years). A 2-component
   GMM with physically seeded means assigns each pixel a water probability;
   water is the component with the higher NDWI mean. If the fit degenerates,
   the epoch falls back to the NDWI baseline and says so.
3. **Quantify** — water area from hard labels × pixel area (from the
   geotransform). The ambiguous shoreline band (0.1 < p < 0.9), the
   GMM/NDWI disagreement, and a 0.5% floor set the per-epoch uncertainty.
4. **Trend** — a weighted least-squares fit across epochs gives the rate in
   km²/yr with a 95% confidence interval.

**Coastal mode** (`mode="coastal"`): the same physics run in reverse for
land reclamation — harbours, artificial islands, coastal construction. The
report additionally states net land change (positive = land gained from the
sea), and a `wide_shoreline_band` QC flag warns when tidal flats make the
shoreline position tide-dependent.

Outputs: `results.json`, `results.csv`, a per-epoch map figure (NDWI /
classification / water probability), and a trend curve with error bars.

## Install

```bash
pip install git+https://github.com/jxbyr/aqualens.git
```

You need a free [Copernicus Data Space](https://dataspace.copernicus.eu/)
account; the first run prints a login URL (device flow — works in Colab).

## Use

The easiest way is the Colab badge above. In Python:

```python
from aqualens import AOI, PipelineConfig, run_pipeline

config = PipelineConfig(
    aoi=AOI(lat=45.0, lon=59.0, buffer_km=60),   # southern Aral Sea
    dates=PipelineConfig.date_range("2018-08-15", "2025-08-15", "yearly"),
    max_cloud_cover=40,
    resolution_m=60,
)
result = run_pipeline(config)
print(result.summary())
```

From the command line:

```bash
aqualens --lat 45.0 --lon 59.0 --buffer-km 60 \
         --start 2018-08-15 --end 2025-08-15 --cadence yearly
```

Reading the output: every epoch reports `area ± σ` plus QC flags
(`low_valid_fraction` = persistent cloud, `gmm_ndwi_divergence` = the two
estimators disagree, `fallback_ndwi_only` = the GMM was unusable,
`wide_shoreline_band` = tide-sensitive shoreline). A trustworthy trend is
one whose epochs are mostly unflagged.

## Development

```bash
pip install -e ".[dev]"
pytest          # runs entirely offline on synthetic rasters
```

CI runs the same suite on every push. The pipeline can also be run fully
offline against pre-downloaded GeoTIFFs via
`run_pipeline(config, scene_paths={date: path, ...})`.

## Acknowledgements

Built on the foundations of UCL's GEOL0069 (AI for Earth Observation),
taught by Dr Michel Tsamados and Weibin Chen. Imagery: ESA Copernicus
Sentinel-2, processed via the Copernicus Data Space Ecosystem openEO API.
