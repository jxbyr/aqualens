"""Command-line interface.

Examples
--------
aqualens --lat 45.0 --lon 59.0 --buffer-km 60 --start 2016-08-01 --end 2026-08-01
aqualens --config aral.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import AOI, CADENCE_MONTHS, PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aqualens",
        description="Water-body drawdown monitoring from Sentinel-2 imagery.",
    )
    parser.add_argument("--config", help="YAML config file; other flags are then ignored")
    parser.add_argument("--lat", type=float, help="AOI centre latitude")
    parser.add_argument("--lon", type=float, help="AOI centre longitude")
    parser.add_argument("--buffer-km", type=float, default=20.0, help="half-width of the square AOI")
    parser.add_argument("--start", help="first target date (YYYY-MM-DD)")
    parser.add_argument("--end", help="last target date (YYYY-MM-DD)")
    parser.add_argument("--cadence", default="yearly", choices=sorted(CADENCE_MONTHS))
    parser.add_argument("--dates", nargs="*", help="explicit target dates instead of --start/--end")
    parser.add_argument("--window-days", type=int, default=45)
    parser.add_argument("--max-cloud-cover", type=float, default=40.0)
    parser.add_argument("--resolution", type=int, default=60, help="output resolution in metres")
    parser.add_argument("--ndwi-threshold", type=float, default=0.0)
    parser.add_argument(
        "--mode", default="drawdown", choices=("drawdown", "coastal"),
        help="'coastal' reports net land change (reclamation) alongside water area",
    )
    parser.add_argument("--out", default="aqualens_output")
    parser.add_argument("--cache", default="aqualens_cache")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        return PipelineConfig.from_yaml(args.config)

    if args.lat is None or args.lon is None:
        raise SystemExit("either --config or both --lat and --lon are required")
    if args.dates:
        dates = args.dates
    elif args.start and args.end:
        dates = PipelineConfig.date_range(args.start, args.end, args.cadence)
    else:
        raise SystemExit("provide --dates, or --start and --end")

    return PipelineConfig(
        aoi=AOI(lat=args.lat, lon=args.lon, buffer_km=args.buffer_km),
        dates=dates,
        window_days=args.window_days,
        max_cloud_cover=args.max_cloud_cover,
        resolution_m=args.resolution,
        ndwi_threshold=args.ndwi_threshold,
        mode=args.mode,
        out_dir=args.out,
        cache_dir=args.cache,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    result = run_pipeline(config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
