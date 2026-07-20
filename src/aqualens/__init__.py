"""aqualens — automated water-body drawdown monitoring from Sentinel-2 imagery."""

from .config import AOI, PipelineConfig
from .pipeline import Result, run_pipeline

__version__ = "0.2.2"
__all__ = ["AOI", "PipelineConfig", "Result", "run_pipeline", "__version__"]
