"""Official Flutter/Dart documentation ingestion pipeline."""

from .classifier import TAXONOMY, classify_chunk
from .coverage import build_coverage_report

__all__ = ["TAXONOMY", "classify_chunk", "build_coverage_report"]
