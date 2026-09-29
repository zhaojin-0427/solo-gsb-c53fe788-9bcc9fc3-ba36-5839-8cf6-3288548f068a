"""Event quality grading from residual RMS and station count."""
from __future__ import annotations

from .. import config


def quality_grade(rms_s: float, station_count: int) -> str:
    """A/B/C/D grade determined by residual RMS and station count."""
    for grade, max_rms, min_sta in config.QUALITY_THRESHOLDS:
        if rms_s <= max_rms and station_count >= min_sta:
            return grade
    return "D"
