"""Runtime configuration (environment driven)."""
from __future__ import annotations

import os

# PostgreSQL by default (docker-compose); SQLite is used by the test-suite.
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://seis:seis@localhost:5432/seisnet"
)

# Fixed processing band / trigger configuration (per product spec).
FILTER_FREQMIN_HZ = 1.0
FILTER_FREQMAX_HZ = 15.0
STA_WINDOW_S = 0.5
LTA_WINDOW_S = 5.0
TRIGGER_ON = 3.0
TRIGGER_OFF = 1.0
MAX_PICKS_PER_TRACE = 200

# Waveform acceptance rules.
MIN_SAMPLING_RATE_HZ = 40.0  # the fixed 1-15 Hz band requires Nyquist > 15 Hz
MIN_DURATION_S = 10.0        # shorter windows cannot support the 5 s LTA
MIN_ELEVATION_M = -1500.0
MAX_ELEVATION_M = 10000.0

# Quality grade thresholds: (max rms [s], min station count) -> grade.
QUALITY_THRESHOLDS = (
    ("A", 0.5, 8),
    ("B", 1.0, 6),
    ("C", 2.0, 4),
)

# Seed model so the API is usable out of the box.
DEFAULT_MODEL_NAME = "default"
DEFAULT_MODEL_VERSION = "1.0"
DEFAULT_MODEL_LAYERS = [
    {"thickness_km": 15.0, "vp_kms": 5.8, "vs_kms": 3.36},
    {"thickness_km": 20.0, "vp_kms": 6.5, "vs_kms": 3.75},
    {"thickness_km": None, "vp_kms": 8.0, "vs_kms": 4.62},
]
