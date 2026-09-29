"""miniSEED decoding and per-waveform validation."""
from __future__ import annotations

import base64
import hashlib
import io

from obspy import UTCDateTime, read

from .. import config


class WaveformError(Exception):
    """Validation failure carrying a stable, client-visible error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def decode_miniseed(data_b64: str):
    """Decode base64 miniSEED; returns (longest_trace, raw_bytes)."""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception:
        raise WaveformError("invalid_base64", "data_b64 is not valid base64")
    if not raw:
        raise WaveformError("invalid_miniseed", "empty miniSEED payload")
    try:
        stream = read(io.BytesIO(raw), format="MSEED")
    except Exception:
        raise WaveformError("invalid_miniseed", "payload is not a readable miniSEED record")
    if len(stream) == 0:
        raise WaveformError("invalid_miniseed", "miniSEED contains no traces")
    try:
        stream = stream.merge(method=0, fill_value=0)
    except Exception:
        pass
    return max(stream, key=lambda t: t.stats.npts), raw


def validate_waveform_meta(wf) -> None:
    """Validate declared metadata before touching the payload."""
    if wf.sampling_rate is None:
        raise WaveformError(
            "missing_sampling_rate", f"station {wf.station}: sampling_rate is required"
        )
    if wf.sampling_rate <= 0:
        raise WaveformError(
            "invalid_sampling_rate",
            f"station {wf.station}: sampling_rate must be positive",
        )
    if wf.sampling_rate < config.MIN_SAMPLING_RATE_HZ:
        raise WaveformError(
            "sampling_rate_too_low",
            f"station {wf.station}: sampling_rate {wf.sampling_rate} Hz cannot support "
            f"the fixed {config.FILTER_FREQMIN_HZ}-{config.FILTER_FREQMAX_HZ} Hz band",
        )
    if not (-90.0 <= wf.latitude <= 90.0) or not (-180.0 <= wf.longitude <= 180.0):
        raise WaveformError(
            "coordinate_out_of_bounds",
            f"station {wf.station}: latitude/longitude out of bounds",
        )
    if not (config.MIN_ELEVATION_M <= wf.elevation_m <= config.MAX_ELEVATION_M):
        raise WaveformError(
            "coordinate_out_of_bounds",
            f"station {wf.station}: elevation_m out of bounds "
            f"[{config.MIN_ELEVATION_M}, {config.MAX_ELEVATION_M}]",
        )


def check_trace_consistency(trace, wf) -> None:
    """Cross-check the miniSEED header against the declared metadata."""
    sr = float(trace.stats.sampling_rate)
    if abs(sr - wf.sampling_rate) > 1e-3 * wf.sampling_rate:
        raise WaveformError(
            "sampling_rate_mismatch",
            f"station {wf.station}: declared {wf.sampling_rate} Hz but miniSEED "
            f"header says {sr} Hz",
        )
    npts = int(trace.stats.npts)
    duration = npts / sr
    if duration < config.MIN_DURATION_S:
        raise WaveformError(
            "insufficient_data",
            f"station {wf.station}: {duration:.2f} s of data, need at least "
            f"{config.MIN_DURATION_S:.0f} s for the {config.LTA_WINDOW_S:.0f} s LTA window",
        )
    declared = UTCDateTime(wf.start_time)
    if abs(trace.stats.starttime - declared) > 0.5 / sr:
        raise WaveformError(
            "start_time_mismatch",
            f"station {wf.station}: declared start {wf.start_time} but miniSEED "
            f"header says {trace.stats.starttime}",
        )


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
