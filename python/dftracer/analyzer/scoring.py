"""Metric-driven severity scoring (analyzer-side).

Produces a CONTINUOUS severity in [0,1] per metric (same unit as the rule
engine's ``severity_score``), rather than discrete 1..5 bands — the band label
is derived from the score via the shared ``_severity_label`` thresholds.

The metric-driven *detection* signal is the SLOPE (WISIO's promise):
``ops_slope = time_share / op_share`` — an entity whose share of time is
disproportionate to its share of operations is slow per-op, i.e. a true
bottleneck (vs. merely busy). ``normalize_slope`` maps that slope onto [0,1] as
the continuous form of WISIO's angle-based slope bands. The metric fact
builder uses ``normalize_slope`` on ``*_slope``.
"""
import math

from dftracer.utils import Series

from .analysis_utils import is_missing, nan_as_null

# Slope severity ramp, centered on the PROPORTIONAL baseline. slope == 1 (45deg,
# time-share == op-share) is NOT a bottleneck (just busy) -> severity 0; only a
# disproportionate slope > 1 (slow per op) ramps up, reaching 1.0 as slope -> inf
# (90deg). This differs from WISIO's tan(15)..tan(75) display banding: we emit
# facts, so the proportional case must score 0, not mid-scale.
SLOPE_LOW_DEG = 45.0
SLOPE_HIGH_DEG = 90.0

# Intensity geometric range (kept for reference; intensity scoring is TODO).
INTENSITY_MIN = 1 / 1024
INTENSITY_MAX = 1 / 1024**3


def normalize_slope(slope):
    """Map an ops-slope (time_share / op_share) onto a [0,1] bottleneck severity.

    ``(atan(slope)_deg - 45) / (90 - 45)`` clipped to [0,1]: slope <= 1
    (proportional or better — not a bottleneck) -> 0; slope > 1 ramps up as the
    entity grows disproportionately slow per op, approaching 1 as slope -> inf.
    A missing or NaN slope gives a missing severity (None on a number).
    """
    if isinstance(slope, Series):
        angle_deg = nan_as_null(slope, as_float=True).arctan().degrees()
        return ((angle_deg - SLOPE_LOW_DEG) / (SLOPE_HIGH_DEG - SLOPE_LOW_DEG)).clip(lower=0.0, upper=1.0)
    if is_missing(slope):
        return None
    sev = (math.degrees(math.atan(slope)) - SLOPE_LOW_DEG) / (SLOPE_HIGH_DEG - SLOPE_LOW_DEG)
    return min(1.0, max(0.0, sev))
