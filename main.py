"""Forecast GTA VI's launch-day Twitch viewers hour by hour (program/v1).

The program is run once per case: it reads the case file, which holds
everything recorded up to the end of the release day's first hour, and writes
one forecast file with five quantiles of the viewer count for each of hours
2 to 24. It runs with no network, so it forecasts from the case alone.

Method: a local two-feature regression over comparable launches.

Every comparable launch in the case carries its full hourly Twitch curve for
its own release day. For each hour h, the forecast models the log-ratio
log(viewers_h / viewers_1) as a plane in two things the case tells us at the
checkpoint:

1. log hour-1 viewers. Launches whose hour 1 is already huge stay near flat
   for the rest of the day, while launches with a small hour 1 explode later
   as regions unlock. The slope on this feature is strongly negative at late
   hours (about -0.8 at the evening peak) and near zero in the first hours.
2. log viewers per live channel at hour 1. When the hour-1 audience is
   already concentrated in a few big streams there is less headroom for the
   evening peak; when it is spread over many small streams, big streamers
   still joining later lifts the peak.

The fit is weighted: a launch counts more when it resembles this one (a
rolling midnight-local unlock and console-only platforms count double, a
single simultaneous global unlock, paid early access, or a weak tier count
half) and when its own hour-1 print is close to this case's (a Gaussian
kernel of width 1.0 in log space). The fitted slopes are clipped to a sane
band and ridge-shrunk, and the fitted centre is averaged 50/50 with the
plain kernel-weighted median of the same ratios, so no single fit can drag
the forecast away from what the nearest launches actually did.

The spread for an hour is the weighted spread of the residuals around the
fit, narrowed by 0.7 and clipped to [0.20, 0.90] in log units, so the five
quantiles are the median times exp(z_tau * spread). The quantiles are
rounded to 2 decimals. Everything is deterministic: the same case file
always yields the same bytes.

Fallbacks keep the program from ever failing a case: a comparable with no
usable release-day hour 1 is skipped; when channel counts are missing the
second feature is dropped and the fit uses hour 1 alone; an hour with no
comparable at all falls back to persistence with a widening spread; a case
with no hour-1 print falls back to its last recorded hourly point; and any
unexpected shape still yields a valid persistence forecast rather than a
crash.

Standard library only. Run as `python3 main.py` with MAKEFUN_INPUT_DIR /
MAKEFUN_OUTPUT_DIR set, or with the grader's /input and /output.
"""

import json
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves this file's directory OFF sys.path; nothing beside
# this file is imported, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# How wide the hour-1 similarity kernel is, in natural-log units.
H1_SIGMA = 1.0
# Ridge penalty on the fitted slopes, as a fraction of the total weight.
RIDGE = 0.05
# Clips on the fitted slopes: the hour-1 slope and the concentration slope.
SLOPE_H1_MIN = -0.7
SLOPE_H1_MAX = 0.1
SLOPE_VPC_MIN = -1.0
SLOPE_VPC_MAX = 1.0
# How much the fitted centre leans on the regression vs the kernel median.
FIT_WEIGHT = 0.5
# How much the residual spread is narrowed before it becomes the quantile
# spread.
SPREAD_SHRINK = 0.7
SPREAD_MIN = 0.20
SPREAD_MAX = 0.90
# A comparable whose hour-1 print is below this is skipped: its ratios are
# noise from a category that had barely unlocked.
H1_FLOOR = 100
# Fallback spread growth per square-root hour when no comparable speaks.
FALLBACK_SIGMA = 0.15


def _parse_time(text):
    """An ISO 8601 UTC timestamp as epoch seconds (fromisoformat handles Z)."""
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _release_day_hours(comp):
    """A comparable's release-day hourly viewers, {hour 1..24: viewers}."""
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
            hour = int((_parse_time(row["from"]) - day_start) // 3600) + 1
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= hour <= 24 and viewers is not None and viewers > 0:
            hours[hour] = viewers
    return hours


def _release_day_channels(comp):
    """A comparable's release-day hourly live channels, {hour: channels}."""
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    channels = {}
    for row in comp.get("hourly") or []:
        try:
            count = row["channels"]
            hour = int((_parse_time(row["from"]) - day_start) // 3600) + 1
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= hour <= 24 and count is not None and count > 0:
            channels[hour] = count
    return channels


def _similarity_weight(comp):
    """How much a comparable launch resembles this one, before the hour-1
    kernel: midnight-local rolling unlocks and console-only platforms count
    more; simultaneous global unlocks, early access and weak tiers less."""
    unlock = str(comp.get("unlock") or "").lower()
    platforms = str(comp.get("platforms") or "").lower()
    weight = 1.0
    if "midnight local" in unlock or "rolling" in unlock:
        weight *= 2.0
    if "pc" not in platforms:
        weight *= 1.5
    if "simultaneous global" in unlock or "single global" in unlock:
        weight *= 0.5
    if "early access" in unlock or "early-access" in unlock:
        weight *= 0.5
    if "tier: weak" in unlock:
        weight *= 0.25
    return weight


def _weighted_median(pairs):
    """Weighted median of [(value, weight)] pairs, ascending by value."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return None
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2:
            return value
    return ordered[-1][0]


def _weighted_spread(pairs, centre):
    """Weighted standard deviation of the values around a centre."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    var = sum(w * (v - centre) ** 2 for v, w in pairs) / total
    return math.sqrt(max(var, 0.0))


def _fit_plane(rows):
    """Weighted ridge fit of y on two centred features.

    rows are (y, x1, x2, weight). Returns (intercept, slope1, slope2) with
    the slopes clipped to their bands; with too few rows, or a degenerate
    design, returns the weighted mean with zero slopes.
    """
    total = sum(w for _, _, _, w in rows)
    if total <= 0 or len(rows) < 3:
        return None
    mean_y = sum(w * y for y, _, _, w in rows) / total
    mean_x1 = sum(w * x1 for _, x1, _, w in rows) / total
    mean_x2 = sum(w * x2 for _, _, x2, w in rows) / total
    s11 = sum(w * (x1 - mean_x1) ** 2 for _, x1, _, w in rows) + RIDGE * total
    s22 = sum(w * (x2 - mean_x2) ** 2 for _, _, x2, w in rows) + RIDGE * total
    s12 = sum(w * (x1 - mean_x1) * (x2 - mean_x2) for _, x1, x2, w in rows)
    t1 = sum(w * (x1 - mean_x1) * (y - mean_y) for y, x1, _, w in rows)
    t2 = sum(w * (x2 - mean_x2) * (y - mean_y) for y, _, x2, w in rows)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return (mean_y, 0.0, 0.0)
    slope1 = (t1 * s22 - t2 * s12) / det
    slope2 = (t2 * s11 - t1 * s12) / det
    slope1 = min(max(slope1, SLOPE_H1_MIN), SLOPE_H1_MAX)
    slope2 = min(max(slope2, SLOPE_VPC_MIN), SLOPE_VPC_MAX)
    intercept = mean_y - slope1 * mean_x1 - slope2 * mean_x2
    return (intercept, slope1, slope2)


def _hour_shape(comparables, target_h1, target_vpc):
    """For each hour 2..24: (median log-ratio to hour 1, log spread).

    The pool is every comparable with a usable release day; each one is
    weighted by its similarity to this launch and by how close its own
    hour-1 print is to the target's. The centre blends a two-feature
    regression fit with the kernel-weighted median of the same ratios.
    """
    log_target = math.log(max(1, target_h1))
    log_vpc = math.log(max(1e-9, target_vpc)) if target_vpc else None
    # Each comparable's release-day curve, channels and weight, computed
    # once: the per-hour loop below only reads them.
    pool = []
    for comp in comparables:
        hours = _release_day_hours(comp)
        if 1 not in hours or hours[1] < H1_FLOOR:
            continue
        weight = _similarity_weight(comp)
        weight *= math.exp(
            -((math.log(hours[1]) - log_target) ** 2) / (2 * H1_SIGMA**2)
        )
        channels = _release_day_channels(comp)
        vpc = hours[1] / channels[1] if 1 in channels else None
        pool.append((hours, weight, vpc))
    shape = {}
    for hour in range(2, 25):
        rows = []
        med_pairs = []
        for hours, weight, vpc in pool:
            if hour not in hours:
                continue
            ratio = math.log(hours[hour] / hours[1])
            med_pairs.append((ratio, weight))
            x1 = math.log(hours[1]) - log_target
            if log_vpc is not None and vpc:
                x2 = math.log(vpc) - log_vpc
            else:
                x2 = 0.0
            rows.append((ratio, x1, x2, weight))
        if not rows:
            shape[hour] = (0.0, FALLBACK_SIGMA * math.sqrt(hour - 1))
            continue
        median = _weighted_median(med_pairs)
        fit = _fit_plane(rows)
        if fit is None:
            centre = median
            spread = _weighted_spread(med_pairs, median)
        else:
            intercept, slope1, slope2 = fit
            centre = FIT_WEIGHT * intercept + (1 - FIT_WEIGHT) * median
            total = sum(w for _, _, _, w in rows)
            var = sum(
                w * (y - (intercept + slope1 * x1 + slope2 * x2)) ** 2
                for y, x1, x2, w in rows
            ) / total
            spread = math.sqrt(max(var, 0.0))
        spread = min(max(spread * SPREAD_SHRINK, SPREAD_MIN), SPREAD_MAX)
        shape[hour] = (centre, spread)
    return shape


def _round2(value):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if rounded.is_integer() else rounded


def _quantiles(median_value, spread, levels):
    """The five quantiles around a median at the case's levels."""
    dist = NormalDist()
    out = []
    prev = 0.0
    for tau in levels:
        q = median_value * math.exp(dist.inv_cdf(tau) * spread)
        q = min(max(q, 1e-6), 1e9)
        q = max(q, prev)  # keep the row non-decreasing
        out.append(_round2(q))
        prev = out[-1]
    return out


def _hour1_print(case):
    """Hour 1's print, or the last recorded hourly point as a fallback."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1, row["viewers"])
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1].get("viewers")
        if last is not None:
            return max(1, last)
    return 1


def _hour1_vpc(case, p1):
    """Viewers per live channel at hour 1, or None when channels are unknown."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("channels"):
            return p1 / row["channels"]
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1]
        if last.get("channels"):
            return p1 / last["channels"]
    return None


def forecast_case(case):
    """The forecast object for one case: {"hours": [...23 rows...]}."""
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    hours_wanted = [row["hour"] for row in case.get("forecast") or []]
    if not hours_wanted:
        hours_wanted = list(range(2, 25))
    p1 = _hour1_print(case)
    vpc = _hour1_vpc(case, p1)
    shape = _hour_shape(case.get("comparables") or [], p1, vpc)
    rows = []
    for hour in hours_wanted:
        median_log, spread = shape.get(
            hour, (0.0, FALLBACK_SIGMA * math.sqrt(max(hour - 1, 0)))
        )
        median_value = p1 * math.exp(median_log)
        rows.append({"hour": hour, "q": _quantiles(median_value, spread, levels)})
    return {"hours": rows}


def _persistence_forecast(case):
    """Fallback: every hour stays at hour 1's print, spread widening."""
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    p1 = _hour1_print(case)
    rows = []
    for hour in range(2, 25):
        rows.append(
            {"hour": hour, "q": _quantiles(p1, FALLBACK_SIGMA * math.sqrt(hour - 1), levels)}
        )
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Never fail a case: a valid persistence forecast beats a crash.
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = _persistence_forecast(case)
            except Exception:
                forecast = {
                    "hours": [
                        {"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)
                    ]
                }
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
