"""Denominator-dilution Shapley surge attribution and pre-event trend stability tests.

Explains why holiday event days (Black Friday / Cyber Monday) systematically exhibit lower
bot *percentages* even when raw bot request volume surges: a +55% to +61% human shopper
influx expands the denominator faster than a +17% to +20% attacker surge expands the numerator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bot_traffic_econometrics._rust_bridge import USING_RUST, require_rust

Backend = Literal["auto", "rust", "python"]


@dataclass(frozen=True)
class SurgeAttribution:
    """Exact two-player Shapley-Owen decomposition of `R(B1, H1) - R(B0, H0)`."""

    baseline_rate: float
    event_rate: float
    delta_rate: float
    bot_volume_growth: float
    human_volume_growth: float
    bot_impulse_effect: float
    human_dilution_effect: float
    is_diluted_bot_surge: bool


def attribute_surge_shapley(
    baseline_bots: float,
    baseline_humans: float,
    event_bots: float,
    event_humans: float,
) -> SurgeAttribution:
    """Decompose rate change `R1 - R0` into bot-volume impulse vs human-denominator dilution.

    Satisfies `bot_impulse_effect + human_dilution_effect == event_rate - baseline_rate`
    identically for any positive traffic counts.
    """
    if min(baseline_bots, baseline_humans, event_bots, event_humans) < 0.0:
        raise ValueError("Traffic counts must be non-negative")
    if baseline_bots + baseline_humans <= 0.0 or event_bots + event_humans <= 0.0:
        raise ValueError("Total traffic in each period must be > 0")

    def _rate(b: float, h: float) -> float:
        return b / (b + h)

    r00 = _rate(baseline_bots, baseline_humans)
    r10 = _rate(event_bots, baseline_humans)
    r01 = _rate(baseline_bots, event_humans)
    r11 = _rate(event_bots, event_humans)

    phi_b = 0.5 * ((r10 - r00) + (r11 - r01))
    phi_h = 0.5 * ((r01 - r00) + (r11 - r10))
    b_growth = (event_bots - baseline_bots) / baseline_bots if baseline_bots > 0.0 else 0.0
    h_growth = (event_humans - baseline_humans) / baseline_humans if baseline_humans > 0.0 else 0.0

    return SurgeAttribution(
        baseline_rate=r00,
        event_rate=r11,
        delta_rate=r11 - r00,
        bot_volume_growth=b_growth,
        human_volume_growth=h_growth,
        bot_impulse_effect=phi_b,
        human_dilution_effect=phi_h,
        is_diluted_bot_surge=(event_bots > baseline_bots) and (r11 < r00),
    )


@dataclass(frozen=True)
class TrendStability:
    """Non-parametric Theil-Sen slope and tie-corrected Mann-Kendall test over `BF-k` windows."""

    theil_sen_slope: float
    theil_sen_intercept: float
    kendall_tau_b: float
    mann_kendall_z: float
    p_value_two_sided: float


def _theil_sen_kendall_python(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float, float, float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    n = min(len(xa), len(ya))
    if n < 2:
        return 0.0, float(ya[0]) if n == 1 else 0.0, 0.0, 0.0

    slopes: list[float] = []
    s_stat = 0
    x_ties = 0
    y_ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(xa[j] - xa[i])
            dy = float(ya[j] - ya[i])
            if abs(dx) > 0.0:
                slopes.append(dy / dx)
            sx = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
            sy = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
            if sx == 0 and sy == 0:
                x_ties += 1
                y_ties += 1
            elif sx == 0:
                x_ties += 1
            elif sy == 0:
                y_ties += 1
            else:
                s_stat += sx * sy

    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(ya[:n] - slope * xa[:n]))
    tot_pairs = n * (n - 1) / 2.0
    denom = math.sqrt(max(0.0, (tot_pairs - x_ties) * (tot_pairs - y_ties)))
    tau_b = s_stat / denom if denom > 0.0 else 0.0

    _, counts = np.unique(ya[:n], return_counts=True)
    tie_term = float(np.sum([t * (t - 1) * (2 * t + 5) for t in counts if t > 1]))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0.0 or s_stat == 0:
        z = 0.0
    elif s_stat > 0:
        z = (s_stat - 1.0) / math.sqrt(var_s)
    else:
        z = (s_stat + 1.0) / math.sqrt(var_s)
    return slope, intercept, tau_b, z


def assess_pre_event_trend(
    days_to_event: Sequence[float],
    values: Sequence[float],
    *,
    backend: Backend = "auto",
) -> TrendStability:
    """Compute exact Theil-Sen median slope and tie-corrected Mann-Kendall trend test."""
    if len(days_to_event) != len(values) or len(values) < 2:
        raise ValueError("days_to_event and values must have equal length >= 2")
    use_rust = (backend == "rust") or (backend == "auto" and USING_RUST)
    if use_rust:
        slope, intercept, tau_b, z = require_rust().py_theil_sen_kendall(
            [float(v) for v in days_to_event], [float(v) for v in values]
        )
    else:
        slope, intercept, tau_b, z = _theil_sen_kendall_python(days_to_event, values)
    p_val = math.erfc(abs(z) / math.sqrt(2.0))
    return TrendStability(
        theil_sen_slope=slope,
        theil_sen_intercept=intercept,
        kendall_tau_b=tau_b,
        mann_kendall_z=z,
        p_value_two_sided=p_val,
    )


def count_sign_consistent_series(
    multi_year_series: Sequence[Sequence[Sequence[float]]],
) -> tuple[int, int]:
    """Return `(consistent_count, total_series)` keeping one strict pre-event Theil-Sen sign."""
    consistent = 0
    for yearly_runs in multi_year_series:
        slopes = [
            assess_pre_event_trend(list(range(len(run))), run).theil_sen_slope
            for run in yearly_runs
        ]
        if all(s > 0.0 for s in slopes) or all(s < 0.0 for s in slopes):
            consistent += 1
    return consistent, len(multi_year_series)
