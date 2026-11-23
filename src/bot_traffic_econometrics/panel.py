"""Dynamic unbalanced-panel shift-share decomposition and Dirichlet cluster bootstrap.

Formalizes the cohort-churn vs within-cohort behavioral identification problem from
Ogunnoiki (F5 Labs, 2026): when enterprise telemetry panels gain and lose retailers across
seasons, aggregate bot-rate changes conflate genuine attacker behavior shifts with entry,
exit, and traffic-share reallocation across heavy-tailed entities.
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
class EntityTraffic:
    """Traffic summary for one entity (retailer, tenant, or autonomous system) in one season."""

    entity_id: str
    total_requests: float
    bot_requests: float

    def __post_init__(self) -> None:
        if self.total_requests <= 0.0:
            raise ValueError(f"total_requests must be > 0 for {self.entity_id!r}")
        if not (0.0 <= self.bot_requests <= self.total_requests):
            raise ValueError(f"bot_requests out of [0, total_requests] for {self.entity_id!r}")

    @property
    def bot_rate(self) -> float:
        return self.bot_requests / self.total_requests


@dataclass(frozen=True)
class PanelDecomposition:
    """Exact 4-way Melitz-Polanec + symmetric Kitagawa decomposition of `r1 - r0`."""

    r0: float
    r1: float
    delta_total: float
    survivor_r0: float
    survivor_r1: float
    within_behavior: float
    survivor_reallocation: float
    entry_selection: float
    exit_selection: float
    n_survivors: int
    n_exiters: int
    n_entrants: int

    @property
    def total_composition(self) -> float:
        """Combined non-behavioral shift: survivor reallocation + entry + exit selection."""
        return self.survivor_reallocation + self.entry_selection + self.exit_selection

    @property
    def identity_residual(self) -> float:
        return abs(
            (
                self.within_behavior
                + self.survivor_reallocation
                + self.entry_selection
                + self.exit_selection
            )
            - self.delta_total
        )


@dataclass(frozen=True)
class PanelBootstrapCI:
    """95% (or `1 - alpha`) Bayesian Dirichlet cluster-bootstrap intervals."""

    point: PanelDecomposition
    delta_total_ci: tuple[float, float]
    within_behavior_ci: tuple[float, float]
    total_composition_ci: tuple[float, float]
    entry_selection_ci: tuple[float, float]
    exit_selection_ci: tuple[float, float]


def _partition_cohorts(
    period0: Sequence[EntityTraffic],
    period1: Sequence[EntityTraffic],
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    map0 = {e.entity_id: e for e in period0}
    map1 = {e.entity_id: e for e in period1}
    if len(map0) != len(period0) or len(map1) != len(period1):
        raise ValueError("Duplicate entity_id within a single period")

    surv_ids = sorted(map0.keys() & map1.keys())
    exit_ids = sorted(map0.keys() - map1.keys())
    entry_ids = sorted(map1.keys() - map0.keys())

    surv_n0 = [map0[k].total_requests for k in surv_ids]
    surv_b0 = [map0[k].bot_requests for k in surv_ids]
    surv_n1 = [map1[k].total_requests for k in surv_ids]
    surv_b1 = [map1[k].bot_requests for k in surv_ids]
    exit_n0 = [map0[k].total_requests for k in exit_ids]
    exit_b0 = [map0[k].bot_requests for k in exit_ids]
    entry_n1 = [map1[k].total_requests for k in entry_ids]
    entry_b1 = [map1[k].bot_requests for k in entry_ids]
    return surv_n0, surv_b0, surv_n1, surv_b1, exit_n0, exit_b0, entry_n1, entry_b1


def _decompose_python(
    surv_n0: Sequence[float],
    surv_b0: Sequence[float],
    surv_n1: Sequence[float],
    surv_b1: Sequence[float],
    exit_n0: Sequence[float],
    exit_b0: Sequence[float],
    entry_n1: Sequence[float],
    entry_b1: Sequence[float],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    sn0 = float(np.sum(surv_n0))
    sb0 = float(np.sum(surv_b0))
    sn1 = float(np.sum(surv_n1))
    sb1 = float(np.sum(surv_b1))
    xn0 = float(np.sum(exit_n0))
    xb0 = float(np.sum(exit_b0))
    mn1 = float(np.sum(entry_n1))
    mb1 = float(np.sum(entry_b1))

    tot0 = sn0 + xn0
    tot1 = sn1 + mn1
    r0 = (sb0 + xb0) / tot0 if tot0 > 0.0 else 0.0
    r1 = (sb1 + mb1) / tot1 if tot1 > 0.0 else 0.0
    sr0 = sb0 / sn0 if sn0 > 0.0 else 0.0
    sr1 = sb1 / sn1 if sn1 > 0.0 else 0.0

    within = 0.0
    realloc = 0.0
    if sn0 > 0.0 and sn1 > 0.0 and len(surv_n0) > 0:
        arr_n0 = np.asarray(surv_n0, dtype=np.float64)
        arr_b0 = np.asarray(surv_b0, dtype=np.float64)
        arr_n1 = np.asarray(surv_n1, dtype=np.float64)
        arr_b1 = np.asarray(surv_b1, dtype=np.float64)
        r0_i = arr_b0 / arr_n0
        r1_i = arr_b1 / arr_n1
        ws0 = arr_n0 / sn0
        ws1 = arr_n1 / sn1
        within = float(np.sum(0.5 * (ws0 + ws1) * (r1_i - r0_i)))
        realloc = float(np.sum((ws1 - ws0) * 0.5 * (r0_i + r1_i)))

    s0 = sn0 / tot0 if tot0 > 0.0 else 1.0
    s1 = sn1 / tot1 if tot1 > 0.0 else 1.0
    rx0 = xb0 / xn0 if xn0 > 0.0 else sr0
    rm1 = mb1 / mn1 if mn1 > 0.0 else sr1
    entry_sel = (1.0 - s1) * (rm1 - sr1)
    exit_sel = (1.0 - s0) * (sr0 - rx0)
    return r0, r1, r1 - r0, sr0, sr1, within, realloc, entry_sel, exit_sel


def decompose_panel(
    period0: Sequence[EntityTraffic],
    period1: Sequence[EntityTraffic],
    *,
    backend: Backend = "auto",
) -> PanelDecomposition:
    """Decompose aggregate bot-rate shift `R1 - R0` across an unbalanced two-period panel."""
    if not period0 or not period1:
        raise ValueError("period0 and period1 must both be non-empty")
    cohorts = _partition_cohorts(period0, period1)
    use_rust = (backend == "rust") or (backend == "auto" and USING_RUST)
    vals = require_rust().py_panel_decompose(*cohorts) if use_rust else _decompose_python(*cohorts)

    return PanelDecomposition(
        r0=vals[0],
        r1=vals[1],
        delta_total=vals[2],
        survivor_r0=vals[3],
        survivor_r1=vals[4],
        within_behavior=vals[5],
        survivor_reallocation=vals[6],
        entry_selection=vals[7],
        exit_selection=vals[8],
        n_survivors=len(cohorts[0]),
        n_exiters=len(cohorts[4]),
        n_entrants=len(cohorts[6]),
    )


def bootstrap_panel(
    period0: Sequence[EntityTraffic],
    period1: Sequence[EntityTraffic],
    *,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 42,
    backend: Backend = "auto",
) -> PanelBootstrapCI:
    """Compute Rubin Bayesian (Dirichlet) entity-cluster bootstrap confidence intervals."""
    point = decompose_panel(period0, period1, backend=backend)
    cohorts = _partition_cohorts(period0, period1)
    use_rust = (backend == "rust") or (backend == "auto" and USING_RUST)

    if use_rust:
        draws = np.asarray(
            require_rust().py_panel_bootstrap(*cohorts, n_boot, seed), dtype=np.float64
        )
    else:
        rng = np.random.default_rng(seed)
        sn0, sb0, sn1, sb1, xn0, xb0, mn1, mb1 = [np.asarray(c, dtype=np.float64) for c in cohorts]
        rows = np.empty((n_boot, 5), dtype=np.float64)
        for b in range(n_boot):
            ws = rng.exponential(1.0, size=len(sn0))
            wx = rng.exponential(1.0, size=len(xn0))
            wm = rng.exponential(1.0, size=len(mn1))
            vals = _decompose_python(
                (sn0 * ws).tolist(),
                (sb0 * ws).tolist(),
                (sn1 * ws).tolist(),
                (sb1 * ws).tolist(),
                (xn0 * wx).tolist(),
                (xb0 * wx).tolist(),
                (mn1 * wm).tolist(),
                (mb1 * wm).tolist(),
            )
            rows[b] = [vals[2], vals[5], vals[6], vals[7], vals[8]]
        draws = rows

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    comp_draws = draws[:, 2] + draws[:, 3] + draws[:, 4]

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(arr, [lo_q, hi_q])
        return float(lo), float(hi)

    return PanelBootstrapCI(
        point=point,
        delta_total_ci=_ci(draws[:, 0]),
        within_behavior_ci=_ci(draws[:, 1]),
        total_composition_ci=_ci(comp_draws),
        entry_selection_ci=_ci(draws[:, 3]),
        exit_selection_ci=_ci(draws[:, 4]),
    )


def wilson_interval(
    successes: float, total: float, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Exact Wilson score interval for a binomial proportion `p = successes / total`."""
    if total <= 0.0:
        raise ValueError("total must be > 0")
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / total + z2 / (4.0 * total * total))
    return max(0.0, center - half), min(1.0, center + half)
