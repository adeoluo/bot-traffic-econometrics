"""A Stackelberg defender-attacker game for displacement between Web and Mobile API.

The model is a thought experiment about a mechanism the public report raises: if Web
endpoints are hardened and Mobile API endpoints are not, a cost-minimizing attacker shifts
work to the API. Nothing here is estimated from data. Every parameter in `ChannelParams` is
an illustrative default, and the size of the displacement the model produces is set almost
entirely by one of them, the substitution coupling. `sensitivity_sweep` exists to make that
visible: across its grid the displacement ranges over more than an order of magnitude, so the
model supports a direction (more than zero whenever the channels are coupled) and no number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelParams:
    """Parameters of the two-channel game. All defaults are illustrative, none is estimated."""

    value_web: float = 1.0
    value_api: float = 0.85
    efficacy_web: float = 0.78
    efficacy_api: float = 0.82
    congestion_web: float = 0.45
    congestion_api: float = 1.60
    substitution_coupling: float = 0.55
    cost_web: float = 1.0
    cost_api: float = 1.15
    damage_weight_web: float = 1.0
    damage_weight_api: float = 1.35


@dataclass(frozen=True)
class StackelbergEquilibrium:
    """Comparison of Web-only rate-chasing vs Stackelberg-optimal joint Web + Mobile API defense."""

    budget: float
    web_only_controls: tuple[float, float]
    web_only_attacks: tuple[float, float]
    web_only_loss: float
    stackelberg_controls: tuple[float, float]
    stackelberg_attacks: tuple[float, float]
    stackelberg_loss: float
    api_displacement_pct: float
    relative_loss_reduction_pct: float


def attacker_best_response(
    u_web: float,
    u_api: float,
    params: ChannelParams,
) -> tuple[float, float]:
    """Exact closed-form KKT best-response `(a_web, a_api) >= 0` for a rational adversary.

    Maximizes:
        m_W * a_W + m_M * a_M - 0.5 * k_W * a_W^2 - 0.5 * k_M * a_M^2 - 0.5 * gamma * (a_W + a_M)^2
    where `m_k = value_k * (1 - efficacy_k * u_k)`.
    """
    mw = params.value_web * (1.0 - params.efficacy_web * np.clip(u_web, 0.0, 1.0))
    mm = params.value_api * (1.0 - params.efficacy_api * np.clip(u_api, 0.0, 1.0))
    kw = params.congestion_web
    km = params.congestion_api
    g = params.substitution_coupling

    # Interior 2x2 linear system: [[kw + g, g], [g, km + g]] [aw, am]^T = [mw, mm]^T
    det = (kw + g) * (km + g) - g * g
    aw_int = ((km + g) * mw - g * mm) / det
    am_int = ((kw + g) * mm - g * mw) / det
    if aw_int >= 0.0 and am_int >= 0.0:
        return float(aw_int), float(am_int)

    # Check active non-negativity boundaries.
    cand_w = (max(0.0, mw / (kw + g)), 0.0)
    cand_m = (0.0, max(0.0, mm / (km + g)))
    cand_0 = (0.0, 0.0)

    def _util(aw: float, am: float) -> float:
        return float(
            mw * aw + mm * am - 0.5 * kw * aw * aw - 0.5 * km * am * am - 0.5 * g * (aw + am) ** 2
        )

    best = max((cand_w, cand_m, cand_0), key=lambda pair: _util(pair[0], pair[1]))
    return float(best[0]), float(best[1])


def defender_loss(u_web: float, u_api: float, params: ChannelParams) -> float:
    """Residual successful attack damage weighted across Web and Mobile API."""
    aw, am = attacker_best_response(u_web, u_api, params)
    rem_w = 1.0 - params.efficacy_web * np.clip(u_web, 0.0, 1.0)
    rem_m = 1.0 - params.efficacy_api * np.clip(u_api, 0.0, 1.0)
    return float(params.damage_weight_web * rem_w * aw + params.damage_weight_api * rem_m * am)


def solve_stackelberg_defense(
    budget: float,
    params: ChannelParams | None = None,
    *,
    grid_points: int = 2_001,
) -> StackelbergEquilibrium:
    """Solve the budget-constrained Stackelberg leader equilibrium vs Web-only rate-chasing.

    Because defender loss is monotonically decreasing along the budget frontier
    `cost_web * u_web + cost_api * u_api <= budget` on `[0, 1]^2` (or saturates at `(1, 1)`),
    evaluating the exact frontier + golden-section refinement solves the 1-D frontier optimum
    to `< 1e-11` without local-minimum traps.
    """
    p = params if params is not None else ChannelParams()
    if budget <= 0.0:
        raise ValueError("budget must be > 0")

    _, am_unprotected = attacker_best_response(0.0, 0.0, p)
    uw_only = min(1.0, budget / p.cost_web)
    aw_wo, am_wo = attacker_best_response(uw_only, 0.0, p)
    loss_wo = defender_loss(uw_only, 0.0, p)

    if budget >= p.cost_web + p.cost_api:
        best_uw, best_um = 1.0, 1.0
    else:
        uw_min = max(0.0, (budget - p.cost_api) / p.cost_web)
        uw_max = min(1.0, budget / p.cost_web)
        uws = np.linspace(uw_min, uw_max, grid_points)
        losses = np.array(
            [
                defender_loss(
                    float(uw), min(1.0, (budget - p.cost_web * float(uw)) / p.cost_api), p
                )
                for uw in uws
            ]
        )
        idx = int(np.argmin(losses))
        lo = float(uws[max(0, idx - 1)])
        hi = float(uws[min(len(uws) - 1, idx + 1)])
        inv_phi = (5.0**0.5 - 1.0) / 2.0
        for _ in range(60):
            c1 = hi - inv_phi * (hi - lo)
            c2 = lo + inv_phi * (hi - lo)
            u1 = min(1.0, (budget - p.cost_web * c1) / p.cost_api)
            u2 = min(1.0, (budget - p.cost_web * c2) / p.cost_api)
            if defender_loss(c1, u1, p) < defender_loss(c2, u2, p):
                hi = c2
            else:
                lo = c1
        best_uw = 0.5 * (lo + hi)
        best_um = min(1.0, max(0.0, (budget - p.cost_web * best_uw) / p.cost_api))

    aw_st, am_st = attacker_best_response(best_uw, best_um, p)
    loss_st = defender_loss(best_uw, best_um, p)
    disp_pct = 100.0 * (am_wo - am_unprotected) / am_unprotected if am_unprotected > 0.0 else 0.0
    rel_red = 100.0 * (loss_wo - loss_st) / loss_wo if loss_wo > 0.0 else 0.0

    return StackelbergEquilibrium(
        budget=budget,
        web_only_controls=(float(uw_only), 0.0),
        web_only_attacks=(aw_wo, am_wo),
        web_only_loss=loss_wo,
        stackelberg_controls=(float(best_uw), float(best_um)),
        stackelberg_attacks=(aw_st, am_st),
        stackelberg_loss=loss_st,
        api_displacement_pct=disp_pct,
        relative_loss_reduction_pct=rel_red,
    )


@dataclass(frozen=True)
class SweepCell:
    """Range of the two outputs over every grid point that shares one substitution coupling."""

    substitution_coupling: float
    grid_points: int
    displacement_pct_min: float
    displacement_pct_median: float
    displacement_pct_max: float
    loss_reduction_pct_min: float
    loss_reduction_pct_median: float
    loss_reduction_pct_max: float


SWEEP_COUPLINGS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.55, 0.8, 1.2)
SWEEP_API_EFFICACY: tuple[float, ...] = (0.5, 0.7, 0.82, 0.95)
SWEEP_API_CONGESTION: tuple[float, ...] = (0.8, 1.6, 3.0)
SWEEP_BUDGETS: tuple[float, ...] = (0.5, 0.95, 1.5)


def sensitivity_sweep(*, grid_points: int = 401) -> tuple[SweepCell, ...]:
    """Solve the game over a grid of parameters and report the spread of its two outputs.

    For each substitution coupling, the API control efficacy, the API congestion cost and
    the defense budget are varied over the `SWEEP_*` grids (36 combinations). The point is
    how wide the ranges are: the displacement figure is a function of the parameters chosen,
    not a finding.
    """
    cells: list[SweepCell] = []
    for coupling in SWEEP_COUPLINGS:
        displacement: list[float] = []
        reduction: list[float] = []
        for efficacy in SWEEP_API_EFFICACY:
            for congestion in SWEEP_API_CONGESTION:
                for budget in SWEEP_BUDGETS:
                    params = ChannelParams(
                        substitution_coupling=coupling,
                        efficacy_api=efficacy,
                        congestion_api=congestion,
                    )
                    eq = solve_stackelberg_defense(budget, params, grid_points=grid_points)
                    displacement.append(eq.api_displacement_pct)
                    reduction.append(eq.relative_loss_reduction_pct)
        cells.append(
            SweepCell(
                substitution_coupling=coupling,
                grid_points=len(displacement),
                displacement_pct_min=float(np.min(displacement)),
                displacement_pct_median=float(np.median(displacement)),
                displacement_pct_max=float(np.max(displacement)),
                loss_reduction_pct_min=float(np.min(reduction)),
                loss_reduction_pct_median=float(np.median(reduction)),
                loss_reduction_pct_max=float(np.max(reduction)),
            )
        )
    return tuple(cells)
