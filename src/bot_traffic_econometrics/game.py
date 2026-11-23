"""Stackelberg defender-attacker game for cross-channel (Web vs Mobile API) bot displacement.

Formalizes the channel-asymmetry and protection-lag mechanism analyzed in Ogunnoiki (F5 Labs,
2026) and Murphy (F5 Field CISO Blog, 2026): hardening Web endpoints without co-allocating
Mobile API defenses displaces rational cost-minimizing attackers onto Mobile APIs, compressing
the Web-to-Mobile-API bot ratio (33x -> 14x -> 12x) while raising unmitigated API risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelParams:
    """Structural parameters of the two-channel Stackelberg displacement game."""

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
