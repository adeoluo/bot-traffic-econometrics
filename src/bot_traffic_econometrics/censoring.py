"""Partial-identification Manski bounds and EM estimation under interstitial challenge censoring.

Addresses Section 5 of Ogunnoiki (F5 Labs, 2026): when 23.9% to 79.5% of flagged bot requests
receive interstitial CAPTCHAs or JS challenges prior to client-side behavioral grading, naive
sophistication averages over uncensored traffic suffer severe non-random selection bias.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bot_traffic_econometrics._rust_bridge import USING_RUST, require_rust

Backend = Literal["auto", "rust", "python"]
TIER_SCORES: tuple[float, float, float] = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class FlowCensoringObs:
    """Observed sophistication counts and challenged (censored) count on one retail flow."""

    flow_name: str
    uncensored_low: float
    uncensored_medium: float
    uncensored_high: float
    challenged_censored: float
    policy_exposure: float = 1.0

    def __post_init__(self) -> None:
        vals = (
            self.uncensored_low,
            self.uncensored_medium,
            self.uncensored_high,
            self.challenged_censored,
        )
        if any(v < 0.0 for v in vals):
            raise ValueError(f"Counts must be >= 0 for {self.flow_name!r}")
        if sum(vals) <= 0.0:
            raise ValueError(f"Total flagged bot count must be > 0 for {self.flow_name!r}")
        if not (0.0 < self.policy_exposure <= 1.0):
            raise ValueError(f"policy_exposure must be in (0, 1] for {self.flow_name!r}")

    @property
    def total_bots(self) -> float:
        return (
            self.uncensored_low
            + self.uncensored_medium
            + self.uncensored_high
            + self.challenged_censored
        )

    @property
    def uncensored_total(self) -> float:
        return self.uncensored_low + self.uncensored_medium + self.uncensored_high

    @property
    def censoring_rate(self) -> float:
        return self.challenged_censored / self.total_bots


@dataclass(frozen=True)
class ManskiBounds:
    """Sharp distribution-free and monotone-selection bounds on latent sophistication `E[v_S]`."""

    naive_uncensored_index: float
    censoring_rate: float
    manski_lower: float
    manski_upper: float
    adverse_interception_bounds: tuple[float, float]
    evasion_trigger_bounds: tuple[float, float]

    @property
    def identification_width(self) -> float:
        return self.manski_upper - self.manski_lower


@dataclass(frozen=True)
class CensoredEMResult:
    """Joint multi-flow Expectation-Maximization estimates of latent bot sophistication."""

    flow_names: tuple[str, ...]
    latent_tier_probs: tuple[tuple[float, float, float], ...]
    latent_sophistication_index: tuple[float, ...]
    naive_uncensored_index: tuple[float, ...]
    tier_challenge_propensity: tuple[float, float, float]
    log_likelihood: float
    iterations: int


def manski_sophistication_bounds(obs: FlowCensoringObs) -> ManskiBounds:
    """Compute sharp Manski bounds and monotone-instrument selection bounds in `[0, 1]`."""
    c = obs.censoring_rate
    u_tot = obs.uncensored_total
    i_u = (0.5 * obs.uncensored_medium + 1.0 * obs.uncensored_high) / u_tot if u_tot > 0.0 else 0.5

    lower = (1.0 - c) * i_u
    upper = (1.0 - c) * i_u + c
    return ManskiBounds(
        naive_uncensored_index=i_u,
        censoring_rate=c,
        manski_lower=lower,
        manski_upper=upper,
        adverse_interception_bounds=(lower, i_u),
        evasion_trigger_bounds=(i_u, upper),
    )


def is_direction_identified(
    bounds_t0: ManskiBounds,
    bounds_t1: ManskiBounds,
    *,
    regime: Literal["manski", "adverse_interception", "evasion_trigger"] = "manski",
) -> Literal["strictly_increasing", "strictly_decreasing", "ambiguous"]:
    """Determine whether the sign of `I_t1 - I_t0` is point-identified without parametric priors."""
    if regime == "manski":
        lo0, hi0 = bounds_t0.manski_lower, bounds_t0.manski_upper
        lo1, hi1 = bounds_t1.manski_lower, bounds_t1.manski_upper
    elif regime == "adverse_interception":
        lo0, hi0 = bounds_t0.adverse_interception_bounds
        lo1, hi1 = bounds_t1.adverse_interception_bounds
    else:
        lo0, hi0 = bounds_t0.evasion_trigger_bounds
        lo1, hi1 = bounds_t1.evasion_trigger_bounds

    if lo1 > hi0:
        return "strictly_increasing"
    if hi1 < lo0:
        return "strictly_decreasing"
    return "ambiguous"


def _censored_em_python(
    uncensored_flat: Sequence[float],
    censored: Sequence[float],
    exposure: Sequence[float],
    max_iter: int,
    tol: float,
) -> tuple[list[float], tuple[float, float, float], float, int]:
    n_flows = len(censored)
    pi = np.full((n_flows, 3), 1.0 / 3.0, dtype=np.float64)
    u = np.asarray(uncensored_flat, dtype=np.float64).reshape(n_flows, 3)
    c = np.asarray(censored, dtype=np.float64)
    exp_arr = np.clip(np.asarray(exposure, dtype=np.float64), 1e-9, 1.0)
    eps = 1e-9

    for f in range(n_flows):
        usum = float(np.sum(u[f]))
        if usum > 0.0:
            pi[f] = (u[f] + 1e-3) / (usum + 3e-3)

    gamma = np.array([0.65, 0.45, 0.25], dtype=np.float64)

    def _loglik(pi_m: np.ndarray, g_v: np.ndarray) -> float:
        ll = 0.0
        for f in range(n_flows):
            ef = float(exp_arr[f])
            chal_p = 0.0
            for k in range(3):
                pfk = max(float(pi_m[f, k]), eps)
                gk = min(max(ef * float(g_v[k]), eps), 1.0 - eps)
                ufk = float(u[f, k])
                if ufk > 0.0:
                    ll += ufk * float(np.log(pfk * (1.0 - gk)))
                chal_p += pfk * gk
            if c[f] > 0.0:
                ll += float(c[f]) * float(np.log(max(chal_p, eps)))
        return ll

    prev_ll = _loglik(pi, gamma)
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        q = pi * gamma[None, :]
        denom = np.sum(q, axis=1, keepdims=True)
        q = np.where(denom > 0.0, q / np.maximum(denom, eps), 1.0 / 3.0)

        totals = np.sum(u, axis=1) + c
        for f in range(n_flows):
            if totals[f] > 0.0:
                pi[f] = np.clip((u[f] + c[f] * q[f]) / totals[f], eps, 1.0 - eps)
                pi[f] /= np.sum(pi[f])

        for k in range(3):
            c_fk = c * q[:, k]
            u_fk = u[:, k]
            if float(np.sum(c_fk)) <= 0.0:
                gamma[k] = eps
                continue
            if float(np.sum(u_fk)) <= 0.0:
                gamma[k] = 1.0 - eps
                continue
            lo, hi = eps, 1.0 - eps
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                score = float(
                    np.sum(c_fk / mid - (u_fk * exp_arr) / np.maximum(1.0 - exp_arr * mid, eps))
                )
                if score > 0.0:
                    lo = mid
                else:
                    hi = mid
            gamma[k] = 0.5 * (lo + hi)

        ll = _loglik(pi, gamma)
        if abs(ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
            prev_ll = ll
            break
        prev_ll = ll

    return (
        pi.reshape(-1).tolist(),
        (float(gamma[0]), float(gamma[1]), float(gamma[2])),
        prev_ll,
        iters,
    )


def fit_censored_sophistication_em(
    flows: Sequence[FlowCensoringObs],
    *,
    max_iter: int = 500,
    tol: float = 1e-10,
    backend: Backend = "auto",
) -> CensoredEMResult:
    """Jointly estimate flow-specific latent sophistication mixes and challenge propensities."""
    if len(flows) < 2:
        raise ValueError(
            "At least 2 flows are required to jointly identify shared challenge propensities"
        )

    uncensored_flat: list[float] = []
    censored: list[float] = []
    exposure: list[float] = []
    naive_idx: list[float] = []
    for f in flows:
        uncensored_flat.extend([f.uncensored_low, f.uncensored_medium, f.uncensored_high])
        censored.append(f.challenged_censored)
        exposure.append(f.policy_exposure)
        naive_idx.append(manski_sophistication_bounds(f).naive_uncensored_index)

    use_rust = (backend == "rust") or (backend == "auto" and USING_RUST)
    if use_rust:
        pi_flat, gamma_arr, ll, iters = require_rust().py_censored_em_fit(
            uncensored_flat, censored, exposure, max_iter, tol
        )
        gamma_tup = (float(gamma_arr[0]), float(gamma_arr[1]), float(gamma_arr[2]))
    else:
        pi_flat, gamma_tup, ll, iters = _censored_em_python(
            uncensored_flat, censored, exposure, max_iter, tol
        )

    probs: list[tuple[float, float, float]] = []
    latent_idx: list[float] = []
    for idx in range(len(flows)):
        p0 = float(pi_flat[idx * 3])
        p1 = float(pi_flat[idx * 3 + 1])
        p2 = float(pi_flat[idx * 3 + 2])
        probs.append((p0, p1, p2))
        latent_idx.append(0.5 * p1 + 1.0 * p2)

    return CensoredEMResult(
        flow_names=tuple(f.flow_name for f in flows),
        latent_tier_probs=tuple(probs),
        latent_sophistication_index=tuple(latent_idx),
        naive_uncensored_index=tuple(naive_idx),
        tier_challenge_propensity=gamma_tup,
        log_likelihood=float(ll),
        iterations=int(iters),
    )
