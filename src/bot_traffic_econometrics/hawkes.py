"""O(N * D) multivariate marked Hawkes processes and retail flow-paradox metrics.

Models calendar-agnostic self-exciting bot bursts (e.g. inventory restock / credential-dump
cascades peaking at BF-23, BF+1, or BF-38 rather than calendar holidays) and cross-flow
rate-vs-volume-share divergence across retail user journeys.
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
class HawkesFit:
    """Maximum-likelihood fit of a D-dimensional exponential Hawkes process."""

    mu: np.ndarray
    alpha: np.ndarray
    beta: float
    log_likelihood: float
    spectral_radius: float
    iterations: int


def spectral_branching_ratio(alpha: np.ndarray, beta: float) -> float:
    """Return spectral radius `rho(alpha / beta)` (< 1 implies stationarity / sub-criticality)."""
    if beta <= 0.0:
        raise ValueError("beta must be > 0")
    eigvals = np.linalg.eigvals(alpha / beta)
    return float(np.max(np.abs(eigvals)))


def _hawkes_loglik_grad_python(
    times: Sequence[float],
    marks: Sequence[int],
    mu: Sequence[float],
    alpha_flat: Sequence[float],
    beta: float,
    horizon: float,
) -> tuple[float, list[float], list[float]]:
    d = len(mu)
    grad_mu = [-horizon for _ in range(d)]
    h_j = [0.0 for _ in range(d)]
    for t, m in zip(times, marks, strict=True):
        h_j[m] += (1.0 - math.exp(-beta * (horizon - t))) / beta

    ll = 0.0
    for i in range(d):
        ll -= mu[i] * horizon
        for j in range(d):
            ll -= alpha_flat[i * d + j] * h_j[j]

    grad_alpha = [-h_j[j] for _i in range(d) for j in range(d)]
    g = [0.0 for _ in range(d)]
    prev_t = 0.0
    for idx, (t, m) in enumerate(zip(times, marks, strict=True)):
        if idx > 0:
            decay = math.exp(-beta * (t - prev_t))
            for j in range(d):
                g[j] *= decay
            g[marks[idx - 1]] += decay
        row = m * d
        lam = mu[m] + sum(alpha_flat[row + j] * g[j] for j in range(d))
        safe_lam = max(lam, 1e-15)
        ll += math.log(safe_lam)
        inv = 1.0 / safe_lam
        grad_mu[m] += inv
        for j in range(d):
            grad_alpha[row + j] += g[j] * inv
        prev_t = t

    return ll, grad_mu, grad_alpha


def hawkes_loglik_and_grad(
    times: Sequence[float],
    marks: Sequence[int],
    mu: np.ndarray,
    alpha: np.ndarray,
    *,
    beta: float,
    horizon: float,
    backend: Backend = "auto",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate exact O(N * D) log-likelihood and gradients `(ll, grad_mu, grad_alpha)`."""
    d = int(mu.shape[0])
    if alpha.shape != (d, d):
        raise ValueError(f"alpha must have shape ({d}, {d}), got {alpha.shape}")
    if beta <= 0.0 or horizon <= 0.0:
        raise ValueError("beta and horizon must be > 0")

    t_list = [float(t) for t in times]
    m_list = [int(m) for m in marks]
    mu_list = [float(v) for v in mu.reshape(-1)]
    alpha_list = [float(v) for v in alpha.reshape(-1)]

    use_rust = (backend == "rust") or (backend == "auto" and USING_RUST)
    if use_rust:
        ll, g_mu, g_alpha = require_rust().py_hawkes_loglik_grad(
            t_list, m_list, mu_list, alpha_list, float(beta), float(horizon)
        )
    else:
        ll, g_mu, g_alpha = _hawkes_loglik_grad_python(
            t_list, m_list, mu_list, alpha_list, float(beta), float(horizon)
        )

    return (
        float(ll),
        np.asarray(g_mu, dtype=np.float64),
        np.asarray(g_alpha, dtype=np.float64).reshape(d, d),
    )


def _hawkes_fit_em_python(
    times: Sequence[float],
    marks: Sequence[int],
    d: int,
    beta: float,
    horizon: float,
    max_iter: int,
    tol: float,
) -> tuple[list[float], list[float], float, int]:
    counts = [0.0 for _ in range(d)]
    h_j = [0.0 for _ in range(d)]
    for t, m in zip(times, marks, strict=True):
        counts[m] += 1.0
        h_j[m] += (1.0 - math.exp(-beta * (horizon - t))) / beta

    mu = [max((0.5 * c) / max(horizon, 1e-9), 1e-4) for c in counts]
    alpha = [0.05 * beta / max(d, 1) for _ in range(d * d)]
    prev_ll = -math.inf
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        g = [0.0 for _ in range(d)]
        m_sum = [0.0 for _ in range(d)]
        s_sum = [0.0 for _ in range(d * d)]
        ll = -sum(mu[i] * horizon for i in range(d)) - sum(
            alpha[i * d + j] * h_j[j] for i in range(d) for j in range(d)
        )
        prev_t = 0.0
        for idx, (t, m) in enumerate(zip(times, marks, strict=True)):
            if idx > 0:
                decay = math.exp(-beta * (t - prev_t))
                for j in range(d):
                    g[j] *= decay
                g[marks[idx - 1]] += decay
            row = m * d
            lam = max(mu[m] + sum(alpha[row + j] * g[j] for j in range(d)), 1e-15)
            ll += math.log(lam)
            inv = 1.0 / lam
            m_sum[m] += mu[m] * inv
            for j in range(d):
                s_sum[row + j] += alpha[row + j] * g[j] * inv
            prev_t = t

        for i in range(d):
            mu[i] = max(m_sum[i] / horizon, 1e-12)
            for j in range(d):
                alpha[i * d + j] = max(s_sum[i * d + j] / h_j[j], 0.0) if h_j[j] > 0.0 else 0.0

        if math.isfinite(prev_ll) and abs(ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = ll

    final_ll, _, _ = _hawkes_loglik_grad_python(times, marks, mu, alpha, beta, horizon)
    return mu, alpha, final_ll, iters


def fit_multivariate_hawkes(
    times: Sequence[float],
    marks: Sequence[int],
    *,
    n_dims: int,
    beta: float,
    horizon: float,
    max_iter: int = 300,
    tol: float = 1e-8,
    backend: Backend = "auto",
) -> HawkesFit:
    """Fit a D-dimensional exponential Hawkes process via exact O(N * D) Lewis-Mohler EM."""
    if n_dims < 1 or beta <= 0.0 or horizon <= 0.0:
        raise ValueError("n_dims >= 1, beta > 0, and horizon > 0 are required")
    t_list = [float(t) for t in times]
    m_list = [int(m) for m in marks]

    use_rust = (backend == "rust") or (backend == "auto" and USING_RUST)
    if use_rust:
        mu_v, alpha_v, ll, iters = require_rust().py_hawkes_fit_em(
            t_list, m_list, n_dims, float(beta), float(horizon), max_iter, tol
        )
    else:
        mu_v, alpha_v, ll, iters = _hawkes_fit_em_python(
            t_list, m_list, n_dims, float(beta), float(horizon), max_iter, tol
        )

    mu_arr = np.asarray(mu_v, dtype=np.float64)
    alpha_arr = np.asarray(alpha_v, dtype=np.float64).reshape(n_dims, n_dims)
    return HawkesFit(
        mu=mu_arr,
        alpha=alpha_arr,
        beta=float(beta),
        log_likelihood=float(ll),
        spectral_radius=spectral_branching_ratio(alpha_arr, float(beta)),
        iterations=int(iters),
    )


def simulate_multivariate_hawkes(
    mu: np.ndarray,
    alpha: np.ndarray,
    *,
    beta: float,
    horizon: float,
    seed: int = 42,
    max_events: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `(times, marks)` from a D-dimensional exponential Hawkes process."""
    d = int(mu.shape[0])
    if alpha.shape != (d, d):
        raise ValueError(f"alpha must have shape ({d}, {d})")
    if USING_RUST:
        times, marks = require_rust().py_simulate_hawkes(
            mu.reshape(-1).tolist(),
            alpha.reshape(-1).tolist(),
            float(beta),
            float(horizon),
            int(seed),
            int(max_events),
        )
        return np.asarray(times, dtype=np.float64), np.asarray(marks, dtype=np.int64)

    rng = np.random.default_rng(seed)
    t_out: list[float] = []
    m_out: list[int] = []
    excit = np.zeros(d, dtype=np.float64)
    mu_sum = float(np.sum(mu))
    t = 0.0
    while t < horizon and len(t_out) < max_events:
        lam_bar = mu_sum + float(np.sum(excit))
        if lam_bar <= 0.0:
            break
        dt = float(rng.exponential(1.0 / lam_bar))
        if t + dt >= horizon:
            break
        excit *= math.exp(-beta * dt)
        t += dt
        lam_vec = mu + excit
        lam_now = float(np.sum(lam_vec))
        if float(rng.random()) * lam_bar <= lam_now:
            chosen = int(rng.choice(d, p=lam_vec / lam_now))
            t_out.append(t)
            m_out.append(chosen)
            excit += alpha[:, chosen]
    return np.asarray(t_out, dtype=np.float64), np.asarray(m_out, dtype=np.int64)


@dataclass(frozen=True)
class FlowRiskSummary:
    """Contrasts per-flow bot rate (`B_f / N_f`) against bot volume share (`B_f / sum_g B_g`)."""

    flow_name: str
    bot_rate: float
    bot_volume_share: float
    rate_rank: int
    volume_share_rank: int


def summarize_flow_paradox(
    flows: Sequence[tuple[str, float, float]],
) -> tuple[FlowRiskSummary, ...]:
    """Rank flows simultaneously by within-flow bot rate and share of total bot volume.

    Captures the Ecommerce/QSR paradox in Ogunnoiki (2026): Ecommerce `add_to_cart` has a
    54.2% bot rate while `session_create` holds 85% of bot volume; QSR `gift_card` has a
    10.1% bot rate while `login` holds 77% of bot volume.
    """
    rates = [b / tot for _, tot, b in flows]
    tot_bots = sum(b for _, _, b in flows)
    shares = [b / tot_bots if tot_bots > 0.0 else 0.0 for _, _, b in flows]

    rate_order = sorted(range(len(flows)), key=lambda i: rates[i], reverse=True)
    share_order = sorted(range(len(flows)), key=lambda i: shares[i], reverse=True)
    rate_ranks = {idx: rank + 1 for rank, idx in enumerate(rate_order)}
    share_ranks = {idx: rank + 1 for rank, idx in enumerate(share_order)}

    return tuple(
        FlowRiskSummary(
            flow_name=flows[i][0],
            bot_rate=rates[i],
            bot_volume_share=shares[i],
            rate_rank=rate_ranks[i],
            volume_share_rank=share_ranks[i],
        )
        for i in range(len(flows))
    )
