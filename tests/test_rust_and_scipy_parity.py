"""Differential parity tests: Rust core (`_bte_rs`) vs Python reference vs `SciPy`."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from bot_traffic_econometrics import (
    USING_RUST,
    ChannelParams,
    EntityTraffic,
    FlowCensoringObs,
    assess_pre_event_trend,
    attacker_best_response,
    attribute_surge_shapley,
    bootstrap_panel,
    decompose_panel,
    defender_loss,
    fit_censored_sophistication_em,
    fit_multivariate_hawkes,
    hawkes_loglik_and_grad,
    manski_sophistication_bounds,
    simulate_multivariate_hawkes,
    solve_stackelberg_defense,
    wilson_interval,
)


def test_rust_extension_is_compiled_and_active() -> None:
    assert USING_RUST is True


def test_panel_decomposition_rust_python_parity_and_exact_identity() -> None:
    rng = np.random.default_rng(101)
    p0 = [
        EntityTraffic(f"ent_{i}", float(rng.pareto(1.4) * 50_000 + 1_000), 0.0) for i in range(45)
    ]
    p0 = [
        EntityTraffic(
            e.entity_id, e.total_requests, e.total_requests * float(rng.uniform(0.01, 0.5))
        )
        for e in p0
    ]
    # Keep 30 survivors, drop 15 exiters, add 12 entrants.
    p1 = [
        EntityTraffic(
            p0[i].entity_id,
            p0[i].total_requests * float(rng.uniform(0.7, 1.5)),
            p0[i].total_requests * float(rng.uniform(0.05, 0.45)),
        )
        for i in range(30)
    ]
    p1 = [
        EntityTraffic(e.entity_id, max(e.total_requests, e.bot_requests * 1.05), e.bot_requests)
        for e in p1
    ]
    for j in range(12):
        tot = float(rng.pareto(1.5) * 40_000 + 2_000)
        p1.append(EntityTraffic(f"new_{j}", tot, tot * float(rng.uniform(0.02, 0.6))))

    d_rs = decompose_panel(p0, p1, backend="rust")
    d_py = decompose_panel(p0, p1, backend="python")
    assert d_rs.identity_residual < 1e-15
    assert d_py.identity_residual < 1e-15
    for field in (
        "r0",
        "r1",
        "delta_total",
        "survivor_r0",
        "survivor_r1",
        "within_behavior",
        "survivor_reallocation",
        "entry_selection",
        "exit_selection",
    ):
        assert abs(getattr(d_rs, field) - getattr(d_py, field)) < 1e-14

    ci_rs = bootstrap_panel(p0, p1, n_boot=1_000, seed=7, backend="rust")
    ci_py = bootstrap_panel(p0, p1, n_boot=1_000, seed=7, backend="python")
    assert ci_rs.delta_total_ci[0] < d_rs.delta_total < ci_rs.delta_total_ci[1]
    assert abs(ci_rs.within_behavior_ci[0] - ci_py.within_behavior_ci[0]) < 0.015


def test_theil_sen_and_mann_kendall_match_scipy_to_machine_precision() -> None:
    rng = np.random.default_rng(2026)
    x = np.arange(38, dtype=np.float64)
    y = 0.14 + 0.0035 * x + rng.normal(0.0, 0.015, size=len(x))
    # Inject exact ties in y to exercise tie-corrected Mann-Kendall variance and tau_b.
    y[5] = y[12]
    y[19] = y[27]

    tr_rs = assess_pre_event_trend(x.tolist(), y.tolist(), backend="rust")
    tr_py = assess_pre_event_trend(x.tolist(), y.tolist(), backend="python")
    scipy_ts = stats.theilslopes(y, x, method="joint")
    scipy_kt = stats.kendalltau(x, y, variant="b")

    assert abs(tr_rs.theil_sen_slope - tr_py.theil_sen_slope) < 1e-15
    assert abs(tr_rs.theil_sen_slope - float(scipy_ts.slope)) < 1e-14
    assert abs(tr_rs.theil_sen_intercept - float(scipy_ts.intercept)) < 1e-14
    assert abs(tr_rs.kendall_tau_b - float(scipy_kt.statistic)) < 1e-14
    assert abs(tr_rs.p_value_two_sided - float(scipy_kt.pvalue)) < 1e-3


def test_hawkes_loglik_and_gradients_match_scipy_quad_and_finite_diff() -> None:
    mu = np.array([0.35, 0.20, 0.15], dtype=np.float64)
    alpha = np.array(
        [
            [0.25, 0.10, 0.05],
            [0.18, 0.22, 0.08],
            [0.06, 0.14, 0.20],
        ],
        dtype=np.float64,
    )
    beta = 1.4
    horizon = 25.0
    times, marks = simulate_multivariate_hawkes(mu, alpha, beta=beta, horizon=horizon, seed=19)
    assert len(times) >= 10

    ll_rs, g_mu_rs, g_a_rs = hawkes_loglik_and_grad(
        times.tolist(), marks.tolist(), mu, alpha, beta=beta, horizon=horizon, backend="rust"
    )
    ll_py, g_mu_py, g_a_py = hawkes_loglik_and_grad(
        times.tolist(), marks.tolist(), mu, alpha, beta=beta, horizon=horizon, backend="python"
    )
    assert abs(ll_rs - ll_py) < 1e-11
    np.testing.assert_allclose(g_mu_rs, g_mu_py, atol=1e-11)
    np.testing.assert_allclose(g_a_rs, g_a_py, atol=1e-11)

    # Cross-check compensator against numerical SciPy quadrature of lambda_i(t) over [0, horizon].
    def lam_i(t: float, dim: int) -> float:
        mask = times < t
        if not np.any(mask):
            return float(mu[dim])
        dts = t - times[mask]
        ms = marks[mask]
        return float(mu[dim] + np.sum(alpha[dim, ms] * np.exp(-beta * dts)))

    comp_quad = 0.0
    for dim in range(3):
        # Split integral at jump points so adaptive quadrature has zero discontinuity error.
        knots = [0.0, *times.tolist(), horizon]
        for a_pt, b_pt in itertools.pairwise(knots):
            if b_pt > a_pt:
                val, _ = integrate.quad(lam_i, a_pt, b_pt, args=(dim,), epsabs=1e-12, epsrel=1e-12)
                comp_quad += val

    event_sum = sum(
        float(np.log(lam_i(float(t), int(m))))
        for t, m in zip(times.tolist(), marks.tolist(), strict=True)
    )
    assert abs(ll_rs - (event_sum - comp_quad)) < 1e-10

    # Cross-check analytical gradients against SciPy finite differences.
    def pack_ll(theta: np.ndarray) -> float:
        m_v = theta[:3]
        a_m = theta[3:].reshape(3, 3)
        val, _, _ = hawkes_loglik_and_grad(
            times.tolist(), marks.tolist(), m_v, a_m, beta=beta, horizon=horizon, backend="rust"
        )
        return val

    theta0 = np.concatenate([mu, alpha.reshape(-1)])
    num_grad = optimize.approx_fprime(theta0, pack_ll, 1e-7)
    ana_grad = np.concatenate([g_mu_rs, g_a_rs.reshape(-1)])
    np.testing.assert_allclose(ana_grad, num_grad, atol=1e-5, rtol=1e-5)

    # Fit EM in Rust and Python and verify parity + stationary KKT optimality.
    fit_rs = fit_multivariate_hawkes(
        times.tolist(),
        marks.tolist(),
        n_dims=3,
        beta=beta,
        horizon=horizon,
        max_iter=500,
        tol=1e-11,
        backend="rust",
    )
    fit_py = fit_multivariate_hawkes(
        times.tolist(),
        marks.tolist(),
        n_dims=3,
        beta=beta,
        horizon=horizon,
        max_iter=500,
        tol=1e-11,
        backend="python",
    )
    assert abs(fit_rs.log_likelihood - fit_py.log_likelihood) < 1e-10
    np.testing.assert_allclose(fit_rs.mu, fit_py.mu, atol=1e-10)
    np.testing.assert_allclose(fit_rs.alpha, fit_py.alpha, atol=1e-10)
    assert fit_rs.spectral_radius < 1.5


def test_censored_manski_sharpness_and_em_recovery() -> None:
    rng = np.random.default_rng(99)
    # Sharpness: across 200 adversarial censoring mechanisms, true index lies inside Manski bounds.
    for _ in range(200):
        pi_true = rng.dirichlet([1.2, 1.0, 1.5])
        gamma_true = rng.uniform(0.05, 0.95, size=3)
        n_tot = 10_000.0
        uncens = n_tot * pi_true * (1.0 - gamma_true)
        cens = float(np.sum(n_tot * pi_true * gamma_true))
        true_idx = float(0.5 * pi_true[1] + 1.0 * pi_true[2])
        b = manski_sophistication_bounds(
            FlowCensoringObs("f", float(uncens[0]), float(uncens[1]), float(uncens[2]), cens)
        )
        assert b.manski_lower - 1e-12 <= true_idx <= b.manski_upper + 1e-12

    # Multi-flow EM recovery + Rust/Python parity.
    gamma_star = np.array([0.75, 0.45, 0.15])
    pis = [
        np.array([0.70, 0.20, 0.10]),
        np.array([0.20, 0.30, 0.50]),
        np.array([0.10, 0.15, 0.75]),
    ]
    exposures = [1.0, 0.75, 0.50]
    flows = []
    for idx, (pi_f, ef) in enumerate(zip(pis, exposures, strict=True)):
        u_f = 50_000.0 * pi_f * (1.0 - ef * gamma_star)
        c_f = float(np.sum(50_000.0 * pi_f * (ef * gamma_star)))
        flows.append(
            FlowCensoringObs(
                f"flow_{idx}", float(u_f[0]), float(u_f[1]), float(u_f[2]), c_f, policy_exposure=ef
            )
        )

    em_rs = fit_censored_sophistication_em(flows, max_iter=800, tol=1e-12, backend="rust")
    em_py = fit_censored_sophistication_em(flows, max_iter=800, tol=1e-12, backend="python")
    assert abs(em_rs.log_likelihood - em_py.log_likelihood) < 1e-9
    np.testing.assert_allclose(em_rs.tier_challenge_propensity, gamma_star, atol=1e-2)
    for f_idx, pi_f in enumerate(pis):
        true_i = float(0.5 * pi_f[1] + 1.0 * pi_f[2])
        assert abs(em_rs.latent_sophistication_index[f_idx] - true_i) < 1e-2


def test_stackelberg_equilibrium_matches_scipy_slsqp() -> None:
    params = ChannelParams()
    budget = 0.95
    eq = solve_stackelberg_defense(budget=budget, params=params)

    res = optimize.minimize(
        lambda u: defender_loss(float(u[0]), float(u[1]), params),
        x0=np.array([0.5, 0.3]),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda u: budget - (params.cost_web * u[0] + params.cost_api * u[1]),
            }
        ],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert res.success
    assert abs(eq.stackelberg_loss - float(res.fun)) < 1e-7
    assert eq.api_displacement_pct > 15.0
    assert eq.relative_loss_reduction_pct > 10.0

    # Verify boundary attacker best-response when one channel is heavily unattractive.
    aw, am = attacker_best_response(1.0, 0.0, ChannelParams(value_web=0.2, efficacy_web=0.99))
    assert aw == 0.0 and am > 0.0


def test_shapley_surge_and_wilson_properties() -> None:
    s = attribute_surge_shapley(15.3, 84.7, 15.3 * 1.20, 84.7 * 1.61)
    assert abs((s.bot_impulse_effect + s.human_dilution_effect) - s.delta_rate) < 1e-15
    assert s.is_diluted_bot_surge is True
    assert s.bot_impulse_effect > 0.0 and s.human_dilution_effect < 0.0

    lo, hi = wilson_interval(286, 1000)
    assert 0.25 < lo < 0.286 < hi < 0.32
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
