"""Reproducible Rust-vs-Python wall-clock & accuracy benchmark suite."""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bot_traffic_econometrics import (
    EntityTraffic,
    FlowCensoringObs,
    bootstrap_panel,
    fit_censored_sophistication_em,
    fit_multivariate_hawkes,
    hawkes_loglik_and_grad,
    simulate_multivariate_hawkes,
)
from bot_traffic_econometrics.cli import _summary_payload


def _median_ms(fn: Callable[[], object], repeats: int = 5) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1_000.0)
    samples.sort()
    return round(samples[len(samples) // 2], 3)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unbalanced panel Dirichlet cluster bootstrap (250 entities x 10,000 draws).
    rng = np.random.default_rng(2026)
    p0 = []
    p1 = []
    for i in range(220):
        n0 = float(rng.pareto(1.3) * 100_000 + 5_000)
        b0 = n0 * float(rng.uniform(0.01, 0.55))
        p0.append(EntityTraffic(f"r_{i}", n0, b0))
        if i < 180:
            n1 = n0 * float(rng.uniform(0.8, 1.4))
            b1 = n1 * float(rng.uniform(0.05, 0.60))
            p1.append(EntityTraffic(f"r_{i}", n1, b1))
    for j in range(40):
        n1 = float(rng.pareto(1.3) * 80_000 + 4_000)
        p1.append(EntityTraffic(f"e_{j}", n1, n1 * float(rng.uniform(0.02, 0.50))))

    boot_rs_ms = _median_ms(
        lambda: bootstrap_panel(p0, p1, n_boot=10_000, seed=42, backend="rust"), repeats=5
    )
    boot_py_ms = _median_ms(
        lambda: bootstrap_panel(p0, p1, n_boot=10_000, seed=42, backend="python"), repeats=3
    )

    # 2. 6-dimensional (Web/API x Search/Login/Checkout) Hawkes MLE over ~50k events.
    d = 6
    mu = np.array([2.4, 1.8, 0.9, 0.5, 0.4, 0.25], dtype=np.float64)
    alpha = np.full((d, d), 0.06, dtype=np.float64) + np.diag([0.35, 0.30, 0.28, 0.22, 0.20, 0.18])
    beta = 1.8
    horizon = 3_600.0
    times, marks = simulate_multivariate_hawkes(mu, alpha, beta=beta, horizon=horizon, seed=7)
    t_list = times.tolist()
    m_list = marks.tolist()

    ll_rs_ms = _median_ms(
        lambda: hawkes_loglik_and_grad(
            t_list, m_list, mu, alpha, beta=beta, horizon=horizon, backend="rust"
        ),
        repeats=7,
    )
    ll_py_ms = _median_ms(
        lambda: hawkes_loglik_and_grad(
            t_list, m_list, mu, alpha, beta=beta, horizon=horizon, backend="python"
        ),
        repeats=5,
    )

    em_rs_ms = _median_ms(
        lambda: fit_multivariate_hawkes(
            t_list,
            m_list,
            n_dims=d,
            beta=beta,
            horizon=horizon,
            max_iter=60,
            tol=0.0,
            backend="rust",
        ),
        repeats=3,
    )
    em_py_ms = _median_ms(
        lambda: fit_multivariate_hawkes(
            t_list,
            m_list,
            n_dims=d,
            beta=beta,
            horizon=horizon,
            max_iter=60,
            tol=0.0,
            backend="python",
        ),
        repeats=1,
    )

    # 3. 14-flow joint censored sophistication EM (300 iterations).
    flows = [
        FlowCensoringObs(
            f"flow_{k}",
            uncensored_low=float(2_000 + 300 * (k % 5)),
            uncensored_medium=float(1_200 + 200 * (k % 3)),
            uncensored_high=float(3_500 + 450 * (k % 4)),
            challenged_censored=float(4_000 + 800 * (k % 6)),
            policy_exposure=0.45 + 0.04 * k,
        )
        for k in range(14)
    ]
    cem_rs_ms = _median_ms(
        lambda: fit_censored_sophistication_em(flows, max_iter=300, tol=0.0, backend="rust"),
        repeats=7,
    )
    cem_py_ms = _median_ms(
        lambda: fit_censored_sophistication_em(flows, max_iter=300, tol=0.0, backend="python"),
        repeats=3,
    )

    load1, load5, load15 = os.getloadavg()
    bench = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "loadavg_1_5_15": [round(load1, 2), round(load5, 2), round(load15, 2)],
        },
        "workloads": {
            "panel_dirichlet_bootstrap_260_entities_10k_draws": {
                "rust_median_ms": boot_rs_ms,
                "python_median_ms": boot_py_ms,
                "speedup_x": round(boot_py_ms / boot_rs_ms, 1),
            },
            "hawkes_6d_loglik_grad_per_eval": {
                "n_events": len(t_list),
                "rust_median_ms": ll_rs_ms,
                "python_median_ms": ll_py_ms,
                "speedup_x": round(ll_py_ms / ll_rs_ms, 1),
            },
            "hawkes_6d_em_60_iters": {
                "n_events": len(t_list),
                "rust_median_ms": em_rs_ms,
                "python_median_ms": em_py_ms,
                "speedup_x": round(em_py_ms / em_rs_ms, 1),
            },
            "censored_14flow_em_300_iters": {
                "rust_median_ms": cem_rs_ms,
                "python_median_ms": cem_py_ms,
                "speedup_x": round(cem_py_ms / cem_rs_ms, 1),
            },
        },
    }

    (out_dir / "rust_speedup.json").write_text(json.dumps(bench, indent=2) + "\n")
    (out_dir / "f5_reproduction.json").write_text(json.dumps(_summary_payload(), indent=2) + "\n")
    print(json.dumps(bench, indent=2))


if __name__ == "__main__":
    main()
