<h1 align="center">Bot Traffic Econometrics</h1>

<p align="center">
  <em>A methods library for bot-traffic measurement: unbalanced-panel rate decomposition, Manski bounds under censored telemetry, O(N * D) multivariate Hawkes processes, and a Stackelberg channel game (compiled Rust core + typed Python). It ships synthetic examples only.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/bot-traffic-econometrics/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/bot-traffic-econometrics/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/bot-traffic-econometrics/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/bot-traffic-econometrics"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue">
  <img alt="Rust" src="https://img.shields.io/badge/rust-PyO3%20%2B%20Rayon-DEA584">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## What this repository is, and is not

Security telemetry looks simple until you ask whether a headline rate moved because
**attackers changed behavior**, **the set of customers being measured changed**, **human
traffic diluted the denominator**, or **challenge pages kept client-side sensors from
running**. I ran into each of those while writing a threat-research report, and this library
implements general-purpose estimators for them.

- **It is** a tested implementation of five methods, each checked against `SciPy` or an
  algebraic identity, with a compiled Rust core.
- **It is not the analysis behind the report.** The report was produced inside F5 on customer
  telemetry with F5's own tooling. None of that data or code is here. The Hawkes, Manski and
  Stackelberg parts are my own extensions of questions the report raises, not methods taken
  from it.
- **It does not reproduce the report's results.** The bundled example uses invented retailer
  panels sized so that their aggregates equal figures printed in the public report (see
  [Worked example](#worked-example-on-synthetic-data-calibrated-to-the-public-report)). That
  shows how the methods behave on inputs with those aggregates and nothing about real
  retailers.
- **Evidence level:** unit tested and validated against reference implementations. Not run on
  real telemetry in this repository, not independently reproduced, not used in production.

The report and two articles that discuss it:

- Adeolu Ogunnoiki, [*Automation leading up to Black Friday*](https://www.f5.com/labs/articles/2026-black-friday-bot-report),
  F5 Labs, Nov 18, 2026 ([PDF](https://cdn.sanity.io/files/k6fem79d/translations-production/1d1028afb94e20d907a0b98ef084535e84fc9cd5.pdf) ·
  [author page](https://www.f5.com/labs/authors/adeolu-ogunnoiki)).
- Sean Murphy (Field CISO, North America, F5), [*The Black Friday automation trend everyone gets wrong*](https://www.f5.com/company/blog/the-black-friday-automation-trend-everyone-gets-wrong),
  F5 blog, Nov 19, 2026.
- Alexander Culafi, [*Why Holiday Security Can't Wait for Black Friday*](https://www.darkreading.com/cybersecurity-operations/why-holiday-security-cant-wait-black-friday),
  Dark Reading, Nov 19, 2026. The article covers the F5 study; it does not name me.

---

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="880">
</p>

<sub>Legend: amber = telemetry inputs and entry points, indigo = compiled Rust hot-path engines (`_bte_rs`), blue = causal identification and equilibrium solvers, green = differential `SciPy` parity gate, grey = empirical and policy outputs.</sub>

Every install compiles the **Rust extension (`bot_traffic_econometrics._bte_rs`, PyO3 + Rayon)**
and routes hot paths through it automatically (`USING_RUST == True`), while keeping pure-Python /
NumPy reference implementations in-tree so every estimator is differential-tested against both
the Python reference and `SciPy` in CI.

---

## Five Measurement Traps & Mathematical Formulations

### 1. Cohort Churn vs Behavioral Shift (`panel.py` + Rust Rayon Bootstrap)

When enterprise panels gain and lose heavy-tailed tenants across seasons, headline rate shifts
$\Delta R = R_1 - R_0$ conflate attacker behavior with entry/exit selection. Partitioning entities
into **Survivors** $S = E_0 \cap E_1$, **Exiters** $X = E_0 \setminus E_1$, and **Entrants**
$M = E_1 \setminus E_0$ yields an **algebraically exact (zero-residual)** 4-way Melitz-Polanec +
symmetric Kitagawa decomposition:

$$\Delta R = \underbrace{\sum_{i \in S} \bar{w}^S_i (r_{i,1} - r_{i,0})}_{\text{Within-survivor behavior}} + \underbrace{\sum_{i \in S} (w^S_{i,1} - w^S_{i,0}) \bar{r}_i}_{\text{Survivor share reallocation}} + \underbrace{(1 - s_1)(R^M_1 - R^S_1)}_{\text{Entry selection}} + \underbrace{(1 - s_0)(R^S_0 - R^X_0)}_{\text{Exit selection}}$$

Uncertainty is quantified via a **Rubin Bayesian (Dirichlet) entity-cluster bootstrap** (`Exp(1)`
entity weights parallelized with `rayon` in Rust), which remains smooth when power-law retail giants
carry 20%+ of cohort volume.

### 2. Event-Day Denominator Dilution & Trend Instability (`surge.py`)

In the F5 Labs study, Black Friday's automation **rate** dipped below the November baseline in all
three years (`11.6%`, `14.8%`, `14.6%` vs `15.3%`, `17.8%`, `24.5%`), even though raw bot volume
**rose** `+20%` (2023) and `+17%` (2024) beneath `+61%` and `+54%` human shopper surges.
`attribute_surge_shapley` applies exact two-player **Shapley-Owen attribution** to
$R(B, H) = \frac{B}{B + H}$:

$$\phi_B = \tfrac{1}{2}\big[(R(B_1, H_0) - R(B_0, H_0)) + (R(B_1, H_1) - R(B_0, H_1))\big], \quad \phi_H = \Delta R - \phi_B$$

cleanly separating upward bot-volume impulses ($\phi_B > 0$) from downward human-denominator
dilution ($\phi_H < 0$). `assess_pre_event_trend` pairs exact **Theil-Sen median slopes** with
tie-corrected **Mann-Kendall $\tau_b$** tests over `BF-k` windows (showing why 13 of 14
flow-subindustry series flip pre-event slope signs across years).

### 3. Interstitial Challenge Censoring: Sharp Manski Bounds & Multi-Flow EM (`censoring.py`)

When $c \in [0.239, 0.795]$ of flagged automated requests hit interstitial CAPTCHAs or JS challenges
before client-side behavioral sensors execute, latent bot sophistication $v_S \in \{0.0, 0.5, 1.0\}$
is non-randomly censored:

- **Sharp Distribution-Free Manski Bounds:** $I^* \in [(1 - c)I_U,\; (1 - c)I_U + c]$ of exact
  width $c$. At the `79.5%` challenge rate the report gives for 2024, the assumption-free
  interval has width `0.795`, so a rise in the uncensored index from `0.07` to `0.23` between
  2023 and 2024 does not fix the sign of the change in the true index. The `0.23 -> 0.98`
  rise to 2025 (at `41.3%` censoring) does fix it, but only under the added assumption that
  challenged traffic is no more sophisticated than graded traffic
  ($\mathbb{E}[v_S \mid C=1] \le I_U$). These are consequences of the published rates, not
  findings about real bots.
- **Joint Multi-Flow EM Solver:** Estimates flow-specific latent mixes $\pi_f \in \Delta^2$ and
  shared tier challenge propensities $\gamma \in (0, 1)^3$ across flows with varying challenge
  policy exposure $e_f \in (0, 1]$ via 1-D concave Newton/bisection M-steps in Rust.

### 4. $O(N \cdot D)$ Multivariate Marked Hawkes Processes (`hawkes.py`)

Models calendar-agnostic self-exciting bot bursts (peaking at `BF-23`, `BF+1`, and `BF-38` rather
than calendar holidays) across $D$ `{Channel x Flow}` dimensions. Using Ozaki's state recursion
$g_j(r) = e^{-\beta(t_r - t_{r-1})}(g_j(r-1) + \mathbf{1}[m_{r-1}=j])$:

- Evaluates exact log-likelihood and analytical gradients $(\nabla_\mu \ell, \nabla_A \ell)$ in
  **$O(N \cdot D)$** time rather than $O(N^2)$, and
- Runs **exact $O(N \cdot D)$ Lewis-Mohler Expectation-Maximization** using summed mark
  responsibilities $S_{i,j} = \sum_{r: m_r=i} \alpha_{i,j} g_j(r) / \lambda_i(t_r^-)$ with
  guaranteed non-negativity and monotonic likelihood ascent.

### 5. Stackelberg Web-vs-Mobile-API Displacement Game (`game.py`)

A thought experiment about a mechanism the report raises (it cautions that where API
protection is weaker, attackers can shift to that channel). A defender with a fixed budget
chooses controls $u_W, u_M$; an attacker with quadratic congestion costs and a cross-channel
capacity coupling $\gamma$ best-responds in closed form (KKT), and the leader problem is
solved along the budget frontier.

**Nothing in this model is estimated from data, and it produces no number worth quoting.**
Every parameter is an illustrative default. `sensitivity_sweep()` solves the game over 216
parameter combinations (API control efficacy, API congestion cost, budget, 36 per coupling):

| Coupling $\gamma$ | Extra attack work on the API when only Web is hardened (min / median / max) | Loss cut by co-allocating at the same budget (min / median / max) |
|---:|---:|---:|
| 0.0 | 0% / 0% / 0% | 0% / 0.3% / 63.7% |
| 0.1 | 10.6% / 20.2% / 21.2% | 0% / 2.3% / 67.6% |
| 0.3 | 34.7% / 65.9% / 69.3% | 0% / 9.8% / 73.4% |
| 0.55 (default) | 71.5% / 119.8% / 143.0% | 0% / 20.1% / 77.2% |
| 0.8 | 118.9% / 175.2% / 237.7% | 0% / 28.5% / 79.3% |
| 1.2 | 231.1% / 290.4% / 448.7% | 0% / 36.3% / 81.4% |

What survives the sweep is a direction: with uncoupled channels nothing moves, with any
coupling hardening Web alone always moves attack work to the API, and co-allocating never
does worse. The size runs from 11% to 449% and is set by a parameter nobody has measured.
An earlier version of this README quoted the default-parameter outputs (135.8% and 22.1%) as
if they were results; they are one cell of this table.

---

## Worked example on synthetic data calibrated to the public report

`uv run bot-econometrics` runs the methods on invented inputs
([`calibrated_example.py`](src/bot_traffic_econometrics/calibrated_example.py)); output is
committed in [`benchmarks/results/calibrated_example.json`](benchmarks/results/calibrated_example.json).
The inputs were sized by hand so the quantities in the middle column equal figures printed in
the report. Those match by construction. The right-hand column is what the methods return on
those invented inputs; it depends on choices the report does not publish (how many retailers,
how large, which ones leave) and is not a statement about F5's data.

| Method | Calibrated to (from the public report) | Method output on the invented inputs |
|---|---|---|
| Panel decomposition, 2023 to 2024 | aggregate `17.4% -> 18.5%`; same-store `21.6% -> 18.1%` | within-cohort `-3.4 pp`, composition `+4.5 pp` (six invented retailers, two of them low-bot exits) |
| Panel decomposition, 2024 to 2025 | aggregate `18.5% -> 28.6%`; same-store `16.7% -> 29.8%` | within-cohort `+13.1 pp`, composition `-3.0 pp` |
| Surge attribution, Black Friday 2023 and 2024 | November rates `15.3%`, `17.8%`; bot volume `+20%`, `+17%`; human volume `+61%`, `+54%` | rate falls to `11.9%` and `14.1%`: bot impulse `+2.15`, `+2.11 pp` against human dilution `-5.58`, `-5.78 pp`. The report prints `11.6%` and `14.8%`; the rounded growth figures do not recover them exactly |
| Surge attribution, Black Friday 2025 | November rate `24.5%`; bot volume `-22%`; human volume `+43%` | rate falls to `15.0%` (report: `14.6%`): bot impulse `-3.88 pp`, human dilution `-5.58 pp` |
| Censoring bounds, Ecommerce Web | challenge rates `23.9%`, `79.5%`, `41.3%`; uncensored index `0.07`, `0.23`, `0.98` | Manski bounds `[0.05, 0.29]`, `[0.05, 0.84]`, `[0.58, 0.99]`: 2023 to 2024 sign not identified; 2024 to 2025 increasing under adverse interception |
| Rate versus volume share | Ecommerce `add_to_cart` rate `54.2%`, `session_create` 85% of bot volume; QSR `gift_card` rate `10.1%`, `login` 77% of bot volume | rank inversion between rate and share (flow sizes invented to hit those four figures) |

The report itself states a different decomposition over the full 2023 to 2025 span (`+11.2 pp`
split into `+5.6 pp` within-cohort and `+5.5 pp` composition). The per-period splits above come
from the invented panels and should not be read against it.

---

## Differential Validation Against `SciPy` & Rust Performance

Correctness is verified in CI (`tests/test_rust_and_scipy_parity.py`) across Python 3.12-3.15
against independent `SciPy` numerical routines and exact algebraic identities:

| Quantity | Independent Reference | Max Absolute Discrepancy |
|---|---|---:|
| 4-way unbalanced panel identity | $\Delta_{\text{within}} + \Delta_{\text{realloc}} + \Delta_{\text{entry}} + \Delta_{\text{exit}} = R_1 - R_0$ | **< 1e-15** |
| Rust vs Python panel decomposition | `_decompose_python` (NumPy float64) | **< 1e-14** |
| Joint Theil-Sen slope & intercept | `scipy.stats.theilslopes(..., method="joint")` | **< 1e-14** |
| Tie-corrected Mann-Kendall $\tau_b$ | `scipy.stats.kendalltau(..., variant="b")` | **< 1e-14** |
| $O(N \cdot D)$ Hawkes log-likelihood | Piecewise `scipy.integrate.quad` + $O(N^2)$ intensity sum | **< 1e-10** |
| Analytical Hawkes gradients $(\nabla_\mu, \nabla_A)$ | `scipy.optimize.approx_fprime` | **< 1e-5** |
| Stackelberg optimal defense loss | `scipy.optimize.minimize(..., method="SLSQP")` | **< 1e-7** |

### Rust (`_bte_rs`) vs Pure-Python/NumPy Wall-Clock Speedups

Measured on Apple M1 (macOS arm64, Python 3.15, 1/5/15-min load average `3.05 / 2.89 / 2.85`);
raw timings committed in [`benchmarks/results/rust_speedup.json`](benchmarks/results/rust_speedup.json):

| Workload | Rust (`_bte_rs`) | Python / NumPy | Speedup |
|---|---:|---:|---:|
| **Dirichlet cluster bootstrap** (260 entities x 10,000 draws, Rayon) | **10.61 ms** | 682.24 ms | **64.3x** |
| **6-D Hawkes EM MLE** (35,251 events, 60 iterations) | **60.12 ms** | 2,589.59 ms | **43.1x** |
| **6-D Ozaki Hawkes log-likelihood + gradient** (35,251 events) | **2.48 ms** | 43.10 ms | **17.4x** |
| **14-flow joint censored-sophistication EM** (300 iterations) | **1.29 ms** | 172.70 ms | **133.7x** |

---

## Limitations & Scope Boundaries

- **No real data.** Every panel, flow table and event stream in the repository is synthetic.
  The estimators have not been run on real telemetry here, so nothing in this repository is
  evidence about attacker behavior.
- **The channel game is uncalibrated.** See section 5: it supports a direction and no size.
- **The Hawkes model has not been fitted to real event data.** The parity and recovery tests
  use streams simulated from the same model family, which is the easiest case.
- **Point identification vs partial identification under censoring:** With a single cross-sectional
  flow, latent sophistication under arbitrary challenge selection is only set-identified (Manski
  width $c$). `fit_censored_sophistication_em` achieves point identification by pooling $\ge 2$
  flows under a shared tier-interception structure $P(C=1 \mid S=k, f) = e_f \gamma_k$; if
  attackers customize evasion per endpoint, report `manski_sophistication_bounds` instead.
- **Fixed decay rate $\beta$ in `fit_multivariate_hawkes`:** Fixing $\beta > 0$ makes the
  $O(N \cdot D)$ Ozaki sufficient-statistic EM globally concave in $(\mu, A)$; joint estimation of
  $\beta$ requires outer profile-likelihood grid search over $\beta$.

---

## Quickstart

```bash
git clone https://github.com/adeoluo/bot-traffic-econometrics.git
cd bot-traffic-econometrics
uv sync --dev --group rust

# Run strict lint, mypy --strict, Rust clippy, and SciPy parity tests
uv run ruff check .
uv run mypy src tests benchmarks
BTE_REQUIRE_RUST=1 uv run pytest -v

# Run the synthetic worked example and the Rust-versus-Python benchmarks
uv run bot-econometrics
uv run python benchmarks/run_benchmarks.py
```

## Citation

```bibtex
@techreport{ogunnoiki2026blackfriday,
  author      = {Adeolu Ogunnoiki},
  title       = {Automation Leading Up to Black Friday},
  institution = {F5 Labs Threat Research},
  year        = {2026},
  month       = nov,
  url         = {https://www.f5.com/labs/articles/2026-black-friday-bot-report}
}
```

