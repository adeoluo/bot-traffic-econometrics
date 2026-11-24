<h1 align="center">Bot Traffic Econometrics</h1>

<p align="center">
  <em>A tested library of five statistical methods that separate a real change in website bot traffic from a change in what is being measured. It has a compiled Rust core and typed Python code, and it ships invented example data only.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/bot-traffic-econometrics/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/bot-traffic-econometrics/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/bot-traffic-econometrics/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/bot-traffic-econometrics"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue">
  <img alt="Rust" src="https://img.shields.io/badge/rust-PyO3%20%2B%20Rayon-DEA584">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Highlights

Bots are automated programs that visit websites. Measuring them is difficult: a headline
number can go up only because the set of websites being measured changed. This library holds
five statistical methods that separate a real change from a change caused by measurement.

- **It shows how a headline bot rate can mislead.** In the bundled example, the overall bot
  rate rises from 17.4% to 18.5% between 2023 and 2024. Over the same years, the bot rate of
  the retailers measured in both years falls from 21.6% to 18.1%. Those four figures come
  from my published report. The inputs behind the example are invented.
- **Each method is checked against an independent reference.** The tests compare results with
  SciPy (the standard Python library for scientific computing), with exact mathematical
  identities, or with known true values in simulated data.
- **The Rust core is 17x to 134x faster than the plain Python version.** This was measured on
  four heavy workloads on one Apple M1 computer. Both versions stay in the repository, and the
  tests compare them.

**Where it falls short:** all example data is made up. It is not the data or the analysis
behind the report, and the methods have not been run on real traffic data here. One of the
five methods, a game model of attackers and defenders, only shows the direction of an effect.
The size of that effect ranges from 11% to 449%, depending on a setting nobody has measured.

## Key terms

Plain definitions of the terms used in this README.

| Term | Meaning |
|---|---|
| Adverse interception | An assumption that challenged bot traffic is no more sophisticated than graded traffic. Under it, the true sophistication index lies between the Manski lower bound and the measured index. |
| Bayesian bootstrap | A way to measure uncertainty. Each draw gives every entity a random positive weight and recomputes the result. The spread of the draws gives a 95% interval. |
| Bot rate | Bot requests divided by all requests, from 0% to 100%. It is the quantity being measured, so neither direction is better in itself. |
| Censoring | Missing measurements that are not missing at random. Requests stopped by a challenge page are never graded, so an average over graded requests can be biased. |
| Challenge page | A CAPTCHA or JavaScript test shown before the real page (an interstitial). It stops a bot before the sensors in the browser or app can grade it. |
| Composition effect | The part of a rate change caused by which entities are measured and how traffic shifts among them: entry, exit and survivor share reallocation. It is not behavior change. |
| Coupling (γ) | In the channel game, how much attack effort on one channel (Web or Mobile API) raises the cost of effort on the other. Its real value has not been measured. |
| Denominator dilution | A fall in the bot rate caused by more human traffic, not by fewer bots. Human growth enlarges the total that bot requests are divided by. |
| Expectation-maximization (EM) | A standard method for fitting a model with hidden quantities. It repeats two steps: estimate the hidden quantities, then refit the model parameters. |
| Hawkes process | A model of events over time in which each event briefly raises the chance of more events. It describes bursts, such as waves of bot requests. |
| Kendall's tau-b and Mann-Kendall test | Tau-b measures how consistently values rise or fall over time, from -1 (always falling) to 1 (always rising). The Mann-Kendall test checks whether that trend could be chance. |
| Manski bounds | The range the true value must lie in when nothing is assumed about the missing data. With a censored share c, the range has width c. Narrower is more informative. |
| Max absolute discrepancy | The largest difference a test allows between this library's result and the reference result. Lower is better. 1e-15 means 0.000000000000001. |
| O(N · D) | How running time grows: in step with the number of events N times the number of dimensions D. O(N²) would grow with the square of the number of events. |
| Panel decomposition | A split of a change in the overall bot rate into four parts that add up exactly: within-survivor behavior, survivor share reallocation, entry and exit. |
| Percentage point (pp) | The plain difference between two percentages. A rate that goes from 17.4% to 18.5% rises by 1.1 pp. |
| Same-store rate | The bot rate over only the entities measured in both periods. It removes the effect of entities joining or leaving. |
| SciPy | The standard Python library for scientific computing. The tests use its routines as independent reference answers. |
| Shapley attribution | A rule from game theory that splits a change fairly between its causes. Here it splits a bot-rate change into a bot-volume part and a human-volume part. |
| Sophistication index | Average bot sophistication, scoring low, medium and high bots as 0, 0.5 and 1. It runs from 0 to 1. Higher means more advanced bots. |
| Speedup | Python time divided by Rust time for the same workload. Higher is better. 64.3x means the Rust version ran 64.3 times faster. |
| Stackelberg game | A game in which a leader moves first and a follower responds. Here the defender sets protections, then the attacker chooses how much to attack each channel. |
| Survivors, exiters and entrants | Entities (customers such as retailers) measured in both periods, only in the first period, and only in the second period. |
| Theil-Sen slope | The median of the slopes between every pair of points. It estimates a trend and is not pulled off course by a few outliers. |
| Within-cohort effect | The part of a rate change that comes from survivors changing their own bot rates. It is the closest measure of a real change in behavior. |

## What this repository is, and is not

Security telemetry (the traffic data that security systems record) can look simple. The hard
question is why a headline rate moved. There are four possible reasons: **attackers changed
behavior**, **the set of customers being measured changed**, **human traffic diluted the
denominator**, or **challenge pages kept client-side sensors from running**. The denominator
is the total that the rate divides by. Client-side sensors run in the visitor's browser or app
and grade the bot. The author ran into each of these while writing a threat-research report. This library provides
general-purpose estimators (statistical methods that compute a quantity from data) for them.

- **It is** a tested implementation of five methods. Each method is checked against `SciPy`,
  an exact algebraic identity, or known true values in simulated data. The heavy computations
  run in a compiled Rust core.
- **It is not the analysis behind the report.** None of the report's data or code is here.
  The Hawkes, Manski and Stackelberg parts are the author's own extensions of questions the
  report raises. They are not methods taken from it.
- **It does not reproduce the report's results.** The bundled example uses invented retailer
  panels (tables of retailers and their traffic in each year). Their sizes were chosen so that
  their totals equal figures printed in the public report (see
  [Worked example](#worked-example-on-synthetic-data-calibrated-to-the-public-report)). So the
  example shows how the methods behave on inputs with those totals. It shows nothing about
  real retailers.
- **Evidence level:** unit tested and validated against reference implementations. The
  methods have not been run on real telemetry in this repository, have not been reproduced by
  anyone else, and are not used in production.

The report and two articles that discuss it:

- Adeolu Ogunnoiki, [*Automation leading up to Black Friday*](https://www.f5.com/labs/articles/2026-black-friday-bot-report),
  F5 Labs, Nov 18, 2026 ([PDF](https://cdn.sanity.io/files/k6fem79d/translations-production/1d1028afb94e20d907a0b98ef084535e84fc9cd5.pdf) ·
  [author page](https://www.f5.com/labs/authors/adeolu-ogunnoiki)).
- Sean Murphy (Field CISO, North America, F5), [*The Black Friday automation trend everyone gets wrong*](https://www.f5.com/company/blog/the-black-friday-automation-trend-everyone-gets-wrong),
  F5 blog, Nov 19, 2026.
- Alexander Culafi, [*Why Holiday Security Can't Wait for Black Friday*](https://www.darkreading.com/cybersecurity-operations/why-holiday-security-cant-wait-black-friday),
  Dark Reading, Nov 19, 2026, which discusses my F5 Labs published research report.

---

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="880">
</p>

<sub>Legend: amber = telemetry inputs and entry points, indigo = compiled Rust engines for the heavy computations (`_bte_rs`), blue = methods that separate causes, bound unknown values and solve the channel game, green = tests that compare Rust, Python and `SciPy` results, grey = outputs.</sub>

Every install compiles the **Rust extension (`bot_traffic_econometrics._bte_rs`, PyO3 + Rayon)**.
PyO3 connects Rust code to Python, and Rayon runs work in parallel on several processor cores.
The library sends the heavy computations through this extension automatically
(`USING_RUST == True`). Plain Python and NumPy (the standard Python array library) versions of
the same computations stay in the repository. The tests check that the Rust and Python
versions agree for the panel decomposition, the trend test, the Hawkes likelihood and fit, and
the censoring EM. The two bootstrap versions use different random numbers, so the tests
compare them only within a loose margin. Where `SciPy` has a matching routine, the tests also
compare against it. All of this runs in CI (continuous integration, the automatic checks that
run on every change).

---

## Five Measurement Traps & Mathematical Formulations

### 1. Cohort Churn vs Behavioral Shift (`panel.py` + Rust Rayon Bootstrap)

A panel is the set of entities (customers, such as retailers) measured in each season.
Entities join and leave between seasons, and a few very large ones carry much of the traffic.
So a change in the headline rate, $\Delta R = R_1 - R_0$, mixes a change in attacker behavior
with a change in which entities are measured. The method splits entities into three groups:
**Survivors** $S = E_0 \cap E_1$ (in both periods), **Exiters** $X = E_0 \setminus E_1$ (only
in the first period) and **Entrants** $M = E_1 \setminus E_0$ (only in the second period). It
then splits the rate change into four parts that add up **exactly, with zero residual**
(nothing left over). This is a 4-way Melitz-Polanec and symmetric Kitagawa decomposition, two
standard ways to split a change in an average rate into parts:

$$\Delta R = \underbrace{\sum_{i \in S} \bar{w}^S_i (r_{i,1} - r_{i,0})}_{\text{Within-survivor behavior}} + \underbrace{\sum_{i \in S} (w^S_{i,1} - w^S_{i,0}) \bar{r}_i}_{\text{Survivor share reallocation}} + \underbrace{(1 - s_1)(R^M_1 - R^S_1)}_{\text{Entry selection}} + \underbrace{(1 - s_0)(R^S_0 - R^X_0)}_{\text{Exit selection}}$$

Uncertainty is measured with a **Rubin Bayesian (Dirichlet) entity-cluster bootstrap**. Each
draw gives every entity a random positive weight (`Exp(1)`, an exponential random number with
mean 1) and recomputes the decomposition. The Rust version runs the draws in parallel with
`rayon`. A standard bootstrap resamples entities, so it can leave a very large retailer out of
a draw. Here every entity keeps a positive weight in every draw. This is meant for panels in
which a few large retailers carry 20% or more of the volume. The tests run it on one invented
panel with a few very large entities and check that the interval contains the point estimate.

### 2. Event-Day Denominator Dilution & Trend Instability (`surge.py`)

In the F5 Labs study, Black Friday's automation **rate** was lower than the November baseline
in all three years: `11.6%`, `14.8%` and `14.6%`, compared with `15.3%`, `17.8%` and `24.5%`.
This happened even though raw bot volume **rose** `+20%` (2023) and `+17%` (2024), because
human shopper traffic rose faster, by `+61%` and `+54%`. `attribute_surge_shapley` applies
exact two-player **Shapley-Owen attribution** to the bot rate $R(B, H) = \frac{B}{B + H}$,
where $B$ is bot requests and $H$ is human requests. It averages the effect of each cause over
both orders in which the two changes can be applied:

$$\phi_B = \tfrac{1}{2}\big[(R(B_1, H_0) - R(B_0, H_0)) + (R(B_1, H_1) - R(B_0, H_1))\big], \quad \phi_H = \Delta R - \phi_B$$

This separates the upward push from more bot requests ($\phi_B > 0$) from the downward pull of
more human requests in the denominator ($\phi_H < 0$).

`assess_pre_event_trend` measures the trend in the days before Black Friday (`BF-k` means $k$
days before Black Friday). It uses the exact **Theil-Sen slope**, the median slope over all
pairs of points. It also uses the tie-corrected **Mann-Kendall $\tau_b$** test, which checks
whether values tend to rise or fall and handles equal values correctly. Each series covers one
flow in one sub-industry, where a flow is one step of a user journey, such as login. Across
years, 13 of 14 such series change the sign of their pre-event slope.
`count_sign_consistent_series` counts series that keep one sign. Its
test uses 14 invented series built to give that same split.

### 3. Interstitial Challenge Censoring: Sharp Manski Bounds & Multi-Flow EM (`censoring.py`)

Some flagged bot requests meet an interstitial challenge (a CAPTCHA or JavaScript test page)
before the client-side behavioral sensors can grade them. The share of such requests, $c$, is
between 0.239 and 0.795. For these requests the bot's hidden sophistication score $v_S$ (0.0,
0.5 or 1.0 for low, medium and high) is never seen. The missing requests are not a random
sample, so this censoring can bias any average over the graded requests:

- **Sharp Distribution-Free Manski Bounds:** with no assumption about the missing requests,
  the true index $I^*$ must lie in $[(1 - c)I_U,\; (1 - c)I_U + c]$. Here $I_U$ is the index
  measured on uncensored (graded) traffic. The range has width exactly $c$, and "sharp" means
  no narrower range is possible without extra assumptions. At the `79.5%` challenge rate the
  report gives for 2024, the range has width `0.795`. So the uncensored index rose from `0.07`
  to `0.23` between 2023 and 2024, but that rise does not show whether the true index went up
  or down. The `0.23 -> 0.98` rise to 2025 (at `41.3%` censoring) does show a rise. That holds
  only under an added assumption: challenged traffic is no more sophisticated than graded traffic
  ($\mathbb{E}[v_S \mid C=1] \le I_U$). These are consequences of the published rates, not
  findings about real bots.
- **Joint Multi-Flow EM Solver:** it estimates, for each flow $f$, the hidden mix of low,
  medium and high sophistication ($\pi_f$). It also estimates one challenge probability per
  sophistication tier ($\gamma$), shared by all flows. Each flow has its own exposure to the
  challenge policy, $e_f$, above 0 and at most 1. In each EM step, the Rust and Python versions
  solve a one-variable equation for each tier by bisection (repeatedly halving an interval).

### 4. $O(N \cdot D)$ Multivariate Marked Hawkes Processes (`hawkes.py`)

A Hawkes process models bursts in which each bot event briefly makes more events likely. It
suits bursts that do not follow the calendar, such as peaks at `BF-23`, `BF+1` and `BF-38`
(days before or after Black Friday) rather than on holidays. The model has $D$ dimensions, one
for each `{Channel x Flow}` pair. Ozaki's recursion updates a running sum at each event
instead of revisiting every past event:
$g_j(r) = e^{-\beta(t_r - t_{r-1})}(g_j(r-1) + \mathbf{1}[m_{r-1}=j])$. With this recursion:

- The library computes the exact log-likelihood (how well the model explains the events) and
  its exact gradients with respect to $\mu$ (the base event rates) and $A$ (how much each event
  type excites the others), $(\nabla_\mu \ell, \nabla_A \ell)$. The cost is
  **$O(N \cdot D)$** time rather than $O(N^2)$, where $N$ is the number of events.
- The library runs **exact $O(N \cdot D)$ Lewis-Mohler Expectation-Maximization**, using summed mark
  responsibilities $S_{i,j} = \sum_{r: m_r=i} \alpha_{i,j} g_j(r) / \lambda_i(t_r^-)$. These
  measure how much of event type $i$ is explained by earlier events of type $j$. Every fitted
  parameter stays non-negative by construction. As in any EM method, each step should not
  lower the likelihood. The tests check that the final fit scores at least as well as the
  true parameters, not each step.

### 5. Stackelberg Web-vs-Mobile-API Displacement Game (`game.py`)

This is a thought experiment about a mechanism the report raises. The report cautions that
where API protection is weaker, attackers can shift to that channel. In this game, a defender
with a fixed budget first chooses how strongly to protect the Web and the Mobile API
($u_W, u_M$). The attacker then responds. Its costs grow with the square of its effort in each
channel (quadratic congestion costs). A cross-channel coupling $\gamma$ makes effort on one
channel raise the cost on the other. The attacker's best response has an exact formula,
from the KKT conditions (the standard optimality conditions for problems with limits). The
defender's best choice is then found by searching along the budget limit.

**Nothing in this model is estimated from data, and it produces no number worth quoting.**
Every parameter is an illustrative default (an example value). `sensitivity_sweep()` solves the
game for 216 parameter combinations. For each of six coupling values, it varies the API control
efficacy, the API congestion cost and the budget, which gives 36 combinations per coupling.

In the table below, each row is one coupling value. The middle column is how much more attack
work goes to the API when only the Web is protected. The right column is how much the
defender's loss falls when the same budget is split between both channels (higher is better
for the defender). Each cell gives the smallest, median and largest value over the 36
combinations.

| Coupling $\gamma$ | Extra attack work on the API when only Web is hardened (min / median / max) | Loss cut by co-allocating at the same budget (min / median / max) |
|---:|---:|---:|
| 0.0 | 0% / 0% / 0% | 0% / 0.3% / 63.7% |
| 0.1 | 10.6% / 20.2% / 21.2% | 0% / 2.3% / 67.6% |
| 0.3 | 34.7% / 65.9% / 69.3% | 0% / 9.8% / 73.4% |
| 0.55 (default) | 71.5% / 119.8% / 143.0% | 0% / 20.1% / 77.2% |
| 0.8 | 118.9% / 175.2% / 237.7% | 0% / 28.5% / 79.3% |
| 1.2 | 231.1% / 290.4% / 448.7% | 0% / 36.3% / 81.4% |

The sweep supports a direction only. With uncoupled channels, nothing moves. With any
coupling, protecting only the Web always moves attack work to the API, and splitting the
budget never does worse. The size runs from 11% to 449% and is set by a parameter nobody has
measured. An earlier version of this README quoted the default-parameter outputs (135.8% and
22.1%) as if they were results. They are one of the 36 points behind the 0.55 (default) row
of this table.

---

## Worked example on synthetic data calibrated to the public report

`uv run bot-econometrics` runs the methods on invented inputs
([`calibrated_example.py`](src/bot_traffic_econometrics/calibrated_example.py)). Its output is
committed in [`benchmarks/results/calibrated_example.json`](benchmarks/results/calibrated_example.json).
The inputs were sized by hand so that the quantities in the middle column equal figures
printed in the report. Those figures therefore match by construction. The right-hand column
is what the methods return on those invented inputs. It depends on choices the report does not
publish (how many retailers, how large, which ones leave), so it is not a statement about the
report's data.

In the table below, each row is one method and period. The middle column lists the report
figures that the invented inputs were built to match, and the right column lists what the
method returns. No direction is better here: the table shows how the methods behave, not how
accurate they are.

| Method | Calibrated to (from the public report) | Method output on the invented inputs |
|---|---|---|
| Panel decomposition, 2023 to 2024 | aggregate `17.4% -> 18.5%`; same-store `21.6% -> 18.1%` | within-cohort `-3.4 pp`, composition `+4.5 pp` (six invented retailers, two of them low-bot exits) |
| Panel decomposition, 2024 to 2025 | aggregate `18.5% -> 28.6%`; same-store `16.7% -> 29.8%` | within-cohort `+13.1 pp`, composition `-3.0 pp` |
| Surge attribution, Black Friday 2023 and 2024 | November rates `15.3%`, `17.8%`; bot volume `+20%`, `+17%`; human volume `+61%`, `+54%` | rate falls to `11.9%` and `14.1%`: bot impulse `+2.15`, `+2.11 pp` against human dilution `-5.58`, `-5.78 pp`. The report prints `11.6%` and `14.8%`; the rounded growth figures do not recover them exactly |
| Surge attribution, Black Friday 2025 | November rate `24.5%`; bot volume `-22%`; human volume `+43%` | rate falls to `15.0%` (report: `14.6%`): bot impulse `-3.88 pp`, human dilution `-5.58 pp` |
| Censoring bounds, Ecommerce Web | challenge rates `23.9%`, `79.5%`, `41.3%`; uncensored index `0.07`, `0.23`, `0.98` | Manski bounds `[0.05, 0.29]`, `[0.05, 0.84]`, `[0.58, 0.99]`: 2023 to 2024 sign not identified; 2024 to 2025 increasing under adverse interception |
| Rate versus volume share | Ecommerce `add_to_cart` rate `54.2%`, `session_create` 85% of bot volume; QSR `gift_card` rate `10.1%`, `login` 77% of bot volume | rank inversion between rate and share (flow sizes invented to hit those four figures) |

The report itself gives a different decomposition over the full 2023 to 2025 span: `+11.2 pp`,
split into `+5.6 pp` within-cohort and `+5.5 pp` composition. The per-period splits above come
from the invented panels, so they should not be compared with it.

---

## Differential Validation Against `SciPy` & Rust Performance

The tests in `tests/test_rust_and_scipy_parity.py` run in CI on Python 3.12 to 3.15. They
compare results with independent `SciPy` numerical routines and with exact algebraic
identities (equations that must hold exactly). In the table below, each row is one quantity
and its reference. The right column is the largest difference the test allows, so lower is
better.

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

The timings come from one Apple M1 computer (macOS arm64, Python 3.15). Its system load
averages over 1, 5 and 15 minutes were `3.05 / 2.89 / 2.85`. Raw timings are committed in
[`benchmarks/results/rust_speedup.json`](benchmarks/results/rust_speedup.json). In the table
below, each row is one workload. The Rust and Python columns give run times (lower is better),
and the speedup is the Python time divided by the Rust time (higher is better).

| Workload | Rust (`_bte_rs`) | Python / NumPy | Speedup |
|---|---:|---:|---:|
| **Dirichlet cluster bootstrap** (260 entities x 10,000 draws, Rayon) | **10.61 ms** | 682.24 ms | **64.3x** |
| **6-D Hawkes EM MLE** (35,251 events, 60 iterations) | **60.12 ms** | 2,589.59 ms | **43.1x** |
| **6-D Ozaki Hawkes log-likelihood + gradient** (35,251 events) | **2.48 ms** | 43.10 ms | **17.4x** |
| **14-flow joint censored-sophistication EM** (300 iterations) | **1.29 ms** | 172.70 ms | **133.7x** |

---

## Limitations & Scope Boundaries

- **No real data.** Every panel, flow table and event stream in the repository is synthetic
  (invented). The methods have not been run on real telemetry here, so nothing in this
  repository is evidence about attacker behavior.
- **The channel game is uncalibrated.** Its parameters are not set from real data. See
  section 5: it supports a direction and no size.
- **The Hawkes model has not been fitted to real event data.** The tests use event streams
  simulated from the same kind of model, which is the easiest case.
- **Under censoring, one flow gives only a range:** with a single flow at one point in time,
  and with no assumption about which bots get challenged, the sophistication index is only
  set-identified. The data give a range (Manski width $c$), not a single value.
  `fit_censored_sophistication_em` gets a single value (point identification) by pooling at
  least two flows. This rests on an assumption: every flow shares the same tier pattern of
  challenges, $P(C=1 \mid S=k, f) = e_f \gamma_k$. If attackers adapt their evasion to each
  endpoint, report the `manski_sophistication_bounds` range instead.
- **Fixed decay rate $\beta$ in `fit_multivariate_hawkes`:** $\beta$ sets how fast the effect
  of an event fades. With $\beta > 0$ fixed, the log-likelihood that the $O(N \cdot D)$ Ozaki
  EM maximizes is concave in $(\mu, A)$, so it has no lower local peak to get stuck on.
  Estimating $\beta$ as well needs an outer search: refit over a grid of $\beta$ values and keep
  the best one (a profile-likelihood search).

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
