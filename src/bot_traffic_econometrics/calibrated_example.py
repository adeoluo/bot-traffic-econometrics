"""A worked example on synthetic data calibrated to published aggregates.

This is not a re-analysis of the report's data and reproduces none of its results. No
customer data is in this repository. The retailer panels below are invented: their sizes and
bot counts were chosen by hand so that the aggregate and same-store rates come out equal to
figures printed in the public report. Matching those figures is therefore true by
construction and is tested only to keep the example honest about what it was calibrated to.

What the example does show is the method: given panels with these aggregates, how the
decomposition, the surge attribution and the censoring bounds behave. Every quantity the
report does not publish (how a rate change splits into within-cohort and composition terms
for these invented retailers, the censored bounds, the flow tables, the channel game) depends
on choices made here and says nothing about real retailers.

Public source of the calibration targets:
- Adeolu Ogunnoiki, *Automation leading up to Black Friday*, F5 Labs, November 18, 2026.
  https://www.f5.com/labs/articles/2026-black-friday-bot-report
Articles that discuss that report:
- Sean Murphy (Field CISO, North America, F5), *The Black Friday automation trend everyone
  gets wrong*, November 19, 2026.
  https://www.f5.com/company/blog/the-black-friday-automation-trend-everyone-gets-wrong
- Alexander Culafi, *Why Holiday Security Can't Wait for Black Friday*, Dark Reading,
  November 19, 2026.
  https://www.darkreading.com/cybersecurity-operations/why-holiday-security-cant-wait-black-friday
"""

from __future__ import annotations

from dataclasses import dataclass

from bot_traffic_econometrics.censoring import (
    FlowCensoringObs,
    ManskiBounds,
    is_direction_identified,
    manski_sophistication_bounds,
)
from bot_traffic_econometrics.game import StackelbergEquilibrium, solve_stackelberg_defense
from bot_traffic_econometrics.hawkes import FlowRiskSummary, summarize_flow_paradox
from bot_traffic_econometrics.panel import EntityTraffic, PanelDecomposition, decompose_panel
from bot_traffic_econometrics.surge import SurgeAttribution, attribute_surge_shapley


@dataclass(frozen=True)
class CalibratedExample:
    """Outputs of the five methods on the synthetic, calibrated example."""

    panel_2023_2024: PanelDecomposition
    panel_2024_2025: PanelDecomposition
    surge_by_year: dict[int, SurgeAttribution]
    censoring_bounds_by_year: dict[int, ManskiBounds]
    ecommerce_2023_vs_2024_direction: str
    ecommerce_2024_vs_2025_direction: str
    ecommerce_flow_paradox: tuple[FlowRiskSummary, ...]
    qsr_flow_paradox: tuple[FlowRiskSummary, ...]
    channel_equilibrium: StackelbergEquilibrium


def calibrated_retail_panels() -> tuple[
    list[EntityTraffic], list[EntityTraffic], list[EntityTraffic]
]:
    """Three invented retailer panels whose aggregates equal figures in the public report:

    - Aggregate automation: `17.4%` (2023) -> `18.5%` (2024) -> `28.6%` (2025)
    - 2023 -> 2024: same-store survivor cohort *falls* `21.6% -> 18.1%`, while exiting low-bot
      (`< 2%`) retailers create an upward `+1.0 pp` headline illusion.
    - 2024 -> 2025: same-store survivor cohort genuinely surges `16.7% -> 29.8%`.
    """
    # 2023 total = 100.0M requests, bots = 17.42M (17.42%):
    # Survivors S_23_24 total = 79.0M, bots = 17.064M (21.60%);
    # Low-bot exiters X_23 total = 21.0M, bots = 0.356M (1.70% < 2%).
    y2023 = [
        EntityTraffic("fashion_flagship_a", 22_000_000.0, 5_720_000.0),
        EntityTraffic("ecommerce_marketplace_b", 26_000_000.0, 6_044_000.0),
        EntityTraffic("qsr_chain_c", 16_000_000.0, 2_900_000.0),
        EntityTraffic("specialty_retail_d", 15_000_000.0, 2_400_000.0),
        EntityTraffic("regional_grocer_exit_1", 12_000_000.0, 204_000.0),
        EntityTraffic("department_store_exit_2", 9_000_000.0, 152_000.0),
    ]

    # 2024: S_23_24 survivors total = 84.0M, bots = 15.204M (18.10%);
    # plus 2024 entrants M_24 total = 16.0M, bots = 3.296M -> overall 2024 = 18.50%.
    # Between 2024 and 2025, `fashion_flagship_a` rotates out while remaining 2024->2025
    # survivors (`b, c, d, entrant_e`) sit at 16.7% in 2024 and surge to 29.8% in 2025.
    y2024 = [
        EntityTraffic("fashion_flagship_a", 22_000_000.0, 4_850_000.0),
        EntityTraffic("ecommerce_marketplace_b", 28_000_000.0, 4_760_000.0),
        EntityTraffic("qsr_chain_c", 18_000_000.0, 2_970_000.0),
        EntityTraffic("specialty_retail_d", 16_000_000.0, 2_624_000.0),
        EntityTraffic("apparel_entrant_e", 16_000_000.0, 2_672_000.0),
        EntityTraffic("footwear_transient_f", 2_500_000.0, 1_089_000.0),
    ]

    # 2025: 2024->2025 survivors (`b, c, d, e`) surge from 16.70% -> 29.80%,
    # plus new 2025 entrant `g` -> overall 2025 = 28.60%.
    y2025 = [
        EntityTraffic("ecommerce_marketplace_b", 32_000_000.0, 10_240_000.0),
        EntityTraffic("qsr_chain_c", 20_000_000.0, 5_400_000.0),
        EntityTraffic("specialty_retail_d", 18_000_000.0, 5_220_000.0),
        EntityTraffic("apparel_entrant_e", 20_000_000.0, 5_960_000.0),
        EntityTraffic("luxury_omni_entrant_g", 15_000_000.0, 3_210_000.0),
    ]
    return y2023, y2024, y2025


def run_calibrated_example() -> CalibratedExample:
    """Run the five methods on the synthetic panels and illustrative parameters."""
    y23, y24, y25 = calibrated_retail_panels()
    p_23_24 = decompose_panel(y23, y24)
    p_24_25 = decompose_panel(y24, y25)

    # Section 3: Black Friday denominator dilution vs raw bot/human volume changes.
    # Published rates & volume shifts:
    # 2023: Nov baseline 15.3%, BF 11.6% (bots +20%, humans +61% -> exact ratio matches 11.8%/11.6%)
    # 2024: Nov baseline 17.8%, BF 14.8% (bots +17%, humans +54%)
    # 2025: Nov baseline 24.5%, BF 14.6% (bots -22%, humans +43%)
    surges = {
        2023: attribute_surge_shapley(
            baseline_bots=15.3,
            baseline_humans=84.7,
            event_bots=15.3 * 1.20,
            event_humans=84.7 * 1.61,
        ),
        2024: attribute_surge_shapley(
            baseline_bots=17.8,
            baseline_humans=82.2,
            event_bots=17.8 * 1.17,
            event_humans=82.2 * 1.54,
        ),
        2025: attribute_surge_shapley(
            baseline_bots=24.5,
            baseline_humans=75.5,
            event_bots=24.5 * 0.78,
            event_humans=75.5 * 1.43,
        ),
    }

    # Section 5: Interstitial challenge censoring (23.9% in 2023, 79.5% in 2024, 41.3% in 2025)
    # applied to Ecommerce Web observed uncensored sophistication (index 0.07 -> 0.23 -> 0.98).
    ecom_obs = {
        2023: FlowCensoringObs(
            "ecommerce_web_2023",
            uncensored_low=0.915 * 761.0,
            uncensored_medium=0.030 * 761.0,
            uncensored_high=0.055 * 761.0,
            challenged_censored=239.0,
        ),
        2024: FlowCensoringObs(
            "ecommerce_web_2024",
            uncensored_low=0.744 * 205.0,
            uncensored_medium=0.052 * 205.0,
            uncensored_high=0.204 * 205.0,
            challenged_censored=795.0,
        ),
        2025: FlowCensoringObs(
            "ecommerce_web_2025",
            uncensored_low=0.010 * 587.0,
            uncensored_medium=0.020 * 587.0,
            uncensored_high=0.970 * 587.0,
            challenged_censored=413.0,
        ),
    }
    bounds = {yr: manski_sophistication_bounds(obs) for yr, obs in ecom_obs.items()}

    # Section 6: Rate vs volume-share paradox across Ecommerce Web and QSR Web flows.
    ecom_flows = summarize_flow_paradox(
        [
            ("session_create", 480_000.0, 136_000.0),  # ~85% of all Ecommerce bot volume
            ("add_to_cart", 22_000.0, 11_924.0),  # 54.2% bot rate, small volume share
            ("product_search", 35_000.0, 8_400.0),
            ("login", 18_000.0, 3_680.0),
        ]
    )
    qsr_flows = summarize_flow_paradox(
        [
            ("login", 920_000.0, 77_000.0),  # 77% of all QSR bot volume (8.4% rate)
            ("gift_card_balance", 65_000.0, 6_565.0),  # 10.1% bot rate (#1 rate, ~6.6% share)
            ("store_locator", 240_000.0, 11_435.0),
            ("checkout", 110_000.0, 5_000.0),
        ]
    )

    eq = solve_stackelberg_defense(budget=0.95)
    return CalibratedExample(
        panel_2023_2024=p_23_24,
        panel_2024_2025=p_24_25,
        surge_by_year=surges,
        censoring_bounds_by_year=bounds,
        ecommerce_2023_vs_2024_direction=is_direction_identified(bounds[2023], bounds[2024]),
        ecommerce_2024_vs_2025_direction=is_direction_identified(
            bounds[2024], bounds[2025], regime="adverse_interception"
        ),
        ecommerce_flow_paradox=ecom_flows,
        qsr_flow_paradox=qsr_flows,
        channel_equilibrium=eq,
    )
