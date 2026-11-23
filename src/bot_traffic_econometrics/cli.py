"""CLI entry point for reproducing the 2026 F5 Labs econometric findings and Rust benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from bot_traffic_econometrics._rust_bridge import USING_RUST
from bot_traffic_econometrics.f5_benchmarks import reproduce_f5_findings


def _summary_payload() -> dict[str, Any]:
    rep = reproduce_f5_findings()
    return {
        "using_rust": USING_RUST,
        "panel_2023_2024": {
            "r0": round(rep.panel_2023_2024.r0, 4),
            "r1": round(rep.panel_2023_2024.r1, 4),
            "delta_total": round(rep.panel_2023_2024.delta_total, 4),
            "survivor_r0": round(rep.panel_2023_2024.survivor_r0, 4),
            "survivor_r1": round(rep.panel_2023_2024.survivor_r1, 4),
            "within_behavior": round(rep.panel_2023_2024.within_behavior, 4),
            "total_composition": round(rep.panel_2023_2024.total_composition, 4),
            "identity_residual": rep.panel_2023_2024.identity_residual,
        },
        "panel_2024_2025": {
            "r0": round(rep.panel_2024_2025.r0, 4),
            "r1": round(rep.panel_2024_2025.r1, 4),
            "delta_total": round(rep.panel_2024_2025.delta_total, 4),
            "survivor_r0": round(rep.panel_2024_2025.survivor_r0, 4),
            "survivor_r1": round(rep.panel_2024_2025.survivor_r1, 4),
            "within_behavior": round(rep.panel_2024_2025.within_behavior, 4),
            "total_composition": round(rep.panel_2024_2025.total_composition, 4),
            "identity_residual": rep.panel_2024_2025.identity_residual,
        },
        "black_friday_surge_shapley": {
            str(yr): {
                "baseline_rate": round(s.baseline_rate, 4),
                "event_rate": round(s.event_rate, 4),
                "bot_impulse_pp": round(100.0 * s.bot_impulse_effect, 2),
                "human_dilution_pp": round(100.0 * s.human_dilution_effect, 2),
                "is_diluted_bot_surge": s.is_diluted_bot_surge,
            }
            for yr, s in rep.surge_by_year.items()
        },
        "censoring_manski_bounds": {
            str(yr): {
                "censoring_rate": round(b.censoring_rate, 4),
                "naive_uncensored_index": round(b.naive_uncensored_index, 4),
                "manski_bounds": [round(b.manski_lower, 4), round(b.manski_upper, 4)],
                "adverse_interception_bounds": [
                    round(b.adverse_interception_bounds[0], 4),
                    round(b.adverse_interception_bounds[1], 4),
                ],
            }
            for yr, b in rep.censoring_bounds_by_year.items()
        },
        "stackelberg_channel_equilibrium": {
            "web_only_controls": [round(x, 4) for x in rep.channel_equilibrium.web_only_controls],
            "stackelberg_controls": [
                round(x, 4) for x in rep.channel_equilibrium.stackelberg_controls
            ],
            "api_displacement_pct_under_web_only": round(
                rep.channel_equilibrium.api_displacement_pct, 2
            ),
            "relative_loss_reduction_pct": round(
                rep.channel_equilibrium.relative_loss_reduction_pct, 2
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bot-econometrics",
        description="Reproduce econometric decompositions and censored-telemetry bounds.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = _summary_payload()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    p1 = payload["panel_2023_2024"]
    p2 = payload["panel_2024_2025"]
    print(f"Backend: {'compiled Rust (_bte_rs)' if USING_RUST else 'Python fallback'}")
    print(
        f"2023->2024 panel: headline {100 * p1['r0']:.1f}% -> {100 * p1['r1']:.1f}% "
        f"(within-survivor {100 * p1['within_behavior']:+.1f} pp, "
        f"composition {100 * p1['total_composition']:+.1f} pp)"
    )
    print(
        f"2024->2025 panel: headline {100 * p2['r0']:.1f}% -> {100 * p2['r1']:.1f}% "
        f"(within-survivor {100 * p2['within_behavior']:+.1f} pp, "
        f"composition {100 * p2['total_composition']:+.1f} pp)"
    )
    for yr, s in payload["black_friday_surge_shapley"].items():
        print(
            f"BF {yr}: rate {100 * s['baseline_rate']:.1f}% -> {100 * s['event_rate']:.1f}% | "
            f"bot impulse {s['bot_impulse_pp']:+.2f} pp, "
            f"human dilution {s['human_dilution_pp']:+.2f} pp "
            f"(diluted surge={s['is_diluted_bot_surge']})"
        )
    eq = payload["stackelberg_channel_equilibrium"]
    disp = eq["api_displacement_pct_under_web_only"]
    print(
        f"Channel game: Web-only defense displaces +{disp:.1f}% extra attack mass onto Mobile API; "
        f"Stackelberg co-allocation cuts residual loss by {eq['relative_loss_reduction_pct']:.1f}%."
    )
    return 0
