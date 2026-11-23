"""End-to-end verification of the 2026 F5 Labs empirical benchmarks and CLI."""

from __future__ import annotations

import json

import pytest

from bot_traffic_econometrics import count_sign_consistent_series, reproduce_f5_findings
from bot_traffic_econometrics.cli import main


def test_f5_report_reproduction_invariants() -> None:
    rep = reproduce_f5_findings()

    # 1. Headline rates: 17.4% (2023) -> 18.5% (2024) -> 28.6% (2025).
    assert round(100.0 * rep.panel_2023_2024.r0, 1) == 17.4
    assert round(100.0 * rep.panel_2023_2024.r1, 1) == 18.5
    assert round(100.0 * rep.panel_2024_2025.r1, 1) == 28.6

    # 2023->2024: same-store survivor cohort fell (21.6% -> 18.1%), lifted by low-bot exits!
    assert round(100.0 * rep.panel_2023_2024.survivor_r0, 1) == 21.6
    assert round(100.0 * rep.panel_2023_2024.survivor_r1, 1) == 18.1
    assert rep.panel_2023_2024.within_behavior < -0.03
    assert rep.panel_2023_2024.exit_selection > 0.03

    # 2024->2025: same-store survivor cohort surged (16.7% -> 29.8%)!
    assert round(100.0 * rep.panel_2024_2025.survivor_r0, 1) == 16.7
    assert round(100.0 * rep.panel_2024_2025.survivor_r1, 1) == 29.8
    assert rep.panel_2024_2025.within_behavior > 0.12

    # 2. Black Friday denominator dilution: 2023 & 2024 had diluted bot surges; 2025 fell genuinely.
    assert rep.surge_by_year[2023].is_diluted_bot_surge is True
    assert rep.surge_by_year[2024].is_diluted_bot_surge is True
    assert rep.surge_by_year[2025].is_diluted_bot_surge is False
    assert rep.surge_by_year[2025].bot_impulse_effect < 0.0

    # 3. Interstitial challenge censoring: 2024's 79.5% censoring creates width 0.795.
    assert round(rep.censoring_bounds_by_year[2024].identification_width, 3) == 0.795
    assert rep.ecommerce_2023_vs_2024_direction == "ambiguous"
    assert rep.ecommerce_2024_vs_2025_direction == "strictly_increasing"

    # 4. Rate vs volume-share rank inversion on Ecommerce and QSR flows.
    ecom_by_name = {f.flow_name: f for f in rep.ecommerce_flow_paradox}
    assert ecom_by_name["add_to_cart"].rate_rank == 1
    assert ecom_by_name["session_create"].volume_share_rank == 1
    assert ecom_by_name["session_create"].bot_volume_share >= 0.84

    qsr_by_name = {f.flow_name: f for f in rep.qsr_flow_paradox}
    assert qsr_by_name["gift_card_balance"].rate_rank == 1
    assert qsr_by_name["login"].volume_share_rank == 1
    assert round(qsr_by_name["login"].bot_volume_share, 2) == 0.77


def test_pre_event_slope_sign_consistency_helper() -> None:
    # 1 of 14 series keeps a consistent positive slope across 3 years; 13 flip signs.
    series = [[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]]]
    for i in range(13):
        series.append([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0 + i, 2.0 + i, 3.0 + i]])
    consistent, total = count_sign_consistent_series(series)
    assert (consistent, total) == (1, 14)


def test_cli_text_and_json_modes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out_txt = capsys.readouterr().out
    assert "compiled Rust (_bte_rs)" in out_txt
    assert "2023->2024 panel" in out_txt

    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["using_rust"] is True
    assert payload["panel_2023_2024"]["identity_residual"] < 1e-15
