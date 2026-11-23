"""Bot Traffic Econometrics: Rust-accelerated causal & point-process telemetry analysis."""

from __future__ import annotations

from bot_traffic_econometrics._rust_bridge import USING_RUST
from bot_traffic_econometrics.calibrated_example import (
    CalibratedExample,
    calibrated_retail_panels,
    run_calibrated_example,
)
from bot_traffic_econometrics.censoring import (
    CensoredEMResult,
    FlowCensoringObs,
    ManskiBounds,
    fit_censored_sophistication_em,
    is_direction_identified,
    manski_sophistication_bounds,
)
from bot_traffic_econometrics.game import (
    ChannelParams,
    StackelbergEquilibrium,
    SweepCell,
    attacker_best_response,
    defender_loss,
    sensitivity_sweep,
    solve_stackelberg_defense,
)
from bot_traffic_econometrics.hawkes import (
    FlowRiskSummary,
    HawkesFit,
    fit_multivariate_hawkes,
    hawkes_loglik_and_grad,
    simulate_multivariate_hawkes,
    spectral_branching_ratio,
    summarize_flow_paradox,
)
from bot_traffic_econometrics.panel import (
    EntityTraffic,
    PanelBootstrapCI,
    PanelDecomposition,
    bootstrap_panel,
    decompose_panel,
    wilson_interval,
)
from bot_traffic_econometrics.surge import (
    SurgeAttribution,
    TrendStability,
    assess_pre_event_trend,
    attribute_surge_shapley,
    count_sign_consistent_series,
)

__version__ = "0.1.0"

__all__ = [
    "USING_RUST",
    "CalibratedExample",
    "CensoredEMResult",
    "ChannelParams",
    "EntityTraffic",
    "FlowCensoringObs",
    "FlowRiskSummary",
    "HawkesFit",
    "ManskiBounds",
    "PanelBootstrapCI",
    "PanelDecomposition",
    "StackelbergEquilibrium",
    "SurgeAttribution",
    "SweepCell",
    "TrendStability",
    "__version__",
    "assess_pre_event_trend",
    "attacker_best_response",
    "attribute_surge_shapley",
    "bootstrap_panel",
    "calibrated_retail_panels",
    "count_sign_consistent_series",
    "decompose_panel",
    "defender_loss",
    "fit_censored_sophistication_em",
    "fit_multivariate_hawkes",
    "hawkes_loglik_and_grad",
    "is_direction_identified",
    "manski_sophistication_bounds",
    "run_calibrated_example",
    "sensitivity_sweep",
    "simulate_multivariate_hawkes",
    "solve_stackelberg_defense",
    "spectral_branching_ratio",
    "summarize_flow_paradox",
    "wilson_interval",
]
