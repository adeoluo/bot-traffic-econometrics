"""Typed bridge to the compiled `_bte_rs` Rust extension."""

from __future__ import annotations

import importlib
import os
from typing import Any

_rs: Any = None
try:
    _rs = importlib.import_module("bot_traffic_econometrics._bte_rs")
except ImportError:  # pragma: no cover - exercised only if built without maturin
    if os.environ.get("BTE_REQUIRE_RUST") == "1":
        raise

USING_RUST: bool = _rs is not None


def require_rust() -> Any:
    if _rs is None:  # pragma: no cover
        raise RuntimeError(
            "Compiled Rust extension `bot_traffic_econometrics._bte_rs` is not installed; "
            "build with `uv sync` or `maturin develop`."
        )
    return _rs
