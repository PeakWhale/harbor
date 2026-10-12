"""Small helpers to show numbers to people."""

from __future__ import annotations

import math


def format_number(x: float | int | None) -> str:
    """Readable number: 215,300 / 52 / 31.4 / 2.75 / 0.835."""
    if x is None:
        return ""
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    size = abs(x)
    if x.is_integer() or size >= 1000:
        return f"{x:,.0f}"
    if size >= 1:
        return f"{x:,.2f}".rstrip("0").rstrip(".")
    return f"{x:.3g}"


def format_percent(x: float | None) -> str:
    if x is None:
        return ""
    return f"{float(x) * 100:.1f}%"
