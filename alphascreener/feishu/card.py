"""Interactive card JSON template and field mapping.

Issue #104: Feishu daily card push.
Reference: PRD 7.5.3 — full card JSON template plus field mapping table.

Provides:
  - CardData: structured input for all card fields.
  - build_card_json(): render the interactive card as a JSON string.
  - build_fallback_text(): build a plain-text fallback message (400 degradation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_NO_ALERTS: str = "ok"

# ============================================================================
# CardData — structured input for all card fields
# ============================================================================


@dataclass
class CardData:
    """Structured input for the Feishu interactive card.

    All fields have sensible defaults so a partially-populated card can
    still be rendered during development or when some data sources are
    unavailable.
    """

    report_date: str = ""
    total_symbols: int | None = None
    coarse_pass: int | None = None
    refine_count: int | None = None

    # Top 5 tickers: list of dicts with keys ticker/rating/confidence/catalyst
    top_five: list[dict[str, Any]] = field(default_factory=list)

    # Alpha acceptance — Ablation dual-track metrics
    p20_pure: float = 0.0
    p20_llm: float = 0.0
    lift_pure: float = 0.0
    lift_llm: float = 0.0
    base_rate: float = 0.0

    # Backtest performance — rolling 7-day window
    win_rate: float = 0.0
    sharpe: float = 0.0
    avg_return: float = 0.0

    # Cost tracking
    daily_cost: float = 0.0
    monthly_cost: float | None = None

    # Alerts summary (or "ok" when no alerts)
    alerts_summary: str = _NO_ALERTS


# ============================================================================
# Card JSON builder
# ============================================================================


def _na(val: Any, fmt_spec: str = "") -> str:
    """Format a value, returning "N/A" when None so zero vs. unavailable are distinct."""
    if val is None:
        return "N/A"
    if fmt_spec:
        return format(val, fmt_spec)
    return str(val)


def _build_top5_table(top_five: list[dict[str, Any]]) -> str:
    """Render the Top 5 tickers as a markdown table string."""
    if not top_five:
        return "今日无精筛标的"
    rows = []
    for i, t in enumerate(top_five[:5], start=1):
        ticker = t.get("ticker", "-")
        rating = t.get("rating", "-")
        confidence = t.get("confidence", 0)
        catalyst = t.get("catalyst", "-")
        rows.append(f"| {i} | {ticker} | {rating} | {confidence:.1f}% | {catalyst} |")
    header = "| 排名 | 标的 | 评级 | 置信度 | 主要催化剂 |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


def build_card_json(data: CardData) -> str:
    """Build the full interactive card JSON string.

    Args:
        data: Populated :class:`CardData` with all field values.

    Returns:
        JSON string suitable for the ``content`` field of the Feishu
        message send API.
    """
    alerts_text = data.alerts_summary if data.alerts_summary != _NO_ALERTS else "ok"

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": f"AlphaScreener | {data.report_date}",
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": (
                        "**Scan**\n"
                        f"Universe: {_na(data.total_symbols)} | "
                        f"Coarse: {_na(data.coarse_pass)} | "
                        f"Refine: {_na(data.refine_count)}"
                    ),
                },
                {
                    "tag": "markdown",
                    "content": (f"**Top 5**\n{_build_top5_table(data.top_five)}"),
                },
                {
                    "tag": "markdown",
                    "content": (
                        "**Alpha (pure / llm)**\n"
                        f"Precision@20: {data.p20_pure:.1f}% / {data.p20_llm:.1f}%\n"
                        f"Lift@20: {data.lift_pure:.2f} / {data.lift_llm:.2f}\n"
                        f"base_rate: {data.base_rate:.1f}%"
                    ),
                },
                {
                    "tag": "markdown",
                    "content": (
                        "**Backtest (7d rolling)**\n"
                        f"Win: {data.win_rate:.1f}% | "
                        f"Sharpe: {data.sharpe:.2f} | "
                        f"Avg ret: {data.avg_return:.1f}%"
                    ),
                },
                {
                    "tag": "markdown",
                    "content": (
                        "**Cost**\n"
                        f"Today: ${data.daily_cost:.2f} | "
                        f"Month: ${_na(data.monthly_cost, '.2f')}/$100"
                    ),
                },
                {"tag": "hr"},
                {
                    "tag": "markdown",
                    "content": f"**Alerts**\n{alerts_text}",
                },
            ],
        },
    }
    return json.dumps(card, ensure_ascii=False)


# ============================================================================
# Fallback plain-text builder (400 card-rendering degradation)
# ============================================================================


def build_fallback_text(
    report_date: str,
    top_five: list[dict[str, Any]],
    p20_pure: float,
    p20_llm: float,
    lift_pure: float,
    lift_llm: float,
    base_rate: float,
    alerts_summary: str,
) -> str:
    """Build a plain-text fallback message when card rendering fails (400).

    Args:
        report_date: Report date string.
        top_five: Top 5 tickers list.
        p20_pure: Precision@20 pure track.
        p20_llm: Precision@20 LLM track.
        lift_pure: Lift@20 pure track.
        lift_llm: Lift@20 LLM track.
        base_rate: Base hit rate (%).
        alerts_summary: Alerts text.

    Returns:
        Plain-text message string.
    """
    lines = [f"AlphaScreener | {report_date}", ""]
    if top_five:
        lines.append("Top 5:")
        for i, t in enumerate(top_five[:5], start=1):
            lines.append(
                f"  {i}. {t.get('ticker', '-')} "
                f"({t.get('rating', '-')}, "
                f"{t.get('confidence', 0):.1f}%)"
            )
    else:
        lines.append("Top 5: --")

    lines.append("")
    lines.append(
        f"Alpha: P@20={p20_pure:.1f}%/{p20_llm:.1f}% "
        f"Lift@20={lift_pure:.2f}/{lift_llm:.2f} "
        f"base_rate={base_rate:.1f}%"
    )
    alerts_text = alerts_summary if alerts_summary != "ok" else "ok"
    lines.append(f"Alerts: {alerts_text}")

    return "\n".join(lines)
