"""Terminal display design system — atomic rich-based components.

All CLI output MUST go through these components for visual consistency.
Adapted from pp_tracer/ppt/display.py design tokens.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel as RichPanel
from rich.table import Table

console = Console()


def get_console(file=None):
    """Return a Console instance — injectable for testing."""
    if file is not None:
        from rich.console import Console
        return Console(file=file)
    return console

# ═══════════════════════════════════════════════════════════════════════════════
# Design Tokens
# ═══════════════════════════════════════════════════════════════════════════════

MAX_WIDTH = 100
GUTTER = 2


class Color:
    """Semantic color tokens."""

    fg_strong = "bold white"
    fg_default = "white"
    fg_muted = "grey50"
    fg_dim = "grey35"

    accent = "bold cyan"
    profit = "bold green"
    loss = "bold red"
    warn = "yellow"
    info = "bold magenta"

    border_dim = "grey23"
    border_ok = "green"
    border_warn = "yellow"
    border_crit = "red"


# ═══════════════════════════════════════════════════════════════════════════════
# Atomic Components
# ═══════════════════════════════════════════════════════════════════════════════


def rule(title: str) -> None:
    """Top-level rule with centered title."""
    console.rule(f"[{Color.accent}]{title}[/]", style=Color.fg_muted)


def panel(
    title: str,
    content: str | list[str],
    accent: str = Color.accent,
    border: str = Color.border_ok,
) -> None:
    """Card container with colored left border and title."""
    body = "\n".join(content) if isinstance(content, list) else content
    rp = RichPanel(
        body,
        title=f"[{accent}]{title}[/]",
        border_style=border,
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(rp)


def kv(label: str, value: str) -> str:
    """Dim-label bright-value line for use inside panels."""
    return f"[{Color.fg_muted}]{label}[/]: [{Color.fg_default}]{value}[/]"


def kv_table(pairs: list[tuple[str, str]]) -> None:
    """Multi-row KV table."""
    t = Table(show_header=False, box=None, padding=(0, GUTTER))
    t.add_column(style=Color.fg_muted, justify="right")
    t.add_column(style=Color.fg_default)
    for label, value in pairs:
        t.add_row(label, value)
    console.print(t)


def kpi(label: str, value: str, sub: str | None = None, value_style: str = Color.fg_strong) -> None:
    """Single KPI: small dim label above big bold value."""
    lines = [f"[{Color.fg_muted}]{label}[/]"]
    lines.append(f"[{value_style}]{value}[/]")
    if sub:
        lines.append(f"[{Color.fg_dim}]{sub}[/]")
    console.print("\n".join(lines))


def kpi_row(kpis: list[tuple[str, str, str]]) -> None:
    """Horizontal KPI row. Each tuple: (label, value, style)."""
    cells = []
    for label, value, style in kpis:
        cells.append(f"[{Color.fg_muted}]{label}[/]\n[{style}]{value}[/]")
    t = Table(show_header=False, box=None, padding=(0, GUTTER))
    for _ in kpis:
        t.add_column(justify="center")
    t.add_row(*cells)
    console.print(t)


def result_table(headers: list[str], rows: list[list[str]]) -> None:
    """Styled data table for prediction results."""
    if not rows:
        console.print(f"  [{Color.fg_dim}](no data)[/]")
        return
    t = Table(box=box.SIMPLE_HEAD, border_style=Color.border_dim)
    for i, h in enumerate(headers):
        style = Color.accent if i == 0 else Color.fg_default
        justify = "left" if i == 0 else "right"
        t.add_column(h, style=style, justify=justify)
    for row in rows:
        styled_row = []
        for i, cell in enumerate(row):
            s = Color.fg_default if i > 0 else Color.fg_strong
            styled_row.append(f"[{s}]{cell}[/]")
        t.add_row(*styled_row)
    console.print(t)


def metric_table(headers: list[str], rows: list[list[str]]) -> None:
    """Styled metrics table with accent header."""
    t = Table(box=box.SIMPLE, border_style=Color.border_dim)
    t.add_column(headers[0], style=Color.fg_muted, justify="right")
    for h in headers[1:]:
        t.add_column(h, style=Color.fg_default, justify="right")
    for row in rows:
        t.add_row(*row)
    console.print(t)


def status_badge(status: str) -> str:
    """Semantic badge from status string."""
    mapping = {
        "ok": (" ✓ OK ", Color.profit),
        "warn": (" ⚠ WARN ", Color.warn),
        "crit": (" ● CRIT ", Color.loss),
        "info": (" ℹ INFO ", Color.info),
    }
    text, style = mapping.get(status, (f" {status} ", Color.fg_muted))
    return f"[{style}]{text}[/]"


def note(text: str) -> str:
    """Dimmed remark line."""
    return f"[{Color.fg_dim}]{text}[/]"


def success_banner(text: str) -> None:
    """Green success banner."""
    console.print(f"[{Color.profit}]✅ {text}[/]")


def warn_card(title: str, body: str | None = None) -> None:
    """Warning card."""
    content = f"[{Color.warn}]⚠ {title}[/]"
    if body:
        content += f"\n[{Color.fg_muted}]{body}[/]"
    console.print(content)


def empty_state(message: str = "暂无数据") -> None:
    """Empty state card."""
    panel("📭", f"[{Color.fg_dim}]{message}[/]", border=Color.border_dim)
