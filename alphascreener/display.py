"""Rich components used by the command-line interface."""

from rich import box
from rich.console import Console
from rich.panel import Panel as RichPanel
from rich.table import Table

console = Console()


class Color:
    """Semantic colors used by the retained CLI components."""

    fg_strong = "bold white"
    fg_default = "white"
    fg_muted = "grey50"
    accent = "bold cyan"
    warn = "yellow"
    border_dim = "grey23"
    border_ok = "green"


def rule(title: str) -> None:
    """Print a top-level section rule."""
    console.rule(f"[{Color.accent}]{title}[/]", style=Color.fg_muted)


def panel(
    title: str,
    content: str | list[str],
    accent: str = Color.accent,
    border: str = Color.border_ok,
) -> None:
    """Print a titled panel."""
    body = "\n".join(content) if isinstance(content, list) else content
    console.print(
        RichPanel(
            body,
            title=f"[{accent}]{title}[/]",
            border_style=border,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def result_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a prediction result table."""
    if not rows:
        console.print("  [grey35](no data)[/]")
        return
    table = Table(box=box.SIMPLE_HEAD, border_style=Color.border_dim)
    for index, header in enumerate(headers):
        table.add_column(
            header,
            style=Color.accent if index == 0 else Color.fg_default,
            justify="left" if index == 0 else "right",
        )
    for row in rows:
        table.add_row(
            *[
                f"[{Color.fg_default if index else Color.fg_strong}]{cell}[/]"
                for index, cell in enumerate(row)
            ]
        )
    console.print(table)


def warn_card(title: str, body: str | None = None) -> None:
    """Print a warning message."""
    content = f"[{Color.warn}]⚠ {title}[/]"
    if body:
        content += f"\n[{Color.fg_muted}]{body}[/]"
    console.print(content)
