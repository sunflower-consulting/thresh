"""Tiny, dependency-free ANSI styling for the CLI surfaces.

Taste rules: cyan carries *structure* (headers, arrows), green/yellow/red carry
*judgement* (how good a number is), and dim-grey carries *secondary detail*
(units, budgets, parentheticals). Bold is for names only. Nothing else.

Colour is emitted only to a real terminal. It honours ``NO_COLOR`` (disable) and
``FORCE_COLOR`` (force on, e.g. for piping into ``less -R``), so redirected or
captured output stays plain text.
"""
from __future__ import annotations

import os
import sys

_CODES = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "grey": "90",
    "bright_green": "92",
}


def supports_color(stream=None) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    stream = stream or sys.stdout
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


ENABLED = supports_color()


def style(text: str, *names: str) -> str:
    """Wrap ``text`` in the named SGR styles, or return it plain if disabled."""
    if not ENABLED or not names:
        return text
    seq = ";".join(_CODES[n] for n in names)
    return f"\033[{seq}m{text}\033[0m"


# --- semantic helpers --------------------------------------------------------
def header(text: str) -> str:
    return style(text, "bold", "cyan")


def name(text: str) -> str:
    return style(text, "bold")


def muted(text: str) -> str:
    return style(text, "grey")


def arrow() -> str:
    return style("->", "cyan")


def quantity(value: float, good: float = 0.75, ok: float = 0.5) -> str:
    """Colour a normalised 0..1 metric: green strong, yellow fair, red weak."""
    s = f"{value:.3f}"
    if value >= good:
        return style(s, "green")
    if value >= ok:
        return style(s, "yellow")
    return style(s, "red")


def signed(value: float) -> str:
    """Colour a signed score: green for positive, red for negative, grey zero."""
    s = f"{value:+.3f}"
    if value > 0:
        return style(s, "green")
    if value < 0:
        return style(s, "red")
    return style(s, "grey")


def good(text: str) -> str:
    return style(text, "green")


def warn(text: str) -> str:
    return style(text, "yellow")


def bad(text: str) -> str:
    return style(text, "red")
