"""Tiny, dependency-free ANSI styling for the CLI surfaces.

Taste rules: cyan carries *structure* (headers, arrows), green/yellow/red carry
*judgement* (how good a number is), and grey carries *secondary detail*
(units, budgets, parentheticals). Bold is for names only. Nothing else.

Colour is emitted only to a real terminal. It honours ``NO_COLOR`` (disable) and
``FORCE_COLOR`` (force on, e.g. for piping into ``less -R``), so redirected or
captured output stays plain text.
"""
from __future__ import annotations

import os
import sys

#: Exactly the SGR names the semantic helpers below request, and nothing else.
#: `style` emits the reset itself, so "reset" was never a lookup either.
#:
#: Kept in step with the helpers deliberately. An entry no helper asks for is a
#: colour this module cannot emit and no caller can reach -- `style` is not
#: re-exported and has no caller outside this file -- so it reads as a palette
#: while being four dead strings. The taste rules above are the whole palette.
_CODES = {
    "bold": "1",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "grey": "90",
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


def good(text: str) -> str:
    return style(text, "green")


def warn(text: str) -> str:
    return style(text, "yellow")


def bad(text: str) -> str:
    return style(text, "red")
