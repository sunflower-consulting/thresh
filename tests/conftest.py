"""Shared fixtures.

The inventory here is the shape a chat engagement actually has: a loopback
endpoint, the chat verbs, and no external network. The private codebase this was
extracted from had an OT inventory of PLCs and CIDRs, which is not what this
package is for.
"""
from datetime import date, timedelta

import pytest

from sunflower_stem.schemas.target import AuthorizationRecord, TargetInventory

#: The tools a chat assessment needs and nothing else. An inventory is an
#: allowlist, so this list is the whole permitted surface.
#:
#: These are the names the tools actually answer to. It said `chat_send` and
#: `chat_history` before, and this package has neither: `tool_allowed` is exact
#: membership, so an inventory built from this fixture permitted nothing that exists
#: and omitted `chat_say`, which is the one verb without which no case can run. It
#: went unnoticed because nothing had yet asked the fixture to run a chat case --
#: `stem preflight` is the first thing that does.
CHAT_TOOLS = ("chat_open", "chat_say", "chat_reset")


def _authorization(**over) -> AuthorizationRecord:
    base = {
        "authorized_by": "owner",
        "authorized_for": "tester",
        "scope_summary": "the bundled lab on loopback",
        "granted_on": date(2026, 1, 1),
        "reference": "personally-owned-lab",
    }
    return AuthorizationRecord(**{**base, **over})


@pytest.fixture
def lab_inventory() -> TargetInventory:
    """A signed scope for the bundled lab. Loopback only, no external network."""
    return TargetInventory(
        project="lab",
        allowed_cidrs=("127.0.0.1/32",),
        allowed_domains=("localhost",),
        external_network=False,
        allowed_tools=CHAT_TOOLS,
        authorization=_authorization(),
    )


@pytest.fixture
def expired_inventory() -> TargetInventory:
    """Same scope, authorization that has run out. An engagement ends."""
    yesterday = date.today() - timedelta(days=1)
    return TargetInventory(
        project="lab",
        allowed_cidrs=("127.0.0.1/32",),
        external_network=False,
        allowed_tools=CHAT_TOOLS,
        authorization=_authorization(granted_on=date(2026, 1, 1), expires_on=yesterday),
    )
