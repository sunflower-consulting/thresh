"""Authorization is decided outside the model, and it fails closed.

WHY THIS MODULE IS A DIFFERENTIATOR RATHER THAN PLUMBING.

The comparable tools in this space take a target URL and start attacking it.
There is no scope document, no ownership attestation and no gate: the thing that
decides what gets attacked is the same thing that writes the attacks. That is
fine for a benchmark and wrong for an engagement, because the question "was this
authorised" then has no answer inside the tool.

Here the model proposes and `PolicyEngine.authorize` decides, reading a signed
inventory rather than anything the request claims about itself. Every test below
is a way that could stop being true while the tool still appeared to work.
"""
from __future__ import annotations

from datetime import date

import pytest

from sunflower_thresh.policy.decision import DenyReason, TargetRef, ToolRequest
from sunflower_thresh.policy.engine import PolicyEngine
from sunflower_thresh.schemas.target import ActionClass

LOOPBACK = TargetRef(ip="127.0.0.1")
ELSEWHERE = TargetRef(ip="203.0.113.10")   # RFC 5737 documentation range
#: Genuinely globally routable. TEST-NET-3 above is reserved, so `is_global`
#: reports False for it and it cannot exercise the external-network check.
ROUTABLE = TargetRef(ip="8.8.8.8")


def _req(**kw) -> ToolRequest:
    kw.setdefault("project", "lab")
    # `chat_say` because that is what the tool is called. This said `chat_send`,
    # which no tool in the package answers to, and it passed only because the
    # fixture inventory permitted the same fiction -- so every test in this file
    # exercised the gate with a name that could never have reached a target.
    kw.setdefault("tool", "chat_say")
    kw.setdefault("action_class", ActionClass.ACTIVE)
    return ToolRequest(**kw)


# --- fail closed -----------------------------------------------------------------

def test_a_fresh_gate_refuses_everything():
    """THE test. A gate that defaults to permitting is not a gate.

    This is the difference between this tool and one that takes a URL on the
    command line and begins: with no inventory loaded there is no signed scope,
    so there is nothing a request can be checked against, and the answer is no.
    """
    d = PolicyEngine().authorize(_req(target=LOOPBACK))
    assert not d.allowed
    assert d.deny_code is DenyReason.NO_INVENTORY


def test_the_refusal_says_why_in_words_a_person_can_act_on():
    d = PolicyEngine().authorize(_req(target=LOOPBACK))
    assert "inventory" in d.reason.lower() and "lab" in d.reason


def test_a_signed_scope_permits_the_target_it_names(lab_inventory):
    eng = PolicyEngine()
    assert eng.load_inventory(lab_inventory) is True
    d = eng.authorize(_req(target=LOOPBACK))
    assert d.allowed, d.reason


def test_a_target_outside_the_signed_scope_is_refused(lab_inventory):
    """The scope is an allowlist of addresses, not a denylist. A target nobody
    wrote down is out of scope by construction."""
    eng = PolicyEngine()
    eng.load_inventory(lab_inventory)
    d = eng.authorize(_req(target=ELSEWHERE))
    assert not d.allowed
    assert d.deny_code is DenyReason.TARGET_OUT_OF_SCOPE


def test_a_tool_not_named_in_the_scope_is_refused(lab_inventory):
    eng = PolicyEngine()
    eng.load_inventory(lab_inventory)
    d = eng.authorize(_req(tool="run_exploit", target=LOOPBACK))
    assert not d.allowed
    assert d.deny_code is DenyReason.TOOL_NOT_PERMITTED


# --- the scope document itself has to be intact ----------------------------------

def test_a_scope_whose_hash_does_not_match_cannot_authorise_anything(lab_inventory):
    """An inventory is loaded with the hash the engagement signed for. If the
    file has moved underneath that hash, the document on disk is not the
    document anybody agreed to, and target-bound work stops.
    """
    eng = PolicyEngine()
    assert eng.load_inventory(lab_inventory, expected_hash="0" * 64) is False
    d = eng.authorize(_req(target=LOOPBACK))
    assert not d.allowed
    assert d.deny_code is DenyReason.INVENTORY_UNVERIFIED


def test_an_authorisation_that_has_expired_refuses(expired_inventory):
    """An engagement ends on a date. After it, the same scope file authorises
    nothing, without anybody having to remember to delete it."""
    eng = PolicyEngine()
    assert eng.load_inventory(expired_inventory) is False
    d = eng.authorize(_req(target=LOOPBACK))
    assert not d.allowed
    assert d.deny_code in (DenyReason.INVENTORY_UNVERIFIED, DenyReason.AUTH_INACTIVE)


# --- the killswitch outranks everything ------------------------------------------

def test_the_killswitch_refuses_a_request_that_would_otherwise_be_allowed(lab_inventory):
    """Ordering, pinned. The killswitch is checked before the inventory, so
    engaging it stops work without needing the scope to be reasoned about. A
    killswitch that were checked last could be reached only by requests that
    were already going to be allowed.
    """
    eng = PolicyEngine()
    eng.load_inventory(lab_inventory)
    assert eng.authorize(_req(target=LOOPBACK)).allowed      # allowed before
    eng.killswitch.trip("stopped by the tester")
    d = eng.authorize(_req(target=LOOPBACK))
    assert not d.allowed
    assert d.deny_code is DenyReason.KILLSWITCH
    assert "stopped by the tester" in d.reason


def test_the_killswitch_outranks_a_missing_inventory_too():
    """So the reason a reader sees is the real one. If NO_INVENTORY were
    reported while the killswitch was engaged, the operator would go looking for
    a scope file instead of noticing they had stopped the run."""
    eng = PolicyEngine()
    eng.killswitch.trip("halt")
    assert eng.authorize(_req(target=LOOPBACK)).deny_code is DenyReason.KILLSWITCH


# --- the internet is not in scope by accident ------------------------------------

def test_a_routable_address_needs_external_network_turned_on(lab_inventory):
    """Separate from scope on purpose. A tester who widens the CIDR list to
    reach a staging host should not thereby acquire permission to attack the
    public internet, so routability is its own check with its own reason.
    """
    wide = lab_inventory.model_copy(update={"allowed_cidrs": ("0.0.0.0/0",)})
    eng = PolicyEngine()
    eng.load_inventory(wide)
    d = eng.authorize(_req(target=ROUTABLE))
    assert not d.allowed
    assert d.deny_code is DenyReason.EXTERNAL_NETWORK_BLOCKED


# --- destructive work is off by default ------------------------------------------

def test_a_destructive_action_is_refused_before_a_human_is_even_asked(lab_inventory):
    """Two separate switches, in this order: the capability is off, and even
    turned on it wants a human token. The default is the customer-safe one, so
    an engagement has to opt in twice to let a model change a target."""
    base = {"tool": "chat_reset", "action_class": ActionClass.DESTRUCTIVE, "target": LOOPBACK}
    eng = PolicyEngine()
    eng.load_inventory(lab_inventory)
    assert eng.authorize(_req(**base, human_token="t")).deny_code is DenyReason.CAPABILITY_DISABLED

    on = lab_inventory.model_copy(update={"capabilities": {"modify_target": True}})
    eng = PolicyEngine()
    eng.load_inventory(on)
    assert eng.authorize(_req(**base)).deny_code is DenyReason.DESTRUCTIVE_NEEDS_HUMAN
    assert eng.authorize(_req(**base, human_token="approved-by-tester")).allowed


# --- a request cannot authorise itself -------------------------------------------

def test_nothing_the_request_claims_about_itself_grants_permission(lab_inventory):
    """The gate reads the inventory, never the request's own account of itself.

    This is the property that makes "the model proposes, the gate decides" true
    rather than aspirational: a model that can write a ToolRequest can write any
    field on it, so no field on it may be what unlocks the request.
    """
    eng = PolicyEngine()
    eng.load_inventory(lab_inventory)
    d = eng.authorize(_req(
        target=ELSEWHERE,
        requested_by="the-owner",
        human_token="approved-by-everyone",
        required_capability=None,
    ))
    assert not d.allowed
    assert d.deny_code is DenyReason.TARGET_OUT_OF_SCOPE


# --- budgets are part of the scope, not a tuning knob ----------------------------

def test_the_message_budget_the_scope_signed_for_is_enforced(lab_inventory):
    """A rate is a measurement of a target under load, and an engagement signs
    for how much load. Exhausting the budget is a refusal with its own reason
    rather than a silent stop, so a truncated run cannot read as a complete one.
    """
    small = lab_inventory.model_copy(update={"max_target_messages": 2})
    eng = PolicyEngine()
    eng.load_inventory(small)
    allowed = [eng.authorize(_req(target=LOOPBACK, message_cost=1)).allowed for _ in range(4)]
    assert allowed[:2] == [True, True]
    assert allowed[2:] == [False, False]
    d = eng.authorize(_req(target=LOOPBACK, message_cost=1))
    assert d.deny_code is DenyReason.MESSAGE_BUDGET_EXHAUSTED


# --- every decision is recorded --------------------------------------------------

def test_a_refusal_is_written_to_the_audit_log(lab_inventory):
    """A gate whose refusals are not recorded cannot be reviewed afterwards,
    and 'what did this tool try to do' is the first question anybody asks."""
    eng = PolicyEngine()
    eng.load_inventory(lab_inventory)
    eng.authorize(_req(target=ELSEWHERE))
    kinds = [e.get("event") for e in eng.audit.entries()]
    assert "inventory_loaded" in kinds


@pytest.mark.parametrize("reason", list(DenyReason))
def test_every_deny_reason_is_a_distinct_readable_string(reason):
    """They reach a report, so they are part of the deliverable surface."""
    assert reason.value and reason.value == reason.value.lower()
    assert " " not in reason.value


# --- a dropped field is an error, not a default ----------------------------------

def test_a_field_the_scope_model_does_not_know_is_rejected():
    """Several of these fail OPEN if silently ignored, which is why `extra` is
    forbidden rather than merely tidy.

    `expires` instead of `expires_on` leaves `expires_on` as None, and an
    authorization with no expiry never stops authorising. `capability` instead
    of `required_capability` leaves the capability check with nothing to check,
    so a request that should have been refused is permitted. Both were accepted
    silently before, and neither would show up anywhere a reader was looking.
    """
    from pydantic import ValidationError

    from sunflower_thresh.schemas.target import AuthorizationRecord

    with pytest.raises(ValidationError, match="extra"):
        ToolRequest(project="lab", tool="chat_say",
                    action_class=ActionClass.ACTIVE, capability="out_of_band_callback")

    with pytest.raises(ValidationError, match="extra"):
        AuthorizationRecord(authorized_by="a", authorized_for="b", scope_summary="s",
                            granted_on=date(2026, 1, 1), reference="r",
                            expires="2026-12-31")


def test_a_mistyped_budget_cannot_quietly_become_unlimited(lab_inventory):
    """The specific fail-open case, named. `max_messages` is not a field; before
    `extra=forbid` it left `max_target_messages` None, which is no budget at
    all, on an engagement that had written a number down."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="extra"):
        lab_inventory.model_copy(update={}).__class__(
            project="lab", allowed_cidrs=("127.0.0.1/32",),
            allowed_tools=("chat_say",), max_messages=10,
            authorization=lab_inventory.authorization)
