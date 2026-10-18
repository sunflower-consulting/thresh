"""Predicting the gate, and the collector assertion nobody checked.

TWO DEFECTS THIS FILE EXISTS FOR.

A preflight that passes an engagement the gate will refuse is worse than no
preflight: the operator opens the window, tells the client they are starting, and
the first request comes back denied. The engine this was ported from could not
predict four of the gate's eleven refusals, and the expensive one is
`AUTH_INACTIVE` -- `AuthorizationRecord.is_active()` has always existed and the gate
has always enforced it, while preflight read `reference` and never asked about
expiry. So the partition test below is the point of the module, not a detail of it.

And `--collector` was an operator ASSERTION fed straight into case selection. The
corpus's out-of-band cases carry a structural oracle as well as a callback one, so
they run either way; what an unchecked assertion bought was those cases being marked
full-ladder when their strongest rung could never fire, on the data-exfiltration
class, where the difference between CALLBACK and STRUCTURAL is the difference
between the target's own traffic and the model having emitted a string.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from sunflower_stem.collector import CALLBACK_CAPABILITY, CallbackCollector
from sunflower_stem.policy.decision import DenyReason
from sunflower_stem.preflight import (
    BASELINE_TRIALS,
    CHAT_TOOL,
    PREDICTS,
    UNPREDICTABLE,
    CollectorReach,
    EngagementPreflight,
    ScopeError,
    Status,
    load_inventory,
    mock_sender,
    probe_collector,
    render,
    unpredicted_denials,
)
from sunflower_stem.schemas.target import AuthorizationRecord, TargetInventory

CORPUS = Path("/Users/napahlm/Projects/GitHub/seed-vault")
needs_corpus = pytest.mark.skipif(
    not (CORPUS / "cases").is_dir(),
    reason="no seed-vault checkout; the corpus is private and this repo is public")


def _auth(**over) -> AuthorizationRecord:
    base = {"authorized_by": "ACME Ltd", "authorized_for": "sunflower",
            "scope_summary": "chatbot assessment", "granted_on": date(2026, 1, 1),
            "expires_on": date.today() + timedelta(days=30),
            "reference": "SOW-2026-114"}
    return AuthorizationRecord(**{**base, **over})


def _inv(**over) -> TargetInventory:
    base = {"project": "acme", "allowed_domains": ("chat.acme.invalid",),
            "allowed_tools": ("chat_open", CHAT_TOOL, "chat_reset"),
            "capabilities": {"plant_content": True, "multi_turn_escalation": True},
            "target_features": ("tools", "stateful"),
            "seedable_channels": ("document",),
            "seeds": {"canary": "ACME-INTERNAL-4417"},
            "max_target_messages": 5000,
            "baseline_turns": ("hours?", "returns?", "shipping?", "payment?"),
            "authorization": _auth()}
    return TargetInventory(**{**base, **over})


def _pf(**over) -> EngagementPreflight:
    """A preflight whose remote path is wired to answer, so offline checks dominate."""
    base = {"inv": _inv(), "host": "chat.acme.invalid", "port": 443, "tls": True,
            "verified": True, "sender": mock_sender("instrumented"),
            "resolver": lambda h: "203.0.113.1", "connector": lambda *a: None}
    base.update(over)
    return EngagementPreflight(**base)


def _row(results, name):
    return next((r for r in results if r.name == name), None)


class _Exploding:
    """A sender that fails the test if anything reaches the target."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        raise AssertionError("preflight sent something it should not have")


class _FakeCollector:
    """A collector whose bind address and recording are under the test's control."""

    def __init__(self, host="203.0.113.9", port=8080, record=True):
        self.host, self.port, self.running = host, port, True
        self._record = record
        self._hits: dict[str, list[dict]] = {}
        self.minted: list[str] = []

    @property
    def netloc(self):
        return f"{self.host}:{self.port}"

    def mint(self, label=""):
        token = f"tok{len(self.minted)}"
        self.minted.append(token)
        self._hits[token] = []
        return token

    def url_for(self, token):
        return f"http://{self.netloc}/c/{token}"

    def hits(self, token):
        return list(self._hits.get(token, []))

    def receive(self, token, client="203.0.113.9"):
        self._hits.setdefault(token, []).append({"client": client})


def _fetcher(collector, *, client="203.0.113.9", answer=True):
    """A fetch that makes the collector record the token in the URL, as a real one would."""
    def fetch(url, *, timeout=5.0):
        if not answer:
            raise OSError("connection refused")
        token = url.rsplit("/", 1)[-1]
        if collector._record:
            collector.receive(token, client=client)
        return 200
    return fetch


# --- the partition against the gate's own refusals -------------------------------

def test_every_refusal_the_gate_can_give_is_predicted_or_written_off():
    """The drift alarm, and the reason this module is arranged the way it is.

    A `DenyReason` that preflight neither predicts nor excuses is a run that dies
    mid-window for a reason nobody looked for. Adding one to the gate fails here
    until this module says what it will do about it."""
    assert unpredicted_denials() == set(), (
        "the gate can refuse a run for a reason preflight does not check and does "
        "not excuse")
    assert not (set(PREDICTS) & set(UNPREDICTABLE)), "a refusal is one or the other"
    assert set(PREDICTS) | set(UNPREDICTABLE) == set(DenyReason)


def test_a_refusal_written_off_has_to_argue_for_itself():
    """An entry in UNPREDICTABLE is an exemption, so it carries its reason. A bare
    name there would be the partition passing while meaning nothing."""
    for reason, why in UNPREDICTABLE.items():
        assert len(why) > 80, f"{reason.value} is excused without a reason"


def test_every_named_check_exists():
    """PREDICTS maps a refusal to a check by name, and a name is not a reference.
    A renamed check would leave the partition passing and the prediction gone."""
    names = {r.name for r in _pf(cases={}).run()}
    missing = {d.value: c for d, c in PREDICTS.items() if c not in names}
    assert not missing, f"PREDICTS names checks that do not appear: {missing}"


# --- fail closed -----------------------------------------------------------------

def test_a_lapsed_authorization_stops_the_run_and_the_gate_agrees():
    """THE defect. The gate enforces `is_active()` on every single call, and the
    preflight this was ported from read `reference` and never asked -- so an
    engagement whose authorization ran out overnight passed its morning preflight
    and then had every request refused, after the client had been told."""
    from sunflower_stem.policy.decision import ActionClass, ToolRequest, target_ref
    from sunflower_stem.policy.engine import PolicyEngine

    yesterday = date.today() - timedelta(days=1)
    inv = _inv(authorization=_auth(expires_on=yesterday))

    row = _row(_pf(inv=inv).run(), "authorization")
    assert row.status is Status.FAIL
    assert "NOT ACTIVE" in row.detail
    assert str(yesterday) in row.detail, "it has to say WHEN, or nobody can act on it"

    # And the same engagement through the gate, so the two cannot be read apart.
    engine = PolicyEngine()
    engine.load_inventory(inv)
    decision = engine.authorize(ToolRequest(
        project="acme", tool=CHAT_TOOL, action_class=ActionClass.ACTIVE,
        target=target_ref("chat.acme.invalid")))
    assert not decision.allowed
    assert decision.deny_code in (DenyReason.AUTH_INACTIVE,
                                  DenyReason.INVENTORY_UNVERIFIED)


def test_nothing_is_sent_when_the_scope_check_fails():
    """The one thing this command must never do. A host the scope does not cover is
    not probed to find out whether it answers -- the finding would be a request to
    somebody we are not authorised to touch, and the readout has to say so."""
    sender = _Exploding()
    results = _pf(host="chat.someone-else.invalid", sender=sender,
                  resolver=sender, connector=sender).run()
    assert sender.calls == 0
    assert _row(results, "scope").status is Status.FAIL
    target = _row(results, "target")
    assert target.status is Status.SKIP
    assert "nothing was sent" in target.hint
    assert _row(results, "dns") is None, "it must not have got as far as resolving"


def test_a_tripped_killswitch_stops_before_anything_else():
    """It wins over everything in the gate, so it wins over everything here."""
    class _Tripped:
        engaged = True
        reason = "watchdog"

    sender = _Exploding()
    results = _pf(killswitch=_Tripped(), sender=sender, resolver=sender).run()
    assert len(results) == 1
    assert results[0].status is Status.FAIL
    assert "watchdog" in results[0].detail
    assert sender.calls == 0


def test_no_killswitch_wired_is_a_gap_in_sight_and_not_a_reason_to_stop():
    """A SKIP here means nobody passed one, which is something preflight cannot see
    rather than something wrong. Treating it as a failure stopped the whole command
    after one row, which is how this was found."""
    results = _pf().run()
    assert results[0].name == "kill-switch"
    assert results[0].status is Status.SKIP
    assert len(results) > 10, "the rest of the checks still have to run"


def test_an_unverified_inventory_is_a_failure_whatever_the_document_says():
    """The gate's own fail-closed point. `authorization` is a required field, so a
    loadable inventory always has one; what actually decides a run is whether the
    gate verified it."""
    assert _row(_pf(verified=False).run(), "inventory").status is Status.FAIL
    assert _row(_pf(verified=None).run(), "inventory").status is Status.WARN


# --- scope, precisely ------------------------------------------------------------

def test_a_target_written_with_its_port_is_still_in_scope():
    """`domain_in_scope` matches a hostname and does not parse a netloc, and an
    operator types what they were given. Passing it through reported a target out of
    scope that the engagement plainly covers, which sends somebody editing a signed
    document to fix a typo in a flag."""
    for written in ("chat.acme.invalid:443", "https://chat.acme.invalid/v1/chat",
                    "CHAT.ACME.INVALID", "chat.acme.invalid."):
        row = _row(_pf(host=written).run(), "scope")
        assert row.status is Status.PASS, f"{written} should be in scope"


def test_a_routable_address_needs_the_external_network_switch():
    """The gate refuses a globally routable ADDRESS unless the inventory says so, and
    the old preflight never looked -- so an engagement with the default
    `external_network: false` passed and then had every request denied."""
    inv = _inv(allowed_cidrs=("8.8.8.0/24",), allowed_domains=())
    row = _row(_pf(inv=inv, host="8.8.8.8").run(), "scope")
    assert row.status is Status.FAIL
    assert "external_network" in row.hint

    allowed = _inv(allowed_cidrs=("8.8.8.0/24",), allowed_domains=(),
                   external_network=True)
    assert _row(_pf(inv=allowed, host="8.8.8.8").run(), "scope").status is Status.PASS


def test_a_routable_hostname_is_not_caught_by_the_external_network_switch():
    """Because the gate does not catch it either: the block reads `target.ip`, and
    `target_ref` routes a hostname to `domain`. Preflight's job is to predict the
    gate, so inventing a stricter rule here would make it disagree -- the worst
    outcome for a tool whose whole value is agreeing."""
    assert _row(_pf().run(), "scope").status is Status.PASS


def test_a_scope_without_the_chat_verb_can_run_nothing():
    inv = _inv(allowed_tools=("chat_open", "chat_reset"))
    row = _row(_pf(inv=inv).run(), "scope")
    assert row.status is Status.FAIL
    assert CHAT_TOOL in row.hint


# --- the two vocabularies that fail open -----------------------------------------

def test_a_misspelt_capability_is_named_rather_than_read_as_off():
    """`extra="forbid"` guards model FIELDS, and `capabilities` is a plain dict -- so
    `plant_contnet: true` validates, reads as off, and demotes every case needing it
    to UNREACHABLE with no diagnostic anywhere. Same shape as the typos
    `extra=forbid` was added for, and the same direction: a document that says yes is
    read as no, and the report shows no gap."""
    inv = _inv(capabilities={"plant_contnet": True, "plant_content": True})
    row = _row(_pf(inv=inv).run(), "capability names")
    assert row.status is Status.FAIL
    assert "plant_contnet" in row.detail
    assert "plant_content" not in row.detail.replace("plant_contnet", "")


def test_a_tool_no_tool_answers_to_grants_nothing():
    """`tool_allowed` is exact membership, so an unknown name is not a permission
    with a typo in it, it is no permission at all. This was live in this repository's
    own fixture, which permitted `chat_send` and `chat_history`."""
    inv = _inv(allowed_tools=("chat_send", CHAT_TOOL, "chat_history"))
    row = _row(_pf(inv=inv).run(), "tool names")
    assert row.status is Status.FAIL
    assert "chat_send" in row.detail
    assert "chat_history" in row.detail
    assert CHAT_TOOL in row.hint, "and it has to say what the real names are"


def test_the_fixtures_this_package_ships_name_tools_that_exist():
    """The drift alarm for the above. A fixture inventory that permits nothing real
    cannot run a chat case, in a package whose whole subject is chat cases."""
    from tests.conftest import CHAT_TOOLS as FIXTURE_TOOLS

    from sunflower_stem.demo import CHAT_TOOLS as DEMO_TOOLS

    for label, tools in (("demo", DEMO_TOOLS), ("conftest", FIXTURE_TOOLS)):
        row = _row(_pf(inv=_inv(allowed_tools=tuple(tools))).run(), "tool names")
        assert row.status is Status.PASS, f"{label} names a tool that does not exist"
        assert CHAT_TOOL in tools, f"{label} omits the one verb a case needs"


# --- the auth header, and the two spellings that go wrong quietly ----------------

def test_a_bare_dollar_var_is_refused_because_it_would_be_sent_verbatim():
    """Only `${VAR}` is expanded. Written `$VAR` it is neither resolved nor reported
    unresolved: it goes to the client's endpoint as the literal characters, which
    authenticates nothing and puts a string that looks like a credential in their
    logs. Nothing anywhere warned about the spelling."""
    sender = _Exploding()
    inv = _inv(auth_headers={"Authorization": "Bearer $ACME_TOKEN"})
    row = _row(_pf(inv=inv, sender=sender).run(), "auth")
    assert row.status is Status.FAIL
    assert "bare $VAR" in row.detail
    assert sender.calls == 0, "and it must be caught before anything is sent"


def test_an_unset_variable_says_how_many_headers_did_resolve(monkeypatch):
    """One stale `${OLD_TOKEN}` in a document with a second working header is a
    different situation from nothing resolving, and the operator has to be able to
    tell which they are in."""
    monkeypatch.setenv("ACME_TOKEN", "live-value")
    monkeypatch.delenv("ACME_OLD", raising=False)
    inv = _inv(auth_headers={"Authorization": "Bearer ${ACME_TOKEN}",
                             "X-Legacy": "${ACME_OLD}"})
    row = _row(_pf(inv=inv, sender=_Exploding()).run(), "auth")
    assert row.status is Status.FAIL
    assert "X-Legacy" in row.detail
    assert "1 of 2 did resolve" in row.detail


# --- what it will cost -----------------------------------------------------------

def test_the_baseline_is_counted_once_and_not_once_per_case():
    """Counting it per case overstates the full corpus by more than the run itself
    costs, and would tell an operator a feasible run is impossible."""
    pf = _pf()
    pf.runnable, pf.skipped = pf.select()
    cases = pf._runnable_cases()
    assert cases, "the built-ins have to be selectable for this to mean anything"
    base = len(pf.inv.baseline_turns)
    once = base * BASELINE_TRIALS
    attack = sum(len(c.turns) * c.trials for c in cases)
    matched = sum(len(c.control_turns) * c.control_trials for c in cases)
    assert pf.estimated_messages() == attack + matched + once + 1

    # The claim, stated as the difference rather than as an inequality that only
    # holds while the baseline term dominates: charging the baseline per case adds
    # one whole baseline arm for every case after the first.
    per_case = attack + matched + once * len(cases) + 1
    assert per_case - pf.estimated_messages() == once * (len(cases) - 1)


def test_the_baseline_trial_count_is_the_one_the_sweep_actually_sends():
    """Not `max(case.control_trials)`, which is what the ported formula used.
    `sweep_builtins` builds `BaselineArm(turns, trials or 10)`, so it is the run's
    override or a hardcoded ten, and the two disagree for any ordinary selection."""
    source = Path(__import__("sunflower_stem.sweep", fromlist=["x"]).__file__).read_text()
    assert f"trials or {BASELINE_TRIALS}" in source, (
        "the sweep's baseline trial count changed and this estimate did not")


def test_a_trials_override_replaces_every_cases_own_figure():
    """Because that is what the sweep does: `_validate_one` passes `trials=trials,
    control_trials=trials`, and `validate_chat_case` falls back to the case's own
    only when the argument is None. An estimate that reads `case.trials` is right
    only for a run that passes no override."""
    pf = _pf(trials=3)
    pf.runnable, pf.skipped = pf.select()
    cases = pf._runnable_cases()
    base = len(pf.inv.baseline_turns)
    attack = sum(len(c.turns) * 3 for c in cases)
    matched = sum(len(c.control_turns) * 3 for c in cases)
    assert pf.estimated_messages() == attack + matched + base * 3 + 1


def test_an_overspent_budget_is_a_failure_before_the_run_finds_out():
    inv = _inv(max_target_messages=3)
    row = _row(_pf(inv=inv).run(), "message budget")
    assert row.status is Status.FAIL
    assert "3 signed" in row.detail


def test_no_signed_budget_is_a_warning_and_says_why():
    row = _row(_pf(inv=_inv(max_target_messages=None)).run(), "message budget")
    assert row.status is Status.WARN
    assert "cannot be undone" in row.hint


# --- the clamp a trials override lifts -------------------------------------------

def test_a_trials_override_that_lifts_the_consumption_cap_is_refused():
    """`to_chat_case` caps a case whose oracle measures COST at two trials, because
    ten trials plus ten controls against a client's assistant is a load test nobody
    signed for. A sweep-level override replaces the case's figure outright, so it
    lifts the cap on exactly the cases it was written for."""
    from sunflower_stem.preflight import CONSUMPTION_TRIALS, EXPENSIVE_RUNGS

    class _Costly:
        id = "LOAD-1"
        turns, control_turns = (), ()
        trials = control_trials = 2
        oracles = ({EXPENSIVE_RUNGS[0]: {"max": 100}},)
        needs = ()
        control_kind = "matched"

    pf = _pf(cases={"LOAD-1": _Costly()})
    pf.runnable = {"LOAD-1": object()}
    assert pf.trials == 0
    assert pf.check_consumption_clamp().status is Status.PASS

    over = _pf(cases={"LOAD-1": _Costly()}, trials=CONSUMPTION_TRIALS + 8)
    over.runnable = {"LOAD-1": object()}
    row = over.check_consumption_clamp()
    assert row.status is Status.FAIL
    assert "LOAD-1" in row.hint
    assert "outage we caused" in row.hint


# --- the collector probe ---------------------------------------------------------

def test_a_loopback_collector_is_reported_unreachable_rather_than_working():
    """It answers, which is not the question. Bound to loopback, nothing outside
    this machine can reach it -- ever -- so this is definitive rather than cautious,
    and `--collector` asserting otherwise was the gap."""
    with CallbackCollector() as col:
        reach, detail = probe_collector(col)
    assert reach is CollectorReach.LOOPBACK_ONLY
    assert "127.0.0.1" in detail


def test_a_collector_that_records_nothing_is_broken_not_absent():
    """Worse than having none, because nothing stops the cases being selected."""
    col = _FakeCollector(record=False)
    reach, detail = probe_collector(col, fetch=_fetcher(col))
    assert reach is CollectorReach.BROKEN
    assert "recorded no hit" in detail


def test_the_probe_refuses_to_fetch_a_host_this_machine_does_not_answer_on():
    """A typo in `--advertise` would put our traffic on somebody else's server, on
    the strength of a mistyped flag. Refusing is the worse readout and the better
    default, and the readout says nothing was sent."""
    col = _FakeCollector()
    reached: list[str] = []
    ours = _fetcher(col)

    def fetch(url, *, timeout=5.0):
        reached.append(url)
        if "not-ours.invalid" in url:
            raise AssertionError("the probe fetched a host we do not answer on")
        return ours(url, timeout=timeout)

    reach, detail = probe_collector(col, advertised="http://not-ours.invalid",
                                    fetch=fetch, local={"203.0.113.9"})
    assert reach is CollectorReach.REFUSED_OFF_HOST
    assert "nothing was sent" in detail
    assert all("not-ours.invalid" not in u for u in reached)
    assert reached, "but it still had to prove the listener works on loopback first"


def test_a_name_that_answers_is_reported_as_exactly_that_and_no_more():
    """NAME_ANSWERS is the strongest state preflight can reach, and the hint has to
    say what it is not: only the target's own request proves the target can reach us,
    and that is what the run measures."""
    col = _FakeCollector()
    reach, detail = probe_collector(
        col, advertised="http://collector.ours.invalid:8080",
        fetch=_fetcher(col), local={"collector.ours.invalid"})
    assert reach is CollectorReach.NAME_ANSWERS
    assert "203.0.113.9" in detail

    inv = _inv(capabilities={CALLBACK_CAPABILITY: True})
    pf = _pf(inv=inv, collector=col, advertised_callback="http://collector.ours.invalid:8080",
             collector_fetch=_fetcher(col), local_addresses={"collector.ours.invalid"})
    row = _row(pf.run(), "collector")
    assert row.status is Status.WARN
    assert "not" in row.hint and "sufficient" in row.hint


def test_the_probe_uses_a_token_it_minted_rather_than_inferring_from_silence():
    """`unknown_hits` exists because an arbitrary request to the port is not evidence
    for anything. A probe that counted any hit would be satisfied by a port scan."""
    with CallbackCollector() as col:
        probe_collector(col)
        assert col.tokens(), "it has to mint one"
        assert any(col.hits(t) for t in col.tokens())
        assert col.unknown_hits == 0, "and its own fetch must not read as noise"


def test_a_reachable_collector_is_still_unusable_without_the_signed_capability():
    """`collector.py` declares CALLBACK_CAPABILITY and says in its docstring that it
    is gated by it; nothing in the module reads the constant. And a corpus callback
    case emits only the `oob` need, which the selection satisfies from
    `have_collector` alone -- so the signed capability was never consulted on the
    corpus path at all, and a run could send callback payloads from a service the
    engagement never agreed we would stand up."""
    col = _FakeCollector()
    wired = {"advertised_callback": "http://collector.ours.invalid:8080",
             "collector_fetch": _fetcher(col),
             "local_addresses": {"collector.ours.invalid"}}

    unsigned = _pf(inv=_inv(capabilities={}), collector=col, **wired)
    unsigned.run()
    assert unsigned.reach is CollectorReach.NAME_ANSWERS, "reachable"
    assert not unsigned.collector_usable, "and still not authorised"

    signed = _pf(inv=_inv(capabilities={CALLBACK_CAPABILITY: True}),
                 collector=col, **wired)
    signed.run()
    assert signed.collector_usable


def test_the_probe_feeds_selection_rather_than_sitting_beside_it():
    """Two authoritative answers to one question drift. The old `--collector` boolean
    went straight into `applicable_chat_cases` unverified; the measurement has to
    REPLACE it, not sit next to it."""
    with CallbackCollector() as col:          # loopback: measurably unreachable
        pf = _pf(inv=_inv(capabilities={CALLBACK_CAPABILITY: True}), collector=col)
        pf.run()
        assert pf.reach is CollectorReach.LOOPBACK_ONLY
        assert pf.collector_usable is False
        with_it, _ = pf.select(have_collector=True)
        assert set(pf.runnable) <= set(with_it)


# --- what the one benign message contradicts -------------------------------------

def test_a_target_declared_stateful_that_is_not_is_a_failure():
    """`target_features` DECIDES WHICH CASES RUN, so a declaration the probe
    contradicts means the selection admitted cases on a false premise. Reported
    rather than acted on: the inventory is signed, and this command does not get to
    quietly re-select against it."""
    row = _row(_pf(sender=mock_sender("stateless")).run(), "statefulness")
    assert row.status is Status.FAIL
    assert "declared stateful" in row.detail
    assert "no conversation id" in row.detail


def test_a_target_that_is_not_declared_stateful_and_is_not_is_no_fault():
    inv = _inv(target_features=("tools",))
    row = _row(_pf(inv=inv, sender=mock_sender("stateless")).run(), "statefulness")
    assert row.status is Status.PASS


def test_tool_call_reporting_the_run_was_not_told_about_is_surfaced():
    """The probe saw the target report tool calls and the run is scoring without
    them, which costs cases for no reason. The old preflight could only hint that
    instrumentation was valuable; it had no way to say the run already had it."""
    row = _row(_pf(sender=mock_sender("instrumented")).run(), "tool visibility")
    assert row.status is Status.WARN
    assert "was not told" in row.detail
    assert "--tool-visibility" in row.hint


def test_an_uninstrumented_target_is_asked_for_instrumentation():
    row = _row(_pf(sender=mock_sender("nested")).run(), "tool visibility")
    assert row.status is Status.WARN
    assert "observed fact" in row.hint


# --- reaching them ---------------------------------------------------------------

def test_an_unauthorised_reply_names_what_was_sent():
    row = _row(_pf(sender=mock_sender("unauthorised")).run(), "auth")
    assert row.status is Status.FAIL
    assert "401" in row.detail
    assert "No auth header was sent" in row.hint


def test_a_two_hundred_with_no_reply_is_a_failure_not_a_pass():
    """The request worked and the parser found nothing, which is the shape that
    scores every case 0.00 against an assistant answering normally."""
    row = _row(_pf(sender=mock_sender("silent")).run(), "auth")
    assert row.status is Status.FAIL
    assert "no reply extracted" in row.detail
    assert "reply-path" in row.hint


def test_plain_http_against_a_production_assistant_is_a_warning():
    row = _row(_pf(inv=_inv(tls=False), tls=False).run(), "tls")
    assert row.status is Status.WARN
    assert "plain HTTP" in row.detail


def test_disabled_certificate_verification_has_to_reach_the_report():
    row = _row(_pf(verify=False).run(), "tls")
    assert row.status is Status.WARN
    assert "DISABLED" in row.detail
    assert "report" in row.hint


# --- the scope document ----------------------------------------------------------

def test_a_scope_document_round_trips(tmp_path):
    """The loader that did not exist. Every inventory in this package was hand-built
    Python, so the signed artifact that says what may be touched had no way to arrive
    from the place it is signed."""
    import yaml

    doc = tmp_path / "scope.yaml"
    doc.write_text(yaml.safe_dump({
        "project": "acme", "allowed_domains": ["chat.acme.invalid"],
        "allowed_tools": ["chat_say"], "max_target_messages": 100,
        "authorization": {"authorized_by": "ACME Ltd", "authorized_for": "sunflower",
                          "scope_summary": "s", "granted_on": "2026-01-01",
                          "reference": "SOW-1"}}))
    inv = load_inventory(doc)
    assert inv.project == "acme"
    assert inv.content_hash().startswith("sha256:")
    assert inv.tool_allowed("chat_say")


def test_a_scope_document_with_an_unknown_field_is_refused(tmp_path):
    """Because a typo in a scope document fails OPEN: `expires` instead of
    `expires_on` is an authorization that never expires."""
    doc = tmp_path / "scope.yaml"
    doc.write_text("project: acme\nexpires: 2020-01-01\n")
    with pytest.raises(ScopeError) as exc:
        load_inventory(doc)
    assert "fails OPEN" in str(exc.value)


def test_a_scope_document_that_is_not_a_mapping_is_refused(tmp_path):
    doc = tmp_path / "scope.yaml"
    doc.write_text("- a\n- b\n")
    with pytest.raises(ScopeError, match="must be a mapping"):
        load_inventory(doc)


def test_a_missing_scope_document_is_refused_by_name(tmp_path):
    with pytest.raises(ScopeError, match="cannot read"):
        load_inventory(tmp_path / "nope.yaml")


# --- the readout -----------------------------------------------------------------

def test_only_the_rows_that_need_acting_on_carry_a_hint():
    """A hint beside every passing check trains an operator to skip the whole block,
    which is how the one that mattered gets missed."""
    results = _pf(sender=mock_sender("stateless")).run()
    text = render(results)
    for row in results:
        if row.status is Status.PASS and row.hint:
            assert f"{row.name}: {row.hint}" not in text


def test_the_verdict_says_do_not_start_when_anything_failed():
    assert "do not start" in render(_pf(verified=False).run())
    assert "do not start" not in render(_pf(inv=_inv(
        capabilities={"plant_content": True, "multi_turn_escalation": True,
                      "capture_harmful_content": True, "modify_target": True,
                      CALLBACK_CAPABILITY: True})).run())


def test_a_failure_is_about_the_window_and_not_about_safety():
    """The gate handles safety, and a readout that blurred the two would teach an
    operator to read a scope failure as a technicality."""
    text = render(_pf(verified=False).run())
    assert "the gate handles safety" in text


# --- against the real corpus -----------------------------------------------------

@needs_corpus
def test_the_collector_is_worth_three_cases_and_not_thirty_four():
    """Measured, and the distinction is the whole reason to probe. 34 cases need the
    URL MINTED as an unguessable marker and nothing has to fetch it; 3 rest on a
    request actually arriving. Reporting the first number would overstate the cost of
    an unreachable collector by an order of magnitude -- in the direction this tool
    exists to stop other tools going."""
    from sunflower_stem.corpus import load_chat_cases, resolve_bindings

    inv = _inv(capabilities={"plant_content": True, "multi_turn_escalation": True,
                             CALLBACK_CAPABILITY: True},
               target_features=("tools", "rag", "memory", "multimodal", "files",
                                "web", "stateful"),
               seedable_channels=("document", "tool_output", "web_page", "email",
                                  "file_metadata", "calendar", "image", "audio"),
               seeds={"canary": "ACME-INTERNAL-4417"})
    # Nothing hand-bound: this test is about what a COLLECTOR changes, and the
    # unbound engagement facts are the same on both sides of the comparison.
    bindings, canaries = resolve_bindings(inventory_seeds=dict(inv.seeds))
    cases, corpus, _ = load_chat_cases(CORPUS, bindings=bindings, canaries=canaries,
                                       have_collector=True)
    pf = _pf(inv=inv, cases=cases, corpus_sha=corpus.sha)
    pf.runnable, pf.skipped = pf.select(have_collector=False)
    without = {c for c, s in pf.runnable.items() if s.partial}
    with_it, _ = pf.select(have_collector=True)
    promoted = without - {c for c, s in with_it.items() if s.partial}
    assert promoted, "the corpus must have cases a collector promotes"
    assert all(c.startswith("OUT-E-") for c in promoted), (
        f"a reachable collector should only promote exfiltration cases, got {promoted}")
    assert len(promoted) < 10, (
        "if this is near 34 the cost is being read off cases that merely need the "
        "URL minted, which nothing has to fetch")
