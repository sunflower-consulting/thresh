"""Nothing sensitive reaches a file this tool writes, and the canary still does.

TWO FAILURES, PULLING IN OPPOSITE DIRECTIONS.

Redact too little and the run log becomes a GDPR liability: a support bot's
replies are made of customer data, getting it to disclose some is the whole
point of a disclosure test, and the log is the artefact that gets handed to a
client's monitoring team to compare against. Redact too much and the seeded
canary is destroyed, which deletes the evidence for the strongest rung on the
ladder. The canary is high-entropy by construction, so it looks exactly like
the secrets the patterns are hunting.

So `keep` is a correctness requirement rather than a nicety, and both directions
are tested here.
"""
from __future__ import annotations

import json
import random

import pytest

from sunflower_thresh.redact import PATTERNS, REDACTION_VERSION, luhn_ok, make_redactor, redact
from sunflower_thresh.runlog import RunLog, read_events

#: Pinned deliberately. The version travels with a run so two runs stay
#: comparable, and bumping it silently makes an old run's redaction claim
#: unverifiable. Changing this number should require editing this test and
#: saying why in the same commit.
EXPECTED_REDACTION_VERSION = 2

#: What must never survive, by pattern name. 4111111111111111 is the standard
#: Visa test number and passes Luhn, which the card pattern requires.
SENSITIVE = {
    "email": "arne.borg@example.no",
    "secret": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdef",
    "api_key": "api_key=sk-live-9f3a91bc77de4411",
    "card": "4111111111111111",
}


def test_the_redaction_version_is_pinned():
    assert REDACTION_VERSION == EXPECTED_REDACTION_VERSION, (
        "the redaction policy changed. Bump EXPECTED_REDACTION_VERSION here and "
        "say in the commit what a reader of an older run should now understand "
        "differently, because the version is what makes two runs comparable")


def test_the_pattern_set_is_small_and_deliberate():
    """Patterns, not a classifier. A regex that is wrong about a Luhn check is a
    bug you can see; a model that is wrong about what counts as personal data is
    a bug you cannot. If this list grows, the argument has changed and the commit
    that grows it should say so."""
    assert [name for name, _, _ in PATTERNS] == [
        "email", "secret", "private_key", "phone"]


def test_a_test_card_passes_luhn_so_the_card_assertions_below_are_not_vacuous():
    """Guards the guard. The card pattern only fires on a Luhn-valid number, so
    a test using an invalid one would assert nothing and pass."""
    assert luhn_ok("4111111111111111")


# --- nothing sensitive reaches the log -------------------------------------------

@pytest.mark.parametrize("name,value", sorted(SENSITIVE.items()))
def test_a_sensitive_value_does_not_reach_the_log_file(tmp_path, name, value):
    log = RunLog("run-1", root=tmp_path, redactor=make_redactor())
    log.emit("http_request", message=f"sent {value}",
             args={"headers": {"Authorization": value}, "body": value})
    raw = (tmp_path / "run-1" / "events.jsonl").read_text()
    assert value not in raw, f"{name} survived into the log verbatim"


def test_a_sensitive_value_nested_arbitrarily_deep_still_does_not_reach_the_log(tmp_path):
    """The walk has to be recursive. A secret one level deeper than the test
    anybody wrote is the shape this fails in."""
    log = RunLog("run-2", root=tmp_path, redactor=make_redactor())
    log.emit("decision", args={"plan": [{"step": {"auth": SENSITIVE["secret"]}}]})
    raw = (tmp_path / "run-2" / "events.jsonl").read_text()
    assert SENSITIVE["secret"] not in raw


def test_the_redacted_log_line_is_still_valid_json(tmp_path):
    """Why the walk redacts values rather than the serialised line: a pattern
    applied to whole JSON can match across a field boundary and produce a line
    that no longer parses, which would lose the run record entirely."""
    log = RunLog("run-3", root=tmp_path, redactor=make_redactor())
    log.emit("http_request", message=f'quotes " braces }} {SENSITIVE["email"]}',
             args={"body": SENSITIVE["secret"]})
    for line in (tmp_path / "run-3" / "events.jsonl").read_text().splitlines():
        json.loads(line)
    assert len(read_events(tmp_path / "run-3" / "events.jsonl")) == 1


def test_a_randomised_sweep_of_secret_shapes_never_leaks(tmp_path):
    """Seeded rather than random so a failure is reproducible from the seed
    printed in the assertion, and broad enough to catch a pattern that only
    anchors at a string boundary."""
    rng = random.Random(20261016)
    log = RunLog("run-4", root=tmp_path, redactor=make_redactor())
    planted = []
    for i in range(60):
        kind = rng.choice(sorted(SENSITIVE))
        value = SENSITIVE[kind]
        filler = "".join(rng.choice("abcdef ,:/\n\t") for _ in range(rng.randint(0, 30)))
        # A delimiter before the value, because that is how a secret appears in
        # captured data: a header, a JSON field or a form parameter all put one
        # there. Gluing a secret to a word character is covered separately by
        # test_the_known_boundary_limits_are_recorded, which documents that it
        # is NOT redacted rather than pretending it is.
        sep = rng.choice([" ", "\n", '": "', "=", ": "])
        planted.append(value)
        log.emit("probe", message=f"{filler}{sep}{value}{filler}",
                 args={"turn": i, "payload": f"{filler}{sep}{value}"})
    raw = (tmp_path / "run-4" / "events.jsonl").read_text()
    leaked = sorted({v for v in planted if v in raw})
    assert not leaked, f"leaked with seed 20261016: {leaked}"


# --- and the canary survives -----------------------------------------------------

def test_a_seeded_canary_survives_redaction_in_the_log(tmp_path):
    """THE test the `keep` list exists for.

    The canary is a high-entropy token, so it looks like exactly what the secret
    pattern hunts. Redacting it would delete the evidence for the strongest rung
    on the ladder, which means the safety mechanism would be destroying the
    findings it was added to make safe to store.
    """
    canary = "STEM-CANARY-7f3a91bc77de4411"
    redactor = make_redactor(keep=(canary,))
    log = RunLog("run-5", root=tmp_path, redactor=redactor)
    log.emit("oracle_hit", message=f"the reply contained {canary}")
    raw = (tmp_path / "run-5" / "events.jsonl").read_text()
    assert canary in raw, "the canary was redacted; the finding's evidence is gone"


def test_a_canary_minted_mid_run_is_protected_in_the_log_too(tmp_path):
    """Cases mint canaries as they run, after the log was constructed. The
    redactor keeps its protected values in a closure precisely so one
    registration covers everything holding that redactor."""
    redactor = make_redactor()
    log = RunLog("run-6", root=tmp_path, redactor=redactor)
    late = "STEM-CANARY-deadbeefcafe1234"
    redactor.keep(late)
    log.emit("oracle_hit", message=f"leaked {late} and also {SENSITIVE['email']}")
    raw = (tmp_path / "run-6" / "events.jsonl").read_text()
    assert late in raw
    assert SENSITIVE["email"] not in raw


def test_keeping_a_canary_does_not_also_keep_a_real_secret(tmp_path):
    """`keep` is an exact-value allowlist, not a relaxation of the patterns."""
    canary = "STEM-CANARY-7f3a91bc77de4411"
    log = RunLog("run-7", root=tmp_path, redactor=make_redactor(keep=(canary,)))
    log.emit("oracle_hit", message=f"{canary} {SENSITIVE['secret']}")
    raw = (tmp_path / "run-7" / "events.jsonl").read_text()
    assert canary in raw and SENSITIVE["secret"] not in raw


# --- the default is a decision, stated -------------------------------------------

def test_without_a_redactor_the_log_writes_what_it_was_given(tmp_path):
    """Recorded so the default is a stated decision rather than a surprise.

    The library default is no redaction, matching `EvidenceStore`, because a
    redactor belongs with the provenance record that says which policy was in
    force. Every path that `thresh` itself drives constructs one; this is the
    escape hatch for a caller who has their own handling, and it is the reason
    `thresh doctor` reports the redaction policy rather than leaving it implied.
    """
    log = RunLog("run-8", root=tmp_path)
    log.emit("http_request", args={"body": SENSITIVE["email"]})
    assert SENSITIVE["email"] in (tmp_path / "run-8" / "events.jsonl").read_text()


def test_redact_reports_what_it_removed():
    """The counts are what lets a manifest say redaction actually ran, rather
    than that a redactor was configured."""
    _, counts = redact(f"{SENSITIVE['email']} and {SENSITIVE['secret']}")
    assert counts.get("email") == 1
    assert counts.get("secret") == 1


# --- the JSON regression, and the limits that remain -----------------------------

@pytest.mark.parametrize("body", [
    '{"api_key":"sk-live-9f3a91bc77de4411"}',       # no space: the original miss
    '{"token": "ghp_9f3a91bc77de4411xx"}',
    "{'password' => 'hunter2hunter2'}",             # ruby/php hash rocket
    'secret: "s3cr3t-value-here"',                  # yaml
    "api_key=sk-live-9f3a91bc77de4411",             # form encoded
    "PASSWORD : correcthorsebattery",               # spaced and upper
])
def test_a_credential_in_a_captured_body_is_redacted_whatever_the_syntax(body):
    """Regression. The first version of the secret pattern required the keyword
    and the separator to be adjacent, so `{"api_key":"..."}` did not match: JSON
    puts a closing quote between them. JSON is the commonest shape a captured
    HTTP body has, which made this the single most likely way for a live
    credential to reach the evidence store verbatim.
    """
    out, counts = redact(body)
    assert counts.get("secret"), f"not redacted: {body}"
    assert "[REDACTED:secret]" in out


@pytest.mark.parametrize("prose", [
    "the token is set",
    '{"token": "abc"}',          # too short to be a credential
    "my password is a secret",
    "tokenizer=fast",
    "password reset requested",
])
def test_widening_the_separator_did_not_start_matching_prose(prose):
    """The other half of the regression. A redactor that fires on the word
    "password" in a sentence destroys the replies this tool exists to read, and
    a disclosure finding is made of prose."""
    assert not redact(prose)[1].get("secret"), f"false positive on: {prose}"


def test_the_known_boundary_limits_are_recorded():
    """What redaction does NOT catch, written down on purpose.

    These are consequences of the patterns being anchored, and anchoring is what
    keeps them from redacting the ordinary text a finding is made of. A limit
    that is recorded is one a reader can account for; a limit that is discovered
    by a client reading a deliverable is a different kind of problem.

    If a future change makes any of these redact, that is an improvement, and
    this test should be updated in the same commit that earns it.
    """
    # a secret glued directly to a word character has no \b to anchor on
    assert not redact("xxxBearer eyJhbGciOiJIUzI1NiJ9abcdef")[1]
    # a digit run preceded by a dot is skipped, which is what stops version
    # strings and IP addresses being redacted as payment cards
    assert not redact("10.0.4111111111111111")[1]
    assert not redact("v.4111111111111111")[1]
    # and the deliberate consequence: a real card IS caught at a boundary
    assert redact("card 4111111111111111")[1].get("card") == 1
