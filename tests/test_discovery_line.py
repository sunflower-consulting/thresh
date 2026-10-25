"""The line between searching for an attack and measuring one.

WHY THERE IS A LINE AT ALL.

An adaptive attacker is the capability thresh does not have and the one the
comparable tools lead on, so it is going to be built. The hazard is not that it
attacks: it is that it attacks DIFFERENTLY each turn. Ten trials of text that
changes are not ten samples of one thing, and the control arm -- the benign twin
that is the same request with only the technique removed -- has nothing fixed to be
a twin of. The rate and the p-value stop meaning anything and go on printing,
which is the one failure this tool exists to prevent in others.

So a discovery run's output is a candidate case file with its turns and no rate, a
person writes its control arm, and the measurement path refuses anything else.

WHAT THE GUARD IS WORTH, PRECISELY.

Within one process it stops a careless line, not a determined one: the marker is a
reassignable attribute and nothing can stop code reassigning it. What it does close
is the route a human actually takes, a file on disk, and the tests below pin both
halves -- that a case file cannot claim authorship, and that a case marked in code
cannot be measured.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sunflower_thresh.tools.chat import (
    AUTHORED,
    BUILTIN_CHAT_CASES,
    GENERATED,
    ChatCase,
    ChatTurn,
)
from sunflower_thresh.validation_chat import validate_chat_case


def _case() -> ChatCase:
    return ChatCase(id="candidate", turns=[ChatTurn(content="hi")],
                    control_turns=[ChatTurn(content="hi")],
                    oracles=[{"structural": {"regex": "x"}}])


def test_a_case_file_cannot_claim_its_turns_were_authored():
    """The guard's first half, and the one that matters most, because a generated
    draft promoted by editing one line is the realistic way model-written turns
    reach a measurement. The origin is not a model field, so a file that names it
    is rejected rather than believed -- and rather than silently ignored, which is
    what pydantic does by default and what `extra="forbid"` is here to stop."""
    with pytest.raises(ValidationError):
        ChatCase(id="x", turns_origin=AUTHORED)
    with pytest.raises(ValidationError):
        ChatCase(id="x", turns_origin=GENERATED)


def test_an_unknown_field_in_a_case_file_is_rejected_rather_than_ignored():
    """Fell out of the same change and is worth its own line. A mistyped field name
    used to read as a case that simply did not set it, so `evalaute: final` left the
    case on the default and nothing said so."""
    with pytest.raises(ValidationError):
        ChatCase(id="x", evalaute="final")


def test_the_measurement_path_refuses_generated_turns(monkeypatch):
    """The guard's second half. Enforced at `validate_chat_case` because that is
    the single entry point to measurement, and before any target message is spent,
    because a refusal that costs messages is a refusal with a side effect."""
    sent = []

    def explode(*_a, **_k):
        sent.append(1)
        raise AssertionError("a generated case reached the target")

    monkeypatch.setattr("sunflower_thresh.validation_chat._ChatRunner", explode)
    finding, report = validate_chat_case(None, _case().as_generated(),
                                         target_ip="127.0.0.1")
    assert finding is None
    assert report["turns_origin"] == GENERATED
    assert not sent, "the refusal must cost no target messages"


def test_the_refusal_says_why_in_words_an_author_can_act_on():
    """A refusal a reader cannot act on gets worked around. This one has to say what
    to do next: freeze the turns and have a person write the control arm."""
    _finding, report = validate_chat_case(None, _case().as_generated(),
                                          target_ip="127.0.0.1")
    why = report["skipped"]
    assert "not authored" in why
    assert "control arm" in why


def test_the_refusal_arrives_as_a_skip_and_not_as_an_error():
    """`sweep._validate_one` wraps this call in a blanket `except Exception` and
    turns anything raised into a generic error row, which is indistinguishable from
    a timeout or a dead endpoint. So the guard reports through `skipped`, the
    channel that already means "nothing was measured, and here is why"."""
    finding, report = validate_chat_case(None, _case().as_generated(),
                                         target_ip="127.0.0.1")
    assert finding is None
    assert report["skipped"]
    assert "error" not in report


def test_marking_a_case_generated_does_not_mark_the_one_it_came_from():
    """A discovery loop draws an exemplar from the library. If marking the candidate
    mutated the exemplar, the first discovery run would quietly make the library
    unmeasurable."""
    original = _case()
    generated = original.as_generated()
    assert original.turns_origin == AUTHORED
    assert generated.turns_origin == GENERATED
    assert generated is not original


def test_every_built_in_case_is_authored():
    """They are written out longhand in the source, which is what authored means.
    If one of these ever reads as generated, the default has moved."""
    assert BUILTIN_CHAT_CASES
    for cid, case in BUILTIN_CHAT_CASES.items():
        assert case.turns_origin == AUTHORED, cid


def test_an_authored_case_still_reaches_the_runner(monkeypatch):
    """The negative control, and the assertion that would catch a default set the
    wrong way round. A guard that refuses everything is not a guard.

    Asserted by watching the runner get CONSTRUCTED rather than by reading the
    report, because the refusal above is proved by the runner never being reached,
    and the mirror of that claim is the only honest negative."""
    built = []

    class _Watch:
        def __init__(self, *_a, **_k):
            built.append(1)
            raise RuntimeError("far enough: the origin check let this through")

    monkeypatch.setattr("sunflower_thresh.validation_chat._ChatRunner", _Watch)
    with pytest.raises(RuntimeError, match="far enough"):
        validate_chat_case(None, _case(), target_ip="127.0.0.1")
    assert built == [1]
