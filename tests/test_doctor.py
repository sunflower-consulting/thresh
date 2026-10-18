"""`stem doctor` is the authoritative statement of what this install can prove.

Tested because it is the one output a reader is told to trust over the README,
and because its claims are the claims: if it says no key is needed while the
default path needs one, the tool is lying in the place it promised not to.

WHY THESE TESTS CHANGED SHAPE.

The three tests that used to be here asserted fixed strings -- that the readout
contained "no API key", and the word "uncalibrated", and that `JUDGE_DEFAULT` was
False. All three passed while the readout said the judge rung was "present but
OFF and uncalibrated" and `judge.py` did not exist. They pinned the wording and
were blind to the only thing that made the wording false.

So the tests below pin the readout against the INSTALLATION instead. Delete
`judge.py` and `test_the_readout_cannot_claim_a_rung_whose_module_is_absent`
turns red, which is the property the old tests could not have.
"""
from __future__ import annotations

import importlib.util

import pytest

from sunflower_stem import doctor
from sunflower_stem.doctor import (
    JUDGE_ABSENT,
    JUDGE_CONFIGURED,
    JUDGE_DEFAULT,
    JUDGE_PRESENT,
    judge_reason,
    judge_state,
    report,
)

ALL_STATES = (JUDGE_ABSENT, JUDGE_PRESENT, JUDGE_CONFIGURED)


def test_the_readout_cannot_claim_a_rung_whose_module_is_absent():
    """THE test in this file, and the one the old ones could not be.

    The readout described the judge rung as "present but OFF" for as long as
    `judge.py` did not exist. "Present but off" is a flag you can turn on; an
    absent module is a rung nothing can reach, and the two are different facts
    about what a reader can do next. This asserts the readout and the import
    system agree, in whichever direction they currently point, so it fails if
    either moves without the other."""
    installed = importlib.util.find_spec("sunflower_stem.judge") is not None
    state, out = judge_state(), report()

    if installed:
        assert state in (JUDGE_PRESENT, JUDGE_CONFIGURED)
        assert "NOT INSTALLED" not in out
    else:
        assert state == JUDGE_ABSENT
        assert "NOT INSTALLED" in out
        # The word that made the old message false. It must not come back while
        # there is no module to be present.
        assert "present but" not in out.lower()


def test_the_ladder_row_for_an_absent_judge_does_not_read_as_a_switch(monkeypatch):
    """`off` sends a reader looking for the switch. There isn't one when the
    module is missing, so the row has to say so in the status column and not only
    in the paragraph underneath."""
    monkeypatch.setattr(doctor, "judge_state", lambda: JUDGE_ABSENT)
    row = next(ln for ln in report().splitlines() if "JUDGE" in ln)
    assert "absent" in row
    assert "off" not in row


def test_the_judge_rung_is_off_by_default():
    """The POLICY, kept separate from whether the rung is reachable. Conflating
    the two is what let the readout call an absent module "off". Shipping a
    model-graded verdict on by default is how this field produces findings a
    human then rejects, and ours has never been measured against human agreement
    per rubric."""
    assert JUDGE_DEFAULT is False


@pytest.mark.parametrize("state", ALL_STATES)
def test_no_state_reports_the_rung_as_usable(monkeypatch, state):
    """Whatever is installed, the verdict is off. A reader must not be able to
    reach a state where the readout implies a judge-scored row is defensible,
    because the agreement figure that would make it so does not exist in any of
    them."""
    monkeypatch.setattr(doctor, "judge_state", lambda: state)
    row = next(ln for ln in report().splitlines() if "JUDGE" in ln)
    assert "available" not in row


@pytest.mark.parametrize("state", ALL_STATES)
def test_every_state_explains_itself_rather_than_only_disabling(monkeypatch, state):
    """A capability that is off without a reason reads as unfinished. Each state
    is off for a DIFFERENT reason and the readout has to carry the right one, or
    a reader cannot tell "install something" from "measure something"."""
    monkeypatch.setattr(doctor, "judge_state", lambda: state)
    out = report()
    assert judge_reason(state) in out
    assert len(judge_reason(state)) > 60, "a reason that short is a label"


def test_the_three_states_do_not_say_the_same_thing():
    """If two states print the same sentence, one of them is wrong, and the
    readout has stopped distinguishing a missing module from an unmeasured
    rubric."""
    assert len({judge_reason(s) for s in ALL_STATES}) == len(ALL_STATES)


def test_the_readout_claims_no_api_key_only_while_that_is_true(monkeypatch):
    """The claim was unconditional and stops being true the moment a judge
    endpoint is configured. It is a statement about the DEFAULT path, and the
    qualifier is what keeps it honest in every state rather than most."""
    monkeypatch.setattr(doctor, "judge_state", lambda: JUDGE_PRESENT)
    assert "API key required:      no." in report()

    monkeypatch.setattr(doctor, "judge_state", lambda: JUDGE_CONFIGURED)
    out = report()
    assert "API key required:      no." not in out
    assert "not for the default path" in out


def test_the_readout_claims_no_network_only_while_that_is_true(monkeypatch):
    """Same failure as the key claim, and worse: a reader on an air-gapped box
    takes "no network required" as permission to expect the whole tool to work."""
    monkeypatch.setattr(doctor, "judge_state", lambda: JUDGE_PRESENT)
    assert "Network required:      no." in report()

    monkeypatch.setattr(doctor, "judge_state", lambda: JUDGE_CONFIGURED)
    out = report()
    assert "Network required:      no." not in out
    assert "reached over the network" in out


def test_the_readout_names_every_rung_and_what_it_needs():
    """A rung that cannot fire is the difference between a clean result and an
    untested one. A reader who cannot see which rungs were reachable cannot
    tell those apart from the output."""
    out = report()
    for rung in ("CANARY", "CALLBACK", "TOOL_CALL", "STRUCTURAL", "JUDGE"):
        assert rung in out, f"{rung} missing from the readout"
    assert "needs a seeded token" in out


def test_the_readout_states_the_redaction_version():
    """Which policy was in force is what makes two runs comparable, so it is a
    fact about the installation rather than a detail of a run."""
    from sunflower_stem.redact import REDACTION_VERSION

    assert f"version {REDACTION_VERSION}" in report()


def test_the_readout_is_plain_text():
    """It gets pasted into tickets and issue trackers. No escape codes."""
    assert "\x1b[" not in report()


def test_the_readout_opens_no_socket(monkeypatch):
    """`stem doctor` is what a reader runs BEFORE trusting the tool, so it has to
    work with no network at all. It must never be the command that discovers the
    judge endpoint is down -- that is a finding about the endpoint, reported by
    whatever actually calls it."""
    import socket

    def refuse(*_a, **_k):
        raise AssertionError("stem doctor opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    report()


@pytest.mark.parametrize("rung", sorted(doctor.ENGAGEMENT_RUNGS))
def test_a_rung_whose_availability_is_an_engagement_fact_is_not_called_available(rung):
    """The defect this file's module was rewritten to fix, one rung over.

    `status` was the constant "available" for every rung but JUDGE, while the same
    line said the rung needs a seeded token, or a listener the target can reach, or
    an agent that reports its own tool calls. Those are facts about an engagement,
    and this command opens no socket and reads no scope document, so it cannot know
    any of them -- and it is the command the README tells a reader to believe over
    the README.
    """
    row = next(ln for ln in report().splitlines() if rung in ln)
    assert "available" not in row, f"{rung} cannot be claimed by this command"
    assert "per engagement" in row
    assert "stem preflight" in report(), "and it has to say what does measure them"


def test_the_one_rung_that_needs_nothing_is_still_called_available():
    """STRUCTURAL needs a deterministic matcher and nothing else, so hedging it
    would be the opposite error: a readout that qualifies everything says nothing."""
    row = next(ln for ln in report().splitlines() if "STRUCTURAL" in ln)
    assert "available" in row
