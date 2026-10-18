"""What this installation can and cannot prove.

Written because every claim this tool makes is conditional on something: a rung
needs a listener bound, a p-value needs a control arm, the judge needs a
calibration figure that does not exist yet. A user who runs one command and
reads one screen should come away knowing which of those hold for them, rather
than discovering it from a finding they cannot defend.

WHY THIS FILE DISCOVERS RATHER THAN DECLARES.

It used to say the judge rung was "present but OFF and uncalibrated". That was a
false statement, in a public repository, in the one command the README tells
readers to trust over the README. `judge.py` did not exist, so nothing could pass
a judge to `build_ladder`, and `oracles._build_rung` returned `None` on every
path: the rung was UNREACHABLE, not disabled. "Present but off" describes a flag
you can turn on, and a reader who went looking for the flag found no module.

The repair is not better wording. A constant string cannot track whether a module
exists, so the next time the two drift the readout lies again. Everything below is
read off the installation at call time, and `tests/test_doctor.py` asserts the
readout agrees with what is actually importable -- which means deleting `judge.py`
turns the test red instead of making the readout wrong.

Nothing here opens a socket. `stem doctor` is what a reader runs BEFORE trusting
the tool, so it has to work on a machine with no network and no endpoint
configured, and it must never be the command that discovers a judge is down.
"""
from __future__ import annotations

import importlib.util
import platform
import sys

from sunflower_stem import __version__

#: The judge rung is off until its agreement with human reviewers is measured per
#: rubric and published. Stated here rather than buried in docs because shipping a
#: model-graded verdict as a default is how the rest of this field produces findings
#: a human then rejects: the rehearsal that measured ours had every rubric firing
#: too eagerly against a hardened assistant. The four rungs above it need no model.
#:
#: This is the POLICY. Whether the rung is even reachable is a separate question
#: that `judge_state` answers, and conflating the two is the bug this file had.
JUDGE_DEFAULT = False

#: The three states the judge rung can really be in, with what each means for a
#: reader deciding whether to trust a judge-scored row.
JUDGE_ABSENT = "absent"
JUDGE_PRESENT = "present"
JUDGE_CONFIGURED = "configured"

_JUDGE_REASONS = {
    JUDGE_ABSENT: ("NOT INSTALLED: this build has no judge, so the rung is "
                   "unreachable rather than switched off. Nothing can score a "
                   "case that needs it, and such a case reports as not probed. "
                   "The four rungs above it need no model."),
    JUDGE_PRESENT: ("installed, and OFF because no judge endpoint is configured. "
                    "The rung is reachable once one is, and stays off until its "
                    "agreement with human reviewers is published per rubric."),
    JUDGE_CONFIGURED: ("installed and configured, and still OFF: "
                       "judge-versus-human agreement has not been measured per "
                       "rubric, so a verdict from it is not defensible yet. The "
                       "rungs above it need no model."),
}


def judge_state() -> str:
    """Which of the three states this installation is in, read at call time.

    `find_spec` rather than an import: asking whether a module is importable must
    not run it, and this is also the check the test pins, so the readout and the
    test cannot disagree about what "present" means.
    """
    try:
        spec = importlib.util.find_spec("sunflower_stem.judge")
    except (ImportError, ValueError):  # a half-installed package is not a judge
        return JUDGE_ABSENT
    if spec is None:
        return JUDGE_ABSENT
    from sunflower_stem import judge

    configured = getattr(judge, "endpoint_configured", None)
    if callable(configured) and configured():
        return JUDGE_CONFIGURED
    return JUDGE_PRESENT


def judge_reason(state: str | None = None) -> str:
    """The sentence the readout prints for a state, and what the tests assert on."""
    return _JUDGE_REASONS[state or judge_state()]


#: Rungs whose availability is a fact about an engagement rather than about this
#: installation. The machinery for all three ships; whether YOUR run can reach them
#: depends on a canary the client holds, a collector the target can reach, and an
#: agent that reports its own tool calls. `stem doctor` opens no socket and reads no
#: scope document, so it must not answer for any of them.
ENGAGEMENT_RUNGS: frozenset[str] = frozenset({"CANARY", "CALLBACK", "TOOL_CALL"})


def _rungs(state: str) -> list[str]:
    """The ladder, strongest first, each with what it needs to be reachable.

    Reported because a rung that cannot fire is the difference between a clean
    result and an untested one, and a reader who does not know which rungs were
    available cannot tell those apart from the output.
    """
    from sunflower_stem.schemas.finding import EvidenceStrength

    judge_needs = {
        JUDGE_ABSENT: "a judge, which this build does not have",
        JUDGE_PRESENT: "a judge endpoint, and a calibration figure that does not exist yet",
        JUDGE_CONFIGURED: "a calibration figure that does not exist yet",
    }[state]
    needs = {
        "CANARY": "a seeded token the target should not be able to produce",
        "CALLBACK": "a listener the target can reach; proof is its own request",
        "TOOL_CALL": "a target that reports the tools it invoked",
        "STRUCTURAL": "nothing: a deterministic matcher",
        "JUDGE": judge_needs,
    }
    out = []
    for s in sorted(EvidenceStrength, key=lambda s: -s.value):
        if s.name not in needs:
            continue
        if s.name in ENGAGEMENT_RUNGS:
            # Not "available". Read the `needs` column: a seeded token, a listener
            # the target can reach, a target that reports its own tool calls. Every
            # one of those is a fact about an ENGAGEMENT, and this command cannot
            # open a socket or read a scope document, so it cannot know any of them.
            #
            # It said "available" for all three, which is the defect this file was
            # rewritten to fix, one rung over: a constant string making a
            # conditional claim unconditionally, in the command the README tells
            # readers to believe over the README. `stem preflight` is what measures
            # them, and for the callback rung it had to be written before the answer
            # existed at all.
            status = "per engagement"
        elif s.name != "JUDGE":
            status = "available"
        elif state == JUDGE_ABSENT:
            # Not "off". Off is a decision; this is an absence, and a reader who
            # reads "off" goes looking for the switch.
            status = "absent"
        else:
            status = "off"
        out.append(f"  {s.value}  {s.name:<11} {status:<14} needs {needs[s.name]}")
    out.append("")
    out.append("  `stem preflight` measures the three marked per engagement, against "
               "one engagement's")
    out.append("  scope document and its target. This command cannot: it opens no "
               "socket.")
    return out


def report() -> str:
    """A plain-text readout. No colour: this gets pasted into tickets."""
    from sunflower_stem.redact import REDACTION_VERSION

    state = judge_state()
    # Both of these were unconditional, and both stop being true the moment a judge
    # endpoint is configured. They are claims about the DEFAULT path, and the
    # qualifier is what keeps them true in every state rather than most of them.
    if state == JUDGE_CONFIGURED:
        keys = ("API key required:      not for the default path. A judge endpoint "
                "may need one; stem ships no target that does.")
        net = ("Network required:      not for the default path, whose target and "
               "callback listener both bind loopback. The configured judge "
               "endpoint is reached over the network.")
    else:
        keys = "API key required:      no. stem ships its own target; see `stem lab`."
        net = ("Network required:      no. The bundled target and the callback "
               "listener both bind loopback.")

    lines = [
        f"stem {__version__}",
        f"python {platform.python_version()} on {sys.platform}",
        "",
        keys,
        net,
        (f"Redaction policy:      version {REDACTION_VERSION}, applied to the "
         f"run log and the evidence store."),
        "",
        "Evidence ladder, strongest first:",
        *_rungs(state),
        "",
        f"judge rung:            {judge_reason(state)}",
    ]
    return "\n".join(lines)
