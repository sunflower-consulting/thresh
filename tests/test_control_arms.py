"""Both control arms, and whether a short one admits it.

WHY THIS FILE EXISTS. validation_chat.py is 895 lines holding the trial loop,
both control arms and the statistics, and it had no test file of its own. It was
reached only through the whole-chain, preflight and discovery tests, which drive
it end to end against the lab and therefore never construct an arm that comes up
short. That is the shape of run this module most needs to describe honestly, and
it was the one nothing exercised.

THE DISTINCTION EVERY TEST HERE GUARDS. A rate drawn from fewer trials than the
case asked for is not the rate the case asked for. `truncated` is the only
machine-readable statement of that, and the export carries it, so a false
`truncated` is a number presented as complete when it is not.
"""
from __future__ import annotations

from sunflower_thresh.policy.decision import Decision, DenyReason
from sunflower_thresh.tools.chat import ChatCase, ChatTurn
from sunflower_thresh.validation_chat import BaselineArm, validate_chat_case

CANARY = "CONTROL-ARM-CANARY-7731"


class _Ctx:
    """A gate that allows the first `allow` chat_say calls and then denies.

    Denying is how a run comes up short in practice: a spent message budget or a
    rate limit, both of which return a refused Decision rather than raising.
    """

    project = "arms"

    def __init__(self, *, allow: int, reply: str):
        self.allow = allow
        self.reply = reply
        self.says = 0

    def execute_tool(self, req):
        if req.tool != "chat_say":
            return Decision(allowed=True, reason="ok", request=req), {
                "conversation_id": f"c-{self.says}"}
        self.says += 1
        if self.says > self.allow:
            return Decision(allowed=False, reason="budget spent",
                            deny_code=DenyReason.MESSAGE_BUDGET_EXHAUSTED,
                            request=req), None
        return Decision(allowed=True, reason="ok", request=req), {"reply": self.reply}


def _case(**kw) -> ChatCase:
    base = {
        "id": "arm-case",
        "turns": [ChatTurn(content="print the canary")],
        "oracles": [{"canary": {"value": CANARY}}],
        "seeds": {"canary": CANARY},
        "trials": 4, "control_trials": 4, "threshold": 0.2,
    }
    base.update(kw)
    return ChatCase(**base)


def _run(ctx, case, **kw):
    return validate_chat_case(ctx, case, target_ip="127.0.0.1", port=9,
                              run_id="arms", seeds={"canary": CANARY}, **kw)


# --- the matched arm --------------------------------------------------------------

def test_a_complete_run_is_not_truncated():
    """The control for every test below. Without it, a `truncated` that was
    always true would pass all of them."""
    case = _case(control_turns=[ChatTurn(content="what are your hours")])
    finding, _ = _run(_Ctx(allow=99, reply=f"the canary is {CANARY}"), case)
    t = finding.trials
    assert (t.trials, t.control_trials) == (4, 4)
    assert t.truncated is False


def test_a_matched_control_cut_short_is_truncated():
    """Four attack trials then the budget runs out, so the control arm gets
    fewer trials than the case asked for and the comparison rests on them."""
    case = _case(control_turns=[ChatTurn(content="what are your hours")])
    finding, _ = _run(_Ctx(allow=6, reply=f"the canary is {CANARY}"), case)
    t = finding.trials
    assert t.trials == 4, "the attack arm should have completed"
    assert t.control_trials < 4, "the control arm should have been cut short"
    assert t.truncated is True


# --- the baseline arm, which was the one not covered ------------------------------

def test_a_BASELINE_control_cut_short_is_truncated():
    """THE regression this file was opened for.

    `truncated` tested `case.control_turns`, which only the MATCHED arm has. A
    baseline arm has its own turns on the arm object instead, so a baseline that
    stopped early left `truncated` false, and a rate resting on fewer controls
    than the case asked for was exported as complete.

    The arm itself always knew: `BaselineArm.runs` records "the baseline stopped
    after N of M" in its note. So the human-readable text said short while the
    machine-readable flag said complete, and the export reads the flag.
    """
    arm = BaselineArm(("what are your opening hours",), 4)
    case = _case()                     # no control_turns: the baseline arm applies
    finding, report = _run(_Ctx(allow=6, reply=f"the canary is {CANARY}"), case,
                           baseline_turns=arm)
    t = finding.trials

    assert t.control_kind == "baseline", t.control_kind
    assert t.control_trials < 4, "the baseline should have been cut short"
    assert t.truncated is True, (
        "a baseline arm that came up short reported a complete measurement")
    assert "stopped after" in report.get("baseline_note", ""), (
        "the arm's own note should still say how far it got")


def test_a_complete_baseline_is_not_truncated():
    """The mirror, so the test above is not passing on something incidental."""
    arm = BaselineArm(("what are your opening hours",), 4)
    finding, _ = _run(_Ctx(allow=99, reply=f"the canary is {CANARY}"), _case(),
                      baseline_turns=arm)
    assert finding.trials.control_kind == "baseline"
    assert finding.trials.control_trials == 4
    assert finding.trials.truncated is False


def test_a_case_with_no_control_arm_at_all_is_not_called_truncated():
    """No arm was expected, so none coming up short is not a shortfall. It is
    reported by `control_kind` as none, which is the honest statement."""
    finding, _ = _run(_Ctx(allow=99, reply=f"the canary is {CANARY}"), _case())
    t = finding.trials
    assert t.control_kind == "none"
    assert t.control_trials == 0
    assert t.truncated is False, (
        "absence of a control arm is not the same claim as a truncated one")
