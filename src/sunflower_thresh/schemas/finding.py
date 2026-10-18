"""Canonical finding object + its state machine.

A finding earns its way to CONFIRMED only through reproduction, validation, and a
human gate. Model confidence is never proof. Transitions are enforced here, not
left to prose.
"""
from __future__ import annotations

import enum
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class FindingState(str, enum.Enum):
    OBSERVED = "observed"
    HYPOTHESIS = "hypothesis"
    UNDER_TEST = "under_test"
    REPRODUCED = "reproduced"
    VALIDATED = "validated"
    HUMAN_REVIEW = "human_review"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


# Allowed transitions. Anything not listed is refused by `can_transition`.
_TRANSITIONS: dict[FindingState, set[FindingState]] = {
    FindingState.OBSERVED: {FindingState.HYPOTHESIS, FindingState.REJECTED, FindingState.DUPLICATE},
    FindingState.HYPOTHESIS: {FindingState.UNDER_TEST, FindingState.REJECTED, FindingState.DUPLICATE},
    FindingState.UNDER_TEST: {FindingState.REPRODUCED, FindingState.REJECTED, FindingState.DUPLICATE},
    # HUMAN_REVIEW is reachable from REPRODUCED, not only from VALIDATED, because
    # of the evidence cap below: a finding whose only oracle is an LLM judge stops
    # at REPRODUCED by construction. Without this edge such a finding is a dead
    # end -- it can never be escalated, only rejected -- which would mean the
    # harness could observe a real jailbreak and have nowhere to put it.
    FindingState.REPRODUCED: {FindingState.VALIDATED, FindingState.REJECTED,
                              FindingState.DUPLICATE, FindingState.HUMAN_REVIEW},
    FindingState.VALIDATED: {FindingState.HUMAN_REVIEW, FindingState.REJECTED},
    # Only a human gate moves HUMAN_REVIEW -> CONFIRMED (enforced by caller flag).
    FindingState.HUMAN_REVIEW: {FindingState.CONFIRMED, FindingState.REJECTED},
    FindingState.CONFIRMED: set(),
    FindingState.REJECTED: set(),
    FindingState.DUPLICATE: set(),
}

TERMINAL = {FindingState.CONFIRMED, FindingState.REJECTED, FindingState.DUPLICATE}

#: How far along the evidence chain each state is. Used by the evidence cap in
#: `Finding.transition`; the terminal verdicts are not on the chain and are
#: exempted there rather than ranked here.
_RANK: dict[FindingState, int] = {
    FindingState.OBSERVED: 0,
    FindingState.HYPOTHESIS: 1,
    FindingState.UNDER_TEST: 2,
    FindingState.REPRODUCED: 3,
    FindingState.VALIDATED: 4,
    FindingState.HUMAN_REVIEW: 5,
    FindingState.CONFIRMED: 6,
}


def can_transition(src: FindingState, dst: FindingState) -> bool:
    return dst in _TRANSITIONS.get(src, set())


class IllegalTransition(Exception):
    pass


class EvidenceStrength(int, enum.Enum):
    """How good the proof is, as an ordered scale.

    A web finding is proved by a deterministic request/response pair. A chatbot
    finding is not: the target is stochastic, and the strongest thing available is
    often an unguessable token that the system could not have invented. So the
    KIND of evidence has to travel with the finding, because it decides how far
    the finding may go -- see MAX_STATE_FOR and `Finding.transition`.

    Ordered deliberately (an IntEnum, so `>=` works): a ladder's job is to report
    the STRONGEST rung that fired, not the first.

    - CANARY     an unguessable seeded token appeared. The system could not have
                 guessed it, so it read something it should not have.
    - CALLBACK   the target made an out-of-band request we recorded. Its own
                 traffic, not its prose -- the only real proof of exfiltration.
    - TOOL_CALL  the agent actually invoked a tool. Observable fact, available
                 when a client instruments their agent; the strongest signal for
                 excessive agency, and the one to push clients toward.
    - STRUCTURAL a deterministic matcher fired: a regex, a schema, a *correct*
                 sha256. Echo is not execution, so this rung is only as good as
                 the matcher discriminates.
    - JUDGE      a different model scored the reply against a versioned rubric.
                 Probabilistic, so it is capped below VALIDATED: the harness's
                 claim is that its verdicts are falsifiable, and a model's
                 opinion is not a falsification.
    """

    NONE = 0
    JUDGE = 1
    STRUCTURAL = 2
    TOOL_CALL = 3
    CALLBACK = 4
    CANARY = 5


class VersionKind(str, enum.Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


#: The furthest a finding may travel on each kind of evidence alone.
#:
#: A table rather than prose in a docstring, because it is a RULE and
#: `Finding.transition` enforces it. Everything deterministic may reach VALIDATED;
#: a judge's verdict stops at REPRODUCED and needs the human gate to go further.
#: NONE is deliberately ABSENT, and that is not an oversight. It is the default on
#: every Finding, including the whole web path, which proves itself with a
#: deterministic request/response pair and never consults this ladder. NONE
#: therefore means "no ladder rung was asserted", not "no evidence" -- so it caps
#: nothing. When a chat oracle genuinely fails to fire, the validator transitions
#: to REJECTED itself and says why; inferring that from an unset default would
#: silently reject every finding the rest of the harness makes.
MAX_STATE_FOR: dict[EvidenceStrength, FindingState] = {
    EvidenceStrength.CANARY: FindingState.VALIDATED,
    EvidenceStrength.CALLBACK: FindingState.VALIDATED,
    EvidenceStrength.TOOL_CALL: FindingState.VALIDATED,
    EvidenceStrength.STRUCTURAL: FindingState.VALIDATED,
    EvidenceStrength.JUDGE: FindingState.REPRODUCED,
}


class TrialStats(BaseModel):
    """What n attempts against a stochastic target actually showed.

    A web reproduction test either works or it does not, so "reproduced twice,
    control clean" is a complete answer. A chatbot is not deterministic: a
    jailbreak that lands two times in five is a REAL finding, and the old
    criterion (`all(repro)` over exactly two runs) rejected it while also calling
    a lucky 2/2 proof. Neither is honest.

    So a chat finding carries its arithmetic: how many times it was tried, how
    often it landed, the same for the benign control, and an interval -- because a
    point estimate from five trials invites the reader to over-read it.
    """

    trials: int = 0
    hits: int = 0
    control_trials: int = 0
    control_hits: int = 0
    asr: float = 0.0
    control_asr: float = 0.0
    #: Wilson score interval on `asr`. Normal-approximation intervals go outside
    #: [0,1] and are useless at the small n a per-case budget allows.
    asr_ci: tuple[float, float] = (0.0, 0.0)
    #: One-sided Fisher exact test of hits/trials against control_hits/control_trials.
    #: This is what makes "2/5 vs 0/5" a different claim from "2/5 vs 2/5".
    p_value: float = 1.0
    threshold: float = 0.0
    #: Set when the run ran out of target messages (or the target died) before
    #: MIN_TRIALS completed. A truncated case is UNTESTED, never "did not reproduce".
    truncated: bool = False
    #: Which kind of control arm ran, because the two support different claims and a
    #: report that calls them both "control" overstates one of them.
    #:
    #: "matched"  -- the SAME turns with only the attack clause removed. Isolates the
    #:               attack: a hit here and not there means the attack did it.
    #: "baseline" -- ordinary customer questions for this target, run once per
    #:               engagement and reused. Shows the signal is not ambient in normal
    #:               traffic. It does NOT isolate the attack, so for an injection case
    #:               it cannot distinguish "the injection worked" from "the bot is
    #:               simply unprotected" -- the report has to say which it is.
    #: "none"     -- nothing was falsified.
    #:
    #: The distinction exists because the corpus carries no matched controls (see
    #: docs/corpus-feedback.md), and without SOME control arm no corpus case could ever
    #: reach VALIDATED -- which would make every number in a report a 0.
    control_kind: str = "none"


class Finding(BaseModel):
    id: str
    title: str = ""
    state: FindingState = FindingState.OBSERVED
    target: str = ""
    asset: str = ""
    vendor: str = ""
    product: str = ""
    version: str = ""
    version_kind: VersionKind = VersionKind.UNKNOWN
    cpe: str = ""
    cwe: str = ""
    existing_cves: list[str] = Field(default_factory=list)
    description: str = ""
    attack_vector: str = ""
    preconditions: list[str] = Field(default_factory=list)
    impact: str = ""
    # Evidence is referenced by content hash, never inlined blobs.
    evidence_refs: list[str] = Field(default_factory=list)
    reproduction_steps: list[str] = Field(default_factory=list)
    test_cases: list[str] = Field(default_factory=list)
    crash_signatures: list[str] = Field(default_factory=list)
    agent_confidence: float = 0.0
    validation_result: str = ""
    review_status: str = "pending"
    #: Which rung of the oracle ladder produced this finding. It caps how far the
    #: finding may travel -- see MAX_STATE_FOR and `transition`.
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    #: Which oracle fired and on what: the rubric id and content hash and judge
    #: model for a JUDGE verdict, the token for a CALLBACK, the tool name for a
    #: TOOL_CALL. A verdict nobody can trace back to its rubric is not evidence.
    oracle_detail: dict = Field(default_factory=dict)
    #: The case's oracles this run could NOT score, each with what it lacked:
    #: [{"oracle": "write_tool_called", "missing": ["tool_visibility"]}].
    #:
    #: Reachability is decided per oracle, so a case carrying a prose regex and a
    #: `tool_called` rubric still runs against a black-box deployment -- on the regex.
    #: That is strictly better than skipping it, and it is only honest if the finding
    #: says so: "0 of 10 trials" scored on one of two oracles is a weaker statement
    #: than the same number scored on both, and a reader cannot tell them apart from
    #: the rate. This field is what makes them different, which is why it is on the
    #: finding rather than in the run log -- a REJECTED finding is exactly the one
    #: that travels into a report as "the target resisted".
    unscored_oracles: list[dict] = Field(default_factory=list)
    #: Present on findings against a stochastic target (chatbots/agents).
    trials: TrialStats | None = None
    #: The test case's own severity HINT, carried from the corpus and never computed on.
    #:
    #: A hint, which is why it is not called severity. The corpus author judged how bad
    #: this class of problem usually is; only the client knows what it is worth against
    #: their system. Multiplying the hint by the measured rate would produce a number
    #: that looks like a CVSS score, is not one, and would be the single most misleading
    #: thing in a report -- so the report prints the hint and the rate side by side and
    #: a human sets severity. `test_severity_is_carried_not_derived` asserts no field
    #: anywhere does otherwise.
    severity_hint: str = ""
    #: The corpus case and revision this came from, so a transcript is traceable to the
    #: exact payload text that produced it. The corpus is edited between engagements,
    #: which is the point of it, so "case INJ-D-0001" alone does not identify anything.
    corpus_id: str = ""
    #: OWASP LLM risk id ("LLM01".."LLM10"), for the coverage table.
    owasp: str = ""
    #: The technique, one level finer than `owasp`, in the corpus's vocabulary.
    #: Carried from the case, never computed: deriving one is the bug the corpus
    #: field was added to fix. Empty for a web finding, which has no corpus case.
    #:
    #: NOT part of `finding_signature`. The signature decides whether two findings
    #: are "the same issue" across a re-test, and widening it is how a re-test
    #: reports a vulnerability as fixed when it was only reclassified.
    taxon: str = ""
    #: The methodology phase the case belongs to ("01".."08"), as the corpus numbered
    #: it. Carried for the same reason as `owasp` and for one more: a coverage table is
    #: organised by phase, and the phase is what joins a measurement to the `category`
    #: a human writes on the finding in the deliverable. `ChatCase` has carried this
    #: since the corpus adapter was written; it was the one field that stopped at the
    #: adapter and never reached the bundle, so nothing downstream could group by it.
    methodology_phase: str = ""
    #: Public references for the technique, so a finding cites the literature rather
    #: than asserting itself. What a client's own security team asks for.
    references: list[str] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)

    def transition(
        self,
        dst: FindingState,
        *,
        reason: str,
        by: str,
        human_gate: bool = False,
    ) -> Finding:
        """Move to a new state or raise IllegalTransition.

        CONFIRMED can only be reached with human_gate=True -- the human review
        gate. This is the authorization boundary for findings.
        """
        if not can_transition(self.state, dst):
            raise IllegalTransition(f"{self.state.value} -> {dst.value} is not allowed")
        if dst is FindingState.CONFIRMED and not human_gate:
            raise IllegalTransition("CONFIRMED requires an explicit human gate")
        # The evidence cap, enforced HERE rather than in the callers.
        #
        # Testing a chatbot for jailbreaks, harmful content, bias or
        # misinformation needs an oracle no regex can provide, so the harness
        # runs a judge model against a versioned rubric. But a judge is another
        # stochastic system, and the harness's whole claim is that what it calls
        # validated was FALSIFIED -- reproduced against a control, by code. A
        # second model agreeing is not a falsification.
        #
        # So a judge-only finding stops at REPRODUCED and a human decides. This
        # lives in the state machine for the same reason the CONFIRMED gate does:
        # a rule that every caller has to remember is a rule that one caller will
        # forget, and the failure mode is a report that overclaims.
        # REJECTED, DUPLICATE and HUMAN_REVIEW are exempt. The first two because
        # the cap restricts PROMOTION, never falsification. HUMAN_REVIEW because it
        # is not a stronger claim about the evidence -- it is the handoff to a
        # person, which is exactly what a capped finding needs in order not to be a
        # dead end. What the cap forbids is code calling something validated.
        ceiling = MAX_STATE_FOR.get(self.evidence_strength)
        if (ceiling is not None and not human_gate
                and dst not in (FindingState.REJECTED, FindingState.DUPLICATE,
                                FindingState.HUMAN_REVIEW)
                and _RANK.get(dst, 0) > _RANK.get(ceiling, 0)):
            raise IllegalTransition(
                f"evidence_strength={self.evidence_strength.name} cannot reach "
                f"{dst.value}; it stops at {ceiling.value} and needs a human gate "
                "beyond that")
        self.history.append(
            {
                "from": self.state.value,
                "to": dst.value,
                "reason": reason,
                "by": by,
                "human_gate": human_gate,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        self.state = dst
        if dst is FindingState.CONFIRMED:
            self.review_status = "confirmed"
        return self
