"""Which regimes a system is in scope of, and the three zeros that differ.

WHY THIS EXISTS. `thresh obligations --system-class X` needs a class, and until now
one was typed from memory. That class is the denominator of the only coverage figure
a report may state, so a wrong one produces no error -- it produces a confident
appendix over the wrong set of duties. The rules to decide it have been in the
catalogue the whole time, under `tier`, and nothing read them.

THE PART WORTH TESTING HARDEST. Of the thirteen classes the rules can reach, three
carry no obligation at all, and an empty mapping means something different for each:
`minimal-risk` is the correct and complete answer, `annex-i-high-risk` is a gap in
the catalogue, and `needs-legal-review` is a refusal to answer about a system that
may well be high-risk. Rendering them alike is the same failure as `probed-clean`
against `not-probed` one layer up, and worse, because the reader is being told what
they owe.
"""
from __future__ import annotations

import pytest

from sunflower_thresh.classify import (
    NOT_A_SCOPE,
    Classification,
    also_ask,
    as_document,
    classes,
    classify,
    questions,
    render,
    why_empty,
)
from sunflower_thresh.obligations import load_snapshot


@pytest.fixture(scope="module")
def snap() -> dict:
    return load_snapshot()


def _answers(**over) -> dict:
    base = {"prohibited": False, "annex-i-embedded": False, "annex-iii-area": False,
            "transparency": False, "dora": False, "dora-third-party": False,
            "cra": False}
    base.update(over)
    return base


# --- the walk ---------------------------------------------------------------------

def test_a_branch_can_name_another_question_and_the_walk_follows(snap):
    """`annex-iii-area` branches to `annex-iii-carve-out`, which is a QUESTION, not
    a class. A walk that treated every branch target as a class would stop there and
    report a class that does not exist."""
    byid = {q["id"]: q for q in questions(snap)}
    assert byid["annex-iii-area"]["if_yes"] == "annex-iii-carve-out"
    assert "annex-iii-carve-out" in byid, "the branch names a question"

    result = classify(snap, _answers(**{"annex-iii-area": True,
                                        "annex-iii-carve-out": False}))
    assert result.classes == ("annex-iii-high-risk",)
    assert [q for q, _a in result.trail][:3] == [
        "prohibited", "annex-i-embedded", "annex-iii-area"]


def test_a_question_with_no_branch_for_the_answer_falls_through(snap):
    """`prohibited: no` has no `if_no`, which is how the catalogue writes "not this,
    carry on". Treating a missing branch as the end would classify everything as
    nothing."""
    byid = {q["id"]: q for q in questions(snap)}
    assert "if_no" not in byid["prohibited"]
    result = classify(snap, _answers(transparency=True))
    assert result.classes[0] == "limited-risk", "it carried on to the last question"


def test_the_independent_questions_add_classes_outside_the_tree(snap):
    """A system can be Annex III high-risk AND an ICT third-party under DORA. Those
    are different instruments and the scope is the union, not a choice."""
    assert {q["id"] for q in also_ask(snap)} == {"dora", "dora-third-party", "cra"}
    result = classify(snap, _answers(**{"annex-iii-area": True,
                                        "annex-iii-carve-out": False,
                                        "dora-third-party": True, "cra": True}))
    assert set(result.classes) == {"annex-iii-high-risk", "ict-third-party",
                                   "product-with-digital-elements"}


def test_an_unanswered_question_stops_rather_than_assuming(snap):
    """The catalogue has a class for "cannot be determined" and guessing an answer
    would bypass it."""
    result = classify(snap, {"prohibited": False})
    assert result.classes == ()
    assert "annex-i-embedded" in result.unanswered
    assert not result.decided


# --- the three zeros --------------------------------------------------------------

def test_a_class_that_cannot_be_determined_is_a_stop_and_not_a_scope(snap):
    """THE one. The Article 6(3) carve-out is the hardest judgement in the Act, and
    `needs-legal-review` is the rules declining to make it. Passing it as a scope
    would filter every obligation away and render a clean page for a system that may
    well be high-risk."""
    result = classify(snap, _answers(**{"annex-iii-area": True,
                                        "annex-iii-carve-out": True}))
    assert result.classes == ("needs-legal-review",)
    assert "needs-legal-review" in result.stops
    assert result.scope_classes == (), "a stop is never a denominator"
    assert not result.decided
    text = render(snap, result)
    assert "STOP" in text
    assert "may well be high-risk" in text
    assert "no class here is a usable denominator" in text


def test_a_prohibited_practice_is_not_a_risk_tier_to_measure(snap):
    """The catalogue's own note: "Not a risk tier to manage; a thing that may not be
    deployed." Whether it may be deployed is not a question an assurance run
    answers."""
    result = classify(snap, _answers(prohibited=True))
    assert result.classes == ("prohibited",)
    assert "prohibited" in result.stops
    assert result.scope_classes == ()
    assert "whether it may be deployed at all" in result.stops["prohibited"]


def test_the_three_empty_classes_are_told_apart(snap):
    """An empty mapping looks identical whichever is true, and they are not the same
    thing to tell a client."""
    carried = {c for o in snap["obligations"].values()
               for c in (o.get("system_class") or ())}
    empties = [c for c in classes(snap) if c not in carried and c not in NOT_A_SCOPE]
    assert set(empties) == {"minimal-risk", "annex-i-high-risk"}, empties

    assert "correct and complete answer" in why_empty(snap, "minimal-risk")
    assert "GAP IN THE DATA" in why_empty(snap, "annex-i-high-risk")
    assert "may well be high-risk" in why_empty(snap, "needs-legal-review")


def test_a_class_carrying_no_obligation_is_kept_out_of_the_scope(snap):
    """Passing it would filter everything away, and the caller is told rather than
    discovering a blank page."""
    result = classify(snap, _answers(**{"annex-i-embedded": True}))
    assert result.classes == ("annex-i-high-risk",)
    assert result.empty == ("annex-i-high-risk",)
    assert result.scope_classes == ()
    assert "gap" in render(snap, result)


def test_minimal_risk_is_an_answer_and_still_not_a_denominator(snap):
    """Zero IS correct for it, and there is still no mapping worth rendering: a page
    of obligations nobody carries teaches the reader to skip the appendix."""
    result = classify(snap, _answers())
    assert result.classes == ("minimal-risk",)
    assert result.scope_classes == ()
    assert "correct and complete" in why_empty(snap, "minimal-risk")


# --- what it hands to the next command --------------------------------------------

def test_a_decided_classification_hands_over_flags_that_work(snap):
    """The point of the whole command: the scope stops being typed from memory."""
    result = classify(snap, _answers(transparency=True, dora=True))
    assert set(result.scope_classes) == {"limited-risk", "important-entity"}
    assert result.decided
    flags = result.as_flags()
    assert flags == ("--system-class limited-risk "
                     "--system-class important-entity")

    # and those classes actually select obligations, which is the whole claim
    from sunflower_thresh.obligations import in_scope

    scoped = in_scope(snap, instruments=("ai-act", "dora"),
                      system_classes=result.scope_classes)
    assert scoped, "the suggested scope must not render an empty mapping"


def test_what_the_catalogue_declined_to_encode_travels_with_the_answer(snap):
    """`open_checks` is the catalogue saying what it saw in secondary sources and
    did NOT encode. In a file nobody opens that is a footnote; attached to the
    answer it is a caveat somebody reads."""
    result = classify(snap, _answers(transparency=True))
    assert result.open_checks
    assert "Digital Omnibus" in " ".join(result.open_checks)
    assert "deliberately did not encode" in render(snap, result)


def test_the_readout_says_it_is_not_legal_advice(snap):
    assert "not legal advice" in render(snap, classify(snap, _answers())).lower()


def test_the_document_records_what_was_answered(snap):
    """An engagement file has to be able to say which questions were answered which
    way, because the scope is a claim somebody made."""
    result = classify(snap, _answers(transparency=True, cra=True))
    doc = as_document(result)
    assert doc["scope_classes"]
    assert {"question": "transparency", "yes": True} in doc["answers"]
    assert {"question": "cra", "yes": True} in doc["answers"]
    assert "Not legal advice" in doc["note"]


def test_an_empty_classification_claims_nothing():
    empty = Classification()
    assert empty.scope_classes == ()
    assert not empty.decided
    assert empty.as_flags() == ""


def test_a_branch_naming_a_class_that_shares_a_question_id_terminates(snap):
    """The bug this walk shipped with, and it was a HANG rather than a wrong answer.

    The first question's id is `prohibited` and its `if_yes` target is the CLASS
    `prohibited`. Resolving the target as a question first sent the walk back to
    index 0 and span forever, which gives an operator nothing at all -- where a
    wrong answer at least gets argued with. The classes table decides what a name
    is, and a cycle guard means malformed rules cannot hang it either.
    """
    byid = {q["id"] for q in questions(snap)}
    assert "prohibited" in byid and "prohibited" in classes(snap), (
        "the collision this guards is real in the shipped catalogue")

    result = classify(snap, _answers(prohibited=True))
    assert result.classes == ("prohibited",)
    tree = {q["id"] for q in questions(snap)}
    walked = [q for q, _a in result.trail if q in tree]
    assert walked == ["prohibited"], "the tree asked once and stopped"
    # The independent questions are still asked: DORA scope does not depend on the
    # AI Act classification, and a prohibited practice can still be supplied to a
    # financial entity.
    assert {q for q, _a in result.trail} - tree == {"dora", "dora-third-party",
                                                    "cra"}


def test_rules_that_loop_are_refused_rather_than_hung():
    """Impossible with the catalogue as written, and cheap to refuse: a hang is the
    one failure mode that leaves somebody with nothing."""
    looping = {
        "tier": {"classes": {"x": {"label": "X"}},
                 "questions": [{"id": "a", "ask": "?", "if_yes": "b"},
                               {"id": "b", "ask": "?", "if_yes": "a"}],
                 "also_ask": [], "open_checks": []},
        "obligations": {},
    }
    result = classify(looping, {"a": True, "b": True})
    assert result.classes == ()
    assert len(result.trail) <= 2, "it stopped instead of spinning"


def test_a_branch_naming_nothing_known_is_taken_at_its_word(snap):
    """A target in neither table is a data fault. It is reported as a class that
    carries no obligation rather than dropped, because a silently missing class is
    how a scope quietly narrows."""
    doc = {
        "tier": {"classes": {"known": {"label": "Known"}},
                 "questions": [{"id": "q", "ask": "?", "if_yes": "typo-class"}],
                 "also_ask": [], "open_checks": []},
        "obligations": {},
    }
    result = classify(doc, {"q": True})
    assert result.classes == ("typo-class",)
    assert result.empty == ("typo-class",)
    assert result.scope_classes == ()
