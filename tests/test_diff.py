"""What closed, what is still there, and the fix that only blocked one phrasing.

THE CLAIM THIS FILE GUARDS.

"Fixed" is the worst thing this toolchain could get wrong, so most of these tests are
about refusing to say it. A chatbot is stochastic: a case that landed 4 times in 10
and now lands 0 in 10 might be fixed or might be the same assistant on a quieter day.
And a client who patches the exact prompt in our report makes that one case stop
reproducing while the technique is untouched -- which a diff keyed on the case alone
calls a fix.

So closure needs the variants. A finding closes only when the proved case AND its
variants fail to reproduce; when a variant still lands it is `narrowed`, and when no
variant was measured at all it is `unverified` and no claim is made in either
direction.

AND THE IDENTITY HAS TO SURVIVE A RETEST.

A finding's id is `{run_id}-{case_id}`, so keying on it reports the whole library as
fixed-and-new. The signature is the case id, and it deliberately excludes the taxon:
widening the identity is precisely how a retest tells a client a vulnerability is
fixed when somebody reclassified it.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from sunflower_thresh import runbundle
from sunflower_thresh.diff import (
    MOVED,
    STOOD_UP,
    Diff,
    as_document,
    diff_runs,
    finding_signature,
    render,
    variants_of,
)
from sunflower_thresh.schemas.finding import FindingState

TAXON = "llm01-direct-override"


class _Case:
    """A library case, as the loader hands one over."""

    def __init__(self, taxon=TAXON, exemplar=""):
        self.taxon = taxon
        self.drafted_by = {"exemplar": exemplar} if exemplar else {}


LIBRARY = {
    "INJ-D-0001": _Case(),
    "INJ-D-0002": _Case(),
    "INJ-D-0003": _Case(),
    "OUT-E-0001": _Case("llm05-markdown-exfil"),
}
ALL_CASES = sorted(LIBRARY)


def _finding(case="INJ-D-0001", state="validated", asr=0.9, taxon=TAXON, **over):
    row = {"id": f"run-{case}", "corpus_id": case, "title": case, "state": state,
           "taxon": taxon, "severity_hint": "high", "test_cases": [case],
           "trials": {"trials": 10, "hits": int(asr * 10), "asr": asr}}
    row.update(over)
    return row


def _bundle(tmp: Path, rid: str, findings, cases=None, **manifest) -> Path:
    base = {"run_id": rid, "quotable": True, "trials": 10, "corpus_sha": "corpus-a",
            "plan_hash": "plan-a", "cases": list(cases if cases is not None
                                                 else ALL_CASES)}
    base.update(manifest)
    return runbundle.write(tmp / rid, runbundle.Manifest(**base), findings)


#: Each `_diff` writes a fresh pair, because a bundle is written once: calling it
#: twice in one test used to reuse the directory, which `runbundle.write` now
#: refuses outright rather than overwriting.
_PAIR = itertools.count()


def _diff(tmp, before, after, *, before_cases=None, after_cases=None,
          library=LIBRARY, **mf) -> Diff:
    slot = tmp / f"pair-{next(_PAIR)}"
    b = _bundle(slot, "before", before, before_cases)
    a = _bundle(slot, "after", after, after_cases, **mf)
    return diff_runs(b, a, library=library)


# --- the identity -----------------------------------------------------------------

def test_the_signature_is_the_case_and_not_the_finding_id():
    """Ids embed the run id, so every finding in a retest has a new one. Keying on
    it would report the entire library as fixed and new in one breath."""
    first = _finding(state="validated")
    first["id"] = "2026-10-01T09-00_run-a-INJ-D-0001"
    second = _finding(state="validated")
    second["id"] = "2026-11-01T09-00_run-b-INJ-D-0001"
    assert first["id"] != second["id"]
    assert finding_signature(first) == finding_signature(second) == "case:INJ-D-0001"


def test_the_taxon_is_deliberately_not_in_the_signature():
    """THE one to get right. Widening the identity is how a retest reports a
    vulnerability as fixed when it was only reclassified, and the taxon is the field
    most likely to be corrected between two runs."""
    before = _finding(taxon="llm01-direct-override")
    after = _finding(taxon="llm01-goal-hijack")
    assert finding_signature(before) == finding_signature(after)


def test_a_hand_written_finding_falls_back_to_its_content_and_not_its_id():
    """Something somebody noticed and wrote up has no case. Its identity is target,
    CWE, product, version and the significant words of its title -- a set, so a
    reworded title is still the same finding rather than a fix plus a regression."""
    # WITH an id, because `Finding.id` is required and the fixture that omitted one
    # is why this passed while the fallback was unreachable. Caught in review.
    one = {"id": "ACME-2026-001", "title": "The bot leaks the system prompt",
           "target": "h", "cwe": "CWE-200"}
    reworded = {"id": "ACME-2026-001-v2",
                "title": "system prompt leaks from the bot", "target": "h",
                "cwe": "CWE-200"}
    other = {"id": "ACME-2026-002", "title": "The bot calls the write tool",
             "target": "h", "cwe": "CWE-200"}
    assert finding_signature(one).startswith("sig:"), (
        "a finding with no case must not key on its own id, which is editable")
    assert finding_signature(one) == finding_signature(reworded)
    assert finding_signature(one) != finding_signature(other)


def test_what_counts_as_real_is_not_defined_twice():
    """Imported from the export seam rather than redefined. "Real enough to cite" and
    "real enough that its disappearance is news" are the same question, and the
    engine this was ported from used a narrower tuple -- so the two would have
    disagreed about every judge-proved finding, which `MAX_STATE_FOR[JUDGE]` caps at
    REPRODUCED."""
    from sunflower_thresh.obligations import CITABLE_STATES

    assert STOOD_UP is CITABLE_STATES
    assert "reproduced" in STOOD_UP
    assert "rejected" not in STOOD_UP


# --- closure needs the variants ---------------------------------------------------

def test_a_fix_that_blocks_the_proved_case_while_a_variant_lands_is_not_closed(
        tmp_path):
    """The row this module exists for. A client who patched the exact prompt in the
    report lands here, and calling it fixed would be the most consequential thing
    this toolchain could say wrongly."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0),
               _finding("INJ-D-0002", "validated", 0.8)])
    assert d.counts()["closed"] == 0
    assert [r.case for r in d.narrowed] == ["INJ-D-0001"]
    assert d.narrowed[0].variants_reproducing == ("INJ-D-0002",)
    text = render(d)
    assert "still landing: INJ-D-0002" in text
    assert "it is not a fix" in text


def test_a_case_closes_only_with_its_variants_retested_clean(tmp_path):
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0),
               _finding("INJ-D-0002", "rejected", 0.0),
               _finding("INJ-D-0003", "rejected", 0.0)])
    assert [r.case for r in d.closed] == ["INJ-D-0001"]
    assert set(d.closed[0].variants_run) == {"INJ-D-0002", "INJ-D-0003"}
    assert d.closed[0].variants_missing == (), "the whole set, or it is not closure"
    assert not d.narrowed and not d.unverified
    assert "all 2 variant(s) re-tested without reproducing" in render(d)


def test_a_case_that_stopped_with_no_variant_retested_makes_no_claim(tmp_path):
    """Not a fix and not a non-fix. With no variant measured there is nothing to
    distinguish a fixed technique from a fixed string, and saying so is the whole
    value of the bucket."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0)],
              after_cases=["INJ-D-0001"])
    assert [r.case for r in d.unverified] == ["INJ-D-0001"]
    assert not d.closed
    text = render(d)
    assert "is not a claim" in text
    assert "fixed technique cannot be told from a fixed string" in text


def test_with_no_library_nothing_can_be_called_closed(tmp_path):
    """The conservative default. Forgetting to pass the library understates closure
    rather than claiming it, which is the right direction to fail in."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0),
               _finding("INJ-D-0002", "rejected", 0.0)],
              library=None)
    assert not d.closed
    assert [r.case for r in d.unverified] == ["INJ-D-0001"]


def test_a_variant_is_a_sibling_on_the_taxon_or_an_explicit_exemplar():
    """Two signals. `drafted_by.exemplar` is the stronger claim because an author
    wrote down which DIFFERENT defence the variant stresses, and no case in the real
    library carries one yet -- so the shared taxon is the signal that works today and
    the exemplar is read where it appears."""
    assert variants_of("INJ-D-0001", taxon=TAXON, library=LIBRARY) == {
        "INJ-D-0002", "INJ-D-0003"}
    assert variants_of("OUT-E-0001", taxon="llm05-markdown-exfil",
                       library=LIBRARY) == set()
    assert variants_of("INJ-D-0001", taxon=TAXON, library=None) == set()

    drafted = dict(LIBRARY)
    drafted["LEAD-x"] = _Case(taxon="something-else", exemplar="OUT-E-0001")
    assert "LEAD-x" in variants_of("OUT-E-0001", taxon="llm05-markdown-exfil",
                                   library=drafted), (
        "an explicit exemplar is a variant whatever its taxon says")


def test_a_case_is_never_its_own_variant():
    assert "INJ-D-0001" not in variants_of("INJ-D-0001", taxon=TAXON,
                                           library=LIBRARY)


# --- not retested is not fixed ----------------------------------------------------

def test_a_case_the_retest_never_attempted_is_not_fixed(tmp_path):
    """The bucket that catches a retest whose scope quietly narrowed: a capability
    switched off, a credential expired, a placeholder nobody rebound."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("OUT-E-0001", "rejected", 0.0, "llm05-markdown-exfil")],
              after_cases=["OUT-E-0001"])
    assert [r.case for r in d.not_retested] == ["INJ-D-0001"]
    assert not d.closed and not d.unverified
    assert d.incomplete is True
    assert "cannot be read as a pass" in render(d)


def test_a_skipped_case_counts_as_not_attempted(tmp_path):
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [],
              skipped={"INJ-D-0001": "no seeding sink was available"})
    assert [r.case for r in d.not_retested] == ["INJ-D-0001"]
    assert d.incomplete


def test_incomplete_and_regressed_are_different_facts(tmp_path):
    """Collapsing them makes both less useful: one means a new finding appeared, the
    other means the retest cannot be read as a pass at all."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("OUT-E-0001", "validated", 0.7, "llm05-markdown-exfil")],
              after_cases=["OUT-E-0001"])
    assert d.regressed is True
    assert d.incomplete is True
    empty = Diff()
    assert empty.regressed is False and empty.incomplete is False


# --- still present, and movement without a verdict change -------------------------

def test_a_finding_that_still_reproduces_is_still_present(tmp_path):
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "validated", 0.9)])
    assert [r.case for r in d.still_present] == ["INJ-D-0001"]
    assert not d.changed, "an unmoved rate is not news"


def test_a_rate_that_moved_without_changing_the_verdict_is_reported(tmp_path):
    """A mitigation that halved the rate did something, and a client who deployed one
    wants to see it even though the finding still stands."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "validated", 0.9 - MOVED)])
    assert [r.case for r in d.changed] == ["INJ-D-0001"]
    assert d.still_present, "and it is still present as well"


def test_a_candidate_that_did_not_come_back_is_not_news(tmp_path):
    """Comparing things that never stood up produces noise rather than news."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "observed", 0.1)],
              [], after_cases=[])
    assert d.counts() == {"closed": 0, "narrowed": 0, "unverified": 0,
                          "still_present": 0, "new": 0, "not_retested": 0,
                          "changed": 0}


def test_a_finding_that_only_appears_in_the_later_run_is_new(tmp_path):
    d = _diff(tmp_path, [], [_finding("INJ-D-0002", "validated", 0.8)])
    assert [r.case for r in d.new] == ["INJ-D-0002"]
    assert d.regressed


# --- the comparison can itself be invalid ----------------------------------------

def test_two_runs_that_selected_different_cases_say_so_loudly(tmp_path):
    """The worst of the two drifts, for a reason specific to this output: a retest
    that selected a narrower set reports what it did not attempt as closed if its
    manifest does not list the cases."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0)],
              plan_hash="plan-b")
    assert d.plan_drifted
    text = render(d)
    assert "selected different cases" in text
    assert "read `not re-tested` before anything else" in text


def test_two_runs_on_different_libraries_say_so(tmp_path):
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "validated", 0.9)],
              corpus_sha="corpus-b")
    assert d.corpus_drifted
    assert "two different questions" in render(d)


def test_a_run_that_predates_plan_hashes_does_not_cry_wolf(tmp_path):
    """Flagging a drift against an absent field would fire on every old bundle."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "validated", 0.9)],
              plan_hash="")
    assert not d.plan_drifted


def test_a_missing_bundle_is_reported_rather_than_raised(tmp_path):
    """A run killed before it finished leaves events and no manifest, and the right
    answer is a diff that says so."""
    b = _bundle(tmp_path, "before", [_finding("INJ-D-0001", "validated")])
    d = diff_runs(b, tmp_path / "never-ran", library=LIBRARY)
    assert d.after_id == "never-ran"
    assert [r.case for r in d.not_retested] == ["INJ-D-0001"] or d.unverified


# --- there is no closed state, and nothing reads a typed one ---------------------

def test_closure_is_not_a_finding_state():
    """It is a property of a PAIR of runs. A member on `FindingState` would let a
    closure be typed by hand into a findings file, and a hand-typed closure is what
    a retest exists to replace."""
    assert not any(s.value == "closed" for s in FindingState)
    # The MEMBER name, not a second look at the values. The line here used to be
    # `"closed" not in {s.value for s in FindingState}`, which is the assertion
    # above spelled differently: either one alone catches `CLOSED = "closed"`, and
    # neither catches `CLOSED = "shut"` -- a typed closure under another spelling,
    # which is the thing the docstring says must not exist.
    assert "CLOSED" not in FindingState.__members__


def test_nothing_in_the_output_came_from_a_status_somebody_typed(tmp_path):
    """A finding carrying `closed: true` changes nothing: every bucket is derived."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "validated", 0.9, closed=True,
                        review_status="closed")])
    assert [r.case for r in d.still_present] == ["INJ-D-0001"]
    assert not d.closed
    assert "derived from the two run bundles" in as_document(d)["note"]


# --- the document -----------------------------------------------------------------

def test_the_document_carries_the_variants_a_closure_rested_on(tmp_path):
    """A closure with nothing behind it is the claim this module avoids making, so a
    reader of the file can see what it was based on without rerunning anything."""
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0),
               _finding("INJ-D-0002", "rejected", 0.0),
               _finding("INJ-D-0003", "rejected", 0.0)])
    doc = as_document(d)
    assert doc["counts"]["closed"] == 1
    assert doc["closed"][0]["case"] == "INJ-D-0001"
    assert doc["closed"][0]["variants_retested"] == ["INJ-D-0002", "INJ-D-0003"]
    assert "variants_not_retested" not in doc["closed"][0]
    assert "closed" in doc and "narrowed" not in doc, "empty buckets are omitted"


def test_a_clean_result_for_part_of_the_variant_set_is_not_closure(tmp_path):
    """Caught in review, and it contradicted this module's own thesis.

    The rule was "any clean variant", so for a technique with variants B and C a
    clean result for the case and B declared closure with no result for C at all.
    The claim being made is about the TECHNIQUE, and two of three cases say nothing
    about the third.
    """
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0),
               _finding("INJ-D-0002", "rejected", 0.0)])
    assert not d.closed
    assert [r.case for r in d.unverified] == ["INJ-D-0001"]
    row = d.unverified[0]
    assert row.variants_run == ("INJ-D-0002",)
    assert row.variants_missing == ("INJ-D-0003",)
    assert set(row.variants_known) == {"INJ-D-0002", "INJ-D-0003"}
    text = render(d)
    assert "1 of 2 variant(s) were not re-tested: INJ-D-0003" in text
    assert as_document(d)["unverified"][0]["variants_not_retested"] == ["INJ-D-0003"]


def test_the_document_names_a_surviving_variant(tmp_path):
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0),
               _finding("INJ-D-0002", "validated", 0.8)])
    doc = as_document(d)
    assert doc["narrowed"][0]["variants_still_reproducing"] == ["INJ-D-0002"]
    assert "closed" not in doc


# --- rates ------------------------------------------------------------------------

def test_a_finding_with_no_trial_stats_reports_its_verdict_not_a_rate(tmp_path):
    """Printing "0% -> 0%" for a case proved by reproducing twice implies a
    measurement that says the opposite of what happened: validated to rejected would
    read as no change."""
    before = _finding("INJ-D-0001", "validated")
    after = _finding("INJ-D-0001", "rejected")
    before["trials"] = None
    after["trials"] = None
    d = _diff(tmp_path, [before], [after])
    row = (d.closed + d.unverified + d.narrowed)[0]
    assert row.measured is False
    assert row.rate("before") == "stood up"
    assert row.rate("after") == "did not reproduce"


def test_a_case_absent_from_the_later_run_reports_not_attempted(tmp_path):
    d = _diff(tmp_path, [_finding("INJ-D-0001", "validated", 0.9)], [],
              after_cases=[])
    assert d.unverified or d.not_retested
    row = (d.unverified + d.not_retested)[0]
    assert row.rate("after") == "not attempted"


@pytest.mark.parametrize("bad", [None, {}, {"asr": "nonsense"}])
def test_a_malformed_rate_does_not_raise(bad):
    from sunflower_thresh.diff import Row

    row = Row(signature="case:X", before={"trials": bad}, after={"trials": bad})
    assert row.asr_before == 0.0
    assert row.asr_after == 0.0


def test_a_variant_on_the_attempt_list_that_produced_no_finding_is_not_a_retest(
        tmp_path):
    """The distinction that decides whether a closure rests on a measurement.

    A case that runs and finds nothing produces a REJECTED finding, not no finding.
    So a case the manifest selected with no finding at all did not complete -- gate
    denied, errored, or killed -- and counting it as "re-tested without reproducing"
    would quote a measurement nobody has. Found by a test asserting the generous
    reading, which was the wrong one.
    """
    # INJ-D-0002 completed and was rejected; INJ-D-0003 was selected and produced
    # nothing at all.
    d = _diff(tmp_path,
              [_finding("INJ-D-0001", "validated", 0.9)],
              [_finding("INJ-D-0001", "rejected", 0.0),
               _finding("INJ-D-0002", "rejected", 0.0)],
              after_cases=["INJ-D-0001", "INJ-D-0002", "INJ-D-0003"])
    assert not d.closed, "INJ-D-0003 was selected and never reported"
    assert d.unverified[0].variants_run == ("INJ-D-0002",)
    assert d.unverified[0].variants_missing == ("INJ-D-0003",), (
        "being on the attempt list is not a result")

    # Reported clean, and it closes: a finding exists either way for a case that ran.
    d2 = _diff(tmp_path,
               [_finding("INJ-D-0001", "validated", 0.9)],
               [_finding("INJ-D-0001", "rejected", 0.0),
                _finding("INJ-D-0002", "rejected", 0.0),
                _finding("INJ-D-0003", "rejected", 0.0)],
               after_cases=["INJ-D-0001", "INJ-D-0002", "INJ-D-0003"])
    assert [r.case for r in d2.closed] == ["INJ-D-0001"]
