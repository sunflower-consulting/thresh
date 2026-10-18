"""A hit from another tool is a lead, and the ways that stops being true.

THREE THINGS THIS FILE EXISTS FOR.

A lead must not be able to become a finding. Not "is not converted" -- `Lead` has
nowhere to put a state or a severity, the way `ChatCase` has nowhere to put a claim
of authorship, and the triage document is asserted to contain no finding state at
all. A tool that says "FAIL: jailbreak" has said where to point a measurement.

No lead may go unrecorded. A scan costs a client's messages, and a hit that is
quietly dropped is a result nobody can account for. Every lead handed to `triage`
comes back as exactly one record, and `dismissed` is a real outcome rather than a
failure to produce one.

And a report that cannot be read must RAISE. Neither format can be checked from the
file alone: garak carries no schema version anywhere, its release notes record that
the `eval` and `digest` entry format changed and that `probe_summary` was added, and
promptfoo's own maintainers say some rows carry no `metadata.pluginId`. An ingester
hard-coded to one version reads zero leads from the other and reports a clean scan,
which is the short-load defect from `corpus.py` in different clothes.

THE FIXTURES ARE THE DOCUMENTED SHAPES, NOT A LIVE RUN.

Neither tool is installed here, so these fixtures encode what the published
references document -- `passed`, `total_evaluated` and `nones` on a garak eval entry,
status 0/1/2 on an attempt, `version: 3` and `gradingResult.componentResults` for
promptfoo. They are not a recording of a real report, which is precisely why the
readers refuse what they do not recognise: the first real report either works or
says what it saw.
"""
from __future__ import annotations

import json

import pytest

from sunflower_thresh.leads import (
    GARAK_EVALUATED,
    LEAD_CASE_PREFIX,
    PROBE_DECLINED,
    PROBE_TAXA,
    PROMPTFOO_VERSION,
    Lead,
    LeadError,
    Outcome,
    read_garak,
    read_promptfoo,
    taxon_for,
    triage,
    write_triage,
)
from sunflower_thresh.schemas.finding import FindingState


def _garak(tmp_path, *rows) -> str:
    p = tmp_path / "garak.report.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def _attempt(**over) -> dict:
    row = {"entry_type": "attempt", "uuid": "a1", "status": GARAK_EVALUATED,
           "probe_classname": "dan.Dan_11_0",
           "detector_results": {"dan.DAN": [1.0, 0.0]}}
    row.update(over)
    return row


def _pf(tmp_path, *results, version=PROMPTFOO_VERSION) -> str:
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"version": version, "timestamp": "t",
                             "results": {"results": list(results)}}))
    return str(p)


def _row(**over) -> dict:
    row = {"testIdx": 0, "success": False, "score": 0,
           "gradingResult": {"pass": False, "score": 0, "reason": "leaked"},
           "testCase": {"metadata": {"pluginId": "harmful:privacy"}}}
    row.update(over)
    return row


# --- a lead cannot become a finding ----------------------------------------------

def test_a_lead_has_nowhere_to_put_a_verdict():
    """Structural, not a convention. A boolean field invites `if lead.passed`, and
    that is the import this forbids: the other tool's pass/fail must never reach the
    finding state machine."""
    fields = set(Lead.__dataclass_fields__)
    for forbidden in ("state", "severity", "passed", "success", "validated",
                      "confirmed"):
        assert forbidden not in fields, (
            f"`Lead` must not carry {forbidden!r}")
    lead = Lead(source="garak", probe="dan", flagged="garak said so", row=1)
    assert isinstance(lead.flagged, str), "the verdict survives only as prose"
    with pytest.raises(Exception):      # noqa: B017 - frozen, by any error it gives
        lead.probe = "changed"


def test_no_triage_document_carries_a_finding_state(tmp_path):
    """The readout version of the same rule. A reader skimming this file must not
    find a word the finding state machine uses, because that is how a lead gets
    quoted as a measured result."""
    leads = [Lead(source="garak", probe="dan", flagged="f", row=1,
                  taxon="jailbreak-dan"),
             Lead(source="promptfoo", probe="nope", flagged="f", row=2)]
    recs = triage(leads, covered_by=lambda t: ["JB-0001"])
    out = write_triage(tmp_path / "leads" / "triage.yaml", recs)
    text = out.read_text()
    present = [s.value for s in FindingState if s.value in text]
    assert present == [], f"a finding state reached a lead document: {present}"
    assert "NONE OF THIS IS A FINDING" in text


def test_the_leads_directory_is_excluded_from_consolidation_in_code():
    """Not by convention and not by a path check at the call site: a garak hit must
    not be able to reach a consolidated measurement even by accident."""
    from sunflower_thresh.consolidate import NEVER_READ

    assert "leads" in NEVER_READ


def test_a_promoted_case_id_cannot_be_mistaken_for_a_library_case(tmp_path):
    """A promoted lead is a DRAFT nobody has reviewed. Giving it a corpus-shaped id
    like `INJ-D-0001` is how it ends up cited as a library case."""
    recs = triage([Lead(source="garak", probe="dan", flagged="f", row=1,
                        taxon="jailbreak-dan")],
                  covered_by=lambda t: [])
    assert recs[0].outcome is Outcome.PROMOTED
    assert recs[0].case_id.startswith(LEAD_CASE_PREFIX)
    row = recs[0].as_row()
    assert row["status"] == "draft"
    assert "no control arm" in row["note"]


# --- no lead goes unrecorded -----------------------------------------------------

def test_every_lead_comes_back_as_exactly_one_record():
    """The invariant. A scan costs a client's messages and a dropped hit is a result
    nobody can account for."""
    leads = [
        Lead(source="garak", probe="dan", flagged="f", row=1, taxon="jailbreak-dan"),
        Lead(source="garak", probe="encoding", flagged="f", row=2,
             taxon="llm01-encoded-payload"),
        Lead(source="garak", probe="snowball.X", flagged="f", row=3),
        Lead(source="promptfoo", probe="pii", flagged="f", row=4,
             taxon="llm02-pii-disclosure"),
    ]
    covered = {"jailbreak-dan": ["JB-0001"]}
    recs = triage(leads, covered_by=covered.get)
    assert len(recs) == len(leads)
    assert [r.lead.probe for r in recs] == [x.probe for x in leads], "and in order"
    assert {r.outcome for r in recs} == {Outcome.COVERED, Outcome.PROMOTED,
                                         Outcome.DISMISSED}


def test_the_three_outcomes_say_different_things(tmp_path):
    leads = [Lead(source="garak", probe="dan", flagged="f", row=1,
                  taxon="jailbreak-dan"),
             Lead(source="garak", probe="encoding", flagged="f", row=2,
                  taxon="llm01-encoded-payload"),
             Lead(source="garak", probe="unmapped.Probe", flagged="f", row=3)]
    recs = triage(leads, covered_by={"jailbreak-dan": ["JB-0001"]}.get)
    rows = {r.outcome: r.as_row() for r in recs}
    assert rows[Outcome.COVERED]["cases"] == ["JB-0001"]
    assert "reason" not in rows[Outcome.COVERED]
    assert rows[Outcome.PROMOTED]["case_id"].startswith(LEAD_CASE_PREFIX)
    assert "cases" not in rows[Outcome.PROMOTED]
    assert "unmapped.Probe" in rows[Outcome.DISMISSED]["reason"]
    assert "case_id" not in rows[Outcome.DISMISSED]


def test_two_leads_on_the_same_uncovered_technique_get_distinct_ids():
    recs = triage([Lead(source="garak", probe="dan", flagged="f", row=1,
                        taxon="jailbreak-dan"),
                   Lead(source="promptfoo", probe="dan", flagged="f", row=2,
                        taxon="jailbreak-dan")],
                  covered_by=lambda t: [])
    ids = [r.case_id for r in recs]
    assert len(set(ids)) == 2, ids


def test_an_id_already_in_use_is_not_minted_again():
    recs = triage([Lead(source="garak", probe="dan", flagged="f", row=1,
                        taxon="jailbreak-dan")],
                  covered_by=lambda t: [],
                  existing_ids=[f"{LEAD_CASE_PREFIX}jailbreak-dan"])
    assert recs[0].case_id != f"{LEAD_CASE_PREFIX}jailbreak-dan"


# --- the mapping is a judgement, never a guess -----------------------------------

def test_an_unmapped_probe_is_dismissed_by_name_rather_than_guessed_at():
    """Attributing a lead to the wrong technique attributes it, through the
    obligation crosswalk, to the wrong regulatory duty. The same reason the
    obligation catalogue gives for its own eight written declines."""
    recs = triage([Lead(source="garak", probe="brandnew.Probe", flagged="f", row=1)],
                  covered_by=lambda t: [])
    assert recs[0].outcome is Outcome.DISMISSED
    assert "brandnew.Probe" in recs[0].reason
    assert "PROBE_TAXA" in recs[0].reason, "and it says where to add the judgement"


def test_a_garak_module_resolves_but_a_sibling_namespace_does_not():
    """garak reports `module.ClassName` and the module is the family, so the head is
    read. promptfoo's `harmful:privacy` is NOT reduced to `harmful`, because
    `harmful:self-harm` is a different technique and a prefix rule would fuse them."""
    assert taxon_for("dan.Dan_11_0") == PROBE_TAXA["dan"]
    assert taxon_for("dan") == PROBE_TAXA["dan"]
    assert taxon_for("harmful:privacy") == PROBE_TAXA["harmful:privacy"]
    assert taxon_for("harmful:self-harm") == "", (
        "an unmapped namespaced plugin must not inherit its namespace's mapping")
    assert taxon_for("") == ""
    assert taxon_for("nothing.Known") == ""


def test_a_taxon_the_vocabulary_does_not_hold_is_dismissed_not_promoted():
    """A draft case against an unresolvable technique crosswalks to no obligation and
    sits in the library confusing people."""
    recs = triage([Lead(source="garak", probe="dan", flagged="f", row=1,
                        taxon="jailbreak-dan")],
                  covered_by=lambda t: [], known_taxa={"something-else"})
    assert recs[0].outcome is Outcome.DISMISSED
    assert "vocabulary does not hold" in recs[0].reason


def test_every_mapping_names_a_technique_THAT_EXISTS():
    """THE test this table needed, and it is here because its absence cost something.

    The table was first written with thirteen entries and SEVEN of them named
    techniques that do not exist -- `jailbreak-dan`, `llm01-encoded-payload`,
    `llm02-pii-disclosure` and four more, invented from memory instead of read from
    the vocabulary. Nothing in the module was wrong: the vocabulary check at triage
    dismissed every one of them correctly, which is how it was found. But a mapping
    table of names that resolve to nothing is worse than an empty one, because each
    entry reads like a judgement somebody made.

    Shape checks cannot catch this -- all seven were shaped exactly like technique
    ids. Only membership can."""
    from sunflower_thresh.obligations import known_taxa, load_snapshot

    vocabulary = known_taxa(load_snapshot())
    unknown = {p: t for p, t in PROBE_TAXA.items() if t not in vocabulary}
    assert not unknown, (
        f"these probes are mapped to techniques the vocabulary does not hold, so "
        f"every lead they produce will be dismissed: {unknown}")


def test_a_probe_is_mapped_or_declined_but_never_both():
    """A decline is a judgement that there is nothing to map it to. Holding both
    would mean the table disagrees with itself, and `taxon_for` reads the decline
    first, so the mapping would be silently dead."""
    assert not (set(PROBE_TAXA) & set(PROBE_DECLINED))


@pytest.mark.parametrize("probe", sorted(PROBE_DECLINED))
def test_a_deliberate_decline_carries_its_reason_into_the_dismissal(probe):
    """The probes with no honest target are a decision, and a reader meeting the
    dismissal should get the decision rather than "nobody has mapped this yet"."""
    assert taxon_for(probe) == ""
    assert len(PROBE_DECLINED[probe]) > 80, "a decline argues for itself"
    recs = triage([Lead(source="promptfoo", probe=probe, flagged="f", row=1)],
                  covered_by=lambda t: [])
    assert recs[0].outcome is Outcome.DISMISSED
    assert "deliberately unmapped" in recs[0].reason
    assert PROBE_DECLINED[probe][:40] in recs[0].reason


@pytest.mark.parametrize("probe,taxon", sorted(PROBE_TAXA.items()))
def test_every_mapping_is_shaped_like_a_technique_id(probe, taxon):
    """Weaker than the membership test above and kept beside it: it is the check that
    passed on all seven invented names, which is the point."""
    assert taxon == taxon.lower()
    assert " " not in taxon
    assert "-" in taxon, f"{probe} -> {taxon} is not shaped like a technique id"


# --- reading garak ----------------------------------------------------------------

def test_only_an_evaluated_attempt_can_carry_a_lead(tmp_path):
    """Its reference says status 0 is not sent and 1 is a response with no
    evaluation. Neither can carry a verdict, and both are counted rather than
    dropped."""
    path = _garak(tmp_path,
                  _attempt(status=0, uuid="s0"),
                  _attempt(status=1, uuid="s1"),
                  _attempt(status=GARAK_EVALUATED, uuid="s2"))
    res = read_garak(path)
    assert res.rows == 3
    assert res.interpreted == 1
    assert sum(res.skipped.values()) == 2
    assert any("status 0" in k for k in res.skipped)
    assert any("status 1" in k for k in res.skipped)


def test_a_detector_that_scored_zero_is_not_a_hit(tmp_path):
    res = read_garak(_garak(tmp_path,
                            _attempt(detector_results={"d": [0.0, 0.0]}),
                            _attempt(uuid="a2",
                                     detector_results={"d": [0.0, 0.5]})),
                     allow_unreadable=True)
    assert res.interpreted == 1
    assert any("no detector flagged" in k for k in res.skipped)


def test_a_detector_score_written_as_a_string_is_not_truthy(tmp_path):
    """`bool("0.0")` is True, and a detector that wrote its score as text would
    otherwise read as a hit on every single attempt."""
    res = read_garak(_garak(tmp_path, _attempt(detector_results={"d": ["0.0"]})),
                     allow_unreadable=True)
    assert res.interpreted == 0


def test_an_eval_row_uses_the_field_names_the_reference_documents(tmp_path):
    """`total_evaluated`, not `total`. The obvious guess is wrong, and a reader built
    on it finds nothing in a report full of results."""
    path = _garak(tmp_path,
                  {"entry_type": "eval", "probe": "latentinjection.X",
                   "passed": 3, "total_evaluated": 10, "nones": 0},
                  {"entry_type": "eval", "probe": "xss.Y",
                   "passed": 10, "total_evaluated": 10})
    res = read_garak(path)
    assert res.interpreted == 1
    assert "3 of 10" in res.leads[0].flagged
    assert any("everything passed" in k for k in res.skipped)

    wrong = _garak(tmp_path, {"entry_type": "eval", "probe": "p",
                              "passed": 1, "total": 10})
    with pytest.raises(LeadError, match="understood none of them"):
        read_garak(wrong)


def test_a_probe_is_found_under_whichever_key_this_version_used(tmp_path):
    """The reference documents neither `probe` nor `detector` on an eval entry, so
    the key is looked for rather than assumed."""
    for key in ("probe_classname", "probe", "probe_name", "classname"):
        row = {"entry_type": "eval", "passed": 1, "total_evaluated": 4, key: "dan.X"}
        res = read_garak(_garak(tmp_path, row))
        assert res.leads[0].probe == "dan.X", key


def test_a_row_naming_no_probe_is_recorded_rather_than_attributed(tmp_path):
    res = read_garak(_garak(tmp_path,
                            {"entry_type": "eval", "passed": 1,
                             "total_evaluated": 4}),
                     allow_unreadable=True)
    assert res.interpreted == 0
    assert any("names no probe" in k for k in res.skipped)


def test_a_future_entry_type_is_counted_rather_than_ignored(tmp_path):
    """The next format change arrives as exactly this, and arriving silently is what
    makes it look like a clean scan."""
    res = read_garak(_garak(tmp_path, _attempt(),
                            {"entry_type": "something_new_in_0_12", "x": 1}))
    assert res.interpreted == 1
    assert any("unknown entry_type" in k for k in res.skipped)


# --- reading promptfoo ------------------------------------------------------------

def test_a_failing_row_is_a_lead_and_a_passing_one_is_not(tmp_path):
    res = read_promptfoo(_pf(tmp_path, _row(), _row(testIdx=1, success=True)))
    assert res.rows == 2
    assert res.interpreted == 1
    assert res.leads[0].probe == "harmful:privacy"
    assert "leaked" in res.leads[0].detail


def test_the_plugin_id_is_read_from_wherever_this_version_put_it(tmp_path):
    """Its own documentation does not list `metadata` as a row field, and a provider
    example puts `pluginId` under the test's metadata, so both are tried."""
    for holder in ("testCase", "test"):
        row = _row(testCase=None, test=None)
        row.pop("testCase", None)
        row.pop("test", None)
        row[holder] = {"metadata": {"pluginId": "pii"}}
        res = read_promptfoo(_pf(tmp_path, row))
        assert res.leads[0].probe == "pii", holder
    flat = _row(testCase=None, metadata={"pluginId": "pii"})
    flat.pop("testCase")
    assert read_promptfoo(_pf(tmp_path, flat)).leads[0].probe == "pii"


def test_a_row_with_no_plugin_id_falls_back_to_its_metric(tmp_path):
    """Its maintainers record that some rows carry no `metadata.pluginId` and rely on
    the metric, so attribution is inconsistent by their own account."""
    row = {"testIdx": 2, "success": False,
           "gradingResult": {"pass": False, "reason": "leaked",
                             "componentResults": [
                                 {"pass": False,
                                  "assertion": {"metric": "prompt-extraction"}}]}}
    res = read_promptfoo(_pf(tmp_path, row))
    assert res.leads[0].probe == "prompt-extraction"


def test_a_failing_row_identifying_nothing_is_recorded_rather_than_guessed(tmp_path):
    row = {"testIdx": 3, "success": False, "gradingResult": {"pass": False}}
    res = read_promptfoo(_pf(tmp_path, row), allow_unreadable=True)
    assert res.interpreted == 0
    assert any("identifies no plugin" in k for k in res.skipped)


def test_a_row_that_says_nothing_about_passing_is_not_a_lead(tmp_path):
    """Treating silence as failure would manufacture leads out of error rows."""
    res = read_promptfoo(_pf(tmp_path, {"testIdx": 4,
                                        "testCase": {"metadata":
                                                     {"pluginId": "pii"}}}),
                         allow_unreadable=True)
    assert res.interpreted == 0


def test_an_output_version_this_reader_does_not_know_is_refused(tmp_path):
    """promptfoo HAS a version, so a reader that ignores it is choosing to find out
    later. garak has none, which is why its reader refuses differently."""
    with pytest.raises(LeadError, match="written against"):
        read_promptfoo(_pf(tmp_path, _row(), version=PROMPTFOO_VERSION + 1))


def test_the_jsonl_form_is_read_too(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in
                           [_row(), _row(testIdx=1, success=True)]) + "\n")
    res = read_promptfoo(p)
    assert res.interpreted == 1


def test_a_results_object_without_a_known_row_list_says_what_it_has(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"version": PROMPTFOO_VERSION,
                             "results": {"table": [], "stats": {}}}))
    with pytest.raises(LeadError) as exc:
        read_promptfoo(p)
    assert "table" in str(exc.value) and "stats" in str(exc.value)


# --- the refusal ------------------------------------------------------------------

def test_a_report_with_rows_and_no_leads_raises_rather_than_reporting_a_clean_scan(
        tmp_path):
    """THE test. An ingester hard-coded against one version reads zero leads from
    the other, and zero leads is indistinguishable from a target that resisted
    everything. garak carries no schema version anywhere, so this is the only place
    a format change can announce itself."""
    path = _garak(tmp_path,
                  {"entry_type": "probe_summary", "probe_id": "dan", "score": 0.3},
                  {"entry_type": "probe_summary", "probe_id": "xss", "score": 0.1})
    with pytest.raises(LeadError) as exc:
        read_garak(path)
    message = str(exc.value)
    assert "understood none of them" in message
    assert "probe_summary x2" in message, "it names the entry types it saw"
    assert "probe_id" in message, "and the keys, so somebody knows what to widen"
    assert "--allow-nothing" in message


def test_a_scan_that_genuinely_found_nothing_can_be_accepted_knowingly(tmp_path):
    res = read_garak(_garak(tmp_path, _attempt(detector_results={"d": [0.0]})),
                     allow_unreadable=True)
    assert res.interpreted == 0
    assert res.rows == 1


def test_an_empty_report_is_not_a_clean_scan(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n")
    with pytest.raises(LeadError, match="not a clean scan"):
        read_garak(p, allow_unreadable=True)


def test_an_unreadable_file_is_named(tmp_path):
    with pytest.raises(LeadError, match="cannot read"):
        read_garak(tmp_path / "nope.jsonl")


def test_a_line_that_is_not_json_is_counted_and_does_not_stop_the_read(tmp_path):
    p = tmp_path / "garak.report.jsonl"
    p.write_text("not json at all\n" + json.dumps(_attempt()) + "\n")
    res = read_garak(p)
    assert res.interpreted == 1
    assert res.skipped.get("not JSON") == 1


# --- the report is untrusted input ------------------------------------------------

def test_a_payload_from_the_report_is_flattened_and_truncated(tmp_path):
    """A report holds attacker-crafted prompts. An excerpt with its own newlines
    rendered into a YAML file an operator skims is how somebody copies a payload out
    of context, so it is collapsed to one line and cut."""
    nasty = "line one\nline two\n" + "x" * 500
    res = read_garak(_garak(tmp_path, _attempt(goal=nasty)))
    detail = res.leads[0].detail
    assert "\n" not in detail
    assert len(detail) <= 170
    assert detail.endswith("...")


def test_the_written_document_records_what_was_read_and_not_only_what_was_found(
        tmp_path):
    """"3 leads" alone cannot be told from "3 leads and 900 rows we could not read"."""
    res = read_garak(_garak(tmp_path, _attempt(),
                            _attempt(uuid="a2", status=1)))
    recs = triage(res.leads, covered_by=lambda t: ["JB-0001"])
    out = write_triage(tmp_path / "leads" / "triage.yaml", recs,
                       sources={"garak": res})
    import yaml

    doc = yaml.safe_load(out.read_text())
    assert doc["leads"] == 1
    assert doc["read"]["garak"]["rows"] == 2
    assert doc["read"]["garak"]["leads"] == 1
    assert doc["read"]["garak"]["not_a_lead"], "and why the other row was not one"
