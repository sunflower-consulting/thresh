"""The seam into the deliverable, and the two ways it has already gone wrong.

WHY THERE IS ONE DIRECTORY PER RUN.

The engine this replaces wrote `<engagement>/measurements.yaml` -- one path per
engagement -- and refused to clobber it. The refusal was right and a test pinned
it. But one destination means retests, tuning runs and breadth output all collide
there, and the only way past the refusal is `--force`, which then destroys the
earlier file. A guard satisfiable only by deleting what it guards is a guard that
gets forced every time.

WHY `leads/` IS EXCLUDED IN CODE AND NOT BY CONVENTION.

A garak hit has no control arm, nothing proved out of band, and a regex that
matched text the target authored. It is a place to look. It reaches a report as a
coverage count and nowhere else, and a convention is what fails the week somebody
is in a hurry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sunflower_stem import runbundle
from sunflower_stem.consolidate import NEVER_READ, collect
from sunflower_stem.consolidate import write as consolidate
from sunflower_stem.export import CASE_FIELDS, measurements, write_run

DOC_ENGINE = Path("/Users/napahlm/Projects/GitHub/doc-engine")


def _finding(case_id="INJ-D-0001", state="validated", **trials) -> dict:
    base = {"trials": 10, "hits": 9, "control_trials": 10, "control_hits": 0,
            "control_kind": "matched", "p_value": 0.0001, "asr": 0.9,
            "asr_ci": [0.6, 0.98], "truncated": False}
    return {"id": f"run-{case_id}", "corpus_id": case_id, "state": state,
            "taxon": "llm01-indirect-doc", "owasp": "LLM01",
            "evidence_strength": "CALLBACK", "trials": {**base, **trials}}


def _bundle(tmp: Path, rid: str, *, quotable: bool, findings) -> Path:
    return runbundle.write(
        tmp / rid,
        runbundle.Manifest(run_id=rid, quotable=quotable,
                           not_quotable_because="" if quotable else "tuning",
                           trials=10, stem_version="0.0.0-test"),
        findings)


# --- one directory per run -------------------------------------------------------

def test_export_never_writes_the_engagement_root(tmp_path):
    """THE test. The root holds human files and the consolidated output; a run
    writing there is how two runs collide and how a hand-written file gets
    clobbered by a tool."""
    eng = tmp_path / "eng"; eng.mkdir()
    (eng / "findings.yaml").write_text("[]")
    before = {p.name for p in eng.iterdir()}

    bundle = _bundle(tmp_path, "2026-01-01T00-00-00_a", quotable=True,
                     findings=[_finding()])
    manifest, findings = runbundle.load(bundle)
    out = write_run(eng, manifest, findings)

    assert out.parent.name == "runs"
    assert {p.name for p in eng.iterdir()} == before | {"runs"}
    assert (eng / "findings.yaml").read_text() == "[]", "a human file was touched"


def test_two_runs_both_survive_with_nothing_forced(tmp_path):
    """The behaviour the old shape could not have. Exporting a second run used to
    need a flag that destroyed the first."""
    eng = tmp_path / "eng"; eng.mkdir()
    for rid in ("2026-01-01T00-00-00_a", "2026-01-02T00-00-00_b"):
        m, f = runbundle.load(_bundle(tmp_path, rid, quotable=True,
                                      findings=[_finding()]))
        write_run(eng, m, f)
    assert len(list((eng / "runs").iterdir())) == 2


def test_a_run_with_no_manifest_is_not_exportable(tmp_path):
    """A run killed partway through leaves events and no manifest. That is a run
    nobody declared anything about, and exporting it would invent a record."""
    eng = tmp_path / "eng"; eng.mkdir()
    with pytest.raises(ValueError, match="no run_id"):
        write_run(eng, {}, [])


# --- what a row may and may not say ---------------------------------------------

def test_a_finding_with_no_trials_yields_no_row(tmp_path):
    """Not a row of zeroes. `attempts: 0` renders as a measured zero, which is a
    different and wronger claim than the case's absence."""
    doc = measurements({"run_id": "r"}, [{"id": "x", "corpus_id": "C-1"}])
    assert doc["cases"] == []


def test_truncated_is_always_emitted_even_when_false():
    """"The budget ran out" and "nobody said" are different claims, and a reader
    cannot tell them apart from an absent key."""
    doc = measurements({"run_id": "r"}, [_finding()])
    assert doc["cases"][0]["truncated"] is False
    assert "truncated" in doc["cases"][0]


def test_rows_come_out_in_the_contract_order():
    """Two runs' files should line up under diff, which is how anybody reviews a
    change to a number."""
    row = measurements({"run_id": "r"}, [_finding()])["cases"][0]
    assert list(row) == [k for k in CASE_FIELDS if k in row]


def test_no_severity_is_emitted():
    """A human sets likelihood and impact, because severity is a judgement about a
    deployment this instrument knows nothing about. Deriving it here would put an
    invented number in a risk matrix."""
    doc = measurements({"run_id": "r"}, [_finding()])
    blob = json.dumps(doc)
    for word in ("severity", "likelihood", "impact", "cvss"):
        assert word not in blob.lower()


# --- consolidation ---------------------------------------------------------------

def test_a_run_not_marked_quotable_never_reaches_the_consolidated_file(tmp_path):
    """What makes running the same case five times while tuning it safe. The run
    stays on disk and stays diffable; it just cannot be quoted."""
    eng = tmp_path / "eng"; eng.mkdir()
    m, f = runbundle.load(_bundle(tmp_path, "2026-01-01T00-00-00_t",
                                  quotable=False, findings=[_finding()]))
    write_run(eng, m, f)
    doc, notes = collect(eng)
    assert doc["cases"] == []
    assert any("not quotable" in n for n in notes)
    assert (eng / "runs" / "2026-01-01T00-00-00_t").is_dir(), "it must stay on disk"


def test_an_unmarked_bundle_is_not_quotable(tmp_path):
    """Fail-closed, and the one default here worth arguing about. A bundle written
    before the field existed is not one somebody declared final; it is one nobody
    declared anything about."""
    assert runbundle.is_quotable({}) is False
    assert runbundle.is_quotable({"quotable": "yes"}) is False


def test_the_newest_quotable_run_wins_per_case(tmp_path):
    """Newest, not best. Quoting the run with the highest rate would be selecting
    the result, which is the thing this tool exists to stop other tools doing."""
    eng = tmp_path / "eng"; eng.mkdir()
    for rid, hits in (("2026-01-01T00-00-00_a", 9), ("2026-01-02T00-00-00_b", 3)):
        m, f = runbundle.load(_bundle(tmp_path, rid, quotable=True,
                                      findings=[_finding(hits=hits)]))
        write_run(eng, m, f)
    doc, _notes = collect(eng)
    assert len(doc["cases"]) == 1
    assert doc["cases"][0]["hits"] == 3, "the later run must win, not the higher rate"


def test_findings_yaml_can_pin_a_particular_run(tmp_path):
    """Pinning is how one particular transcript gets quoted when the newest run is
    not the one the write-up describes."""
    eng = tmp_path / "eng"; eng.mkdir()
    for rid, hits in (("2026-01-01T00-00-00_a", 9), ("2026-01-02T00-00-00_b", 3)):
        m, f = runbundle.load(_bundle(tmp_path, rid, quotable=True,
                                      findings=[_finding(hits=hits)]))
        write_run(eng, m, f)
    (eng / "findings.yaml").write_text(yaml.safe_dump(
        [{"id": "F-001", "case": "INJ-D-0001", "run": "2026-01-01T00-00-00_a"}]))
    doc, _notes = collect(eng)
    assert doc["cases"][0]["hits"] == 9, "the pin must beat the newest run"


def test_consolidation_is_idempotent(tmp_path):
    """Delete the output, run it again, get the same bytes. A consolidated file
    that drifts is one nobody can review by diffing, and diffing is the only way
    anybody notices a change to a number."""
    eng = tmp_path / "eng"; eng.mkdir()
    m, f = runbundle.load(_bundle(tmp_path, "2026-01-01T00-00-00_a",
                                  quotable=True, findings=[_finding()]))
    write_run(eng, m, f)
    first = consolidate(eng)[0].read_text()
    (eng / "measurements.yaml").unlink()
    assert consolidate(eng)[0].read_text() == first


def test_a_leads_directory_is_excluded_in_code(tmp_path):
    """A garak hit has no control arm, nothing proved out of band, and a regex
    that matched text the target authored. It is a place to look.

    Asserted on the exclusion set rather than only by behaviour, because the
    behaviour would also pass if `leads/` simply held no measurements file -- and
    the requirement is that it could not be read even if it did."""
    assert "leads" in NEVER_READ

    eng = tmp_path / "eng"; eng.mkdir()
    (eng / "runs").mkdir()
    lead = eng / "runs" / "leads"; lead.mkdir()
    (lead / "manifest.yaml").write_text(yaml.safe_dump(
        {"run_id": "leads", "quotable": True}))
    (lead / "measurements.yaml").write_text(yaml.safe_dump(
        {"run_id": "leads", "generated_by": "garak",
         "cases": [{"case_id": "xss.MarkdownImageExfil", "attempts": 10,
                    "hits": 7, "truncated": False}]}))
    doc, _notes = collect(eng)
    assert doc["cases"] == [], "a lead reached the consolidated measurements"


def test_the_consolidated_file_names_every_run_it_drew_from(tmp_path):
    """The consumer's schema types `run_id` as one string and its renderer prints
    one "Run" line, because it was written when one run made one file.
    Consolidation spans runs, so the full provenance goes in the header comment:
    honest, traceable, and schema-safe."""
    eng = tmp_path / "eng"; eng.mkdir()
    for rid in ("2026-01-01T00-00-00_a", "2026-01-02T00-00-00_b"):
        m, f = runbundle.load(_bundle(tmp_path, rid, quotable=True,
                                      findings=[_finding(case_id=rid[-1])]))
        write_run(eng, m, f)
    text = consolidate(eng)[0].read_text()
    assert "2026-01-01T00-00-00_a" in text and "2026-01-02T00-00-00_b" in text
    assert "not the only one" in text


# --- the consumer contract -------------------------------------------------------

@pytest.mark.skipif(not DOC_ENGINE.is_dir(),
                    reason="no doc-engine checkout beside this one")
def test_every_field_emitted_is_one_the_consumer_allows(tmp_path):
    """The consuming schema is `additionalProperties: false`, so a field this
    emits and that does not allow is a build failure in the other repo.

    This caught `runs_quoted`, which was useful, honest, and would have failed
    doc-engine's validator the first time anyone ran it."""
    schema = json.loads(
        (DOC_ENGINE / "schema/measurements.schema.json").read_text())
    eng = tmp_path / "eng"; eng.mkdir()
    m, f = runbundle.load(_bundle(tmp_path, "2026-01-01T00-00-00_a",
                                  quotable=True, findings=[_finding()]))
    write_run(eng, m, f)
    doc = yaml.safe_load(consolidate(eng)[0].read_text())

    assert set(doc) <= set(schema["properties"]), (
        f"not in the consumer's schema: {sorted(set(doc) - set(schema['properties']))}")
    allowed = set(schema["properties"]["cases"]["items"]["properties"])
    for case in doc["cases"]:
        assert set(case) <= allowed, f"case keys: {sorted(set(case) - allowed)}"
    for key in schema["required"]:
        assert key in doc, f"required by the consumer and absent: {key}"
    assert set(CASE_FIELDS) <= allowed


# --- the second machine file, which arrived with the obligation lookup -----------

def _mapping(run_id="r", **over) -> dict:
    """A control mapping shaped like `obligations.coverage()`'s output."""
    doc = {"run_id": run_id, "trellis_version": "sha256:abcdef0123456789",
           "trellis_sha256": "abcdef", "scope": {"instruments": ["x"],
                                                 "system_classes": []},
           "counts": {"tested": 1, "review-only": 1},
           "testable_total": 1, "testable_with_evidence": 1, "review_only_total": 1,
           "obligations": [
               {"obligation": "x:Art.1", "instrument": "x", "citation": "Art.1",
                "verifiable": "test", "status": "tested", "findings": ["F-1"]},
               {"obligation": "x:Art.2", "instrument": "x", "citation": "Art.2",
                "verifiable": "review", "status": "review-only",
                "evidence_expected": "a policy"}],
           "taxa": {"counts": {"mapped": 1, "declined": 0, "unknown": 1},
                    "techniques": [
                        {"taxon": "fam-mapped", "state": "mapped",
                         "obligations": ["x:Art.1"]},
                        {"taxon": "fam-new", "state": "unknown",
                         "note": "never crosswalked", "family": "fam",
                         "family_crosswalked": ["fam-mapped"]}]}}
    doc.update(over)
    return doc


def test_a_run_writes_the_control_mapping_with_its_own_run_id(tmp_path):
    """The consuming schema requires `run_id` at the top level and the engine this
    was ported from never emitted it, so every control mapping it wrote would have
    failed validation on a required key. The run's id wins over whatever the mapping
    document happened to carry."""
    from sunflower_stem.export import CONTROL_MAPPING

    bundle = _bundle(tmp_path, "2026-10-18T09-00-00_run-a", quotable=True,
                     findings=[_finding()])
    manifest, findings = runbundle.load(bundle)
    eng = tmp_path / "engagement"
    eng.mkdir()
    out = write_run(eng, manifest, findings,
                    control_mapping=_mapping(run_id="something-else"))
    doc = yaml.safe_load((out / CONTROL_MAPPING).read_text())
    assert doc["run_id"] == manifest["run_id"]
    assert [r["status"] for r in doc["obligations"]] == ["tested", "review-only"]
    assert doc["taxa"]["counts"]["unknown"] == 1


def test_a_run_with_no_obligation_lookup_writes_no_control_mapping(tmp_path):
    """An engagement that was never scoped against an obligation catalogue has no
    control mapping, which is a fact about the engagement rather than a failure."""
    from sunflower_stem.export import CONTROL_MAPPING

    bundle = _bundle(tmp_path, "2026-10-18T09-00-00_run-a", quotable=True,
                     findings=[_finding()])
    manifest, findings = runbundle.load(bundle)
    eng = tmp_path / "engagement"
    eng.mkdir()
    out = write_run(eng, manifest, findings)
    assert not (out / CONTROL_MAPPING).exists()
    assert (out / "measurements.yaml").is_file(), "the first file still arrives"


def test_the_control_mapping_is_consolidated_from_one_run_entire(tmp_path):
    """Chosen differently from the measurements, deliberately. A measurement is per
    case so the newest quotable run wins per case; a control mapping's rows interlock
    -- `probed-clean` here and `not-probed` there are two statements about the same
    selection -- so stitching them from two runs would describe a page no run ever
    measured."""
    from sunflower_stem.export import CONTROL_MAPPING

    eng = tmp_path / "engagement"
    eng.mkdir()
    for rid, status in (("2026-10-18T09-00-00_old", "not-probed"),
                        ("2026-10-18T10-00-00_new", "tested")):
        bundle = _bundle(tmp_path, rid, quotable=True, findings=[_finding()])
        manifest, findings = runbundle.load(bundle)
        mapping = _mapping()
        mapping["obligations"][0]["status"] = status
        if status == "not-probed":
            mapping["obligations"][0].pop("findings", None)
            mapping["obligations"][0]["reason"] = "nothing bearing on it ran"
        write_run(eng, manifest, findings, control_mapping=mapping)

    _out, notes = consolidate(eng)
    doc = yaml.safe_load((eng / CONTROL_MAPPING).read_text())
    assert doc["run_id"].endswith("_new")
    assert doc["obligations"][0]["status"] == "tested", (
        "the newer run wins entire, rather than per obligation")
    assert any("superseded" in n for n in notes), (
        "and the earlier one is in the notes rather than merged")
