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
from tests.paths import DOC_ENGINE

from sunflower_stem import runbundle
from sunflower_stem.consolidate import NEVER_READ, collect
from sunflower_stem.consolidate import write as consolidate
from sunflower_stem.export import CASE_FIELDS, measurements, write_run


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


# --- the obligation lookup reaching the export path ------------------------------

def _engagement(eng: Path, **ai_system) -> Path:
    """An engagement directory, with `ai_system` when a scope is wanted."""
    eng.mkdir(parents=True, exist_ok=True)
    doc: dict = {"client": "Acme Bank AS", "client_short": "Acme",
                 "engagement": "assistant assessment", "reference": "SF-2026-EX01",
                 "version": "1.0", "date": "2026-10-18",
                 "classification": "CONFIDENTIAL", "consultancy": "Sunflower",
                 "authors": [{"name": "A Tester", "role": "Consultant"}],
                 "methodology_version": "v1.0.0",
                 "test_window": {"start": "2026-10-18", "end": "2026-10-19"},
                 "scope": {"in": ["the assistant"], "out": []},
                 "executive_summary": "Placeholder."}
    if ai_system:
        doc["ai_system"] = dict(ai_system)
    (eng / "engagement.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return eng


def _run(tmp_path: Path, eng: Path, rid: str, findings, **kw) -> int:
    """`stem export` over a bundle, through the real CLI."""
    from sunflower_stem.cli import main

    _bundle(tmp_path, rid, quotable=kw.pop("quotable", True), findings=findings)
    argv = ["export", rid, str(eng), "--runs-dir", str(tmp_path)]
    argv += kw.pop("extra", [])
    return main(argv)


def test_an_engagement_that_named_no_regimes_gets_no_control_mapping(tmp_path, capsys):
    """The renderer's schema makes `ai_system` optional and says why it matters:
    without it every obligation in the catalogue renders as not-probed, which is a
    page of noise. A client reading duties they do not carry learns to skip the
    appendix, and the row that mattered goes with it. So nothing is written, and the
    readout says what to add."""
    from sunflower_stem.export import CONTROL_MAPPING

    eng = _engagement(tmp_path / "eng")
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [_finding()]) == 0
    out = capsys.readouterr().out
    assert not list(eng.rglob(CONTROL_MAPPING))
    assert "no ai_system block" in out
    assert "page of noise" in out
    assert (eng / "runs" / "2026-01-01T00-00-00_a" / "measurements.yaml").is_file(), (
        "the first machine file still arrives")


def test_the_regimes_come_from_the_engagement_file_and_not_from_a_flag(tmp_path):
    """The scope is an engagement fact with a home the renderer already defined. It
    is not on the signed inventory: that model is the policy gate's allowlist, it is
    frozen and hash-covered so a new field unverifies every inventory that exists,
    it is read at run time where this is needed at export time, and one run can
    honestly be reported against more than one scope."""
    from sunflower_stem.export import CONTROL_MAPPING

    eng = _engagement(tmp_path / "eng", instruments=["dora"],
                      classes=["ict-third-party"], role="deployer")
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [_finding()]) == 0
    doc = yaml.safe_load(
        (eng / "runs" / "2026-01-01T00-00-00_a" / CONTROL_MAPPING).read_text())
    assert doc["scope"] == {"instruments": ["dora"],
                            "system_classes": ["ict-third-party"]}
    assert {r["instrument"] for r in doc["obligations"]} == {"dora"}, (
        "an engagement is shown the duties it carries and no others")


def test_a_rejected_measurement_is_never_cited_as_evidence(tmp_path):
    """THE overclaim this wiring would otherwise introduce.

    A run bundle holds every completed measurement, including the ones whose control
    arm fired as often as the attack -- which is the measurement that says the
    technique did NOT cause the result. Handing the whole list to the lookup marks
    obligations `tested` and prints those ids as the evidence, so a hardened target
    that proved nothing reads as four obligations tested in a client's regulatory
    appendix. Measured on the bundled lab at `--safety strong`, which is exactly
    what it did.

    What it must say instead is `probed-clean`: a technique bearing on it ran and
    nothing stood up. That is a result, and a better one for the client."""
    from sunflower_stem.export import CONTROL_MAPPING

    eng = _engagement(tmp_path / "eng", instruments=["ai-act", "dora"],
                      classes=["annex-iii-high-risk"])
    rejected = _finding(case_id="C-1", state="rejected")
    rejected["taxon"] = "jailbreak-crescendo"
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [rejected]) == 0
    doc = yaml.safe_load(
        (eng / "runs" / "2026-01-01T00-00-00_a" / CONTROL_MAPPING).read_text())

    assert doc["testable_with_evidence"] == 0
    assert not [r for r in doc["obligations"] if r["status"] == "tested"]
    clean = [r for r in doc["obligations"] if r["status"] == "probed-clean"]
    assert clean, "it ran, so the obligations bearing on it are probed-clean"
    assert all("jailbreak-crescendo" in r["techniques_run"] for r in clean)
    for row in doc["obligations"]:
        assert rejected["id"] not in (row.get("findings") or [])


def test_a_validated_measurement_is_cited(tmp_path):
    """The other half, or the test above would pass on a mapping that cites
    nothing at all."""
    from sunflower_stem.export import CONTROL_MAPPING

    eng = _engagement(tmp_path / "eng", instruments=["ai-act", "dora"],
                      classes=["annex-iii-high-risk"])
    hit = _finding(case_id="C-1", state="validated")
    hit["taxon"] = "jailbreak-crescendo"
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [hit]) == 0
    doc = yaml.safe_load(
        (eng / "runs" / "2026-01-01T00-00-00_a" / CONTROL_MAPPING).read_text())
    tested = [r for r in doc["obligations"] if r["status"] == "tested"]
    assert tested
    assert all(hit["id"] in r["findings"] for r in tested)
    assert doc["testable_with_evidence"] == len(tested)


def test_a_technique_that_ran_clean_is_told_from_one_that_never_ran(tmp_path):
    """The pair the whole module exists to keep apart, end to end through the CLI.
    `attempted` is derived from every completed measurement and the citations from
    the subset a report would quote, so one bundle answers both questions."""
    from sunflower_stem.export import CONTROL_MAPPING

    eng = _engagement(tmp_path / "eng", instruments=["ai-act"],
                      classes=["annex-iii-high-risk"])
    rejected = _finding(case_id="C-1", state="rejected")
    rejected["taxon"] = "jailbreak-crescendo"
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [rejected]) == 0
    doc = yaml.safe_load(
        (eng / "runs" / "2026-01-01T00-00-00_a" / CONTROL_MAPPING).read_text())
    by_status: dict[str, list[str]] = {}
    for row in doc["obligations"]:
        by_status.setdefault(row["status"], []).append(row["obligation"])
    assert by_status.get("probed-clean"), "something ran"
    assert by_status.get("not-probed"), "and something did not"
    assert not set(by_status["probed-clean"]) & set(by_status["not-probed"])
    for oid in by_status["not-probed"]:
        row = next(r for r in doc["obligations"] if r["obligation"] == oid)
        assert row.get("reason"), "and the one that did not run says why"


def test_the_run_that_measured_it_is_the_run_the_mapping_names(tmp_path):
    """The consuming schema requires `run_id` at the top level, and the engine this
    was ported from never emitted one."""
    from sunflower_stem.export import CONTROL_MAPPING

    eng = _engagement(tmp_path / "eng", instruments=["dora"])
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [_finding()]) == 0
    doc = yaml.safe_load(
        (eng / "runs" / "2026-01-01T00-00-00_a" / CONTROL_MAPPING).read_text())
    assert doc["run_id"] == "2026-01-01T00-00-00_a"


def test_the_mapping_can_be_declined_for_one_export(tmp_path):
    from sunflower_stem.export import CONTROL_MAPPING

    eng = _engagement(tmp_path / "eng", instruments=["dora"])
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [_finding()],
                extra=["--no-control-mapping"]) == 0
    assert not list(eng.rglob(CONTROL_MAPPING))


def test_a_catalogue_that_fails_its_own_digest_stops_the_export(tmp_path):
    """Exit 2, because it is a configuration fault rather than a measured one, and
    an export that silently skipped the mapping would look like an engagement that
    was never scoped."""
    eng = _engagement(tmp_path / "eng", instruments=["dora"])
    bad = tmp_path / "snapshot.yaml"
    bad.write_text(yaml.safe_dump({"obligations": {"x": {}}, "body_sha256": "nope"}))
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [_finding()],
                extra=["--snapshot", str(bad)]) == 2


def test_both_machine_files_reach_the_engagement_root(tmp_path, capsys):
    """What the seam is for. The measurements are per case from the newest quotable
    run that measured each; the mapping is one run entire, because its rows
    interlock."""
    from sunflower_stem.cli import main
    from sunflower_stem.export import CONTROL_MAPPING, MEASUREMENTS

    eng = _engagement(tmp_path / "eng", instruments=["ai-act"],
                      classes=["annex-iii-high-risk"])
    for rid in ("2026-01-01T00-00-00_a", "2026-01-02T00-00-00_b"):
        assert _run(tmp_path, eng, rid, [_finding()]) == 0
    assert main(["consolidate", str(eng)]) == 0

    assert (eng / MEASUREMENTS).is_file()
    assert (eng / CONTROL_MAPPING).is_file()
    out = capsys.readouterr().out
    assert "from 2 quotable run(s)" in out, (
        "the count read a key removed when it broke the consumer's schema, so it "
        "printed 0 over a file drawn from several")
    assert CONTROL_MAPPING in out


@pytest.mark.skipif(not DOC_ENGINE.is_dir(), reason="the renderer is another repo")
def test_the_control_mapping_satisfies_the_consumer_contract(tmp_path):
    """Structural, so it runs without the renderer's own validator installed:
    every key the schema requires is present, every status is in its closed enum,
    and the two combinations the schema refuses are absent."""
    from sunflower_stem.cli import main
    from sunflower_stem.export import CONTROL_MAPPING

    schema = json.loads(
        (DOC_ENGINE / "schema/control-mapping.schema.json").read_text())
    eng = _engagement(tmp_path / "eng", instruments=["ai-act", "dora"],
                      classes=["annex-iii-high-risk"])
    hit = _finding(case_id="C-1", state="validated")
    hit["taxon"] = "jailbreak-crescendo"
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [hit]) == 0
    assert main(["consolidate", str(eng)]) == 0
    doc = yaml.safe_load((eng / CONTROL_MAPPING).read_text())

    for key in schema["required"]:
        assert key in doc, f"required by the consumer and absent: {key}"
    rows = schema["properties"]["obligations"]["items"]
    statuses = set(rows["properties"]["status"]["enum"])
    for row in doc["obligations"]:
        for key in rows["required"]:
            assert key in row, f"{row.get('obligation')}: missing {key}"
        assert row["status"] in statuses
        assert row["verifiable"] in set(rows["properties"]["verifiable"]["enum"])
        if row["status"] == "review-only":
            assert not row.get("findings"), "no test speaks to a document"
        if row["status"] == "not-probed":
            assert row.get("reason"), "'no findings' and 'nobody looked' must differ"
        if row["status"] == "tested":
            assert row.get("findings")


@pytest.mark.skipif(not DOC_ENGINE.is_dir(), reason="the renderer is another repo")
def test_the_control_mapping_passes_the_renderers_own_validator(tmp_path):
    """The structural test above states the rules in this repository's words. This
    one hands the file to the renderer's actual validator, so the two cannot drift."""
    jsonschema = pytest.importorskip("jsonschema")
    from sunflower_stem.cli import main
    from sunflower_stem.export import CONTROL_MAPPING

    schema = json.loads(
        (DOC_ENGINE / "schema/control-mapping.schema.json").read_text())
    eng = _engagement(tmp_path / "eng", instruments=["ai-act"],
                      classes=["annex-iii-high-risk"])
    assert _run(tmp_path, eng, "2026-01-01T00-00-00_a", [_finding()]) == 0
    assert main(["consolidate", str(eng)]) == 0

    validator = jsonschema.Draft202012Validator(schema)
    for path in sorted(eng.rglob(CONTROL_MAPPING)):
        errors = sorted(validator.iter_errors(yaml.safe_load(path.read_text())),
                        key=lambda e: list(e.path))
        assert not errors, f"{path.name}: {[e.message for e in errors[:3]]}"
