"""One run, carried all the way to the files a report renders from.

THE BAR THIS MEETS, from the plan the package was built against: "CI runs a full
case to evidence to export to obligations chain against the bundled lab."

Every stage of that chain is tested already, and every one of those tests builds its
findings by hand. A hand-built fixture is where a contract drifts: it encodes what
the author believed `validate_chat_case` produces, and it keeps passing after the
producer changes. This test never writes a finding. It runs cases against the
bundled target, takes whatever comes out, and pushes it through export,
consolidation and the obligation mapping to the two files the renderer reads.

What it is therefore able to catch, and the piecewise tests are not: a field the
sweep stopped emitting, a state the export rows cannot represent, a measurement row
whose shape the consuming schema refuses, and a taxon the obligation catalogue
cannot resolve. All four of those look fine from inside one stage.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaml
from tests.paths import DOC_ENGINE, HAVE_DOC_ENGINE

from sunflower_thresh import runbundle
from sunflower_thresh.consolidate import write as consolidate
from sunflower_thresh.context import AgentContext
from sunflower_thresh.evidence.store import EvidenceStore
from sunflower_thresh.export import CASE_FIELDS, CONTROL_MAPPING, MEASUREMENTS, write_run
from sunflower_thresh.lab import serve
from sunflower_thresh.obligations import attempted_taxa as taxa_of
from sunflower_thresh.obligations import citable_findings, coverage, load_snapshot
from sunflower_thresh.policy.engine import PolicyEngine
from sunflower_thresh.preflight import load_inventory, transport_for
from sunflower_thresh.redact import make_redactor
from sunflower_thresh.sweep import applicable_chat_cases, sweep_builtins
from sunflower_thresh.tools.conversation import ConversationStore, chat_tools

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def chain(tmp_path_factory) -> dict:
    """A real run, exported and consolidated. Nothing here is hand-written."""
    tmp = tmp_path_factory.mktemp("chain")
    inv = load_inventory(EXAMPLES / "scope.yaml")
    canary = dict(inv.seeds)["canary"]

    srv, _lab = serve(port=0, deterministic=True, safety="weak")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    store = EvidenceStore(redactor=make_redactor(keep=(canary,)))
    policy = PolicyEngine()
    verified = policy.load_inventory(inv)
    convos = ConversationStore()
    ctx = AgentContext(inv.project, policy,
                       tool_impls=chat_tools(convos, store, policy=policy))

    runnable, unreachable = applicable_chat_cases(
        inv.allowed_tools, capabilities=inv, have_tool_visibility=True,
        seeds=dict(inv.seeds))
    rid = runbundle.run_id("chain")
    try:
        findings, _stats = sweep_builtins(
            ctx, runnable, target_ip="127.0.0.1", port=port, run_id=rid, trials=3,
            seeds=dict(inv.seeds), scoring=runnable, unreachable=unreachable,
            transport=transport_for(inv))
    finally:
        srv.shutdown()

    bundle = runbundle.write(tmp / "runs" / rid, runbundle.Manifest(
        run_id=rid, quotable=True, trials=3, cases=sorted(runnable),
        skipped=dict(unreachable), corpus_sha="builtins"), findings)
    manifest, raw = runbundle.load(bundle)

    snapshot = load_snapshot()
    mapping = coverage(snapshot, citable_findings(raw),
                       attempted=taxa_of(raw), run_id=rid,
                       instruments=("ai-act", "dora"),
                       system_classes=("annex-iii-high-risk",))

    engagement = tmp / "engagement"
    engagement.mkdir()
    out = write_run(engagement, manifest, raw, control_mapping=mapping)
    consolidate(engagement)

    return {"verified": verified, "findings": findings, "raw": raw,
            "runnable": runnable, "run": out, "engagement": engagement,
            "snapshot": snapshot, "mapping": mapping}


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# --- the chain holds --------------------------------------------------------------

def test_a_real_run_reaches_both_machine_files(chain):
    """The bar, stated plainly. Not that each stage works, but that one run's own
    output survives every stage without anybody retyping it."""
    assert chain["verified"] is True, "the gate verified the signed scope"
    assert chain["findings"], "the run measured something"
    assert (chain["run"] / MEASUREMENTS).is_file()
    assert (chain["run"] / CONTROL_MAPPING).is_file()
    assert (chain["engagement"] / MEASUREMENTS).is_file()
    assert (chain["engagement"] / CONTROL_MAPPING).is_file()


def test_every_case_that_ran_has_a_measurement_row(chain):
    """A case that ran and found nothing produces a REJECTED finding, not no
    finding, so the row count is the run's own denominator. A case missing here
    would be a measurement that evaporated between the sweep and the file."""
    rows = _yaml(chain["engagement"] / MEASUREMENTS)["cases"]
    measured = {f.corpus_id or (f.test_cases[0] if f.test_cases else f.id)
                for f in chain["findings"] if f.trials}
    assert {r["case_id"] for r in rows} == measured
    assert rows, "the chain produced no rows at all"


def test_the_row_fields_are_exactly_what_the_contract_names(chain):
    """`CASE_FIELDS` is a contract with another repository, and the rows here come
    from a real Finding rather than a fixture -- so this catches the sweep emitting
    something the row builder drops, or dropping something it needs."""
    for row in _yaml(chain["engagement"] / MEASUREMENTS)["cases"]:
        assert set(row) <= set(CASE_FIELDS), (
            f"{set(row) - set(CASE_FIELDS)} is not in the agreed field set")
        assert "truncated" in row, (
            "always emitted: 'the budget ran out' and 'nobody said' must not look "
            "alike")
        assert isinstance(row["attempts"], int) and row["attempts"] > 0


def test_a_real_finding_carries_a_state_the_export_can_represent(chain):
    """The states a run actually produces have to be ones the consuming schema's
    enum permits. A new FindingState reaching a client's file is a validation
    failure in the other repository."""
    from sunflower_thresh.schemas.finding import FindingState

    produced = {f.state for f in chain["findings"]}
    assert produced, "nothing was produced to check"
    assert produced <= set(FindingState)
    rows = _yaml(chain["engagement"] / MEASUREMENTS)["cases"]
    assert {r["state"] for r in rows} <= {s.value for s in FindingState}


def test_the_obligation_mapping_resolves_the_taxa_the_run_actually_touched(chain):
    """The join between the two halves. A taxon the run emits and the catalogue has
    never seen contributes to no obligation row, which is exactly the state the
    third taxon state exists to report rather than swallow."""
    doc = _yaml(chain["engagement"] / CONTROL_MAPPING)
    touched = taxa_of(chain["raw"])
    reported = {t["taxon"] for t in doc["taxa"]["techniques"]}
    assert touched <= reported, "every technique the run touched is accounted for"
    assert set(doc["taxa"]["counts"]) == {"mapped", "declined", "unknown"}
    assert sum(doc["taxa"]["counts"].values()) == len(reported)


def test_a_rejected_finding_is_measured_but_never_cited(chain):
    """The split the export seam needed. A rejected finding is a real measurement
    and belongs in the measurements file; citing it under `status: tested` would put
    an obligation's evidence list in a client's regulatory appendix for a run that
    established the technique was not the cause."""
    rejected = [f for f in chain["findings"] if f.state.value == "rejected"]
    if not rejected:
        pytest.skip("the lab produced no rejected finding on this run")
    rows = _yaml(chain["engagement"] / MEASUREMENTS)["cases"]
    assert any(r["state"] == "rejected" for r in rows), "measured, and reported"

    cited = {f.get("id") for f in citable_findings(chain["raw"])}
    assert all(f.id not in cited for f in rejected), "and never cited as evidence"
    doc = _yaml(chain["engagement"] / CONTROL_MAPPING)
    for row in doc["obligations"]:
        if row["status"] == "tested":
            assert row.get("findings"), "a tested row names what it rests on"


def test_consolidation_is_idempotent_over_a_real_run(chain):
    """Delete the engagement-root pair, run it again, get identical files. Asserted
    on real output rather than fixtures, because the header is assembled from a
    run's own manifest and that is where a timestamp or a dict order would leak in."""
    before = {name: (chain["engagement"] / name).read_text()
              for name in (MEASUREMENTS, CONTROL_MAPPING)}
    for name in before:
        (chain["engagement"] / name).unlink()
    consolidate(chain["engagement"])
    for name, text in before.items():
        assert (chain["engagement"] / name).read_text() == text, name


# --- and the consuming repository accepts them ------------------------------------

@pytest.mark.skipif(not HAVE_DOC_ENGINE, reason="the report renderer is a separate repository")
def test_both_files_validate_against_the_renderers_own_schemas(chain):
    """The end of the chain, and the only check that the agreed shape is agreed. Both
    schemas live in the other repository and are read from there rather than copied,
    because a copied schema is one that can quietly stop matching."""
    jsonschema = pytest.importorskip("jsonschema")

    for name, schema_file in ((MEASUREMENTS, "measurements.schema.json"),
                              (CONTROL_MAPPING, "control-mapping.schema.json")):
        schema = json.loads((DOC_ENGINE / "schema" / schema_file).read_text())
        doc = _yaml(chain["engagement"] / name)
        errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(doc),
                        key=lambda e: list(e.path))
        assert not errors, (
            f"{name} is not what {schema_file} permits: "
            + "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:3]))


@pytest.mark.skipif(not HAVE_DOC_ENGINE, reason="the report renderer is a separate repository")
def test_the_renderer_declares_every_revision_key_the_mapping_emits(chain):
    """The catalogue was renamed, and both spellings are emitted while the renderer
    catches up. Neither may be one the renderer does not declare, because its schema
    allows extra properties -- so an unknown key is ignored rather than refused, and
    the report silently stops citing the dataset revision it resolved against."""
    schema = json.loads(
        (DOC_ENGINE / "schema" / "control-mapping.schema.json").read_text())
    declared = set(schema["properties"])
    doc = _yaml(chain["engagement"] / CONTROL_MAPPING)
    emitted = {k for k in doc if "version" in k or "sha256" in k}
    assert emitted, "the mapping has to say what it resolved against"
    assert emitted <= declared, (
        f"{sorted(emitted - declared)} is emitted here and not declared by the "
        f"renderer's control-mapping schema. Its additionalProperties is true, so "
        f"an undeclared key is IGNORED rather than refused -- the report would "
        f"silently stop citing the dataset revision it resolved against. Add the "
        f"key there (the catalogue was renamed from trellis to almanac, and both "
        f"spellings are emitted here while the renderer catches up).")
