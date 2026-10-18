"""The two axes that look like one, and the state that did not exist.

THE DEFECT THIS FILE EXISTS FOR.

`for_taxon` returned an empty list for two unrelated situations: a technique the
catalogue deliberately declines to map, FOR A WRITTEN REASON, and a technique the
catalogue has never heard of. The first is a regulatory judgement somebody made and
recorded; the second is nobody's judgement at all. Rounding the second to "no
obligations" is how the differentiating part of a run's output leaves no trace in the
appendix, and nothing notices, because an absence renders the same as a decision.

Measured against the vendored catalogue and the corpus: 82 techniques mapped, 0
declined, 48 unknown. So the declined branch fires on nothing in the library and only
a hand-written finding reaches it -- which is exactly why the tests here drive it
with findings rather than only with a corpus.

AND THE SECOND AXIS, WHICH MUST NOT MERGE WITH THE FIRST.

An obligation is `review-only` because no probe could ever establish it. A technique
is `declined` because no obligation in the catalogue is about it. Three of the eight
declines say the technique "Reaches ai-act:Art.10 and the CRA SBOM duties by REVIEW"
-- it has regulatory relevance, routed to review -- so rendering a declined technique
as a review-only obligation would double-count the same fact in two columns.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from tests.paths import CORPUS
from tests.paths import DOC_ENGINE_SNAPSHOT as DOC_ENGINE

from sunflower_thresh.obligations import (
    DEFAULT_SNAPSHOT,
    SNAPSHOT_SHA256,
    STATUSES,
    VERIFIABLE,
    AlmanacError,
    TaxonState,
    body_sha256,
    cite_as,
    coverage,
    for_taxon,
    in_scope,
    known_taxa,
    load_snapshot,
    provenance,
    read_taxon,
    render,
    snapshot_faults,
    taxa_coverage,
    unreachable_obligations,
)


@pytest.fixture(scope="module")
def snap() -> dict:
    return load_snapshot()


def _mini(**over) -> dict:
    """A snapshot small enough to reason about, with all three taxon states in it."""
    doc = {
        "schema_version": 1,
        "generated_from": "abc123",
        "obligations": {
            "x:Art.1": {"id": "x:Art.1", "instrument": "x", "citation": "Art.1",
                        "title": "testable and mapped", "verifiable": "test",
                        "system_class": ["high"], "applies_from": "2026-01-01"},
            "x:Art.2": {"id": "x:Art.2", "instrument": "x", "citation": "Art.2",
                        "title": "a document", "verifiable": "review",
                        "system_class": ["high"], "evidence_expected": "a policy"},
            "x:Art.3": {"id": "x:Art.3", "instrument": "x", "citation": "Art.3",
                        "title": "testable, nothing mapped to it",
                        "verifiable": "hybrid", "system_class": ["high"]},
            "y:Art.9": {"id": "y:Art.9", "instrument": "y", "citation": "Art.9",
                        "title": "another instrument", "verifiable": "test",
                        "system_class": ["other"]},
        },
        "crosswalk": {"techniques": {
            "fam-mapped": {"obligations": ["x:Art.1"], "rationale": "because"},
            "fam-second": {"obligations": ["x:Art.1"], "rationale": "because"},
        }},
        "unmapped": {"techniques": {
            "fam-declined": {"reason": "no obligation here is about it, and it "
                                       "reaches x:Art.2 by REVIEW"},
        }},
        "taxonomy": {"techniques": {"fam-mapped": {}, "fam-second": {},
                                    "fam-declined": {}}},
        "tier": {}, "calendar": {},
    }
    doc.update(over)
    doc["body_sha256"] = body_sha256(doc)
    return doc


# --- the three taxon states ------------------------------------------------------

def test_a_declined_technique_and_an_unknown_one_do_not_look_alike():
    """THE defect. Both produce no obligations, and that is the only thing they have
    in common: one is a written judgement and the other is a gap nobody has looked
    at."""
    doc = _mini()
    declined = read_taxon(doc, "fam-declined")
    unknown = read_taxon(doc, "fam-never-heard-of")

    assert for_taxon(doc, "fam-declined") == []
    assert for_taxon(doc, "fam-never-heard-of") == []

    assert declined.state is TaxonState.DECLINED
    assert unknown.state is TaxonState.UNKNOWN
    assert declined.reason and "REVIEW" in declined.reason
    assert not unknown.reason
    assert str(declined) != str(unknown)
    assert "declined" in str(declined)
    assert "UNKNOWN" in str(unknown)


def test_a_mapped_technique_carries_what_the_catalogue_said_about_it():
    read = read_taxon(_mini(), "fam-mapped")
    assert read.state is TaxonState.MAPPED
    assert read.obligations == ("x:Art.1",)
    assert read.rationale == "because"
    assert read.speaks_to_obligations


def test_membership_in_the_crosswalk_is_not_the_same_as_being_mapped():
    """Nothing in the catalogue's own tests forbids `obligations: []` -- they assert
    that SOME technique maps to several, which an entry mapping to none satisfies. So
    keying the state on membership would report `mapped` for a technique that speaks
    to nothing, reintroducing the ambiguity these three states remove."""
    doc = _mini(crosswalk={"techniques": {"fam-hollow": {"obligations": []},
                                          "fam-mapped": {"obligations": ["x:Art.1"]}}})
    read = read_taxon(doc, "fam-hollow")
    assert read.state is not TaxonState.MAPPED
    assert not read.speaks_to_obligations
    assert any("no obligations" in f for f in snapshot_faults(doc)), (
        "and it is reported as a data fault, because it is neither state")


def test_a_technique_both_crosswalked_and_declined_resolves_the_same_way_every_time():
    """Contradictory, and nothing forbids it: the vocabulary test is satisfied by a
    technique in either set or both. Mapped wins, in writing, rather than by whichever
    branch happens to be first -- and the contradiction is reported rather than just
    resolved."""
    doc = _mini()
    doc["unmapped"]["techniques"]["fam-mapped"] = {"reason": "contradiction"}
    doc["body_sha256"] = body_sha256(doc)
    assert read_taxon(doc, "fam-mapped").state is TaxonState.MAPPED
    assert any("both crosswalked and declined" in f for f in snapshot_faults(doc))


def test_a_technique_in_the_vocabulary_with_no_position_is_not_the_corpus_fault():
    """A different sentence and a different person to talk to: the catalogue lists the
    technique and has taken no position, where `unknown` means the corpus named
    something the catalogue has never heard of."""
    doc = _mini()
    doc["taxonomy"]["techniques"]["fam-listed-only"] = {}
    doc["body_sha256"] = body_sha256(doc)
    read = read_taxon(doc, "fam-listed-only")
    assert read.state is TaxonState.UNKNOWN
    assert read.in_vocabulary is True
    assert "NO POSITION" in str(read)
    assert read_taxon(doc, "fam-never-listed").in_vocabulary is False
    assert any("no position taken" in f for f in snapshot_faults(doc))


def test_an_unknown_technique_names_its_neighbours_and_claims_nothing():
    """All 48 unknowns in the real corpus are variants inside a family the catalogue
    already crosswalks, so a family's obligation set is one line of code away.
    Inheriting it would have this tool inventing a regulatory judgement, which is the
    thing the eight written declines exist to forbid."""
    read = read_taxon(_mini(), "fam-variant-nobody-mapped")
    assert read.state is TaxonState.UNKNOWN
    assert read.obligations == (), "it must claim no obligation"
    assert read.family == "fam"
    assert read.family_siblings == ("fam-mapped", "fam-second"), (
        "but it must say where a human would look")


def test_a_decline_with_no_reason_is_reported_as_one():
    """A decline is a judgement, and one without its reason cannot be told from
    somebody forgetting to write the entry."""
    doc = _mini(unmapped={"techniques": {"fam-silent": {}}})
    read = read_taxon(doc, "fam-silent")
    assert read.state is TaxonState.DECLINED
    assert "records no reason" in read.reason
    assert any("carry no reason" in f for f in snapshot_faults(doc))


# --- the second axis, which must not merge with the first ------------------------

def test_a_declined_technique_is_not_rendered_as_a_review_only_obligation():
    """Different facts that look alike. An obligation is review-only because no probe
    could establish it; a technique is declined because no obligation here is about
    it. Three real declines say the technique reaches a duty BY REVIEW, so merging
    the two would count the same fact in both columns."""
    doc = _mini()
    cov = coverage(doc, [{"id": "F-1", "taxon": "fam-declined"}],
                   attempted={"fam-declined"}, instruments=("x",))
    assert cov["taxa"]["counts"]["declined"] == 1
    # The declined technique contributed no finding to any obligation row.
    assert all(not r.get("findings") for r in cov["obligations"])
    review = [r for r in cov["obligations"] if r["status"] == "review-only"]
    assert len(review) == 1 and review[0]["obligation"] == "x:Art.2"
    assert cov["review_only_total"] == 1


def test_the_four_outcome_states_are_the_ones_the_consumer_permits():
    """A closed enum in the consuming schema, so a fifth value invented here is a
    validation failure in the other repository rather than a richer readout."""
    doc = _mini()
    cov = coverage(doc, [{"id": "F-1", "taxon": "fam-mapped"}],
                   attempted={"fam-mapped", "fam-second"}, instruments=("x",))
    seen = {r["status"] for r in cov["obligations"]}
    assert seen <= set(STATUSES)
    assert {r["verifiable"] for r in cov["obligations"]} <= set(VERIFIABLE)


def test_probed_clean_and_not_probed_stay_apart():
    """The pair a client most needs kept apart. "No findings against this" and
    "nobody looked" are not the same sentence."""
    doc = _mini()
    clean = coverage(doc, [], attempted={"fam-mapped"}, instruments=("x",))
    row = next(r for r in clean["obligations"] if r["obligation"] == "x:Art.1")
    assert row["status"] == "probed-clean"
    assert row["techniques_run"] == ["fam-mapped"]

    nothing = coverage(doc, [], attempted=set(), instruments=("x",),
                       skipped={"C-1": "no seeding sink was available"})
    row = next(r for r in nothing["obligations"] if r["obligation"] == "x:Art.1")
    assert row["status"] == "not-probed"
    assert "no seeding sink" in row["reason"], (
        "and it carries the run's own words for why")


def test_a_tested_row_names_its_findings_and_a_review_row_cannot():
    """Both pinned by the consuming schema: `tested` requires findings, and
    `review-only` permits none, because no test result can speak to a document."""
    doc = _mini()
    cov = coverage(doc, [{"id": "F-7", "taxon": "fam-mapped"}],
                   attempted={"fam-mapped"}, instruments=("x",))
    tested = next(r for r in cov["obligations"] if r["status"] == "tested")
    assert tested["findings"] == ["F-7"]
    for row in cov["obligations"]:
        if row["status"] == "review-only":
            assert not row.get("findings")
        if row["status"] == "not-probed":
            assert row.get("reason"), "the schema requires it"


def test_an_obligation_no_technique_reaches_is_named_as_a_crosswalk_gap():
    """It has to be labelled `not-probed`, because the status enum is closed. Without
    naming it separately an operator reads "we did not get to it" where the truth is
    that nothing in the library points at it and no run ever would."""
    doc = _mini()
    cov = coverage(doc, [], attempted={"fam-mapped"}, instruments=("x",))
    gap = cov["unreachable_by_any_technique"]
    assert gap["count"] == 1
    assert gap["obligations"] == [{"obligation": "x:Art.3", "verifiable": "hybrid"}]
    assert "crosswalk change, not a bigger run" in gap["note"]
    row = next(r for r in cov["obligations"] if r["obligation"] == "x:Art.3")
    assert row["status"] == "not-probed"


def test_two_numbers_and_never_one_percentage():
    """"Evidence for 3 of 8" and "30 review-only" are different claims, and a single
    figure covering both is the specific way this appendix would mislead."""
    doc = _mini()
    cov = coverage(doc, [{"id": "F-1", "taxon": "fam-mapped"}],
                   attempted={"fam-mapped"}, instruments=("x",))
    assert cov["testable_total"] == 2
    assert cov["testable_with_evidence"] == 1
    assert cov["review_only_total"] == 1
    text = render(cov)
    assert "%" not in text, "a percentage over both would be the misleading figure"


# --- scope ------------------------------------------------------------------------

def test_scope_is_the_engagement_and_not_the_catalogue():
    doc = _mini()
    assert len(in_scope(doc)) == 4
    assert set(in_scope(doc, instruments=("x",))) == {"x:Art.1", "x:Art.2", "x:Art.3"}
    assert set(in_scope(doc, system_classes=("other",))) == {"y:Art.9"}


def test_an_obligation_tagged_all_is_in_scope_whatever_class_was_asked_for():
    """Subtle and deliberate: `all` is a tag an obligation carries meaning the duty
    applies to everything, not a way of asking for every class. Asking for `all` as a
    filter selects only the obligations tagged with it, which is a trap worth having
    a test say out loud."""
    doc = _mini()
    doc["obligations"]["z:Art.0"] = {
        "id": "z:Art.0", "instrument": "z", "citation": "Art.0", "title": "everyone",
        "verifiable": "review", "system_class": ["all"]}
    doc["body_sha256"] = body_sha256(doc)
    assert "z:Art.0" in in_scope(doc, system_classes=("high",))
    assert set(in_scope(doc, system_classes=("all",))) == {"z:Art.0"}


# --- the vendored bytes -----------------------------------------------------------

def test_the_vendored_snapshot_holds_the_bytes_it_claims(snap):
    """The digest has always been in the file and no consumer ever checked it. A hand
    edit to a vendored regulatory dataset is invisible otherwise, and the file's own
    header says a hand edit is exactly what it fears."""
    assert body_sha256(snap) == snap["body_sha256"]


def test_a_snapshot_that_does_not_match_its_digest_is_refused(tmp_path):
    import yaml

    doc = _mini()
    doc["obligations"]["x:Art.1"]["title"] = "edited by hand"
    p = tmp_path / "snapshot.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(AlmanacError) as exc:
        load_snapshot(p)
    assert "does not hold the bytes it claims" in str(exc.value)
    assert load_snapshot(p, verify=False), "and can be read knowingly"


def test_a_snapshot_with_no_digest_is_refused(tmp_path):
    import yaml

    doc = _mini()
    del doc["body_sha256"]
    p = tmp_path / "snapshot.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(AlmanacError, match="no body_sha256"):
        load_snapshot(p)


def test_a_missing_or_empty_snapshot_raises_rather_than_returning_nothing(tmp_path):
    """It RAISES, which inverts the convention a taxon table uses. An empty taxon
    table is a destination state; an empty obligation set never is -- "0 obligations
    covered" renders as a result, and the thing that produced it was a missing
    file."""
    with pytest.raises(AlmanacError, match="vendored, not fetched"):
        load_snapshot(tmp_path / "nope.yaml")
    p = tmp_path / "empty.yaml"
    p.write_text("obligations: {}\n")
    with pytest.raises(AlmanacError, match="no obligations"):
        load_snapshot(p)


def test_the_vendored_file_is_the_sha_this_package_recorded():
    """The only drift detector the design provides. Two vendored copies drift in
    opposite directions the first week nobody is watching, and then an id will not
    resolve and the fastest fix under deadline is to loosen something."""
    actual = hashlib.sha256(DEFAULT_SNAPSHOT.read_bytes()).hexdigest()
    assert actual == SNAPSHOT_SHA256, (
        "the vendored snapshot changed. If that was deliberate, update "
        "SNAPSHOT_SHA256 and re-vendor the report renderer's copy in the same "
        "change, because a report cites one and renders from the other")


@pytest.mark.skipif(not DOC_ENGINE.is_file(),
                    reason="the report renderer is a separate repository")
def test_the_renderer_holds_the_same_bytes():
    """Otherwise a report cites obligation text its own appendix does not contain."""
    assert hashlib.sha256(DOC_ENGINE.read_bytes()).hexdigest() == SNAPSHOT_SHA256


def test_the_licence_for_the_vendored_data_is_published():
    """It is CC BY 4.0 and the package is Apache-2.0. Redistributing it in a wheel
    needs the attribution the licence asks for, and a statement that the copy is
    unmodified."""
    root = Path(__file__).resolve().parents[1]
    data_licence = (root / "LICENSE-DATA").read_text()
    assert "CC BY 4.0" in data_licence
    assert "almanac" in data_licence
    assert "NO CHANGES WERE MADE" in data_licence
    notice = (root / "NOTICE").read_text()
    assert "LICENSE-DATA" in notice
    assert "OWASP" in notice, "the vocabulary is derived from OWASP material"
    packaged = (root / "pyproject.toml").read_text()
    assert "LICENSE-DATA" in packaged, "and it has to ship in the wheel"
    assert 'almanac/*.yaml' in packaged, "as does the data it covers"


# --- provenance -------------------------------------------------------------------

def test_the_vendored_snapshot_cites_a_revision_that_exists(snap):
    """It did not, for the whole life of this module until now.

    `generated_from` said `64579f63...-dirty`, and the commit before that recorded
    the literal string `working-tree`, so every report cited a tree state that no
    longer existed -- which `cite_as` correctly refused, falling back to the content
    digest. The catalogue has since been regenerated from a clean tree, so the
    fallback is no longer exercised by our own copy, and asserting that it still is
    would mean shipping a file we know to be wrong to keep a test green.
    """
    rev, fault = provenance(snap)
    assert not fault, f"the vendored copy should cite a real commit: {fault}"
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), rev
    assert cite_as(snap) == rev, "and it is cited as given, not by digest"


def test_an_untrustworthy_revision_is_still_refused_and_shown(snap):
    """The fallback the vendored copy no longer needs, tested where it belongs: on a
    snapshot that actually carries a bad revision, rather than on ours.

    Both halves matter. A revision that names nothing must not be cited, because a
    report rebuilt later has to be able to resolve what it cited -- and the claim
    must still be SHOWN, or nobody learns the dataset needs regenerating.
    """
    for bad in ("64579f63ca235c8d19cdc8dfcab16cec7e1cd03c-dirty", "working-tree-dirty"):
        doc = _mini(generated_from=bad)
        rev, fault = provenance(doc)
        assert rev == bad
        assert "edited working tree" in fault
        assert cite_as(doc) == "sha256:" + doc["body_sha256"][:16]

        cov = coverage(doc, [], attempted=set(), instruments=("x",))
        assert cov["almanac_version"].startswith("sha256:")
        assert cov["almanac_revision_claimed"] == bad
        assert "cited by digest" in render(cov)

    clean = _mini(generated_from="deadbeef")
    assert provenance(clean) == ("deadbeef", "")
    assert cite_as(clean) == "deadbeef"
    assert "almanac_provenance_fault" not in coverage(
        clean, [], attempted=set(), instruments=("x",))


def test_a_snapshot_with_no_revision_at_all_says_so():
    doc = _mini(generated_from="")
    rev, fault = provenance(doc)
    assert rev == ""
    assert "records no revision" in fault


# --- against the real catalogue and corpus ---------------------------------------

def test_the_catalogue_partitions_its_own_vocabulary(snap):
    """Measured: 101 = 93 mapped + 8 declined, exactly, with no overlap and no
    remainder. Asserted because `read_taxon` consults all three and the third state
    would mean something different if they ever stopped agreeing."""
    mapped = set(snap["crosswalk"]["techniques"])
    declined = set(snap["unmapped"]["techniques"])
    assert len(known_taxa(snap)) == 101
    assert len(mapped) == 93
    assert len(declined) == 8
    assert known_taxa(snap) == mapped | declined
    assert not (mapped & declined)
    assert snapshot_faults(snap) == [], "the vendored copy is internally consistent"


def test_two_thirds_of_what_a_client_carries_admits_no_test_evidence(snap):
    """The figure that belongs in the conversation rather than in a footnote: 44 of
    66 are review duties no run can evidence."""
    obs = snap["obligations"]
    assert len(obs) == 66
    review = [o for o in obs.values() if o.get("verifiable") == "review"]
    assert len(review) == 44
    assert len({o["id"] for o in obs.values()
                if o.get("verifiable") in ("test", "hybrid")}) == 22


def test_most_testable_obligations_have_no_technique_pointed_at_them(snap):
    """The ceiling on what any run can ever evidence, and it is not a run's fault. Of
    22 that admit test evidence only 13 have any technique mapped, and one of the
    nine that do not is `verifiable: test` -- directly testable, with nothing in the
    library pointed at it."""
    gap = unreachable_obligations(snap)
    assert len(gap) == 9
    assert gap["ai-act:Art.50(1)"] == "test"
    assert sum(1 for v in gap.values() if v == "hybrid") == 8


@pytest.mark.skipif(not (CORPUS / "cases").is_dir(),
                    reason="no seed-vault checkout; the corpus is private")
def test_a_fifth_of_the_corpus_lands_in_the_state_that_did_not_exist(snap):
    """82 mapped, 0 declined, 48 unknown over 130 taxa. The declined bucket being
    empty is why the tests above drive it with hand-written findings: a three-state
    implementation exercised only against this corpus would never run that branch."""
    from sunflower_thresh.corpus import load_corpus

    corpus = load_corpus(CORPUS)
    taxa = {c.taxon for c in corpus.cases.values() if c.taxon}
    cov = taxa_coverage(snap, taxa)
    assert cov.total == 130
    assert len(cov.mapped) == 82
    assert len(cov.declined) == 0
    assert len(cov.unknown) == 48
    # And every one of them is a variant inside a family the catalogue crosswalks,
    # which is what makes inheriting their obligations tempting and wrong.
    assert all(r.family_siblings for r in cov.unknown)
    assert not any(r.in_vocabulary for r in cov.unknown)


@pytest.mark.skipif(not (CORPUS / "cases").is_dir(),
                    reason="no seed-vault checkout; the corpus is private")
def test_an_unknown_technique_leaves_a_trace_in_the_appendix(snap):
    """The whole point. It contributes to no obligation row, so without the taxa
    section it would leave no trace at all -- and the consuming schema's
    `additionalProperties` is true, so the section travels without a schema change."""
    from sunflower_thresh.corpus import load_corpus

    corpus = load_corpus(CORPUS)
    taxa = {c.taxon for c in corpus.cases.values() if c.taxon}
    cov = coverage(snap, [], attempted=taxa, instruments=("ai-act",))
    rows = {r["taxon"]: r for r in cov["taxa"]["techniques"]}
    unknown = rows["llm01-indirect-screenshot"]
    assert unknown["state"] == "unknown"
    assert "never crosswalked" in unknown["note"]
    assert unknown["family"] == "llm01"
    assert unknown["family_crosswalked"], "and says where a human would look"
    assert "obligations" not in unknown, "while claiming none"


# --- the readout -----------------------------------------------------------------

def test_the_readout_groups_declines_by_the_reason_given():
    """The eight declines carry only three distinct paragraphs, because the catalogue
    generates them per family. Printed per technique, the same wall of text appears
    four times running and teaches the reader to skip it."""
    doc = _mini()
    doc["unmapped"]["techniques"]["fam-declined-two"] = {
        "reason": doc["unmapped"]["techniques"]["fam-declined"]["reason"]}
    doc["taxonomy"]["techniques"]["fam-declined-two"] = {}
    doc["body_sha256"] = body_sha256(doc)
    cov = coverage(doc, [], attempted={"fam-declined", "fam-declined-two"},
                   instruments=("x",))
    text = render(cov)
    reason = doc["unmapped"]["techniques"]["fam-declined"]["reason"]
    assert text.count(reason) == 1, "one paragraph, both techniques"
    assert "fam-declined, fam-declined-two" in text


def test_the_readout_refuses_to_infer_and_says_so():
    doc = _mini()
    cov = coverage(doc, [], attempted={"fam-variant"}, instruments=("x",))
    text = render(cov)
    assert "regulatory judgement, not a prefix match" in text


def test_a_snapshot_fault_is_printed_rather_than_rendered_over():
    doc = _mini(crosswalk={"techniques": {"fam-hollow": {"obligations": []}}})
    cov = coverage(doc, [], attempted=set(), instruments=("x",))
    assert cov["almanac_faults"]
    assert "no obligations" in render(cov)


def test_the_snapshot_lives_inside_the_package_and_not_beside_it():
    """Otherwise it resolves in a checkout and not in an installed wheel, and
    `thresh obligations` works for the person who wrote it and nobody else. The engine
    this was ported from kept it at `config/almanac/` two directories above the
    package, which is correct there and unshippable here."""
    import sunflower_thresh

    package = Path(sunflower_thresh.__file__).resolve().parent
    assert DEFAULT_SNAPSHOT.is_relative_to(package)
    assert DEFAULT_SNAPSHOT.is_file()


def test_the_revision_is_emitted_under_the_renderers_own_key_too(snap):
    """The catalogue was renamed from `trellis` to `almanac`, and these two keys are
    the only ones in this document the renderer DECLARES. Its schema allows extra
    properties, so renaming them alone would not fail validation -- the report would
    just silently stop citing the dataset revision it resolved against, which is the
    one fact a rebuilt report has to carry. Both pairs are emitted until the
    renderer reads the new names.
    """
    from sunflower_thresh.obligations import LEGACY_REVISION_KEYS

    cov = coverage(snap, [], attempted=set(), instruments=("ai-act",))
    for old, new in zip(LEGACY_REVISION_KEYS,
                        ("almanac_version", "almanac_sha256"), strict=True):
        assert old in cov, f"the renderer still reads {old}"
        assert cov[old] == cov[new], "and the two must not drift"


# --- a duty that does not bind yet is not a duty you failed ----------------------

def test_a_duty_that_does_not_bind_yet_is_outside_the_assurance_figure(snap):
    """Measured: THIRTY of sixty-six obligations do not apply yet, most from
    2027-12-02 after the Digital Omnibus moved them. They were reported exactly like
    the ones in force, and the worst of those rows was not `not-probed` but
    `probed-clean` -- which a client reads as "we checked this and you are fine"
    about a duty with no force for another year.

    For an Annex III high-risk system today that is 21 of 22 in scope, so the honest
    headline is that nothing measurable binds yet and everything measured is
    preparation.
    """
    from sunflower_thresh.obligations import in_force

    cov = coverage(snap, [], attempted={"jailbreak-crescendo"},
                   instruments=("ai-act",),
                   system_classes=("annex-iii-high-risk",))
    future = cov["not_yet_in_force"]
    assert future["count"] >= 15, future["count"]
    assert "preparation rather than assurance" in future["note"]

    rows = {r["obligation"]: r for r in cov["obligations"]}
    # Every row says whether it binds, so a reader never infers it from a date.
    assert all("in_force" in r for r in rows.values())
    binding = [r for r in rows.values() if r["in_force"]]
    assert cov["testable_total"] == sum(
        1 for r in binding if r["status"] != "review-only")
    assert cov["review_only_total"] == sum(
        1 for r in binding if r["status"] == "review-only")

    # And the one that read as assurance is marked.
    art15 = rows["ai-act:Art.15(1)"]
    assert art15["in_force"] is False
    assert not in_force(snap["obligations"]["ai-act:Art.15(1)"])


def test_the_report_can_speak_as_of_a_date(snap):
    """A report rebuilt later has to describe the duties as they stood when it was
    written, not as they stand now -- the same reason the dataset revision is cited.
    """
    import datetime

    from sunflower_thresh.obligations import in_force

    art15 = snap["obligations"]["ai-act:Art.15(1)"]
    assert str(art15["applies_from"]) == "2027-12-02"
    assert not in_force(art15, datetime.date(2026, 10, 19))
    assert in_force(art15, datetime.date(2027, 12, 2)), "the day it starts, it binds"
    assert in_force(art15, datetime.date(2028, 1, 1))

    later = coverage(snap, [], attempted=set(), instruments=("ai-act",),
                     system_classes=("annex-iii-high-risk",),
                     on=datetime.date(2028, 1, 1))
    assert "not_yet_in_force" not in later, "by then they all bind"
    assert later["testable_total"] > 0


def test_a_missing_or_unreadable_date_counts_as_binding():
    """The alternative is excusing a duty from a report on the strength of a blank
    field, which is the wrong direction to fail in."""
    from sunflower_thresh.obligations import in_force

    assert in_force({})
    assert in_force({"applies_from": ""})
    assert in_force({"applies_from": "not a date"})
    assert in_force({"applies_from": None})


def test_the_two_numbers_beside_each_other_count_the_same_set(snap):
    """It said "4 of those 0" once the figure started excluding duties that do not
    bind yet: the crosswalk-gap count was over every scoped obligation while the
    figure was over the binding ones. A reader stops trusting a page over a sentence
    like that."""
    cov = coverage(snap, [], attempted=set(), instruments=("ai-act",),
                   system_classes=("annex-iii-high-risk",))
    gap = cov.get("unreachable_by_any_technique") or {"count": 0}
    assert gap["count"] <= cov["testable_total"], (
        f"{gap['count']} cannot be 'of those {cov['testable_total']}'")
    text = render(cov)
    assert "of those 0" not in text


def test_a_scope_whose_duties_all_bind_reads_normally(snap):
    """The ordinary path must not acquire a caveat it does not need."""
    cov = coverage(snap, [], attempted=set(), instruments=("dora",),
                   system_classes=("important-entity",))
    assert "not_yet_in_force" not in cov
    text = render(cov)
    assert "DO NOT BIND YET" not in text
    assert "bind today" in text
