"""What a run's results mean against the obligations a client actually carries.

THE FOUR OUTCOME STATES, AND WHY THEY ARE FOUR.

An obligation can end a run without test evidence in three different ways, and a
report that renders them identically is misleading in the one direction that matters:

    tested        a technique that bears on it ran, and something stood up
    probed-clean  a technique that bears on it ran, and nothing stood up. A RESULT.
    not-probed    techniques bear on it and none of them ran. NOT a result.
    review-only   no test could ever bear on it; it is a document or a process

`probed-clean` and `not-probed` are the pair that has to stay apart. A client reading
"no findings against Article 15(5)" when every indirect-injection case was skipped for
want of a seeding sink has been told something false by omission, and `corpus.py`'s own
comment calls that the most dangerous failure in the system: an indirect-injection case
run without its seeding looks like a clean pass.

THE THIRD TAXON STATE, WHICH IS A DIFFERENT AXIS AND IS NEW.

Those four are per OBLIGATION. This module also reports, per TECHNIQUE, how trellis
stands on it -- and the engine this was ported from could not, because `for_taxon`
returned an empty list for two unrelated situations:

    mapped    trellis crosswalks it to at least one obligation
    declined  trellis has seen it and maps it to nothing, FOR A WRITTEN REASON
    unknown   trellis has never heard of it

A declined technique is a regulatory judgement someone made and recorded: eight of
them, each saying why a test result cannot be evidence about a documentation duty. An
unknown one is nobody's judgement at all. Rounding the second to "no obligations" is
how a run's most differentiating output disappears from the appendix without anybody
noticing it is missing.

Measured against seed-vault 379a45b and trellis's current snapshot: of 130 taxa in the
corpus, 82 are mapped (189 cases), 0 are declined, and 48 are unknown (55 cases). So
roughly a fifth of a full run lands in a state nothing previously reported.

WHAT THE UNKNOWN STATE DOES NOT DO.

It does not guess. All 48 are variants within a family trellis already crosswalks --
`llm01-indirect-screenshot` beside its `llm01-indirect-upload`, and so on -- so a
family's obligation set is sitting right there and inheriting it would be one line of
code. That line would have stem inventing a regulatory judgement, which is the whole
thing trellis's eight written declines exist to say you must not do. The readout names
the neighbouring family so a human can make the call, and claims nothing.

WHAT THIS DOES NOT DO AT ALL.

It does not filter findings. Every finding reaches the report whether or not it maps to
an obligation, and the mapping is an appendix rather than the spine. A finding that maps
to nothing is still a finding, and a report organised by obligation would quietly lose
the eight technique families the crosswalk deliberately maps to nothing.
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Vendored, not fetched. stem runs offline by design, and a regulatory dataset
#: resolved at run time would mean a report could not be rebuilt later from its own
#: bundle -- which is the only reason the copy exists. See LICENSE-DATA: the file is
#: CC BY 4.0 rather than Apache-2.0, and the copy is verbatim.
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "trellis" / "snapshot.yaml"

#: The sha256 of the whole vendored file, as the report renderer's copy holds it.
#:
#: Two vendored copies drift in opposite directions the first week nobody is watching:
#: this one moves because a run needed an obligation, doc-engine's does not because
#: nobody rebuilt a report. Then an id will not resolve and the fastest fix under
#: deadline is to loosen something. This constant is what makes that a test failure
#: here instead of a judgement call there.
SNAPSHOT_SHA256 = "155dc291392c64a1376d3aef349e892238cbc6c655fa3c911a8dc22f2a417ecd"

#: Keys that describe the snapshot rather than being its content. `body_sha256` is
#: computed over everything else, so it cannot cover these.
_META_KEYS = frozenset({"schema_version", "generated_from", "body_sha256"})

#: What `verifiable` can say, from the snapshot's own schema. `review` means no test
#: could ever bear on it. Measured: 44 review, 20 hybrid, 2 test of 66 -- so two
#: thirds of what a client carries is work no run can evidence, which belongs in the
#: conversation rather than in a footnote.
VERIFIABLE = ("test", "hybrid", "review")

#: The statuses doc-engine's control-mapping schema permits. A closed enum there, so
#: a fifth value invented here is a validation failure in the consuming repo.
STATUSES = ("tested", "probed-clean", "not-probed", "review-only")


class TrellisError(RuntimeError):
    """The snapshot is missing, empty, the wrong shape, or not the bytes it claims."""


class TaxonState(str, enum.Enum):
    """How trellis stands on one technique. Three, because two of them were one."""

    MAPPED = "mapped"
    DECLINED = "declined"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaxonRead:
    """Trellis's position on a technique, with whatever it said about it."""

    taxon: str
    state: TaxonState
    #: Obligations this technique produces evidence about. Empty unless MAPPED.
    obligations: tuple[str, ...] = ()
    #: Why trellis declined to map it. Present only when DECLINED, and never empty
    #: then: a decline without a reason is indistinguishable from an oversight.
    reason: str = ""
    #: Trellis's own rationale for the mapping, when MAPPED.
    rationale: str = ""
    #: The `llmNN` family, and whether trellis crosswalks anything in it. For an
    #: UNKNOWN technique this is the useful part: it says where a human would look.
    family: str = ""
    family_siblings: tuple[str, ...] = ()
    #: UNKNOWN only: whether trellis's VOCABULARY holds the technique even though
    #: neither the crosswalk nor the declines take a position on it. True means a
    #: trellis-side gap; False means the corpus names something trellis has never
    #: heard of. Different sentences, and different people to go and talk to.
    in_vocabulary: bool = False

    @property
    def speaks_to_obligations(self) -> bool:
        return bool(self.obligations)

    def __str__(self) -> str:
        if self.state is TaxonState.MAPPED:
            return f"{self.taxon}: {len(self.obligations)} obligation(s)"
        if self.state is TaxonState.DECLINED:
            return f"{self.taxon}: declined -- {self.reason}"
        kin = (f"; trellis crosswalks {len(self.family_siblings)} other "
               f"{self.family} technique(s)" if self.family_siblings else "")
        if self.in_vocabulary:
            return (f"{self.taxon}: in trellis's vocabulary with NO POSITION taken, "
                    f"so no obligation can be claimed for it{kin}")
        return (f"{self.taxon}: UNKNOWN to trellis, so no obligation can be claimed "
                f"for it{kin}")


def load_snapshot(path: str | Path | None = None, *, verify: bool = True) -> dict:
    """The vendored trellis data, checked against its own digest.

    RAISES rather than returning empty, which inverts the convention a taxon table
    uses. An empty taxon table is a destination state; an empty obligation set never
    is. "0 obligations covered" renders as a result, and the thing that produced it
    was a missing file.

    The digest check is new. `body_sha256` has always been in the snapshot and the
    bundler's docstring says it is there so "a consumer can assert it holds the bytes
    it thinks it holds" -- and no consumer asserted it. A hand edit to a vendored
    regulatory file is invisible otherwise, and the file's own header says a hand
    edit is exactly what it fears.
    """
    import yaml

    p = Path(path or DEFAULT_SNAPSHOT)
    if not p.exists():
        raise TrellisError(
            f"no trellis snapshot at {p}. It is vendored, not fetched: copy "
            f"dist/snapshot.yaml from the trellis checkout. Without it a control "
            f"mapping would render zero rows and look like a clean result")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not doc.get("obligations"):
        raise TrellisError(f"the trellis snapshot at {p} holds no obligations")
    if verify:
        declared = str(doc.get("body_sha256") or "")
        actual = body_sha256(doc)
        if not declared:
            raise TrellisError(
                f"the trellis snapshot at {p} carries no body_sha256, so the bytes "
                f"cannot be checked against what they claim to be")
        if declared != actual:
            raise TrellisError(
                f"the trellis snapshot at {p} does not hold the bytes it claims: "
                f"declared {declared[:16]}, computed {actual[:16]}. Either it was "
                f"edited by hand -- which its own header forbids, because an edit "
                f"here is invisible to the sources it came from -- or the copy is "
                f"damaged. Re-copy dist/snapshot.yaml from trellis")
    return doc


def body_sha256(doc: dict) -> str:
    """The digest over the snapshot's CONTENT, the way the bundler computes it.

    Canonical serialisation, and over the content only: a digest that moved when a
    key was reordered would be a digest nobody trusts and everyone bypasses.
    """
    body = {k: v for k, v in doc.items() if k not in _META_KEYS}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


def cite_as(snapshot: dict) -> str:
    """What a report should name as the dataset revision it resolved against.

    The snapshot's `generated_from` is not trustworthy for this and never has been.
    The copy vendored here says `...-dirty`, the commit before it recorded the
    literal string `working-tree`, and trellis's own test asserts only that the field
    is truthy while its CI deliberately refuses to diff the file -- so nothing on
    either side catches a revision that names a tree state no longer in existence.
    Meanwhile the content IS identifiable: `body_sha256` is over a canonical
    serialisation and is verified on load.

    So a usable revision is cited as given, and an unusable one is replaced by the
    digest, which is a real identity. The schema's own words for this field are that
    "a report rebuilt later must cite what it cited then", and a citation to a dirty
    tree at a commit cites nothing.
    """
    rev, fault = provenance(snapshot)
    if not fault:
        return rev
    return "sha256:" + str(snapshot.get("body_sha256") or "")[:16]


def provenance(snapshot: dict) -> tuple[str, str]:
    """(revision, what is wrong with it). The second is "" when nothing is.

    A vendored dataset exists so a report can be rebuilt from its own bundle, and a
    revision ending `-dirty` means the snapshot cannot be traced to any commit, so
    that promise does not hold for it. Reported rather than refused, because the
    content can still be perfectly good -- measured on the copy shipped here, it is
    byte-identical to what trellis's current HEAD produces, and only the label is
    unreproducible.
    """
    rev = str(snapshot.get("generated_from") or "")
    if not rev:
        return "", "the snapshot records no revision, so a report cannot say what it cited"
    if rev.endswith("-dirty"):
        return rev, ("built from an edited working tree, so it cannot be regenerated "
                     "from any commit. A report citing it cannot be rebuilt from its "
                     "own bundle, which is the only reason to vendor at all")
    return rev, ""


# ----- trellis's position on one technique ---------------------------------------

def _techniques(snapshot: dict, section: str) -> dict:
    return (snapshot.get(section) or {}).get("techniques") or {}


def known_taxa(snapshot: dict) -> frozenset[str]:
    """Every technique trellis has an opinion about, mapped or declined.

    Read from `taxonomy` rather than assembled from the other two, because taxonomy
    is the vocabulary and the others are positions on it. Measured, and asserted by a
    test: the three agree exactly, 101 = 93 mapped + 8 declined.
    """
    return frozenset(_techniques(snapshot, "taxonomy"))


def read_taxon(snapshot: dict, taxon: str) -> TaxonRead:
    """Trellis's position on `taxon`, as one of three states and never as silence.

    Three things here are not what the obvious implementation would do, and each is
    a way the snapshot's own tests allow data this has to survive.

    MAPPED keys on a NON-EMPTY obligation list, not on membership. Nothing in trellis
    forbids `obligations: []` -- its mapping tests assert that SOME technique maps to
    several, and pass for an entry that maps to none -- so `if taxon in crosswalk`
    would report `mapped` for a technique that speaks to nothing, reintroducing
    exactly the ambiguity these three states exist to remove.

    Appearing in BOTH the crosswalk and the declined set is contradictory, and
    nothing forbids that either, because the vocabulary test is satisfied by a
    technique in one or both. Mapped wins, deliberately and in writing, rather than
    by whichever branch happens to be first: a decline says "no obligation here is
    about this", and an actual list of obligations is the better evidence that
    somebody looked. `snapshot_faults` reports it so the contradiction is not merely
    resolved silently.

    UNKNOWN consults `taxonomy` as well. The vocabulary holds 101 techniques and the
    crosswalk and declines partition it exactly today, so a technique in the
    vocabulary with no position is a trellis-side gap rather than a corpus one, and
    the two deserve different sentences. The vocabulary's own header makes the point:
    membership is asserted by the consumers that resolve a taxon to something, not by
    the corpus that names one.
    """
    taxon = str(taxon or "")
    family = taxon.split("-", 1)[0] if "-" in taxon else taxon
    mapped = _techniques(snapshot, "crosswalk")
    declined = _techniques(snapshot, "unmapped")
    vocabulary = _techniques(snapshot, "taxonomy")

    entry = mapped.get(taxon) or {}
    obligations = tuple(entry.get("obligations") or ())
    if obligations:
        return TaxonRead(taxon=taxon, state=TaxonState.MAPPED,
                         obligations=obligations,
                         rationale=str(entry.get("rationale") or ""),
                         family=family)
    if taxon in declined:
        d = declined[taxon] or {}
        reason = str(d.get("reason") or "")
        return TaxonRead(taxon=taxon, state=TaxonState.DECLINED,
                         reason=reason or "trellis records no reason, and a decline "
                                          "without one cannot be told from an "
                                          "oversight",
                         family=family)
    siblings = tuple(sorted(
        t for t, e in mapped.items()
        if t.split("-", 1)[0] == family and ((e or {}).get("obligations") or ())))
    return TaxonRead(taxon=taxon, state=TaxonState.UNKNOWN, family=family,
                     family_siblings=siblings,
                     in_vocabulary=taxon in vocabulary)


def snapshot_faults(snapshot: dict) -> list[str]:
    """Contradictions in the snapshot that its own tests permit.

    Every one of these is empty against the copy vendored here, which is why they are
    a function rather than a load-time refusal: the data is good today and these are
    the shapes that would make a three-state answer quietly wrong tomorrow. `stem
    obligations` prints them, so the first bad bundle says so instead of rendering.
    """
    mapped = _techniques(snapshot, "crosswalk")
    declined = _techniques(snapshot, "unmapped")
    vocabulary = _techniques(snapshot, "taxonomy")
    out = []

    empty = sorted(t for t, e in mapped.items()
                   if not ((e or {}).get("obligations") or ()))
    if empty:
        out.append(f"{len(empty)} technique(s) are in the crosswalk with no "
                   f"obligations, which is neither mapped nor declined: "
                   f"{', '.join(empty[:6])}")
    both = sorted(set(mapped) & set(declined))
    if both:
        out.append(f"{len(both)} technique(s) are both crosswalked and declined, "
                   f"which contradicts; this reads them as mapped: "
                   f"{', '.join(both[:6])}")
    unpositioned = sorted(set(vocabulary) - set(mapped) - set(declined))
    if unpositioned:
        out.append(f"{len(unpositioned)} technique(s) are in the vocabulary with no "
                   f"position taken, neither crosswalked nor declined: "
                   f"{', '.join(unpositioned[:6])}")
    stray = sorted((set(mapped) | set(declined)) - set(vocabulary))
    if stray:
        out.append(f"{len(stray)} technique(s) carry a position but are not in the "
                   f"vocabulary: {', '.join(stray[:6])}")
    missing_reason = sorted(t for t, e in declined.items()
                            if not str((e or {}).get("reason") or "").strip())
    if missing_reason:
        out.append(f"{len(missing_reason)} decline(s) carry no reason, which cannot "
                   f"be told from an oversight: {', '.join(missing_reason[:6])}")
    return out


def unreachable_obligations(snapshot: dict, *, instruments=(),
                            system_classes=()) -> dict[str, str]:
    """{obligation id: its `verifiable`} for ones NO technique is crosswalked to.

    The ceiling on what any run can ever evidence, and it is not a run's fault.
    Measured on the current snapshot: of 22 obligations that admit test evidence,
    only 13 have any technique mapped to them, and one of the nine that do not is
    `verifiable: test` -- directly testable, with nothing in the library pointed at
    it. `coverage` has to label those `not-probed`, because the consuming schema's
    status enum is closed, so without this they are a crosswalk gap wearing an
    outcome-state label and an operator reads them as "we did not get to it".
    """
    reachable = {o for e in _techniques(snapshot, "crosswalk").values()
                 for o in ((e or {}).get("obligations") or ())}
    scoped = in_scope(snapshot, instruments=instruments,
                      system_classes=system_classes)
    return {oid: str(o.get("verifiable") or "")
            for oid, o in sorted(scoped.items())
            if o.get("verifiable") != "review" and oid not in reachable}


def for_taxon(snapshot: dict, taxon: str) -> list[str]:
    """The obligations a result for this technique produces evidence about.

    Kept as a list for the callers that only want the mapping. Use `read_taxon` to
    tell an empty list that means "trellis decided this proves nothing" from one that
    means "trellis has never heard of this".
    """
    return list(read_taxon(snapshot, taxon).obligations)


@dataclass
class TaxonCoverage:
    """How the taxa a run touched stand against trellis. The new section."""

    mapped: list[TaxonRead] = field(default_factory=list)
    declined: list[TaxonRead] = field(default_factory=list)
    unknown: list[TaxonRead] = field(default_factory=list)

    def add(self, read: TaxonRead) -> None:
        getattr(self, read.state.value).append(read)

    @property
    def total(self) -> int:
        return len(self.mapped) + len(self.declined) + len(self.unknown)

    def as_rows(self) -> list[dict]:
        """The three states as data, each saying what it is rather than how many."""
        out = []
        for read in sorted(self.mapped + self.declined + self.unknown,
                           key=lambda r: r.taxon):
            row: dict[str, Any] = {"taxon": read.taxon, "state": read.state.value}
            if read.state is TaxonState.MAPPED:
                row["obligations"] = list(read.obligations)
            elif read.state is TaxonState.DECLINED:
                row["reason"] = read.reason
            else:
                row["note"] = ("trellis has never crosswalked this technique, so no "
                               "obligation is claimed for it")
                if read.family_siblings:
                    row["family"] = read.family
                    row["family_crosswalked"] = list(read.family_siblings)
            out.append(row)
        return out


def taxa_coverage(snapshot: dict, taxa) -> TaxonCoverage:
    """Every technique a run touched, in one of three states."""
    cov = TaxonCoverage()
    for taxon in sorted({str(t) for t in taxa if t}):
        cov.add(read_taxon(snapshot, taxon))
    return cov


# ----- the obligations a client carries ------------------------------------------

def in_scope(snapshot: dict, *, instruments=(), system_classes=()) -> dict[str, dict]:
    """The obligations this client carries, which is what the denominator is over.

    Scoped by the engagement rather than by the catalogue: a Norwegian bank deploying a
    support assistant carries DORA and the Digital Security Act and is nowhere near
    Annex III, and a mapping that showed it every high-risk obligation as "not probed"
    would be a page of noise that teaches the reader to skip the appendix.
    """
    want_i, want_c = set(instruments or ()), set(system_classes or ())
    out = {}
    for oid, o in (snapshot.get("obligations") or {}).items():
        if want_i and o.get("instrument") not in want_i:
            continue
        classes = set(o.get("system_class") or ())
        if want_c and not (classes & want_c) and "all" not in classes:
            continue
        out[oid] = o
    return out


def coverage(snapshot: dict, findings: list[dict], *, attempted: set[str],
             skipped: dict[str, str] | None = None,
             instruments=(), system_classes=(),
             run_id: str = "") -> dict[str, Any]:
    """The control mapping for one run.

    `attempted` is the set of taxa the run actually measured, and `skipped` is
    {case id: why}, carried straight from the run manifest so the reason a technique
    did not run travels into the client's report in the words the run recorded.

    `findings` is whichever set the report will cite. When an engagement has a
    written-up `findings.yaml`, pass THOSE, because the appendix has to be
    cross-referenceable: a reader who sees an obligation marked tested needs to find
    the finding by the id printed in the findings table, and a run id appears nowhere
    in the report.

    That is also what lets a finding made by hand reach this mapping. A written-up
    finding carries its own `taxon:`, chosen by the tester, and nothing here cares
    whether the technique came from the corpus or from something somebody noticed at
    two in the morning. `attempted` still comes from the run, so a hand-written
    finding adds evidence without claiming its technique was systematically exercised.

    `run_id` is REQUIRED by the consuming schema and the port arrived without it, so
    every control mapping the old engine wrote would have failed validation on a
    required field. It is accepted here and emitted whether or not it is set, because
    an empty string fails loudly in the consumer while a missing key fails obscurely.
    """
    scoped = in_scope(snapshot, instruments=instruments, system_classes=system_classes)
    skipped = dict(skipped or {})
    crosswalk = _techniques(snapshot, "crosswalk")

    by_obligation: dict[str, list[str]] = {}
    for f in findings:
        for oid in for_taxon(snapshot, f.get("taxon") or ""):
            by_obligation.setdefault(oid, []).append(f.get("id") or "")

    rows = []
    for oid, o in sorted(scoped.items()):
        bearing = [t for t, e in crosswalk.items()
                   if oid in ((e or {}).get("obligations") or ())]
        row: dict[str, Any] = {
            "obligation": oid,
            "instrument": o.get("instrument"),
            "citation": o.get("citation"),
            "title": o.get("title", ""),
            "verifiable": o.get("verifiable"),
            "applies_from": o.get("applies_from"),
        }
        if o.get("verifiable") == "review":
            row["status"] = "review-only"
            row["evidence_expected"] = o.get("evidence_expected", "")
        elif not bearing:
            # testable in principle, but nothing in the corpus speaks to it yet
            row["status"] = "not-probed"
            row["reason"] = ("no technique in the library is crosswalked to this "
                             "obligation yet")
        elif by_obligation.get(oid):
            row["status"] = "tested"
            row["findings"] = sorted(set(by_obligation[oid]))
        elif any(t in attempted for t in bearing):
            row["status"] = "probed-clean"
            row["techniques_run"] = sorted(t for t in bearing if t in attempted)
        else:
            row["status"] = "not-probed"
            why = sorted(set(skipped.values())) if skipped else []
            row["reason"] = ("techniques bearing on this obligation were not run"
                             + (f": {why[0]}" if why else ""))
        if o.get("verification") == "unconfirmed":
            row["citation_unconfirmed"] = True
        rows.append(row)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    testable = [r for r in rows if r["status"] != "review-only"]

    # Every technique the run touched OR reported a finding for, which is not the
    # same set: a hand-written finding can carry a taxon no case in the library has.
    touched = set(attempted) | {f.get("taxon") for f in findings if f.get("taxon")}
    taxa = taxa_coverage(snapshot, touched)
    rev, rev_fault = provenance(snapshot)
    faults = snapshot_faults(snapshot)
    unreachable = unreachable_obligations(
        snapshot, instruments=instruments, system_classes=system_classes)

    out: dict[str, Any] = {
        "run_id": run_id,
        "trellis_version": cite_as(snapshot),
        "trellis_sha256": snapshot.get("body_sha256", ""),
        "scope": {"instruments": sorted(instruments or ()),
                  "system_classes": sorted(system_classes or ())},
        "counts": counts,
        # Two numbers, never one percentage. "14 of 31 that admit test evidence" and
        # "19 review-only, not tested" are different claims and a single figure
        # covering both is the specific way this appendix would mislead.
        "testable_total": len(testable),
        "testable_with_evidence": sum(1 for r in testable if r["status"] == "tested"),
        "review_only_total": counts.get("review-only", 0),
        "obligations": rows,
        # The new axis. `additionalProperties` is true on the consuming schema, so
        # this travels without a schema change -- and it has to travel, because a
        # technique trellis has never seen contributes to no row above and would
        # otherwise leave no trace in the appendix at all.
        "taxa": {
            "counts": {"mapped": len(taxa.mapped), "declined": len(taxa.declined),
                       "unknown": len(taxa.unknown)},
            "techniques": taxa.as_rows(),
        },
    }
    if rev_fault:
        out["trellis_revision_claimed"] = rev
        out["trellis_provenance_fault"] = rev_fault
    if faults:
        out["trellis_faults"] = faults
    if unreachable:
        # A crosswalk gap, not a run's gap. These are labelled `not-probed` above
        # because the consuming schema's status enum is closed, so without naming
        # them here an operator reads "we did not get to it" where the truth is that
        # nothing in the library is pointed at it and no run ever would.
        out["unreachable_by_any_technique"] = {
            "count": len(unreachable),
            "obligations": [{"obligation": k, "verifiable": v}
                            for k, v in unreachable.items()],
            "note": ("no technique in the crosswalk is mapped to these, so no run "
                     "can produce evidence about them. Closing that is a crosswalk "
                     "change, not a bigger run"),
        }
    return out


def render(cov: dict, *, title: str = "control mapping") -> str:
    """The readout. Three numbers that must not become one, and the third state."""
    from sunflower_stem import termstyle as ts

    counts = cov.get("counts") or {}
    taxa = cov.get("taxa") or {}
    tc = taxa.get("counts") or {}
    rows = list(taxa.get("techniques") or ())
    lines = [ts.header(f" {title}")]
    lines.append(f"  {ts.name('trellis')}        "
                 f"{ts.muted(cov.get('trellis_version') or 'unknown')}")
    if cov.get("trellis_provenance_fault"):
        claimed = cov.get("trellis_revision_claimed") or ""
        lines.append(f"  {ts.arrow()} {ts.warn(f'cited by digest: it says {claimed}')}")
        lines.append(f"    {ts.muted(cov['trellis_provenance_fault'])}")
    for fault in cov.get("trellis_faults") or ():
        lines.append(f"  {ts.arrow()} {ts.bad(fault)}")

    testable = int(cov.get("testable_total") or 0)
    evidenced = int(cov.get("testable_with_evidence") or 0)
    review = int(cov.get("review_only_total") or 0)
    lines.append("")
    lines.append(f"  {ts.name('obligations')}")
    for status in STATUSES:
        n = counts.get(status, 0)
        paint = ts.good if status == "tested" else (
            ts.muted if status == "review-only" else ts.warn)
        lines.append(f"    {paint(f'{n:4}')}  {status}")
    summary = (f"evidence for {evidenced} of {testable} that admit test evidence; "
               f"{review} review-only, which no run can speak to")
    lines.append(f"  {ts.muted(summary)}")

    gap = cov.get("unreachable_by_any_technique") or {}
    if gap:
        lines.append("")
        lines.append("  " + ts.warn(
            f"{gap['count']} of those {testable} have NO technique crosswalked to "
            f"them, so no run reaches them"))
        for row in (gap.get("obligations") or ())[:10]:
            lines.append(f"    {row['obligation']}  {ts.muted(row['verifiable'])}")
        lines.append("  " + ts.muted(gap.get("note", "")))

    lines.append("")
    lines.append(f"  {ts.name('techniques')}  "
                 f"{ts.muted('(how trellis stands on each)')}")
    for state in ("mapped", "declined", "unknown"):
        n = tc.get(state, 0)
        paint = ts.good if state == "mapped" else (
            ts.muted if state == "declined" else ts.warn)
        lines.append(f"    {paint(f'{n:4}')}  {state}")

    # Declines grouped by reason. The eight carry only three distinct paragraphs,
    # because trellis generates them per family rather than per technique, so a
    # per-technique rendering prints the same wall of text four times running and
    # teaches the reader to skip it.
    declined: dict[str, list[str]] = {}
    for row in rows:
        if row.get("state") == "declined":
            declined.setdefault(row.get("reason") or "", []).append(row["taxon"])
    if declined:
        lines.append("")
        lines.append("  " + ts.name("declined, by the reason trellis gave"))
        for reason, techs in sorted(declined.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"    {', '.join(techs)}")
            lines.append(f"      {ts.muted(reason)}")

    unknown = [r for r in rows if r.get("state") == "unknown"]
    if unknown:
        lines.append("")
        lines.append("  " + ts.warn(
            f"{len(unknown)} technique(s) trellis has never crosswalked. No "
            f"obligation is claimed for them,"))
        lines.append("  " + ts.warn(
            "  and none is inferred from a sibling: a mapping is a regulatory "
            "judgement, not a prefix match."))
        # Grouped by family, because 48 one-line rows is a page nobody reads and the
        # families are the unit somebody would actually close them in.
        fams: dict[str, list[str]] = {}
        for row in unknown:
            fams.setdefault(row.get("family") or "?", []).append(row["taxon"])
        for fam, techs in sorted(fams.items(), key=lambda kv: -len(kv[1])):
            kin = next((len(r.get("family_crosswalked") or ())
                        for r in unknown if r.get("family") == fam), 0)
            lines.append(f"    {ts.name(fam)}  {len(techs)} here"
                         + (ts.muted(f", {kin} crosswalked in trellis") if kin
                            else ts.muted(", and trellis crosswalks none in this "
                                          "family")))
            lines.append(f"      {ts.muted(', '.join(techs))}")
    return "\n".join(lines)
