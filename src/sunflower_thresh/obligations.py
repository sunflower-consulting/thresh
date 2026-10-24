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

Those four are per OBLIGATION. This module also reports, per TECHNIQUE, how almanac
stands on it -- and the engine this was ported from could not, because `for_taxon`
returned an empty list for two unrelated situations:

    mapped    almanac crosswalks it to at least one obligation
    declined  almanac has seen it and maps it to nothing, FOR A WRITTEN REASON
    unknown   almanac has never heard of it

A declined technique is a regulatory judgement someone made and recorded: eight of
them, each saying why a test result cannot be evidence about a documentation duty. An
unknown one is nobody's judgement at all. Rounding the second to "no obligations" is
how a run's most differentiating output disappears from the appendix without anybody
noticing it is missing.

Measured against seed-vault 379a45b and almanac's current snapshot: of 130 taxa in the
corpus, 82 are mapped (189 cases), 0 are declined, and 48 are unknown (55 cases). So
roughly a fifth of a full run lands in a state nothing previously reported.

WHAT THE UNKNOWN STATE DOES NOT DO.

It does not guess. All 48 are variants within a family almanac already crosswalks --
`llm01-indirect-screenshot` beside its `llm01-indirect-upload`, and so on -- so a
family's obligation set is sitting right there and inheriting it would be one line of
code. That line would have thresh inventing a regulatory judgement, which is the whole
thing almanac's eight written declines exist to say you must not do. The readout names
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

from sunflower_thresh.schemas.finding import FindingState

#: Vendored, not fetched. thresh runs offline by design, and a regulatory dataset
#: resolved at run time would mean a report could not be rebuilt later from its own
#: bundle -- which is the only reason the copy exists. See LICENSE-DATA: the file is
#: CC BY 4.0 rather than Apache-2.0, and the copy is verbatim.
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "almanac" / "snapshot.yaml"

#: The sha256 of the whole vendored file, as the report renderer's copy holds it.
#:
#: Two vendored copies drift in opposite directions the first week nobody is watching:
#: this one moves because a run needed an obligation, doc-engine's does not because
#: nobody rebuilt a report. Then an id will not resolve and the fastest fix under
#: deadline is to loosen something. This constant is what makes that a test failure
#: here instead of a judgement call there.
SNAPSHOT_SHA256 = "4adbb3b7776caa27aa4d4b26e363058e84892faa0bbac0f3c743cf13c68214f1"

#: The two keys the report renderer declares, under the names it still uses. The
#: catalogue was renamed from `trellis` to `almanac`; this package emits both pairs
#: so a report built by the current renderer keeps citing the revision it resolved
#: against. Remove once the renderer's own schema names the new pair.
LEGACY_REVISION_KEYS = ("trellis_version", "trellis_sha256")

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


class AlmanacError(RuntimeError):
    """The snapshot is missing, empty, the wrong shape, or not the bytes it claims."""


class TaxonState(str, enum.Enum):
    """How almanac stands on one technique. Three, because two of them were one."""

    MAPPED = "mapped"
    DECLINED = "declined"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaxonRead:
    """Almanac's position on a technique, with whatever it said about it."""

    taxon: str
    state: TaxonState
    #: Obligations this technique produces evidence about. Empty unless MAPPED.
    obligations: tuple[str, ...] = ()
    #: Why almanac declined to map it. Present only when DECLINED, and never empty
    #: then: a decline without a reason is indistinguishable from an oversight.
    reason: str = ""
    #: Almanac's own rationale for the mapping, when MAPPED.
    rationale: str = ""
    #: The `llmNN` family, and whether almanac crosswalks anything in it. For an
    #: UNKNOWN technique this is the useful part: it says where a human would look.
    family: str = ""
    family_siblings: tuple[str, ...] = ()
    #: UNKNOWN only: whether almanac's VOCABULARY holds the technique even though
    #: neither the crosswalk nor the declines take a position on it. True means a
    #: almanac-side gap; False means the corpus names something almanac has never
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
        kin = (f"; almanac crosswalks {len(self.family_siblings)} other "
               f"{self.family} technique(s)" if self.family_siblings else "")
        if self.in_vocabulary:
            return (f"{self.taxon}: in almanac's vocabulary with NO POSITION taken, "
                    f"so no obligation can be claimed for it{kin}")
        return (f"{self.taxon}: UNKNOWN to almanac, so no obligation can be claimed "
                f"for it{kin}")


def load_snapshot(path: str | Path | None = None, *, verify: bool = True) -> dict:
    """The vendored almanac data, checked against its own digest.

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
        raise AlmanacError(
            f"no almanac snapshot at {p}. It is vendored, not fetched: copy "
            f"dist/snapshot.yaml from the almanac checkout. Without it a control "
            f"mapping would render zero rows and look like a clean result")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not doc.get("obligations"):
        raise AlmanacError(f"the almanac snapshot at {p} holds no obligations")
    if verify:
        declared = str(doc.get("body_sha256") or "")
        actual = body_sha256(doc)
        if not declared:
            raise AlmanacError(
                f"the almanac snapshot at {p} carries no body_sha256, so the bytes "
                f"cannot be checked against what they claim to be")
        if declared != actual:
            raise AlmanacError(
                f"the almanac snapshot at {p} does not hold the bytes it claims: "
                f"declared {declared[:16]}, computed {actual[:16]}. Either it was "
                f"edited by hand -- which its own header forbids, because an edit "
                f"here is invisible to the sources it came from -- or the copy is "
                f"damaged. Re-copy dist/snapshot.yaml from almanac")
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

    `generated_from` cannot be trusted blindly, and for the whole life of this
    module it was wrong: the vendored copy said `...-dirty` and the commit before it
    recorded the literal string `working-tree`, so every report cited a tree state
    that no longer existed. almanac's own test asserts only that the field is truthy
    and its CI deliberately refuses to diff the file, so nothing on either side
    caught it. The catalogue has since been regenerated from a clean tree and the
    copy here now names a real commit.

    The check stays, because the next regeneration can be dirty again and nothing
    upstream prevents it. A usable revision is cited as given; an unusable one is
    replaced by `body_sha256`, which is over a canonical serialisation, verified on
    load, and a real identity. The consuming schema's own words for this field are
    that "a report rebuilt later must cite what it cited then", and a citation to a
    dirty tree at a commit cites nothing.
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
    content can still be perfectly good: when the copy here carried a dirty label it
    was byte-identical to what almanac's HEAD produced, and only the label was
    unreproducible. Refusing to load it would have been the wrong trade.
    """
    rev = str(snapshot.get("generated_from") or "")
    if not rev:
        return "", "the snapshot records no revision, so a report cannot say what it cited"
    if rev.endswith("-dirty"):
        return rev, ("built from an edited working tree, so it cannot be regenerated "
                     "from any commit. A report citing it cannot be rebuilt from its "
                     "own bundle, which is the only reason to vendor at all")
    return rev, ""


# ----- almanac's position on one technique ---------------------------------------

def _techniques(snapshot: dict, section: str) -> dict:
    return (snapshot.get(section) or {}).get("techniques") or {}


def known_taxa(snapshot: dict) -> frozenset[str]:
    """Every technique almanac has an opinion about, mapped or declined.

    Read from `taxonomy` rather than assembled from the other two, because taxonomy
    is the vocabulary and the others are positions on it. Measured, and asserted by a
    test: the three agree exactly, 101 = 93 mapped + 8 declined.
    """
    return frozenset(_techniques(snapshot, "taxonomy"))


def read_taxon(snapshot: dict, taxon: str) -> TaxonRead:
    """Almanac's position on `taxon`, as one of three states and never as silence.

    Three things here are not what the obvious implementation would do, and each is
    a way the snapshot's own tests allow data this has to survive.

    MAPPED keys on a NON-EMPTY obligation list, not on membership. Nothing in almanac
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
    vocabulary with no position is a almanac-side gap rather than a corpus one, and
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
                         reason=reason or "almanac records no reason, and a decline "
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
    the shapes that would make a three-state answer quietly wrong tomorrow. `thresh
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
                            system_classes=(), on=None,
                            binding_only: bool = False) -> dict[str, str]:
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
            if o.get("verifiable") != "review" and oid not in reachable
            and (not binding_only or in_force(o, on))}


def for_taxon(snapshot: dict, taxon: str) -> list[str]:
    """The obligations a result for this technique produces evidence about.

    Kept as a list for the callers that only want the mapping. Use `read_taxon` to
    tell an empty list that means "almanac decided this proves nothing" from one that
    means "almanac has never heard of this".
    """
    return list(read_taxon(snapshot, taxon).obligations)


@dataclass
class TaxonCoverage:
    """How the taxa a run touched stand against almanac. The new section."""

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
                row["note"] = ("almanac has never crosswalked this technique, so no "
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

def in_force(obligation: dict, on=None) -> bool:
    """Does this duty bind yet, on the date the report speaks as of?

    Measured on the current catalogue: THIRTY of sixty-six obligations do not apply
    yet, most from 2027-12-02. Without this they were reported exactly like the ones
    in force -- and the worst of those rows was not `not-probed` but `probed-clean`,
    which a client reads as "we checked this and you are fine" about a duty with no
    force for another year.

    A missing or unparseable `applies_from` counts as IN FORCE, because the
    alternative is excusing a duty from the report on the strength of a blank field.
    """
    import datetime

    raw = str(obligation.get("applies_from") or "").strip()
    if not raw:
        return True
    on = on or datetime.date.today()
    if isinstance(on, datetime.datetime):
        on = on.date()
    try:
        starts = datetime.date.fromisoformat(raw[:10])
    except ValueError:
        return True
    return starts <= on


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


#: The engagement file the report renderer reads, and where the regulatory scope
#: already has a defined home.
ENGAGEMENT_FILE = "engagement.yaml"


@dataclass(frozen=True)
class EngagementScope:
    """Which regimes a client is in scope of, and as what kind of system.

    WHY THIS IS NOT ON THE TARGET INVENTORY, which is the obvious place and the
    wrong one. The inventory is the signed allowlist the POLICY GATE reads: what we
    may touch, and nothing else. The instruments a client carries have no bearing on
    what the gate permits, and three things argue against putting them there anyway.

    It is frozen and hash-covered, so a new field moves `content_hash()` for every
    inventory that already exists and an engagement that recorded a hash stops
    verifying -- which the gate reads as an unverified inventory and refuses every
    call under. It is read at RUN time and this is needed at EXPORT time, so the
    run would carry a fact it never uses. And one run can honestly be reported
    against more than one scope -- a client who wants an AI-Act view and a DORA
    view -- which baking it into the signed document forecloses.

    WHY NOT A FLAG EITHER. An engagement fact retyped per command is one that gets
    mistyped, and the instruments decide the denominator of the only coverage figure
    the report may state.

    So it is read from `engagement.yaml`, where the renderer's own schema already
    declares it, under `ai_system`, with a description that says what it is for:
    "Scopes the control mapping: without it every obligation in the catalogue
    renders as not-probed, which is a page of noise." Nothing had written it yet.
    """

    instruments: tuple[str, ...] = ()
    system_classes: tuple[str, ...] = ()
    #: provider, deployer or both. Carried because the catalogue keys obligations by
    #: it and a later filter will want it; nothing here reads it yet, and inventing
    #: a filter the renderer does not ask for would narrow a client's duties on this
    #: tool's opinion.
    role: str = ""

    def __bool__(self) -> bool:
        return bool(self.instruments or self.system_classes)


def read_engagement_scope(engagement_dir: str | Path) -> EngagementScope:
    """The `ai_system` block from an engagement's own file, or an empty scope.

    Empty rather than raising, and the caller decides. An engagement with no
    `ai_system` has not been scoped against the catalogue, which is a fact about the
    engagement; the schema makes the block optional and this honours that. What must
    not happen is rendering every obligation in the catalogue as `not-probed`,
    because a client reading a page of duties they do not carry learns to skip the
    appendix, and the one row that mattered goes with it.
    """
    import yaml

    path = Path(engagement_dir) / ENGAGEMENT_FILE
    if not path.is_file():
        return EngagementScope()
    # NOT caught. A file that is there and does not parse is a read failure, and
    # returning the empty scope made it indistinguishable from an engagement that
    # simply carries no `ai_system` block -- `__bool__` is False for both.
    #
    # What that cost, reproduced: a good export writes a 22-obligation control
    # mapping; the operator edits engagement.yaml and drops one `]`; the re-export
    # reads an empty scope, writes no mapping, and `export.write_run` then UNLINKS
    # the mapping already on disk because this run has none. `consolidate` removes
    # the engagement-root copy too. The client's entire regulatory appendix is
    # gone, exit 0 throughout, and the operator is told to "add `ai_system:
    # {instruments: [...], classes: [...]}`" to a file where that block is sitting
    # one bracket short.
    #
    # The unlink in export.py is right and stays: it is correct for
    # `--no-control-mapping` and for a genuinely de-scoped engagement. It was only
    # destructive because its caller could not tell "de-scoped" from "did not
    # parse". The `is_file()` guard above still answers the absent-file case this
    # function's docstring exists for.
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # These two stay caught, because they are scoping FACTS rather than read
    # failures: a file that parses to a non-dict, or an `ai_system` that is not a
    # mapping, really does declare no scope.
    block = (doc.get("ai_system") or {}) if isinstance(doc, dict) else {}
    if not isinstance(block, dict):
        return EngagementScope()
    return EngagementScope(
        instruments=tuple(str(x) for x in (block.get("instruments") or ())),
        system_classes=tuple(str(x) for x in (block.get("classes") or ())),
        role=str(block.get("role") or ""))


#: Finding states a report CITES as evidence an obligation was spoken to.
#:
#: Narrow on purpose, and the omission that matters is `rejected`. A rejected
#: finding is not a weak finding: it is the measurement that says the technique did
#: NOT cause the result, because the control arm fired as often as the attack. It is
#: still a measurement worth quoting -- `consolidate.QUOTABLE_STATES` includes it,
#: and the rate belongs in the measurements appendix -- but citing it under
#: `status: tested` would put "Article 15(1): tested, evidence: [five ids]" in a
#: client's regulatory appendix for a run that established nothing.
#:
#: `human_review` is out for the opposite reason: it is waiting for a person to
#: decide, and an appendix that cites it has decided on their behalf. Nothing is
#: lost by leaving both out, because their techniques are still in `attempted_taxa`,
#: so the obligations bearing on them read `probed-clean` -- we looked and nothing
#: stands -- which is the true statement and a better one.
#:
#: Derived from `FindingState`, not written out as strings. Both halves of this
#: pair used to be raw literals, and they drift in the one direction nothing
#: catches: a state RENAMED in the enum leaves the literal matching nothing, so an
#: obligation a finding speaks to silently stops being cited and the appendix
#: reports it `probed-clean`. Measured: leaving this set stale while syncing its
#: sibling left the suite 560/560 green with `coverage()` reporting the wrong
#: status. The enum is now the single statement, so a rename breaks at import.
#: REPRODUCED is NOT here, and that is the point of the set.
#:
#: `_decide` reaches REPRODUCED on `hits > 0`, BEFORE it looks at the control arm
#: at all, so the state means only "the oracle fired at least once". It then parks
#: a finding there on three outcomes that establish nothing: no control arm ran,
#: the arms did not separate (p >= ALPHA, or the rate under the case's threshold),
#: and the judge-only verdict that says in its own words that it needs human
#: review before it counts as validated.
#:
#: Citing those stamped `status: tested` on the obligation and printed their ids
#: as the evidence, so a client's regulatory appendix read "ai-act:Art.15(1):
#: tested, evidence: [two ids]" about a run whose own two findings say "nothing
#: here is falsified and nothing is established". That is the sentence this
#: function's docstring already refuses to write for a REJECTED finding, and the
#: argument is the same one.
#:
#: Nothing is lost. The measurement keeps its full verdict text in the
#: measurements appendix, because `consolidate.QUOTABLE_STATES` still carries
#: REPRODUCED -- a reader sees the number and the hedge together, which is the
#: honest place for it. The obligation reads `probed-clean`: we looked and nothing
#: stands, which is true.
CITABLE_STATES = frozenset({FindingState.VALIDATED.value,
                            FindingState.CONFIRMED.value})


def citable_findings(findings: list[dict]) -> list[dict]:
    """The findings a report would cite, out of everything a run wrote down.

    `coverage` takes "whichever set the report will cite", which for an engagement
    with a written-up `findings.yaml` is that file: every row in it is a finding
    somebody decided to report. A run bundle is not that. It holds every completed
    measurement, including the ones whose control arm fired as often as the attack,
    and handing those to `coverage` marks obligations `tested` on the strength of
    results that proved the technique was not what caused them.

    Measured on the bundled lab: a run at `--safety strong` finds nothing -- every
    finding comes back rejected -- and without this filter it reported four
    obligations as tested, citing five rejected measurements as the evidence.
    """
    return [f for f in findings
            if str(f.get("state") or "").lower() in CITABLE_STATES]


def attempted_taxa(findings: list[dict]) -> set[str]:
    """The techniques a run actually MEASURED, read off the findings it wrote.

    Recoverable from the bundle rather than needing a new manifest field, and the
    reason is a property of the sweep rather than a convenience: a finding exists if
    and only if a measurement completed. A case denied by the gate, skipped for want
    of a scorable oracle, or killed by an error produces no finding, and each of
    those is correctly absent here -- none of them established anything.

    The half that is easy to get wrong is the other one: a case that RAN and found
    nothing produces a REJECTED finding, not no finding, so its technique is counted
    and the obligations bearing on it can reach `probed-clean`. Deriving this from
    validated findings alone would collapse `probed-clean` into `not-probed` and
    turn every clean result into "nobody looked" -- which is the exact pair this
    whole module exists to keep apart.
    """
    return {str(f.get("taxon")) for f in findings if f.get("taxon")}


def coverage(snapshot: dict, findings: list[dict], *, attempted: set[str],
             skipped: dict[str, str] | None = None,
             instruments=(), system_classes=(),
             run_id: str = "", on=None) -> dict[str, Any]:
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
        # Whether it binds yet. On every row rather than only the future ones, so a
        # reader never has to infer it from a date they did not check.
        row["in_force"] = in_force(o, on)
        if not row["in_force"] and row["status"] == "not-probed":
            row["reason"] = (f"this duty does not apply until "
                             f"{o.get('applies_from')}. "
                             f"{row.get('reason', '')}").strip()
        rows.append(row)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    # The assurance figure is over duties that BIND. A denominator including ones
    # that start in a year makes "evidence for 3 of 7" the answer to a question
    # nobody asked, and the numerator cannot grow to meet it.
    binding = [r for r in rows if r["in_force"]]
    testable = [r for r in binding if r["status"] != "review-only"]
    future = [r for r in rows if not r["in_force"]]

    # Every technique the run touched OR reported a finding for, which is not the
    # same set: a hand-written finding can carry a taxon no case in the library has.
    touched = set(attempted) | {f.get("taxon") for f in findings if f.get("taxon")}
    taxa = taxa_coverage(snapshot, touched)
    rev, rev_fault = provenance(snapshot)
    faults = snapshot_faults(snapshot)
    # The crosswalk gap, over EVERY in-scope duty that admits test evidence, and
    # partitioned afterwards rather than filtered before.
    #
    # Computing it binding-only kept one sentence honest and deleted the report.
    # The sentence was real -- it read "4 of those 0" once the coverage figure
    # started excluding duties that do not bind yet -- but restricting the gap to
    # fix it meant that under `annex-iii-high-risk`, where every testable duty
    # binds on 2027-12-02, the gap was empty, the block was never emitted, and
    # FOUR duties with no technique pointed at them left no trace in the appendix.
    # The fourteen months before they bind is exactly when building a technique is
    # cheap, and the one reader who could act on that was the one told nothing.
    #
    # So the figure stays binding-only and the gap does not. They are different
    # questions: "what can you show today" and "what can this library never show".
    scoped_all = in_scope(snapshot, instruments=instruments,
                          system_classes=system_classes)
    gap_all = unreachable_obligations(
        snapshot, instruments=instruments, system_classes=system_classes,
        on=on, binding_only=False)
    unreachable = {k: v for k, v in gap_all.items()
                   if in_force(scoped_all[k], on)}
    gap_future = {k: v for k, v in gap_all.items() if k not in unreachable}

    out: dict[str, Any] = {
        "run_id": run_id,
        "almanac_version": cite_as(snapshot),
        "almanac_sha256": snapshot.get("body_sha256", ""),
        # The same two values under their old names, because these are the only
        # keys in this document the RENDERER declares, and its schema allows extra
        # properties -- so a rename alone would not fail validation, it would be
        # silently ignored and the report would lose the dataset revision it cited.
        # Both are emitted until the renderer reads the new names; drop these two
        # lines, and `LEGACY_REVISION_KEYS`, once it does.
        **{k: v for k, v in (("trellis_version", cite_as(snapshot)),
                             ("trellis_sha256",
                              snapshot.get("body_sha256", "")))},
        "scope": {"instruments": sorted(instruments or ()),
                  "system_classes": sorted(system_classes or ())},
        "counts": counts,
        # Two numbers, never one percentage. "14 of 31 that admit test evidence" and
        # "19 review-only, not tested" are different claims and a single figure
        # covering both is the specific way this appendix would mislead.
        "testable_total": len(testable),
        "testable_with_evidence": sum(1 for r in testable if r["status"] == "tested"),
        "review_only_total": sum(1 for r in binding
                                 if r["status"] == "review-only"),
        "obligations": rows,
        # The new axis. `additionalProperties` is true on the consuming schema, so
        # this travels without a schema change -- and it has to travel, because a
        # technique almanac has never seen contributes to no row above and would
        # otherwise leave no trace in the appendix at all.
        "taxa": {
            "counts": {"mapped": len(taxa.mapped), "declined": len(taxa.declined),
                       "unknown": len(taxa.unknown)},
            "techniques": taxa.as_rows(),
        },
    }
    if future:
        # Not dropped: a duty starting in fourteen months is worth preparing for,
        # and a client told about it late has a worse problem than one told early.
        # Reported APART, because folding it into the assurance figure claims a
        # check that could not have been meaningful.
        out["not_yet_in_force"] = {
            "count": len(future),
            # Evidence against a duty that does not bind yet is still evidence, and
            # zeroing it hides real work: a client with a proven weakness in
            # something that becomes a duty in December 2027 wants to know now. It
            # is counted here rather than in the figure above, because the figure
            # answers "what can you show today" and this answers "what is coming".
            "with_evidence": sum(1 for r in future if r["status"] == "tested"),
            "obligations": [{"obligation": r["obligation"],
                             "applies_from": r.get("applies_from"),
                             "verifiable": r.get("verifiable"),
                             "status": r["status"],
                             **({"findings": r["findings"]}
                                if r.get("findings") else {})} for r in future],
            "note": ("these are in scope and do not bind yet, so they are outside "
                     "the coverage figure above. Measuring against them now is "
                     "preparation rather than assurance -- and evidence found "
                     "against one is still evidence, reported here so it is not "
                     "lost in a figure it does not belong to."),
        }
    if rev_fault:
        out["almanac_revision_claimed"] = rev
        out["almanac_provenance_fault"] = rev_fault
    if faults:
        out["almanac_faults"] = faults
    if gap_all:
        # A crosswalk gap, not a run's gap. These are labelled `not-probed` above
        # because the consuming schema's status enum is closed, so without naming
        # them here an operator reads "we did not get to it" where the truth is that
        # nothing in the library is pointed at it and no run ever would.
        #
        # Emitted whenever EITHER half is non-empty, so a scope whose every gap is
        # in the future still reports one. `count` stays the binding half, because
        # the sentence it feeds names the same denominator as the figure above.
        block: dict[str, Any] = {
            "count": len(unreachable),
            "obligations": [{"obligation": k, "verifiable": v}
                            for k, v in unreachable.items()],
            "note": ("no technique in the crosswalk is mapped to these, so no run "
                     "can produce evidence about them. Closing that is a crosswalk "
                     "change, not a bigger run."),
        }
        if gap_future:
            # Reported apart for the same reason `not_yet_in_force` is: folding a
            # duty that does not bind yet into today's figure claims a check that
            # could not have been meaningful. Kept, rather than dropped, because
            # this is the half with lead time on it.
            block["not_yet_binding"] = {
                "count": len(gap_future),
                "obligations": [
                    {"obligation": k, "verifiable": v,
                     "applies_from": scoped_all[k].get("applies_from")}
                    for k, v in gap_future.items()],
                # Says only what the rendered sentence does not. It used to repeat
                # "these do not bind yet and nothing bears on them", which is the
                # sentence the reader has just finished, and a client report that
                # says a thing twice reads as one nobody proofread.
                "note": ("building a technique before the date is preparation; "
                         "after it, the same gap is in an appendix."),
            }
        out["unreachable_by_any_technique"] = block
    return out


def render(cov: dict, *, title: str = "control mapping") -> str:
    """The readout. Three numbers that must not become one, and the third state."""
    from sunflower_thresh import termstyle as ts

    counts = cov.get("counts") or {}
    taxa = cov.get("taxa") or {}
    tc = taxa.get("counts") or {}
    rows = list(taxa.get("techniques") or ())
    lines = [ts.header(f" {title}")]
    lines.append(f"  {ts.name('almanac')}        "
                 f"{ts.muted(cov.get('almanac_version') or 'unknown')}")
    if cov.get("almanac_provenance_fault"):
        claimed = cov.get("almanac_revision_claimed") or ""
        lines.append(f"  {ts.arrow()} {ts.warn(f'cited by digest: it says {claimed}')}")
        lines.append(f"    {ts.muted(cov['almanac_provenance_fault'])}")
    for fault in cov.get("almanac_faults") or ():
        lines.append(f"  {ts.arrow()} {ts.bad(fault)}")

    testable = int(cov.get("testable_total") or 0)
    evidenced = int(cov.get("testable_with_evidence") or 0)
    review = int(cov.get("review_only_total") or 0)
    lines.append("")
    lines.append(f"  {ts.name('obligations')}"
                 + ts.muted("  (every row, including ones that do not bind yet)"))
    for status in STATUSES:
        n = counts.get(status, 0)
        paint = ts.good if status == "tested" else (
            ts.muted if status == "review-only" else ts.warn)
        lines.append(f"    {paint(f'{n:4}')}  {status}")
    if testable:
        summary = (f"evidence for {evidenced} of {testable} that admit test "
                   f"evidence AND bind today; {review} review-only, which no run "
                   f"can speak to")
    else:
        summary = (f"nothing in this scope both admits test evidence and binds "
                   f"today; {review} review-only, which no run can speak to")
    lines.append(f"  {ts.muted(summary)}")

    future = cov.get("not_yet_in_force") or {}
    if future:
        lines.append("")
        proven = int(future.get("with_evidence") or 0)
        lines.append("  " + ts.warn(
            f"{future['count']} obligation(s) in scope DO NOT BIND YET, and are "
            f"outside the figure above"))
        if proven:
            lines.append("  " + ts.bad(
                f"{proven} of them already "
                f"{'has' if proven == 1 else 'have'} evidence against "
                f"{'it' if proven == 1 else 'them'}. Not a finding about today's "
                f"compliance, and worth acting on before the date"))
        soonest = sorted(future.get("obligations") or (),
                         key=lambda r: (r.get("status") != "tested",
                                        str(r.get("applies_from") or "")))
        for row in soonest[:6]:
            mark = ts.bad(" evidence") if row.get("status") == "tested" else ""
            lines.append(f"    {row.get('applies_from')}  {row['obligation']}"
                         f"  {ts.muted(str(row.get('verifiable') or ''))}{mark}")
        if len(soonest) > 6:
            lines.append(f"    {ts.muted(f'+{len(soonest) - 6} more')}")
        lines.append("  " + ts.muted(
            "measuring against these now is preparation rather than assurance, and "
            "a client told late about a duty has the worse problem"))

    gap = cov.get("unreachable_by_any_technique") or {}
    if gap:
        lines.append("")
        # Guarded, because the block is now emitted when only the future half is
        # non-empty: "0 of those 0 have NO technique" is the sentence this whole
        # section exists to avoid printing.
        if gap.get("count"):
            lines.append("  " + ts.warn(
                f"{gap['count']} of those {testable} have NO technique crosswalked "
                f"to them, so no run reaches them"))
            for row in (gap.get("obligations") or ())[:10]:
                lines.append(f"    {row['obligation']}  {ts.muted(row['verifiable'])}")
            lines.append("  " + ts.muted(gap.get("note", "")))
        pending = gap.get("not_yet_binding") or {}
        if pending:
            lines.append("  " + ts.warn(
                f"{pending['count']} obligation(s) that do not bind yet also have "
                f"NO technique crosswalked to them"))
            soonest = sorted((pending.get("obligations") or ()),
                             key=lambda r: str(r.get("applies_from") or ""))
            for row in soonest[:10]:
                lines.append(f"    {row.get('applies_from')}  {row['obligation']}"
                             f"  {ts.muted(str(row.get('verifiable') or ''))}")
            if len(soonest) > 10:
                lines.append(f"    {ts.muted(f'+{len(soonest) - 10} more')}")
            lines.append("  " + ts.muted(pending.get("note", "")))

    lines.append("")
    lines.append(f"  {ts.name('techniques')}  "
                 f"{ts.muted('(how almanac stands on each)')}")
    for state in ("mapped", "declined", "unknown"):
        n = tc.get(state, 0)
        paint = ts.good if state == "mapped" else (
            ts.muted if state == "declined" else ts.warn)
        lines.append(f"    {paint(f'{n:4}')}  {state}")

    # Declines grouped by reason. The eight carry only three distinct paragraphs,
    # because almanac generates them per family rather than per technique, so a
    # per-technique rendering prints the same wall of text four times running and
    # teaches the reader to skip it.
    declined: dict[str, list[str]] = {}
    for row in rows:
        if row.get("state") == "declined":
            declined.setdefault(row.get("reason") or "", []).append(row["taxon"])
    if declined:
        lines.append("")
        lines.append("  " + ts.name("declined, by the reason almanac gave"))
        for reason, techs in sorted(declined.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"    {', '.join(techs)}")
            lines.append(f"      {ts.muted(reason)}")

    unknown = [r for r in rows if r.get("state") == "unknown"]
    if unknown:
        lines.append("")
        lines.append("  " + ts.warn(
            f"{len(unknown)} technique(s) almanac has never crosswalked. No "
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
                         + (ts.muted(f", {kin} crosswalked in almanac") if kin
                            else ts.muted(", and almanac crosswalks none in this "
                                          "family")))
            lines.append(f"      {ts.muted(', '.join(techs))}")
    return "\n".join(lines)
