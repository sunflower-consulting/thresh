"""Compare two runs: what closed, what is still there, what is new.

A retest whose output is a second report is not much use. The question a client asked
is what CHANGED, and the dangerous answer is "fixed", so most of this module is
arranged around being slow to say it.

FOUR DECISIONS, EACH ONE A WAY TO BE WRONG.

**Keyed on the case, not the finding id.** A finding's id is `{run_id}-{case_id}`, so
every finding in a retest has a new one and keying on it would report the entire
library as fixed-and-new. The join is the case id, which is this project's own
invariant -- join on case id, not path -- and it comes from `export._case_id` rather
than a second copy, because two definitions of "the same issue" drift and the failure
mode of a drifted key is a client told a vulnerability is fixed when it was renamed.

**The signature deliberately excludes the taxon.** It is the most tempting field to
add and the worst one: widening the identity is exactly how a retest reports a
vulnerability as fixed when somebody reclassified it. A case whose taxon was corrected
between runs is the same case and must compare as one.

**Fixed is the claim most likely to be wrong, so it is the most hedged.** A chatbot is
stochastic. A finding that landed 4 times in 10 and now lands 0 in 10 might be fixed
or might be the same assistant on a quieter day, so nothing is called closed unless
the case actually RAN and failed to reproduce. A case that was real and was not
attempted the second time is not fixed, it is `not_retested` -- a separate bucket,
and the one that catches a retest whose scope quietly narrowed: a capability switched
off, a credential expired, a placeholder nobody rebound.

**And closure needs the variants, not just the chain.** This is the part that exists
nowhere else. A client who patches the exact prompt in our report makes that one case
stop reproducing, and a diff that keys on the case alone calls it fixed. The honest
question is whether the TECHNIQUE still lands, so a finding closes only when the
proved case and its variants all fail to reproduce -- and when a variant does still
land, that is the most valuable row in the output: the fix blocks the phrasing we
demonstrated and the technique is untouched.

Measured on the corpus: 36 taxa carry more than one case, covering 150 of 244, so the
variant set is substantial rather than theoretical. `drafted_by.exemplar` is the
explicit, stronger signal and no case carries one yet, so it is preferred where it
appears and the shared taxon is the fallback.

THERE IS NO `closed` FINDING STATE AND THIS DOES NOT ADD ONE.

Closure is a property of a PAIR of runs, not of a finding, and `FindingState` has no
member for it. Putting one there would let a status be typed by hand into a findings
file, and a hand-typed closure is the thing a retest exists to replace. Every bucket
here is derived from the two bundles and nothing reads a status anybody wrote.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sunflower_thresh.obligations import CITABLE_STATES

#: States that mean "we established this was real". Imported rather than redefined:
#: the export seam already had to decide what a report may cite as evidence, and
#: "real enough to cite" and "real enough that its disappearance is news" are the
#: same question. A second tuple here is a second thing to keep true, and the engine
#: this was ported from used a narrower one, so the two would have disagreed about
#: every judge-proved finding -- `MAX_STATE_FOR[JUDGE]` caps those at REPRODUCED.
STOOD_UP = CITABLE_STATES

#: How much an attack success rate must move before it is worth a row of its own.
#: A mitigation that halved a rate did something, and a client who deployed one wants
#: to see it even when the verdict did not change.
MOVED = 0.2

#: Words too common to identify anything, for the fallback signature.
_NOISE = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "via",
    "is", "can", "be", "its", "it", "by", "from", "at", "as", "that", "this",
})
_WORDS = re.compile(r"[a-z0-9]+")


def case_of(finding: dict) -> str:
    """The CASE a finding measured, or "" when it measured none.

    Narrower than `export._case_id`, deliberately, and this is the difference. That
    one answers "what goes in the case_id column of a measurement row", where
    falling back to the finding's own id is a reasonable last resort. This one
    answers "is this finding's identity a case", and the finding's id is the one
    answer it must never accept: `Finding.id` is REQUIRED, so taking it would mean a
    hand-written finding never reaches the content signature at all -- and editing
    its id between runs would report the same issue as one fixed and one new.

    Caught in review. The test that was supposed to cover the fallback passed only
    because its fixture omitted `id`, which a real finding cannot.
    """
    if finding.get("corpus_id"):
        return str(finding["corpus_id"])
    cases = finding.get("test_cases") or []
    return str(cases[0]) if cases else ""


def finding_signature(finding: dict) -> str:
    """The identity that has to survive a retest.

    A case id when there is one, which for this package is almost always: a chat
    finding IS its case, and the case id is what a report cites and a reader greps
    for. Only a finding with no case at all -- something somebody noticed by hand and
    wrote up -- falls back to target, CWE, product, version and the significant words
    of its title.

    The taxon is not in either branch. See the module docstring: it is the field whose
    inclusion turns a reclassification into a fix.
    """
    cid = case_of(finding)
    if cid:
        return f"case:{cid}"
    parts = [str(finding.get(k) or "").strip().lower()
             for k in ("target", "cwe", "product", "version")]
    words = sorted({w for w in _WORDS.findall(str(finding.get("title") or "").lower())
                    if w not in _NOISE and len(w) > 2})
    blob = "|".join(parts + words)
    return "sig:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Row:
    """One signature's story across the two runs."""

    signature: str
    title: str = ""
    before: dict | None = None
    after: dict | None = None
    #: Variants of this case that the later run measured, and what became of them.
    #: Empty when the library offered none, which is a different fact from "none
    #: reproduced" and is why closure distinguishes the two.
    variants_run: tuple[str, ...] = ()
    variants_reproducing: tuple[str, ...] = ()
    #: Every variant the library knows of, and the ones the retest did not measure.
    #: Closure needs the second to be empty: a clean result for two of five variants
    #: says nothing about the other three, and the claim is about the technique.
    variants_known: tuple[str, ...] = ()
    variants_missing: tuple[str, ...] = ()

    @property
    def asr_before(self) -> float:
        return _asr(self.before)

    @property
    def asr_after(self) -> float:
        return _asr(self.after)

    @property
    def measured(self) -> bool:
        """Did either side carry a rate at all?

        A case proved by reproducing twice rather than by a success rate has no
        trial stats, and printing its rate as "0% -> 0%" implies a measurement that
        says the opposite of what happened: a finding that went from validated to
        rejected would read as no change.
        """
        return bool(_trials(self.before).get("trials")
                    or _trials(self.after).get("trials"))

    def rate(self, which: str) -> str:
        side = self.before if which == "before" else self.after
        if side is None:
            return "not attempted"
        if not _trials(side).get("trials"):
            # No rate to report, so report the verdict, which is what changed.
            return ("stood up" if str(side.get("state") or "") in STOOD_UP
                    else "did not reproduce")
        return f"{(self.asr_before if which == 'before' else self.asr_after):.0%}"

    @property
    def hint(self) -> str:
        return str((self.after or self.before or {}).get("severity_hint") or "")

    @property
    def case(self) -> str:
        return self.signature.removeprefix("case:")


def _trials(side: dict | None) -> dict:
    t = (side or {}).get("trials")
    return t if isinstance(t, dict) else {}


def _asr(side: dict | None) -> float:
    try:
        return float(_trials(side).get("asr") or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Diff:
    """The buckets, and the two ways the comparison itself can be invalid."""

    #: Proved before, the case ran again and did not reproduce, AND its variants were
    #: measured and none of them reproduced either. The only bucket that may be
    #: reported to a client as closed.
    closed: list[Row] = field(default_factory=list)
    #: The case stopped reproducing and a VARIANT still lands. The fix blocks the
    #: phrasing we demonstrated; the technique is untouched. Usually the most
    #: valuable row in the output, and it is deliberately not in `closed`.
    narrowed: list[Row] = field(default_factory=list)
    #: The case stopped reproducing and NO variant was measured, so there is nothing
    #: to distinguish a fixed technique from a fixed string. Not a claim either way.
    unverified: list[Row] = field(default_factory=list)
    still_present: list[Row] = field(default_factory=list)
    new: list[Row] = field(default_factory=list)
    #: Was real, and the retest did not attempt it. NOT fixed.
    not_retested: list[Row] = field(default_factory=list)
    #: Reproduced either way, but the rate moved. Movement without a verdict change.
    changed: list[Row] = field(default_factory=list)

    before_id: str = ""
    after_id: str = ""
    before_corpus: str = ""
    after_corpus: str = ""
    before_plan: str = ""
    after_plan: str = ""
    #: The two runs did not use the same library, so part of what follows compares
    #: two different questions.
    corpus_drifted: bool = False
    #: Worse, and for a reason specific to this output: a retest that selected a
    #: NARROWER set reports what it did not attempt as fixed if its manifest does not
    #: list the cases, and understates `not_retested`. Called out rather than
    #: inferred from the buckets.
    plan_drifted: bool = False

    @property
    def regressed(self) -> bool:
        return bool(self.new)

    @property
    def incomplete(self) -> bool:
        """Did this retest fail to attempt something previously established?

        Separate from `regressed`, because they are different facts and collapsing
        them makes both less useful. Why it matters more than it sounds: the buckets
        are counted, so a retest that quietly narrowed produces a cheerful "0 still
        present" beside a `not_retested` list nobody reads, and a release gated on
        the exit code goes green on exactly the run that proved nothing.
        """
        return bool(self.not_retested)

    @property
    def claims_closure(self) -> bool:
        return bool(self.closed)

    def counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in
                ("closed", "narrowed", "unverified", "still_present", "new",
                 "not_retested", "changed")}


# ----- variants ------------------------------------------------------------------

def variants_of(case_id: str, *, taxon: str, library: dict | None) -> set[str]:
    """Cases that stress the same technique by a different route.

    Two signals, and the weaker one is the one that works today. An explicit
    `drafted_by.exemplar` names the case a variant was varied from, which is the
    stronger claim because an author wrote it down along with which DIFFERENT defence
    it stresses -- and no case in the library carries one yet, so it is read where
    present and cannot be relied on.

    The fallback is the shared taxon: other active cases for the same technique.
    Measured on the corpus, 36 taxa carry more than one case covering 150 of 244, so
    this is a real set rather than a theoretical one. It is a weaker signal because
    two cases on one taxon may be two phrasings of one idea rather than two routes --
    which is why a surviving variant is reported as `narrowed` and never silently
    folded into `still_present`, and why `closed` says how many variants it rested on.
    """
    if not library:
        return set()
    out: set[str] = set()
    for cid, case in library.items():
        if cid == case_id:
            continue
        drafted = getattr(case, "drafted_by", None) or {}
        if isinstance(drafted, dict) and str(drafted.get("exemplar") or "") == case_id:
            out.add(cid)
            continue
        if taxon and getattr(case, "taxon", "") == taxon:
            out.add(cid)
    return out


# ----- the comparison -------------------------------------------------------------

def diff_runs(before_dir: str | Path, after_dir: str | Path, *,
              library: dict | None = None) -> Diff:
    """Compare two run bundles. `library` enables the variant half of closure.

    Without a library nothing can reach `closed`: a case that stopped reproducing
    goes to `unverified` instead, because with no variant set there is no way to tell
    a fixed technique from a fixed string. That is the conservative direction and it
    is the default, so forgetting to pass the library understates closure rather than
    claiming it.
    """
    from sunflower_thresh import runbundle

    before_m, before_f = runbundle.load(before_dir)
    after_m, after_f = runbundle.load(after_dir)

    out = Diff(
        before_id=str(before_m.get("run_id") or Path(before_dir).name),
        after_id=str(after_m.get("run_id") or Path(after_dir).name),
        before_corpus=str(before_m.get("corpus_sha") or ""),
        after_corpus=str(after_m.get("corpus_sha") or ""),
        before_plan=str(before_m.get("plan_hash") or ""),
        after_plan=str(after_m.get("plan_hash") or ""))
    out.corpus_drifted = bool(out.before_corpus and out.after_corpus
                              and out.before_corpus != out.after_corpus)
    # Only when BOTH runs recorded one: a retest of a run that predates plan hashes
    # has nothing to differ from, and flagging that cries wolf on every old bundle.
    out.plan_drifted = bool(out.before_plan and out.after_plan
                            and out.before_plan != out.after_plan)

    b = {finding_signature(f): f for f in before_f}
    a = {finding_signature(f): f for f in after_f}

    # What the retest actually attempted. A case absent from `after` could be one
    # that did not run, which is a different thing from one that ran and failed.
    after_skipped = set(after_m.get("skipped") or {})
    after_attempted = set(after_m.get("cases") or ()) - after_skipped
    # For the variant check, two sets and the difference between them matters.
    # `measured_now` is the cases the later run produced a FINDING for, in any state:
    # a case that ran and found nothing yields a rejected finding, so a case on the
    # attempt list with no finding at all did not complete -- gate-denied, errored,
    # or killed. Counting one of those as "re-tested without reproducing" would rest
    # a closure on a measurement nobody has.
    measured_now = {cid for cid in (case_of(f) for f in after_f) if cid}
    reproducing_now = {
        case_of(f) for f in after_f
        if str(f.get("state") or "") in STOOD_UP and case_of(f)}

    for sig, old in b.items():
        row = Row(signature=sig, title=str(old.get("title") or ""), before=old,
                  after=a.get(sig))
        was_real = str(old.get("state") or "") in STOOD_UP
        if row.after is None:
            if not was_real:
                continue        # a candidate that did not come back is not news
            cases = set(old.get("test_cases") or ())
            if row.case:
                cases.add(row.case)
            if after_attempted and not (cases & after_attempted):
                # Explicitly not attempted, and the bucket that catches a retest
                # whose scope narrowed. It also catches "cannot tell": a finding
                # recorded with no cases has nothing to match against the attempt
                # list. That lands here rather than in a closure bucket on purpose,
                # and the asymmetry is the point -- closure is a claim we make to a
                # client, this is a question we ask ourselves, and `incomplete` gates
                # the exit code on it so the conservative reading fails loudly.
                out.not_retested.append(row)
            else:
                _place_stopped(out, row, old, library, measured_now,
                               reproducing_now)
            continue
        now_real = str(row.after.get("state") or "") in STOOD_UP
        if was_real and not now_real:
            _place_stopped(out, row, old, library, measured_now, reproducing_now)
        elif now_real and not was_real:
            out.new.append(row)
        elif now_real and was_real:
            out.still_present.append(row)
            if row.measured and abs(row.asr_after - row.asr_before) >= MOVED:
                out.changed.append(row)

    for sig, now in a.items():
        if sig in b or str(now.get("state") or "") not in STOOD_UP:
            continue
        out.new.append(Row(signature=sig, title=str(now.get("title") or ""),
                           after=now))

    for bucket in (out.closed, out.narrowed, out.unverified, out.still_present,
                   out.new, out.not_retested, out.changed):
        bucket.sort(key=lambda r: (_HINT_ORDER.get(r.hint.lower(), 5), r.title))
    return out


#: Worst first, so the row a reader must not miss is the row they see.
_HINT_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _place_stopped(out: Diff, row: Row, old: dict, library: dict | None,
                   measured_now: set[str], reproducing_now: set[str]) -> None:
    """The case stopped reproducing. Closed, narrowed, or neither.

    Three outcomes from one observation, and collapsing them into "fixed" is the
    single most consequential thing this module could get wrong. A client who patched
    the exact prompt in our report makes this case stop; whether the TECHNIQUE stopped
    is a different question, and the variants are the only thing that answers it.

    A variant counts as re-tested only when the later run produced a finding for it,
    not merely when its manifest listed the case. Those differ exactly when a case
    was selected and did not complete, and resting a closure on one of those would
    quote a measurement that does not exist.
    """
    taxon = str(old.get("taxon") or "")
    siblings = variants_of(row.case, taxon=taxon, library=library)
    ran = sorted(siblings & measured_now)
    still = sorted(siblings & reproducing_now)
    row.variants_run = tuple(ran)
    row.variants_reproducing = tuple(still)
    row.variants_known = tuple(sorted(siblings))
    row.variants_missing = tuple(sorted(siblings - measured_now))
    if still:
        out.narrowed.append(row)
    elif siblings and not row.variants_missing:
        # The WHOLE set, measured and clean. One clean sibling is not closure: for a
        # technique with variants B and C, clean results for the case and B say
        # nothing about C, and the claim being made is about the technique rather
        # than about the two cases that happened to run. Caught in review; the rule
        # here was "any clean variant", which contradicted this module's own thesis.
        out.closed.append(row)
    else:
        out.unverified.append(row)


def render(d: Diff) -> str:
    """The readout. Ordered so the thing a reader must not miss comes first."""
    from sunflower_thresh import termstyle as ts

    lines = [ts.header(f" {d.before_id} -> {d.after_id}")]
    if d.plan_drifted:
        lines.append(" " + ts.bad("the two runs selected different cases: ")
                     + ts.muted(f"{_short(d.before_plan)} -> {_short(d.after_plan)}"))
        lines.append("   " + ts.muted(
            "a case the retest never attempted is reported as closed if its manifest "
            "does not list it, so read `not re-tested` before anything else"))
    if d.corpus_drifted:
        lines.append(" " + ts.bad("the two runs used different libraries: ")
                     + ts.muted(f"{_short(d.before_corpus)} -> "
                                f"{_short(d.after_corpus)}"))
        lines.append("   " + ts.muted(
            "a case added or edited between the runs makes part of this a comparison "
            "of two different questions"))
    lines.append("")
    for label, bucket, paint in (
            ("new", d.new, ts.bad),
            ("still present", d.still_present, ts.warn),
            ("narrowed", d.narrowed, ts.warn),
            ("not re-tested", d.not_retested, ts.warn),
            ("fix unverified", d.unverified, ts.muted),
            ("closed", d.closed, ts.good)):
        if not bucket:
            continue
        lines.append(f" {paint(f'{len(bucket):3}')}  {ts.name(label)}")
        for row in bucket[:12]:
            detail = f"{row.rate('before')} -> {row.rate('after')}"
            lines.append(f"      {row.case or row.title[:40]:42} {ts.muted(detail)}")
            if label == "narrowed":
                lines.append("        " + ts.warn(
                    f"still landing: {', '.join(row.variants_reproducing[:6])}"))
            elif label == "closed":
                lines.append("        " + ts.muted(
                    f"and all {len(row.variants_known)} variant(s) re-tested "
                    f"without reproducing"))
            elif label == "fix unverified" and row.variants_missing:
                lines.append("        " + ts.muted(
                    f"{len(row.variants_missing)} of {len(row.variants_known)} "
                    f"variant(s) were not re-tested: "
                    f"{', '.join(row.variants_missing[:4])}"))
        if len(bucket) > 12:
            lines.append(f"      {ts.muted(f'+{len(bucket) - 12} more')}")
        lines.append("")
    if d.narrowed:
        lines.append(" " + ts.warn(
            "`narrowed` means the phrasing in the report stopped working and the "
            "technique did not."))
        lines.append("   " + ts.muted(
            "A client who patched the prompt we showed them lands here, and it is "
            "not a fix."))
    if d.unverified:
        lines.append(" " + ts.muted(
            "`fix unverified` is not a claim: the case stopped reproducing and no "
            "variant was re-tested,"))
        lines.append("   " + ts.muted(
            "so a fixed technique cannot be told from a fixed string. Re-run with "
            "the variants selected."))
    if d.incomplete:
        lines.append(" " + ts.bad(
            "this retest did not attempt everything the first run established, so it "
            "cannot be read as a pass."))
    return "\n".join(lines)


def _short(digest: str) -> str:
    """`sha256:abc123...` -> 12 characters. The prefix is the same on every hash."""
    return digest.split(":", 1)[-1][:12]


def as_document(d: Diff) -> dict[str, Any]:
    """The retest as data, for a document generator.

    `closed` carries the variants it rested on, because a closure with no variants
    behind it is the claim this module exists to avoid making, and a reader of the
    file should be able to see what it was based on without rerunning anything.
    """
    def rows(bucket: list[Row]) -> list[dict]:
        out = []
        for r in bucket:
            row: dict[str, Any] = {"case": r.case or r.signature,
                                   "title": r.title,
                                   "before": r.rate("before"),
                                   "after": r.rate("after")}
            if r.variants_run:
                row["variants_retested"] = list(r.variants_run)
            if r.variants_reproducing:
                row["variants_still_reproducing"] = list(r.variants_reproducing)
            if r.variants_missing:
                row["variants_not_retested"] = list(r.variants_missing)
            out.append(row)
        return out

    doc: dict[str, Any] = {
        "before": d.before_id,
        "after": d.after_id,
        "counts": d.counts(),
        "incomplete": d.incomplete,
        "regressed": d.regressed,
    }
    if d.plan_drifted:
        doc["plan_drifted"] = True
    if d.corpus_drifted:
        doc["corpus_drifted"] = True
    for name in ("new", "still_present", "narrowed", "not_retested", "unverified",
                 "closed", "changed"):
        bucket = getattr(d, name)
        if bucket:
            doc[name] = rows(bucket)
    doc["note"] = (
        "Every bucket here is derived from the two run bundles. There is no `closed` "
        "finding state and nothing here reads a status anybody typed: a hand-typed "
        "closure is what a retest exists to replace."
    )
    return doc
