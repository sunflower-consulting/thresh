"""Rented breadth, joined to owned proof. A hit from another tool is a LEAD.

garak, promptfoo and PyRIT carry hundreds of probes and the multi-turn generators,
maintained by NVIDIA and Microsoft for nothing, and reimplementing that would be
foolish. What they produce is a place to look, not a thing that is true. A tool that
says "FAIL: jailbreak" has told us where to point a measurement; it has not measured
anything against a control, on a channel the target does not own, inside an
authorisation anybody signed.

So this module reads their output and turns each hit into exactly one of three
records, and the point of it is the LAST one:

    covered    an active case in the library already exercises this technique
    promoted   nothing covers it, so a draft case is proposed for an author
    dismissed  it cannot become either, and the reason is written down

With a 244-case library `covered` is the common outcome and is enormously cheaper
than authoring. `dismissed` is the one that makes the whole thing honest, because a
lead that is quietly dropped is a scan somebody paid for whose result nobody can
account for.

A LEAD CANNOT BECOME A FINDING, AND THAT IS STRUCTURAL.

`Lead` has no field for a state and no field for a severity. Not "it is ignored" --
there is nowhere to put one, the way `ChatCase` has nowhere to put a claim of
authorship. The tool's own verdict survives only as `flagged`, a string in the
tool's words, clearly attributed to the tool. `consolidate.NEVER_READ` excludes
`leads/` in code, so a lead directory cannot reach a consolidated measurement even
by accident, and a test asserts no triage document contains a finding state.

WHY THE READERS REFUSE RATHER THAN RETURN NOTHING.

Neither format is stable and neither can be checked from the file alone.

garak writes JSONL with an `entry_type` per row and NO SCHEMA VERSION ANYWHERE. Its
release notes record that the format of `eval` and `digest` entries changed and that
a `probe_summary` type was added, and its published reference documents `passed`,
`total_evaluated` and `nones` on an eval entry while documenting neither `probe` nor
`detector` -- so the fields an ingester most wants are the ones least pinned down.

promptfoo does carry a top-level `version`, which is checkable, but its own
documentation says `gradingResult` and `componentResults` may be absent, and its
maintainers note that some rows carry no `metadata.pluginId` at all, so attribution
is inconsistent by their own account.

An ingester hard-coded against one version of either reads ZERO leads from the other
and reports a clean scan. That is the short-load defect from `corpus.py` wearing a
different hat, and it is refused the same way: a report with rows in it that yields
no interpretable lead raises, naming the entry types and keys it actually saw.

The field names here come from the published documentation of both tools and have
NOT been checked against a report produced by a live run, because neither tool is
installed here. That is exactly why the reader is shape-detecting and refuses what
it does not recognise: the first real report either works or says what it saw, which
is the only pair of outcomes worth having when the format cannot be verified in
advance.

WHAT THIS DOES NOT DO.

It does not run anything. stem has no tool catalog and no business invoking a scanner
against a client's endpoint; the operator runs the tool and hands over the file. The
file is therefore UNTRUSTED INPUT -- it holds attacker-crafted prompts and a model's
replies to them -- so nothing read from it is executed, interpolated into a payload,
or allowed to name a case id that collides with the library.

And it never invents a taxon. A garak probe name is not a trellis technique and
neither is a promptfoo plugin id; resolving one to the other is a judgement, recorded
in `PROBE_TAXA` below where it can be argued with in a diff. An unmapped probe is
dismissed by name rather than guessed at, for the reason the obligation catalogue
gives for its own eight written declines.
"""
from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


#: What a lead can become. Three, and the third is not a failure mode: a lead that
#: cannot be covered or promoted still has to be accounted for.
class Outcome(str, enum.Enum):
    COVERED = "covered"
    PROMOTED = "promoted"
    DISMISSED = "dismissed"


#: Tools whose output this reads. Named, because a reader is written per format and
#: a fourth tool is a reader rather than a configuration value.
SOURCES = ("garak", "promptfoo")

#: garak entry types this knows. `attempt` and `eval` are documented in its stable
#: reference; `probe_summary` appears in later releases and is read when present.
#: A row whose `entry_type` is outside this set is counted as uninterpreted rather
#: than ignored, because the next format change arrives as exactly that.
GARAK_ENTRY_TYPES = frozenset({
    "start_run setup", "init", "attempt", "eval", "digest", "probe_summary",
    "completion", "config",
})

#: garak attempt statuses, from its reference: 0 not sent, 1 response but not
#: evaluated, 2 response and evaluation. Only 2 can carry a verdict, so only 2 can
#: produce a lead; 0 and 1 are recorded as uninterpretable-for-triage, not dropped.
GARAK_EVALUATED = 2

#: The promptfoo output version this reader was written against. A different major
#: is refused rather than parsed hopefully.
PROMPTFOO_VERSION = 3

#: A case id this module may mint. Deliberately unlike the corpus's own
#: `INJ-D-0001`: a promoted lead is a DRAFT nobody has reviewed, and giving it a
#: library-shaped id is how it ends up cited as one.
LEAD_CASE_PREFIX = "LEAD-"
_SAFE_SLUG = re.compile(r"[^a-z0-9]+")


class LeadError(ValueError):
    """The report cannot be read, or is not the format it claims."""


@dataclass(frozen=True)
class Lead:
    """One place another tool says is worth looking at.

    No state and no severity, and their absence is the design rather than an
    omission: there is nowhere here to put the claim that something is true. The
    tool's verdict survives as `flagged`, in the tool's own words, attributed.
    """

    source: str
    #: The tool's own identifier for what it ran, verbatim. Not interpreted.
    probe: str
    #: What the tool said, in its words. A string, never a boolean or a state: a
    #: boolean invites `if lead.passed`, and that is the import this forbids.
    flagged: str
    #: Where in the file, so a reader can go and look at the row themselves.
    row: int
    #: The trellis technique this probe was RESOLVED to, or "" when none is mapped.
    taxon: str = ""
    #: A short excerpt for a human triaging, truncated and never re-sent anywhere.
    detail: str = ""

    def __post_init__(self) -> None:
        for name in ("state", "severity", "passed", "success"):
            if name in self.__dataclass_fields__:        # pragma: no cover - guard
                raise AssertionError(
                    f"`Lead` must not carry {name!r}: a scanner's verdict is not a "
                    f"finding state, and a field for it is how it becomes one")


@dataclass
class Triage:
    """What was decided about one lead, and why. Every lead gets exactly one."""

    lead: Lead
    outcome: Outcome
    #: COVERED: the active cases that already exercise this technique.
    cases: tuple[str, ...] = ()
    #: PROMOTED: the draft id minted for an author to accept or throw away.
    case_id: str = ""
    #: DISMISSED: why, in a sentence somebody can disagree with.
    reason: str = ""

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source": self.lead.source,
            "probe": self.lead.probe,
            "row": self.lead.row,
            "flagged": self.lead.flagged,
            "outcome": self.outcome.value,
        }
        if self.lead.taxon:
            row["taxon"] = self.lead.taxon
        if self.outcome is Outcome.COVERED:
            row["cases"] = list(self.cases)
        elif self.outcome is Outcome.PROMOTED:
            row["case_id"] = self.case_id
            row["status"] = "draft"
            row["note"] = ("a stub for an author. It carries no control arm and no "
                           "oracle, because deleting the words a tool guessed were "
                           "the attack falsifies nothing")
        else:
            row["reason"] = self.reason
        if self.lead.detail:
            row["detail"] = self.lead.detail
        return row


# ----- the judgement, written where it can be argued with ------------------------

#: garak probe family / promptfoo plugin id -> trellis technique.
#:
#: A judgement per line, and the reason it is a table rather than a function is that
#: a function would invite a rule. There is no rule: `dan` and `encoding` are garak's
#: names for things the corpus calls something else, and the mapping is somebody's
#: opinion about what a probe actually exercises. An entry here is reviewable in a
#: diff; a regex over probe names would not be.
#:
#: Deliberately short. Everything absent is DISMISSED BY NAME, which is a cheap,
#: honest outcome and a prompt to add a line here -- where guessing from a prefix
#: would silently attribute a client's lead to the wrong technique and, through the
#: obligation crosswalk, to the wrong regulatory duty.
PROBE_TAXA: dict[str, str] = {
    # garak probe modules, by the name its reports use for the family.
    "dan": "jailbreak-dan",
    "encoding": "llm01-encoded-payload",
    "promptinject": "llm01-direct-override",
    "latentinjection": "llm01-indirect-doc",
    "xss": "llm05-markdown-exfil",
    "leakreplay": "llm02-training-data-echo",
    # promptfoo redteam plugin ids.
    "harmful:privacy": "llm02-pii-disclosure",
    "pii": "llm02-pii-disclosure",
    "prompt-extraction": "llm07-system-prompt-extraction",
    "indirect-prompt-injection": "llm01-indirect-doc",
    "bola": "llm06-tool-idor",
    "bfla": "llm06-tool-unapproved-send",
    "excessive-agency": "llm06-unapproved-action",
}


def taxon_for(probe: str) -> str:
    """The technique this probe is mapped to, or "" when nobody has mapped it.

    Exact lookup, then the part before the first `:` for promptfoo's namespaced
    plugin ids (`harmful:privacy`), then nothing. No prefix matching beyond that:
    `harmful:self-harm` and `harmful:privacy` are not the same technique and a
    prefix rule would make them one.
    """
    probe = str(probe or "").strip()
    if not probe:
        return ""
    if probe in PROBE_TAXA:
        return PROBE_TAXA[probe]
    # garak reports a probe as `module.ClassName`; the module is the family.
    head = probe.split(".", 1)[0]
    if head in PROBE_TAXA:
        return PROBE_TAXA[head]
    return ""


# ----- reading, and refusing what cannot be read ---------------------------------

@dataclass
class ReadResult:
    """Leads, and an account of everything that did not become one."""

    leads: list[Lead] = field(default_factory=list)
    #: Rows read in total, including ones that produced nothing.
    rows: int = 0
    #: {why: how many}. A row that produced no lead is counted under a reason, so
    #: "no leads" can always be told apart from "nothing was understood".
    skipped: dict[str, int] = field(default_factory=dict)

    def note(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    @property
    def interpreted(self) -> int:
        return len(self.leads)


def read_garak(path: str | Path, *, allow_unreadable: bool = False) -> ReadResult:
    """Leads from a garak JSONL report.

    Reads `attempt` rows at status 2, which is the only status its reference says
    carries an evaluation, and treats a detector result above zero as a flag. Falls
    back to `eval` rows when attempts carry no detector results, because the
    reference documents `passed` and `total_evaluated` there and documents neither
    on an attempt.

    Refuses a report whose rows it could not interpret at all, naming the entry
    types and keys it saw. There is no schema version in a garak report to check, so
    this is the only way a format change announces itself rather than arriving as a
    clean scan.
    """
    p = Path(path)
    out = ReadResult()
    seen_types: dict[str, int] = {}
    seen_keys: set[str] = set()

    for n, line in enumerate(_lines(p), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.rows += 1
            out.note("not JSON")
            continue
        if not isinstance(row, dict):
            out.rows += 1
            out.note("not an object")
            continue
        out.rows += 1
        kind = str(row.get("entry_type") or "")
        seen_types[kind or "(none)"] = seen_types.get(kind or "(none)", 0) + 1
        seen_keys.update(map(str, row))
        if kind not in GARAK_ENTRY_TYPES:
            out.note(f"unknown entry_type {kind!r}")
            continue
        lead = _garak_lead(row, n, out)
        if lead is not None:
            out.leads.append(lead)

    _refuse_if_nothing_understood(
        p, out, allow_unreadable,
        what=f"entry types {_counts(seen_types)}",
        keys=sorted(seen_keys)[:12],
        tool="garak")
    return out


def _garak_lead(row: dict, n: int, out: ReadResult) -> Lead | None:
    kind = str(row.get("entry_type") or "")
    if kind == "attempt":
        status = row.get("status")
        if status != GARAK_EVALUATED:
            out.note(f"attempt at status {status!r}, which carries no evaluation")
            return None
        detectors = row.get("detector_results")
        if not isinstance(detectors, dict) or not detectors:
            out.note("attempt with no detector_results")
            return None
        hits = {name: scores for name, scores in detectors.items()
                if _any_positive(scores)}
        if not hits:
            out.note("attempt no detector flagged")
            return None
        probe = _garak_probe(row)
        if not probe:
            out.note("attempt that names no probe")
            return None
        return Lead(source="garak", probe=probe, row=n,
                    flagged=f"garak detectors {', '.join(sorted(hits))} scored above 0",
                    taxon=taxon_for(probe),
                    detail=_excerpt(row.get("goal") or row.get("prompt")))
    if kind in ("eval", "probe_summary"):
        total = row.get("total_evaluated")
        passed = row.get("passed")
        if not isinstance(total, int) or not isinstance(passed, int) or total <= 0:
            out.note(f"{kind} row with no usable passed/total_evaluated")
            return None
        if passed >= total:
            out.note(f"{kind} row where everything passed")
            return None
        probe = _garak_probe(row)
        if not probe:
            out.note(f"{kind} row that names no probe")
            return None
        return Lead(source="garak", probe=probe, row=n,
                    flagged=f"garak scored {passed} of {total} evaluated as passing",
                    taxon=taxon_for(probe))
    out.note(f"{kind} rows carry no lead")
    return None


def _garak_probe(row: dict) -> str:
    """The probe a row names, under whichever key this garak version used.

    Its reference documents neither `probe` nor `detector` on an eval entry, and
    reports in the wild carry `probe_classname` on attempts, so the key is looked
    for rather than assumed. The first match wins and the order is deliberate:
    the most specific name available, because `probe_classname` identifies a class
    where `probe` may be a module.
    """
    for key in ("probe_classname", "probe", "probe_name", "classname"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def read_promptfoo(path: str | Path, *,
                   allow_unreadable: bool = False) -> ReadResult:
    """Leads from a promptfoo `-o results.json` or `.jsonl`.

    The top-level `version` is checked, because promptfoo has one and a reader that
    ignores an available version number is choosing to find out later. Rows are read
    from `results.results` or `results.outputs` for the JSON form, and one per line
    for the JSONL form.

    A failing row is a lead. promptfoo's own documentation says `gradingResult` may
    be absent and its maintainers say `metadata.pluginId` sometimes is too, so both
    are read defensively and a row that identifies no plugin is counted rather than
    guessed at.
    """
    p = Path(path)
    out = ReadResult()
    rows, version = _promptfoo_rows(p, out)
    if version is not None and int(version) != PROMPTFOO_VERSION:
        raise LeadError(
            f"{p}: promptfoo output version {version}, and this reader was written "
            f"against {PROMPTFOO_VERSION}. The row shape changes between majors, so "
            f"reading it anyway would most likely find no leads in a report that has "
            f"them. Check the row keys and widen this reader deliberately")

    seen_keys: set[str] = set()
    for n, row in enumerate(rows, start=1):
        out.rows += 1
        if not isinstance(row, dict):
            out.note("not an object")
            continue
        seen_keys.update(map(str, row))
        if _promptfoo_passed(row):
            out.note("row the grader passed")
            continue
        probe = _promptfoo_plugin(row)
        if not probe:
            out.note("failing row that identifies no plugin")
            continue
        out.leads.append(Lead(
            source="promptfoo", probe=probe, row=n,
            flagged="promptfoo graded this row as failing",
            taxon=taxon_for(probe),
            detail=_excerpt(_promptfoo_reason(row))))

    _refuse_if_nothing_understood(
        p, out, allow_unreadable,
        what=f"{out.rows} row(s)", keys=sorted(seen_keys)[:12], tool="promptfoo")
    return out


def _promptfoo_rows(p: Path, out: ReadResult) -> tuple[list, int | None]:
    """(rows, version). Handles the JSON document and the JSONL stream."""
    text = _text(p)
    stripped = text.lstrip()
    if stripped.startswith("{") and "\n{" not in stripped:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LeadError(f"{p}: not valid JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise LeadError(f"{p}: the top level is not an object")
        version = doc.get("version")
        results = doc.get("results")
        if isinstance(results, dict):
            for key in ("results", "outputs"):
                if isinstance(results.get(key), list):
                    return list(results[key]), version
            raise LeadError(
                f"{p}: `results` holds none of the row lists this reader knows "
                f"(results, outputs); it has {sorted(map(str, results))}")
        if isinstance(results, list):
            return list(results), version
        raise LeadError(f"{p}: no `results` to read rows from")
    rows = []
    for line in _lines(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            out.note("not JSON")
    return rows, None


def _promptfoo_passed(row: dict) -> bool:
    """Whether the grader passed this row, under either of the shapes it writes."""
    if isinstance(row.get("success"), bool):
        return row["success"]
    grading = row.get("gradingResult")
    if isinstance(grading, dict) and isinstance(grading.get("pass"), bool):
        return grading["pass"]
    if isinstance(row.get("pass"), bool):
        return row["pass"]
    # Nothing said. Not a pass and not a lead: `_promptfoo_plugin` will decline it,
    # and treating silence as failure would manufacture leads from error rows.
    return True


def _promptfoo_plugin(row: dict) -> str:
    for holder in (row, row.get("testCase"), row.get("test")):
        meta = (holder or {}).get("metadata") if isinstance(holder, dict) else None
        if isinstance(meta, dict):
            for key in ("pluginId", "plugin"):
                value = str(meta.get(key) or "").strip()
                if value:
                    return value
    # The documented fallback when a row carries no pluginId: its metric name.
    grading = row.get("gradingResult")
    if isinstance(grading, dict):
        for component in grading.get("componentResults") or ():
            if not isinstance(component, dict):
                continue
            metric = str(((component.get("assertion") or {})
                          .get("metric")) or "").strip()
            if metric:
                return metric
    return ""


def _promptfoo_reason(row: dict) -> str:
    grading = row.get("gradingResult")
    if isinstance(grading, dict):
        return str(grading.get("reason") or "")
    return ""


# ----- triage ---------------------------------------------------------------------

def triage(leads: list[Lead], *, covered_by, known_taxa=None,
           existing_ids=()) -> list[Triage]:
    """One record per lead, in order, and never fewer than were handed in.

    `covered_by` answers "which active cases exercise this technique" -- a callable
    rather than a corpus, so this can be reasoned about without one and so the
    library stays the corpus's business.

    `known_taxa` is the technique vocabulary, when there is one. A probe mapped to a
    taxon the vocabulary does not hold is DISMISSED rather than promoted: minting a
    draft case against a technique nothing can resolve produces a case that will
    never crosswalk to an obligation and will sit in the library confusing people.
    """
    out: list[Triage] = []
    minted = {str(i) for i in existing_ids}
    for lead in leads:
        if not lead.taxon:
            out.append(Triage(lead, Outcome.DISMISSED, reason=(
                f"no technique is mapped to {lead.source} probe {lead.probe!r}. "
                f"Add a line to PROBE_TAXA if it exercises one: nothing here guesses "
                f"from the name, because attributing a lead to the wrong technique "
                f"attributes it to the wrong regulatory duty")))
            continue
        if known_taxa is not None and lead.taxon not in known_taxa:
            out.append(Triage(lead, Outcome.DISMISSED, reason=(
                f"{lead.probe!r} is mapped to {lead.taxon!r}, which the technique "
                f"vocabulary does not hold. Fix the mapping or add the technique; a "
                f"draft case against an unresolvable taxon crosswalks to nothing")))
            continue
        cases = tuple(sorted(covered_by(lead.taxon) or ()))
        if cases:
            out.append(Triage(lead, Outcome.COVERED, cases=cases))
            continue
        case_id = _mint(lead, minted)
        minted.add(case_id)
        out.append(Triage(lead, Outcome.PROMOTED, case_id=case_id))
    if len(out) != len(leads):      # pragma: no cover - the invariant, asserted
        raise AssertionError("a lead went unrecorded, which is the one thing this "
                             "must never do")
    return out


def _mint(lead: Lead, taken: set[str]) -> str:
    """A draft id that cannot be mistaken for a library case."""
    slug = _SAFE_SLUG.sub("-", lead.taxon.lower()).strip("-") or "unmapped"
    base = f"{LEAD_CASE_PREFIX}{slug}"
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def write_triage(path: str | Path, records: list[Triage], *,
                 sources: dict[str, ReadResult] | None = None) -> Path:
    """Write `leads/triage.yaml`. Every lead appears, whatever became of it."""
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.outcome.value] = counts.get(r.outcome.value, 0) + 1
    doc: dict[str, Any] = {
        "leads": len(records),
        "counts": counts,
        "triage": [r.as_row() for r in records],
    }
    if sources:
        doc["read"] = {
            name: {"rows": res.rows, "leads": res.interpreted,
                   "not_a_lead": dict(sorted(res.skipped.items()))}
            for name, res in sorted(sources.items())
        }
    p.write_text(_TRIAGE_HEADER + yaml.safe_dump(doc, sort_keys=False, width=100,
                                                 allow_unicode=True),
                 encoding="utf-8")
    return p


_TRIAGE_HEADER = (
    "# Generated by `stem leads`. Every lead another tool produced appears here,\n"
    "# including the ones that became nothing, because a lead quietly dropped is a\n"
    "# scan somebody paid for whose result nobody can account for.\n"
    "#\n"
    "# NONE OF THIS IS A FINDING. `flagged` is the other tool's verdict in its own\n"
    "# words; it says where to point a measurement, not that anything is true. A\n"
    "# promoted case is a DRAFT with no control arm and no oracle, for an author to\n"
    "# accept or throw away. `stem consolidate` excludes this directory in code.\n")


# ----- plumbing -------------------------------------------------------------------

def _text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LeadError(f"cannot read {p}: {exc}") from exc


def _lines(p: Path) -> list[str]:
    return [ln for ln in _text(p).splitlines() if ln.strip()]


def _any_positive(scores) -> bool:
    """Whether a detector scored anything above zero, under either shape.

    garak detectors report a list of per-generation scores; some versions report a
    single number. A string is deliberately not truthy here: a detector that wrote
    "0.0" is not a hit, and `bool("0.0")` is True.
    """
    if isinstance(scores, bool):
        return scores
    if isinstance(scores, (int, float)):
        return scores > 0
    if isinstance(scores, (list, tuple)):
        return any(_any_positive(s) for s in scores)
    return False


def _excerpt(value, limit: int = 160) -> str:
    """A short, flattened excerpt. Never re-sent anywhere; this is for a human.

    Collapsed to one line because a report holds attacker-crafted prompts, and a
    payload with its own newlines rendered into a YAML file an operator skims is how
    somebody copies one out of context.
    """
    text = " ".join(str(value or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _counts(d: dict[str, int]) -> str:
    return ", ".join(f"{k} x{v}" for k, v in sorted(d.items())) or "none"


def _refuse_if_nothing_understood(p: Path, out: ReadResult, allow: bool, *,
                                  what: str, keys: list[str], tool: str) -> None:
    """A report with rows that yields no lead is a parse failure, not a clean scan.

    The same defect as a short corpus load and refused the same way. Neither format
    carries enough version information to check in advance -- garak carries none at
    all -- so this is the only place a format change can announce itself, and the
    message has to name what was actually seen or the next person debugs blind.
    """
    if out.rows == 0:
        raise LeadError(f"{p}: no rows at all. An empty report is not a clean scan")
    if out.leads or allow:
        return
    raise LeadError(
        f"{p}: read {out.rows} row(s) from {tool} and understood none of them as a "
        f"lead, which is how a format change reads as a clean scan. Saw {what}; "
        f"top-level keys {', '.join(keys) or 'none'}. Reasons: "
        f"{_counts(out.skipped)}. If the scan genuinely found nothing, pass "
        f"--allow-nothing and that fact is recorded")
