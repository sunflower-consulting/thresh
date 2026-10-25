"""Which regimes a system is in scope of, from the catalogue's own rules.

THE QUESTION THIS ANSWERS, AND WHY GUESSING AT IT IS EXPENSIVE.

`thresh obligations --system-class annex-iii-high-risk` needs a class, and until now
an operator typed one from memory. That is the denominator of the only coverage
figure a report may state, so getting it wrong does not produce an error -- it
produces a confident appendix over the wrong set of duties. Passing `all`, which
reads like "every class", selects the single obligation tagged `all` and reports one
row. Passing a class the client does not carry pads the page with duties they never
had, and the catalogue's own warning about that is why scoping exists at all.

The rules to decide it have been in the catalogue the whole time, under `tier`, and
nothing read them. Five branching questions, three independent ones, and thirteen
classes, written by the same people who wrote the obligations.

THREE ZEROS THAT MEAN DIFFERENT THINGS.

Measured against the current catalogue: of the thirteen classes these rules can
produce, THREE carry no obligation at all -- `minimal-risk`, `annex-i-high-risk` and
`needs-legal-review`. Hand any of them to the scope filter and you get an empty
mapping, and the three empties mean:

    minimal-risk        no specific obligations under the Act. Zero IS the answer.
    annex-i-high-risk   high-risk embedded in a regulated product, and the
                        catalogue has no rows for it yet. Zero is a GAP in the
                        data, not an absence of duty.
    needs-legal-review  the rules could not decide. Zero is a refusal to answer,
                        and the system may well be high-risk.

Rendering those alike is the same failure as `probed-clean` against `not-probed` one
layer up, and it is worse here because the reader is being told what they owe. So a
classification carries each class with what an empty result would mean for it, and
`scope_classes` returns only the ones that are a usable denominator.

NOT LEGAL ADVICE, and the catalogue says so in its own README. These are reusable
questions somebody still has to answer about a specific deployment, and
`needs-legal-review` exists precisely because the rules decline to guess. `tier`
also carries `open_checks` -- things deliberately NOT encoded because they were seen
in secondary sources and not confirmed against the official text -- and those travel
with every answer rather than sitting in a file nobody opens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: A class the rules can reach that is not a scope to measure against. Each is
#: returned and reported; none is passed to the obligation filter.
NOT_A_SCOPE = {
    "needs-legal-review": (
        "the rules could not decide from the description. This is not a low-risk "
        "answer: the Article 6(3) carve-out is the hardest judgement in the Act and "
        "the system may well be high-risk. Somebody has to read the deployment"),
    "prohibited": (
        "Article 5 forbids the practice outright. Not a risk tier to manage and not "
        "a scope to measure: the question is whether it may be deployed at all, "
        "which is not a question an assurance run answers"),
}


@dataclass
class Classification:
    """What the rules concluded, and what each conclusion is worth as a denominator."""

    #: Every class reached, in the order reached.
    classes: tuple[str, ...] = ()
    #: {class: label}, from the catalogue.
    labels: dict[str, str] = field(default_factory=dict)
    #: {class: why it is not a scope}, for the ones that are a stop rather than an
    #: answer. Non-empty means the run cannot simply proceed.
    stops: dict[str, str] = field(default_factory=dict)
    #: Classes that are a real scope but carry no obligation in this catalogue --
    #: a gap in the data, which must not read as an absence of duty.
    empty: tuple[str, ...] = ()
    #: Questions asked, in order, with the answer given.
    trail: tuple[tuple[str, bool], ...] = ()
    #: Questions the rules reached and nobody answered.
    unanswered: tuple[str, ...] = ()
    #: What the catalogue deliberately did not encode.
    open_checks: tuple[str, ...] = ()

    @property
    def scope_classes(self) -> tuple[str, ...]:
        """The classes that are a usable denominator. Only these.

        A stop is excluded because it is not a scope, and an empty class is excluded
        because passing it filters every obligation away -- the caller is told about
        both rather than discovering a blank page.
        """
        return tuple(c for c in self.classes
                     if c not in self.stops and c not in self.empty)

    @property
    def decided(self) -> bool:
        return bool(self.scope_classes) and not self.stops and not self.unanswered

    def as_flags(self) -> str:
        """The `--system-class` arguments, ready to paste."""
        return " ".join(f"--system-class {c}" for c in self.scope_classes)


def questions(snapshot: dict) -> list[dict]:
    """The branching questions, in the order the rules ask them."""
    return list((snapshot.get("tier") or {}).get("questions") or ())


def also_ask(snapshot: dict) -> list[dict]:
    """The independent questions. Each adds a class on its own, outside the tree."""
    return list((snapshot.get("tier") or {}).get("also_ask") or ())


def classes(snapshot: dict) -> dict[str, dict]:
    return dict((snapshot.get("tier") or {}).get("classes") or {})


def classify(snapshot: dict, answers: dict[str, bool]) -> Classification:
    """Walk the rules with `answers`, a mapping of question id to yes or no.

    The tree first: each question's `if_yes`/`if_no` names either another QUESTION
    or a CLASS, and the walk follows until it reaches a class. A question with no
    branch for the answer given falls through to the next one in order, which is how
    the catalogue writes "not this, carry on".

    Then the independent questions, each of which adds its class on a yes. A system
    can be Annex III high-risk AND an ICT third-party under DORA; those are different
    instruments and the scope is the union.

    An unanswered question stops the walk rather than assuming. The catalogue has a
    class for "cannot be determined" and inventing an answer here would bypass it.
    """
    tier_classes = classes(snapshot)
    by_id = {q["id"]: q for q in questions(snapshot) if q.get("id")}
    order = [q["id"] for q in questions(snapshot) if q.get("id")]

    reached: list[str] = []
    trail: list[tuple[str, bool]] = []
    unanswered: list[str] = []

    index = 0
    seen: set[str] = set()
    while index < len(order):
        qid = order[index]
        question = by_id[qid]
        if qid in seen:
            # A cycle in the rules. Impossible with the catalogue as written and
            # cheap to refuse anyway, because the failure without this guard is a
            # HANG -- which gives an operator nothing at all, where a wrong answer
            # at least gets argued with.
            break
        seen.add(qid)
        if qid not in answers:
            unanswered.append(qid)
            break
        answer = bool(answers[qid])
        trail.append((qid, answer))
        target = question.get("if_yes" if answer else "if_no")
        if not target:
            index += 1                      # no branch for this answer: carry on
            continue
        # A CLASS first, and the order is not arbitrary: the first question's id is
        # `prohibited` and its `if_yes` target is the class `prohibited`, so
        # resolving questions first sent the walk back to index 0 and span forever.
        # The classes table is what says a name is a class.
        if target in tier_classes:
            reached.append(target)
            break
        if target in by_id:                 # the branch names another question
            index = order.index(target)
            continue
        reached.append(target)              # neither: taken at its word, and
        break                               # reported as carrying no obligation

    for extra in also_ask(snapshot):
        qid = extra.get("id")
        if not qid:
            continue
        if qid not in answers:
            unanswered.append(qid)
            continue
        answer = bool(answers[qid])
        trail.append((qid, answer))
        target = extra.get("if_yes" if answer else "if_no")
        if target and target not in reached:
            reached.append(target)

    carried = {c for o in (snapshot.get("obligations") or {}).values()
               for c in (o.get("system_class") or ())}
    return Classification(
        classes=tuple(reached),
        labels={c: str((tier_classes.get(c) or {}).get("label") or c)
                for c in reached},
        stops={c: NOT_A_SCOPE[c] for c in reached if c in NOT_A_SCOPE},
        empty=tuple(c for c in reached
                    if c not in NOT_A_SCOPE and c not in carried),
        trail=tuple(trail),
        unanswered=tuple(dict.fromkeys(unanswered)),
        open_checks=tuple((snapshot.get("tier") or {}).get("open_checks") or ()),
    )


def why_empty(snapshot: dict, klass: str) -> str:
    """Why a class that carries no obligation carries none. Three different answers.

    The whole reason this module reports `empty` apart from `classes`: an empty
    mapping looks the same whichever of these is true, and they are not the same
    thing to tell a client.
    """
    if klass in NOT_A_SCOPE:
        return NOT_A_SCOPE[klass]
    note = str((classes(snapshot).get(klass) or {}).get("note") or "")
    if klass == "minimal-risk":
        return ("no specific obligations under the Act, so an empty mapping is the "
                "correct and complete answer" + (f". {note}" if note else ""))
    return ("this catalogue carries no obligation for it yet, so an empty mapping "
            "is a GAP IN THE DATA rather than an absence of duty. Closing it is a "
            "catalogue change" + (f". {note}" if note else ""))


def render(snapshot: dict, result: Classification, *,
           title: str = "classification") -> str:
    """The readout. What was concluded, what it is worth, and what was not asked."""
    from sunflower_thresh import termstyle as ts

    lines = [ts.header(f" {title}")]
    if result.trail:
        lines.append("")
        for qid, answer in result.trail:
            mark = ts.warn(" yes") if answer else ts.muted("  no")
            lines.append(f"   {mark}  {qid}")

    lines.append("")
    if not result.classes:
        lines.append("  " + ts.warn("no class reached"))
    for klass in result.classes:
        label = result.labels.get(klass, klass)
        if klass in result.stops:
            lines.append(f"  {ts.bad('STOP')}  {ts.name(klass)}  {ts.muted(label)}")
            lines.append(f"        {ts.warn(result.stops[klass])}")
        elif klass in result.empty:
            lines.append(f"  {ts.warn('gap ')}  {ts.name(klass)}  {ts.muted(label)}")
            lines.append(f"        {ts.muted(why_empty(snapshot, klass))}")
        else:
            lines.append(f"  {ts.good('ok  ')}  {ts.name(klass)}  {ts.muted(label)}")

    if result.unanswered:
        lines.append("")
        lines.append("  " + ts.warn(
            f"{len(result.unanswered)} question(s) unanswered: "
            f"{', '.join(result.unanswered)}"))
        lines.append("  " + ts.muted(
            "the walk stops rather than assuming, because the rules have a class "
            "for 'cannot be determined' and guessing would bypass it"))

    lines.append("")
    if result.scope_classes:
        lines.append("  " + ts.name("scope for the obligation mapping"))
        lines.append(f"    thresh obligations {result.as_flags()}")
    else:
        lines.append("  " + ts.warn(
            "no class here is a usable denominator, so there is no mapping to run"))

    if result.open_checks:
        lines.append("")
        lines.append("  " + ts.name("the catalogue deliberately did not encode"))
        for check in result.open_checks:
            lines.append(f"    {ts.muted('- ' + str(check))}")

    lines.append("")
    lines.append("  " + ts.muted(
        "These are the catalogue's own questions, not legal advice. Somebody still "
        "has to answer them about a specific deployment."))
    return "\n".join(lines)


def as_document(result: Classification) -> dict[str, Any]:
    """The classification as data, for an engagement file to record."""
    doc: dict[str, Any] = {
        "classes": list(result.classes),
        "scope_classes": list(result.scope_classes),
        "answers": [{"question": q, "yes": a} for q, a in result.trail],
    }
    if result.stops:
        doc["stops"] = [{"class": k, "why": v} for k, v in result.stops.items()]
    if result.empty:
        doc["carry_no_obligation"] = list(result.empty)
    if result.unanswered:
        doc["unanswered"] = list(result.unanswered)
    if result.open_checks:
        doc["open_checks"] = list(result.open_checks)
    doc["note"] = ("The catalogue's own classification questions, answered for this "
                   "deployment. Not legal advice.")
    return doc
