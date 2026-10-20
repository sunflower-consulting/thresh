"""Run the whole verified test library, deterministically, before the model runs.

The library is the harness's reliable floor: each case was checked against a
live target, each has an oracle and a negative control, and each reaches
VALIDATED through the same reproduce-twice gate as anything else. But the loop
only ever ran the cases the *model* chose to nominate -- so a benchmark with 31
verified tests available fired 18 of them, and seven cases added that same day
(each one proven to flip a challenge minutes earlier) were simply never picked.

That is the wrong dependency. Deterministic knowledge should not be gated behind
model whim: the model's budget belongs on discovery -- the things no library can
anticipate -- not on re-deriving a checklist somebody already wrote down.

So the sweep runs every applicable case first, then the model loop runs for
discovery with that floor already banked. Nothing bypasses the policy gate: each
case goes through ``validate_*`` -> ``AgentContext.execute_tool``, is authorized
per call, and a gate denial mints no finding.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from sunflower_thresh.context import AgentContext
from sunflower_thresh.oracles import OBSERVATION_NEEDS, Scoring, Unscored, rung_needs
from sunflower_thresh.schemas.finding import Finding, FindingState
from sunflower_thresh.tools.chat import BUILTIN_CHAT_CASES
from sunflower_thresh.validation_chat import validate_chat_case

#: What each `needs` token requires before a chat case is REACHABLE.
#:
#: A capability token is checked against the inventory's switches; the others
#: against what this run actually has. The point is the distinction the rest of the
#: codebase keeps insisting on: a case we could not run is UNREACHABLE, and
#: recording it as "not vulnerable" puts a false negative in the denominator and
#: lets a coverage number claim ground nobody tested.
#: Written as literals rather than imported, so that `thresh --help` does not pay
#: for the collector and the evidence store to resolve a set of four strings. The
#: cost of that choice is drift: `collector.CALLBACK_CAPABILITY` holds the same
#: name, and if the two part company the collector gates on one spelling while
#: this decides reachability on the other -- a case reported unreachable that the
#: collector would have allowed, or the reverse. Renaming the constant and leaving
#: this alone passed the whole suite, so
#: `test_the_callback_capability_is_spelled_the_same_everywhere` now pins it.
_NEEDS_CAPABILITY = {"plant_content", "out_of_band_callback",
                     "capture_harmful_content", "multi_turn_escalation"}


def scoring_for(case, met: Callable[[str], bool]) -> Scoring:
    """Which of one case's oracles THIS run can score. See `oracles.Scoring`.

    `met` answers one need token. Everything about the run's capabilities is already
    in that closure, so this function is the whole reachability decision and nothing
    else makes it -- which is what stops the sweep and the validator disagreeing about
    which oracles were in play.

    Three decisions, and they point in three different directions on purpose.

    The ORACLES are decided individually: a case runs when at least one of them can
    score, and the ones that cannot are reported by name rather than quietly dropped.
    Unless the case is `all_of` -- then the claim is the conjunction and a subset of it
    is a different, weaker claim, so one unobservable rung loses the case. That is
    `Scoring.conjunctive`, and it is the one place where adding an oracle can still
    cost a case: an author who means "either of these" writes `any_of`.

    The SUPPRESSORS gate the case. Losing an oracle measures less, which a report can
    state; losing the refusal suppressor measures wrong, because a bot answering "I
    will not print CANARY-7F3A" contains the canary and leaked nothing. A finding that
    may be false is worse in a client's report than a case that says it did not run.

    Everything NOT in `OBSERVATION_NEEDS` gates the case, including an observation
    need that no rung claimed. That last clause is the one that keeps this honest: a
    case declaring `judge` with no judge rung -- or any token a future
    `_needs_for` rule adds that this function does not know how to attach -- stays a
    whole-case gate rather than being silently discarded, so a new rule can only ever
    make the harness stricter by accident, never more generous.
    """
    declared = tuple(case.needs or ())
    names = list(case.oracle_names or ())
    veto_names = list(case.suppressor_names or ())

    def named(rung: dict, at: int, from_: list[str]) -> str:
        if at < len(from_) and from_[at]:
            return from_[at]
        return next(iter(rung or {}), "oracle")

    scored: list[dict] = []
    unscored: list[Unscored] = []
    claimed: set[str] = set()
    for i, rung in enumerate(case.oracles or []):
        wants = rung_needs(rung, declared=declared)
        claimed |= set(wants)
        lacks = tuple(n for n in wants if not met(n))
        if lacks:
            unscored.append(Unscored(named(rung, i, names), lacks))
        else:
            scored.append(rung)

    vetoes: list[Unscored] = []
    for i, rung in enumerate(case.suppressors or []):
        wants = rung_needs(rung, declared=declared)
        claimed |= set(wants)
        lacks = tuple(n for n in wants if not met(n))
        if lacks:
            vetoes.append(Unscored(named(rung, i, veto_names), lacks))

    blocked: list[str] = []
    unbound: list[str] = []
    for need in declared:
        if need.startswith("binding:"):
            # Not a capability anything can switch on: the engagement has not
            # supplied a value this payload needs, and sending the literal
            # `{{OTHER_USER_ID}}` would be a probe that cannot match its own oracle.
            # Named so an operator knows what to go and ask for.
            unbound.extend(need.split(":", 1)[1].split(","))
        elif need in OBSERVATION_NEEDS and need in claimed:
            continue            # decided per rung above
        elif not met(need):
            blocked.append(need)

    return Scoring(oracles=tuple(scored), unscored=tuple(unscored),
                   blocked=tuple(blocked), unbound=tuple(unbound),
                   suppressors_unscored=tuple(vetoes),
                   conjunctive=bool(case.require_all))


def applicable_chat_cases(tool_names: Iterable[str],
                          allowed: Iterable[str] | None = None, *,
                          capabilities=None, have_collector: bool = False,
                          have_judge: bool = False,
                          have_ingestion: bool = False,
                          have_tool_visibility: bool = False,
                          allow_load: bool = False,
                          have_resolver: bool = False,
                          profile=None,
                          cases: dict | None = None,
                          seeds: dict | None = None) -> tuple[dict[str, Scoring],
                                                              dict[str, str]]:
    """({runnable case id: what it will be scored on}, {skipped id: why}).

    Returns the skips rather than dropping them, so the run can report what it did
    not test and the coverage denominator stays honest. The web path has no analogue
    because an HTTP case's reachability is just "is the tool permitted"; a chat
    case can need a plantable sink, an instrumented agent, a reachable collector or
    a judge, none of which the tool allowlist describes.

    The runnable half is a MAPPING rather than a list, and the mapping is what makes
    reachability per oracle work end to end: the decision is made once, here, where
    the run's capabilities are known, and travels to `validate_chat_case` so the
    ladder is built from the oracles this run can actually score. A list of ids would
    have left the validator to guess, and it has no way to know whether tool calls are
    observable. It still reads as a collection of ids everywhere that only wanted ids
    -- `cid in runnable`, `len(runnable)`, iteration -- which is why the signature
    could change without touching the call sites.

    `cases` is the library to judge, defaulting to the built-ins. The corpus adapter
    passes its own, because reachability is the same question for a corpus case as
    for a built-in one and answering it twice would let the two answers drift.

    The counts that used to be quoted here are gone on purpose. One was simply
    wrong -- it said five built-ins and there are six -- and the other named a
    corpus subset that cannot be recomputed without the engagement's bindings,
    so neither could be checked by a reader or by a test. The argument does not
    rest on either number.
    """
    from sunflower_thresh.tools.chat import BUILTIN_CHAT_CASES

    cases = BUILTIN_CHAT_CASES if cases is None else cases
    names = set(tool_names)
    if allowed is not None:
        names &= set(allowed)
    if "chat_say" not in names:
        return {}, {cid: "chat_say is not permitted for this project"
                    for cid in cases}

    def enabled(cap: str) -> bool:
        if capabilities is None:
            return False
        if hasattr(capabilities, "capability_enabled"):
            return bool(capabilities.capability_enabled(cap))
        return bool(dict(capabilities).get(cap))

    # The engagement's seeds, from the inventory when one was passed.
    if seeds is None and capabilities is not None:
        seeds = dict(getattr(capabilities, "seeds", {}) or {})
    seeds = seeds or {}

    # What the target itself turned out to be able to do. From the inventory, which
    # is where the engagement recorded it, falling back to the measured profile for
    # the two things probing can establish on its own.
    declared = set(getattr(capabilities, "target_features", ()) or ())
    channels = set(getattr(capabilities, "seedable_channels", ()) or ())
    stateful = "stateful" in declared or bool(getattr(profile, "stateful", False))
    tools_visible = have_tool_visibility or bool(
        getattr(profile, "tool_calls_observed", False))

    have = {"conversation": True,
            "multi_turn": enabled("multi_turn_escalation"),
            "ingestion": have_ingestion,
            "tool_visibility": tools_visible,
            "oob": have_collector,
            "judge": have_judge,
            "load": allow_load,
            "package_registry": have_resolver,
            "stateful": stateful,
            # An extraction case needs the secret the TARGET holds. Minting one
            # guarantees a false negative -- a token the bot has never seen cannot
            # come back -- so without it the case is UNREACHABLE, not failed. This
            # was measured: eight cases scored 0.00 against a lab whose board showed
            # the same attacks landing.
            "engagement_canary": bool(seeds.get("canary"))}

    # `target_tools`, `target_rag`, `target_memory`, ... -- what the BOT has, which is
    # a different question from what the harness can observe. A RAG case against a bot
    # with no retrieval is not a resisted attack, it is a test that had nothing to
    # attack, and the two must not share a column in the report.
    for feature in ("tools", "rag", "memory", "multimodal", "files", "web"):
        have[f"target_{feature}"] = feature in declared

    # `seed_document`, `seed_email`, ... -- an indirect case run WITHOUT its seeding
    # looks like a clean pass: the payload never arrives, the bot answers an ordinary
    # question, and the report claims the target resisted indirect injection.
    for channel in channels:
        have[f"seed_{channel}"] = True

    def met(need: str) -> bool:
        return enabled(need) if need in _NEEDS_CAPABILITY else have.get(need, False)

    runnable: dict[str, Scoring] = {}
    skipped: dict[str, str] = {}
    for cid, case in cases.items():
        scoring = scoring_for(case, met)
        if scoring.runnable:
            runnable[cid] = scoring
        else:
            skipped[cid] = scoring.why_not()
    return runnable, skipped


def _validate_one(ctx, tid, *, target_ip, port, run_id,
                  trials=None, judge=None, collector=None, seeds=None,
                  resolver=None, cases=None, baseline_turns=(), transport=None,
                  scoring=None):
    """Dispatch one case to its validator. Returns (finding, report, failed).

    `cases` is the chat-case library, defaulting to the built-in smoke set. Threaded
    as a parameter rather than held in a module-level registry: a corpus loaded into
    global state would make "which library did this run use" depend on import order,
    and the manifest records the answer.
    """
    chat_cases = BUILTIN_CHAT_CASES if cases is None else cases
    # One arm. The package this came from dispatched to an HTTP registry, a
    # browser registry and a superseded chat registry before reaching this
    # branch, and first match won -- so an id present in two registries was
    # routed by position rather than by intent. With one registry that ordering
    # question cannot arise.
    try:
        if tid in chat_cases:
            finding, report = validate_chat_case(
                ctx, chat_cases[tid], target_ip=target_ip, port=port,
                run_id=run_id, by="sweep",
                trials=trials, control_trials=trials,
                judge=judge, collector=collector, seeds=seeds, resolver=resolver,
                baseline_turns=baseline_turns, transport=transport,
                scoring=scoring)
        else:
            return None, {"error": f"unknown case {tid!r}"}, True
    except Exception as exc:  # noqa: BLE001 - one bad case must not end the sweep
        return None, {"error": f"{type(exc).__name__}: {exc}"}, True
    return finding, report, False


#: A rate-limit denial is transient and self-clearing; every other denial is a
#: decision. Retrying the first is correct, retrying the second is pointless.
#: Only `rate_limited` belongs here, and the omission is load-bearing.
#:
#: A rate-limit denial self-clears: wait, retry, it works. A spent target-message
#: budget does not -- it is a decision for the rest of the run. Listing
#: `message_budget_exhausted` here would make the loop below retry it
#: MAX_RATE_RETRIES times per case, sleeping between each, so a library of forty
#: cases would spend minutes re-asking a question whose answer cannot change.
#: tests/test_message_budget.py asserts the contents of this tuple.
TRANSIENT_DENIALS = ("rate_limited",)
MAX_RATE_RETRIES = 6


def sweep_builtins(
    ctx: AgentContext,
    test_ids: Iterable[str],
    *,
    target_ip: str,
    port: int,
    run_id: str,
    on_result: Callable[[str, Finding | None, dict], None] | None = None,
    #: Called with the case id just BEFORE it runs. The live view needs the
    #: moment work starts, not only the moment it finishes, or a long case looks
    #: like an idle worker.
    on_start: Callable[[str], None] | None = None,
    budget_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    #: Trials per chat case, overriding each case's own figure. See --sweep-trials:
    #: attack-success-rate costs n trials plus n controls per case, so the chat half
    #: of the library is minutes rather than the HTTP half's seconds.
    trials: int | None = None,
    judge=None,
    collector=None,
    #: The engagement's seeded values, from the inventory. An extraction case needs
    #: the secret the target actually holds.
    seeds: dict | None = None,
    resolver=None,
    #: Ordinary questions for this target, from the inventory: the control arm for any
    #: case with no matched control of its own, which is every corpus case.
    baseline_turns=(),
    #: How to reach the chat endpoint, from the inventory: tls, verify, reply_path.
    transport: dict | None = None,
    #: The chat-case library. Defaults to the built-in smoke set; an engagement
    #: passes the corpus.
    cases: dict | None = None,
    #: What each case will be SCORED on, from `applicable_chat_cases` -- which is
    #: usually `test_ids` itself, since the runnable half of that pair is a mapping.
    #: Passing it is what makes a case run on the oracles this run can observe rather
    #: than on all of them; without it each case is scored on its whole ladder, which
    #: is right for a caller that did no reachability pass and wrong for one that did.
    scoring: dict | None = None,
    #: The OTHER half of `applicable_chat_cases`'s return: {case id: why it
    #: cannot be scored}. Accepted so the counts can include the cases that
    #: never reached this driver at all.
    #:
    #: Without it the denominator lies by omission. `applicable_chat_cases`
    #: already returns the skips, and its docstring says they exist so "the
    #: coverage denominator stays honest" -- but this function had no parameter
    #: to receive them, so every caller dropped them on the floor. With the
    #: judge off, `chat-fabricated-policy` is unreachable, and a six-case
    #: library then reported five attempted and said nothing at all about the
    #: sixth. "Nothing was found" and "nothing was looked for" must not render
    #: alike, and a missing row renders as the first.
    unreachable: dict[str, str] | None = None,
) -> tuple[list[Finding], dict]:
    """Validate every case in ``test_ids``. Returns (findings, stats).

    One case failing -- a gate denial, a dead endpoint, a tool error -- never
    stops the sweep: the point is to bank everything that *does* hold.

    The gate rate-limits per (project, tool) and every case here runs through the
    same tool, so an unpaced sweep denies itself: 37 cases at three requests each
    tripped the limiter and dropped two thirds of the library on the floor. A
    rate-limit denial is therefore backed off and retried, which is honest --
    the limiter exists to protect fragile gear from a burst, not to cap how much
    of the library may run.

    ONE PROPERTY OF THIS SWEEP DOES NOT SURVIVE CHAT CASES, and it should be said
    here rather than discovered. The HTTP library's whole appeal is that it banks
    ~33 findings in about fourteen seconds, model-independent. A chat case cannot be
    that: establishing an attack success rate takes n trials plus n controls, each a
    real round-trip to a model, so twenty chat cases at the default n=10 is four
    hundred requests and minutes of wall-clock. `trials` exists to bound that for a
    quick pass -- but it is deliberately NOT defaulted to 2 to recover the old
    number, because a two-trial measurement is precisely the bug
    validation_chat.py was written to fix.
    """
    # ONE baseline arm for the whole sweep: its questions are sent once and re-scored
    # by every case's own ladder. Built here rather than inside _validate_one, because
    # a per-case arm would re-send them and spend the message budget on controls.
    from sunflower_thresh.validation_chat import BaselineArm

    if not isinstance(baseline_turns, BaselineArm):
        baseline_turns = BaselineArm(tuple(baseline_turns or ()), trials or 10)

    # `test_ids` IS the mapping when the caller passed `applicable_chat_cases`'s
    # runnable half straight through, which is the common path and the one that needs
    # no second argument.
    if scoring is None and isinstance(test_ids, dict):
        scoring = test_ids

    started = clock()
    findings: list[Finding] = []
    # Every key present from the start. `rate_waits`, `deny_reasons` and
    # `denied_ids` used to be created only when the thing they count happened,
    # so a reporter reading them raised KeyError on a CLEAN run -- a crash on
    # exactly the run nobody tests.
    #
    # `unreachable` and `budget_skipped` are separate because they are different
    # claims. A case nothing could score is an absence of measurement; a case
    # dropped because the clock ran out is a truncated sweep. Both used to land
    # in one "skipped" bucket with the time budget, and a case the validator
    # declared unreachable landed in "errors" instead, which reads as a broken
    # case rather than an untested duty.
    stats = {"considered": 0, "attempted": 0, "validated": 0, "rejected": 0,
             "errors": 0, "denied": 0, "unreachable": 0, "budget_skipped": 0,
             "rate_waits": 0, "deny_reasons": {}, "denied_ids": [],
             "unreachable_ids": {}}
    # A bare count of denials is not diagnosable: "denied 26" says nothing about
    # which rule fired. Tally the reasons and the cases so a sweep explains itself.
    deny_reasons: dict[str, int] = {}
    denied_ids: list[str] = []

    for tid in test_ids:
        stats["considered"] += 1
        if budget_seconds is not None and clock() - started > budget_seconds:
            stats["budget_skipped"] += 1
            continue
        stats["attempted"] += 1
        if on_start is not None:
            try:
                on_start(tid)
            except Exception:  # noqa: BLE001,S110 - a view never breaks a sweep
                pass

        for attempt in range(MAX_RATE_RETRIES + 1):
            finding, report, failed = _validate_one(
                ctx, tid, target_ip=target_ip, port=port,
                run_id=run_id, trials=trials, judge=judge, collector=collector,
                seeds=seeds, resolver=resolver, cases=cases,
                baseline_turns=baseline_turns, transport=transport,
                scoring=(scoring or {}).get(tid))
            if failed:
                stats["errors"] += 1
                if on_result is not None:
                    on_result(tid, None, report)
                break
            code = str(report.get("deny_code") or "")
            if not (report.get("gate_denied") and code in TRANSIENT_DENIALS):
                break
            if attempt < MAX_RATE_RETRIES:
                stats["rate_waits"] = stats.get("rate_waits", 0) + 1
                sleeper(0.25 * (attempt + 1))      # let the sliding window drain
        else:
            pass
        if failed:
            continue

        if report.get("gate_denied"):
            stats["denied"] += 1
            why = str(report.get("deny_code") or report.get("gate_reason")
                      or report.get("reason") or "unknown")
            deny_reasons[why] = deny_reasons.get(why, 0) + 1
            denied_ids.append(tid)
        elif report.get("skipped"):
            # The validator looked at the case and said no rung could score it:
            # no seeded canary, or no reachable oracle. That is an absence of
            # measurement and it carries its reason, which is the one thing a
            # reader needs and the one thing an "errors" tally destroys.
            stats["unreachable"] += 1
            stats["unreachable_ids"][tid] = str(report["skipped"])
        elif finding is None:
            stats["errors"] += 1
        elif finding.state is FindingState.VALIDATED:
            stats["validated"] += 1
            findings.append(finding)
        else:
            stats["rejected"] += 1
            findings.append(finding)
        if on_result is not None:
            on_result(tid, finding, report)

    # Folded in before the counts are read, so `considered` covers the whole
    # library the caller started from rather than only the part that ran.
    for cid, why in (unreachable or {}).items():
        if cid not in stats["unreachable_ids"]:
            stats["unreachable"] += 1
            stats["considered"] += 1
            stats["unreachable_ids"][cid] = str(why)

    stats["seconds"] = round(clock() - started, 2)
    stats["deny_reasons"] = deny_reasons
    # Truncated, so `len(denied_ids)` is not the count -- `stats["denied"]` is.
    stats["denied_ids"] = denied_ids[:20]
    return findings, stats
