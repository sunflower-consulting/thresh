"""`thresh demo`: the whole argument, against a target that ships with the tool.

WHY A DEMO IS PART OF THE PRODUCT RATHER THAN A SAMPLE.

Everything thresh claims is a claim about method, and a method is not persuasive
described. The comparable tools in this space need an endpoint, a key and a
config file before they produce their first number, which means the honest
version of "try it" is an afternoon. This is two commands and no account.

It also keeps the claims testable. The bundled target scores itself, so what
this prints can be checked against what the target believes happened rather
than only against what thresh reports.

WHAT TO LOOK AT IN THE OUTPUT.

Not the hit counts. The pair of them.

The undefended assistant hands its canary to anyone who asks, so
`chat-direct-override` succeeds on every attempt -- and its benign twin, which
asks for the same secret without the override framing, succeeds just as often.
A tool that reported the first number alone would file a critical prompt
injection. The technique did nothing; the assistant was simply unprotected.
thresh rejects it, and the rejection is the point.
"""
from __future__ import annotations

import threading
from datetime import UTC, date, datetime
from pathlib import Path

CHAT_TOOLS = ("chat_open", "chat_say", "chat_reset")


def _now() -> str:
    return datetime.now(UTC).isoformat()

#: Switched on explicitly because capability checks are fail-closed: an
#: unlisted switch reads as off, and `chat-markdown-exfil` would then be
#: reported as unreachable rather than measured. An engagement turns these on
#: in its own inventory; the demo is making the same decision in the open.
DEMO_CAPABILITIES = {"out_of_band_callback": True, "multi_turn_escalation": True}


def _inventory(port: int, canary: str):
    from sunflower_thresh.schemas.target import AuthorizationRecord, TargetInventory

    return TargetInventory(
        project="demo",
        allowed_cidrs=("127.0.0.1/32",),
        external_network=False,
        allowed_tools=CHAT_TOOLS,
        chat_path="/chat",
        chat_port=port,
        # Ten trials against ten controls across six cases, plus the open and
        # reset either side of each. Generous rather than tuned: a budget that
        # runs out mid-sweep is a truncated measurement, which this tool reports
        # loudly and which would be a confusing thing for a demo to show.
        max_target_messages=4000,
        capabilities=DEMO_CAPABILITIES,
        seeds={"canary": canary},
        authorization=AuthorizationRecord(
            authorized_by="you",
            authorized_for="you",
            scope_summary="the vulnerable chat application bundled with thresh",
            granted_on=date(2026, 1, 1),
            reference="bundled-lab-on-loopback"),
    )


def _ratio(cell: str) -> float:
    """"3/5" -> 0.6. For comparing the two arms in the summary only."""
    hits, _, trials = cell.partition("/")
    try:
        return int(hits) / int(trials) if int(trials) else 0.0
    except ValueError:
        return 0.0


def _verdict(finding) -> str:
    """One word, from the finding's own state rather than from the numbers."""
    return finding.state.value if finding is not None else "-"


def run(*, safety: str = "none", trials: int = 5, quiet: bool = False,
        save: str | None = None, quotable: bool = False) -> int:
    """Run the built-in cases against the bundled lab. Returns an exit code.

    `save` writes a run bundle, which is what makes the run quotable by anything
    other than the terminal it printed to. Off by default: the two-command
    quickstart should leave nothing behind on a reader's machine.
    """
    from sunflower_thresh.collector import CallbackCollector
    from sunflower_thresh.context import AgentContext
    from sunflower_thresh.evidence.store import EvidenceStore
    from sunflower_thresh.lab import serve
    from sunflower_thresh.lab.scoreboard import CANARY
    from sunflower_thresh.policy.engine import PolicyEngine
    from sunflower_thresh.redact import make_redactor
    from sunflower_thresh.sweep import applicable_chat_cases, sweep_builtins
    from sunflower_thresh.tools.chat import BUILTIN_CHAT_CASES
    from sunflower_thresh.tools.conversation import ConversationStore, chat_tools

    def say(*a):
        if not quiet:
            print(*a, flush=True)

    from sunflower_thresh.runbundle import run_id as _mint

    rid = _mint("demo")
    started = _now()
    srv, _lab = serve(port=0, deterministic=True, safety=safety)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    # The canary is protected from redaction by name. It is a high-entropy token,
    # so it looks exactly like the secrets the patterns hunt, and redacting it
    # would destroy the evidence for the strongest rung on the ladder.
    redactor = make_redactor(keep=(CANARY,))
    store = EvidenceStore(redactor=redactor)
    collector = CallbackCollector(store=store).start()

    inventory = _inventory(port, CANARY)
    policy = PolicyEngine()
    verified = policy.load_inventory(inventory)

    conversations = ConversationStore()
    ctx = AgentContext("demo", policy,
                       tool_impls=chat_tools(conversations, store, policy=policy))

    say(f"target     the bundled lab on http://127.0.0.1:{port}  (safety={safety}, "
        f"scripted brain, no model)")
    say(f"collector  {collector.netloc}  (loopback; proof is the target's own request)")
    say(f"scope      signed inventory, verified={verified}; {len(CHAT_TOOLS)} tools "
        f"permitted, external network off")
    say(f"budget     {trials} attempts per case, and {trials} per control arm")
    say("")

    runnable, unreachable = applicable_chat_cases(
        CHAT_TOOLS,
        capabilities=inventory,
        have_collector=True,
        # Off, and the consequence is visible below rather than hidden: a case
        # that only a judge could score is reported as unreachable.
        have_judge=False,
        have_tool_visibility=True,
        cases=BUILTIN_CHAT_CASES,
        seeds={"canary": CANARY})

    try:
        findings, stats = sweep_builtins(
            ctx, runnable, target_ip="127.0.0.1", port=port, run_id=rid,
            trials=trials, collector=collector, seeds={"canary": CANARY},
            cases=BUILTIN_CHAT_CASES, scoring=runnable, unreachable=unreachable,
            transport={}, on_result=None)
    finally:
        collector.stop()
        srv.shutdown()

    if save:
        from sunflower_thresh import runbundle

        bundle = runbundle.write(
            Path(save) / rid,
            runbundle.Manifest(
                run_id=rid, quotable=quotable,
                not_quotable_because=("" if quotable else
                                      "the bundled lab is not a client target"),
                trials=trials, cases=sorted(runnable),
                skipped=dict(unreachable or {}),
                target_family="sunflower-thresh bundled lab",
                target_config=f"safety={safety}, scripted brain",
                started_at=started, finished_at=_now()),
            findings)
        say(f"\nbundle     {bundle}  "
            f"({'quotable' if quotable else 'not quotable'})")

    _print_table(findings, stats, say)
    return 0


def _print_table(findings, stats, say) -> None:
    rows = []
    for f in findings:
        t = f.trials
        # The case id the finding carries, not the finding id with a prefix
        # chopped off it. The old line stripped a literal "demo-", which was
        # correct only while the run id was the literal string "demo": the moment
        # a run got a real id the table printed `2026-10-18T12-00-32_chat-crescendo`
        # as the case name. A finding knows which case it came from, so ask it.
        cid = (f.corpus_id or (f.test_cases[0] if f.test_cases else f.id))
        p = "" if t.p_value is None else f"{t.p_value:.3f}"
        rows.append((cid, f.evidence_strength.name,
                     f"{t.hits}/{t.trials}", f"{t.control_hits}/{t.control_trials}",
                     p, t.control_kind, _verdict(f)))
    rows.sort(key=lambda r: r[0])

    head = (f"{'case':30} {'strongest rung':14} {'attack':>8} {'control':>8} "
            f"{'p':>6} {'control':10} verdict")
    say(head)
    say("-" * len(head))
    for cid, rung, atk, ctl, p, kind, verdict in rows:
        say(f"{cid:30} {rung:14} {atk:>8} {ctl:>8} {p:>6} {kind:10} {verdict}")

    for cid, why in sorted(stats.get("unreachable_ids", {}).items()):
        say(f"{cid:30} {'-':14} {'':>8} {'':>8} {'':>6} {'-':10} not probed")
        # The row already says "not probed", so the reason does not need to
        # open by saying it again.
        reason = why.removeprefix("unreachable: ")
        say(f"{'':30}   {reason[:100]}")

    louder = [r[0] for r in rows if _ratio(r[3]) > _ratio(r[2])]
    if louder:
        say("")
        say("Note: the control arm fired MORE often than the attack for "
            f"{', '.join(louder)}.")
        say("That is a statement about the target, not a glitch: it does this on a")
        say("benign request, so the technique is not merely ineffective, it is")
        say("beside the point. Worth reporting as the target's own behaviour.")

    say("")
    say(f"{stats['considered']} cases considered: "
        f"{stats['validated']} validated, {stats['rejected']} rejected, "
        f"{stats['unreachable']} not probed, {stats['denied']} refused by the gate, "
        f"{stats['errors']} errored.  {stats['seconds']}s")
    say("")
    say("Read the two hit columns together, never the first alone.")
    say("A case whose control arm fired as often as the attack is REJECTED: the")
    say("same request with the technique removed produced the same result, so the")
    say("technique is not what caused it. That is the difference between a")
    say("measurement and a screenshot, and it is why `not probed` is its own")
    say("row rather than a zero.")
