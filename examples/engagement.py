#!/usr/bin/env python3
"""One engagement, driven from a script. `thresh` as a library rather than a command.

    python examples/engagement.py [--out DIR]

WHY THIS FILE EXISTS.

The commands are a convenience over this. Everything `thresh` does is reachable from
Python, and a real engagement is a script: the client's endpoint, their signed scope,
their canary, the cases that apply to them, and whatever reporting they bought. The
only other end-to-end caller in the package is `demo.py`, which is wired to the
bundled lab and builds its inventory in code -- useful as a smoke test and no use as
a pattern. This is the pattern.

It runs against the bundled lab so it works anywhere, and the two lines a real
engagement changes are marked. A test executes it, so it cannot rot into advice that
no longer works.

THE ORDER MATTERS, and it is the one thing a reader should take away. Six objects,
and each one exists because something downstream refuses to work without it:

    1. the inventory      the signed scope. The gate refuses what it does not permit
    2. the evidence store  content-addressed, redacted, with the canary kept by name
    3. the policy engine   the gate, which must VERIFY the inventory before anything
    4. the context         the one path a tool call reaches a target through
    5. the selection       which cases this engagement can actually run, and why not
    6. the run bundle      written once, and the input to export, diff and consolidate

What is NOT here, deliberately: a facade. There is no single object you construct and
drive, the way PyRIT gives you an orchestrator. The assembly below IS the API, and it
is this explicit because every step is a decision somebody has to be able to point
at afterwards -- which scope, which evidence policy, which cases, which budget.
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

from sunflower_thresh import runbundle
from sunflower_thresh.collector import CallbackCollector
from sunflower_thresh.context import AgentContext
from sunflower_thresh.evidence.store import EvidenceStore
from sunflower_thresh.lab import serve
from sunflower_thresh.policy.engine import PolicyEngine
from sunflower_thresh.preflight import (
    EngagementPreflight,
    load_inventory,
    transport_for,
)
from sunflower_thresh.preflight import render as render_preflight
from sunflower_thresh.redact import make_redactor
from sunflower_thresh.sweep import applicable_chat_cases, sweep_builtins
from sunflower_thresh.tools.conversation import (
    ChatOpen,
    ChatReset,
    ChatSay,
    ConversationStore,
)

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scope", type=Path, default=HERE / "scope.yaml")
    ap.add_argument("--out", type=Path, default=Path("runs"),
                    help="where to write the run bundle")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--corpus", default="",
                    help="a case library to run instead of the built-ins")
    args = ap.parse_args(argv)

    # 1. THE SIGNED SCOPE, read from the document it was signed as. An unknown field
    #    is refused rather than ignored, so a typo cannot quietly widen the scope.
    inv = load_inventory(args.scope)
    canary = dict(inv.seeds).get("canary", "")

    # --- the two lines a real engagement changes -----------------------------------
    # A client run points at their endpoint and does not stand up a target at all:
    #     host, port = "chat.client.example", inv.port_for_chat()
    srv, _lab = serve(port=0, deterministic=True, safety="weak")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = "127.0.0.1", srv.server_address[1]
    # -------------------------------------------------------------------------------

    # 2. EVIDENCE. Redacted on the way in, with the canary protected BY NAME: it is a
    #    high-entropy token, so it looks exactly like the secrets the patterns hunt,
    #    and redacting it would destroy the evidence for the strongest rung there is.
    store = EvidenceStore(redactor=make_redactor(keep=(canary,) if canary else ()))

    # The collector is optional and gated: binding anything but loopback needs
    # `out_of_band_callback` signed, and an unreachable one costs the callback rung.
    collector = CallbackCollector(store=store, inventory=inv).start()

    # 3. THE GATE, which has to verify the inventory before it permits anything. A
    #    lapsed authorization makes `verified` false and every call is then refused.
    policy = PolicyEngine()
    verified = policy.load_inventory(inv)

    # 4. THE CONTEXT. Every tool call goes through `execute_tool`, which authorizes
    #    first. There is no second path, and that is the whole safety argument.
    convos = ConversationStore()
    ctx = AgentContext(inv.project, policy, tool_impls={
        "chat_open": ChatOpen(convos, store, policy=policy),
        "chat_say": ChatSay(convos, store, policy=policy),
        "chat_reset": ChatReset(convos, store, policy=policy)})

    # The case library: the built-ins, or a corpus checkout.
    cases = None
    if args.corpus:
        from sunflower_thresh.corpus import load_chat_cases, resolve_bindings

        bindings, canaries = resolve_bindings(inventory_seeds=dict(inv.seeds))
        cases, _corpus, _skipped = load_chat_cases(
            args.corpus, bindings=bindings, canaries=canaries)

    # 5. LOOK BEFORE SENDING. Preflight predicts every refusal the gate can give and
    #    stops before the remote checks if the scope does not cover the target, so a
    #    host nobody authorised is never probed to find out whether it answers.
    pf = EngagementPreflight(inv=inv, host=host, port=port, verified=verified,
                             collector=collector, cases=cases or {},
                             trials=0 if args.trials <= 2 else args.trials)
    results = pf.run()
    print(render_preflight(results, title=f"preflight -- {inv.project}"))
    print()

    # 6. SELECT, RUN, WRITE. The selection answers "what can we test" once, here, and
    #    travels into the sweep so the ladder is built from the oracles this run can
    #    actually score. `unreachable` is kept, because a case that did not run is
    #    not a case that found nothing.
    runnable, unreachable = applicable_chat_cases(
        inv.allowed_tools, capabilities=inv,
        have_collector=False,          # measured by preflight; see its collector row
        have_tool_visibility=True,
        cases=cases, seeds=dict(inv.seeds))

    rid = runbundle.run_id(inv.project)
    try:
        findings, _stats = sweep_builtins(
            ctx, runnable, target_ip=host, port=port, run_id=rid,
            trials=args.trials, collector=collector, seeds=dict(inv.seeds),
            cases=cases, scoring=runnable, unreachable=unreachable,
            # From the signed scope, not retyped: tls, verification, path and port
            # all come from the document preflight just checked.
            transport=transport_for(inv))
    finally:
        collector.stop()
        srv.shutdown()

    bundle = runbundle.write(args.out / rid, runbundle.Manifest(
        run_id=rid, trials=args.trials, cases=sorted(runnable),
        skipped=dict(unreachable),
        # A client run sets this true. The bundled lab is not a client target, and
        # `consolidate` reads this flag rather than a judgement made later.
        quotable=False,
        not_quotable_because="the bundled lab is not a client target",
    ), findings)

    print(f"{len(runnable)} case(s) run, {len(unreachable)} not reachable, "
          f"{len(findings)} finding(s)")
    for f in findings:
        case = f.corpus_id or (f.test_cases[0] if f.test_cases else f.id)
        print(f"  {case:24} {f.state.value:11} {f.title[:48]}")
    print(f"\nbundle: {bundle}")
    print("\nnext, with the same library:\n"
          "  thresh export <bundle> <engagement>   measurements + control mapping\n"
          "  thresh consolidate <engagement>       the engagement's machine files\n"
          "  thresh diff <before> <after>          what closed, and what only narrowed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
