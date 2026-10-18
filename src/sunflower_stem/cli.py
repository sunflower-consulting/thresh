"""The `stem` command.

Subcommands are registered here only once they work. A help listing that
advertises a command which then prints "not implemented" costs a reader more
trust than the missing feature costs them time, and this tool's entire argument
is about not overclaiming.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sunflower_stem import __version__
from sunflower_stem.calibrate import MOCK_JUDGES
from sunflower_stem.corpus import CORPUS_ENV, corpus_root_from_env
from sunflower_stem.judge import ENDPOINT_ENV, MODEL_ENV


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stem",
        description="Measure whether an attack on an LLM application actually "
                    "worked, against a control, proved out of band.",
        epilog="Authorized targets only. `stem doctor` reports what this "
               "installation can and cannot prove.",
    )
    p.add_argument("--version", action="version", version=f"stem {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    d = sub.add_parser("doctor", help="report what this installation can prove")
    d.set_defaults(func=_doctor)

    demo = sub.add_parser(
        "demo", help="run the built-in cases against the bundled target",
        description="Start the bundled vulnerable chat application, run every "
                    "built-in case and its benign control arm against it, and "
                    "report what was proved. Needs no API key, no target of "
                    "your own and no configuration.")
    demo.add_argument(
        "--safety", choices=("none", "weak", "strong"), default="none",
        help="how hard the bundled application defends itself. Running the same "
             "cases against 'strong' is the control for the defences themselves.")
    demo.add_argument(
        "--trials", type=int, default=5,
        help="attempts per case, and per control arm (default: 5). A rate needs "
             "a denominator worth having; 1 is not a measurement.")
    demo.add_argument(
        "--save", metavar="DIR",
        help="write a run bundle under DIR/<run-id>: the manifest, the findings "
             "and the events. Off by default, because the quickstart should leave "
             "nothing behind -- but a run nobody wrote down cannot be exported, "
             "quoted or diffed against.")
    demo.set_defaults(func=_demo)

    lab = sub.add_parser(
        "lab", help="serve the bundled vulnerable chat app on loopback",
        description="Serve the deliberately vulnerable chat application that "
                    "ships with stem. Loopback only, and no model is needed.")
    lab.add_argument("--port", type=int, default=5005)
    lab.add_argument(
        "--safety", choices=("none", "weak", "strong"), default="none",
        help="how hard the app defends itself. Running the same assessment "
             "against 'strong' is the control for the defences themselves: if "
             "the result does not move, the assessment is measuring something "
             "other than what the defences do.")
    lab.add_argument(
        "--model", default="",
        help="use a local Ollama model as the app's brain instead of the "
             "scripted one. Optional, and off by default: the scripted brain "
             "needs no model, no GPU and no network, which is what makes a run "
             "reproducible and what makes the quickstart keyless.")
    lab.set_defaults(func=_lab)

    judge = sub.add_parser(
        "judge", help="report the judge rung's configuration and rubric library",
        description="What the judge rung would do if a case asked for it. Reads "
                    "the rubric library, resolves the configured endpoint, and "
                    "says per rubric whether it is reachable. The rung is off "
                    "until a per-rubric agreement figure is published, so this "
                    "normally reports every rubric as withheld, which is the "
                    "honest state rather than a fault.")
    judge.add_argument("--endpoint", default="",
                       help=f"OpenAI-compatible base URL; overrides ${ENDPOINT_ENV}.")
    judge.add_argument("--model", default="",
                       help=f"served model name; overrides ${MODEL_ENV}.")
    judge.add_argument("--probe", action="store_true",
                       help="also reach the endpoint. This is the only part of "
                            "this command that touches the network, which is why "
                            "it is a flag: `stem doctor` and the listing below "
                            "must both work offline.")
    judge.set_defaults(func=_judge)

    cal = sub.add_parser(
        "calibrate", help="report whether each judge rubric's threshold is calibrated",
        description="Run every rubric against the bundled lab at two guardrail "
                    "levels and report, per rubric, whether it separates a "
                    "hardened target from an unguarded one. With no endpoint "
                    "configured this runs the DRY PASS: three constant judges "
                    "whose read-out proves the detector works before anyone "
                    "spends GPU time on the real question. Produces a "
                    "discrimination verdict, never an agreement figure.")
    cal.add_argument("--dry", action="store_true",
                     help="force the dry pass even when an endpoint is configured.")
    cal.add_argument("--judge", default="",
                     help="one constant judge by name, instead of all three: "
                          + ", ".join(MOCK_JUDGES))
    cal.add_argument("--endpoint", default="",
                     help=f"OpenAI-compatible base URL; overrides ${ENDPOINT_ENV}.")
    cal.add_argument("--model", default="",
                     help=f"served model name; overrides ${MODEL_ENV}.")
    cal.set_defaults(func=_calibrate)

    ex = sub.add_parser(
        "export", help="write one run's measurements into an engagement directory",
        description="Write a run's numbers where a document generator reads them. "
                    "Writes `<engagement>/runs/<run-id>/` and never the engagement "
                    "root, so retests and tuning runs do not collide and nothing "
                    "needs overwriting. `stem consolidate` produces the single "
                    "file a report renders from.")
    ex.add_argument("run_id", help="a run id under --runs-dir, or a path to a bundle")
    ex.add_argument("engagement_dir", help="the engagement directory to write into")
    ex.add_argument("--runs-dir", default="runs",
                    help="where run bundles live (default: runs/)")
    ex.set_defaults(func=_export)

    con = sub.add_parser(
        "consolidate", help="build the engagement's measurements from its quotable runs",
        description="Read every exported run marked quotable and write the one "
                    "measurements file a document reads. Idempotent: delete the "
                    "output, run it again, get an identical file. Runs marked not "
                    "quotable stay on disk and never reach a client, which is what "
                    "makes testing the same case five times safe.")
    con.add_argument("engagement_dir")
    con.set_defaults(func=_consolidate)

    cor = sub.add_parser(
        "corpus", help="load a corpus checkout and report what it would run",
        description="Read a corpus of test cases, bind its placeholders to this "
                    "engagement's own values, and report what would run and what "
                    "would not. Sends nothing. A case with an unbound placeholder "
                    "is reported UNREACHABLE rather than sent half-expanded, which "
                    "is the point of looking before a run.")
    cor.add_argument("--corpus", metavar="DIR", default="",
                     help=f"the checkout. Defaults to ${CORPUS_ENV}.")
    cor.add_argument("--bindings", metavar="FILE", type=Path,
                     help="YAML of placeholder values. Lowest precedence: a plan "
                          "beats it and the signed inventory beats both.")
    cor.add_argument("--collector", action="store_true",
                     help="the engagement has a callback collector, so the cases "
                          "that need one become reachable. Their URL is minted per "
                          "trial and cannot be hand-bound: a hardcoded host proves "
                          "something fetched it, not which case caused the fetch.")
    cor.add_argument("--only", metavar="IDS",
                     help="comma-separated case ids, instead of every active case")
    cor.add_argument("--show", action="store_true",
                     help="print each case's expanded turns. This is the "
                          "sign-off view: what would actually be sent.")
    cor.add_argument("--allow-broken", action="store_true",
                     help="proceed when a case fails to load. Refused by default, "
                          "because 243 of 244 reads exactly like a corpus and the "
                          "case it dropped is the case nobody ran.")
    cor.set_defaults(func=_corpus)
    return p


def _demo(args: argparse.Namespace) -> int:
    from sunflower_stem.demo import run

    if args.trials < 1:
        print("--trials must be at least 1", file=sys.stderr)
        return 2
    # No `--quotable` here, deliberately. `quotable` means "may be quoted to a
    # client" and the bundled lab is the target stem ships so it can attack
    # itself, so a run against it can never be evidence in a deliverable. The
    # manifest records that reason rather than leaving a reader to infer it from
    # a flag they did not pass.
    return run(safety=args.safety, trials=args.trials, save=args.save)


def _lab(args: argparse.Namespace) -> int:
    from sunflower_stem.lab import serve

    # Loopback is not a flag. This is an application built to be exploitable,
    # and a --host option is how one ends up answering on a laptop's wifi
    # address in a cafe. Anyone who genuinely needs it reachable can forward a
    # port deliberately, which is a decision with a person attached to it.
    srv, _app = serve(port=args.port, safety=args.safety,
                     deterministic=not args.model,
                     **({"model": args.model} if args.model else {}))
    brain = f"model={args.model}" if args.model else "scripted brain (no model)"
    port = srv.server_address[1]
    # flush=True because what follows is serve_forever(): without it the banner
    # sits in a buffer until the process exits, so anyone piping or redirecting
    # the output sees nothing at all while the server is in fact up.
    print(f"lab on http://127.0.0.1:{port}  safety={args.safety}  {brain}", flush=True)
    print(f"  scoreboard: http://127.0.0.1:{port}/api/Challenges", flush=True)
    print("  stop with ctrl-c", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
        print()
    return 0


def _judge(args: argparse.Namespace) -> int:
    """Say what the judge rung would do, and why it will probably do nothing."""
    import os

    from sunflower_stem.judge import LOAD_ERRORS, load_rubrics
    from sunflower_stem.modelcall import ChatTransport

    endpoint = args.endpoint or os.environ.get(ENDPOINT_ENV, "").strip()
    model = args.model or os.environ.get(MODEL_ENV, "").strip()
    rubrics = load_rubrics()

    print(f"rubric library:   {len(rubrics)} loaded from the installed package")
    for name, err in LOAD_ERRORS:
        print(f"  ! {name}: {err}")
    calibrated = sorted(r.id for r in rubrics.values() if r.calibrated)
    print(f"calibrated:       {len(calibrated)} of {len(rubrics)}"
          f"{': ' + ', '.join(calibrated) if calibrated else ''}")
    if not calibrated:
        # Said plainly, because a reader who points stem at a working endpoint and
        # sees every case report `not probed` will otherwise think it is broken.
        print("                  every rubric is WITHHELD: a rung whose agreement "
              "with a human")
        print("                  has never been measured scores nothing, so a case "
              "resting on")
        print("                  one reports as not probed rather than as a pass.")

    # Per rubric, because "uncalibrated" is one word for three different situations
    # and they send a reader to three different places: measure it, measure more of
    # it, or fix the rubric. The polarity is shown beside it because it decides
    # which arm a calibration run expects the rubric to fire on, and an inverse
    # rubric read as positive inverts its whole read-out.
    print()
    width = max(len(name) for name in rubrics)
    print(f"  {'rubric':<{width}}  {'polarity':<9}  state")
    print("  " + "-" * (width + 58))
    for name in sorted(rubrics):
        r = rubrics[name]
        state = "CALIBRATED" if r.calibrated else f"withheld: {r.withheld_because}"
        print(f"  {name:<{width}}  {r.polarity:<9}  {state}")
    print()
    print(f"endpoint:         {endpoint or f'not configured (set ${ENDPOINT_ENV})'}")
    print(f"model:            {model or f'not configured (set ${MODEL_ENV})'}")

    if not args.probe:
        return 0
    # Flushed before every stderr write below. Redirected stdout is block-buffered
    # while stderr is not, so without this the failure line lands ABOVE the listing
    # it refers to and reads as though it came first. `_lab` carries the same
    # flushing for the same reason.
    if not (endpoint and model):
        sys.stdout.flush()
        print("\nnothing to probe: configure an endpoint and a model first.",
              file=sys.stderr)
        return 2
    result = ChatTransport(endpoint, model).probe()
    if not result.get("ok"):
        sys.stdout.flush()
        print(f"\nprobe:            UNREACHABLE  {result.get('error', '')}",
              file=sys.stderr)
        return 1
    served = result.get("served_models") or []
    print(f"\nprobe:            reachable in {result.get('latency_ms')} ms, "
          f"{len(served)} model(s) served")
    if model not in served:
        print(f"                  ! {model!r} is not in the listing")
    # Named rather than implied: on the reference box a server launched outside its
    # tray app lists every model and returns HTTP 500 on the largest one. A listing
    # is what succeeds while the model you need is the one thing that is broken.
    print("                  a listing is not a generation: only a completion "
          "against the")
    print("                  model you need proves it can run.")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    """The dry pass by default, the live one when an endpoint says so."""
    import os

    from sunflower_stem.calibrate import (
        collect_replies,
        mock_judge,
        rehearse,
        render,
    )
    from sunflower_stem.judge import Judge, load_rubrics
    from sunflower_stem.modelcall import ChatTransport

    rubrics = load_rubrics()
    if not rubrics:
        print("no rubrics to calibrate: the installed package has none.",
              file=sys.stderr)
        return 2
    if args.judge and args.judge not in MOCK_JUDGES:
        print(f"unknown constant judge {args.judge!r}; choose from "
              f"{', '.join(MOCK_JUDGES)}", file=sys.stderr)
        return 2

    endpoint = args.endpoint or os.environ.get(ENDPOINT_ENV, "").strip()
    model = args.model or os.environ.get(MODEL_ENV, "").strip()
    dry = args.dry or bool(args.judge) or not (endpoint and model)

    # Collected once and reused across judges: the lab's scripted brain is
    # deterministic, so three judges scoring the SAME replies is what makes their
    # read-outs comparable. Re-collecting per judge would also triple the run for
    # identical input.
    replies = collect_replies()

    if dry:
        if not (endpoint and model):
            print(f"no endpoint configured (${ENDPOINT_ENV}, ${MODEL_ENV}), so this "
                  f"is the dry pass.\n", flush=True)
        names = [args.judge] if args.judge else list(MOCK_JUDGES)
        failures = []
        for name in names:
            _reply, expected = MOCK_JUDGES[name]
            verdicts = rehearse(mock_judge(name, rubrics), rubrics=rubrics,
                                replies=replies)
            got = {v.verdict for v in verdicts}
            # The dry pass INVERTS the usual success condition: a constant judge is
            # supposed to provoke a pathology, and the detector is sound only if it
            # reports that pathology for every rubric. A clean read-out here would
            # mean the detector cannot see anything.
            ok = got == {expected}
            print(render(verdicts, judge_name=name, replies=len(replies)), flush=True)
            print(f"\nexpected every rubric {expected}: "
                  f"{'OK' if ok else 'MISMATCH, got ' + ', '.join(sorted(got))}\n",
                  flush=True)
            if not ok:
                failures.append(name)
        if failures:
            sys.stdout.flush()
            print(f"the detector is not sound: {', '.join(failures)} did not produce "
                  f"the read-out it must.", file=sys.stderr)
            return 1
        print("the detector behaves. A live run against a real judge is now worth "
              "the GPU time.")
        return 0

    judge = Judge(ChatTransport(endpoint, model), rubrics=rubrics,
                  require_calibrated=False)
    verdicts = rehearse(judge, rubrics=rubrics, replies=replies)
    print(render(verdicts, judge_name=f"{model} at {endpoint}", replies=len(replies)))
    # Non-zero when there is something to fix, so this can gate a pipeline.
    return 1 if any(not v.ok for v in verdicts) else 0


def _corpus(args: argparse.Namespace) -> int:
    """Report what a corpus would run. Sends nothing."""
    from collections import Counter

    import yaml

    from sunflower_stem.corpus import (
        COLLECTOR_PLACEHOLDERS,
        CorpusError,
        load_chat_cases,
        resolve_bindings,
    )

    root = args.corpus or corpus_root_from_env()
    if not root:
        print(f"no corpus. Pass --corpus or set ${CORPUS_ENV}.", file=sys.stderr)
        return 2
    file_bindings = {}
    if args.bindings:
        try:
            file_bindings = yaml.safe_load(args.bindings.read_text()) or {}
        except OSError as exc:
            print(f"cannot read {args.bindings}: {exc}", file=sys.stderr)
            return 2
    bindings, canaries = resolve_bindings(file_bindings=file_bindings)

    try:
        cases, corpus, skipped = load_chat_cases(
            root, bindings=bindings, canaries=canaries,
            have_collector=args.collector,
            only=[c.strip() for c in args.only.split(",")] if args.only else None,
            allow_broken=args.allow_broken)
    except CorpusError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    unreachable = {cid: [n for n in c.needs if n.startswith("binding:")]
                   for cid, c in cases.items()}
    blocked = {cid: n for cid, n in unreachable.items() if n}
    print(f"corpus      {corpus.root}  at {corpus.sha}")
    print(f"cases       {len(corpus)} loaded, {len(corpus.broken)} broken")
    print(f"selected    {len(cases)}, {len(skipped)} not selected")
    families: dict[str, list[str]] = {}
    for cid, why in sorted(skipped.items()):
        families.setdefault(why.split(":", 1)[0].strip(), []).append(cid)
    for family, ids in sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        named = ", ".join(ids[:4]) + (f", +{len(ids) - 4} more" if len(ids) > 4 else "")
        print(f"            {len(ids)}x {family}: {named}")
    print(f"unbound     {len(blocked)} case(s) would be UNREACHABLE for want of a "
          f"binding")
    missing = Counter(name for needs in blocked.values() for need in needs
                      for name in need.removeprefix("binding:").split(","))
    # Two different operator actions, and conflating them sends somebody to write a
    # value that must not be written. A collector placeholder is minted per trial;
    # binding it by hand gives every case the same URL, and a hit on a shared host
    # proves that something fetched it rather than that this case caused the fetch.
    for label, names in (
            ("write a value", [(k, v) for k, v in missing.most_common()
                               if k not in COLLECTOR_PLACEHOLDERS]),
            ("run --collector", [(k, v) for k, v in missing.most_common()
                                 if k in COLLECTOR_PLACEHOLDERS])):
        if names:
            print(f"            {label}: " + ", ".join(
                f"{k} ({v})" for k, v in names[:8])
                + (f", +{len(names) - 8} more" if len(names) > 8 else ""))
    # A reachable case is not an automatic case. A `seeded` or `inseparable` control
    # is the corpus stating that no matched control can be SENT and why, and without
    # a control arm nothing reaches VALIDATED -- so a count of reachable cases alone
    # implies more than a run can deliver, which is the readout defect this tool
    # exists to catch in other tools.
    reachable = [c for cid, c in cases.items() if not unreachable[cid]]
    by_kind = Counter(c.control_kind or "none" for c in reachable
                      if not c.control_turns)
    if by_kind:
        print(f"controls    {sum(1 for c in reachable if c.control_turns)} of "
              f"{len(reachable)} reachable case(s) can send a matched control")
        for kind, n in by_kind.most_common():
            why = {"seeded": "the control is a re-plant an operator does out of band",
                   "inseparable": "the technique cannot be removed and leave the "
                                  "request intact",
                   "none": "NOBODY WROTE ONE -- this is a gap, not a declaration"}
            print(f"            {n}x {kind}: {why.get(kind, kind)}")
    if corpus.unmapped_taxa():
        print(f"no taxon    {len(corpus.unmapped_taxa())} case(s): a reporting gap, "
              f"not a reason to drop a measurement")
    if not args.show:
        print("\n--show prints the expanded turns, which is the sign-off view: "
              "what would actually be sent.")
        return 0
    for cid in sorted(cases):
        case = cases[cid]
        flag = "  UNREACHABLE" if unreachable[cid] else ""
        print(f"\n=== {cid}  {case.taxon or 'no taxon'}{flag}")
        for i, turn in enumerate(case.turns, start=1):
            print(f"  [{i}] {' '.join(turn.content.split())[:150]}")
        if case.seed_note:
            print(f"  PLANT FIRST: {' '.join(case.seed_note.split())[:150]}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from sunflower_stem import runbundle
    from sunflower_stem.export import write_run

    bundle = Path(args.run_id)
    if not bundle.is_dir():
        bundle = Path(args.runs_dir) / args.run_id
    if not bundle.is_dir():
        print(f"no run bundle at {bundle}. `stem demo --save runs/` writes one.",
              file=sys.stderr)
        return 2
    manifest, findings = runbundle.load(bundle)
    if not manifest:
        print(f"{bundle} has no manifest, so it is not a finished run. A run "
              f"killed partway through leaves events and no manifest.",
              file=sys.stderr)
        return 2
    try:
        out = write_run(args.engagement_dir, manifest, findings)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rows = len((out / "measurements.yaml").read_text().splitlines())
    quotable = manifest.get("quotable") is True
    print(f"wrote {out}")
    print(f"  {len(findings)} finding(s), {rows} lines of measurements")
    if not quotable:
        why = manifest.get("not_quotable_because") or "not marked quotable"
        print(f"  NOT QUOTABLE ({why}), so `stem consolidate` will skip it. The "
              f"run stays on disk and stays diffable.")
    return 0


def _consolidate(args: argparse.Namespace) -> int:
    from sunflower_stem.consolidate import write

    try:
        out, notes = write(args.engagement_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    import yaml

    doc = yaml.safe_load(out.read_text()) or {}
    cases = doc.get("cases") or []
    quoted = doc.get("runs_quoted") or []
    print(f"wrote {out}")
    print(f"  {len(cases)} case(s) from {len(quoted)} quotable run(s)")
    for note in notes:
        print(f"  - {note}")
    if not cases:
        # Said rather than left as an empty file, because an empty appendix and a
        # missing one look identical in a PDF and mean different things.
        print("  nothing was consolidated. A report rendering this would show an "
              "empty measurement appendix,")
        print("  which reads like a clean result and is not one.")
    return 0


def _doctor(_args: argparse.Namespace) -> int:
    from sunflower_stem.doctor import report

    print(report())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
