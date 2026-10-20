"""The `thresh` command.

Subcommands are registered here only once they work. A help listing that
advertises a command which then prints "not implemented" costs a reader more
trust than the missing feature costs them time, and this tool's entire argument
is about not overclaiming.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sunflower_thresh import __version__
from sunflower_thresh.calibrate import MOCK_JUDGES
from sunflower_thresh.corpus import CORPUS_ENV, corpus_root_from_env
from sunflower_thresh.judge import ENDPOINT_ENV, MODEL_ENV, judge_env


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="thresh",
        description="Measure whether an attack on an LLM application actually "
                    "worked, against a control, proved out of band.",
        epilog="Authorized targets only. `thresh doctor` reports what this "
               "installation can and cannot prove.",
    )
    p.add_argument("--version", action="version", version=f"thresh {__version__}")
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
                    "ships with thresh. Loopback only, and no model is needed.")
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
                            "it is a flag: `thresh doctor` and the listing below "
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
                    "needs overwriting. `thresh consolidate` produces the single "
                    "file a report renders from.")
    ex.add_argument("run_id", help="a run id under --runs-dir, or a path to a bundle")
    ex.add_argument("engagement_dir", help="the engagement directory to write into")
    ex.add_argument("--runs-dir", default="runs",
                    help="where run bundles live (default: runs/)")
    ex.add_argument("--snapshot", metavar="FILE", type=Path,
                    help="an obligation catalogue other than the vendored one. "
                         "Verified against its own digest either way.")
    ex.add_argument("--no-control-mapping", action="store_true",
                    help="write only the measurements. The mapping is written "
                         "whenever the engagement file says which regimes the "
                         "client carries, because a run that produced evidence "
                         "about an obligation and did not say so is the gap this "
                         "whole seam exists to close.")
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

    pre = sub.add_parser(
        "preflight", help="can this engagement run, and what will it send",
        description="Check one engagement before it spends a message budget. "
                    "Answers four questions: is the scope sound, can we reach them, "
                    "what can we test, and what will it cost. Every check that can "
                    "be answered offline is, and the only thing sent to the target "
                    "is one ordinary question -- last, and only once everything "
                    "before it passed.")
    pre.add_argument("--scope", metavar="FILE", type=Path, required=True,
                     help="the signed scope document: a YAML target inventory.")
    pre.add_argument("--corpus", metavar="DIR", default="",
                     help=f"the case library. Defaults to ${CORPUS_ENV}. Without "
                          f"one the built-in cases are used, which answers the shape "
                          f"of the report and not the size of the run.")
    pre.add_argument("--bindings", metavar="FILE", type=Path,
                     help="YAML of placeholder values. The scope document's `seeds` "
                          "beat it, because the inventory is the signed record.")
    pre.add_argument("--target", metavar="HOST", default="",
                     help="the chat endpoint's host. Defaults to the single entry in "
                          "the scope's allowed_domains when there is exactly one.")
    pre.add_argument("--port", type=int, default=0,
                     help="default 443 with tls, else 80.")
    pre.add_argument("--trials", type=int, default=0,
                     help="override every case's own trial count. Off by default, "
                          "and leaving it off is what keeps the two-trial cap on "
                          "cases that measure cost rather than disclosure.")
    pre.add_argument("--collector", action="store_true",
                     help="stand up a callback collector and PROBE it, rather than "
                          "asserting it is reachable. Binding anything but loopback "
                          "also needs out_of_band_callback in the scope.")
    pre.add_argument("--collector-host", metavar="HOST", default="127.0.0.1",
                     help="what the collector binds. Loopback by default, which "
                          "nothing outside this machine can reach.")
    pre.add_argument("--advertise", metavar="URL", default="",
                     help="the collector URL cases are given, when it differs from "
                          "the bind address. Probed, and only if it names this host.")
    pre.add_argument("--tool-visibility", action="store_true",
                     help="the target's replies report which tools it called.")
    pre.add_argument("--judge", action="store_true",
                     help="a calibrated judge is available to this run.")
    pre.add_argument("--insecure", action="store_true",
                     help="do not verify the certificate. Recorded in the readout, "
                          "because a finding attributed to a host whose identity was "
                          "not checked is a weaker finding.")
    pre.add_argument("--mock", metavar="PROFILE", default="",
                     help="rehearse against a fake target instead of sending "
                          "anything: healthy, instrumented, stateless, unauthorised, "
                          "streaming, nested, silent.")
    pre.set_defaults(func=_preflight)

    obl = sub.add_parser(
        "obligations", help="what a run's results mean against a client's duties",
        description="Resolve techniques against the vendored obligation catalogue. "
                    "Reports four states per obligation -- tested, probed-clean, "
                    "not-probed, review-only -- and three per technique: mapped, "
                    "declined with the reason the catalogue gave, or unknown to it. "
                    "Claims no obligation for an unknown technique and infers none "
                    "from a sibling, because a mapping is a regulatory judgement "
                    "rather than a prefix match.")
    obl.add_argument("--snapshot", metavar="FILE", type=Path,
                     help="an obligation snapshot other than the vendored one. "
                          "Verified against its own digest either way.")
    obl.add_argument("--corpus", metavar="DIR", default="",
                     help=f"a case library whose taxa to resolve. Defaults to "
                          f"${CORPUS_ENV}.")
    obl.add_argument("--findings", metavar="FILE", type=Path,
                     help="YAML list of findings, each with an `id` and a `taxon`. "
                          "Pass the engagement's written-up findings.yaml: the "
                          "appendix has to be cross-referenceable by the ids a "
                          "reader sees in the findings table.")
    obl.add_argument("--instrument", action="append", default=[], metavar="NAME",
                     help="scope to this instrument; repeatable. Without any, the "
                          "denominator is the whole catalogue, which shows a client "
                          "duties they do not carry.")
    obl.add_argument("--system-class", action="append", default=[], metavar="NAME",
                     help="scope to this system class; repeatable.")
    obl.add_argument("--list-instruments", action="store_true",
                     help="print the instruments and system classes the catalogue "
                          "holds, with a count of obligations for each, and exit.")
    obl.set_defaults(func=_obligations)

    cls = sub.add_parser(
        "classify", help="which regimes a system is in scope of, from the rules",
        description="Answer the obligation catalogue's own classification "
                    "questions and get the system classes to scope a mapping "
                    "with. Until now that scope was typed from memory, and it is "
                    "the denominator of the only coverage figure a report may "
                    "state. Not legal advice: these are reusable questions "
                    "somebody still has to answer about a deployment.")
    cls.add_argument("--answers", metavar="FILE", type=Path,
                     help="YAML mapping of question id to true/false. "
                          "`--questions` prints the ids to fill in.")
    cls.add_argument("--ask", action="store_true",
                     help="answer them interactively.")
    cls.add_argument("--questions", action="store_true",
                     help="print the questions and exit, as a template to answer.")
    cls.add_argument("--out", metavar="FILE", type=Path,
                     help="also write the classification as YAML, for the "
                          "engagement file to record what was answered.")
    cls.add_argument("--snapshot", metavar="FILE", type=Path,
                     help="an obligation snapshot other than the vendored one.")
    cls.set_defaults(func=_classify)

    lead = sub.add_parser(
        "leads", help="triage another tool's hits against the case library",
        description="Read a garak or promptfoo report and record every hit as one "
                    "of three things: covered by a case that already exercises the "
                    "technique, promoted to a draft for an author, or dismissed "
                    "with the reason. NONE of it becomes a finding -- another "
                    "tool's verdict says where to point a measurement, not that "
                    "anything is true. Runs nothing and sends nothing.")
    lead.add_argument("--garak", metavar="FILE", type=Path, action="append",
                      default=[], help="a garak JSONL report; repeatable.")
    lead.add_argument("--promptfoo", metavar="FILE", type=Path, action="append",
                      default=[], help="a promptfoo results.json or .jsonl; "
                                       "repeatable.")
    lead.add_argument("--corpus", metavar="DIR", default="",
                      help=f"the case library to check coverage against. Defaults "
                           f"to ${CORPUS_ENV}. Without one every mapped lead is "
                           f"promoted, which overstates how much is new.")
    lead.add_argument("--out", metavar="FILE", type=Path,
                      help="where to write the triage. Defaults to "
                           "leads/triage.yaml under the current directory, which "
                           "`thresh consolidate` excludes in code.")
    lead.add_argument("--allow-nothing", action="store_true",
                      help="accept a report whose rows this reader could not "
                           "understand. A clean scan needs no flag: a report whose "
                           "rows are all understood and hold no hit is an ordinary "
                           "result. This is for one whose SHAPE has moved, which "
                           "neither tool's format carries the version information "
                           "to detect -- garak's carries none at all.")
    lead.set_defaults(func=_leads)

    dif = sub.add_parser(
        "diff", help="compare two runs: what closed, what is still there, what is new",
        description="Read two run bundles and report what changed. A finding is "
                    "CLOSED only when the proved case and its variants all failed "
                    "to reproduce; when a variant still lands it is NARROWED, which "
                    "is what a fix that blocked only the phrasing in the report "
                    "looks like. A case the retest never attempted is never closed.")
    dif.add_argument("before", help="the earlier run's bundle directory")
    dif.add_argument("after", help="the retest's bundle directory")
    dif.add_argument("--corpus", metavar="DIR", default="",
                     help=f"the case library, for the variant half of closure. "
                          f"Defaults to ${CORPUS_ENV}. WITHOUT IT NOTHING CAN BE "
                          f"CALLED CLOSED, because there is no way to tell a fixed "
                          f"technique from a fixed string.")
    dif.add_argument("--out", metavar="FILE", type=Path,
                     help="also write the comparison as YAML, for a document "
                          "generator.")
    dif.set_defaults(func=_diff_runs)
    return p


def _leads(args: argparse.Namespace) -> int:
    """Triage another tool's hits. Reads files, writes one; sends nothing."""
    from sunflower_thresh.leads import (
        LeadError,
        Outcome,
        read_garak,
        read_promptfoo,
        triage,
        write_triage,
    )

    if not (args.garak or args.promptfoo):
        print("nothing to read. Pass --garak and/or --promptfoo with a report file.",
              file=sys.stderr)
        return 2

    sources, leads = {}, []
    for label, paths, reader in (("garak", args.garak, read_garak),
                                 ("promptfoo", args.promptfoo, read_promptfoo)):
        for path in paths:
            try:
                res = reader(path, allow_unreadable=args.allow_nothing)
            except LeadError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            sources[f"{label}:{Path(path).name}"] = res
            leads.extend(res.leads)

    # The library decides `covered` against `promoted`, and without one every
    # mapped lead looks new. That is the expensive direction -- authoring a case
    # the library already has -- so it is stated rather than left implicit.
    covered_by = {}
    known_taxa = None
    root = args.corpus or corpus_root_from_env()
    if root:
        from sunflower_thresh.corpus import CorpusError, load_corpus

        try:
            corpus = load_corpus(root)
        except CorpusError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        for cid, case in corpus.cases.items():
            if case.taxon and case.selectable:
                covered_by.setdefault(case.taxon, []).append(cid)
    try:
        from sunflower_thresh.obligations import known_taxa as _known
        from sunflower_thresh.obligations import load_snapshot

        known_taxa = _known(load_snapshot())
    except Exception:  # noqa: BLE001 - no catalogue is not a reason to refuse leads
        known_taxa = None

    records = triage(leads, covered_by=covered_by.get, known_taxa=known_taxa)
    out = write_triage(args.out or (Path("leads") / "triage.yaml"), records,
                       sources=sources)

    counts = {}
    for r in records:
        counts[r.outcome.value] = counts.get(r.outcome.value, 0) + 1
    rows = sum(s.rows for s in sources.values())
    print(f"read        {rows} row(s) from {len(sources)} report(s)")
    print(f"leads       {len(records)}")
    for outcome in Outcome:
        print(f"  {outcome.value:9} {counts.get(outcome.value, 0)}")
    if not root:
        print("\nno corpus loaded, so nothing could be found already covered. "
              f"Pass --corpus or set ${CORPUS_ENV}.")
    print(f"\nwrote {out}")
    print("None of it is a finding. A promoted case is a draft with no control arm "
          "and no oracle,\nfor an author to accept or throw away.")
    # Zero. A dismissed lead is a recorded outcome rather than a fault, and exiting
    # non-zero for one would make the ordinary case look broken.
    return 0


def _diff_runs(args: argparse.Namespace) -> int:
    """Compare two runs. Reads two bundles, writes at most one file."""
    from sunflower_thresh.diff import as_document, diff_runs, render

    library = None
    root = args.corpus or corpus_root_from_env()
    if root:
        from sunflower_thresh.corpus import CorpusError, load_corpus

        try:
            library = load_corpus(root).cases
        except CorpusError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    d = diff_runs(args.before, args.after, library=library)
    print(render(d))
    if not root:
        sys.stdout.flush()
        print(f"\nno library loaded, so nothing can be reported closed: a case that "
              f"stopped reproducing\nis `fix unverified` instead. Pass --corpus or "
              f"set ${CORPUS_ENV}.", file=sys.stderr)
    if args.out:
        import yaml

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(_DIFF_HEADER + yaml.safe_dump(
            as_document(d), sort_keys=False, width=100, allow_unicode=True),
            encoding="utf-8")
        print(f"\nwrote {args.out}")
    # Non-zero when the retest cannot be read as a pass, so this can gate a release:
    # a NEW finding appeared, or something previously established was never retried.
    # Those are different facts and either one makes a green build a lie.
    return 1 if (d.regressed or d.incomplete) else 0


_DIFF_HEADER = (
    "# Generated by `thresh diff`. Do not edit.\n"
    "#\n"
    "# `closed` carries the variants it rested on. A finding closes only when the\n"
    "# proved case AND its variants failed to reproduce, because a client who\n"
    "# patched the exact prompt in the report makes one case stop while the\n"
    "# technique is untouched -- that is `narrowed`, and it is not a fix. Nothing\n"
    "# here reads a status anybody typed; every bucket is derived from the bundles.\n")


def _demo(args: argparse.Namespace) -> int:
    from sunflower_thresh.demo import run

    if args.trials < 1:
        print("--trials must be at least 1", file=sys.stderr)
        return 2
    # No `--quotable` here, deliberately. `quotable` means "may be quoted to a
    # client" and the bundled lab is the target thresh ships so it can attack
    # itself, so a run against it can never be evidence in a deliverable. The
    # manifest records that reason rather than leaving a reader to infer it from
    # a flag they did not pass.
    return run(safety=args.safety, trials=args.trials, save=args.save)


def _lab(args: argparse.Namespace) -> int:
    from sunflower_thresh.lab import serve

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

    from sunflower_thresh.judge import LOAD_ERRORS, load_rubrics
    from sunflower_thresh.modelcall import ChatTransport

    endpoint = args.endpoint or judge_env(ENDPOINT_ENV)
    model = args.model or judge_env(MODEL_ENV)
    rubrics = load_rubrics()

    print(f"rubric library:   {len(rubrics)} loaded from the installed package")
    for name, err in LOAD_ERRORS:
        print(f"  ! {name}: {err}")
    calibrated = sorted(r.id for r in rubrics.values() if r.calibrated)
    print(f"calibrated:       {len(calibrated)} of {len(rubrics)}"
          f"{': ' + ', '.join(calibrated) if calibrated else ''}")
    if not calibrated:
        # Said plainly, because a reader who points thresh at a working endpoint and
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

    from sunflower_thresh.calibrate import (
        collect_replies,
        mock_judge,
        rehearse,
        render,
    )
    from sunflower_thresh.judge import Judge, load_rubrics
    from sunflower_thresh.modelcall import ChatTransport

    rubrics = load_rubrics()
    if not rubrics:
        print("no rubrics to calibrate: the installed package has none.",
              file=sys.stderr)
        return 2
    if args.judge and args.judge not in MOCK_JUDGES:
        print(f"unknown constant judge {args.judge!r}; choose from "
              f"{', '.join(MOCK_JUDGES)}", file=sys.stderr)
        return 2

    endpoint = args.endpoint or judge_env(ENDPOINT_ENV)
    model = args.model or judge_env(MODEL_ENV)
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

    from sunflower_thresh.corpus import (
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


def _preflight(args: argparse.Namespace) -> int:
    """Check an engagement. Sends one ordinary question, last, or nothing at all."""
    import contextlib

    import yaml

    from sunflower_thresh.collector import CallbackCollector, CollectorRefused
    from sunflower_thresh.corpus import CorpusError, load_chat_cases, resolve_bindings
    from sunflower_thresh.preflight import (
        EngagementPreflight,
        ScopeError,
        Status,
        load_inventory,
        mock_sender,
        render,
    )

    try:
        inv = load_inventory(args.scope)
    except ScopeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cases, sha = {}, ""
    root = args.corpus or corpus_root_from_env()
    if root:
        file_bindings = {}
        if args.bindings:
            try:
                file_bindings = yaml.safe_load(args.bindings.read_text()) or {}
            except OSError as exc:
                print(f"cannot read {args.bindings}: {exc}", file=sys.stderr)
                return 2
        # `inventory_seeds` is the layer the CLI never passed. Without it `canaries`
        # is empty, so every `contains` rubric grades as STRUCTURAL where the
        # engagement's own canary would have made it CANARY -- the top of the ladder
        # reported as the second rung, on every extraction case in the library.
        bindings, canaries = resolve_bindings(
            file_bindings=file_bindings,
            inventory_seeds=dict(getattr(inv, "seeds", {}) or {}))
        try:
            cases, corpus, _ = load_chat_cases(root, bindings=bindings,
                                               canaries=canaries,
                                               have_collector=args.collector)
        except CorpusError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        sha = corpus.sha

    host = args.target
    if not host:
        domains = list(getattr(inv, "allowed_domains", ()) or ())
        host = domains[0] if len(domains) == 1 else ""

    with contextlib.ExitStack() as stack:
        collector = None
        if args.collector:
            try:
                # The inventory goes in because the collector's bind gate reads it:
                # anything but loopback needs the engagement to have signed for a
                # service the operator runs on someone else's behalf.
                collector = stack.enter_context(
                    CallbackCollector(host=args.collector_host, inventory=inv))
            except CollectorRefused as exc:
                print(str(exc), file=sys.stderr)
                return 2
        pf = EngagementPreflight(
            inv=inv, cases=cases, corpus_sha=sha, verified=True,
            host=host, port=args.port, path=getattr(inv, "chat_path", "/chat"),
            tls=bool(getattr(inv, "tls", False)),
            verify=not args.insecure and bool(getattr(inv, "verify_tls", True)),
            trials=args.trials,
            collector=collector, advertised_callback=args.advertise,
            have_tool_visibility=args.tool_visibility, have_judge=args.judge,
            # `--mock` stands in for the whole remote path, not only the send. A
            # rehearsal that still resolves and connects for real is not a rehearsal:
            # it fails at DNS against a hostname the operator invented, and the shape
            # of the readout -- which is the thing being rehearsed -- never appears.
            sender=mock_sender(args.mock) if args.mock else None,
            resolver=(lambda h: "203.0.113.1") if args.mock else None,
            connector=(lambda *a: None) if args.mock else None)
        results = pf.run()

    print(render(results, title=f"preflight -- {inv.project}"))
    failures = [r for r in results if r.status is Status.FAIL]
    if failures:
        # Flushed before stderr: redirected stdout is block-buffered while stderr is
        # not, so without this the summary lands ABOVE the table it summarises.
        sys.stdout.flush()
        print(f"\n{len(failures)} check(s) say do not start: "
              f"{', '.join(r.name for r in failures)}", file=sys.stderr)
    # Non-zero when there is something to fix, so this can gate a pipeline. A
    # preflight that exited 0 because it successfully established that there was no
    # authorization would be the defect this command exists to prevent.
    return 1 if failures else 0


def _classify(args: argparse.Namespace) -> int:
    """Answer the catalogue's classification questions. Reads files, sends nothing."""
    import yaml

    from sunflower_thresh.classify import (
        also_ask,
        as_document,
        classify,
        questions,
        render,
    )
    from sunflower_thresh.obligations import AlmanacError, load_snapshot

    try:
        snapshot = load_snapshot(args.snapshot)
    except AlmanacError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    asked = questions(snapshot) + also_ask(snapshot)
    if args.questions:
        print("# Answer each with true or false, and pass with --answers.\n"
              "# The branching ones are asked in order and the walk stops at the\n"
              "# first class it reaches, so later ones may not be needed.\n")
        for q in asked:
            for line in _wrap(str(q.get("ask") or ""), width=76):
                print(f"# {line}")
            if q.get("note"):
                for line in _wrap("note: " + str(q["note"]), width=76):
                    print(f"#   {line}")
            print(f"{q['id']}: \n")
        return 0

    answers: dict[str, bool] = {}
    if args.answers:
        try:
            raw = yaml.safe_load(args.answers.read_text()) or {}
        except OSError as exc:
            print(f"cannot read {args.answers}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(raw, dict):
            print(f"{args.answers} must be a mapping of question id to true/false",
                  file=sys.stderr)
            return 2
        answers = {str(k): bool(v) for k, v in raw.items() if v is not None}
    elif args.ask:
        known = {q["id"] for q in asked}
        for q in asked:
            if q["id"] not in known:
                continue
            print()
            for line in _wrap(str(q.get("ask") or ""), width=76):
                print(f"  {line}")
            reply = input(f"  {q['id']} [y/N/?] ").strip().lower()
            if reply == "?":
                print(f"  {q.get('note') or 'no further note'}")
                reply = input(f"  {q['id']} [y/N] ").strip().lower()
            answers[q["id"]] = reply.startswith("y")
        print()
    else:
        print("nothing to answer. Pass --answers FILE, --ask, or --questions to "
              "print a template.", file=sys.stderr)
        return 2

    result = classify(snapshot, answers)
    print(render(snapshot, result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "# Generated by `thresh classify`. The catalogue's own questions,\n"
            "# answered for this deployment. Not legal advice.\n"
            + yaml.safe_dump(as_document(result), sort_keys=False,
                             width=100, allow_unicode=True),
            encoding="utf-8")
        print(f"\nwrote {args.out}")
    # Non-zero when the answer is not a usable scope, so this can gate a pipeline:
    # a stop or an unanswered question means the run has no denominator, and
    # proceeding would measure against the wrong set of duties.
    return 0 if result.decided else 1


def _wrap(text: str, width: int = 76) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def _obligations(args: argparse.Namespace) -> int:
    """Resolve a library's techniques against the obligation catalogue."""
    from collections import Counter

    import yaml

    from sunflower_thresh.obligations import (
        AlmanacError,
        coverage,
        in_scope,
        load_snapshot,
        render,
        snapshot_faults,
    )

    try:
        snapshot = load_snapshot(args.snapshot)
    except AlmanacError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.list_instruments:
        obs = snapshot.get("obligations") or {}
        print(f"catalogue  {len(obs)} obligations\n")
        print("instruments")
        for name, n in Counter(o.get("instrument") for o in obs.values()).most_common():
            testable = sum(1 for o in obs.values()
                           if o.get("instrument") == name
                           and o.get("verifiable") != "review")
            print(f"  {n:4}  {name:28} {testable} admit test evidence")
        print("\nsystem classes")
        classes = Counter(c for o in obs.values()
                          for c in (o.get("system_class") or ()))
        for name, n in classes.most_common():
            print(f"  {n:4}  {name}")
        return 0

    taxa: set[str] = set()
    root = args.corpus or corpus_root_from_env()
    if root:
        from sunflower_thresh.corpus import CorpusError, load_corpus

        try:
            corpus = load_corpus(root)
        except CorpusError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        taxa = {c.taxon for c in corpus.cases.values() if c.taxon}

    findings: list[dict] = []
    if args.findings:
        try:
            raw = yaml.safe_load(args.findings.read_text()) or []
        except OSError as exc:
            print(f"cannot read {args.findings}: {exc}", file=sys.stderr)
            return 2
        if isinstance(raw, dict):
            raw = raw.get("findings") or []
        findings = [f for f in raw if isinstance(f, dict)]

    if not in_scope(snapshot, instruments=args.instrument,
                    system_classes=args.system_class):
        sys.stdout.flush()
        print("nothing is in scope with those filters, so there is no denominator "
              "to report over. `--list-instruments` prints what the catalogue "
              "holds; note that `all` is a tag one obligation carries rather than "
              "a way of saying every class.", file=sys.stderr)
        return 2

    cov = coverage(snapshot, findings, attempted=taxa,
                   instruments=args.instrument, system_classes=args.system_class)
    print(render(cov))
    if not root:
        print("\nno corpus loaded, so no technique was resolved. Pass --corpus or "
              f"set ${CORPUS_ENV} to see the three technique states.")
    # Non-zero when the snapshot itself is inconsistent, so this can gate a
    # pipeline. An unknown technique is not a fault: it is a reportable fact, and
    # exiting non-zero for one would make the common case look broken.
    return 1 if snapshot_faults(snapshot) else 0


def _export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from sunflower_thresh import runbundle
    from sunflower_thresh.export import write_run
    from sunflower_thresh.obligations import (
        ENGAGEMENT_FILE,
        AlmanacError,
        attempted_taxa,
        citable_findings,
        coverage,
        load_snapshot,
        read_engagement_scope,
    )

    bundle = Path(args.run_id)
    if not bundle.is_dir():
        bundle = Path(args.runs_dir) / args.run_id
    if not bundle.is_dir():
        print(f"no run bundle at {bundle}. `thresh demo --save runs/` writes one.",
              file=sys.stderr)
        return 2
    manifest, findings = runbundle.load(bundle)
    if not manifest:
        print(f"{bundle} has no manifest, so it is not a finished run. A run "
              f"killed partway through leaves events and no manifest.",
              file=sys.stderr)
        return 2
    # The regulatory scope comes from the engagement's own file rather than from a
    # flag or from the signed inventory. See `obligations.EngagementScope`.
    scope = read_engagement_scope(args.engagement_dir)
    mapping = None
    scope_note = ""
    if scope and not args.no_control_mapping:
        try:
            snapshot = load_snapshot(args.snapshot)
        except AlmanacError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        # Two different sets out of one bundle, and the split is the point.
        # `attempted` is every technique MEASURED, rejected ones included, which is
        # what lets an obligation reach `probed-clean`. The findings are only the
        # ones a report would cite, which is what keeps a rejected measurement out
        # of `tested`.
        mapping = coverage(
            snapshot, citable_findings(findings),
            attempted=attempted_taxa(findings),
            skipped=dict(manifest.get("skipped") or {}),
            instruments=scope.instruments, system_classes=scope.system_classes,
            run_id=str(manifest.get("run_id") or ""))
    elif not scope:
        scope_note = (
            f"no ai_system block in {ENGAGEMENT_FILE}, so no control mapping was "
            f"written. Add `ai_system: {{instruments: [...], classes: [...]}}` to "
            f"say which regimes this client carries; without it every obligation "
            f"in the catalogue renders as not-probed, which is a page of noise.")

    try:
        out = write_run(args.engagement_dir, manifest, findings,
                        control_mapping=mapping)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rows = len((out / "measurements.yaml").read_text().splitlines())
    quotable = manifest.get("quotable") is True
    print(f"wrote {out}")
    print(f"  {len(findings)} finding(s), {rows} lines of measurements")
    if mapping is not None:
        counts = mapping.get("counts") or {}
        taxa = (mapping.get("taxa") or {}).get("counts") or {}
        print(f"  control mapping: "
              f"{mapping.get('testable_with_evidence', 0)} of "
              f"{mapping.get('testable_total', 0)} obligations that admit test "
              f"evidence have it, {counts.get('review-only', 0)} review-only")
        print(f"  techniques: {taxa.get('mapped', 0)} mapped, "
              f"{taxa.get('declined', 0)} declined, {taxa.get('unknown', 0)} "
              f"unknown to the catalogue")
    if scope_note:
        print(f"  {scope_note}")
    if not quotable:
        why = manifest.get("not_quotable_because") or "not marked quotable"
        print(f"  NOT QUOTABLE ({why}), so `thresh consolidate` will skip it. The "
              f"run stays on disk and stays diffable.")
    return 0


def _consolidate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from sunflower_thresh.consolidate import quoted_runs, write
    from sunflower_thresh.export import CONTROL_MAPPING

    try:
        out, notes = write(args.engagement_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    import yaml

    doc = yaml.safe_load(out.read_text()) or {}
    cases = doc.get("cases") or []
    # `quoted_runs` rather than a key on the document. It used to read
    # `runs_quoted` from the document itself, and that key was removed when the
    # consumer's schema turned out to be additionalProperties: false, so the
    # provenance moved into the header comment and this line has printed "from 0
    # quotable run(s)" over a file drawn from several ever since.
    quoted = quoted_runs(doc, args.engagement_dir)
    print(f"wrote {out}")
    print(f"  {len(cases)} case(s) from {len(quoted)} quotable run(s)")
    mapping = Path(args.engagement_dir) / CONTROL_MAPPING
    if mapping.is_file():
        print(f"  and {mapping.name}, from the newest quotable run entire")
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
    from sunflower_thresh.doctor import report

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
