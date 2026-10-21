"""`thresh demo` is the headline, so it is tested like one.

The claim is not that a demo exists. It is that two commands on a machine with
no API key, no target and no configuration produce a MEASURED result -- one with
a denominator, a control arm and an honest row for the case nothing could score.
Each of those is asserted separately, because any of them could disappear while
the command still printed a confident-looking table.
"""
from __future__ import annotations

import pytest

from sunflower_thresh.demo import DEMO_CAPABILITIES, _ratio, run


@pytest.fixture
def demo(capsys):
    def _run(**kw):
        kw.setdefault("trials", 2)
        assert run(**kw) == 0
        return capsys.readouterr().out
    return _run


def test_the_demo_runs_with_no_key_and_no_configuration(demo):
    out = demo()
    assert "cases considered" in out


def test_every_built_in_case_appears(demo):
    """A case that silently stopped being dispatched would shrink the library
    without shrinking any number a reader sees."""
    from sunflower_thresh.tools.chat import BUILTIN_CHAT_CASES

    out = demo()
    for cid in BUILTIN_CHAT_CASES:
        assert cid in out, f"{cid} is missing from the demo output"


def test_a_case_nothing_could_score_is_reported_as_not_probed(demo):
    """THE honesty row, and the reason the judge being off is visible.

    `chat-fabricated-policy` can only be scored by a judge, and the judge ships
    off. If this ever renders as 0/5 instead, the demo is reporting an absence of
    measurement as a clean result, which is the single failure this whole package
    is arranged to prevent.
    """
    out = demo()
    assert "not probed" in out
    assert "chat-fabricated-policy" in out
    # and it must NOT appear as a measured zero
    for line in out.splitlines():
        if line.startswith("chat-fabricated-policy"):
            assert "0/" not in line, f"reported as a measured zero: {line}"


def test_every_measured_row_carries_a_control_arm(demo):
    """A rate without its control is the thing this tool exists not to print."""
    out = demo()
    measured = [ln for ln in out.splitlines()
                if ln.startswith("chat-") and "not probed" not in ln]
    assert measured, "no measured rows at all"
    for line in measured:
        assert "matched" in line or "baseline" in line, f"no control arm: {line}"


def test_the_summary_counts_add_up_to_the_cases_considered(demo):
    """A total that does not reconcile is how a dropped case hides."""
    import re

    out = demo()
    m = re.search(r"(\d+) cases considered: (\d+) validated, (\d+) rejected, "
                  r"(\d+) unconfirmed, (\d+) untested, "
                  r"(\d+) not probed, (\d+) refused by the gate, (\d+) errored", out)
    assert m, out
    considered, *parts = (int(g) for g in m.groups())
    assert sum(parts) == considered, f"{parts} does not sum to {considered}"


def test_the_undefended_target_has_a_case_its_control_also_fires(demo):
    """The demonstration itself. On the unguarded assistant at least one case
    succeeds on every attempt AND its benign twin succeeds just as often, which
    is the result a tool without a control arm would file as critical."""
    out = demo(safety="none")
    rejected_with_live_control = [
        ln for ln in out.splitlines()
        if ln.startswith("chat-") and ln.rstrip().endswith("rejected")
        and _ratio(ln.split()[2]) > 0 and _ratio(ln.split()[3]) > 0]
    assert rejected_with_live_control, (
        "no case was rejected by a control arm that also fired; the demo no "
        "longer demonstrates the thing it exists to demonstrate")


def test_the_out_of_band_rung_is_demonstrated(demo):
    """The one headline claim the demo could not show.

    Out-of-band proof is the rung that distinguishes this tool from one that
    greps replies, and for as long as `chat-markdown-exfil` never reached the
    lab's embed branch, the strongest-rung column held NONE, CANARY, STRUCTURAL
    and TOOL_CALL and nothing else. The failure this prevents is that regressing
    silently: every other assertion in this file would still pass with the
    CALLBACK rung dead, because the row would still be present, still carry a
    control arm and still reconcile with the summary.
    """
    out = demo()
    row = next((ln for ln in out.splitlines()
                if ln.startswith("chat-markdown-exfil")), None)
    assert row, f"the exfiltration case is not in the table at all\n{out}"
    _, rung, attack, control, *_ = row.split()
    assert rung == "CALLBACK", f"not proved by the target's own traffic: {row}"
    # Separated, not merely fired. Both arms are told to append the image, so
    # both reach the collector; only the attack's fetch carries the canary, and
    # a control that fired as often would make this row rejected.
    assert _ratio(attack) == 1.0, row
    assert _ratio(control) == 0.0, row
    # Not `validated`, which also needs the p-value under the threshold and so
    # depends on the trial budget -- at two trials the strongest honest verdict
    # is `reproduced`. What must never come back is the rejection.
    assert not row.rstrip().endswith("rejected"), row


def test_turning_the_defences_on_changes_what_is_proved(demo):
    """The defence arm. Not that the numbers drop to zero everywhere -- a
    guardrail that filters text does nothing about a tool authorization flaw --
    but that the two runs differ at all."""
    assert demo(safety="none") != demo(safety="strong")


def test_the_callback_capability_is_switched_on_explicitly():
    """Capability checks are fail-closed, so an unlisted switch reads as off and
    the exfiltration case would be reported unreachable rather than measured.
    The demo turns it on in the open rather than relying on a default."""
    assert DEMO_CAPABILITIES["out_of_band_callback"] is True


@pytest.mark.parametrize("cell,expected", [
    ("5/5", 1.0), ("0/5", 0.0), ("2/4", 0.5), ("0/0", 0.0), ("", 0.0), ("x/y", 0.0),
])
def test_the_arm_ratio_helper_never_raises(cell, expected):
    """It parses a rendered table cell, so it has to tolerate anything the table
    can hold rather than crash the summary that follows the numbers."""
    assert _ratio(cell) == expected


def test_the_readme_quotes_the_not_probed_reason_the_demo_actually_prints():
    """The README prints a sample run and then tells the reader to look at the last
    row: "Nothing could score it, so it says so and carries the reason." That
    sentence is only worth anything if the reason in the README is the reason the
    tool prints.

    It has already drifted once. The row read "no scorable oracle: judge needs
    judge", because the oracle's name falls back to the rung kind and the
    capability it needs is called the same thing -- so the one row the README
    points at was the one carrying the clumsiest string in the output."""
    import io
    import pathlib
    from contextlib import redirect_stdout

    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
    quoted = next(ln.strip() for ln in readme.splitlines()
                  if "no scorable oracle" in ln)

    buf = io.StringIO()
    with redirect_stdout(buf):
        run(trials=1)
    assert quoted in buf.getvalue(), (
        f"the README quotes {quoted!r}, which the demo does not print")


def test_the_table_names_the_case_and_not_the_run(capsys):
    """The row label is the case id, taken from the finding rather than parsed out
    of its finding id.

    The old line stripped a literal `"demo-"` prefix, which was right only while
    the run id was the literal string `demo`. The moment a run got a real id, every
    row read `2026-10-18T12-00-32_chat-crescendo`, and three tests that assert on
    the table caught it only indirectly. This asserts the thing itself."""
    run(trials=1)
    out = capsys.readouterr().out
    # Rows sit between the rule and the blank line before the summary. Skip the
    # rule itself and the indented continuation lines that carry a skip reason.
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if set(ln.strip()) == {"-"}) + 1
    labels = []
    for ln in lines[start:]:
        if not ln.strip():
            break
        if ln.startswith(" "):
            continue
        labels.append(ln.split()[0])
    assert len(labels) == 6, f"expected six rows, got {labels}"
    for label in labels:
        assert label.startswith("chat-"), f"row labelled with a run id: {label}"


def test_the_library_example_still_works(tmp_path):
    """The example is documentation that answers "how do I drive this from a script",
    and documentation that nobody runs becomes advice that no longer works.

    It is the only pattern in the repository for a client-specific engagement:
    `demo.py` builds its inventory in code and is wired to the bundled lab, which
    makes it a smoke test rather than something to copy. This asserts the example's
    six-object assembly -- inventory, evidence store, gate, context, selection,
    bundle -- actually produces a run.
    """
    import runpy
    import sys
    from pathlib import Path

    from sunflower_thresh import runbundle

    examples = Path(__file__).resolve().parents[1] / "examples"
    assert (examples / "scope.yaml").is_file(), "the example's scope travels with it"

    out = tmp_path / "runs"
    argv = sys.argv[:]
    sys.argv = ["engagement.py", "--out", str(out), "--trials", "2"]
    try:
        runpy.run_path(str(examples / "engagement.py"), run_name="__main__")
    except SystemExit as exit_:
        assert exit_.code == 0, f"the example exited {exit_.code}"
    finally:
        sys.argv = argv

    bundles = sorted(out.iterdir())
    assert len(bundles) == 1, bundles
    manifest, findings = runbundle.load(bundles[0])
    assert manifest["run_id"] == bundles[0].name
    assert manifest["quotable"] is False, "the bundled lab is not a client target"
    assert findings, "the example has to actually measure something"
    assert {f.get("state") for f in findings} <= {
        "validated", "reproduced", "rejected", "confirmed", "observed",
        "hypothesis", "under_test", "human_review", "duplicate"}
