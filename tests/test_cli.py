"""The command surface, and the promise the module docstring makes about it.

`cli.py` says a subcommand is registered only once it works, because a help
listing that advertises something which then prints "not implemented" costs a
reader more trust than the missing feature costs them time. That is a claim worth
a test, and so is the exit code: a command whose failure exits 0 is one that
cannot be used in a script.
"""
from __future__ import annotations

import argparse
import pathlib

import pytest

from sunflower_thresh.cli import _build_parser, main

REPO = pathlib.Path(__file__).resolve().parents[1]


def _help(argv: list[str]) -> None:
    """argparse's --help action exits rather than returning, so every caller here
    has to unwrap that. A non-zero code means the parser rejected the arguments.

    REQUIRED, not merely handled. The assertion used to sit inside
    `except SystemExit`, so a `--help` that RETURNED skipped it entirely and
    `_help` reported success -- and a `--help` that returns is a `--help` that ran
    the command instead of describing it. Measured: registering `doctor` with
    `add_help=False` plus its own `--help` flag made `main(["doctor", "--help"])`
    print the whole doctor readout and return 0, and the suite stayed at 560
    passed.

    The `-> str` and `return ""` went with it: no caller ever read the value, and
    they are what let the silent path look like a result.
    """
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code in (0, None), f"{argv} exited {exc.value.code}"


def test_every_advertised_command_runs(capsys):
    """Each name in the help listing has to do something. Discovered from the
    listing rather than hardcoded, so a command added without a test is caught
    here instead of by a reader."""
    _help(["--help"])
    listing = capsys.readouterr().out
    body = listing.split("positional arguments:", 1)[1]
    names = [ln.split()[0] for ln in body.splitlines()
             if ln.startswith("    ") and ln.strip() and not ln.startswith("     ")]
    assert set(names) >= {"doctor", "judge"}, f"unexpected listing: {names}"
    for name in names:
        # `demo` and `lab` bind sockets and serve; their behaviour is covered in
        # their own files. What is asserted here is that the name resolves to a
        # real parser with its own help, which is what "registered only once it
        # works" means -- a bare `set_defaults` with no function would exit 2 here.
        _help([name, "--help"])
        assert capsys.readouterr().out.strip(), f"{name} --help printed nothing"


def test_no_subcommand_prints_help_rather_than_an_error(capsys):
    """`thresh` with no arguments is a reader orienting themselves, not a mistake."""
    assert main([]) == 0
    assert "usage: thresh" in capsys.readouterr().out


def test_the_judge_readout_works_with_no_endpoint_and_no_network(capsys):
    """The normal state of a fresh install, and it must not read as a fault."""
    assert main(["judge"]) == 0
    out = capsys.readouterr().out
    assert "12 loaded" in out
    assert "WITHHELD" in out
    assert "not configured" in out


def test_probing_without_configuration_exits_nonzero(capsys):
    """Asking for a probe when there is nothing to probe is a usage error, not a
    clean run that happened to do nothing."""
    assert main(["judge", "--probe"]) == 2
    assert "nothing to probe" in capsys.readouterr().err


def test_an_unreachable_judge_endpoint_exits_nonzero_and_says_so(capsys):
    """A script that points thresh at a dead endpoint has to be able to tell. Port 1
    is reserved and refuses immediately, so this costs no wall-clock."""
    code = main(["judge", "--probe", "--endpoint", "http://127.0.0.1:1/v1",
                 "--model", "nothing-serves-this"])
    captured = capsys.readouterr()
    assert code == 1
    assert "UNREACHABLE" in captured.err


def test_the_failure_line_does_not_precede_the_context_it_refers_to(capsys):
    """Redirected stdout is block-buffered and stderr is not, so an unflushed
    readout puts the error above the listing it is about. Harmless on a terminal
    and confusing in a log, which is where this output actually gets read."""
    main(["judge", "--probe", "--endpoint", "http://127.0.0.1:1/v1", "--model", "m"])
    captured = capsys.readouterr()
    assert "rubric library:" in captured.out
    assert "UNREACHABLE" in captured.err


def test_the_readme_documents_exactly_the_commands_that_exist():
    """The README publishes a command list and then says `thresh doctor` is
    authoritative if the two disagree. That is the right fallback and a poor
    substitute for them agreeing: a reader meets the README first, and a command
    missing from it is a capability nobody finds.

    This repository has already shipped one readout that drifted from the code it
    described, so the list is compared rather than trusted."""
    readme = (REPO / "README.md").read_text()
    block = readme.split("## Commands", 1)[1].split("```", 2)[1]
    documented = {ln.split()[1] for ln in block.splitlines() if ln.startswith("thresh ")}

    sub = next(a for a in _build_parser()._actions
               if isinstance(a, argparse._SubParsersAction))
    advertised = set(sub.choices)

    assert documented == advertised, (
        f"README and --help disagree: only in README {documented - advertised}, "
        f"only in --help {advertised - documented}")


def test_the_judge_readout_names_every_rubric_and_its_state(capsys):
    """One word, "uncalibrated", covers three situations that send a reader to
    three different places: measure it, measure more of it, or fix the rubric. So
    the state is printed per rubric rather than as a count."""
    from sunflower_thresh.judge import load_rubrics

    assert main(["judge"]) == 0
    out = capsys.readouterr().out
    for name in load_rubrics():
        assert name in out, f"{name} missing from the readout"
    assert "withheld:" in out


def test_the_judge_readout_shows_polarity_because_it_changes_the_verdict(capsys):
    """Polarity decides which arm a calibration run expects a rubric to fire on.
    An inverse rubric read as positive inverts its whole read-out, so which one a
    rubric is belongs in the output and not only in the file."""
    assert main(["judge"]) == 0
    out = capsys.readouterr().out
    assert "inverse" in out and "positive" in out
    row = next(ln for ln in out.splitlines() if ln.strip().startswith("refusal"))
    assert "inverse" in row
