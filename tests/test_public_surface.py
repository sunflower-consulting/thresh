"""The names a client script imports, and the promise about them.

`__init__.py` used to export `__version__` and nothing else, so every useful import
was a deep submodule path and there was no statement anywhere about what was stable.
That mattered little while the command was the surface and matters now that the
package is meant to be driven from a script.

Three properties are worth a test rather than a convention: the surface is what it
says it is, resolving a name is LAZY so `thresh --help` does not pay for pydantic,
and a name that is deliberately not re-exported says so when you ask for it.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import sunflower_thresh


def test_the_declared_surface_and_the_table_behind_it_agree():
    """`__all__` is written out so a linter and a type checker can read it
    statically, which means it can drift from the table `__getattr__` uses. It
    cannot drift silently."""
    declared = set(sunflower_thresh.__all__) - {"__version__"}
    assert declared == set(sunflower_thresh._EXPORTS)
    assert sunflower_thresh.__all__ == sorted(sunflower_thresh.__all__)
    assert set(dir(sunflower_thresh)) == set(sunflower_thresh.__all__)


@pytest.mark.parametrize("name", sorted(sunflower_thresh._EXPORTS))
def test_every_re_exported_name_resolves_to_the_real_object(name):
    """A table entry naming a module that does not define it would be a surface
    that looks present and fails on first use."""
    value = getattr(sunflower_thresh, name)
    assert value is not None
    where = sunflower_thresh._EXPORTS[name]
    module = __import__(f"sunflower_thresh.{where}", fromlist=[name])
    assert getattr(module, name) is value, "the re-export must be the same object"


def test_a_bare_import_pays_for_nothing():
    """`thresh --help` and `thresh doctor` must not import pydantic and the bundled
    lab to print a line -- `cli.py` keeps only four top-level imports for exactly
    that reason, and an eager re-export here would have undone it.

    Measured in a subprocess, because this test process has long since imported
    everything and would report a false pass.
    """
    code = (
        "import sys, sunflower_thresh;"
        "loaded = [m for m in sys.modules"
        " if m.startswith(('pydantic', 'yaml')) or m.startswith('sunflower_thresh.')];"
        "print(loaded)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", f"a bare import pulled in {out}"


def test_resolving_a_name_is_what_costs():
    """The other half: lazy must still WORK, not merely be cheap."""
    code = (
        "import sys, sunflower_thresh;"
        "assert 'pydantic' not in sys.modules;"
        "sunflower_thresh.PolicyEngine;"
        "assert 'pydantic' in sys.modules;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "ok"


def test_a_name_that_is_not_re_exported_says_where_to_find_it():
    """Four modules define a `render` and two define a `coverage`, so flattening
    them would make the shortest name the most ambiguous. Asking for one has to say
    that rather than just failing."""
    missing = "coverage"
    with pytest.raises(AttributeError) as exc:
        getattr(sunflower_thresh, missing)
    message = str(exc.value)
    assert "no attribute 'coverage'" in message
    assert "submodule, which is the API for it" in message
    assert "render" not in sunflower_thresh._EXPORTS
    assert "coverage" not in sunflower_thresh._EXPORTS


def test_the_stability_statement_exists_and_is_honest():
    """At 0.1.0.dev0 there is no stability promise, and saying so beats leaving it
    to be discovered. What IS pinned are the machine-file schemas other repositories
    assert against, and the docstring has to name that distinction."""
    doc = sunflower_thresh.__doc__ or ""
    assert "STABILITY" in doc
    assert "There is none yet" in doc
    assert "measurements.yaml" in doc and "control-mapping.yaml" in doc
    assert sunflower_thresh.__version__.endswith(".dev0"), (
        "when this stops being a dev version the statement above needs rewriting")


def test_the_policy_package_re_exports_nothing():
    """A facade nobody imports still costs every importer of the package.

    `policy/__init__.py` re-exported seven names and nothing in the repository
    imported any of them -- all twenty call sites name a submodule directly. The
    cost was not hypothetical. `tools/base.py` and `tools/conversation.py` need
    `policy.decision`, 125 lines of request shapes, and importing it ran the
    facade first: measured at 9 `sunflower_thresh` modules including the engine,
    the kill switch and the audit log, against 5 with the facade empty.

    `thresh --help` advertising itself as cheap is a claim this guards, and
    `schemas/__init__.py` already made the same decision for the same reason.
    """
    import sys

    from sunflower_thresh import policy

    assert not getattr(policy, "__all__", None), (
        f"policy/__init__.py re-exports {policy.__all__}; import the submodule "
        f"you need instead, as schemas/__init__.py does")

    for name in ("Decision", "PolicyEngine", "KillSwitch", "RateLimiter",
                 "ToolRequest", "TargetRef", "DenyReason"):
        assert not hasattr(policy, name), (
            f"sunflower_thresh.policy.{name} is re-exported again")

    # The consequence, not just the shape: the request module must not drag the
    # engine in behind it. Asserted on a subprocess-clean import graph, because
    # by the time this test runs the suite has imported everything already.
    import subprocess
    code = (
        "import sys;"
        "import sunflower_thresh.policy.decision;"
        "print(int('sunflower_thresh.policy.engine' in sys.modules),"
        "      int('sunflower_thresh.policy.killswitch' in sys.modules),"
        "      int('sunflower_thresh.audit.log' in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["0", "0", "0"], (
        f"importing policy.decision loaded engine/killswitch/audit = {out}; "
        f"something re-introduced an eager import in the policy package")


def test_the_import_examples_in_prose_are_valid_python_and_name_the_real_package():
    """Prose that tells a reader what to type has to be typeable.

    Two defects this guards, both found by review and both in the first thing a
    reader meets:

    `__init__.py`'s "the submodule is the API" block carried
    `render as render_diff as render as render_diff` -- a doubled alias clause
    that raises `SyntaxError: cannot use name as import target`. It sat in the
    docstring that exists specifically to say how to import from this package.

    README.md said "so `import thresh` stays cheap". There is no module named
    `thresh`; `import thresh` raises ModuleNotFoundError. The import package is
    `sunflower_thresh` and only the CLI command is `thresh`, which is exactly the
    pair a reader is most likely to confuse, and the sentence confirmed the wrong
    half. The claim around it is true and stays: the top-level import does not
    load pydantic.

    Scanned rather than pinned, so a new example cannot arrive broken.
    """
    import ast
    from pathlib import Path

    import sunflower_thresh

    root = Path(__file__).resolve().parents[1]
    prose = {
        "sunflower_thresh.__doc__": sunflower_thresh.__doc__ or "",
        "README.md": (root / "README.md").read_text(),
        "docs/method.md": (root / "docs" / "method.md").read_text(),
    }

    for where, text in prose.items():
        for raw in text.split("\n"):
            line = raw.strip()
            if not line.startswith(("from sunflower_thresh", "import sunflower_thresh")):
                continue
            if line.endswith(("(", ",", "\\")):
                continue  # a real multi-line import in code, not an example
            try:
                ast.parse(line)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{where} shows an import a reader cannot type: {line!r} "
                    f"({exc.msg})") from None

        assert "import thresh" not in text.replace("import sunflower_thresh", ""), (
            f"{where} says `import thresh`; the import package is "
            f"`sunflower_thresh` and only the CLI command is `thresh`")

    # The claim the README makes around that sentence, so correcting the name
    # cannot quietly leave a false promise beside it.
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sunflower_thresh, sys; print(int('pydantic' in sys.modules))"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == "0", "the top-level import now loads pydantic; the README says it does not"


def test_the_palette_holds_only_colours_the_module_can_emit():
    """`_CODES` is the palette, and an entry no helper requests is a dead string.

    It carried `dim`, `italic`, `blue` and `magenta`, none of which appears
    anywhere else in the package. `style` is not re-exported and has no caller
    outside `termstyle.py`, so those four were unreachable -- a palette that read
    as an offer the module does not make, beside a docstring laying out strict
    taste rules for the colours it does.

    Scanned from the helpers rather than pinned to a list, so adding a helper
    that needs a new colour is not a test failure, and adding a colour no helper
    needs is.
    """
    import re
    from pathlib import Path

    from sunflower_thresh import termstyle

    src = Path(termstyle.__file__).read_text()
    # Every quoted name inside every `style(...)` call. `_CODES` is read in exactly
    # one place -- the comprehension inside `style` -- so a name is reachable only
    # by being passed to that function.
    requested = set()
    for call in re.findall(r"style\(([^)]*)\)", src):
        requested |= set(re.findall(r"""["']([a-z_]+)["']""", call))

    declared = set(termstyle._CODES)
    assert requested, "found no style() call to read the palette from"
    assert declared - requested == set(), (
        f"{sorted(declared - requested)} are in the palette and no helper requests "
        f"them; `style` has no caller outside termstyle.py, so they cannot be "
        f"reached")
    assert requested - declared == set(), (
        f"a helper requests {sorted(requested - declared)}, which the palette does "
        f"not define, so it would raise KeyError at the first call")


#: Paths this package's comments mention ON PURPOSE while saying they are not
#: here. They belong to the private engine `sunflower_thresh` was extracted from,
#: and the sentences citing them exist to tell a reader that the guarantee lives
#: on the other side. Adding to this set is a deliberate act; the citing line must
#: say the file is absent, or a reader will go looking for it.
CITED_BUT_DELIBERATELY_ABSENT = frozenset({
    "config/labs/defenses.yaml",
    "docs/corpus-feedback.md",
    "config/projects/chatbot-demo.example.yaml",
    "tests/test_lab_defenses.py",
    "tests/test_lab_characterisation.py",
})


def test_every_file_a_comment_points_at_is_either_here_or_declared_absent():
    """A comment that vouches for a guarantee by citing a test is worth exactly as
    much as the test.

    This class of defect recurred in five files in one review pass, and the
    expensive ones were not the dangling links -- they were the claims:

      * `sweep.py` said "tests/test_message_budget.py asserts the contents of this
        tuple". No such file, so `TRANSIENT_DENIALS` -- which decides whether a
        refused case is re-sent six times -- was unpinned while a comment said it
        was not. That test now exists, under the name it actually has.
      * `lab/server.py` said `tests/test_lab_characterisation.py` "proves it did
        not" about branch ordering. Nothing proved it.
      * `schemas/finding.py` cited `docs/corpus-feedback.md` for "the corpus
        carries no matched controls". The doc is absent and the claim was false:
        190 of 244 cases carry one. That false sentence was the stated reason the
        baseline arm exists.

    So the rule is not "no dead links". It is that a reader must be able to tell a
    citation they can follow from one describing another repository, and the only
    way to keep that true is to check it.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "src" / "sunflower_thresh"
    pattern = re.compile(
        r"\b((?:tests|config|docs|examples|labs|data|schema)"
        r"/[A-Za-z0-9_./-]+\.(?:py|ya?ml|md|json|typ))")

    dangling: dict[str, list[str]] = {}
    for path in sorted(src.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().split("\n"), 1):
            for match in pattern.finditer(line):
                cited = match.group(1)
                if (root / cited).exists() or cited in CITED_BUT_DELIBERATELY_ABSENT:
                    continue
                dangling.setdefault(cited, []).append(
                    f"{path.relative_to(root)}:{lineno}")

    assert not dangling, (
        "these comments point at files that are not in this repository. Either the "
        "path is wrong, or the file lives in the private engine -- in which case say "
        "so on the citing line and add it to CITED_BUT_DELIBERATELY_ABSENT, so the "
        f"next reader is not sent looking: {dangling}")

    # The allowlist must not outlive its entries: a path that comes BACK should be
    # checked normally again, and one nothing cites any more is just stale.
    cited_anywhere = {m.group(1) for path in src.rglob("*.py")
                      for m in pattern.finditer(path.read_text())}
    for absent in sorted(CITED_BUT_DELIBERATELY_ABSENT):
        assert absent in cited_anywhere, (
            f"{absent} is allowlisted as deliberately absent and nothing cites it; "
            f"drop it from CITED_BUT_DELIBERATELY_ABSENT")
        assert not (root / absent).exists(), (
            f"{absent} now EXISTS, so it should be checked like any other citation; "
            f"drop it from CITED_BUT_DELIBERATELY_ABSENT")
