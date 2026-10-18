"""Nothing in `sunflower_thresh` may import anything outside its allowlist.

THE FAILURE THIS EXISTS TO STOP.

This package is an extraction from a much larger private codebase whose modules
import a knowledge graph, an agent runtime, a Playwright browser surface and a
regulatory layer. Every one of those is reachable in one hop from a module that
belongs here, so the natural end state of porting by hand is a package that
quietly depends on half the thing it was extracted from and cannot be installed
from PyPI by anyone.

An AST walk is used rather than importing the modules, because an import-time
check only catches what the import graph actually executes: a dependency behind
`if TYPE_CHECKING` or inside a function body would pass while still being a
dependency a reader has to satisfy.
"""
from __future__ import annotations

import ast
import pathlib
import sys

PACKAGE = "sunflower_thresh"
SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / PACKAGE

#: Third-party distributions this package is allowed to require. Keep this
#: short and make additions argue for themselves in the diff: every entry is a
#: thing a user must install before `thresh demo` can run, and the two-command
#: quickstart is the single most valuable property this package has.
THIRD_PARTY = frozenset({"pydantic", "yaml"})

#: Modules from the private source tree that must NEVER appear. Listed by name
#: so the failure message can say what went wrong rather than only that
#: something did. These are the heavy neighbours, not an exhaustive list.
FORBIDDEN_ROOTS = frozenset({
    "harness",      # the private package this was extracted from
    "playwright",   # the browser surface is deliberately out of scope
})


def _modules() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def _imported_roots(tree: ast.AST) -> set[str]:
    """Top-level module name of every import, however it is spelled."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        # level > 0 is a relative import, which is by definition in-package
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_source_tree_is_not_empty():
    """Guards the guard. A glob that matches nothing passes every assertion
    below it, so a renamed package directory would silently disable the whole
    firewall rather than failing."""
    mods = _modules()
    assert mods, f"no modules found under {SRC}; the firewall is not checking anything"


def test_every_module_imports_only_stdlib_pydantic_yaml_or_itself():
    allowed = set(sys.stdlib_module_names) | THIRD_PARTY | {PACKAGE}
    offences: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for root in sorted(_imported_roots(tree) - allowed):
            offences.append(f"{path.relative_to(SRC)} imports {root!r}")
    assert not offences, (
        "modules outside the allowlist are imported:\n  "
        + "\n  ".join(offences)
        + "\n\nEither vendor what is needed into this package, or add the "
          "distribution to THIRD_PARTY here AND to pyproject dependencies. "
          "Do not add it in only one of the two places."
    )


def test_the_private_source_package_is_never_imported():
    """Separate from the allowlist test so the failure reads unambiguously. An
    accidental `from harness.x import y` left in a ported file is the single
    most likely porting mistake, and it must not be reported as a generic
    allowlist miss."""
    offences = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for root in sorted(_imported_roots(tree) & FORBIDDEN_ROOTS):
            offences.append(f"{path.relative_to(SRC)} imports {root!r}")
    assert not offences, "\n  ".join(["ported code still reaches its old home:"] + offences)


def test_third_party_imports_are_declared_in_pyproject():
    """The firewall and the package metadata must agree. An allowlisted import
    that is not a declared dependency installs fine for the person who wrote it
    and fails for everybody else, which is the worst shape of bug to ship on a
    tool whose headline is a two-command install."""
    import tomllib

    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    # map the import name to the distribution name where they differ
    dist_of = {"yaml": "pyyaml", "pydantic": "pydantic"}
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in declared}
    for mod in sorted(THIRD_PARTY):
        assert dist_of[mod] in names, (
            f"{mod!r} is allowlisted in the firewall but {dist_of[mod]!r} is not "
            f"a declared dependency in pyproject.toml")
