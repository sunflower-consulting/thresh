"""Where the private checkouts are, for the tests that need one.

Every test that reads the corpus or the report renderer skips without it, which is
deliberate: this repository is public and both of those are private, so the contract
tests are the ones a stranger cannot run and everything else must be.

What was wrong was HOW they were found. Five constants hardcoded one developer's home
directory, so those tests ran for exactly one person and skipped silently for anybody
else who had the checkouts somewhere else -- and a silent skip of a contract test is
indistinguishable from a passing one. `THRESH_CORPUS` is already the documented way to
point this package at a corpus, so it is read here too, and the old absolute paths
remain only as the last fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Sibling checkouts, when this repository sits beside them. The common layout and
#: the one that needs no configuration.
_SIBLINGS = Path(__file__).resolve().parents[2]


def _resolve(env: str, name: str, *, marker: str) -> Path:
    """The first of: the environment variable, a sibling checkout, a legacy path.

    `marker` is a file or directory that must exist inside a real checkout, so a
    variable pointing at an empty or wrong directory resolves to something the
    skipif can reject rather than to a path that merely exists.
    """
    for candidate in (os.environ.get(env), _SIBLINGS / name,
                      Path.home() / "Projects" / "GitHub" / name):
        if not candidate:
            continue
        path = Path(candidate)
        if (path / marker).exists():
            return path
    return _SIBLINGS / name


#: The private case library. `THRESH_CORPUS` is the same variable `thresh corpus` reads.
CORPUS = _resolve("THRESH_CORPUS", "seed-vault", marker="cases")

#: The report renderer, for the two cross-repository contract tests.
DOC_ENGINE = _resolve("THRESH_DOC_ENGINE", "doc-engine", marker="schema")

#: The renderer's vendored copy of the obligation catalogue, which this package's
#: copy must match byte for byte.
#:
#: Both spellings, because the catalogue was renamed from `trellis` to `almanac`
#: HERE and the renderer is a separate repository that has not been renamed yet. A
#: rename in this repo must not relocate another repo's files -- doing so turned the
#: byte-for-byte contract test into a silent skip, which is the failure the whole of
#: this module exists to stop.
def _renderer_snapshot() -> Path:
    for flavour in ("almanac", "trellis"):
        candidate = DOC_ENGINE / "data" / flavour / "snapshot.yaml"
        if candidate.is_file():
            return candidate
    return DOC_ENGINE / "data" / "almanac" / "snapshot.yaml"


DOC_ENGINE_SNAPSHOT = _renderer_snapshot()

HAVE_CORPUS = (CORPUS / "cases").is_dir()
#: A file every renderer-contract test actually reads, not just the directory that
#: would contain it. A failed `actions/checkout` of another repository leaves the
#: target directory BEHIND AND EMPTY, so a directory-shaped predicate said "present"
#: and the tests then died on FileNotFoundError instead of skipping. Measured in CI.
DOC_ENGINE_MARKER = DOC_ENGINE / "schema" / "control-mapping.schema.json"

HAVE_DOC_ENGINE = DOC_ENGINE_MARKER.is_file()
