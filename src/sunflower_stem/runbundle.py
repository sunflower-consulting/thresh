"""What a run leaves behind, so something other than a terminal can read it.

`stem demo` printed a table and kept nothing. That is fine for a demo and useless
for an engagement: a report cannot quote a run that was never written down, and a
retest cannot diff against one. So a run now writes a BUNDLE, and the bundle is
the only thing downstream reads.

    runs/<run-id>/
      manifest.json      what this run was: scope, config, budget, quotable
      findings.jsonl     one finding per line, as dicts
      events.jsonl       the append-only event stream (runlog writes this)

Findings are stored as DICTS rather than validated models, which is deliberate and
borrowed from the engine this is replacing: the readers want fields the model does
not promise to keep, and round-tripping through pydantic to get a dict back only
adds a way for an older bundle to fail to load. A bundle is a record, and a record
that stops being readable when the code moves on is not a record.

WHY `quotable` IS IN THE MANIFEST AND NOT A FLAG AT EXPORT TIME.

A run is tuning, or it is evidence. That is a fact about the run -- the scope was
not final, the config was being changed, somebody was testing the test -- and it is
known when the run happens, not later when somebody is assembling a deliverable.
Putting it in the manifest means the five times you ran the same case while getting
it right stay on disk, stay diffable, and cannot reach a client. Deciding it at
export time would mean remembering, and the whole point of this file is to stop
anything depending on remembering.

A missing manifest is an empty result and never an exception. A run killed partway
through leaves events and no manifest, and the honest response to that is a reader
that says so rather than a traceback.
"""
from __future__ import annotations

import json
import platform
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sunflower_stem import __version__

MANIFEST = "manifest.json"
FINDINGS = "findings.jsonl"
EVENTS = "events.jsonl"


class BundleExists(FileExistsError):
    """A bundle is written once. Writing a second time would lose the first."""


def run_id(prefix: str = "run") -> str:
    """A run id that sorts chronologically and carries no client's name.

    Timestamp first because the directory listing is the history, and `runs/` read
    in order is what a retest walks back through.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    # Seconds are not enough to be unique and `write` does not refuse a collision:
    # two runs started in the same second produced one directory and the second
    # overwrote the first's manifest and findings, losing a measurement silently.
    # Caught in review. Milliseconds plus four random hex, so neither a fast loop
    # nor two concurrent workers can land on one id.
    return f"{stamp}-{datetime.now(UTC).microsecond // 1000:03d}" \
           f"_{prefix}_{secrets.token_hex(4)}"


@dataclass
class Manifest:
    """What this run was, as distinct from what it found.

    Every field here exists to answer one question a reader of the findings will
    ask: are these numbers comparable with those other numbers? `config_hash` is
    the short answer and the rest is why.
    """

    run_id: str
    #: Whether this run may be quoted to a client. See the module docstring: a
    #: fact about the run, decided when it happens.
    quotable: bool = False
    #: Reason, when it is not. "scope not signed yet" and "tuning the threshold"
    #: are different and a reader three weeks later cannot reconstruct either.
    not_quotable_because: str = ""
    #: Identity of the thing measured and the instrument that measured it.
    stem_version: str = field(default_factory=lambda: __version__)
    python: str = field(default_factory=platform.python_version)
    scope_hash: str = ""
    config_hash: str = ""
    corpus_sha: str = ""
    plan_hash: str = ""
    roe_hash: str = ""
    #: What the target IS and what posture it was IN. Declared, never inferred:
    #: guessing a model family from a reply is how a report asserts a fact about a
    #: vendor that nobody checked.
    target_family: str = ""
    target_config: str = ""
    #: The attempt budget this run was given, so a rate has a denominator that was
    #: decided in advance rather than read off the result.
    trials: int = 0
    #: Case ids selected, and the ones skipped with why. A case selected and never
    #: reached is not a case that found nothing.
    cases: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def write(run_dir: str | Path, manifest: Manifest, findings) -> Path:
    """Write a bundle. Returns the directory.

    Findings are serialised one per line so a partial write is still readable up to
    the break, which matters when a long run is interrupted.
    """
    d = Path(run_dir)
    if (d / MANIFEST).is_file():
        # Refused rather than overwritten, and this is the guarantee the id's
        # entropy only makes unlikely. A collision used to replace the earlier
        # run's manifest and findings in place, so a measurement vanished with
        # nothing said; widening the id makes that rare and refusing makes it
        # impossible. A bundle is written once, which is this project's own rule
        # about machine files.
        raise BundleExists(
            f"{d} already holds a {MANIFEST}: a run bundle is written once. Two "
            f"runs sharing an id would lose the earlier one's measurement, so "
            f"mint a fresh id with `run_id()` rather than writing here again")
    d.mkdir(parents=True, exist_ok=True)
    # Findings first, and the ORDER is the point. The manifest is what marks a
    # bundle finished and possibly quotable, so writing it first meant a crash
    # during serialisation left a directory that `export` reads as a completed run
    # with no findings -- a quotable manifest advertising a measurement that was
    # never written. Caught in review. Serialise, write the findings, and only then
    # publish the manifest.
    lines = []
    for f in findings:
        raw = f.model_dump(mode="json") if hasattr(f, "model_dump") else dict(f)
        lines.append(json.dumps(raw, sort_keys=True))
    (d / FINDINGS).write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")
    (d / MANIFEST).write_text(json.dumps(manifest.as_dict(), indent=2,
                                         sort_keys=False) + "\n",
                              encoding="utf-8")
    return d


def load(run_dir: str | Path) -> tuple[dict, list[dict]]:
    """(manifest, findings-as-dicts) for one run directory.

    A missing file is an empty result, never an error: a run killed before it
    finished leaves events and no manifest, and the right response is a reader that
    reports that rather than a traceback.
    """
    d = Path(run_dir)
    manifest: dict = {}
    mf = d / MANIFEST
    if mf.is_file():
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            manifest = {}
    findings: list[dict] = []
    ff = d / FINDINGS
    if ff.is_file():
        for line in ff.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                findings.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                # One unreadable line does not discard the rest. A truncated final
                # line is the ordinary shape of an interrupted run.
                continue
    return manifest, findings


def is_quotable(manifest: dict) -> bool:
    """Whether this run may be quoted. Absent means NO.

    Fail-closed on purpose, and it is the one default in this file worth arguing
    about. An older bundle written before this field existed is not a bundle
    somebody declared final; it is one nobody declared anything about, and the
    difference between those matters more than the inconvenience of marking it.
    """
    return manifest.get("quotable") is True
