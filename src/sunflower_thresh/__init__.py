"""thresh: measure whether an attack on an LLM application actually worked.

Three things distinguish a measurement here from a transcript somebody read:

  * every case has a benign twin, so a hit is compared against what the target
    does when nobody is attacking it. A rate without a control arm is an
    anecdote with a denominator.
  * proof is ranked, and the strongest available rung is what gets reported. A
    model emitting a plausible-looking string is the weakest thing this tool
    will call evidence, and it says so.
  * authorization is decided outside the model. The model proposes and the gate
    decides, so a case cannot talk its way into a request nobody signed.

The package is `sunflower_thresh` and the command is `thresh`; the names differ
because the `thresh` distribution on PyPI is taken by a data-collection library
(github.com/davidheineman/thresh.tools) that has every right to it. Checked rather
than assumed: that project's wheel ships no entry point, so the COMMAND name is
free and only the distribution name needed the prefix.

USING IT FROM A SCRIPT

The commands are a convenience over a library, and a real engagement is a script.
`examples/engagement.py` is that script, runnable as it stands, and a test executes
it so it cannot rot. The names below are re-exported for it:

    from sunflower_thresh import load_inventory as load_inventory, PolicyEngine as PolicyEngine, AgentContext as AgentContext

There is deliberately NO facade -- no single object you construct and drive, the way
PyRIT gives you an orchestrator. Six things are assembled in order (the signed
inventory, the evidence store, the policy gate, the context every tool call passes
through, the case selection, the run bundle) and that assembly IS the API. It is
explicit because each step is a decision somebody has to be able to point at
afterwards: which scope, which evidence policy, which cases, which budget.

Re-exported LAZILY, through a module `__getattr__`. `import sunflower_thresh` stays
cheap, which matters because `thresh --help` and `thresh doctor` must not pay for
pydantic and the bundled lab to print a line -- `cli.py` keeps only four top-level
imports for the same reason, and an eager re-export here would have undone that.

WHAT IS NOT RE-EXPORTED, and why it is not an oversight: four modules define a
`render`, two define a `coverage`, and flattening those would make the shortest name
the most ambiguous. They stay where they are, and the submodule is the API:

    from sunflower_thresh.obligations import coverage as coverage, render as render
    from sunflower_thresh.diff import diff_runs as diff_runs, render as render_diff

STABILITY

There is none yet, and this says so rather than leaving it to be discovered. At
0.1.0.dev0 any name here can move. What will NOT move quietly, because other
repositories assert them, are the two machine-file schemas this writes
(`measurements.yaml` and `control-mapping.yaml`) and the vendored catalogue's
digest; those are versioned by the consuming repository's schemas and a change
there is a change in both places or it is a bug.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0.dev0"

#: name -> the submodule that defines it. `__all__` below is written out rather than
#: derived from this, because a linter and a type checker both have to read it
#: statically -- and a test asserts the two agree, so they cannot drift.
_EXPORTS: dict[str, str] = {
    # the signed scope, and the endpoint facts it carries
    "load_inventory": "preflight",
    "transport_for": "preflight",
    "EngagementPreflight": "preflight",
    "TargetInventory": "schemas.target",
    "AuthorizationRecord": "schemas.target",
    # the gate, and the one path a tool call reaches a target through
    "PolicyEngine": "policy.engine",
    "AgentContext": "context",
    # evidence
    "EvidenceStore": "evidence.store",
    "make_redactor": "redact",
    "CallbackCollector": "collector",
    # what can be tested, and running it
    "applicable_chat_cases": "sweep",
    "sweep_builtins": "sweep",
    "validate_chat_case": "validation_chat",
    # the case library
    "load_corpus": "corpus",
    "load_chat_cases": "corpus",
    "resolve_bindings": "corpus",
    # what a run leaves behind, and comparing two of them
    "Manifest": "runbundle",
    "run_id": "runbundle",
    "diff_runs": "diff",
    # findings and the evidence ladder
    "Finding": "schemas.finding",
    "FindingState": "schemas.finding",
    "EvidenceStrength": "schemas.finding",
}

__all__ = [
    "AgentContext",
    "AuthorizationRecord",
    "CallbackCollector",
    "EngagementPreflight",
    "EvidenceStore",
    "EvidenceStrength",
    "Finding",
    "FindingState",
    "Manifest",
    "PolicyEngine",
    "TargetInventory",
    "__version__",
    "applicable_chat_cases",
    "diff_runs",
    "load_chat_cases",
    "load_corpus",
    "load_inventory",
    "make_redactor",
    "resolve_bindings",
    "run_id",
    "sweep_builtins",
    "transport_for",
    "validate_chat_case",
]


def __getattr__(name: str):
    """Resolve a re-exported name on first use. See the module docstring."""
    where = _EXPORTS.get(name)
    if where is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. The re-exported names "
            f"are {', '.join(sorted(_EXPORTS))}; everything else lives on a "
            f"submodule, which is the API for it")
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{where}"), name)
    globals()[name] = value          # resolved once, not on every attribute read
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

if TYPE_CHECKING:  # pragma: no cover - for editors and type checkers only
    # The redundant-alias form (`X as X`) is how a module says "this is a
    # re-export" to a type checker, and it is what stops these reading as unused
    # imports. They exist only so an editor can resolve the lazy names above.
    from sunflower_thresh.collector import CallbackCollector as CallbackCollector
    from sunflower_thresh.context import AgentContext as AgentContext
    from sunflower_thresh.corpus import load_chat_cases as load_chat_cases
    from sunflower_thresh.corpus import load_corpus as load_corpus
    from sunflower_thresh.corpus import resolve_bindings as resolve_bindings
    from sunflower_thresh.diff import diff_runs as diff_runs
    from sunflower_thresh.evidence.store import EvidenceStore as EvidenceStore
    from sunflower_thresh.policy.engine import PolicyEngine as PolicyEngine
    from sunflower_thresh.preflight import EngagementPreflight as EngagementPreflight
    from sunflower_thresh.preflight import load_inventory as load_inventory
    from sunflower_thresh.preflight import transport_for as transport_for
    from sunflower_thresh.redact import make_redactor as make_redactor
    from sunflower_thresh.runbundle import Manifest as Manifest
    from sunflower_thresh.runbundle import run_id as run_id
    from sunflower_thresh.schemas.finding import EvidenceStrength as EvidenceStrength
    from sunflower_thresh.schemas.finding import Finding as Finding
    from sunflower_thresh.schemas.finding import FindingState as FindingState
    from sunflower_thresh.schemas.target import (
        AuthorizationRecord as AuthorizationRecord,
    )
    from sunflower_thresh.schemas.target import TargetInventory as TargetInventory
    from sunflower_thresh.sweep import applicable_chat_cases as applicable_chat_cases
    from sunflower_thresh.sweep import sweep_builtins as sweep_builtins
    from sunflower_thresh.validation_chat import (
        validate_chat_case as validate_chat_case,
    )
