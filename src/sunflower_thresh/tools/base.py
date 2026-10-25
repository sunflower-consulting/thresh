"""Real tool adapters -- deterministic runners that live *below* the agents.

A tool never decides whether it may run; the PolicyEngine does, and every call
reaches a tool only through ``AgentContext.execute_tool`` (which authorizes
first). A tool's job is narrow: do one bounded thing, write what it observed
into the content-addressed evidence store, and return a structured result that
references evidence by hash. No tool invents an external fact; lookups carry a
``source`` and ``retrieved_at`` so provenance survives into the knowledge graph.

The callable contract matches what ``AgentContext`` expects: ``tool(req) ->
dict``. The returned dict always includes ``evidence_refs``.
"""
from __future__ import annotations

import abc
import json

from sunflower_thresh.evidence.store import EvidenceStore
from sunflower_thresh.policy.decision import ToolRequest
from sunflower_thresh.schemas.evidence import EvidenceKind
from sunflower_thresh.schemas.target import ActionClass


class Tool(abc.ABC):
    """Base class for a gated, evidence-producing tool."""

    name: str = ""
    action_class: ActionClass = ActionClass.PASSIVE
    evidence_kind: EvidenceKind = EvidenceKind.LOG

    def __init__(self, store: EvidenceStore | None = None):
        self.store = store or EvidenceStore()

    def _record(self, data: bytes | None = None, *, obj=None,
                target: str = "", source: str | None = None, meta: dict | None = None) -> str:
        """Store evidence as a typed, provenanced item; return its content hash.

        The kind and producer come from the tool (``evidence_kind`` / ``name``),
        so a stored artifact always knows what it is and what made it.
        """
        if obj is not None:
            data = json.dumps(obj, sort_keys=True).encode()
        item = self.store.put_item(data or b"", kind=self.evidence_kind,
                                   produced_by=self.name, target=target,
                                   source=source, meta=meta)
        return item.sha256

    @abc.abstractmethod
    def __call__(self, req: ToolRequest) -> dict:
        ...


#: Lifted from the session module rather than importing it: that module is 336
#: lines of session storage this package has no use for, and this is a pure
#: twelve-line function. It lives on the base rather than in one transport,
#: because both the single-shot and the conversational client need it.
#: The ORDER is load-bearing and not cosmetic -- see the docstring.
def merge_auth_headers(caller: dict, session: dict) -> dict:
    """Session headers UNDER the caller's, never over them.

    This was `headers.update(session)`, so the stored session won -- and
    `authed` defaults to True. A forged JWT passed in `headers`, which is the
    whole point of the jwt_tool workflow the prompt teaches, was silently
    dropped, the real token was sent instead, and the result reported success
    as the session user. The forgery test could not fail and could not pass:
    it was never run. An explicit header is the caller saying what it wants.
    """
    out = dict(session)
    out.update({k: v for k, v in (caller or {}).items() if v is not None})
    return out
