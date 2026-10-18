"""Content-addressed evidence store.

Store raw bytes by their sha256. Same content -> same ref, so a finding's
evidence is deduplicated, immutable, and tamper-evident (the ref *is* the hash).
In-memory by default; give it a directory to persist.

Beyond the raw bytes, the store keeps a typed ``EvidenceItem`` per artifact --
what kind it is, which tool produced it, against what target, and (for external
facts) its source and retrieval time. That record is what makes evidence
auditable and reproducible: a finding points at a hash, and the hash resolves to
both the exact bytes and their provenance. The item index persists as
``evidence_index.jsonl`` next to the blobs.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from sunflower_thresh.schemas.evidence import EvidenceItem, EvidenceKind


class EvidenceStore:
    def __init__(self, root: str | Path | None = None, redactor=None):
        self.root = Path(root) if root else None
        #: Optional ``bytes -> (bytes, meta)`` applied by `put_item` BEFORE hashing.
        #:
        #: Before, not after, because an artifact's identity IS its hash -- so
        #: redaction cannot be retrofitted to a store that already holds the
        #: original. The consequence is deliberate and documented in redact.py: the
        #: stored artifact and the live reply have different refs, which is the
        #: trade for not keeping personal data in an immutable directory.
        #:
        #: Only `put_item` applies it. `put` is the raw content-addressed primitive
        #: and several callers rely on getting back the hash of exactly what they
        #: passed; redaction belongs with the provenance record, which is where a
        #: caller declares what an artifact IS.
        self.redactor = redactor
        self._mem: dict[str, bytes] = {}
        self._items: dict[str, EvidenceItem] = {}
        # Content addressing already makes a `put` idempotent -- two agents
        # storing the same bytes get the same ref and neither is wrong, which
        # is most of what a shared store needs. The index is the part that is
        # not: "first record for a ref wins" is a check and then a write, and
        # the append to evidence_index.jsonl is a second one.
        self._lock = threading.Lock()
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ref(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def put(self, data: bytes) -> str:
        ref = self._ref(data)
        self._mem[ref] = data
        if self.root:
            p = self.root / ref.replace(":", "_")
            if not p.exists():
                p.write_bytes(data)
        return ref

    def path_for(self, ref: str) -> Path | None:
        """Where these bytes live on disk, or None for an in-memory store.

        A file tool -- exiftool, strings, binwalk, steghide -- needs a path, and
        the blob already has one. Handing back the store's own name rather than
        letting a caller choose is what keeps the artifact auditable and keeps
        the filename out of anyone else's control: it is the content hash.
        """
        if not self.root:
            return None
        p = self.root / ref.replace(":", "_")
        return p if p.exists() else None

    def put_json(self, obj) -> str:
        return self.put(json.dumps(obj, sort_keys=True).encode())

    def put_item(
        self,
        data: bytes,
        *,
        kind: EvidenceKind = EvidenceKind.LOG,
        produced_by: str = "",
        target: str = "",
        source: str | None = None,
        meta: dict | None = None,
    ) -> EvidenceItem:
        """Store bytes AND a typed provenance record; return the EvidenceItem.

        First record for a ref wins, so re-storing identical bytes keeps the
        original provenance (content addressing means the ref is the same).

        When a redactor is configured it runs here, before the bytes are hashed, and
        what it removed is recorded in the item's meta -- so the provenance says the
        artifact was redacted and by which policy version, rather than leaving a
        reader to wonder whether the bot really said "[REDACTED:email]".
        """
        if self.redactor is not None:
            try:
                data, redaction_meta = self.redactor(data)
            except Exception:  # noqa: BLE001 - a redactor that fails must not drop
                # evidence, but it must not silently store unredacted bytes either.
                # Recording the failure keeps that visible.
                redaction_meta = {"error": "redactor raised; bytes stored as captured"}
            if redaction_meta:
                meta = {**(meta or {}), "redacted": redaction_meta}
        ref = self.put(data)
        with self._lock:
            if ref not in self._items:
                item = EvidenceItem(kind=kind, sha256=ref, produced_by=produced_by,
                                    target=target, source=source, meta=meta or {})
                self._items[ref] = item
                if self.root:
                    with (self.root / "evidence_index.jsonl").open("a") as fh:
                        fh.write(item.model_dump_json() + "\n")
            return self._items[ref]

    def item(self, ref: str) -> EvidenceItem | None:
        return self._items.get(ref)

    def items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def get(self, ref: str) -> bytes:
        if ref in self._mem:
            return self._mem[ref]
        if self.root:
            p = self.root / ref.replace(":", "_")
            if p.exists():
                return p.read_bytes()
        raise KeyError(ref)

    def verify(self, ref: str) -> bool:
        """Re-hash stored content and confirm it still matches its ref."""
        try:
            return self._ref(self.get(ref)) == ref
        except KeyError:
            return False
