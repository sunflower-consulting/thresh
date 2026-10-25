"""Append-only audit log.

Every policy decision -- allow and deny alike -- is recorded. This is what lets
you answer "why did the agent do that?" months later, and it is the evidence
that nothing ran out of scope. Writes only ever append.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._mem: list[dict] = []

    def record(self, event_type: str, **fields) -> dict:
        entry = {
            "at": datetime.now(UTC).isoformat(),
            "event": event_type,
            **fields,
        }
        with self._lock:
            self._mem.append(entry)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._mem)
