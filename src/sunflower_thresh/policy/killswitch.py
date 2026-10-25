"""Emergency stop, outside the agent's control.

A single flag that, once set, makes the policy engine deny every active action.
Wire OT stop conditions (watchdog trip, unexpected reboot) to `.trip()`.
"""
from __future__ import annotations

import threading


class KillSwitch:
    def __init__(self) -> None:
        self._engaged = False
        self._reason = ""
        self._lock = threading.Lock()

    def trip(self, reason: str) -> None:
        with self._lock:
            self._engaged = True
            self._reason = reason

    def reset(self) -> None:
        # Resetting is a deliberate human action; the engine never resets itself.
        with self._lock:
            self._engaged = False
            self._reason = ""

    @property
    def engaged(self) -> bool:
        with self._lock:
            return self._engaged

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason
