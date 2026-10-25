"""One place that opens a connection, so the harness can reach the internet.

Every native target-facing tool built its own `http.client.HTTPConnection`:
`tools/chat.py`, `tools/conversation.py`, `tools/session.py`, `tools/validate.py`,
`tools/http_recon.py`, `benchmark.py`. Plain HTTP, every one of them. STATUS.md
already names the consequence: "there is no target-facing TLS at all, so nothing on
the internet is reachable."

The loopback labs work, which is why this survived. Every real chat endpoint is
HTTPS, so "chatbot is the primary target type" was false for any engagement outside
a container on this machine — the harness could test the lab and nothing else.

This is the helper. It is small on purpose: one function that decides http vs https
and returns a connection the callers already know how to use, so adopting it is a
one-line change per call site rather than a rewrite of six tools.

On verification. `verify=True` is the default and it is the right default: an agent
that silently accepts any certificate cannot tell the target from whatever is between
it and the target, and a finding attributed to the wrong host is worse than no
finding. But self-signed certificates are *ordinary* on internal targets and in labs,
so `verify=False` exists — and when it is used, the caller is handed back the fact,
so a run can record that a connection was unverified rather than discovering it in
a report later.
"""
from __future__ import annotations

import http.client
import ssl

#: Ports that mean TLS when nobody said. 443 and the usual alternates; everything
#: else defaults to plain HTTP, which is what every existing caller assumed.
TLS_PORTS = frozenset({443, 8443, 9443, 10443})


def wants_tls(port: int, tls: bool | None = None) -> bool:
    """Should this connection be TLS? An explicit answer wins over the port."""
    if tls is not None:
        return bool(tls)
    return int(port) in TLS_PORTS


def connect(host: str, port: int, *, tls: bool | None = None, timeout: float = 30.0,
            verify: bool = True):
    """An HTTP or HTTPS connection to host:port.

    Returns the connection; the caller drives it exactly as before. `tls=None` infers
    from the port, so an existing call site that passes nothing keeps its current
    behaviour on 80 and gains TLS on 443.
    """
    if not wants_tls(port, tls):
        return http.client.HTTPConnection(host, port, timeout=timeout)
    return http.client.HTTPSConnection(host, port, timeout=timeout,
                                       context=tls_context(verify=verify))


def tls_context(*, verify: bool = True) -> ssl.SSLContext:
    """A TLS context, verified by default.

    Unverified is a real need -- self-signed certificates are ordinary on internal
    targets -- but it is never the default, because an agent that accepts any
    certificate cannot tell the target from whatever sits in front of it, and a
    finding attributed to the wrong host is worse than no finding.
    """
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def scheme_for(port: int, tls: bool | None = None) -> str:
    """"http" or "https", for building a URL that matches the connection."""
    return "https" if wants_tls(port, tls) else "http"
