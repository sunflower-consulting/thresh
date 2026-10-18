"""Can this engagement run, and what exactly will it send?

Run before burning a message budget, and run in front of the customer. It answers
the four questions an operator otherwise has no way to answer:

1. **Is the scope sound?** The inventory verifies, the authorization is on file AND
   still active, the limits are signed, and the capabilities say what we may do.
2. **Can we reach them?** DNS resolves, TLS verifies, the auth header works, and the
   reply parser actually extracted a reply rather than a protocol frame.
3. **What can we test?** Reachable cases, and for every unreachable one the reason --
   a missing binding, a capability the client did not grant, a channel the bot cannot
   seed. The report's coverage table is this list, so getting it wrong here is how a
   report claims ground nobody tested.
4. **What will it cost?** Messages against the signed budget, before rather than after.

THE RULE THIS MODULE IS ORGANISED AROUND.

Every active call in a run passes `PolicyEngine.authorize`, which can refuse it for
one of eleven reasons. A preflight that passes an engagement the gate will refuse is
worse than no preflight: the operator opens the window, the first request is denied,
and the finding is discovered after the budget is gone. So the checks here are
ordered to mirror `policy.engine._evaluate` and are **partitioned against
`DenyReason` itself** -- every member is either predicted here or named in
`UNPREDICTABLE` with a written reason, and a test asserts the partition is total. Add
a reason to the gate and this module fails until it says what it will do about it.

That partition found four the engine this was ported from could not predict:
`KILLSWITCH`, `INVENTORY_UNVERIFIED`, `EXTERNAL_NETWORK_BLOCKED`, and -- the one that
matters most -- `AUTH_INACTIVE`. `AuthorizationRecord.is_active()` has always existed
and the gate has always enforced it; the old preflight read `reference` and never
asked about expiry, so an engagement whose authorization lapsed overnight passed its
morning preflight and then had every single request refused.

Everything external is injected -- the resolver, the connector, the message sender,
the collector's own fetcher -- so the whole thing runs offline as a unit test and
`--mock` shows the output shape and each failure shape with no customer endpoint at
all. That is not only for tests: an operator rehearsing the morning of a window wants
to see what "auth rejected" looks like before it happens for real.
"""
from __future__ import annotations

import enum
import ipaddress
import re
import socket
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sunflower_thresh import nethttp
from sunflower_thresh import termstyle as ts
from sunflower_thresh.collector import CALLBACK_CAPABILITY
from sunflower_thresh.corpus import CONSUMPTION_TRIALS, EXPENSIVE_RUNGS
from sunflower_thresh.policy.decision import DenyReason


class Status(str, enum.Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""
    hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (Status.PASS, Status.WARN)


#: A question so ordinary that a bot which cannot answer it is not working, and so
#: harmless that sending it needs no capability. One message, used to prove the whole
#: path -- DNS, TLS, auth, the reply parser -- before anything adversarial is sent.
HELLO = "Hello, what can you help me with?"

#: What `sweep_builtins` gives the baseline arm when no --trials is passed:
#: `BaselineArm(turns, trials or 10)`. Here so the estimate and the sweep
#: cannot drift silently.
BASELINE_TRIALS = 10

#: A `$VAR` the inventory will NOT expand, anchored so `${VAR}` does not match.
_BARE_ENV_VAR = re.compile(r"\$(?!\{)[A-Za-z_][A-Za-z0-9_]*")

#: Every capability switch this package understands. The inventory stores them in a
#: plain `dict[str, bool]`, which `extra="forbid"` cannot police, so this is the only
#: place a misspelt switch can be caught -- and a misspelt switch reads as OFF.
#:
#: Sourced from the comment block on `TargetInventory.capabilities`, which is where
#: the vocabulary was documented and nowhere enforced.
KNOWN_CAPABILITIES: frozenset[str] = frozenset({
    "modify_target", "require_human_for_destructive", "solve_captcha", "osint",
    "plant_content", "out_of_band_callback", "capture_harmful_content",
    "multi_turn_escalation",
})

#: The one tool without which no case can run. `chat_say` is how a turn reaches the
#: target; an inventory that omits it would have every case denied TOOL_NOT_PERMITTED.
CHAT_TOOL = "chat_say"

#: Every tool name this package answers to, plus the two local verbs the gate permits
#: everywhere. An `allowed_tools` entry outside this set grants nothing: `tool_allowed`
#: is exact membership, so the operator believes they permitted a tool and the gate
#: hands out nothing -- the same fail-open shape as a misspelt capability, and it is
#: live in this repository's own test fixture, which permits `chat_send` and
#: `chat_history`, neither of which exists.
KNOWN_TOOLS: frozenset[str] = frozenset({
    "chat_open", "chat_say", "chat_reset", "chat_probe",
    "tool_search", "skill_detail",
})

# ----- the gate's refusals, and which of them are knowable in advance ------------

#: `DenyReason` -> the check in this module that predicts it. A denial with a check
#: here cannot surprise an operator mid-window.
PREDICTS: dict[DenyReason, str] = {
    DenyReason.KILLSWITCH: "kill-switch",
    DenyReason.NO_INVENTORY: "inventory",
    DenyReason.INVENTORY_UNVERIFIED: "inventory",
    DenyReason.AUTH_INACTIVE: "authorization",
    DenyReason.TOOL_NOT_PERMITTED: "scope",
    DenyReason.TARGET_OUT_OF_SCOPE: "scope",
    DenyReason.EXTERNAL_NETWORK_BLOCKED: "scope",
    DenyReason.CAPABILITY_DISABLED: "capabilities",
    DenyReason.MESSAGE_BUDGET_EXHAUSTED: "message budget",
}

#: Refusals this command deliberately does not predict, each with the reason it
#: cannot be. Written down rather than omitted, because the difference between "we
#: cannot know this yet" and "nobody thought about it" is the whole value of the
#: partition below.
UNPREDICTABLE: dict[DenyReason, str] = {
    DenyReason.RATE_LIMITED: (
        "a sliding window asks when calls land, not whether the run is feasible. A "
        "run comfortably inside its budget can still trip it transiently, which is "
        "why sweep.TRANSIENT_DENIALS retries it rather than stopping. Predicting it "
        "would mean predicting the scheduler."
    ),
    DenyReason.DESTRUCTIVE_NEEDS_HUMAN: (
        "no tool on the chat path is DESTRUCTIVE -- the chat verbs are PASSIVE or "
        "ACTIVE -- so this cannot fire on a chat engagement today. It becomes "
        "predictable the moment a planting tool that modifies the target is added, "
        "and this entry is what will fail then."
    ),
}


def unpredicted_denials() -> set[DenyReason]:
    """Refusals the gate can give that this module neither predicts nor excuses.

    Empty is the only acceptable answer, and the test that asserts so is the drift
    alarm: a new `DenyReason` lands in the gate and preflight says it cannot see it.
    """
    return set(DenyReason) - set(PREDICTS) - set(UNPREDICTABLE)


# ----- the collector probe -------------------------------------------------------

class CollectorReach(str, enum.Enum):
    """How far a collector has actually been shown to be reachable.

    Graded because the honest answer is graded, and because the state that matters
    to a report -- "the TARGET can reach it" -- is not on this list and cannot be.
    Nothing short of the target's own request proves the target's own reach, so that
    is what the run measures and what preflight must not pretend to have measured.
    """

    #: No collector. The corpus's out-of-band cases are UNREACHABLE, which the loader
    #: already reports; nothing here is wrong, there is simply nothing to probe.
    UNCONFIGURED = "unconfigured"
    #: A token was minted and fetched on loopback and the collector recorded nothing.
    #: The listener is broken, which is worse than absent: cases will run.
    BROKEN = "broken"
    #: It works, and it is bound to loopback, so NOTHING outside this machine can
    #: ever reach it. Definitive rather than cautious.
    LOOPBACK_ONLY = "loopback_only"
    #: Bound off-loopback, and the advertised URL resolved and answered with the
    #: minted token recorded. Proves the name and the port work from HERE. Does not
    #: prove the target's network can get here.
    NAME_ANSWERS = "name_answers"
    #: Bound off-loopback, and the advertised URL did not answer or recorded no hit.
    UNREACHABLE_BY_NAME = "unreachable_by_name"
    #: The advertised host is not this machine, so the probe refused to fetch it and
    #: sent nothing. Fetching it would have put traffic on a third party's host on
    #: the strength of a typo.
    REFUSED_OFF_HOST = "refused_off_host"


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() in ("localhost", "localhost.localdomain")


def _local_addresses(resolver: Callable[[str], list[str]] | None = None) -> set[str]:
    """Every address this machine answers on, for the off-host refusal.

    Best effort on purpose: a machine behind NAT does not know its public address, so
    a miss here yields REFUSED_OFF_HOST rather than a fetch. Refusing to probe is a
    worse readout and a better default than sending a request to whatever the
    operator mistyped.
    """
    if resolver is not None:
        return set(resolver(""))
    out = {"0.0.0.0", "::", "127.0.0.1", "::1", "localhost"}
    try:
        name = socket.gethostname()
        out.add(name)
        for info in socket.getaddrinfo(name, None):
            out.add(info[4][0])
    except OSError:
        pass
    return out


def probe_collector(collector, *, advertised: str = "", fetch=None,
                    local: set[str] | None = None,
                    timeout: float = 5.0) -> tuple[CollectorReach, str]:
    """(how far it reaches, what was observed). Sends nothing to the target.

    The sequence matters. A token is minted and fetched on LOOPBACK first, because a
    collector that does not record its own traffic is broken and every later answer
    would be about the network instead. Only then is the advertised URL tried, and
    only when it names this machine.
    """
    if collector is None or not getattr(collector, "running", False):
        return CollectorReach.UNCONFIGURED, "no collector"

    token = collector.mint("preflight")
    do_fetch = fetch if fetch is not None else _fetch
    try:
        do_fetch(collector.url_for(token), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - any failure here means "did not arrive"
        return CollectorReach.BROKEN, f"loopback fetch raised {type(exc).__name__}: {exc}"
    if not collector.hits(token):
        return CollectorReach.BROKEN, "fetched its own URL and recorded no hit"

    if _is_loopback(collector.host):
        return (CollectorReach.LOOPBACK_ONLY,
                f"works, bound to {collector.netloc}")

    if not advertised:
        return (CollectorReach.LOOPBACK_ONLY,
                f"bound to {collector.netloc} but no URL is advertised to cases")

    host = _advertised_host(advertised)
    known = local if local is not None else _local_addresses()
    if host not in known:
        return (CollectorReach.REFUSED_OFF_HOST,
                f"{host} is not an address of this machine; nothing was sent")

    probe = collector.mint("preflight-advertised")
    url = advertised.rstrip("/") + "/c/" + probe
    try:
        do_fetch(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return (CollectorReach.UNREACHABLE_BY_NAME,
                f"{url} raised {type(exc).__name__}: {exc}")
    hits = collector.hits(probe)
    if not hits:
        return CollectorReach.UNREACHABLE_BY_NAME, f"{url} answered but recorded no hit"
    return (CollectorReach.NAME_ANSWERS,
            f"{url} recorded a hit from {hits[0].get('client', '?')}")


def _advertised_host(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url if "//" in url else "//" + url)
    return (parsed.hostname or "").lower()


def _fetch(url: str, *, timeout: float = 5.0) -> int:
    """GET a URL with the package's own transport. Returns the status code."""
    from urllib.parse import urlparse

    parsed = urlparse(url if "//" in url else "http://" + url)
    tls = parsed.scheme == "https"
    port = parsed.port or (443 if tls else 80)
    conn = nethttp.connect(parsed.hostname or "", port, tls=tls, timeout=timeout)
    try:
        conn.request("GET", parsed.path or "/")
        return conn.getresponse().status
    finally:
        conn.close()


class ScopeError(ValueError):
    """The scope document cannot be read, or is not one."""


def load_inventory(path, *, env: dict | None = None):
    """A `TargetInventory` from a YAML scope document. The loader that was missing.

    Nothing in this package turned a file into an inventory: every inventory in the
    repository is hand-written Python, in `demo` and in two test fixtures. So the
    signed artifact that defines what may be touched had no way to arrive from the
    place it is actually signed, and `thresh preflight` is the command that needs it.

    Deliberately thin. `TargetInventory` already forbids unknown fields and freezes
    itself, and both of those are load-bearing: the model's own comment records that
    `expires` for `expires_on` yields an authorization that never expires, which is a
    typo that fails OPEN. So this function's whole job is to hand pydantic the
    mapping and turn its complaint into a sentence an operator can act on, rather
    than to validate anything itself -- a second opinion about the schema is a second
    thing to keep true.

    `${VAR}` is NOT expanded here. The inventory expands it at read time, in
    `resolved_auth_headers`, specifically so a credential never sits in the loaded
    object; expanding it on load would put it there.
    """
    import yaml

    from sunflower_thresh.schemas.target import TargetInventory

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ScopeError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScopeError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScopeError(f"{path} must be a mapping, not {type(raw).__name__}")
    try:
        return TargetInventory.model_validate(raw)
    except Exception as exc:          # pydantic's message is the useful part
        raise ScopeError(
            f"{path} is not a scope document this package understands:\n{exc}\n"
            f"An unknown field is refused rather than ignored, because a typo in a "
            f"scope document fails OPEN -- `expires` for `expires_on` is an "
            f"authorization that never expires.") from exc


def mock_sender(profile: str = "healthy"):
    """A fake target. Profiles: healthy, instrumented, unauthorised, streaming,
    nested, silent, stateless.

    Each profile is a failure actually seen in the wild rather than an invented one:
    a bearer token that was never activated, an SSE endpoint whose frames were being
    read as the reply, a reply nested under `data.message.text`, and a 200 with an
    empty body (a gateway that accepted the request and dropped it).
    """
    def send(_host, _port, _path, _message, **_kw):
        if profile == "unauthorised":
            return {"status": 401, "reply": "", "error": "401 Unauthorized"}
        if profile == "silent":
            return {"status": 200, "reply": "", "raw": ""}
        if profile == "streaming":
            return {"status": 200, "reply": "Hi! I can help with orders.",
                    "stream": True, "tool_calls": []}
        if profile == "nested":
            return {"status": 200, "reply": "Hi! I can help with orders.",
                    "reply_path": "data.message.text"}
        if profile == "stateless":
            # No conversation id. Ordinary for a stateless HTTP wrapper, and the most
            # expensive single fact about a target: most of the corpus is multi-turn.
            return {"status": 200, "reply": "Hi! I can help with orders."}
        if profile == "instrumented":
            # The one every engagement should be asking for: the reply reports which
            # tools the turn called, which is what makes excessive agency observable
            # rather than inferred.
            return {"status": 200, "reply": "Hi! I can help with orders.",
                    "conversation_id": "c-1", "tool_calls": []}
        return {"status": 200, "reply": "Hi! I can help with orders and returns.",
                "conversation_id": "c-1"}
    return send


@dataclass
class EngagementPreflight:
    """Ordered checks against one chatbot engagement."""

    inv: object
    #: The gate itself, when the caller has one. Given it, the scope checks ASK the
    #: gate instead of re-deriving its rules, which is the only way to guarantee
    #: preflight and the gate cannot disagree -- and they disagreed in the engine
    #: this was ported from, whose scope check skipped the external-network branch.
    engine: object | None = None
    host: str = ""
    port: int = 0
    path: str = "/chat"
    tls: bool = False
    verify: bool = True
    #: The case library to judge: the corpus when one was loaded, else the built-ins.
    cases: dict = field(default_factory=dict)
    corpus_sha: str = ""
    #: The run's collector, if it has one, and the URL cases would be given.
    collector: object | None = None
    advertised_callback: str = ""
    #: The other run-shaped facts `applicable_chat_cases` needs. `have_collector` is
    #: absent on purpose: it is MEASURED here by the probe rather than asserted.
    have_judge: bool = False
    have_tool_visibility: bool = False
    have_ingestion: bool = False
    allow_load: bool = False
    have_resolver: bool = False
    #: True when the gate reports a verified inventory for this project. Separate
    #: from `inv` because an inventory object exists whether or not the gate accepted
    #: it, and the gate's answer is the one a run lives by.
    verified: bool | None = None
    #: A sweep-level `--trials`, which REPLACES every case's own figure.
    #: 0 means the run uses each case's, which is the default and the
    #: only setting under which the consumption clamp survives.
    trials: int = 0
    killswitch: object | None = None
    #: Injected so the whole preflight is offline-testable.
    resolver: Callable[[str], str] | None = None
    connector: Callable[[str, int, bool, bool], None] | None = None
    sender: Callable | None = None
    collector_fetch: Callable | None = None
    local_addresses: set | None = None
    results: list = field(default_factory=list)

    # ----- what the probe and the selection established -----
    #: How far the collector was shown to reach, and what was observed.
    reach: CollectorReach = CollectorReach.UNCONFIGURED
    reach_detail: str = ""
    #: {case id: Scoring} and {case id: why not}, from `applicable_chat_cases`.
    runnable: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)

    # ----- plumbing -----

    @property
    def library(self) -> dict:
        """The cases this preflight is about: the corpus, or the built-ins.

        One property rather than a fallback inside `select`, because the fallback WAS
        inside `select` and `_runnable_cases` read `self.cases` -- so without a
        corpus the selection ran against the six built-ins while the estimate read an
        empty dict, and `thresh preflight` reported a message budget of one.
        """
        if self.cases:
            return self.cases
        from sunflower_thresh.tools.chat import BUILTIN_CHAT_CASES

        return BUILTIN_CHAT_CASES

    @property
    def bare_host(self) -> str:
        """`host` with any port or scheme removed, which is what scope is keyed on.

        An operator types what they were given, and what they were given is usually a
        URL or a netloc. `TargetInventory.domain_in_scope` matches a hostname and
        nothing else, so passing `chat.acme.example:443` through reports a target out
        of scope that the engagement plainly covers -- a FAIL that sends somebody
        editing the signed document to fix a typo in a flag.
        """
        raw = (self.host or "").strip()
        if "//" in raw:
            from urllib.parse import urlparse
            raw = urlparse(raw).hostname or ""
        if raw.startswith("["):                       # [2001:db8::1]:443
            return raw[1:raw.index("]")] if "]" in raw else raw.strip("[")
        if raw.count(":") == 1:                       # host:port, never bare IPv6
            raw = raw.split(":", 1)[0]
        return raw.lower().rstrip(".")

    def _add(self, name, status, detail="", hint="") -> CheckResult:
        r = CheckResult(name=name, status=status, detail=detail, hint=hint)
        self.results.append(r)
        return r

    def run(self) -> list[CheckResult]:
        """Every check, in the order the gate itself would refuse them."""
        self.results = []
        if self.check_killswitch().status is Status.FAIL:
            # Nothing else is worth reporting: the gate denies every active call
            # while this is engaged, so the run cannot start whatever else is true.
            #
            # FAIL rather than `not ok`, because a SKIP here means no switch was
            # wired to this preflight, which is a gap in what we can SEE and not a
            # reason to stop looking at everything else.
            return self.results
        self.check_inventory()
        self.check_authorization()
        scoped = self.check_scope()
        # The probe runs BEFORE the selection and feeds it, rather than reporting a
        # second opinion beside it. `--collector` was an operator assertion that
        # `applicable_chat_cases` took on trust; a measurement sitting next to that
        # boolean instead of replacing it is two answers waiting to disagree.
        self.reach, self.reach_detail = probe_collector(
            self.collector, advertised=self.advertised_callback,
            fetch=self.collector_fetch, local=self.local_addresses)
        self.runnable, self.skipped = self.select()
        self.check_tool_names()
        self.check_capability_names()
        self.check_capabilities()
        self.check_budget()
        self.check_consumption_clamp()
        self.check_baseline()
        self.check_bindings()
        self.check_coverage()
        self.check_collector()
        if not scoped.ok:
            # Stop here rather than carrying on to the remote checks. The gate would
            # refuse the request anyway, but a preflight that sends a message to a
            # host the scope does not cover -- even to find out whether it answers --
            # is the one thing this command must never do.
            self._add("target", Status.SKIP, "skipped: the scope check failed",
                      "nothing was sent")
            return self.results
        if not self.host:
            self._add("target", Status.SKIP, "no host given",
                      "pass --target <hostname> or put it in allowed_domains")
            return self.results
        if not self.check_dns().ok:
            return self.results
        if not self.check_tcp().ok:
            return self.results
        self.check_tls()
        self.check_auth()
        return self.results

    # ----- the scope -----

    def check_killswitch(self) -> CheckResult:
        """Predicts KILLSWITCH, which the gate checks before anything else.

        Cheap, and the engine this was ported from did not have it: a tripped switch
        made every call fail with a reason no readout mentioned.
        """
        ks = self.killswitch
        if ks is None:
            return self._add("kill-switch", Status.SKIP, "not wired to this preflight")
        if getattr(ks, "engaged", False):
            return self._add("kill-switch", Status.FAIL,
                             f"ENGAGED: {getattr(ks, 'reason', '') or 'no reason given'}",
                             "the gate denies every active call while it is engaged. "
                             "Reset it deliberately, as a human action")
        return self._add("kill-switch", Status.PASS, "clear")

    def check_inventory(self) -> CheckResult:
        """Predicts NO_INVENTORY and INVENTORY_UNVERIFIED. The signed record."""
        inv = self.inv
        if inv is None:
            return self._add("inventory", Status.FAIL, "no inventory loaded",
                             "the gate is fail-closed: with no inventory for this "
                             "project, nothing active is permitted")
        project = getattr(inv, "project", "") or "?"
        digest = getattr(inv, "content_hash", lambda: "")()
        if self.verified is False:
            return self._add(
                "inventory", Status.FAIL, f"{project}: loaded but NOT VERIFIED",
                "the gate verifies the content hash and the authorization together, "
                "so this is a bad expected hash or a lapsed authorization. Every "
                "target-bound call would be denied")
        detail = f"{project}, {digest[:19] if digest else 'no hash'}"
        if self.verified is None:
            return self._add("inventory", Status.WARN, detail + ", not asked of the gate",
                             "pass the engine's verified flag so this check can "
                             "predict INVENTORY_UNVERIFIED rather than assume it")
        return self._add("inventory", Status.PASS, detail + ", verified")

    def check_authorization(self) -> CheckResult:
        """Predicts AUTH_INACTIVE, and reports who signed for what.

        The expiry half is the one the old preflight was missing, and it is the
        expensive kind of miss: the gate enforces `is_active()` on every single call,
        so an authorization that lapsed overnight turns the whole window into a
        column of denials -- after the operator has told the client they are starting.
        """
        auth = getattr(self.inv, "authorization", None)
        if auth is None:
            return self._add("authorization", Status.FAIL, "no authorization record",
                             "an engagement without one is unauthorised access")
        by = getattr(auth, "authorized_by", "?")
        for_ = getattr(auth, "authorized_for", "?")
        ref = getattr(auth, "reference", "") or ""
        if not getattr(auth, "is_active", lambda: True)():
            granted = getattr(auth, "granted_on", "?")
            expires = getattr(auth, "expires_on", None)
            return self._add(
                "authorization", Status.FAIL,
                f"NOT ACTIVE: granted {granted}, "
                + (f"expired {expires}" if expires else "not yet in force"),
                "the gate refuses every call under an inactive authorization. Get the "
                "scope re-signed before the window, not during it")
        window = ""
        if getattr(auth, "expires_on", None):
            window = f", expires {auth.expires_on}"
        if not ref or ref.startswith("personally-owned"):
            return self._add(
                "authorization", Status.WARN,
                f"{by} -> {for_} ({ref or 'no reference'}){window}",
                "a client engagement needs the signed statement of work as `reference`")
        return self._add("authorization", Status.PASS,
                         f"{by} -> {for_}, ref {ref}{window}")

    def check_scope(self) -> CheckResult:
        """Predicts TARGET_OUT_OF_SCOPE, TOOL_NOT_PERMITTED, EXTERNAL_NETWORK_BLOCKED."""
        inv = self.inv
        domains = list(getattr(inv, "allowed_domains", []) or [])
        cidrs = list(getattr(inv, "allowed_cidrs", []) or [])
        if not (domains or cidrs):
            return self._add("scope", Status.FAIL, "nothing is in scope",
                             "the gate is fail-closed, so every request would be denied")
        if self.host:
            # `domain_in_scope` lowercases and strips a trailing dot and does NOT
            # parse a netloc, so `chat.acme.example:443` is out of scope against an
            # allowlist that contains `chat.acme.example`. Ask about the host alone.
            bare = self.bare_host
            in_scope = (inv.domain_in_scope(bare) if domains else False) or (
                inv.ip_in_scope(bare) if cidrs else False)
            if not in_scope:
                return self._add(
                    "scope", Status.FAIL, f"{bare} is not in scope",
                    f"add it to allowed_domains (have: {', '.join(domains) or 'none'})")
            # The gate's external-network block applies only to an ADDRESS, because
            # `target_ref` routes a hostname to `domain` and the check reads `ip`. So
            # this fires for an engagement that named its target by address, which is
            # exactly the case nobody notices: `external_network` defaults to false.
            blocked = self._external_block()
            if blocked is not None:
                return blocked
        detail = ", ".join(domains + cidrs)
        tools = list(getattr(inv, "allowed_tools", []) or [])
        if CHAT_TOOL not in tools:
            return self._add("scope", Status.FAIL, detail,
                             f"{CHAT_TOOL} is not in allowed_tools, so no case can run")
        return self._add("scope", Status.PASS, detail)

    def _external_block(self) -> CheckResult | None:
        """The FAIL for a globally-routable address with external_network off."""
        try:
            addr = ipaddress.ip_address(self.bare_host)
        except ValueError:
            return None             # a hostname never reaches the gate's ip branch
        if not addr.is_global or getattr(self.inv, "external_network", False):
            return None
        return self._add(
            "scope", Status.FAIL,
            f"{self.bare_host} is globally routable and external_network is false",
            "the gate denies a routable address unless the inventory says external_"
            "network: true. Set it, or name the target by hostname if that is what "
            "the scope actually covers -- the gate's block reads the address field only")

    @property
    def collector_usable(self) -> bool:
        """Reachable AND signed for. Both, and the second half was missing entirely.

        NAME_ANSWERS is the strongest state preflight can reach, and it is not proof
        the target can reach us -- but it is the operator's best available answer, and
        selecting on it beats selecting on an assertion. Every weaker state is false,
        including LOOPBACK_ONLY, where the answer is not "probably not" but "nothing
        outside this machine can, ever".

        The capability half closes a gap neither side had. `collector.py` declares
        `CALLBACK_CAPABILITY = "out_of_band_callback"` and says in its docstring that
        it is "gated by" that switch; nothing in the module reads the constant. And a
        corpus callback case emits only the `oob` need, which `applicable_chat_cases`
        satisfies from `have_collector` alone -- so the signed capability was never
        consulted on the corpus path at all, and a run could send callback payloads
        from a service the engagement never agreed we would stand up. A routable
        collector is a service the operator runs on someone else's behalf, which is
        precisely the kind of thing the inventory exists to record.
        """
        if self.reach is not CollectorReach.NAME_ANSWERS:
            return False
        enabled = getattr(self.inv, "capability_enabled", None)
        return bool(enabled(CALLBACK_CAPABILITY)) if enabled else False

    def select(self, **override) -> tuple[dict, dict]:
        """({case id: Scoring}, {case id: why not}) for this engagement.

        Delegates to `sweep.applicable_chat_cases`, which is the one place that
        answers "what can we test". Its docstring says answering it twice would let
        the two answers drift, and the engine this was ported from did exactly that:
        its capability check built a view of its own, so a preflight could report
        coverage the sweep would not deliver.
        """
        from sunflower_thresh.sweep import applicable_chat_cases

        inv = override.pop("inv", None) or self.inv
        kw = {"have_collector": self.collector_usable,
              "have_judge": self.have_judge,
              "have_tool_visibility": self.have_tool_visibility,
              "have_ingestion": self.have_ingestion,
              "allow_load": self.allow_load,
              "have_resolver": self.have_resolver}
        kw.update(override)
        return applicable_chat_cases(
            tuple(getattr(inv, "allowed_tools", ()) or ()),
            capabilities=inv, cases=self.library, **kw)

    def check_capabilities(self) -> CheckResult:
        """Predicts CAPABILITY_DISABLED, and prices each switch in cases.

        A WARN rather than a FAIL on purpose: a client who declined to let us plant
        content has made a decision, not a mistake. What matters is that the decision
        shows up as untested coverage rather than as a clean result -- and, now that
        a corpus can be loaded, as a NUMBER. "plant_content is off" is a
        configuration note; "plant_content is off, which is 57 cases" is a
        conversation with the client.

        The number comes from flipping the switch and asking the selection function
        again, rather than from a table of which need maps to which capability. A
        second table is a second thing to keep true, and this one cannot drift.
        """
        inv = self.inv
        interesting = ("plant_content", "out_of_band_callback", "multi_turn_escalation",
                       "capture_harmful_content", "modify_target")
        enabled = getattr(inv, "capability_enabled", lambda *_: False)
        off = [c for c in interesting if not enabled(c)]
        feats = list(getattr(inv, "target_features", ()) or ())
        chans = list(getattr(inv, "seedable_channels", ()) or ())
        detail = (f"target: {', '.join(feats) or 'nothing declared'}; "
                  f"seedable: {', '.join(chans) or 'none'}")
        if not off:
            return self._add("capabilities", Status.PASS, detail)
        priced = [f"{cap} (+{n})" if n else cap
                  for cap, n in ((c, self.cases_gained_by(c)) for c in off)]
        return self._add("capabilities", Status.WARN, detail,
                         f"off: {', '.join(priced)} -- the figure is how many more "
                         f"cases would run with the switch on. Cases needing these "
                         f"are reported UNREACHABLE, not clean")

    def check_tool_names(self) -> CheckResult:
        """An `allowed_tools` entry no tool answers to grants nothing.

        `tool_allowed` is exact membership, so a name this package does not have is
        not a permission with a typo in it -- it is no permission at all, and the
        gate refuses the call the operator believed they had allowed. Checked
        alongside the capability names because it is the same mistake in the other
        half of the scope document, and both read as "I granted that" right up to
        the denial.
        """
        tools = tuple(getattr(self.inv, "allowed_tools", ()) or ())
        unknown = sorted(set(tools) - KNOWN_TOOLS)
        if not unknown:
            return self._add("tool names", Status.PASS,
                             ", ".join(tools) if tools else "none permitted")
        return self._add(
            "tool names", Status.FAIL, f"no such tool: {', '.join(unknown)}",
            f"the gate matches a tool name exactly, so these permit nothing. This "
            f"package has {', '.join(sorted(KNOWN_TOOLS))}")

    def check_capability_names(self) -> CheckResult:
        """A misspelt capability reads as off, and nothing anywhere says so.

        `extra="forbid"` on the inventory guards MODEL FIELDS, and `capabilities` is
        a `dict[str, bool]` -- so `plant_contnet: true` validates perfectly, reads as
        off through `capability_enabled`'s safe default, and silently demotes every
        case needing it to UNREACHABLE. The vocabulary exists only in a comment, so
        there is nothing else in either package that could catch this.

        It is the same shape as the typos `extra="forbid"` was added for, and the
        same direction: a scope document that says yes is read as no, and the run
        reports a clean result for cases it never attempted.
        """
        caps = dict(getattr(self.inv, "capabilities", {}) or {})
        unknown = sorted(set(caps) - KNOWN_CAPABILITIES)
        if not unknown:
            return self._add("capability names", Status.PASS,
                             f"{len(caps)} switch(es), all recognised" if caps
                             else "none set, so every switch takes its safe default")
        return self._add(
            "capability names", Status.FAIL, f"not a capability: {', '.join(unknown)}",
            "a name this package does not know reads as OFF, whatever the document "
            "says, because `capabilities` is a plain dict and extra=forbid does not "
            "reach inside it. Cases needing the switch report UNREACHABLE and the "
            "report shows no gap. Fix the spelling or remove the line")

    def cases_gained_by(self, capability: str) -> int:
        """How many more cases would run if this switch were on. 0 when none.

        Measured by flipping it on a copy of the inventory and re-selecting, so it
        is the sweep's own answer rather than an estimate of it.
        """
        if not hasattr(self.inv, "model_copy"):
            return 0
        caps = dict(getattr(self.inv, "capabilities", {}) or {})
        caps[capability] = True
        try:
            flipped = self.inv.model_copy(update={"capabilities": caps})
        except Exception:  # noqa: BLE001 - an inventory that will not copy prices nothing
            return 0
        gained, _ = self.select(inv=flipped)
        return max(0, len(gained) - len(self.runnable))

    def check_budget(self) -> CheckResult:
        """Predicts MESSAGE_BUDGET_EXHAUSTED, before the sweep finds out mid-run."""
        signed = getattr(self.inv, "max_target_messages", 0) or 0
        spend = self.estimated_messages()
        if not signed:
            return self._add("message budget", Status.WARN, f"~{spend} needed, no cap set",
                             "set max_target_messages: an uncapped run against "
                             "production is the one mistake that cannot be undone")
        if spend > signed:
            return self._add(
                "message budget", Status.FAIL, f"~{spend} needed, {signed} signed",
                "raise the signed figure, cut --sweep-trials, or select fewer cases; "
                "the gate will stop the run mid-sweep otherwise")
        return self._add("message budget", Status.PASS,
                         f"~{spend} of {signed} signed "
                         f"({spend * 100 // max(1, signed)}%)")

    def estimated_messages(self) -> int:
        """Attack arm per case, plus matched controls, plus ONE shared baseline.

        The baseline is counted once because that is how it is sent: its questions do
        not depend on which case is being measured, so the sweep sends them once and
        re-scores them per case. Counting it per case overstates the full library by
        more than the run itself costs -- 12,588 messages against a 10,037-message
        run, measured on the corpus -- and would tell an operator a feasible run is
        impossible.

        Two corrections to the formula this was ported with, both found by reading
        what the sweep actually sends rather than what the estimate assumed:

        The baseline's trial count is NOT `max(case.control_trials)`. `sweep_builtins`
        builds `BaselineArm(turns, trials or 10)`, so it is the run's override or a
        hardcoded ten, and the two disagree for any ordinary corpus selection.

        And a sweep-level `--trials` REPLACES every case's own figure, for the attack
        and control arms both. An estimate that reads `case.trials` is only right for
        a run that passes no override, so the override belongs in the estimate.
        """
        runnable = self._runnable_cases()
        n = self.trials
        attack = sum(len(c.turns) * (max(1, n) if n else c.trials) for c in runnable)
        matched = sum(len(c.control_turns) * (n if n else c.control_trials)
                      for c in runnable)
        base = tuple(getattr(self.inv, "baseline_turns", ()) or ())
        # `sweep_builtins`: BaselineArm(turns, trials or 10). Not derived from a case.
        baseline = len(base) * (n or BASELINE_TRIALS)
        # +1 for the hello probe this preflight itself sends.
        return attack + matched + baseline + 1

    def check_consumption_clamp(self) -> CheckResult:
        """A `--trials` override discards the clamp that stops a load test.

        `to_chat_case` caps a case whose oracle measures COST rather than disclosure
        -- `response_length`, `tool_call_count` -- at two trials, because ten trials
        plus ten controls against a client's assistant is not a measurement, it is a
        load test nobody signed for. A sweep-level override replaces `case.trials`
        outright, so it silently lifts that cap on exactly the cases it was written
        for.
        """
        n = self.trials
        if not n or n <= CONSUMPTION_TRIALS:
            return self._add("trial cap", Status.PASS,
                             f"consumption cases stay at {CONSUMPTION_TRIALS} trials")
        expensive = [cid for cid, case in sorted(self.library.items())
                     if cid in self.runnable
                     and {k for rung in (getattr(case, "oracles", ()) or ())
                          for k in rung} & set(EXPENSIVE_RUNGS)]
        if not expensive:
            return self._add("trial cap", Status.PASS,
                             f"--trials {n}, and no selected case measures cost")
        return self._add(
            "trial cap", Status.FAIL,
            f"--trials {n} lifts the {CONSUMPTION_TRIALS}-trial cap on "
            f"{len(expensive)} case(s)",
            f"{', '.join(expensive[:6])} measure cost rather than disclosure, and the "
            f"cap exists because one long reply is a finding while a thousand is an "
            f"outage we caused. Drop --trials, or deselect them with --only")

    def _runnable_cases(self) -> list:
        return [c for cid, c in self.library.items() if cid in self.runnable]

    def check_baseline(self) -> CheckResult:
        base = tuple(getattr(self.inv, "baseline_turns", ()) or ())
        if not base:
            return self._add(
                "control arm", Status.FAIL, "no baseline_turns",
                "with no control arm NOTHING reaches VALIDATED -- the report would "
                "show an attack success rate of 1.00 beside 0 validated. Add 4-6 "
                "ordinary questions a real user of this bot would ask")
        if len(base) < 3:
            return self._add("control arm", Status.WARN,
                             f"{len(base)} baseline question(s)",
                             "3-6 is the useful range: one question tests one phrasing")
        runnable = self._runnable_cases()
        matched = sum(1 for c in runnable if c.control_turns)
        declared = Counter(getattr(c, "control_kind", "") or "none"
                           for c in runnable if not c.control_turns)
        detail = f"{len(base)} baseline questions, {matched} with a matched control"
        if declared:
            detail += ", " + ", ".join(f"{n} {k}" for k, n in declared.most_common())
        # A case whose corpus entry says `seeded` or `inseparable` has DECLARED that
        # no matched control can be sent, and why. One with neither has nobody's
        # reason, and the two must not share a column.
        if declared.get("none"):
            return self._add("control arm", Status.WARN, detail,
                             f"{declared['none']} case(s) have no control and no "
                             f"reason for not having one -- their findings rest on "
                             f"the shared baseline alone")
        return self._add("control arm", Status.PASS, detail)

    def unbound_names(self) -> list[str]:
        """Distinct placeholders nothing bound, as the selection reports them.

        From the `Scoring.unbound` of the SKIPPED half, because a case whose
        placeholder is unbound does not run -- so reading the runnable half would
        report an empty list and a clean preflight.
        """
        out: set[str] = set()
        for scoring in self._all_scorings():
            for name in getattr(scoring, "unbound", ()) or ():
                out.update(n for n in str(name).split(",") if n)
        return sorted(out)

    def _all_scorings(self) -> list:
        """Every case's Scoring, with nothing this run lacks held against it.

        A placeholder is unbound whatever the engagement's capabilities are, so the
        names are read from a fully-permissive selection. Reading the real one would
        report nothing: a case with an unbound placeholder does not run, so its
        Scoring is not in the runnable half at all.
        """
        permissive, _ = self.select(have_collector=True, have_judge=True,
                                    have_tool_visibility=True, have_ingestion=True,
                                    allow_load=True, have_resolver=True)
        return list(permissive.values())

    def check_bindings(self) -> CheckResult:
        if not self.cases:
            return self._add("bindings", Status.SKIP, "no corpus loaded",
                             "pass --corpus <checkout> to run the real library")
        unbound = self.unbound_names()
        if unbound:
            return self._add(
                "bindings", Status.WARN, f"{len(unbound)} unbound",
                f"{', '.join(unbound[:6])} -- cases needing these are UNREACHABLE. "
                f"The markers come from the client; the rest are engagement facts")
        return self._add("bindings", Status.PASS, "every placeholder bound")

    def check_coverage(self) -> CheckResult:
        """What will run, and -- the number a coverage table must not round off --
        how much of it will run on less than it was written for.

        `Scoring.partial` is a case that runs on a subset of its own oracles. It is
        not a lesser pass, it is a different claim: "we looked one of the two ways we
        meant to". A coverage figure that adds those to the full ones is the sort of
        number that gets copied into a report unqualified.
        """
        total = len(self.library)
        n = len(self.runnable)
        detail = f"{n} of {total} runnable ({n * 100 // max(1, total)}%)"
        detail += (f", corpus {self.corpus_sha[:12]}" if self.corpus_sha
                   else ", BUILT-IN cases only")
        partial = sum(1 for s in self.runnable.values()
                      if getattr(s, "partial", False))
        if partial:
            detail += f", {partial} on part of its ladder"
        if n == 0:
            return self._add("coverage", Status.FAIL, detail,
                             "nothing would run; see the unreachable list below")
        if n < total // 2:
            return self._add("coverage", Status.WARN, detail,
                             "under half the library -- the report's coverage table "
                             "will be mostly 'not tested', which is honest but thin")
        return self._add("coverage", Status.PASS, detail)

    def check_collector(self) -> CheckResult:
        """Is the collector reachable, and what does it buy?

        The check that did not exist. `--collector` was an operator ASSERTION fed
        straight into case selection -- "the collector will be reachable" -- and
        nothing ever probed it. What that buys is narrow and real: the out-of-band
        cases carry a `structural` oracle as well as the `callback` one, so they run
        either way and `Scoring.partial` records that they ran on less. The run is
        therefore honest about it. What was NOT honest is the selection: an operator
        who asserted a collector got those cases marked full-ladder when their
        strongest rung could never fire, and the exfiltration class is exactly where
        the difference between CALLBACK and STRUCTURAL is the difference between the
        target's own traffic and the model having emitted a string.
        """
        cost = self._collector_cost()
        if self.reach is CollectorReach.UNCONFIGURED:
            if not cost:
                return self._add("collector", Status.SKIP,
                                 "none, and no case would gain one")
            return self._add("collector", Status.SKIP, self.reach_detail, cost)
        if self.reach is CollectorReach.BROKEN:
            return self._add("collector", Status.FAIL, self.reach_detail,
                             "the listener is not recording its own traffic, so no "
                             "callback can ever be proved. Worse than having none: "
                             "nothing stops the cases being selected")
        if self.reach is CollectorReach.REFUSED_OFF_HOST:
            return self._add("collector", Status.FAIL, self.reach_detail,
                             "check the advertised URL. The probe will not fetch a "
                             "host this machine does not answer on, because a typo "
                             "would put our traffic on somebody else's server")
        if self.reach in (CollectorReach.LOOPBACK_ONLY,
                          CollectorReach.UNREACHABLE_BY_NAME):
            return self._add("collector", Status.FAIL, self.reach_detail,
                             (cost or "no case would gain one")
                             + ". Those cases are now selected WITHOUT the callback "
                               "rung rather than with a rung that cannot fire")
        return self._add(
            "collector", Status.WARN, self.reach_detail,
            "the name and port answer FROM HERE, which is necessary and not "
            "sufficient: only the target's own request proves the target can reach "
            "us, and that is what the run measures" + (f". {cost}" if cost else ""))

    def _collector_cost(self) -> str:
        """What a reachable collector changes, asked of the selection function.

        Deliberately reported as cases that move off the PARTIAL list rather than as
        cases unlocked. Measured on the corpus: a collector unlocks no additional
        case and moves three off partial, where counting every case that merely needs
        the URL MINTED as an unguessable marker would have claimed thirty-four.
        Nothing has to fetch a marker.
        """
        with_it, _ = self.select(have_collector=True)
        without, _ = self.select(have_collector=False)
        gained = sorted(set(with_it) - set(without))
        def part(d):
            return {cid for cid, s in d.items() if getattr(s, "partial", False)}
        promoted = sorted(part(without) - part(with_it))
        bits = []
        if gained:
            bits.append(f"{len(gained)} case(s) only run with one: "
                        f"{', '.join(gained[:6])}")
        if promoted:
            bits.append(f"{len(promoted)} case(s) run on their FULL ladder only with "
                        f"one, and on a weaker rung without: {', '.join(promoted[:6])}")
        return ". ".join(bits)

    # ----- reaching them -----

    def check_dns(self) -> CheckResult:
        resolve = self.resolver or (lambda h: socket.gethostbyname(h))
        try:
            addr = resolve(self.host)
        except Exception as exc:  # noqa: BLE001
            return self._add("dns", Status.FAIL, f"{self.host}: {exc}",
                             "check the hostname, or whether it resolves only inside "
                             "their network (then you need their VPN or a jump host)")
        return self._add("dns", Status.PASS, f"{self.host} -> {addr}")

    def check_tcp(self) -> CheckResult:
        port = self.port or (443 if self.tls else 80)
        connect = self.connector
        if connect is None:
            def connect(host, p, _tls, _verify):
                with socket.create_connection((host, p), timeout=10):
                    pass
        try:
            connect(self.host, port, False, False)
        except Exception as exc:  # noqa: BLE001
            return self._add("tcp", Status.FAIL, f"{self.host}:{port}: {exc}",
                             "firewalled, wrong port, or the service is down. Ask "
                             "whether our source address needs allowlisting")
        return self._add("tcp", Status.PASS, f"{self.host}:{port} open")

    def check_tls(self) -> CheckResult:
        if not self.tls:
            return self._add(
                "tls", Status.WARN, "plain HTTP",
                "a production chatbot is HTTPS; plain HTTP means either a staging "
                "instance or a misconfigured target URL. Confirm which")
        port = self.port or 443
        connect = self.connector
        if connect is None:
            def connect(host, p, tls, verify):
                conn = nethttp.connect(host, p, tls=True, timeout=10, verify=verify)
                try:
                    conn.connect()
                finally:
                    conn.close()
        try:
            connect(self.host, port, True, self.verify)
        except Exception as exc:  # noqa: BLE001
            if self.verify:
                return self._add(
                    "tls", Status.FAIL, f"certificate rejected: {exc}",
                    "if this is their internal CA or a self-signed staging cert, "
                    "re-run with --insecure AND write that down in the report: an "
                    "agent that accepts any certificate cannot tell the target from "
                    "whatever sits in front of it")
            return self._add("tls", Status.FAIL, f"{exc}", "the handshake itself failed")
        if not self.verify:
            return self._add("tls", Status.WARN, "connected, verification DISABLED",
                             "record this in the report; findings are attributed to a "
                             "host whose identity was not checked")
        return self._add("tls", Status.PASS, "certificate verifies")

    def check_auth(self) -> CheckResult:
        """One benign message. Proves auth, the reply parser and liveness together.

        Deliberately last and deliberately one message: everything before it is free,
        and this is the first thing that touches the customer's system.
        """
        headers = {}
        unresolved = []
        configured = dict(getattr(self.inv, "auth_headers", {}) or {})
        if hasattr(self.inv, "resolved_auth_headers"):
            headers = self.inv.resolved_auth_headers()
            unresolved = self.inv.unresolved_auth_headers()
        # `${VAR}` is the only spelling the inventory expands. A bare `$VAR` is not a
        # placeholder to it, so it is neither resolved nor reported unresolved: it is
        # sent to the client's endpoint as the literal seven characters. That looks
        # like a credential in their logs and authenticates nothing, and the operator
        # has no way to find out except by reading their logs with them.
        literal = sorted(n for n, v in configured.items()
                         if _BARE_ENV_VAR.search(str(v)))
        if literal:
            return self._add(
                "auth", Status.FAIL,
                f"{', '.join(literal)} contains a bare $VAR",
                "only ${VAR} is expanded. Written $VAR it is sent to the client "
                "verbatim, which authenticates nothing and puts a string that looks "
                "like a credential in their logs. Add the braces")
        if unresolved:
            return self._add(
                "auth", Status.FAIL, f"unset: {', '.join(unresolved)}"
                + (f" ({len(headers)} of {len(configured)} did resolve)"
                   if headers else ""),
                "export the environment variable. An unset one sends NO header rather "
                "than a literal ${TOKEN}, so the request would go out with less "
                "authority than the inventory was signed for")
        send = self.sender
        if send is None:
            return self._add("auth", Status.SKIP, "no sender configured")
        try:
            out = send(self.host, self.port or (443 if self.tls else 80), self.path,
                       HELLO, headers=headers, tls=self.tls, verify=self.verify)
        except Exception as exc:  # noqa: BLE001
            return self._add("auth", Status.FAIL, f"{type(exc).__name__}: {exc}",
                             "the request did not complete")
        status = int(out.get("status") or 0)
        if status in (401, 403):
            return self._add(
                "auth", Status.FAIL, f"HTTP {status}",
                "the credential is wrong, expired, or not yet activated. "
                f"{'Sent ' + ', '.join(headers) if headers else 'No auth header was sent'}")
        if status >= 400:
            return self._add("auth", Status.FAIL,
                             f"HTTP {status}: {out.get('error', '')}",
                             "wrong path, or the body shape is not what they expect "
                             "(--path, and check their API docs)")
        reply = (out.get("reply") or "").strip()
        if not reply:
            return self._add(
                "auth", Status.FAIL, "HTTP 200 with no reply extracted",
                "the request worked and the parser found nothing. Either a gateway "
                "accepted and dropped it, or the reply is nested -- find the field in "
                "their response and pass --reply-path a.b.c")
        shape = []
        if out.get("stream"):
            shape.append("streamed")
        if out.get("reply_path"):
            shape.append(f"nested at {out['reply_path']}")
        if out.get("conversation_id"):
            shape.append("stateful")
        if out.get("tool_calls") is not None:
            shape.append("reports tool calls")
        result = self._add(
            "auth", Status.PASS,
            f"HTTP 200, {len(reply)} chars"
            + (f" [{', '.join(shape)}]" if shape else "")
            + (f", {len(headers)} auth header(s)" if headers else ", unauthenticated"))
        self._compare_probe_with_declaration(out)
        return result

    def _compare_probe_with_declaration(self, out: dict) -> None:
        """Where the one benign message contradicts what the inventory declared.

        This is the check with the most leverage in the whole command, and it only
        became possible once a corpus could be loaded. The inventory is the signed
        record of what the client said their system is; the hello probe is one
        measurement of what it does. A disagreement between the two is not a detail,
        because `target_features` DECIDES WHICH CASES RUN -- so a declaration of
        `stateful` that the probe contradicts means the selection admitted every
        multi-turn case in the library on a false premise, and most of the library is
        multi-turn.

        Reported rather than acted on: the inventory is signed and this command does
        not get to quietly re-select against it. The operator takes it to the client,
        and the fix is a corrected scope document, not a flag.
        """
        runnable = self._runnable_cases()
        declared = set(getattr(self.inv, "target_features", ()) or ())

        stateful_seen = bool(out.get("conversation_id"))
        if "stateful" in declared and not stateful_seen:
            n = sum(1 for c in runnable
                    if len(getattr(c, "turns", ())) > 1
                    or "stateful" in (getattr(c, "needs", ()) or ()))
            self._add(
                "statefulness", Status.FAIL,
                "declared stateful, and the reply carries no conversation id",
                f"{n} of {len(runnable)} selected case(s) were admitted because the "
                f"inventory says this target is stateful. Either the session field "
                f"is somewhere this probe did not look -- find it and set "
                f"reply_path/transport -- or the declaration is wrong and those "
                f"cases cannot be measured at all")
        elif not stateful_seen:
            n = sum(1 for cid, why in self.skipped.items() if "stateful" in why)
            self._add("statefulness", Status.PASS,
                      "not declared, and the reply carries no conversation id"
                      + (f"; {n} case(s) skipped accordingly" if n else ""))

        if out.get("tool_calls") is None:
            gained = len(self.select(have_tool_visibility=True)[0]) - len(self.runnable)
            self._add(
                "tool visibility", Status.WARN,
                "the reply does not report tool calls",
                "ask the client for one line of logging on their side"
                + (f": {gained} more case(s) would run" if gained > 0 else "")
                + ". It is the highest-value thing they can give us, because it "
                  "turns excessive agency from inference into observed fact")
        elif not self.have_tool_visibility:
            gained = len(self.select(have_tool_visibility=True)[0]) - len(self.runnable)
            self._add("tool visibility", Status.WARN,
                      "the reply DOES report tool calls, and this run was not told",
                      f"pass --tool-visibility: {gained} more case(s) would run, and "
                      f"tool-call evidence is a rung above a structural match"
                      if gained > 0 else
                      "pass --tool-visibility so tool-call oracles can score")


_MARK = {Status.PASS: ("ok  ", ts.good), Status.WARN: ("warn", ts.warn),
         Status.FAIL: ("FAIL", ts.bad), Status.SKIP: ("--  ", ts.muted)}


def render(results: list[CheckResult], *, title: str = "engagement preflight") -> str:
    """Aligned verdict table, then the hints that matter.

    Hints only for FAIL and WARN: a hint beside every passing check trains an operator
    to skip the whole block, which is how the one that mattered gets missed.
    """
    width = max((len(r.name) for r in results), default=10)
    lines = [ts.header(f" {title}")]
    for r in results:
        mark, paint = _MARK[r.status]
        lines.append(f"  {paint(mark)}  {ts.name(r.name.ljust(width))}  "
                     f"{ts.muted(r.detail)}")
    bad = [r for r in results if r.status is Status.FAIL]
    warn = [r for r in results if r.status is Status.WARN]
    if bad or warn:
        lines.append("")
        for r in bad + warn:
            if r.hint:
                lines.append(f"  {ts.arrow()} {ts.name(r.name)}: {r.hint}")
    verdict = (ts.bad("do not start") if bad else
               ts.warn("can run, with gaps") if warn else ts.good("ready"))
    lines.append("")
    lines.append(f" {len(results)} checks  {ts.muted('->')}  {verdict}")
    if bad:
        lines.append(" " + ts.muted(
            "a FAIL means the run would waste the window or the budget, or the gate "
            "would refuse it -- not that it would be unsafe; the gate handles safety"))
    return "\n".join(lines)
