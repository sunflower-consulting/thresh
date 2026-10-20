"""Target inventory + authorization record.

The inventory is the single signed artifact that defines what the harness is
allowed to touch. It is loaded once, verified, and frozen. Nothing at runtime
mutates it -- least of all an agent.
"""
from __future__ import annotations

import enum
import hashlib
import ipaddress
import json
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

# Safe defaults for known capability switches: everything risky is OFF unless the
# project opts in, except require_human_for_destructive which defaults ON.
_CAPABILITY_SAFE_DEFAULTS = {"require_human_for_destructive": True}


class ActionClass(str, enum.Enum):
    """How much an action can affect a target.

    The policy engine treats these very differently: PASSIVE is cheap to allow,
    ACTIVE needs the target in scope, DESTRUCTIVE always needs a human token.
    """

    PASSIVE = "passive"        # read-only: banner read, pcap read, CVE lookup
    ACTIVE = "active"          # touches the target: scan, crawl, request
    DESTRUCTIVE = "destructive"  # may change target state: write, reset, fuzz-with-writes


class AuthorizationRecord(BaseModel):
    """Who authorized this scope, what, and when. A first-class artifact.

    For a lab target this may be self-authorization (personally owned gear).
    For a client engagement it records the signed statement of work.
    """

    #: A field this model does not know is a typo, and a typo in a scope
    #: document must not be silently dropped. Several of them fail OPEN:
    #: `expires` instead of `expires_on` leaves expires_on None, which is an
    #: authorization that never expires, and `capability` instead of
    #: `required_capability` leaves the capability check with nothing to
    #: check, so a request that should be refused is permitted.
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorized_by: str = Field(..., description="Name/identity granting authorization")
    authorized_for: str = Field(..., description="Operator who may act under it")
    scope_summary: str = Field(..., description="Plain-language description of the scope")
    granted_on: date
    expires_on: date | None = None
    reference: str | None = Field(
        default=None, description="SoW id, ticket, or 'personally-owned-lab'"
    )
    signature: str | None = Field(
        default=None, description="Detached signature over the inventory hash"
    )

    def is_active(self, on: date | None = None) -> bool:
        # Authorization dates are in the operator's locale ("granted on <date>"),
        # so compare against the local date; a UTC comparison makes an
        # authorization granted "today" read as inactive when local is a day
        # ahead of UTC.
        on = on or datetime.now().date()
        if on < self.granted_on:
            return False
        if self.expires_on is not None and on > self.expires_on:
            return False
        return True


class TargetInventory(BaseModel):
    """Explicit allowlist for one project. Frozen after load."""

    #: A field this model does not know is a typo, and a typo in a scope
    #: document must not be silently dropped. Several of them fail OPEN:
    #: `expires` instead of `expires_on` leaves expires_on None, which is an
    #: authorization that never expires, and `capability` instead of
    #: `required_capability` leaves the capability check with nothing to
    #: check, so a request that should be refused is permitted.
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str
    allowed_cidrs: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    allowed_devices: tuple[str, ...] = ()
    external_network: bool = False
    # Per-project tool allowlist. Empty tuple => no tools permitted (fail-closed).
    allowed_tools: tuple[str, ...] = ()
    # Kali tool CATEGORIES to permit wholesale, e.g. ["web", "vulnerability"].
    # A full Kali build is ~1200 binaries and `tool_allowed` is exact membership,
    # so without this a project would have to enumerate twelve hundred names to
    # use the catalog at all. Expanded into `allowed_tools` AT LOAD TIME by
    # harness.toolcatalog.expand_categories, so the inventory an engagement is
    # judged against still lists every tool by name -- the convenience is in
    # writing the file, never in what the gate checks.
    allowed_tool_categories: tuple[str, ...] = ()
    # Per-project capability switches -- "absolutely everything on/off". Anything
    # not listed falls back to its SAFE default (see capability_enabled), so a bare
    # inventory is customer-safe and a lab opts features in explicitly. Known switches:
    #   modify_target                 target-changing / DESTRUCTIVE actions (the lab switch)
    #   require_human_for_destructive  even with modify_target on, still gate on a token
    #   solve_captcha                 may answer a CAPTCHA to proceed
    #   osint                         may use OSINT sources/tools (web/Shodan/CLI)
    # Chatbot/agent engagements add four:
    #   plant_content                 may write into an ingestion sink the bot later
    #                                 reads (a document, a profile field, a ticket).
    #                                 Indirect prompt injection cannot be tested
    #                                 without it, and it MODIFIES the target, so it
    #                                 also needs modify_target.
    #   out_of_band_callback          may run a collector the target can reach, to
    #                                 prove exfiltration by the target's own traffic.
    #                                 Off by default because a routable collector is
    #                                 a service the operator runs on someone else's
    #                                 behalf, which an engagement signs for.
    #   capture_harmful_content       may store a harmful reply VERBATIM. Off by
    #                                 default: the case still runs and the oracle
    #                                 still fires, but what is kept is the verdict,
    #                                 the rubric hash and a digest of the reply --
    #                                 provable without the harness holding the
    #                                 artifact in an immutable store.
    #   multi_turn_escalation         may run staged multi-turn attacks (crescendo,
    #                                 many-shot). Off by default because they cost
    #                                 many target messages and a client may want
    #                                 single-turn testing only.
    # A disabled switch is a logged missed opportunity, not a hard failure.
    #: Calls per second the gate will allow for this project, across EVERY worker
    #: in the run. None means no limit, and that is the default on purpose.
    #:
    #: The limiter was hardcoded at 30/s and applied identically to a disposable
    #: container on loopback and to a live PLC. A throttle nobody chose for the lab
    #: is not a safety property, it is a number that quietly caps a benchmark --
    #: and the design always said this belonged here: schemas/engagement.py
    #: describes the inventory as the layer that "enforces (hosts, tools, rate
    #: limits)". Restraint is written down per engagement, next to modify_target
    #: and require_human_for_destructive, where somebody signs for it.
    max_calls_per_second: float | None = None
    #: How many workers may act against this project at once. None means no cap.
    #: The same reasoning: a lab runs as wide as it likes, a plant segment does not.
    max_concurrent_workers: int | None = None
    #: Seconds a single command tool may run. A full-port -sV sweep or a long
    #: fuzz legitimately exceeds the old hardcoded 120, and in a lab there is no
    #: reason to cut it off; a customer engagement can bound it here.
    command_timeout_s: float = 120.0
    #: Target messages this engagement may send, across EVERY worker and the
    #: whole run. None means no cap.
    #:
    #: The right unit for a chatbot engagement, and it is not interchangeable with
    #: the others. max_calls_per_second protects the target from a burst;
    #: max_tokens bounds what WE spend on our own models. Neither bounds what the
    #: CLIENT spends serving us -- and attack-success-rate means every case costs
    #: n trials plus m controls, so a forty-case library is hundreds of inference
    #: calls on someone else's bill. Denial-of-wallet is a finding class here, and
    #: a harness that can produce it can also commit it by accident.
    max_target_messages: int | None = None
    #: Turns one conversation may run to. A sixty-turn crescendo is itself a
    #: denial-of-wallet, and a runaway loop looks exactly like one.
    max_messages_per_conversation: int | None = None
    #: Largest file the upload tool will send. 1MB was hardcoded; upload handling
    #: is itself a vulnerability class, so a lab may want to exceed it.
    max_upload_bytes: int = 1_000_000
    #: Static headers every request to this target carries: a bearer token, an
    #: `x-api-key`, a cookie. Most chat endpoints that are worth assessing sit behind
    #: one, and the harness could not send any.
    #:
    #: A value of the form `${ENV_VAR}` is read from the environment when the
    #: inventory loads, so the credential is never committed to a YAML file and never
    #: appears in argv -- `--header` would put it in `ps` output for anyone on the
    #: box. See `resolved_auth_headers`.
    #:
    #: Here rather than in a tool argument because it is part of the SCOPE: these
    #: headers are the authority under which the engagement acts, they are signed
    #: with everything else, and the model must never be able to set one.
    auth_headers: dict[str, str] = Field(default_factory=dict)
    #: Values the client has confirmed should never surface in a reply, keyed by
    #: name: {"canary": "ACME-INTERNAL-4417"}.
    #:
    #: A disclosure case needs the secret THE TARGET HOLDS. The harness cannot invent
    #: one -- a freshly minted token the bot has never seen can never come back, so a
    #: case that mints its own scores 0.00 against a bot that leaked everything it
    #: had. That is a false negative on the most basic chatbot test there is, and it
    #: is worse than no test because it reads as a clean result.
    #:
    #: Only a case that PLANTS its marker (indirect injection) can mint one. Extraction
    #: cases declare `needs=("engagement_canary",)` and are reported UNREACHABLE when
    #: this is empty, rather than quietly failing.
    seeds: dict[str, str] = Field(default_factory=dict)
    #: What the TARGET can do -- "tools", "rag", "memory", "multimodal", "files",
    #: "web", "stateful". The corpus's `requires:` vocabulary, and a different
    #: question from the capabilities below: those say what WE are permitted to do,
    #: these say what the bot is. A retrieval-poisoning case against a bot with no
    #: retrieval is not a resisted attack, it is a test with nothing to attack, and
    #: putting the two in the same column is how a coverage number claims ground
    #: nobody tested.
    #:
    #: Signed with the scope because it decides which cases run, and because the
    #: client told us -- "yes it has tools" is a statement of fact about their system
    #: that belongs in the engagement record next to who authorised it.
    target_features: tuple[str, ...] = ()
    #: Channels the harness can plant indirect content into: "document",
    #: "tool_output", "web_page", "email", "file_metadata", "calendar", "image",
    #: "audio". Empty means no indirect case can run.
    #:
    #: The most dangerous omission in the whole system if this is wrong in the
    #: optimistic direction. An indirect-injection case run without its seeding sends
    #: an innocuous question, gets an innocuous answer, and reports the target as
    #: resistant to a class of attack that was never attempted.
    seedable_channels: tuple[str, ...] = ()
    #: Ordinary questions a real user of THIS bot would ask. The baseline control arm:
    #: run once per engagement and reused as the control for every case that carries no
    #: matched control of its own.
    #:
    #: Needed because the corpus carries no controls, and without a control arm nothing
    #: reaches VALIDATED -- so a report would show an attack success rate of 1.00 and a
    #: validated count of 0, which reads as a broken tool rather than as the honest
    #: "measured but not falsified" that it is.
    #:
    #: What it establishes: the oracle does not fire on normal traffic. A bot that
    #: cheerfully quotes internal document titles to everyone fails
    #: `restricted_title_leaked` on every case, and without this the report calls that
    #: a finding rather than a house style.
    #:
    #: What it does NOT establish: that the ATTACK caused the result. For an injection
    #: case, only a matched control separates "the injection worked" from "the bot is
    #: unprotected and would have told anyone". `TrialStats.control_kind` records which
    #: ran so the report cannot blur them.
    #:
    #: Write them in the client's own domain -- a shipping question for a retailer, a
    #: claims question for an insurer. Generic questions test a generic bot.
    baseline_turns: tuple[str, ...] = ()

    # ----- how to reach the chat endpoint ------------------------------------
    #
    # On the inventory rather than as command-line flags, for the same reason
    # `auth_headers` is: these are engagement facts the operator establishes once and
    # should not have to retype, `thresh preflight` verifies them, and the run then
    # uses the thing preflight verified rather than something retyped at 9am. A flag
    # that has to be remembered is a flag that gets forgotten, and forgetting `--tls`
    # against a production endpoint means a plaintext request to a host that expects
    # TLS -- which at best fails and at worst sends the auth header in clear.
    #: Does the endpoint speak HTTPS? A production assistant does.
    tls: bool = False
    #: Verify the certificate. Off ONLY for an internal CA or a self-signed staging
    #: cert, and the report has to say so: findings attributed to a host whose identity
    #: was not checked are weaker, because an agent that accepts any certificate cannot
    #: tell the target from whatever sits in front of it.
    verify_tls: bool = True
    #: Path of the chat endpoint.
    chat_path: str = "/chat"
    #: Dotted path to the reply inside their JSON, e.g. `data.message.text`. Empty means
    #: the common shapes are tried. Getting this wrong does not raise -- it degrades to
    #: "no reply found", which `thresh preflight` reports as a FAIL rather than letting
    #: every case score 0.00 against an assistant that is answering normally.
    reply_path: str = ""
    #: Default port. 443 when `tls` and nothing is set, else 80.
    chat_port: int = 0

    def port_for_chat(self) -> int:
        return self.chat_port or (443 if self.tls else 80)

    capabilities: dict[str, bool] = Field(default_factory=dict)
    authorization: AuthorizationRecord

    # ----- scope checks (pure, deterministic) -----

    def ip_in_scope(self, ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        for cidr in self.allowed_cidrs:
            try:
                if addr in ipaddress.ip_network(cidr, strict=False):
                    return True
            except ValueError:
                continue
        return False

    def domain_in_scope(self, domain: str) -> bool:
        d = domain.lower().rstrip(".")
        for allowed in self.allowed_domains:
            a = allowed.lower().rstrip(".")
            if d == a or d.endswith("." + a):
                return True
        return False

    def device_in_scope(self, device: str) -> bool:
        return device in self.allowed_devices

    def tool_allowed(self, tool: str) -> bool:
        """Exact membership, and no name is permitted outside the allowlist.

        There was a second clause: `or tool in LOCAL_VERBS`, two verbs granted
        everywhere because they only read the tool catalogue and the skill
        library. Neither of those exists here -- they stayed in the package this
        was extracted from, along with the wide tool surface and the model-driven
        loop that called them. So the clause widened the gate by two names that
        nothing in this package serves and nothing asks for.

        If a local read-only verb arrives, the clause comes back WITH its reason.
        An empty set left in the condition would read as live and guard nothing.
        """
        return tool in self.allowed_tools

    def capability_enabled(self, name: str, default: bool | None = None) -> bool:
        """Is capability ``name`` switched on for this project? Fail-closed: an
        unlisted switch returns its safe default (OFF for anything risky)."""
        if name in self.capabilities:
            return bool(self.capabilities[name])
        if default is not None:
            return default
        return _CAPABILITY_SAFE_DEFAULTS.get(name, False)

    def resolved_auth_headers(self) -> dict[str, str]:
        """`auth_headers` with `${ENV_VAR}` expanded, dropping what is unset.

        An unset variable yields NO HEADER rather than the literal `${TOKEN}`: sending
        that to a customer endpoint would be an authentication attempt with a
        nonsense credential, which looks like a brute-force probe in their logs. The
        header is simply absent, and preflight is where the operator finds out --
        before the window opens rather than during it.
        """
        import os
        import re

        out: dict[str, str] = {}
        for name, raw in (self.auth_headers or {}).items():
            value = str(raw)
            missing = False

            def _sub(m, _missing=None):
                nonlocal missing
                got = os.environ.get(m.group(1), "")
                if not got:
                    missing = True
                return got

            value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, value)
            if missing or not value.strip():
                continue
            out[name] = value
        return out

    def unresolved_auth_headers(self) -> list[str]:
        """Header names whose `${ENV_VAR}` is not set. Preflight reports these."""
        import os
        import re

        out: list[str] = []
        for name, raw in (self.auth_headers or {}).items():
            for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(raw)):
                if not os.environ.get(var):
                    out.append(f"{name} (needs ${{{var}}})")
        return out

    def content_hash(self) -> str:
        """Stable hash of the inventory content (signature excluded)."""
        payload = self.model_dump(mode="json")
        payload["authorization"].pop("signature", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
