"""The policy engine -- plain code, outside the LLM.

Every active tool call passes through `authorize()` before it runs. The engine
is fail-closed: with no verified inventory, nothing active is permitted. Agents
can call this; they cannot change its rules or its loaded inventories.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from sunflower_thresh.audit.log import AuditLog
from sunflower_thresh.policy.decision import Decision, DenyReason, ToolRequest
from sunflower_thresh.policy.killswitch import KillSwitch
from sunflower_thresh.schemas.target import ActionClass, TargetInventory


class RateLimiter:
    """Sliding-window limiter keyed by (project, tool). Protects fragile gear.

    The rate is PER PROJECT, which makes it the whole run's budget rather than
    each worker's: two agents hitting one PLC share the allowance, which is the
    only reading that means anything when the point is to not overwhelm the gear.

    ``max_calls=None`` means no limit. That is the default for a project that does
    not ask for one, because a disposable container on loopback and a live PLC do
    not deserve the same throttle, and the one this class used to invent (30/s)
    was applied to both.
    """

    def __init__(self, max_calls: int | None, window_s: float = 1.0):
        self.max_calls = max_calls
        self.window_s = window_s
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        #: Per-project overrides from the loaded inventories.
        self._per_project: dict[str, float | None] = {}

    def configure(self, project: str, max_calls: float | None) -> None:
        """Set this project's allowance from its inventory.

        A None from the inventory means "this project did not ask for a limit",
        not "clear whatever was set" -- so an explicitly constructed limiter is
        left alone. The default construction is already unlimited, so a project
        that stays silent is unthrottled either way; this only preserves the
        caller who passed a limiter on purpose.
        """
        if max_calls is None:
            return
        with self._lock:
            self._per_project[project] = max_calls

    def limit_for(self, project: str) -> float | None:
        return self._per_project.get(project, self.max_calls)

    def allow(self, project: str, tool: str, now: float | None = None) -> bool:
        limit = self.limit_for(project)
        if limit is None:
            return True                      # no limit asked for, none applied
        now = now if now is not None else time.monotonic()
        key = (project, tool)
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


class MessageBudget:
    """Target messages spent per project, for the whole engagement.

    A sibling of RateLimiter, and deliberately a different question. The limiter
    asks "am I hitting this target too fast right now"; a sliding window forgets,
    which is exactly right for protecting fragile gear and exactly wrong for a
    budget. This one asks "has this engagement spent its allowance", and the
    answer is monotonic: once spent, it stays spent.

    Why it exists at all: validating a chat finding takes n adversarial trials plus
    m controls, so a forty-case library is hundreds of inference calls on the
    CLIENT'S bill. Nothing else in the gate bounds that -- max_calls_per_second
    bounds the rate, max_tokens bounds what we spend on our own models. Unbudgeted,
    the first sweep against a real endpoint is itself the denial-of-wallet attack
    the harness is meant to report.

    reserve() is atomic: a fleet means several workers asking at once, and a budget
    that can be overspent by a race is not a budget.
    """

    def __init__(self) -> None:
        self._limits: dict[str, int | None] = {}
        self._spent: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def configure(self, project: str, limit: int | None) -> None:
        """Set the project's allowance from its signed inventory.

        Unlike the rate limiter, a None here is recorded: "this project has no
        message cap" is a real answer and must survive a reload that follows a
        project which did have one.
        """
        with self._lock:
            self._limits[project] = limit

    def limit_for(self, project: str) -> int | None:
        return self._limits.get(project)

    def reserve(self, project: str, n: int) -> bool:
        """Spend n messages if the allowance covers them. Atomic; all or nothing."""
        if n <= 0:
            return True
        with self._lock:
            limit = self._limits.get(project)
            if limit is None:
                self._spent[project] += n
                return True
            if self._spent[project] + n > limit:
                return False
            self._spent[project] += n
            return True

    def refund(self, project: str, n: int) -> None:
        """Give back messages a denied or failed call never actually sent.

        Reserving at authorize() time means a call the gate allowed but the
        transport failed has been charged for a message nobody received. Refunding
        on failure keeps the count honest; over-counting is the safe direction, so
        a refund that does not happen costs the run a message rather than letting
        it exceed what was signed for.
        """
        if n <= 0:
            return
        with self._lock:
            self._spent[project] = max(0, self._spent[project] - n)

    def spent(self, project: str) -> int:
        with self._lock:
            return self._spent[project]

    def remaining(self, project: str) -> int | None:
        with self._lock:
            limit = self._limits.get(project)
            if limit is None:
                return None
            return max(0, limit - self._spent[project])


class PolicyEngine:
    def __init__(
        self,
        audit: AuditLog | None = None,
        killswitch: KillSwitch | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.audit = audit or AuditLog()
        self.killswitch = killswitch or KillSwitch()
        # No global default: an unconfigured project is unlimited, and a project
        # that wants a limit says so in its inventory.
        self.rate_limiter = rate_limiter or RateLimiter(max_calls=None, window_s=1.0)
        self.message_budget = MessageBudget()
        # project -> (inventory, verified?)
        self._inventories: dict[str, tuple[TargetInventory, bool]] = {}
        self._lock = threading.Lock()

    # ----- inventory loading (a human/config action, logged) -----

    def load_inventory(
        self, inventory: TargetInventory, *, expected_hash: str | None = None
    ) -> bool:
        """Load and verify an inventory. Verification = hash + active authorization.

        If `expected_hash` is given it must match the computed content hash. The
        authorization record must be active. Only a verified inventory enables
        active tools for its project.
        """
        computed = inventory.content_hash()
        verified = inventory.authorization.is_active()
        if expected_hash is not None and expected_hash != computed:
            verified = False
        with self._lock:
            self._inventories[inventory.project] = (inventory, verified)
        # The project's own allowance, or none. Applied whether or not the
        # inventory verified: an unverified project cannot run active tools at
        # all, so its limit is moot, but a verified one must get exactly the
        # figure its RoE signed for.
        self.rate_limiter.configure(inventory.project, inventory.max_calls_per_second)
        self.message_budget.configure(inventory.project, inventory.max_target_messages)
        self.audit.record(
            "inventory_loaded",
            project=inventory.project,
            content_hash=computed,
            verified=verified,
        )
        return verified

    # ----- the gate -----

    def authorize(self, req: ToolRequest) -> Decision:
        decision = self._evaluate(req)
        self.audit.record(
            "authorization",
            allowed=decision.allowed,
            deny_code=decision.deny_code.value if decision.deny_code else None,
            project=req.project,
            tool=req.tool,
            action_class=req.action_class.value,
            target=req.target.describe(),
            requested_by=req.requested_by,
        )
        return decision

    def _evaluate(self, req: ToolRequest) -> Decision:
        def deny(code: DenyReason, msg: str) -> Decision:
            return Decision(allowed=False, reason=msg, deny_code=code, request=req)

        # 0. Kill-switch wins over everything.
        if self.killswitch.engaged:
            return deny(DenyReason.KILLSWITCH, f"kill-switch engaged: {self.killswitch.reason}")

        # 1. Must have a verified inventory for this project (fail-closed).
        with self._lock:
            entry = self._inventories.get(req.project)
        if entry is None:
            return deny(DenyReason.NO_INVENTORY, f"no inventory loaded for project {req.project!r}")
        inventory, verified = entry

        # Passive actions still require a loaded inventory, but we allow a
        # verified-or-not read ONLY when it touches no target (pure lookups like
        # query_cve). Any target-bound action requires verification.
        target_bound = any(
            [req.target.ip, req.target.domain, req.target.device]
        )

        if not verified and (target_bound or req.action_class is not ActionClass.PASSIVE):
            return deny(
                DenyReason.INVENTORY_UNVERIFIED,
                "inventory is not verified (bad hash or inactive authorization)",
            )

        if not inventory.authorization.is_active():
            return deny(DenyReason.AUTH_INACTIVE, "authorization record is not active")

        # 2. Tool must be permitted for this project.
        if not inventory.tool_allowed(req.tool):
            return deny(
                DenyReason.TOOL_NOT_PERMITTED,
                f"tool {req.tool!r} not in project allowlist",
            )

        # 2b. Capability switch: the action is in-scope and possible, but this
        # project has its capability turned off -> a reviewable missed opportunity,
        # not a scope breach.
        if req.required_capability and not inventory.capability_enabled(req.required_capability):
            return deny(
                DenyReason.CAPABILITY_DISABLED,
                f"capability {req.required_capability!r} is disabled for this project",
            )

        # 3. Target must be explicitly in scope (for any target-bound action).
        if target_bound:
            in_scope = (
                (req.target.ip and inventory.ip_in_scope(req.target.ip))
                or (req.target.domain and inventory.domain_in_scope(req.target.domain))
                or (req.target.device and inventory.device_in_scope(req.target.device))
            )
            if not in_scope:
                return deny(
                    DenyReason.TARGET_OUT_OF_SCOPE,
                    f"target {req.target.describe()!r} not explicitly in scope",
                )
            # External-network block: a routable target needs external_network=True.
            if (req.target.ip and not inventory.external_network
                    and _is_global(req.target.ip)):
                    return deny(
                        DenyReason.EXTERNAL_NETWORK_BLOCKED,
                        "target is on an external network but external_network is false",
                    )

        # 4. Target-modifying (DESTRUCTIVE) actions. The "lab switch": modify_target
        # is OFF by default (customer-safe), so a destructive action off-switch is a
        # logged missed opportunity. When a lab turns it ON, a human token is still
        # required unless require_human_for_destructive is also turned off.
        if req.action_class is ActionClass.DESTRUCTIVE:
            if not inventory.capability_enabled("modify_target"):
                return deny(
                    DenyReason.CAPABILITY_DISABLED,
                    "target-modifying action blocked: modify_target is off for this project",
                )
            if inventory.capability_enabled("require_human_for_destructive") and not req.human_token:
                return deny(
                    DenyReason.DESTRUCTIVE_NEEDS_HUMAN,
                    "destructive action requires a human approval token",
                )

        # 5. Rate limit (protects fragile OT gear from agent loops).
        if not self.rate_limiter.allow(req.project, req.tool):
            return deny(DenyReason.RATE_LIMITED, "rate limit exceeded for tool")

        # 6. The engagement's target-message budget. LAST, so a call denied for any
        # other reason is not charged for a message it never sent.
        if req.message_cost and not self.message_budget.reserve(
                req.project, req.message_cost):
                limit = self.message_budget.limit_for(req.project)
                return deny(
                    DenyReason.MESSAGE_BUDGET_EXHAUSTED,
                    f"target-message budget spent: "
                    f"{self.message_budget.spent(req.project)}/{limit} for this engagement",
                )

        return Decision(allowed=True, reason="authorized", request=req)


    # ----- scope check for a single host token (used by freeform command tools) -----

    def inventory_for(self, project: str):
        """The VERIFIED inventory for a project, or None.

        Exposed so a tool can read a capability switch rather than hardcoding a
        policy of its own -- the gate stays the authority, but a tool sometimes
        needs to know what the gate would decide. Returns None for an unknown,
        unverified or expired project, so callers fail closed.
        """
        with self._lock:
            entry = self._inventories.get(project)
        if entry is None:
            return None
        inventory, verified = entry
        if not verified or not inventory.authorization.is_active():
            return None
        return inventory

    def target_in_scope(self, project: str, token: str) -> bool:
        """Is one host/IP/URL token inside the verified scope for ``project``?

        Freeform command tools (where an LLM composes the argv) call this for
        every host-like token they find, so scope enforcement stays in the
        policy engine, not in the tool. Fail-closed: unknown project, unverified
        inventory, or an unparseable token all return False.
        """
        with self._lock:
            entry = self._inventories.get(project)
        if entry is None:
            return False
        inventory, verified = entry
        if not verified or not inventory.authorization.is_active():
            return False
        host = _host_from_token(token)
        if not host:
            return False
        import ipaddress
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return inventory.domain_in_scope(host)
        if not inventory.ip_in_scope(host):
            return False
        if not inventory.external_network and _is_global(host):
            return False
        return True


def _host_from_token(token: str) -> str:
    """Pull a bare host out of a URL / host:port / user@host / bare host token."""
    t = token.strip()
    if "://" in t:
        t = t.split("://", 1)[1]
    t = t.split("/", 1)[0]          # drop path
    if "@" in t:
        t = t.rsplit("@", 1)[1]     # drop userinfo
    if t.startswith("["):           # [ipv6]:port
        return t[1:].split("]", 1)[0]
    if t.count(":") == 1:           # host:port
        t = t.split(":", 1)[0]
    return t.lower()


def _is_global(ip: str) -> bool:
    import ipaddress

    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False
