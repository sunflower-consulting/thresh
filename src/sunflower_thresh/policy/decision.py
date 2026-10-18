"""Request and Decision objects for the policy gate."""
from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field

from sunflower_thresh.schemas.target import ActionClass


class TargetRef(BaseModel):
    """Identifies what an action is aimed at. Exactly one field is meaningful."""
    #: A field this model does not know is a typo, and a typo in a scope
    #: document must not be silently dropped. Several of them fail OPEN:
    #: `expires` instead of `expires_on` leaves expires_on None, which is an
    #: authorization that never expires, and `capability` instead of
    #: `required_capability` leaves the capability check with nothing to
    #: check, so a request that should be refused is permitted.
    model_config = ConfigDict(extra="forbid")

    ip: str | None = None
    domain: str | None = None
    device: str | None = None

    def describe(self) -> str:
        return self.ip or self.domain or self.device or "<none>"


def host_of(req: ToolRequest) -> str:
    """The host a tool should actually connect to: the DOMAIN if there is one.

    Every chat tool read `req.target.ip` and nothing else, so a customer endpoint at
    `chat.customer.com` could not be targeted at all -- the harness could only ever
    test something it could name by address. Resolving the hostname ourselves first
    is not a workaround: it breaks TLS SNI and certificate verification, and breaks
    virtual hosting, which is how most chat endpoints are served.

    Domain first because when both are present the domain is the more specific
    statement of intent -- an operator who wrote a hostname meant the hostname.
    """
    return (req.target.domain or req.target.ip or "").strip()


def target_ref(host: str) -> TargetRef:
    """A TargetRef for `host`, as a domain unless it is an IP literal.

    The validators and the profiler hardcoded `TargetRef(ip=...)`, so even once the
    tools accepted a hostname the GATE would have refused it: scope is checked
    against `ip_in_scope` or `domain_in_scope` depending on which field is set, and
    a hostname in the ip field matches no CIDR.
    """
    import ipaddress

    host = (host or "").strip()
    if not host:
        return TargetRef()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return TargetRef(domain=host)
    return TargetRef(ip=host)


class ToolRequest(BaseModel):
    #: A field this model does not know is a typo, and a typo in a scope
    #: document must not be silently dropped. Several of them fail OPEN:
    #: `expires` instead of `expires_on` leaves expires_on None, which is an
    #: authorization that never expires, and `capability` instead of
    #: `required_capability` leaves the capability check with nothing to
    #: check, so a request that should be refused is permitted.
    model_config = ConfigDict(extra="forbid")
    project: str
    tool: str
    action_class: ActionClass
    target: TargetRef = Field(default_factory=TargetRef)
    args: dict = Field(default_factory=dict)
    requested_by: str = Field(default="", description="agent role proposing the action")
    # A human approval token, required for DESTRUCTIVE actions.
    human_token: str | None = None
    # Named capability this action needs (e.g. "osint", "solve_captcha"). When the
    # project has that switch off, the gate denies with CAPABILITY_DISABLED, which
    # the loop records as a reviewable "missed opportunity" rather than a hard error.
    required_capability: str | None = None
    #: How many TARGET messages this call spends. Chat verbs set 1; everything
    #: else leaves it 0.
    #:
    #: On the request rather than inferred from the tool name, for the same reason
    #: required_capability is: the policy belongs in the engine, and a gate that
    #: has to keep a list of which tools are chatty is a gate that will be wrong
    #: the first time a tool is added.
    message_cost: int = 0


class DenyReason(str, enum.Enum):
    KILLSWITCH = "killswitch_engaged"
    NO_INVENTORY = "no_inventory_loaded"
    INVENTORY_UNVERIFIED = "inventory_not_verified"
    AUTH_INACTIVE = "authorization_inactive"
    TOOL_NOT_PERMITTED = "tool_not_permitted"
    TARGET_OUT_OF_SCOPE = "target_out_of_scope"
    EXTERNAL_NETWORK_BLOCKED = "external_network_blocked"
    DESTRUCTIVE_NEEDS_HUMAN = "destructive_needs_human_token"
    # A capability switch for this project is off. NOT a scope violation -- the
    # action was in-scope and possible, but disabled by configuration, so the loop
    # records it as a reviewable "missed opportunity" (maybe a human should do it).
    CAPABILITY_DISABLED = "capability_disabled"
    RATE_LIMITED = "rate_limited"
    #: The engagement's target-message budget is spent. Unlike RATE_LIMITED this
    #: does NOT self-clear, so it must never be retried -- see
    #: sweep.TRANSIENT_DENIALS.
    MESSAGE_BUDGET_EXHAUSTED = "message_budget_exhausted"


class Decision(BaseModel):
    #: A field this model does not know is a typo, and a typo in a scope
    #: document must not be silently dropped. Several of them fail OPEN:
    #: `expires` instead of `expires_on` leaves expires_on None, which is an
    #: authorization that never expires, and `capability` instead of
    #: `required_capability` leaves the capability check with nothing to
    #: check, so a request that should be refused is permitted.
    model_config = ConfigDict(extra="forbid")
    allowed: bool
    reason: str
    deny_code: DenyReason | None = None
    request: ToolRequest
