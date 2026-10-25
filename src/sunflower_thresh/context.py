"""The gated execution context: the one way a tool call reaches a target.

Everything that acts -- the orchestrator, a fleet worker, the sweep, a validator --
goes through `AgentContext.execute_tool`, which authorizes first. That is the
boundary the whole safety argument rests on: the model proposes, the PolicyEngine
decides, and there is no second path.

This module also held an `Agent` ABC and a `RoleSpec`, a role-bound-to-a-model
abstraction with one vestigial caller. Both are gone from this package.
"""
from __future__ import annotations

from sunflower_thresh.audit.log import AuditLog
from sunflower_thresh.policy.decision import Decision, ToolRequest
from sunflower_thresh.policy.engine import PolicyEngine


class AgentContext:
    """Everything an agent needs to act safely.

    Tool execution ALWAYS goes through here so the policy gate cannot be skipped.
    """

    def __init__(
        self,
        project: str,
        policy: PolicyEngine,
        audit: AuditLog | None = None,
        tool_impls: dict | None = None,
        observer=None,
    ):
        self.project = project
        self.policy = policy
        self.audit = audit or policy.audit
        self._tool_impls = tool_impls or {}
        #: Called after every gated call with (req, decision, out). Set by whoever owns
        #: the run log, so the gate's work reaches the event stream.
        #:
        #: Here rather than in each caller because this is the one chokepoint every tool
        #: call passes through. The orchestrator logs its own `policy` and `tool_result`
        #: events by hand, which worked until a second driver appeared: the deterministic
        #: chat sweep calls `execute_tool` directly, so on the one path a customer
        #: engagement uses, every gate decision was invisible. The live view showed a
        #: run with no traffic and the bench lamp sat on a steady colour -- the safety
        #: story is the gate, and nothing was reporting it.
        self.observer = observer

    def register_tool(self, name: str, impl) -> None:
        self._tool_impls[name] = impl

    def execute_tool(self, req: ToolRequest) -> tuple[Decision, dict | None]:
        """Authorize then (only if allowed) run. Returns (decision, output).

        A tool that raises returns an error dict rather than ending the run.
        `impl(req)` used to be called bare, and so did this method, so a
        ConnectionRefusedError from fingerprinting a closed port propagated out
        of `Orchestrator.run()` and out of `harness run`: no RunResult, no
        telemetry, no run_end event, and every finding gathered in the previous
        fifty-nine steps lost. Tools raise on completely ordinary target
        conditions -- a closed port, a container that has just died, a login
        endpoint that answers with a JSON array -- and none of those is a reason
        to throw the run away.

        The error reads like a failed connection on purpose: the orchestrator's
        dead-target detector inspects error strings, so a crash that means "the
        target is gone" now reaches the detector that exists for it.

        BaseException subclasses that are not Exception -- KeyboardInterrupt,
        SystemExit -- pass straight through. Ctrl-C is the operator, not a tool
        fault, and swallowing it would make a run unstoppable.
        """
        decision = self.policy.authorize(req)
        if not decision.allowed:
            self._observe(req, decision, None)
            return decision, None
        impl = self._tool_impls.get(req.tool)
        if impl is None:
            self._refund_messages(req)
            out = {"error": f"no implementation registered for {req.tool!r}"}
            self._observe(req, decision, out)
            return decision, out
        try:
            out = impl(req)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see above
            self.audit.record("tool_exception", tool=req.tool, project=req.project,
                              exception=type(exc).__name__, error=str(exc)[:500],
                              target=req.target.describe())
            self._refund_messages(req)
            out = {"error": f"{req.tool} raised {type(exc).__name__}: {exc}"[:500],
                   "exception": type(exc).__name__, "evidence_refs": []}
            self._observe(req, decision, out)
            return decision, out
        # A target message is charged at authorize() time, because the budget has
        # to hold across workers and only the gate sees every request. So a call
        # the gate allowed but that never reached the target -- a refused socket, a
        # tool that returned an error before sending -- has been charged for a
        # message nobody received. Evidence is the test: a chat verb that talked to
        # the target recorded the exchange.
        if isinstance(out, dict) and out.get("error") and not out.get("evidence_refs"):
            self._refund_messages(req)
        self._observe(req, decision, out)
        return decision, out

    def _observe(self, req: ToolRequest, decision: Decision, out: dict | None) -> None:
        """Tell the observer, and never let it affect the call.

        Exception-swallowed on purpose, like `RunLog.on_event`: a view that cannot draw
        must not stop a run that is working. The run is the product; the lamp is not.
        """
        if self.observer is None:
            return
        try:
            self.observer(req, decision, out)
        except Exception:  # noqa: BLE001,S110 - a broken view never breaks a run
            pass

    def _refund_messages(self, req: ToolRequest) -> None:
        """Give back a reservation whose call never reached the target.

        Best-effort and quiet: over-counting is the safe direction, so a refund
        that fails costs the run a message rather than letting it exceed what the
        inventory was signed for.
        """
        if not getattr(req, "message_cost", 0):
            return
        budget = getattr(self.policy, "message_budget", None)
        if budget is not None:
            try:
                budget.refund(req.project, req.message_cost)
            except Exception:  # noqa: BLE001,S110 - never on the way out
                pass
